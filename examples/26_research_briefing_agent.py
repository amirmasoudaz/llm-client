from __future__ import annotations

import asyncio
import json
import uuid
from collections import Counter
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage
from cookbook_expansion_support import QdrantRetriever, RetrieverDocument, chunk_text, embed_text_or_fail, excerpt

from llm_client.agent import Agent, AgentDefinition, AgentExecutionPolicy, ToolExecutionMode
from llm_client.engine import ExecutionEngine
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import Message, StreamEventType, ToolCall, ToolCallDelta
from llm_client.spec import RequestContext
from llm_client.structured import StructuredOutputConfig, extract_structured
from llm_client.tools import Tool, ToolResult


RESEARCH_SCOPE = "research-briefing-agent"
QUESTION = "Prepare a commercialization-first research briefing for a robotics safety grant application."
BRIEFING_PACKET = {
    "program": "robotics safety commercialization grant",
    "deployment_context": "supervised warehouse/logistics robotics operations",
    "goal": "reduce operator intervention and safety events with a tightly scoped pilot",
    "budget_cap_usd": 220000,
    "timeline_months": 9,
    "commercialization_priority": "commercialization readiness over novelty claims",
    "known_constraints": [
        "single named adopter site or tightly scoped pilot",
        "explicit safety governance and fallback behavior",
        "no unsupported autonomy or multi-site claims",
    ],
}

RESEARCH_CORPUS = [
    RetrieverDocument(
        doc_id="doc_1",
        title="Warehouse Safety Pilot Results",
        text=(
            "A supervised warehouse robotics pilot across two shifts reduced operator intervention by 31% and "
            "near-miss events by 18% while preserving all hardware safety stops. Reviewers responded positively "
            "to concrete deployment metrics and a narrow operating envelope."
        ),
        source="paper://warehouse-safety-pilot",
        metadata={"kind": "pilot", "citation": "R1"},
    ),
    RetrieverDocument(
        doc_id="doc_2",
        title="Commercialization Reviewer Feedback",
        text=(
            "Commercialization reviewers reward measurable operational impact, deployment feasibility, safety "
            "governance, and named adopter partners. They penalize ambitious autonomy claims unsupported by "
            "pilot validation or a realistic rollout plan."
        ),
        source="memo://reviewer-feedback",
        metadata={"kind": "reviewer", "citation": "R2"},
    ),
    RetrieverDocument(
        doc_id="doc_3",
        title="Budget and Scope Risk Note",
        text=(
            "Grant budgets under $250k usually force a single-site or tightly scoped dual-site pilot. Teams that "
            "claim broad generalizability at this budget level are often marked down unless they frame scale-up as "
            "future work after validation."
        ),
        source="memo://budget-risk",
        metadata={"kind": "risk", "citation": "R3"},
    ),
    RetrieverDocument(
        doc_id="doc_4",
        title="Customer Discovery Interview Summary",
        text=(
            "Three logistics operators said the strongest purchase trigger is reduced manual exception handling with "
            "clear safety accountability. They care more about deployment readiness and operator trust than raw autonomy."
        ),
        source="interview://logistics-operators",
        metadata={"kind": "customer", "citation": "R4"},
    ),
    RetrieverDocument(
        doc_id="doc_5",
        title="Safety Board Governance Note",
        text=(
            "Internal safety boards expect explicit fallback behavior, auditability of interventions, and a staged "
            "operator training plan before approving broader deployment. Governance maturity can offset skepticism "
            "about constrained pilot scope."
        ),
        source="board://safety-governance-note",
        metadata={"kind": "safety", "citation": "R5"},
    ),
    RetrieverDocument(
        doc_id="doc_6",
        title="Commercial Rollout Plan",
        text=(
            "The strongest rollout plans commit to one concrete post-pilot conversion milestone, a named site partner, "
            "and a 90-day validation plan with cost, safety, and adoption metrics. Reviewers prefer a believable wedge "
            "over a broad platform story."
        ),
        source="plan://commercial-rollout",
        metadata={"kind": "commercial", "citation": "R6"},
    ),
]


def _truncate(value: Any, max_chars: int = 220) -> str:
    text = str(value)
    return text if len(text) <= max_chars else f"{text[:max_chars].rstrip()}..."


def _available_citations() -> list[str]:
    return sorted({str(doc.metadata["citation"]) for doc in RESEARCH_CORPUS})


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    seed_entries = [
        MemoryWrite(
            scope=RESEARCH_SCOPE,
            content="Proposal strategy: prioritize commercialization readiness over novelty claims.",
            relevance=0.96,
            metadata={"kind": "strategy"},
        ),
        MemoryWrite(
            scope=RESEARCH_SCOPE,
            content="Budget ceiling is $220k with a 9-month delivery window and no appetite for a multi-country rollout claim.",
            relevance=0.94,
            metadata={"kind": "constraint"},
        ),
        MemoryWrite(
            scope=RESEARCH_SCOPE,
            content="Review panel wants one named deployment milestone and explicit safety governance before broader autonomy claims.",
            relevance=0.93,
            metadata={"kind": "reviewer_preference"},
        ),
        MemoryWrite(
            scope=RESEARCH_SCOPE,
            content="Commercial sponsor asked for evidence that operator trust and exception-handling burden will improve in the first pilot site.",
            relevance=0.91,
            metadata={"kind": "sponsor_signal"},
        ),
    ]
    written: list[dict[str, Any]] = []
    for entry in seed_entries:
        record = await memory.write(entry)
        written.append({"kind": record.metadata.get("kind"), "content": record.content})
    return written


async def _search_corpus(
    retriever: QdrantRetriever,
    embed_engine: ExecutionEngine,
    query: str,
    *,
    allowed_kinds: set[str] | None = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    query_vector = await embed_text_or_fail(
        embed_engine,
        query,
        failure_message="Embedding generation failed while searching the research corpus.",
    )
    raw_hits = await retriever.search(query_vector, limit=max(limit * 3, 6))
    evidence: list[dict[str, Any]] = []
    for hit in raw_hits:
        payload = hit.get("payload", {})
        kind = str(payload.get("kind") or payload.get("metadata", {}).get("kind") or "")
        if allowed_kinds and kind not in allowed_kinds:
            continue
        evidence.append(
            {
                "citation": payload.get("citation"),
                "title": payload.get("title"),
                "source": payload.get("source"),
                "kind": kind,
                "score": hit.get("score"),
                "excerpt": payload.get("text"),
            }
        )
        if len(evidence) >= limit:
            break
    if not evidence:
        for hit in raw_hits[:limit]:
            payload = hit.get("payload", {})
            evidence.append(
                {
                    "citation": payload.get("citation"),
                    "title": payload.get("title"),
                    "source": payload.get("source"),
                    "kind": payload.get("kind"),
                    "score": hit.get("score"),
                    "excerpt": payload.get("text"),
                }
            )
    return evidence


def _build_tools(retriever: QdrantRetriever, embed_engine: ExecutionEngine, memory: ShortTermMemoryStore) -> list[Tool]:
    async def pilot_evidence(topic: str) -> dict[str, Any]:
        evidence = await _search_corpus(retriever, embed_engine, topic, allowed_kinds={"pilot", "customer"}, limit=3)
        return {
            "lens": "pilot_evidence",
            "topic": topic,
            "evidence": evidence,
        }

    async def reviewer_feedback(topic: str) -> dict[str, Any]:
        evidence = await _search_corpus(retriever, embed_engine, topic, allowed_kinds={"reviewer", "safety"}, limit=3)
        return {
            "lens": "reviewer_feedback",
            "topic": topic,
            "evidence": evidence,
        }

    async def commercialization_path(topic: str) -> dict[str, Any]:
        evidence = await _search_corpus(
            retriever,
            embed_engine,
            topic,
            allowed_kinds={"commercial", "customer", "pilot"},
            limit=3,
        )
        return {
            "lens": "commercialization_path",
            "topic": topic,
            "evidence": evidence,
        }

    async def risk_scan(topic: str) -> dict[str, Any]:
        evidence = await _search_corpus(retriever, embed_engine, topic, allowed_kinds={"risk", "safety"}, limit=3)
        return {
            "lens": "risk_scan",
            "topic": topic,
            "evidence": evidence,
        }

    async def budget_guardrails(max_budget_usd: int) -> dict[str, Any]:
        evidence = await _search_corpus(
            retriever,
            embed_engine,
            f"Budget fit and pilot scope for {max_budget_usd} USD robotics safety commercialization grant",
            allowed_kinds={"risk", "commercial", "reviewer"},
            limit=3,
        )
        return {
            "max_budget_usd": max_budget_usd,
            "guidance": (
                "Keep the proposal to one tightly scoped pilot with one named adopter site, "
                "and frame scale-up as post-validation future work."
            ),
            "evidence": evidence,
        }

    async def briefing_memory(topic: str) -> dict[str, Any]:
        records = await memory.retrieve(MemoryQuery(scope=RESEARCH_SCOPE, query=topic, limit=4))
        return {
            "topic": topic,
            "notes": [
                {
                    "kind": record.metadata.get("kind"),
                    "content": record.content,
                }
                for record in records
            ],
        }

    return [
        Tool(
            name="pilot_evidence",
            description="Retrieve pilot and customer evidence supporting a commercialization-oriented robotics proposal.",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            handler=pilot_evidence,
        ),
        Tool(
            name="reviewer_feedback",
            description="Retrieve reviewer and governance evidence about what the grant panel rewards or penalizes.",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            handler=reviewer_feedback,
        ),
        Tool(
            name="commercialization_path",
            description="Retrieve evidence for a believable commercialization wedge and rollout story.",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            handler=commercialization_path,
        ),
        Tool(
            name="risk_scan",
            description="Retrieve deployment, safety, and scope risks that could weaken the proposal.",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            handler=risk_scan,
        ),
        Tool(
            name="budget_guardrails",
            description="Return deterministic budget-fit guidance and evidence for tightly scoped robotics pilots.",
            parameters={
                "type": "object",
                "properties": {"max_budget_usd": {"type": "integer"}},
                "required": ["max_budget_usd"],
                "additionalProperties": False,
            },
            handler=budget_guardrails,
        ),
        Tool(
            name="briefing_memory",
            description="Retrieve memory-backed strategy notes, reviewer preferences, and sponsor signals for the briefing.",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            handler=briefing_memory,
        ),
    ]


def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [{"tool_name": call.name, "arguments": call.parse_arguments()} for call in tool_calls]


def _serialize_tool_results(tool_results: list[ToolResult]) -> list[dict[str, Any]]:
    return [
        {
            "success": result.success,
            "error": result.error,
            "content_preview": _truncate(result.to_string(), 260),
        }
        for result in tool_results
    ]


def _serialize_turns(turns: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "turn_number": turn.turn_number + 1,
            "assistant_preview": _truncate(turn.content or "", 260),
            "tool_calls": _serialize_tool_calls(turn.tool_calls),
            "tool_results": _serialize_tool_results(turn.tool_results),
        }
        for turn in turns
    ]


def _collect_evidence_ledger(turns: list[Any]) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for turn in turns:
        for tool_name, result in zip([call.name for call in turn.tool_calls], turn.tool_results, strict=False):
            content = result.content if isinstance(result.content, dict) else None
            if not isinstance(content, dict):
                continue
            evidence = content.get("evidence")
            if not isinstance(evidence, list):
                continue
            for item in evidence:
                citation = str(item.get("citation") or "")
                source = str(item.get("source") or "")
                key = (citation, source)
                if not citation or key in seen:
                    continue
                seen.add(key)
                ledger.append(
                    {
                        "tool_name": tool_name,
                        "citation": citation,
                        "title": item.get("title"),
                        "source": source,
                        "kind": item.get("kind"),
                        "score": item.get("score"),
                        "excerpt": excerpt(item.get("excerpt")),
                    }
                )
    return sorted(ledger, key=lambda item: (str(item["citation"]), str(item["tool_name"])))


def _citation_audit(available: list[str], structured_data: dict[str, Any] | None) -> dict[str, Any]:
    used = sorted({str(item) for item in (structured_data or {}).get("citations_used", [])})
    available_set = set(available)
    used_set = set(used)
    return {
        "available_citations": available,
        "citations_used": used,
        "all_citations_resolved": used_set.issubset(available_set),
        "missing_citations": sorted(used_set - available_set),
        "unused_citations": sorted(available_set - used_set),
    }


def _assembled_briefing(data: dict[str, Any] | None) -> str | None:
    if not data:
        return None
    evidence_lines = [
        f"- {item['claim']} ({item['confidence']}; citations: {', '.join(item['citations'])})"
        for item in data.get("evidence_cards", [])
    ]
    objection_lines = [f"- {item}" for item in data.get("reviewer_objections", [])]
    action_lines = [f"- {item}" for item in data.get("next_actions", [])]
    return (
        "Core Thesis\n"
        f"- {data.get('core_thesis')}\n\n"
        "Evidence Cards\n"
        f"{chr(10).join(evidence_lines) if evidence_lines else '- None'}\n\n"
        "Commercialization Case\n"
        f"- {data.get('commercialization_case')}\n\n"
        "Budget Fit\n"
        f"- {data.get('budget_fit')}\n\n"
        "Reviewer Objections\n"
        f"{chr(10).join(objection_lines) if objection_lines else '- None'}\n\n"
        "Next Actions\n"
        f"{chr(10).join(action_lines) if action_lines else '- None'}\n\n"
        "Citations Used\n"
        f"- {', '.join(data.get('citations_used', []))}"
    )


async def _run_agent_stream(agent: Agent, prompt: str, context: RequestContext) -> tuple[Any, dict[str, Any]]:
    event_counts: Counter[str] = Counter()
    token_preview_parts: list[str] = []
    tool_call_events: list[dict[str, Any]] = []
    tool_result_events: list[dict[str, Any]] = []
    meta_events: list[dict[str, Any]] = []
    usage_events: list[dict[str, Any]] = []
    final_result: Any = None

    async for event in agent.stream(prompt, context=context):
        event_counts[event.type.value] += 1

        if event.type == StreamEventType.TOKEN:
            if sum(len(part) for part in token_preview_parts) < 280:
                token_preview_parts.append(str(event.data))
            continue

        if event.type in {StreamEventType.TOOL_CALL_START, StreamEventType.TOOL_CALL_DELTA, StreamEventType.TOOL_CALL_END}:
            payload = event.data
            if isinstance(payload, ToolCallDelta):
                tool_call_events.append(
                    {
                        "event": event.type.value,
                        "tool_name": payload.name,
                        "arguments_delta": _truncate(payload.arguments_delta, 160),
                    }
                )
            elif isinstance(payload, ToolCall):
                tool_call_events.append(
                    {
                        "event": event.type.value,
                        "tool_name": payload.name,
                        "arguments": payload.parse_arguments(),
                    }
                )
            continue

        if event.type == StreamEventType.META:
            payload = event.data if isinstance(event.data, dict) else {"value": str(event.data)}
            if payload.get("event") == "tool_result":
                tool_result_events.append(
                    {
                        "tool_name": payload.get("tool_name"),
                        "success": payload.get("success"),
                        "content_preview": _truncate(payload.get("content"), 220),
                    }
                )
            else:
                meta_events.append(payload)
            continue

        if event.type == StreamEventType.USAGE and hasattr(event.data, "to_dict"):
            usage_events.append(event.data.to_dict())
            continue

        if event.type == StreamEventType.DONE:
            final_result = event.data

    if final_result is None:
        raise RuntimeError("Agent stream completed without a final result.")

    return final_result, {
        "event_type_counts": dict(event_counts),
        "token_preview": "".join(token_preview_parts).strip(),
        "tool_call_events": tool_call_events,
        "tool_result_events": tool_result_events,
        "meta_events": meta_events,
        "usage_events": usage_events,
    }


async def main() -> None:
    chat_handle = build_live_provider()
    embed_handle = build_live_provider(capability="embeddings")
    collection = f"research_agent_{uuid.uuid4().hex[:10]}"
    try:
        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)

        chat_lifecycle = LifecycleRecorder()
        chat_diagnostics = EngineDiagnosticsRecorder()
        chat_hooks = HookManager([chat_lifecycle, chat_diagnostics])
        chat_engine = ExecutionEngine(provider=chat_handle.provider, hooks=chat_hooks)
        embed_engine = ExecutionEngine(provider=embed_handle.provider)

        probe_vector = await embed_text_or_fail(
            embed_engine,
            RESEARCH_CORPUS[0].text,
            failure_message="Embedding generation failed while sizing the Qdrant collection.",
        )
        retriever = QdrantRetriever(collection=collection, vector_size=len(probe_vector))
        await retriever.recreate_collection()

        points: list[dict[str, Any]] = []
        indexed_corpus: list[dict[str, Any]] = []
        point_id = 1
        for doc in RESEARCH_CORPUS:
            doc_chunks = chunk_text(doc.text, max_chars=190)
            indexed_corpus.append(
                {
                    "citation": doc.metadata["citation"],
                    "title": doc.title,
                    "source": doc.source,
                    "kind": doc.metadata["kind"],
                    "chunk_count": len(doc_chunks),
                }
            )
            for chunk_index, chunk in enumerate(doc_chunks):
                embedding = await embed_text_or_fail(
                    embed_engine,
                    chunk,
                    failure_message="Embedding generation failed while preparing research notes.",
                )
                points.append(
                    {
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "citation": doc.metadata["citation"],
                            "title": doc.title,
                            "source": doc.source,
                            "text": chunk,
                            "kind": doc.metadata["kind"],
                            "chunk_index": chunk_index,
                        },
                    }
                )
                point_id += 1
        await retriever.upsert(points)

        tools = _build_tools(retriever, embed_engine, memory)
        agent = Agent(
            engine=chat_engine,
            definition=AgentDefinition(
                name="research-briefing-agent",
                system_message=(
                    "You are a research briefing agent for a commercialization-first robotics grant application. "
                    "Before finalizing, gather evidence with at least five distinct tools. "
                    "Use citations from tool outputs in the form [R#]. "
                    "Do not invent product form factors, customer segments, interfaces, hardware architectures, "
                    "or regulatory claims that are not present in the briefing packet, memory, or retrieved evidence. "
                    "If a needed detail is absent, label it as an open question or future work. "
                    "Return sections: Core Thesis, Evidence Cards, Reviewer Lens, Commercialization Path, Risks, Confidence Map, Next Actions."
                ),
                execution_policy=AgentExecutionPolicy(
                    max_turns=5,
                    max_tool_calls_per_turn=8,
                    tool_execution_mode=ToolExecutionMode.PARALLEL,
                    stop_on_tool_error=False,
                ),
            ),
            tools=tools,
        )

        request_context = RequestContext(
            session_id="cookbook-research-briefing",
            job_id="research-briefing",
            tags={"question": QUESTION, "collection": collection},
        )
        prompt = (
            f"Research brief request: {QUESTION}\n\n"
            f"Briefing packet: {BRIEFING_PACKET}\n\n"
            "Context: the team has a $220k cap, a 9-month window, and wants a believable commercialization wedge, "
            "not a moonshot autonomy story. Build an operator-ready and reviewer-aware briefing. "
            "Stay inside the facts in the briefing packet, memory, and retrieved evidence; treat missing details as open questions instead of filling them in."
        )
        result, stream_summary = await _run_agent_stream(agent, prompt, request_context)

        evidence_ledger = _collect_evidence_ledger(result.turns)
        structured = await extract_structured(
            chat_handle.provider,
            [
                Message.system(
                    "Convert the research briefing into a structured commercialization packet. "
                    "Every evidence card must list citation IDs and a confidence bucket. "
                    "Do not introduce unsupported specifics that are absent from the briefing."
                ),
                Message.user(result.content or ""),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "core_thesis": {"type": "string"},
                        "evidence_cards": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "citations": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "confidence": {"type": "string"},
                                },
                                "required": ["claim", "citations", "confidence"],
                                "additionalProperties": False,
                            },
                        },
                        "commercialization_case": {"type": "string"},
                        "reviewer_objections": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "budget_fit": {"type": "string"},
                        "next_actions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "citations_used": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "core_thesis",
                        "evidence_cards",
                        "commercialization_case",
                        "reviewer_objections",
                        "budget_fit",
                        "next_actions",
                        "citations_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )

        citation_audit = _citation_audit([item["citation"] for item in evidence_ledger], structured.data)
        assembled_briefing = _assembled_briefing(structured.data)

        await memory.write(
            MemoryWrite(
                scope=RESEARCH_SCOPE,
                content=json.dumps(structured.data, ensure_ascii=True, sort_keys=True),
                relevance=0.97,
                metadata={"kind": "briefing_packet"},
            )
        )
        memory_after = await memory.retrieve(MemoryQuery(scope=RESEARCH_SCOPE, limit=6))

        distinct_tool_names = sorted({call.name for call in result.all_tool_calls})
        latest_request_report = list(chat_lifecycle.requests.values())[-1] if chat_lifecycle.requests else None
        latest_session_report = chat_lifecycle.sessions.get(request_context.session_id or "")

        print_heading("Research Briefing Agent")
        print_json(
            {
                "chat_provider": {"provider": chat_handle.name, "model": chat_handle.model},
                "embeddings_provider": {"provider": embed_handle.name, "model": embed_handle.model},
                "question": QUESTION,
                "briefing_packet": BRIEFING_PACKET,
                "qdrant_collection": collection,
                "indexed_corpus": indexed_corpus,
                "memory_bootstrap": memory_bootstrap,
                "tool_catalog": [{"name": tool.name, "description": tool.description} for tool in tools],
                "stream_summary": stream_summary,
                "agent_result": {
                    "status": result.status,
                    "num_turns": result.num_turns,
                    "tool_names_used": distinct_tool_names,
                    "turns": _serialize_turns(result.turns),
                    "usage": summarize_usage(result.total_usage),
                    "final_content": result.content,
                },
                "evidence_ledger": evidence_ledger,
                "structured_briefing": {
                    "valid": structured.valid,
                    "repair_attempts": structured.repair_attempts,
                    "usage": summarize_usage(getattr(structured, "usage", None)),
                    "data": structured.data,
                },
                "citation_audit": citation_audit,
                "assembled_briefing": assembled_briefing,
                "observability": {
                    "hook_event_counts": dict(Counter(event for event, _, _ in chat_diagnostics.events)),
                    "lifecycle_event_counts": dict(Counter(event.type.value for event in chat_lifecycle.events)),
                    "latest_request_report": latest_request_report.to_dict() if latest_request_report else None,
                    "latest_session_report": latest_session_report.to_dict() if latest_session_report else None,
                },
                "memory_after_action": [
                    {
                        "kind": record.metadata.get("kind"),
                        "content": record.content,
                    }
                    for record in memory_after
                ],
                "showcase_verdict": {
                    "streamed_agent_run": bool(stream_summary["event_type_counts"]),
                    "used_five_plus_tools": len(distinct_tool_names) >= 5,
                    "citation_grounded": citation_audit["all_citations_resolved"] and bool(citation_audit["citations_used"]),
                    "memory_backed": any(record.metadata.get("kind") == "briefing_packet" for record in memory_after),
                    "reviewer_ready": structured.valid and bool(assembled_briefing),
                },
            }
        )
    finally:
        await close_provider(chat_handle.provider)
        if embed_handle.provider is not chat_handle.provider:
            await close_provider(embed_handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
