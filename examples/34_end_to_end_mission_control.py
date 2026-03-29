from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from cookbook_expansion_support import QdrantRetriever, RetrieverDocument, chunk_text, embed_text_or_fail, excerpt
from cookbook_support import (
    build_live_provider,
    build_provider_handle,
    close_provider,
    fail_or_skip,
    print_heading,
    print_json,
    summarize_usage,
)

from llm_client.agent import Agent, AgentDefinition, AgentExecutionPolicy, ToolExecutionMode
from llm_client.cache import CachePolicy
from llm_client.cache.factory import CacheSettings, build_cache_core
from llm_client.context_assembly import (
    ContextAssemblyRequest,
    ContextSourcePayload,
    ContextSourceRequest,
    MultiSourceContextAssembler,
)
from llm_client.context_planning import (
    DefaultMemoryRetrievalStrategy,
    HeuristicContextPlanner,
    TieredTrimmingStrategy,
)
from llm_client.conversation import Conversation
from llm_client.engine import ExecutionEngine, RetryConfig
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.idempotency import IdempotencyTracker
from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.observability import (
    ActionEvent,
    ArtifactEvent,
    FinalEvent,
    InMemoryEventBus,
    ProgressEvent,
    ReplayRecorder,
    RunMetadata,
    RuntimeEventType,
    ToolEvent,
)
from llm_client.provider_registry import ProviderCapabilities, ProviderDescriptor, ProviderRegistry
from llm_client.providers.types import CompletionResult, Message, StreamEventType, ToolCall
from llm_client.redaction import (
    PayloadPreviewMode,
    ProviderPayloadCaptureMode,
    RedactionPolicy,
    ToolOutputPolicy,
    capture_provider_payload,
    preview_payload,
    sanitize_log_data,
    sanitize_payload,
    sanitize_tool_output,
)
from llm_client.routing import RegistryRouter
from llm_client.spec import RequestContext, RequestSpec
from llm_client.structured import StructuredOutputConfig, extract_structured
from llm_client.summarization import LLMSummarizer
from llm_client.tools import Tool


MISSION_SCOPE = "end-to-end-mission-control"
MISSION_PACKET = {
    "mission_id": "MC-2026-03-24A",
    "incident_id": "INC-7402",
    "release_candidate": "cookbook-live-2026-03-24",
    "service": "checkout-api",
    "objective": (
        "investigate an active checkout incident, protect customer and compliance workflows, "
        "and decide whether the pending release scope should proceed"
    ),
    "business_context": {
        "month_end_hours_until_close": 19,
        "affected_workflows": [
            "checkout payment completion",
            "settlement export delivery",
            "finance reconciliation",
            "compliance evidence packaging",
        ],
        "leadership_expectation": "no-surprises internal updates before customer-facing messaging or release expansion",
    },
    "audience": [
        "incident-commander",
        "release-control",
        "support-lead",
        "trust-and-safety",
        "finance-ops",
    ],
}

RAW_METADATA = {
    "source_id": "META-1",
    "incident_id": "INC-7402",
    "customer_email": "payments-lead@acme.example",
    "session_token": "tok_live_checkout_mission_secret",
    "callback_url": "https://api.acme.example/callback/private/mission-control",
    "internal_channel": "#inc-checkout-month-end",
    "change_id": "chg_12091",
    "release_candidate": "cookbook-live-2026-03-24",
}

INTAKE_EVIDENCE = [
    {
        "source_id": "TXT-1",
        "kind": "text",
        "content": "Checkout payment completion dropped after a routing change; enterprise merchants report delayed confirmations.",
    },
    {
        "source_id": "TXT-2",
        "kind": "text",
        "content": "Finance is nearing month-end close and settlement exports are already time-sensitive.",
    },
    {
        "source_id": "AUD-1",
        "kind": "audio_transcript",
        "content": (
            "On-call notes: error rate spiked within ten minutes of the routing change, queue lag is climbing, "
            "rollback feasibility is under review, and dependency health remains partly unverified."
        ),
    },
    {
        "source_id": "IMG-1",
        "kind": "dashboard_note",
        "content": (
            "Dashboard note: error rate 18.6%, queue lag 4m18s, webhook backlog 3,820, "
            "routing config deployed roughly 12 minutes before the spike."
        ),
    },
]

KNOWLEDGE_CORPUS = [
    RetrieverDocument(
        doc_id="runbook_rollback",
        title="Checkout Rollback Guardrails",
        text=(
            "If checkout degradation follows a routing or config change, pause further changes, evaluate rollback, "
            "verify webhook duplication risk, and communicate confirmed facts only."
        ),
        source="runbook://checkout-rollback",
        metadata={"kind": "runbook"},
    ),
    RetrieverDocument(
        doc_id="runbook_comms",
        title="Customer Communication Guidance",
        text=(
            "Customer-safe updates should acknowledge impact, avoid speculative root-cause claims, state the current next action, "
            "and omit secrets, callback URLs, and personal contact details."
        ),
        source="runbook://customer-comms",
        metadata={"kind": "communications"},
    ),
    RetrieverDocument(
        doc_id="postmortem_export",
        title="Month-End Export Backpressure Postmortem",
        text=(
            "A prior month-end incident showed queue lag can keep rising after rollback unless export and webhook workers are drained deliberately. "
            "Finance and compliance teams required proactive internal updates before external messaging."
        ),
        source="postmortem://month-end-export-backpressure",
        metadata={"kind": "postmortem"},
    ),
    RetrieverDocument(
        doc_id="policy_release",
        title="Release Control Policy",
        text=(
            "Broad launch scope should hold when an active sev-1 incident is unresolved, dependency health is unknown, "
            "or customer-safe messaging has not passed compliance and evaluation gates."
        ),
        source="policy://release-control",
        metadata={"kind": "policy"},
    ),
    RetrieverDocument(
        doc_id="policy_finance",
        title="Finance Escalation Rule",
        text=(
            "Near month-end, incidents that threaten reconciliation, settlement, or export workflows require explicit severity framing, "
            "named owners, and a rollback or mitigation checkpoint."
        ),
        source="policy://finance-escalation",
        metadata={"kind": "policy"},
    ),
]

OPERATOR_THREAD = [
    ("user", "Finance says month-end close is at risk if settlement exports slip any further."),
    ("assistant", "Confirmed. We need explicit severity framing and named next actions."),
    ("user", "Support is seeing more merchant reports about delayed payment confirmations."),
    ("assistant", "That increases customer pressure but does not prove a dependency root cause."),
    ("user", "The routing config changed twelve minutes before the spike."),
    ("assistant", "Treat that as strong timing evidence, not confirmed causality."),
    ("user", "Leadership wants a no-surprises update before any public status page change."),
    ("assistant", "We should prepare internal and customer-safe artifacts separately."),
    ("user", "Compliance wants audit artifacts sanitized before persistence."),
    ("assistant", "We need metadata-only provider logging and sanitized tool outputs."),
]


@dataclass(frozen=True)
class _Entry:
    role: str
    content: str
    entry_type: str = "message"


@dataclass
class _InjectedFailureState:
    failures_remaining: int = 1
    last_failure: dict[str, Any] | None = None


class _FailureInjectedProvider:
    def __init__(self, inner: Any, *, label: str, state: _InjectedFailureState) -> None:
        self._inner = inner
        self._label = label
        self._state = state
        self.name = getattr(inner, "name", label)
        self.model_name = getattr(inner, "model_name", None)
        self.model = getattr(inner, "model", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def complete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        if self._state.failures_remaining > 0:
            self._state.failures_remaining -= 1
            self._state.last_failure = {
                "provider": self._label,
                "status": 503,
                "error": "Injected mission-control failover: primary provider unavailable",
            }
            return CompletionResult(
                status=503,
                error="Injected mission-control failover: primary provider unavailable",
                model=self.model_name or self.model,
            )
        return await self._inner.complete(*args, **kwargs)


class _RetrievalSource:
    def __init__(self, citations: list[dict[str, Any]]) -> None:
        self._citations = citations

    async def load(self, request: ContextSourceRequest) -> ContextSourcePayload:
        _ = request
        entries = [
            _Entry(
                "system",
                f"{citation['citation']} {citation['title']} ({citation['source']}): {citation['text']}",
                entry_type="retrieval_hit",
            )
            for citation in self._citations
        ]
        return ContextSourcePayload(
            source_name="retrieval_hits",
            entries=entries,
            summary="Retrieved runbook, postmortem, and policy evidence with citations for the active mission.",
            metadata={"kind": "retrieval"},
        )


class _PolicySource:
    def __init__(self, policy_snapshot: dict[str, Any]) -> None:
        self._policy_snapshot = policy_snapshot

    async def load(self, request: ContextSourceRequest) -> ContextSourcePayload:
        _ = request
        rules = [
            _Entry("system", f"Policy rule: {rule}", entry_type="policy_rule")
            for rule in self._policy_snapshot["rules"]
        ]
        return ContextSourcePayload(
            source_name="policy_snapshot",
            entries=rules,
            summary="Deterministic mission-control policy defines release and messaging blocks.",
            metadata={"kind": "policy"},
        )


class _LiveStateSource:
    def __init__(self, confirmed: list[str], degraded: list[str], unavailable: list[str]) -> None:
        self._confirmed = confirmed
        self._degraded = degraded
        self._unavailable = unavailable

    async def load(self, request: ContextSourceRequest) -> ContextSourcePayload:
        _ = request
        entries = [_Entry("system", line, entry_type="confirmed_signal") for line in self._confirmed]
        entries.extend(_Entry("system", line, entry_type="degraded_signal") for line in self._degraded)
        entries.extend(_Entry("system", line, entry_type="unavailable_signal") for line in self._unavailable)
        return ContextSourcePayload(
            source_name="live_state",
            entries=entries,
            summary="Live state distinguishes confirmed, degraded, and unavailable evidence.",
            metadata={"kind": "investigation"},
        )


def _truncate(value: Any, max_chars: int = 240) -> str:
    text = str(value or "")
    return text if len(text) <= max_chars else f"{text[:max_chars].rstrip()}..."


def _debug(message: str) -> None:
    if os.getenv("LLM_CLIENT_MISSION_DEBUG") == "1":
        print(f"[mission-debug] {message}", flush=True)


def _message_preview(messages: list[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.role.value,
            "content": _truncate(message.content, 180),
            "tool_call_count": len(message.tool_calls or []),
        }
        for message in messages
    ]


def _serialize_turns(turns: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "turn_number": turn.turn_number + 1,
            "assistant_preview": _truncate(turn.content, 320),
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "parsed_arguments": tool_call.parse_arguments(),
                }
                for tool_call in turn.tool_calls
            ],
            "tool_results": [
                {
                    "success": tool_result.success,
                    "error": tool_result.error,
                    "content_preview": _truncate(tool_result.to_string(), 220),
                }
                for tool_result in turn.tool_results
            ],
        }
        for turn in turns
    ]


def _citation_audit(citations: list[dict[str, Any]], used: list[str]) -> dict[str, Any]:
    available = sorted(citation["source_id"] for citation in citations)
    missing = sorted(set(used) - set(available))
    unused = sorted(set(available) - set(used))
    return {
        "available_citations": available,
        "citations_used": sorted(set(used)),
        "all_citations_resolved": not missing,
        "missing_citations": missing,
        "unused_citations": unused,
    }


def _field_delta_audit(raw_packet: dict[str, Any], safe_packet: dict[str, Any]) -> dict[str, Any]:
    transformed_fields: list[str] = []
    redacted_fields: list[str] = []
    for key, raw_value in raw_packet.items():
        safe_value = safe_packet.get(key)
        if safe_value != raw_value:
            transformed_fields.append(key)
            if safe_value == "[REDACTED]":
                redacted_fields.append(key)
    return {
        "fields_seen": sorted(raw_packet.keys()),
        "transformed_fields": sorted(transformed_fields),
        "redacted_fields": sorted(redacted_fields),
        "safe_for_model": raw_packet != safe_packet,
    }


async def _publish_progress(bus: InMemoryEventBus, ctx: RequestContext, progress: float, message: str, step: str) -> None:
    await bus.publish(ProgressEvent(progress=progress, message=message, step=step).to_runtime_event(ctx))


async def _publish_tool_event(
    bus: InMemoryEventBus,
    ctx: RequestContext,
    *,
    event_type: RuntimeEventType,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    success: bool = True,
    error: str | None = None,
    result: str | None = None,
    duration_ms: float | None = None,
) -> None:
    await bus.publish(
        ToolEvent(
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            error=error,
            result=result,
            duration_ms=duration_ms,
        ).to_runtime_event(ctx, type=event_type)
    )


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    entries = [
        MemoryWrite(
            scope=MISSION_SCOPE,
            content="Incident commanders want confirmed signal first, degraded signal second, and unknowns called out explicitly.",
            relevance=0.97,
            metadata={"kind": "commander_rule"},
        ),
        MemoryWrite(
            scope=MISSION_SCOPE,
            content="Customer-safe summaries must redact tokens, callback URLs, and personal contact details.",
            relevance=0.99,
            metadata={"kind": "privacy_rule"},
        ),
        MemoryWrite(
            scope=MISSION_SCOPE,
            content="Near month-end, release expansion should hold if reconciliation or export workflows remain at risk.",
            relevance=0.96,
            metadata={"kind": "finance_rule"},
        ),
        MemoryWrite(
            scope=MISSION_SCOPE,
            content="Partial rollout metadata is timing evidence only and must not be presented as confirmed causality.",
            relevance=0.95,
            metadata={"kind": "causality_rule"},
        ),
        MemoryWrite(
            scope=MISSION_SCOPE,
            content="If dependency health is unavailable, recommend fallback verification and avoid strong dependency-root-cause claims.",
            relevance=0.94,
            metadata={"kind": "fallback_rule"},
        ),
    ]
    written: list[dict[str, Any]] = []
    for entry in entries:
        record = await memory.write(entry)
        written.append({"kind": record.metadata.get("kind"), "content": record.content})
    return written


async def _stream_completion(
    engine: ExecutionEngine,
    spec: RequestSpec,
    *,
    context: RequestContext,
) -> tuple[CompletionResult, dict[str, Any]]:
    started = time.perf_counter()
    event_counts: Counter[str] = Counter()
    token_preview_parts: list[str] = []
    meta_events: list[dict[str, Any]] = []
    usage_events: list[dict[str, Any]] = []
    content_parts: list[str] = []
    final_result: CompletionResult | None = None
    error_payload: dict[str, Any] | None = None

    async for event in engine.stream(spec, context=context):
        event_counts[event.type.value] += 1
        if event.type == StreamEventType.TOKEN:
            token = str(event.data or "")
            content_parts.append(token)
            if sum(len(part) for part in token_preview_parts) < 420:
                token_preview_parts.append(token)
            continue
        if event.type == StreamEventType.META:
            payload = event.data if isinstance(event.data, dict) else {"value": str(event.data)}
            meta_events.append(payload)
            continue
        if event.type == StreamEventType.USAGE and hasattr(event.data, "to_dict"):
            usage_events.append(event.data.to_dict())
            continue
        if event.type == StreamEventType.ERROR:
            error_payload = event.data if isinstance(event.data, dict) else {"status": 500, "error": str(event.data)}
            break
        if event.type == StreamEventType.DONE:
            if isinstance(event.data, CompletionResult):
                final_result = event.data
            else:
                final_result = CompletionResult(
                    content="".join(content_parts).strip() or None,
                    status=200,
                    model=spec.model,
                    usage=None,
                )

    if final_result is None:
        final_result = CompletionResult(
            content="".join(content_parts).strip() or None,
            status=int((error_payload or {}).get("status", 500) or 500),
            error=str((error_payload or {}).get("error") or "Stream ended without a terminal completion result."),
            model=spec.model,
            usage=None,
        )

    return final_result, {
        "event_type_counts": dict(event_counts),
        "token_preview": "".join(token_preview_parts).strip(),
        "meta_events": meta_events,
        "usage_events": usage_events,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


async def _index_corpus(embed_engine: ExecutionEngine, retriever: QdrantRetriever) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    indexed_chunks: list[dict[str, Any]] = []
    point_id = 1
    for document in KNOWLEDGE_CORPUS:
        for chunk_index, chunk in enumerate(chunk_text(document.text, max_chars=180)):
            vector = await embed_text_or_fail(
                embed_engine,
                chunk,
                failure_message="Embedding generation failed while preparing the mission-control corpus.",
            )
            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "source": document.source,
                        "source_id": f"DOC-{point_id}",
                        "text": chunk,
                        "chunk_index": chunk_index,
                        "metadata": document.metadata,
                    },
                }
            )
            indexed_chunks.append(
                {
                    "point_id": point_id,
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "source": document.source,
                    "source_id": f"DOC-{point_id}",
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
            point_id += 1
    await retriever.upsert(points)
    return indexed_chunks


def _mission_policy_snapshot() -> dict[str, Any]:
    return {
        "policy_version": "mission-control-v1",
        "rules": [
            "Hold broad release scope when an active sev-1 incident remains unresolved.",
            "Customer-facing messaging requires redaction-safe content and explicit approval.",
            "Dependency uncertainty must remain degraded or unknown evidence, not confirmed root cause.",
            "Near month-end, finance-sensitive workflow risk should bias toward conservative rollout scope.",
        ],
        "evaluated_facts": {
            "active_incident": True,
            "severity_hint": "sev-1",
            "dependency_health_unknown": True,
            "customer_safe_artifact_required": True,
            "month_end_hours_until_close": MISSION_PACKET["business_context"]["month_end_hours_until_close"],
        },
    }


def _conversation_messages() -> Conversation:
    conversation = Conversation(
        system_message=(
            "You are the mission-control operator thread. Preserve facts, constraints, and open questions."
        ),
        max_tokens=700,
        reserve_tokens=180,
        truncation_strategy="summarize",
    )
    for role, content in OPERATOR_THREAD:
        if role == "user":
            conversation.add_user(content)
        else:
            conversation.add_assistant(content)
    return conversation


def _build_registry(primary: Any, secondary: Any, *, primary_family: str, secondary_family: str) -> ProviderRegistry:
    registry = ProviderRegistry()
    capabilities = ProviderCapabilities(completions=True, streaming=True, embeddings=False, tool_calling=True)
    registry.register(
        ProviderDescriptor(
            name="mission_primary",
            default_model=primary.model_name or primary.model,
            priority=10,
            capabilities=capabilities,
            metadata={"provider_family": primary_family},
            factory=lambda **_: primary,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="mission_secondary",
            default_model=secondary.model,
            priority=20,
            capabilities=capabilities,
            metadata={"provider_family": secondary_family},
            factory=lambda **_: secondary.provider,
        )
    )
    return registry


async def _probe_handle(handle: Any) -> dict[str, Any]:
    engine = ExecutionEngine(provider=handle.provider)
    result = await engine.complete(
        RequestSpec(
            provider=handle.name,
            model=handle.model,
            messages=[Message.system("Reply with exactly: ok"), Message.user("ok")],
            max_tokens=8,
        )
    )
    return {
        "provider": handle.name,
        "model": handle.model,
        "status": result.status,
        "ok": result.ok,
        "error": result.error,
    }


async def _resolve_secondary(primary: Any, configured_secondary: Any) -> tuple[Any, dict[str, Any]]:
    configured_probe = await _probe_handle(configured_secondary)
    if configured_probe["ok"]:
        return configured_secondary, {"mode": "configured_secondary", "probe": configured_probe}

    backup_model = {
        "openai": "gpt-5-mini",
        "anthropic": "claude-sonnet-4",
        "google": "gemini-2.5-flash",
    }[primary.name]
    backup_handle = build_provider_handle(primary.name, backup_model)
    backup_probe = await _probe_handle(backup_handle)
    if backup_probe["ok"]:
        return backup_handle, {
            "mode": "same_provider_backup",
            "configured_probe": configured_probe,
            "backup_probe": backup_probe,
        }

    await close_provider(backup_handle.provider)
    fail_or_skip(
        "Mission control could not find a working secondary routing path. "
        f"Configured secondary failed: {configured_probe}. Same-provider backup failed: {backup_probe}."
    )
    raise AssertionError("unreachable")


async def _run_manual_tool(
    *,
    bus: InMemoryEventBus,
    ctx: RequestContext,
    name: str,
    args: dict[str, Any],
    timeout_seconds: float,
    retries: int,
    trust_level: str,
    func,
) -> dict[str, Any]:
    await _publish_tool_event(bus, ctx, event_type=RuntimeEventType.TOOL_START, tool_name=name, arguments=args)
    started = time.perf_counter()
    attempts = 0
    last_error: str | None = None
    while attempts <= retries:
        attempts += 1
        try:
            payload = await asyncio.wait_for(func(**args), timeout=timeout_seconds)
            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            metadata = payload.get("_metadata", {}) if isinstance(payload, dict) else {}
            status = str(metadata.get("status") or "success")
            result_content = payload.get("content", payload) if isinstance(payload, dict) else payload
            record = {
                "tool_name": name,
                "status": status,
                "success": status != "error",
                "attempts": attempts,
                "duration_ms": duration_ms,
                "result": {
                    "content": result_content,
                    "error": None,
                    "metadata": metadata,
                },
                "execution": {
                    "timeout_seconds": timeout_seconds,
                    "retry_attempts": retries,
                    "trust_level": trust_level,
                },
            }
            event_type = RuntimeEventType.TOOL_END if status != "error" else RuntimeEventType.TOOL_ERROR
            await _publish_tool_event(
                bus,
                ctx,
                event_type=event_type,
                tool_name=name,
                arguments=args,
                success=status != "error",
                result=_truncate(result_content, 200),
                error=None,
                duration_ms=duration_ms,
            )
            return record
        except asyncio.TimeoutError:
            last_error = f"Tool '{name}' timed out after {timeout_seconds:.2f}s"
        except Exception as exc:
            last_error = str(exc)

    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    await _publish_tool_event(
        bus,
        ctx,
        event_type=RuntimeEventType.TOOL_ERROR,
        tool_name=name,
        arguments=args,
        success=False,
        error=last_error,
        duration_ms=duration_ms,
    )
    return {
        "tool_name": name,
        "status": "error",
        "success": False,
        "attempts": attempts,
        "duration_ms": duration_ms,
        "result": {
            "content": None,
            "error": last_error,
            "metadata": {},
        },
        "execution": {
            "timeout_seconds": timeout_seconds,
            "retry_attempts": retries,
            "trust_level": trust_level,
        },
    }


async def _metrics_snapshot(service: str) -> dict[str, Any]:
    _ = service
    await asyncio.sleep(0.05)
    return {
        "content": {
            "service": "checkout-api",
            "error_rate": "18.6%",
            "queue_lag": "4m18s",
            "webhook_backlog": 3820,
            "status": "degrading",
        },
        "_metadata": {"status": "success"},
    }


async def _rollout_state(service: str) -> dict[str, Any]:
    _ = service
    await asyncio.sleep(0.08)
    return {
        "content": {
            "service": "checkout-api",
            "latest_change": "payment routing config changed 12 minutes ago",
            "confidence": "partial",
            "note": "rollout metadata is delayed on one control-plane replica",
        },
        "_metadata": {
            "status": "partial",
            "partial": True,
            "warning": "timestamp may lag by one replica heartbeat",
        },
    }


async def _dependency_probe(service: str) -> dict[str, Any]:
    _ = service
    await asyncio.sleep(0.22)
    return {
        "content": {
            "payment_gateway": "degraded",
            "webhook_fanout": "lagging",
            "audit_log_pipeline": "healthy",
        },
        "_metadata": {"status": "success"},
    }


async def _support_pressure(service: str) -> dict[str, Any]:
    _ = service
    await asyncio.sleep(0.04)
    return {
        "content": {
            "open_cases": 23,
            "merchant_reports": 9,
            "top_issue": "delayed payment confirmations",
        },
        "_metadata": {"status": "success"},
    }


async def _sql_impact_estimate(service: str, severity: str) -> dict[str, Any]:
    _ = service
    await asyncio.sleep(0.07)
    return {
        "content": {
            "estimated_failed_transactions": 1842,
            "estimated_finance_exports_at_risk": 146,
            "severity_frame": severity,
            "query": "SELECT failed_tx, export_jobs_at_risk FROM simulated_operational_impact;",
        },
        "_metadata": {"status": "success"},
    }


async def _rollback_readiness(service: str, rollout_confidence: str, dependency_state: str) -> dict[str, Any]:
    _ = service
    await asyncio.sleep(0.06)
    return {
        "content": {
            "rollback_possible": True,
            "needs_webhook_duplication_check": True,
            "dependency_state": dependency_state,
            "rollout_confidence": rollout_confidence,
            "owner": "payments-platform",
        },
        "_metadata": {"status": "success"},
    }


def _deterministic_investigation_summary(
    parallel_results: list[dict[str, Any]],
    sequential_results: list[dict[str, Any]],
) -> dict[str, Any]:
    confirmed: list[str] = []
    degraded: list[str] = []
    unavailable: list[str] = []
    for item in parallel_results:
        if item["status"] == "success" and item["tool_name"] == "metrics_snapshot":
            content = item["result"]["content"]
            confirmed.append(
                f"metrics_snapshot confirms checkout degradation with error_rate={content['error_rate']} and queue_lag={content['queue_lag']}."
            )
        elif item["status"] == "success" and item["tool_name"] == "support_pressure":
            content = item["result"]["content"]
            confirmed.append(
                f"support_pressure confirms rising support load with {content['open_cases']} open cases and {content['merchant_reports']} merchant reports."
            )
        elif item["status"] == "partial":
            content = item["result"]["content"]
            degraded.append(
                f"rollout_state is partial: {content['latest_change']}, but rollout metadata is delayed on one control-plane replica."
            )
        else:
            unavailable.append(f"{item['tool_name']} unavailable: {item['result']['error']}")

    for item in sequential_results:
        if item["tool_name"] == "sql_impact_estimate":
            content = item["result"]["content"]
            confirmed.append(
                f"sql_impact_estimate suggests {content['estimated_failed_transactions']} failed transactions and "
                f"{content['estimated_finance_exports_at_risk']} finance exports at risk."
            )
        if item["tool_name"] == "rollback_readiness":
            content = item["result"]["content"]
            degraded.append(
                "rollback_readiness indicates rollback is possible but requires explicit webhook duplication verification."
            )

    return {
        "overall_status": "active_incident_with_degraded_dependencies",
        "confirmed_signal": confirmed,
        "degraded_evidence": degraded,
        "unavailable_evidence": unavailable,
        "immediate_actions": [
            "Use metrics and support pressure as the confirmed signal base for commander decisions.",
            "Treat rollout timing as degraded evidence only; do not present it as confirmed causality.",
            "Run fallback dependency verification before naming a dependency root cause.",
            "Keep release scope narrow while finance-sensitive workflows remain exposed.",
        ],
        "evidence_used": [item["tool_name"] for item in [*parallel_results, *sequential_results]],
    }


def _normalize_case(text: str | None) -> str:
    return " ".join(str(text or "").lower().split())


def _approval_packet(customer_summary: str) -> dict[str, Any]:
    return {
        "action_id": "approve-customer-safe-update",
        "action_type": "external_customer_update",
        "requested_scope": "customer-safe incident update only",
        "requested_content_preview": _truncate(customer_summary, 180),
        "status": "approved_with_conditions",
        "conditions": [
            "Customer-safe summary only; no release launch language.",
            "Keep dependency root cause unconfirmed until fallback verification completes.",
            "Do not expand rollout scope while month-end finance workflows remain at risk.",
        ],
        "resolver": "mission-control-operator",
    }


def _normalize_classification(data: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(data or {})
    normalized["severity"] = "sev-1"
    owner = str(normalized.get("likely_owner") or "").lower()
    if "payments" not in owner:
        normalized["likely_owner"] = "payments-platform"
    normalized["business_risk"] = str(
        normalized.get("business_risk")
        or (
            "High risk: checkout degradation, rising support pressure, and finance-sensitive export workflows "
            "remain exposed during the month-end window."
        )
    )
    normalized["release_sensitivity"] = str(
        normalized.get("release_sensitivity")
        or "High: broad release scope should pause while incident stabilization and customer-safe messaging remain active."
    )
    normalized["commander_focus"] = str(
        normalized.get("commander_focus")
        or "Stabilize checkout, verify dependencies, evaluate rollback, and hold broad release scope."
    )
    return normalized


def _deterministic_customer_summary(investigation: dict[str, Any]) -> str:
    metrics_line = next(
        (line for line in investigation["confirmed_signal"] if "metrics_snapshot confirms" in line),
        "metrics_snapshot confirms checkout degradation with elevated error rate and queue lag.",
    )
    support_line = next(
        (line for line in investigation["confirmed_signal"] if "support_pressure confirms" in line),
        "support_pressure confirms rising merchant reports of delayed confirmations.",
    )
    return (
        "We are investigating checkout degradation affecting payment completion for a subset of users. "
        f"Current verified indicators show {metrics_line.split(' confirms ', 1)[-1]} "
        f"and {support_line.split(' confirms ', 1)[-1]} "
        "Rollback feasibility and dependency verification are still in progress, so no confirmed root cause is being shared yet. "
        "Sensitive technical details and personal contact information were redacted from this update."
    )


def _judge_deterministic_summary(
    *,
    customer_summary: str,
    operator_summary: str,
    safe_metadata: dict[str, Any],
    judge_data: dict[str, Any] | None,
) -> dict[str, Any]:
    sensitive_values = [
        RAW_METADATA["customer_email"],
        RAW_METADATA["session_token"],
        RAW_METADATA["callback_url"],
    ]
    lowered_customer = _normalize_case(customer_summary)
    lowered_operator = _normalize_case(operator_summary)
    redaction_safe = not any(secret.lower() in lowered_customer for secret in sensitive_values)
    non_speculative = all(
        phrase not in lowered_customer
        for phrase in ("definitely caused", "certain root cause", "proved root cause")
    )
    grounded = any(
        phrase in lowered_customer
        for phrase in ("error_rate=18.6%", "queue_lag=4m18s", "merchant reports", "payment completion")
    )
    operator_useful = "next" in lowered_operator or "action" in lowered_operator
    avg_score = 0.0
    if judge_data is not None:
        avg_score = round(
            sum(
                float(judge_data.get(key, 0))
                for key in (
                    "groundedness_score",
                    "redaction_score",
                    "non_speculation_score",
                    "operator_usefulness_score",
                )
            )
            / 4.0,
            2,
        )
    return {
        "status": "pass" if all((redaction_safe, non_speculative, grounded, operator_useful)) and avg_score >= 80 else "fail",
        "criteria": {
            "redaction_safe": redaction_safe,
            "non_speculative": non_speculative,
            "grounded": grounded,
            "operator_useful": operator_useful,
        },
        "judge_average_score": avg_score,
        "safe_metadata_keys": sorted(safe_metadata.keys()),
    }


def _deterministic_mission_packet(
    *,
    classification: dict[str, Any],
    retrieval_citations: list[dict[str, Any]],
    investigation: dict[str, Any],
    approval: dict[str, Any],
    eval_gate: dict[str, Any],
    customer_summary: str,
    compliance_status: str,
    specialist_outputs: list[dict[str, Any]],
    failover_summary: dict[str, Any],
) -> dict[str, Any]:
    incident_status = "stabilize_checkout_and_verify_dependencies"
    release_recommendation = "hold_broad_release"
    rollout_scope = "internal_only_no_external_launch"
    next_actions = [
        "Complete fallback dependency verification and webhook duplication check.",
        "Prepare the approved customer-safe update and internal finance update.",
        "Keep broad release scope on hold until dependency health and finance workflow risk improve.",
        "Re-run the eval gate after updated investigation facts land.",
    ]
    return {
        "overall_status": "mission_control_ready_with_release_hold",
        "incident_status": incident_status,
        "severity": classification.get("severity", "sev-1"),
        "likely_owner": classification.get("likely_owner", "payments-platform"),
        "business_risk": classification.get("business_risk", "high"),
        "release_sensitivity": classification.get("release_sensitivity", "high"),
        "confirmed_evidence": list(investigation["confirmed_signal"]),
        "degraded_evidence": list(investigation["degraded_evidence"]),
        "unavailable_evidence": list(investigation["unavailable_evidence"]),
        "retrieval_citations": [citation["source_id"] for citation in retrieval_citations],
        "customer_comms_status": "approved_customer_safe_update",
        "customer_safe_summary": customer_summary,
        "compliance_status": compliance_status,
        "approval_status": approval["status"],
        "eval_gate_status": eval_gate["status"],
        "release_recommendation": release_recommendation,
        "rollout_scope": rollout_scope,
        "next_actions": next_actions,
        "artifacts": [
            "triage_update",
            "lead_agent_checkpoint",
            "customer_safe_update",
            "mission_control_memo",
            "sanitized_audit_artifact",
        ],
        "specialist_roles": [item["role"] for item in specialist_outputs],
        "failover_used": bool(failover_summary.get("request_succeeded")),
        "evidence_used": [
            "intake_bundle",
            "retrieval_hits",
            "tool_batch",
            "sequential_tools",
            "lead_agent",
            "specialist_batch",
            "approval_checkpoint",
            "judge_packet",
            "failover_memo",
        ],
    }


def _assembled_summary(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    confirmed = "\n".join(f"- {item}" for item in data.get("confirmed_evidence", []))
    degraded = "\n".join(f"- {item}" for item in data.get("degraded_evidence", []))
    unavailable = "\n".join(f"- {item}" for item in data.get("unavailable_evidence", []))
    actions = "\n".join(f"- {item}" for item in data.get("next_actions", []))
    citations = "\n".join(f"- {item}" for item in data.get("retrieval_citations", []))
    return (
        f"Mission Control Status\n- {data.get('overall_status')}\n- incident_status={data.get('incident_status')} | "
        f"release_recommendation={data.get('release_recommendation')} | eval_gate={data.get('eval_gate_status')}\n\n"
        f"Confirmed Evidence\n{confirmed}\n\n"
        f"Degraded Evidence\n{degraded}\n\n"
        f"Unavailable Evidence\n{unavailable}\n\n"
        f"Customer Comms Status\n- {data.get('customer_comms_status')}\n- {data.get('customer_safe_summary')}\n\n"
        f"Next Actions\n{actions}\n\n"
        f"Retrieval Citations\n{citations}"
    ).strip()


def _combine_usage_dicts(*usage_values: Any) -> dict[str, Any]:
    combined = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }
    for usage in usage_values:
        summary = summarize_usage(usage)
        combined["input_tokens"] += int(summary.get("input_tokens") or 0)
        combined["output_tokens"] += int(summary.get("output_tokens") or 0)
        combined["total_tokens"] += int(summary.get("total_tokens") or 0)
        combined["total_cost"] += float(summary.get("total_cost") or 0.0)
    combined["total_cost"] = round(combined["total_cost"], 8)
    return combined


async def main() -> None:
    chat_handle = build_live_provider()
    embed_handle = build_live_provider(capability="embeddings")
    configured_secondary = build_live_provider(secondary=True)

    extra_providers: list[Any] = []
    if embed_handle.provider is not chat_handle.provider:
        extra_providers.append(embed_handle.provider)
    if configured_secondary.provider is not chat_handle.provider:
        extra_providers.append(configured_secondary.provider)

    event_bus = InMemoryEventBus()
    job_id = "cookbook-end-to-end-mission-control"
    session_id = "cookbook-end-to-end-mission-control"
    replay_recorder = ReplayRecorder(
        event_bus,
        metadata=RunMetadata.create(
            runtime_version="cookbook",
            llm_client_version="local",
            model_version=chat_handle.model,
            config={"scenario": "mission_control"},
            policy=_mission_policy_snapshot(),
            tools=["metrics_snapshot", "rollout_state", "dependency_probe", "support_pressure"],
            job_id=job_id,
            session_id=session_id,
            tags={"example": "34_end_to_end_mission_control"},
        ),
    )

    lifecycle = LifecycleRecorder()
    diagnostics = EngineDiagnosticsRecorder()
    hooks = HookManager([lifecycle, diagnostics])
    chat_engine = ExecutionEngine(provider=chat_handle.provider, hooks=hooks)
    embed_engine = ExecutionEngine(provider=embed_handle.provider, hooks=hooks)

    mission_ctx = RequestContext(session_id=session_id, job_id=job_id, tags={"scenario": "mission-control"})
    primary_state = _InjectedFailureState(failures_remaining=1)
    injected_primary = _FailureInjectedProvider(chat_handle.provider, label="mission_primary", state=primary_state)

    try:
        await replay_recorder.start(
            initial_input={"mission_packet": MISSION_PACKET, "incident_id": MISSION_PACKET["incident_id"]},
            job_id=job_id,
        )
        _debug("bootstrapped recorder")
        await _publish_progress(event_bus, mission_ctx, 0.02, "Bootstrapping mission control state", "bootstrap")

        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)
        memory_notes = await memory.retrieve(MemoryQuery(scope=MISSION_SCOPE, limit=6))

        redaction_policy = RedactionPolicy(
            sensitive_keys=(
                "api_key",
                "apikey",
                "authorization",
                "token",
                "secret",
                "password",
                "access_token",
                "refresh_token",
                "customer_email",
                "session_token",
                "callback_url",
                "internal_channel",
            ),
            provider_payload_capture=ProviderPayloadCaptureMode.METADATA_ONLY,
            preview_mode=PayloadPreviewMode.SUMMARY,
        )
        safe_metadata = sanitize_payload(RAW_METADATA, redaction_policy)
        metadata_preview = preview_payload(RAW_METADATA, redaction_policy)
        redaction_audit = _field_delta_audit(RAW_METADATA, safe_metadata)
        provider_payload_capture = capture_provider_payload(
            {
                "provider": chat_handle.name,
                "model": chat_handle.model,
                "messages": [item["content"] for item in INTAKE_EVIDENCE],
                "authorization": "Bearer demo-secret",
            },
            redaction_policy,
        )
        _debug("redaction complete")

        await _publish_progress(event_bus, mission_ctx, 0.1, "Streaming intake triage", "intake")
        _debug("starting triage stream")
        triage_result, triage_stream = await _stream_completion(
            chat_engine,
            RequestSpec(
                provider=chat_handle.name,
                model=chat_handle.model,
                messages=[
                    Message.system(
                        "You are a mission-control intake assistant. Respond in 4 markdown bullets labeled Situation, Risk, Constraint, Next."
                    ),
                    Message.user(
                        "Create an intake checkpoint from this mission packet, safe metadata, and evidence bundle.\n\n"
                        f"Mission packet: {MISSION_PACKET}\n\n"
                        f"Safe metadata: {safe_metadata}\n\n"
                        f"Evidence: {INTAKE_EVIDENCE}"
                    ),
                ],
            ),
            context=RequestContext(session_id=session_id, job_id="triage-stream"),
        )
        _debug("triage stream complete")

        await _publish_progress(event_bus, mission_ctx, 0.18, "Embedding and indexing mission corpus", "retrieval")
        _debug("starting retrieval indexing")
        probe_vector = await embed_text_or_fail(
            embed_engine,
            KNOWLEDGE_CORPUS[0].text,
            failure_message="Embedding generation failed while sizing the mission-control Qdrant collection.",
        )
        collection = f"cookbook_mission_control_{uuid.uuid4().hex[:10]}"
        retriever = QdrantRetriever(collection=collection, vector_size=len(probe_vector))
        await retriever.recreate_collection()
        indexed_chunks = await _index_corpus(embed_engine, retriever)
        retrieval_query = (
            f"Mission: {MISSION_PACKET['objective']}. "
            f"Business context: {MISSION_PACKET['business_context']}. "
            f"Evidence: {INTAKE_EVIDENCE}."
        )
        query_vector = await embed_text_or_fail(
            embed_engine,
            retrieval_query,
            failure_message="Embedding generation failed for the mission-control retrieval query.",
        )
        hits = await retriever.search(query_vector, limit=5)
        _debug("retrieval search complete")
        retrieval_citations: list[dict[str, Any]] = []
        for index, hit in enumerate(hits, start=1):
            payload = hit.get("payload", {})
            retrieval_citations.append(
                {
                    "citation": f"[{index}]",
                    "source_id": str(payload.get("source_id") or f"DOC-{index}"),
                    "title": payload.get("title"),
                    "source": payload.get("source"),
                    "kind": (payload.get("metadata") or {}).get("kind"),
                    "score": hit.get("score"),
                    "text": payload.get("text"),
                }
            )

        citation_block = "\n".join(
            f"{citation['citation']} {citation['title']} ({citation['source']}): {citation['text']}"
            for citation in retrieval_citations
        )

        classification = await extract_structured(
            chat_handle.provider,
            [
                Message.system(
                    "Classify the mission packet using only the supplied facts. "
                    "Be conservative, non-speculative, and explicit about business risk."
                ),
                Message.user(
                    "Mission packet:\n"
                    f"{MISSION_PACKET}\n\n"
                    f"Safe metadata:\n{safe_metadata}\n\n"
                    f"Triage update:\n{triage_result.content}\n\n"
                    f"Retrieved evidence:\n{citation_block}\n\n"
                    "Return severity, likely_owner, business_risk, release_sensitivity, and commander_focus."
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string"},
                        "likely_owner": {"type": "string"},
                        "business_risk": {"type": "string"},
                        "release_sensitivity": {"type": "string"},
                        "commander_focus": {"type": "string"},
                    },
                    "required": [
                        "severity",
                        "likely_owner",
                        "business_risk",
                        "release_sensitivity",
                        "commander_focus",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )
        classification_data = _normalize_classification(classification.data if classification.valid else {})
        _debug("classification complete")

        await _publish_progress(event_bus, mission_ctx, 0.26, "Managing operator conversation context", "conversation")
        _debug("starting conversation management")
        operator_thread = _conversation_messages()
        operator_thread.config.summarizer = LLMSummarizer(engine=chat_engine)
        operator_thread.config.max_tokens = 140
        operator_thread.config.reserve_tokens = 40
        raw_token_count = operator_thread.count_tokens(chat_handle.provider.model)
        condensed_messages = await operator_thread.get_messages_async(model=chat_handle.provider.model)
        condensed_token_count = sum(
            chat_handle.provider.model.count_tokens(message.content or "") for message in condensed_messages if message.content
        )
        summary_inserted = any("[Earlier context]" in (message.content or "") for message in condensed_messages)

        planner = HeuristicContextPlanner(
            trimming_strategy=TieredTrimmingStrategy(tier1_tail=6),
            memory_reader=memory,
            retrieval_strategy=DefaultMemoryRetrievalStrategy(default_scope=MISSION_SCOPE, default_limit=5),
        )
        assembler = MultiSourceContextAssembler(
            planner=planner,
            source_loaders=[
                _RetrievalSource(retrieval_citations),
                _PolicySource(_mission_policy_snapshot()),
                _LiveStateSource(
                    confirmed=[f"Triage note: {triage_result.content or ''}".strip()],
                    degraded=[],
                    unavailable=[],
                ),
            ],
        )
        base_entries = [
            _Entry(message.role.value, message.content or "", entry_type="operator_thread")
            for message in condensed_messages
            if message.content
        ]
        assembled_context = await assembler.assemble(
            ContextAssemblyRequest(
                current_message=(
                    "Build mission-control context for incident stabilization, customer-safe messaging, and release control."
                ),
                base_entries=base_entries,
                source_request=ContextSourceRequest(current_message="mission control synthesis", scope=MISSION_SCOPE),
                max_entries=10,
                memory_query=MemoryQuery(scope=MISSION_SCOPE, query="mission control synthesis", limit=5),
                max_memory_entries=5,
                metadata={"scenario": "mission-control"},
            )
        )
        _debug("conversation management complete")

        await _publish_progress(event_bus, mission_ctx, 0.38, "Running live investigation tools", "investigation")
        _debug("starting manual tools")
        parallel_ctx = RequestContext(session_id=session_id, job_id="parallel-tools")
        parallel_specs = [
            {
                "name": "metrics_snapshot",
                "args": {"service": MISSION_PACKET["service"]},
                "timeout_seconds": 5.0,
                "retries": 0,
                "trust_level": "high",
                "func": _metrics_snapshot,
            },
            {
                "name": "rollout_state",
                "args": {"service": MISSION_PACKET["service"]},
                "timeout_seconds": 5.0,
                "retries": 0,
                "trust_level": "medium",
                "func": _rollout_state,
            },
            {
                "name": "dependency_probe",
                "args": {"service": MISSION_PACKET["service"]},
                "timeout_seconds": 0.15,
                "retries": 1,
                "trust_level": "low",
                "func": _dependency_probe,
            },
            {
                "name": "support_pressure",
                "args": {"service": MISSION_PACKET["service"]},
                "timeout_seconds": 5.0,
                "retries": 0,
                "trust_level": "high",
                "func": _support_pressure,
            },
        ]
        started_tools = time.perf_counter()
        parallel_results = await asyncio.gather(
            *[
                _run_manual_tool(
                    bus=event_bus,
                    ctx=parallel_ctx,
                    name=spec["name"],
                    args=spec["args"],
                    timeout_seconds=spec["timeout_seconds"],
                    retries=spec["retries"],
                    trust_level=spec["trust_level"],
                    func=spec["func"],
                )
                for spec in parallel_specs
            ]
        )
        parallel_duration_ms = round((time.perf_counter() - started_tools) * 1000.0, 2)

        sequential_ctx = RequestContext(session_id=session_id, job_id="sequential-tools")
        rollout_record = next(item for item in parallel_results if item["tool_name"] == "rollout_state")
        dependency_record = next(item for item in parallel_results if item["tool_name"] == "dependency_probe")
        sequential_results = [
            await _run_manual_tool(
                bus=event_bus,
                ctx=sequential_ctx,
                name="sql_impact_estimate",
                args={
                    "service": MISSION_PACKET["service"],
                    "severity": str(classification_data.get("severity", "sev-1")),
                },
                timeout_seconds=5.0,
                retries=0,
                trust_level="high",
                func=_sql_impact_estimate,
            ),
            await _run_manual_tool(
                bus=event_bus,
                ctx=sequential_ctx,
                name="rollback_readiness",
                args={
                    "service": MISSION_PACKET["service"],
                    "rollout_confidence": str(rollout_record["result"]["content"].get("confidence", "unknown"))
                    if rollout_record["result"]["content"]
                    else "unknown",
                    "dependency_state": "unavailable" if not dependency_record["success"] else "checked",
                },
                timeout_seconds=5.0,
                retries=0,
                trust_level="medium",
                func=_rollback_readiness,
            ),
        ]
        investigation_summary = _deterministic_investigation_summary(parallel_results, sequential_results)
        _debug("manual tools complete")

        await _publish_progress(event_bus, mission_ctx, 0.52, "Running cached digest and idempotent replay", "cache")
        _debug("starting cache/idempotency")
        cache_dir = Path(gettempdir()) / f"cookbook_mission_cache_{uuid.uuid4().hex[:8]}"
        cache_collection = f"mission_control_digest_{uuid.uuid4().hex[:8]}"
        cached_engine = ExecutionEngine(
            provider=chat_handle.provider,
            cache=build_cache_core(
                CacheSettings(
                    backend="fs",
                    client_type="completions",
                    default_collection=cache_collection,
                    cache_dir=cache_dir,
                )
            ),
            hooks=hooks,
        )
        cache_policy = CachePolicy.default_response(collection=cache_collection)
        digest_spec = RequestSpec(
            provider=chat_handle.name,
            model=chat_handle.model,
            messages=[
                Message.system("Summarize the investigation in 4 concise bullets with stable wording."),
                Message.user(
                    f"Classification: {classification_data}\n\n"
                    f"Investigation summary: {investigation_summary}\n\n"
                    f"Retrieved citations: {retrieval_citations}"
                ),
            ],
        )
        cache_cold_ctx = RequestContext(session_id=session_id, job_id="cache-cold")
        cache_warm_ctx = RequestContext(session_id=session_id, job_id="cache-warm")
        cold_started = time.perf_counter()
        cache_cold_result = await cached_engine.complete(digest_spec, context=cache_cold_ctx, cache_policy=cache_policy)
        cold_latency_ms = round((time.perf_counter() - cold_started) * 1000.0, 2)
        warm_started = time.perf_counter()
        cache_warm_result = await cached_engine.complete(digest_spec, context=cache_warm_ctx, cache_policy=cache_policy)
        warm_latency_ms = round((time.perf_counter() - warm_started) * 1000.0, 2)

        idem_engine = ExecutionEngine(
            provider=chat_handle.provider,
            idempotency_tracker=IdempotencyTracker(),
            hooks=hooks,
        )
        idem_spec = RequestSpec(
            provider=chat_handle.name,
            model=chat_handle.model,
            messages=[
                Message.system("Return one sentence with the current operator approval posture."),
                Message.user(
                    f"Active incident classification: {classification_data}. "
                    "State whether external messaging can proceed and whether broad release scope should proceed."
                ),
            ],
        )
        idem_key = f"mission-control-idem-{uuid.uuid4().hex[:8]}"
        idem_first_ctx = RequestContext(session_id=session_id, job_id="idem-first")
        idem_second_ctx = RequestContext(session_id=session_id, job_id="idem-second")
        idem_first_started = time.perf_counter()
        idem_first_result = await idem_engine.complete(idem_spec, context=idem_first_ctx, idempotency_key=idem_key)
        idem_first_latency_ms = round((time.perf_counter() - idem_first_started) * 1000.0, 2)
        idem_second_started = time.perf_counter()
        idem_second_result = await idem_engine.complete(idem_spec, context=idem_second_ctx, idempotency_key=idem_key)
        idem_second_latency_ms = round((time.perf_counter() - idem_second_started) * 1000.0, 2)
        _debug("cache/idempotency complete")

        await _publish_progress(event_bus, mission_ctx, 0.64, "Running lead conversational agent", "lead-agent")
        _debug("starting lead agent")
        commander_tool = Tool(
            name="get_confirmed_signal",
            description="Return the deterministic investigation summary with confirmed, degraded, and unavailable evidence.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda: investigation_summary,
        )
        retrieval_tool = Tool(
            name="get_retrieval_evidence",
            description="Return the retrieved citation block and score-ordered evidence.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda: {
                "citations": retrieval_citations,
                "top_sources": [item["source_id"] for item in retrieval_citations[:3]],
            },
        )
        memory_tool = Tool(
            name="get_memory_rules",
            description="Return the mission-control memory rules and operator preferences.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda: {
                "notes": [{"kind": note.metadata.get("kind"), "content": note.content} for note in memory_notes]
            },
        )
        release_tool = Tool(
            name="get_release_guardrails",
            description="Return the deterministic policy snapshot for release and comms control.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=_mission_policy_snapshot,
        )
        context_tool = Tool(
            name="get_context_plan",
            description="Return the assembled context plan entries and summary.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda: {
                "entries": [
                    {"entry_type": entry.entry_type, "content": entry.content}
                    for entry in assembled_context.plan.entries
                ],
                "summary": assembled_context.plan.summary,
            },
        )

        lead_agent = Agent(
            provider=chat_handle.provider,
            tools=[commander_tool, retrieval_tool, memory_tool, release_tool, context_tool],
            conversation=Conversation(
                system_message=(
                    "You are the lead mission-control conversational agent. "
                    "Call tools before answering. Keep confirmed facts separate from degraded evidence and unknowns. "
                    "Do not present partial rollout timing as confirmed causality."
                ),
                max_tokens=900,
                reserve_tokens=300,
                truncation_strategy="sliding",
            ),
            definition=AgentDefinition(
                name="mission_control_lead",
                system_message=(
                    "You are the lead mission-control conversational agent. "
                    "Use tools to ground every answer before making recommendations."
                ),
                execution_policy=AgentExecutionPolicy(
                    max_turns=4,
                    tool_execution_mode=ToolExecutionMode.PARALLEL,
                    max_tool_calls_per_turn=8,
                ),
                metadata={"scenario": "mission-control"},
            ),
        )
        lead_result = await lead_agent.run(
            "Prepare the first command-post checkpoint with sections Confirmed Signal, Degraded Evidence, Unknowns, Operator Priorities, and Release Scope."
        )
        lead_followup = await lead_agent.run(
            "Now produce a tighter follow-up focused on what customer-safe messaging can say right now and what must wait for verification."
        )
        lead_raw_token_count = lead_agent.conversation.count_tokens(chat_handle.provider.model)
        lead_dispatched_messages = lead_agent.conversation.get_messages(model=chat_handle.provider.model)
        lead_dispatched_token_count = sum(
            chat_handle.provider.model.count_tokens(message.content or "")
            for message in lead_dispatched_messages
            if message.content
        )
        _debug("lead agent complete")

        await _publish_progress(event_bus, mission_ctx, 0.74, "Running specialist batch", "specialists")
        _debug("starting specialist batch")
        specialist_specs = [
            RequestSpec(
                provider=chat_handle.name,
                model=chat_handle.model,
                messages=[
                    Message.system("You are the incident analyst. Respond in 3 bullets labeled Risk, Evidence, Next Action."),
                    Message.user(
                        f"Mission packet: {MISSION_PACKET}\n\nClassification: {classification_data}\n\n"
                        f"Investigation: {investigation_summary}\n\nLead checkpoint: {lead_result.content}"
                    ),
                ],
            ),
            RequestSpec(
                provider=chat_handle.name,
                model=chat_handle.model,
                messages=[
                    Message.system(
                        "You draft customer-safe updates. Respond in 3 bullets labeled Summary, Impact, Redaction Note. "
                        "Avoid speculative root cause language and never include secrets."
                    ),
                    Message.user(
                        f"Safe metadata: {safe_metadata}\n\n"
                        f"Investigation: {investigation_summary}\n\n"
                        f"Lead follow-up: {lead_followup.content}"
                    ),
                ],
            ),
            RequestSpec(
                provider=chat_handle.name,
                model=chat_handle.model,
                messages=[
                    Message.system("You are the compliance reviewer. Respond in 3 bullets labeled Exposure, Guardrail, Audit."),
                    Message.user(
                        f"Safe metadata: {safe_metadata}\n\nRedaction audit: {redaction_audit}\n\n"
                        f"Provider capture policy: {provider_payload_capture}\n\nLead follow-up: {lead_followup.content}"
                    ),
                ],
            ),
            RequestSpec(
                provider=chat_handle.name,
                model=chat_handle.model,
                messages=[
                    Message.system("You are the release risk analyst. Respond in 3 bullets labeled Release Status, Finance Risk, Recommendation."),
                    Message.user(
                        f"Classification: {classification_data}\n\n"
                        f"Investigation summary: {investigation_summary}\n\nPolicy: {_mission_policy_snapshot()}"
                    ),
                ],
            ),
        ]
        specialist_roles = [
            "incident_analyst",
            "customer_comms",
            "compliance_reviewer",
            "release_risk_analyst",
        ]
        specialist_results = await chat_engine.batch_complete(specialist_specs, max_concurrency=4)
        _debug("specialist batch complete")
        specialist_batch = [
            {
                "role": role,
                "status": result.status,
                "usage": summarize_usage(result.usage),
                "content": result.content,
            }
            for role, result in zip(specialist_roles, specialist_results, strict=False)
        ]

        customer_specialist = next(item for item in specialist_batch if item["role"] == "customer_comms")
        customer_summary = _deterministic_customer_summary(investigation_summary)
        approval = _approval_packet(customer_summary)
        await event_bus.publish(
            ActionEvent(
                action_id=approval["action_id"],
                action_type=approval["action_type"],
                payload={
                    "requested_scope": approval["requested_scope"],
                    "requested_content_preview": approval["requested_content_preview"],
                },
            ).to_runtime_event(mission_ctx, type=RuntimeEventType.ACTION_REQUIRED)
        )
        await event_bus.publish(
            ActionEvent(
                action_id=approval["action_id"],
                action_type=approval["action_type"],
                payload={"requested_scope": approval["requested_scope"]},
                resolution={
                    "status": approval["status"],
                    "conditions": approval["conditions"],
                    "resolver": approval["resolver"],
                },
            ).to_runtime_event(mission_ctx, type=RuntimeEventType.ACTION_RESOLVED)
        )

        judge_packet = await extract_structured(
            chat_handle.provider,
            [
                Message.system(
                    "You are an evaluation judge. Score groundedness, redaction safety, non-speculative language, "
                    "and operator usefulness from 0 to 100 and provide one short note."
                ),
                Message.user(
                    f"Operator summary:\n{lead_followup.content}\n\n"
                    f"Customer-safe summary:\n{customer_summary}\n\n"
                    f"Investigation summary:\n{investigation_summary}\n\n"
                    f"Safe metadata:\n{safe_metadata}"
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "groundedness_score": {"type": "number"},
                        "redaction_score": {"type": "number"},
                        "non_speculation_score": {"type": "number"},
                        "operator_usefulness_score": {"type": "number"},
                        "note": {"type": "string"},
                    },
                    "required": [
                        "groundedness_score",
                        "redaction_score",
                        "non_speculation_score",
                        "operator_usefulness_score",
                        "note",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )
        judge_data = judge_packet.data if judge_packet.valid else None
        eval_gate = _judge_deterministic_summary(
            customer_summary=customer_summary,
            operator_summary=lead_followup.content or "",
            safe_metadata=safe_metadata,
            judge_data=judge_data,
        )
        _debug("judge packet complete")

        await _publish_progress(event_bus, mission_ctx, 0.86, "Routing final mission memo with failover", "failover")
        _debug("starting failover memo")
        secondary_handle, secondary_strategy = await _resolve_secondary(chat_handle, configured_secondary)
        gateway_registry = _build_registry(
            injected_primary,
            secondary_handle,
            primary_family=chat_handle.name,
            secondary_family=secondary_handle.name,
        )
        gateway_engine = ExecutionEngine(
            router=RegistryRouter(registry=gateway_registry),
            retry=RetryConfig(attempts=1, backoff=0.0, max_backoff=0.0),
            hooks=hooks,
        )
        memo_spec = RequestSpec(
            provider="auto",
            model="",
            messages=[
                Message.system(
                    "You are the mission-control final presenter. Return markdown with headings: Situation, Decision, Evidence, Approval, Next Actions."
                ),
                Message.user(
                    f"Mission packet: {MISSION_PACKET}\n\n"
                    f"Classification: {classification_data}\n\n"
                    f"Investigation summary: {investigation_summary}\n\n"
                    f"Lead follow-up: {lead_followup.content}\n\n"
                    f"Specialists: {specialist_batch}\n\n"
                    f"Approval: {approval}\n\n"
                    f"Eval gate: {eval_gate}\n\n"
                    f"Retrieval citations: {retrieval_citations}"
                ),
            ],
        )
        failover_context = RequestContext(session_id=session_id, job_id="mission-failover-memo")
        memo_result = await gateway_engine.complete(memo_spec, context=failover_context)
        _debug("failover memo complete")
        failover_report = lifecycle.requests.get(failover_context.request_id)
        failover_summary = {
            "primary_failure": primary_state.last_failure,
            "secondary_strategy": secondary_strategy,
            "fallback_provider_selected": failover_report.provider if failover_report else None,
            "fallback_model_selected": failover_report.model if failover_report else None,
            "request_succeeded": memo_result.ok,
            "request_report": failover_report.to_dict() if failover_report else None,
        }

        mission_packet = _deterministic_mission_packet(
            classification=classification_data,
            retrieval_citations=retrieval_citations,
            investigation=investigation_summary,
            approval=approval,
            eval_gate=eval_gate,
            customer_summary=customer_summary,
            compliance_status="safe" if redaction_audit["safe_for_model"] else "blocked",
            specialist_outputs=specialist_batch,
            failover_summary=failover_summary,
        )

        structured_packet = await extract_structured(
            chat_handle.provider,
            [
                Message.system(
                    "Extract the mission-control packet. Use the supplied memo and facts only."
                ),
                Message.user(
                    f"Mission memo:\n{memo_result.content}\n\n"
                    f"Deterministic mission packet:\n{mission_packet}\n\n"
                    "Return the structured mission-control status."
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "overall_status": {"type": "string"},
                        "incident_status": {"type": "string"},
                        "severity": {"type": "string"},
                        "likely_owner": {"type": "string"},
                        "business_risk": {"type": "string"},
                        "release_sensitivity": {"type": "string"},
                        "confirmed_evidence": {"type": "array", "items": {"type": "string"}},
                        "degraded_evidence": {"type": "array", "items": {"type": "string"}},
                        "unavailable_evidence": {"type": "array", "items": {"type": "string"}},
                        "retrieval_citations": {"type": "array", "items": {"type": "string"}},
                        "customer_comms_status": {"type": "string"},
                        "customer_safe_summary": {"type": "string"},
                        "compliance_status": {"type": "string"},
                        "approval_status": {"type": "string"},
                        "eval_gate_status": {"type": "string"},
                        "release_recommendation": {"type": "string"},
                        "rollout_scope": {"type": "string"},
                        "next_actions": {"type": "array", "items": {"type": "string"}},
                        "artifacts": {"type": "array", "items": {"type": "string"}},
                        "specialist_roles": {"type": "array", "items": {"type": "string"}},
                        "failover_used": {"type": "boolean"},
                        "evidence_used": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "overall_status",
                        "incident_status",
                        "severity",
                        "likely_owner",
                        "business_risk",
                        "release_sensitivity",
                        "confirmed_evidence",
                        "degraded_evidence",
                        "unavailable_evidence",
                        "retrieval_citations",
                        "customer_comms_status",
                        "customer_safe_summary",
                        "compliance_status",
                        "approval_status",
                        "eval_gate_status",
                        "release_recommendation",
                        "rollout_scope",
                        "next_actions",
                        "artifacts",
                        "specialist_roles",
                        "failover_used",
                        "evidence_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )
        _debug("structured packet complete")
        structured_data = dict(structured_packet.data or {})
        structured_data.update(mission_packet)

        artifact_payload = sanitize_log_data(
            {
                "mission_packet": MISSION_PACKET,
                "safe_metadata": safe_metadata,
                "provider_payload": {
                    "provider": chat_handle.name,
                    "model": chat_handle.model,
                    "messages": [item["content"] for item in INTAKE_EVIDENCE],
                    "authorization": "Bearer demo-secret",
                },
                "tool_batch": parallel_results,
                "sequential_tools": sequential_results,
                "mission_control_status": structured_data,
                "request_report": failover_summary["request_report"],
            },
            redaction_policy,
        )

        sanitized_tool_output = sanitize_tool_output(
            {
                "tool_batch": parallel_results,
                "sequential_tools": sequential_results,
                "customer_summary": customer_summary,
            },
            ToolOutputPolicy(),
        )
        await event_bus.publish(
            ArtifactEvent(
                artifact_id="artifact-mission-audit",
                artifact_type="json",
                name="mission_control_audit.json",
                content_preview=_truncate(json.dumps(artifact_payload, ensure_ascii=True), 220),
            ).to_runtime_event(mission_ctx, type=RuntimeEventType.ARTIFACT_CREATED)
        )

        memory_writes = [
            MemoryWrite(
                scope=MISSION_SCOPE,
                content=json.dumps(structured_data, ensure_ascii=True, sort_keys=True),
                relevance=0.99,
                metadata={"kind": "mission_control_status"},
            ),
            MemoryWrite(
                scope=MISSION_SCOPE,
                content=json.dumps(eval_gate, ensure_ascii=True, sort_keys=True),
                relevance=0.94,
                metadata={"kind": "eval_gate"},
            ),
            MemoryWrite(
                scope=MISSION_SCOPE,
                content=json.dumps(approval, ensure_ascii=True, sort_keys=True),
                relevance=0.93,
                metadata={"kind": "approval_status"},
            ),
        ]
        for entry in memory_writes:
            await memory.write(entry)
        _debug("memory persistence complete")
        memory_after_action = [
            {"kind": record.metadata.get("kind"), "content": record.content}
            for record in await memory.retrieve(MemoryQuery(scope=MISSION_SCOPE, limit=8))
        ]

        citation_audit = _citation_audit(retrieval_citations, structured_data.get("retrieval_citations", []))
        request_report = lifecycle.requests.get(mission_ctx.request_id)
        session_report = lifecycle.sessions.get(session_id)
        diagnostics_snapshot = diagnostics.latest_request(failover_context.request_id)
        await event_bus.publish(
            FinalEvent(
                content=_assembled_summary(structured_data),
                status="success",
                usage=summarize_usage(memo_result.usage),
            ).to_runtime_event(mission_ctx, type=RuntimeEventType.FINAL_RESULT)
        )
        await _publish_progress(event_bus, mission_ctx, 1.0, "Mission control run complete", "complete")

        recording = await replay_recorder.stop()
        _debug("replay stopped")
        replay_valid, replay_error = recording.validate_chain()
        replay_event_counts = Counter(event.event.type.value for event in recording.events)

        output = {
            "provider": chat_handle.name,
            "model": chat_handle.model,
            "embeddings_provider": {"provider": embed_handle.name, "model": embed_handle.model},
            "scenario_packet": MISSION_PACKET,
            "raw_metadata": RAW_METADATA,
            "safe_metadata": safe_metadata,
            "metadata_preview": metadata_preview,
            "redaction_audit": redaction_audit,
            "provider_payload_capture": provider_payload_capture,
            "memory_bootstrap": memory_bootstrap,
            "triage_stream": {
                "status": triage_result.status,
                "usage": summarize_usage(triage_result.usage),
                "content": triage_result.content,
                "stream_summary": triage_stream,
            },
            "retrieval": {
                "qdrant_collection": collection,
                "indexed_chunks": indexed_chunks,
                "retrieval_query": retrieval_query,
                "retrieved_citations": retrieval_citations,
                "citation_audit": citation_audit,
            },
            "classification": {
                "valid": classification.valid,
                "repair_attempts": classification.repair_attempts,
                "usage": summarize_usage(classification.usage),
                "data": classification_data,
            },
            "conversation_management": {
                "operator_thread_message_count": len(OPERATOR_THREAD),
                "raw_token_count": raw_token_count,
                "condensed_token_count": condensed_token_count,
                "summary_inserted": summary_inserted,
                "condensed_messages": _message_preview(condensed_messages),
                "assembled_context": {
                    "selected_entries": [
                        {"entry_type": getattr(entry, "entry_type", "entry"), "content": entry.content}
                        for entry in assembled_context.plan.entries
                    ],
                    "memory": [
                        {"kind": record.metadata.get("kind"), "content": record.content}
                        for record in assembled_context.plan.memory
                    ],
                    "summary": assembled_context.plan.summary,
                },
            },
            "tool_batch": {
                "mode": "parallel_then_sequential",
                "parallel_results": parallel_results,
                "parallel_duration_ms": parallel_duration_ms,
                "status_counts": dict(Counter(item["status"] for item in parallel_results)),
                "sequential_results": sequential_results,
            },
            "deterministic_investigation_summary": investigation_summary,
            "cache_and_idempotency": {
                "fs_cache": {
                    "collection": cache_collection,
                    "cache_dir": str(cache_dir),
                    "cold_latency_ms": cold_latency_ms,
                    "warm_latency_ms": warm_latency_ms,
                    "cache_hit": bool((lifecycle.requests.get(cache_warm_ctx.request_id) or {}).cache_hit)
                    if lifecycle.requests.get(cache_warm_ctx.request_id)
                    else False,
                    "cold_usage": summarize_usage(cache_cold_result.usage),
                    "warm_usage": summarize_usage(cache_warm_result.usage),
                    "cache_stats": cached_engine.cache.get_stats().to_dict() if cached_engine.cache else {},
                },
                "idempotency_replay": {
                    "first_latency_ms": idem_first_latency_ms,
                    "second_latency_ms": idem_second_latency_ms,
                    "idempotency_hit": bool((lifecycle.requests.get(idem_second_ctx.request_id) or {}).idempotency_hit)
                    if lifecycle.requests.get(idem_second_ctx.request_id)
                    else False,
                    "same_content": (idem_first_result.content or "") == (idem_second_result.content or ""),
                    "content_excerpt": _truncate(idem_second_result.content, 180),
                },
            },
            "lead_agent": {
                "status": lead_followup.status,
                "turn_count": len(lead_agent.conversation),
                "tool_call_count": len(lead_result.all_tool_calls) + len(lead_followup.all_tool_calls),
                "usage": _combine_usage_dicts(lead_result.total_usage, lead_followup.total_usage),
                "initial_checkpoint": lead_result.content,
                "followup_checkpoint": lead_followup.content,
                "turns": _serialize_turns([*lead_result.turns, *lead_followup.turns]),
                "conversation_cap": {
                    "raw_token_count": lead_raw_token_count,
                    "dispatched_token_count": lead_dispatched_token_count,
                    "stored_message_count": len(list(lead_agent.conversation)),
                    "dispatched_message_count": len(lead_dispatched_messages),
                    "messages_preview": _message_preview(lead_dispatched_messages),
                },
            },
            "specialist_batch": specialist_batch,
            "approval_checkpoint": approval,
            "judge_packet": {
                "valid": judge_packet.valid,
                "repair_attempts": judge_packet.repair_attempts,
                "usage": summarize_usage(judge_packet.usage),
                "model_packet": judge_data,
                "deterministic_eval_gate": eval_gate,
            },
            "failover_route": {
                "stream_summary": None,
                "memo_status": memo_result.status,
                "memo_usage": summarize_usage(memo_result.usage),
                "memo_content": memo_result.content,
                "gateway_story": failover_summary,
            },
            "structured_packet": {
                "valid": structured_packet.valid,
                "repair_attempts": structured_packet.repair_attempts,
                "usage": summarize_usage(structured_packet.usage),
                "data": structured_data,
            },
            "sanitized_audit_artifact": artifact_payload,
            "sanitized_tool_output": sanitized_tool_output,
            "assembled_summary": _assembled_summary(structured_data),
            "observability": {
                "request_report": request_report.to_dict() if request_report else None,
                "failover_request_report": failover_summary["request_report"],
                "diagnostics": diagnostics_snapshot.payload if diagnostics_snapshot else None,
                "session_report": session_report.to_dict() if session_report else None,
            },
            "replay_recording": {
                "event_count": recording.metadata.event_count,
                "model_response_hashes": recording.metadata.model_response_hashes,
                "valid_chain": replay_valid,
                "chain_error": replay_error,
                "event_type_counts": dict(replay_event_counts),
            },
            "memory_after_action": memory_after_action,
            "showcase_verdict": {
                "streamed_triage": triage_result.status == 200,
                "embeddings_rag_used": bool(retrieval_citations),
                "context_managed": bool(assembled_context.plan.entries) and (
                    summary_inserted or lead_dispatched_token_count < lead_raw_token_count
                ),
                "parallel_and_sequential_tools": bool(parallel_results) and bool(sequential_results),
                "conversational_agent_run": str(lead_followup.status).lower() in {"200", "success"}
                and bool(lead_followup.content),
                "specialist_batch_run": len(specialist_batch) == 4 and all(item["status"] == 200 for item in specialist_batch),
                "cache_and_idempotency_used": (
                    bool((lifecycle.requests.get(cache_warm_ctx.request_id) or {}).cache_hit)
                    if lifecycle.requests.get(cache_warm_ctx.request_id)
                    else False
                )
                and (
                    bool((lifecycle.requests.get(idem_second_ctx.request_id) or {}).idempotency_hit)
                    if lifecycle.requests.get(idem_second_ctx.request_id)
                    else False
                ),
                "failover_routed": bool(failover_summary["request_succeeded"])
                and bool(primary_state.last_failure)
                and bool((failover_summary["request_report"] or {}).get("fallbacks", 0) >= 1),
                "progress_recorded": recording.metadata.event_count > 0 and replay_valid,
                "structured_outputs_validated": classification.valid and judge_packet.valid and structured_packet.valid,
                "operator_ready": structured_data.get("release_recommendation") == "hold_broad_release"
                and structured_data.get("customer_comms_status") == "approved_customer_safe_update",
            },
        }

        print_heading("End To End Mission Control")
        print_json(output)
    finally:
        if replay_recorder.is_recording:
            await replay_recorder.stop()
        await event_bus.close()
        await close_provider(chat_handle.provider)
        for provider in extra_providers:
            await close_provider(provider)


if __name__ == "__main__":
    asyncio.run(main())
