from __future__ import annotations

import asyncio
import uuid

from cookbook_support import (
    build_live_provider,
    close_provider,
    print_heading,
    print_json,
    summarize_usage,
)
from cookbook_expansion_support import QdrantRetriever, RetrieverDocument, chunk_text, embed_text_or_fail, excerpt

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec
from llm_client.structured import StructuredOutputConfig, extract_structured


CORPUS = [
    RetrieverDocument(
        doc_id="runbook_1",
        title="Checkout Incident Runbook",
        text=(
            "If checkout 5xx spikes after a routing or config change, pause further changes, "
            "evaluate safe rollback, inspect queue lag, and confirm webhook backlog before external communications."
        ),
        source="runbook://checkout-incident",
        metadata={"kind": "runbook"},
    ),
    RetrieverDocument(
        doc_id="runbook_2",
        title="Communications Guidance",
        text=(
            "Customer updates should acknowledge impact, state one concrete next action, and avoid speculative root-cause claims. "
            "Leadership prefers proactive no-surprises internal updates before escalation."
        ),
        source="runbook://communications-guidance",
        metadata={"kind": "communications"},
    ),
    RetrieverDocument(
        doc_id="postmortem_1",
        title="Prior Export Backpressure Postmortem",
        text=(
            "A previous finance export incident showed audit-log fanout can amplify queue backpressure without immediate data loss. "
            "Observed symptoms included rising queue depth and delayed downstream webhooks."
        ),
        source="postmortem://finance-export-backpressure",
        metadata={"kind": "postmortem"},
    ),
    RetrieverDocument(
        doc_id="runbook_3",
        title="Revenue Operations Escalation Thresholds",
        text=(
            "When checkout degradation occurs near month-end, revenue operations and finance stakeholders should receive "
            "a no-surprises internal update before any external customer communication. Escalation severity should bias upward "
            "if settlement, export, or reconciliation workflows may be affected."
        ),
        source="runbook://revops-escalation-thresholds",
        metadata={"kind": "stakeholder"},
    ),
]


INCIDENT_PACKET = {
    "service": "checkout-api",
    "business_impact": "Payment completion is degraded for a subset of users and finance is nearing month-end reconciliation.",
    "symptoms": [
        "5xx errors spiked after a payment routing change",
        "checkout queue lag is rising",
        "support reports intermittent webhook delays",
    ],
    "recent_change": "payment routing config changed 12 minutes ago",
}


def _citation_audit(citations: list[dict], structured_data: dict | None) -> dict[str, object]:
    available = {index for index, _ in enumerate(citations, start=1)}
    cited = {int(item) for item in list((structured_data or {}).get("citations_used") or [])}
    missing = sorted(cited - available)
    unused = sorted(available - cited)
    return {
        "available_citations": sorted(available),
        "citations_used": sorted(cited),
        "all_citations_resolved": len(missing) == 0,
        "missing_citations": missing,
        "unused_citations": unused,
    }


def _assembled_brief(data: dict | None) -> str | None:
    if not data:
        return None
    evidence_lines = "\n".join(f"- {line}" for line in list(data.get("evidence_bullets") or []))
    citations = ", ".join(f"[{item}]" for item in list(data.get("citations_used") or []))
    return (
        f"Situation\n- {data.get('situation_summary')}\n\n"
        f"Evidence\n{evidence_lines}\n\n"
        f"Recommended Next Action\n- {data.get('recommended_next_action')}\n\n"
        f"Communication Guardrail\n- {data.get('communication_guardrail')}\n\n"
        f"Citations Used\n- {citations or 'None'}"
    )


async def _index_corpus(embed_engine: ExecutionEngine, retriever: QdrantRetriever) -> list[dict]:
    points: list[dict] = []
    indexed_chunks: list[dict] = []
    point_id = 1
    for document in CORPUS:
        for chunk_index, chunk in enumerate(chunk_text(document.text, max_chars=180)):
            vector = await embed_text_or_fail(
                embed_engine,
                chunk,
                failure_message="Embedding generation failed while preparing the retrieval corpus.",
            )
            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "doc_id": document.doc_id,
                        "title": document.title,
                        "source": document.source,
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
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
            point_id += 1
    await retriever.upsert(points)
    return indexed_chunks


async def main() -> None:
    chat_handle = build_live_provider()
    embed_handle = build_live_provider(capability="embeddings")
    collection = f"cookbook_rag_{uuid.uuid4().hex[:10]}"
    try:
        chat_engine = ExecutionEngine(provider=chat_handle.provider)
        embed_engine = ExecutionEngine(provider=embed_handle.provider)
        probe_vector = await embed_text_or_fail(
            embed_engine,
            CORPUS[0].text,
            failure_message="Embedding generation failed while sizing the Qdrant collection.",
        )
        retriever = QdrantRetriever(collection=collection, vector_size=len(probe_vector))
        await retriever.recreate_collection()
        indexed_chunks = await _index_corpus(embed_engine, retriever)

        question = (
            f"Prepare a grounded incident brief for this packet: {INCIDENT_PACKET}. "
            "Use runbook guidance and prior evidence only."
        )
        query_vector = await embed_text_or_fail(
            embed_engine,
            question,
            failure_message="Embedding generation failed for the retrieval query.",
        )
        hits = await retriever.search(query_vector, limit=4)
        citations = []
        for index, hit in enumerate(hits, start=1):
            payload = hit.get("payload", {})
            citations.append(
                {
                    "citation": f"[{index}]",
                    "title": payload.get("title"),
                    "source": payload.get("source"),
                    "score": hit.get("score"),
                    "kind": (payload.get("metadata") or {}).get("kind"),
                    "text": payload.get("text"),
                }
            )

        citation_block = "\n".join(
            f"{item['citation']} {item['title']} ({item['source']}): {item['text']}" for item in citations
        )
        structured = await extract_structured(
            chat_handle.provider,
            [
                Message.system(
                    "Use only the retrieved evidence. Produce a grounded incident brief. "
                    "Every evidence bullet must include one or more [n] citation markers tied to the supplied sources."
                ),
                Message.user(
                    f"Incident packet: {INCIDENT_PACKET}\n\nQuestion: {question}\n\nRetrieved evidence:\n{citation_block}\n\n"
                    "Return structured output with a concise situation summary, evidence bullets, "
                    "one recommended next action, one communication guardrail, and an explicit citations_used list."
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "situation_summary": {"type": "string"},
                        "evidence_bullets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                        "recommended_next_action": {"type": "string"},
                        "communication_guardrail": {"type": "string"},
                        "citations_used": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                            "minItems": 1,
                        },
                    },
                    "required": [
                        "situation_summary",
                        "evidence_bullets",
                        "recommended_next_action",
                        "communication_guardrail",
                        "citations_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            )
        )
        data = structured.data if structured.valid else None
        answer_text = _assembled_brief(data)
        citation_audit = _citation_audit(citations, data)
        score_order = [
            {
                "citation": item["citation"],
                "score": item["score"],
                "kind": item["kind"],
                "source": item["source"],
                "excerpt": excerpt(item["text"], limit=120),
            }
            for item in citations
        ]

        print_heading("RAG With Citations")
        print_json(
            {
                "chat_provider": {"provider": chat_handle.name, "model": chat_handle.model},
                "embeddings_provider": {"provider": embed_handle.name, "model": embed_handle.model},
                "incident_packet": INCIDENT_PACKET,
                "qdrant_collection": collection,
                "indexed_chunks": indexed_chunks,
                "retrieval_summary": {
                    "query": question,
                    "hit_count": len(citations),
                    "score_order": score_order,
                },
                "retrieved_citations": citations,
                "citation_audit": citation_audit,
                "structured_answer": {
                    "valid": structured.valid,
                    "repair_attempts": structured.repair_attempts,
                    "usage": summarize_usage(getattr(structured, "usage", None)),
                    "data": data,
                },
                "assembled_brief": answer_text,
                "showcase_verdict": {
                    "grounded": citation_audit["all_citations_resolved"],
                    "uses_multiple_sources": len(set(item["source"] for item in citations)) >= 2,
                    "ready_for_operator_use": bool(answer_text),
                },
            }
        )
    finally:
        await close_provider(chat_handle.provider)
        if embed_handle.provider is not chat_handle.provider:
            await close_provider(embed_handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
