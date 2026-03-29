from __future__ import annotations

import asyncio
import uuid
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage
from cookbook_expansion_support import QdrantRetriever, RetrieverDocument, chunk_text, embed_text_or_fail

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.redaction import RedactionPolicy, sanitize_payload
from llm_client.structured import StructuredOutputConfig, extract_structured


HISTORY = [
    RetrieverDocument(
        doc_id="ticket_1",
        title="Prior export timeout escalation",
        text=(
            "Workspace export jobs timed out after audit logging increased write fanout. "
            "Recommended owner was data platform and customer messaging emphasized impact plus next update timing."
        ),
        source="ticket://SUP-1821",
        metadata={"kind": "prior_ticket"},
    ),
    RetrieverDocument(
        doc_id="ticket_2",
        title="Webhook lag after routing update",
        text=(
            "Payment routing changes caused webhook lag without data loss. "
            "Primary mitigation was rollback evaluation and queue drain monitoring."
        ),
        source="ticket://SUP-1732",
        metadata={"kind": "prior_ticket"},
    ),
    RetrieverDocument(
        doc_id="runbook_1",
        title="Enterprise support comms guidance",
        text=(
            "Enterprise incident replies should acknowledge impact, state the active mitigation, provide the next update time, "
            "and avoid unconfirmed root-cause claims."
        ),
        source="runbook://support-comms-enterprise",
        metadata={"kind": "runbook"},
    ),
    RetrieverDocument(
        doc_id="runbook_2",
        title="Month-end reconciliation escalation",
        text=(
            "If finance or compliance workflows are blocked near month-end, route to Data Platform ownership, "
            "flag executive visibility, and keep Support informed with a short internal brief."
        ),
        source="runbook://month-end-reconciliation",
        metadata={"kind": "runbook"},
    ),
]

SUPPORT_REDACTION_POLICY = RedactionPolicy(
    sensitive_keys=(
        "api_key",
        "apikey",
        "authorization",
        "token",
        "secret",
        "password",
        "access_token",
        "refresh_token",
        "auth_token",
        "customer_email",
        "email",
    )
)


def _redaction_audit(raw_packet: dict[str, Any], safe_packet: dict[str, Any]) -> dict[str, Any]:
    redacted_fields: list[str] = []
    transformed_fields: list[str] = []
    for key, raw_value in raw_packet.items():
        safe_value = safe_packet.get(key)
        if safe_value != raw_value:
            transformed_fields.append(key)
            if raw_value is not None and safe_value in {None, "<redacted>", "[REDACTED]"}:
                redacted_fields.append(key)
    return {
        "fields_seen": sorted(raw_packet.keys()),
        "transformed_fields": sorted(transformed_fields),
        "redacted_fields": sorted(redacted_fields),
        "safe_for_model": safe_packet != raw_packet,
    }


def _retrieval_summary(retrieval_context: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hit_count": len(retrieval_context),
        "sources": [
            {
                "source": item.get("source"),
                "title": item.get("title"),
                "kind": item.get("kind"),
                "score": item.get("score"),
            }
            for item in retrieval_context
        ],
    }


def _assembled_summary(result: dict[str, Any]) -> str:
    return (
        "Internal Triage Note\n"
        f"- Severity: {result['severity']}\n"
        f"- Owning Team: {result['owning_team']}\n"
        f"- Escalation: {result['escalation_recommendation']}\n"
        f"- Customer Ready: {result['customer_ready']}\n\n"
        "Internal Note\n"
        f"- {result['internal_note']}\n\n"
        "Customer Reply Draft\n"
        f"- {result['customer_reply']}"
    )


async def main() -> None:
    chat_handle = build_live_provider()
    embed_handle = build_live_provider(capability="embeddings")
    collection = f"copilot_{uuid.uuid4().hex[:10]}"
    try:
        packet = {
            "case_id": "SUP-7014",
            "customer_email": "ops-lead@acme.example",
            "auth_token": "tok_demo_sensitive",
            "customer_tier": "enterprise",
            "issue_summary": "Workspace export jobs started timing out after audit logging was enabled.",
            "business_impact": "Finance and compliance teams cannot complete month-end reconciliation.",
            "symptoms": [
                "timeouts begin near the 5-minute mark",
                "queue drains slowly after large exports",
                "support suspects audit-log fanout overhead",
            ],
        }
        safe_packet = sanitize_payload(packet, policy=SUPPORT_REDACTION_POLICY)
        redaction_audit = _redaction_audit(packet, safe_packet)

        chat_engine = ExecutionEngine(provider=chat_handle.provider)
        embed_engine = ExecutionEngine(provider=embed_handle.provider)
        probe_vector = await embed_text_or_fail(
            embed_engine,
            HISTORY[0].text,
            failure_message="Embedding generation failed while sizing the Qdrant collection.",
        )
        retriever = QdrantRetriever(collection=collection, vector_size=len(probe_vector))
        await retriever.recreate_collection()
        points = []
        point_id = 1
        for doc in HISTORY:
            for chunk_index, chunk in enumerate(chunk_text(doc.text, max_chars=180)):
                embedding = await embed_text_or_fail(
                    embed_engine,
                    chunk,
                    failure_message="Embedding generation failed while preparing support history.",
                )
                points.append(
                    {
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "title": doc.title,
                            "source": doc.source,
                            "text": chunk,
                            "metadata": doc.metadata,
                            "chunk_index": chunk_index,
                        },
                    }
                )
                point_id += 1
        await retriever.upsert(points)

        query_text = f"{packet['issue_summary']} {' '.join(packet['symptoms'])}"
        query_vector = await embed_text_or_fail(
            embed_engine,
            query_text,
            failure_message="Embedding generation failed for support retrieval.",
        )
        hits = await retriever.search(
            query_vector,
            limit=3,
        )
        retrieval_context = [
            {
                "source": hit.get("payload", {}).get("source"),
                "title": hit.get("payload", {}).get("title"),
                "text": hit.get("payload", {}).get("text"),
                "kind": hit.get("payload", {}).get("metadata", {}).get("kind"),
                "score": hit.get("score"),
            }
            for hit in hits
        ]

        structured = await extract_structured(
            chat_handle.provider,
            [
                Message.system(
                    "Produce a support-copilot routing decision. Use only the sanitized packet and retrieved history. "
                    "Keep internal notes concise and operator-ready. Keep the customer reply factual, calm, and free of unconfirmed root-cause claims."
                ),
                Message.user(
                    f"Packet: {safe_packet}\n\nRetrieved history: {retrieval_context}\n\n"
                    "Return severity, owning_team, escalation_recommendation, customer_ready, internal_note, customer_reply."
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string"},
                        "owning_team": {"type": "string"},
                        "escalation_recommendation": {"type": "string"},
                        "customer_ready": {"type": "boolean"},
                        "internal_note": {"type": "string"},
                        "customer_reply": {"type": "string"},
                    },
                    "required": [
                        "severity",
                        "owning_team",
                        "escalation_recommendation",
                        "customer_ready",
                        "internal_note",
                        "customer_reply",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )

        assembled_summary = _assembled_summary(structured.data or {})

        print_heading("Customer Support Copilot")
        print_json(
            {
                "chat_provider": {"provider": chat_handle.name, "model": chat_handle.model},
                "embeddings_provider": {"provider": embed_handle.name, "model": embed_handle.model},
                "raw_packet_fields": sorted(packet.keys()),
                "safe_packet": safe_packet,
                "redaction_audit": redaction_audit,
                "retrieval_summary": _retrieval_summary(retrieval_context),
                "retrieval_context": retrieval_context,
                "copilot_result": {
                    "valid": structured.valid,
                    "repair_attempts": structured.repair_attempts,
                    "usage": summarize_usage(getattr(structured, "usage", None)),
                    "data": structured.data,
                },
                "assembled_summary": assembled_summary,
                "showcase_verdict": {
                    "sanitized_before_model": redaction_audit["safe_for_model"],
                    "retrieval_backed": len(retrieval_context) >= 2,
                    "operator_ready": bool(structured.data and structured.data.get("internal_note")),
                    "customer_reply_present": bool(structured.data and structured.data.get("customer_reply")),
                },
            }
        )
    finally:
        await close_provider(chat_handle.provider)
        if embed_handle.provider is not chat_handle.provider:
            await close_provider(embed_handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
