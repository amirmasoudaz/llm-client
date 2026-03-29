from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import replace
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.content import (
    AudioBlock,
    ContentHandlingMode,
    ContentMessage,
    ContentRequestEnvelope,
    FileBlock,
    ImageBlock,
    MetadataBlock,
    TextBlock,
    ensure_completion_result,
    ensure_content_response_envelope,
    project_content_blocks,
)
from llm_client.engine import ExecutionEngine
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import CompletionResult, Message, Role, StreamEventType
from llm_client.redaction import PayloadPreviewMode, RedactionPolicy, preview_payload, sanitize_payload
from llm_client.spec import RequestContext
from llm_client.structured import StructuredOutputConfig, extract_structured


INTAKE_SCOPE = "multimodal-intake-pipeline"
INCIDENT_PACKET = {
    "incident_id": "INC-4821",
    "service": "checkout-api",
    "business_impact": "payment completion is degraded for a subset of users after a routing change",
    "suspected_owner": "payments-platform",
    "severity_hint": "sev-1",
}

CITATION_MODALITY_MAP = {
    "TXT-1": "text",
    "TXT-2": "text",
    "AUD-1": "audio",
    "IMG-1": "image",
    "FILE-1": "file",
    "META-1": "metadata",
    "MEM-1": "memory",
}

IMAGE_METRIC_MARKERS = (
    "18.2%",
    "5m12s",
    "5m 12s",
    "14,220",
    "queue depth",
    "webhook confirmation lag",
    "error rate",
)


def _dashboard_image_data_url() -> str:
    # Small provider-safe PNG data URL; the attached `IMG-1` text block carries
    # the human-readable dashboard observations for grounded citation.
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAABUElEQVR4nO3csU7CUBhA4WJI3HwFu8liFxsHmRjcfAxW"
        "n8DZJ3B1cfAN3ByccCCw4KIbrq4MrA4uxLacvyq0xPNNFG7DzcmluSSUznLxkajaXtMTaDsDAQOB7upB7+ykqXm0zev"
        "z9OuBKwh0i0/NJ7Ptz6M90jxbPXQFAQMBAwEDAQMBAwEDgZJ9UFGaZzu6Obo6TiPDrl/mVS+5goCBgIGAgYCBgIGAgY"
        "CBgIFAaCfdrPeni8iww8HDJt7dFQQMBAwEDAQMBAwEDAS2tA96uzmPDDu6fNz0TOpyBQEDAQMBAwEDAQMBA4Ea+6Dxs"
        "B8Zdno7+ulk2sgVBAwEDAQMBAwEDAQMBAwEDAQMBAwEDAQMBKLf5r/dRbVOfGTR3S/OTcrO3T+InHmfZ0nFjXKhQDv6"
        "K/I/4UcMGAgYCJRcg2pcj/8BVxDo+Ocm67mCgIHAJ3T3I7SyunHNAAAAAElFTkSuQmCC"
    )


def _redaction_audit(raw_packet: dict[str, Any], safe_packet: dict[str, Any]) -> dict[str, Any]:
    transformed_fields: list[str] = []
    for key, original_value in raw_packet.items():
        if safe_packet.get(key) != original_value:
            transformed_fields.append(key)
    return {
        "fields_seen": sorted(raw_packet.keys()),
        "transformed_fields": sorted(transformed_fields),
        "redacted_fields": sorted(
            [key for key in transformed_fields if safe_packet.get(key) == "[REDACTED]"]
        ),
        "safe_for_model": not any(
            safe_packet.get(key) == raw_packet.get(key) and key in {"customer_email", "session_token", "callback_url"}
            for key in raw_packet
        ),
    }


def _projection_summary(provider_name: str, source_blocks: list[object]) -> dict[str, object]:
    normalized_provider = provider_name.lower()
    projection_kwargs: dict[str, Any] = {
        "provider": provider_name,
        "mode": ContentHandlingMode.LOSSY,
        "include_metadata": True,
    }
    if normalized_provider == "openai":
        projection_kwargs.update(
            {
                "supports_images": True,
                "supports_audio_data": True,
                "supports_audio_url": False,
                "supports_files": False,
            }
        )
    projection = project_content_blocks(source_blocks, **projection_kwargs)
    return {
        "projected_block_count": len(projection.blocks),
        "projected_blocks": [block.to_dict() for block in projection.blocks],
        "degradations": [item.reason for item in projection.degradations],
    }


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    seed_entries = [
        MemoryWrite(
            scope=INTAKE_SCOPE,
            content="Payments incidents with visible customer impact default to payments-platform unless evidence points elsewhere.",
            relevance=0.94,
            metadata={"kind": "triage_rule"},
        ),
        MemoryWrite(
            scope=INTAKE_SCOPE,
            content="Customer-safe summaries must exclude session tokens, callback URLs, and personal contact details.",
            relevance=0.96,
            metadata={"kind": "privacy_rule"},
        ),
        MemoryWrite(
            scope=INTAKE_SCOPE,
            content="Missing data should explicitly ask for customer impact count, precise change identifier, and rollback status.",
            relevance=0.9,
            metadata={"kind": "intake_gap_rule"},
        ),
    ]
    written: list[dict[str, Any]] = []
    for entry in seed_entries:
        record = await memory.write(entry)
        written.append({"kind": record.metadata.get("kind"), "content": record.content})
    return written


def _citation_audit(available: list[str], structured_data: dict[str, Any] | None) -> dict[str, Any]:
    used = sorted({str(item) for item in (structured_data or {}).get("citations_used", [])})
    available_set = sorted({str(item) for item in available})
    missing = [item for item in used if item not in available_set]
    unused = [item for item in available_set if item not in used]
    return {
        "available_citations": available_set,
        "citations_used": used,
        "all_citations_resolved": not missing,
        "missing_citations": missing,
        "unused_citations": unused,
    }


def _normalize_structured_packet(structured_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(structured_data, dict):
        return {}

    normalized = dict(structured_data)
    evidence_items: list[dict[str, Any]] = []
    for item in list(normalized.get("evidence_items") or []):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "").strip()
        finding = str(item.get("finding") or "").strip()
        lowered_finding = finding.lower()
        if finding and any(marker in lowered_finding for marker in IMAGE_METRIC_MARKERS):
            source_id = "IMG-1"
        modality = CITATION_MODALITY_MAP.get(source_id, str(item.get("modality") or "text").strip() or "text")
        evidence_items.append(
            {
                "source_id": source_id,
                "modality": modality,
                "finding": finding,
            }
        )
    normalized["evidence_items"] = evidence_items

    valid_citations = set(CITATION_MODALITY_MAP)
    citations_used = {
        str(item).strip()
        for item in list(normalized.get("citations_used") or [])
        if str(item).strip() in valid_citations
    }
    citations_used.update(item["source_id"] for item in evidence_items if item.get("source_id") in valid_citations)
    normalized["citations_used"] = sorted(citations_used)
    return normalized


async def _run_content_stream(
    engine: ExecutionEngine,
    envelope: ContentRequestEnvelope,
    context: RequestContext,
) -> tuple[Any, dict[str, Any]]:
    event_counts: Counter[str] = Counter()
    token_preview_parts: list[str] = []
    meta_events: list[dict[str, Any]] = []
    usage_events: list[dict[str, Any]] = []
    content_buffer_parts: list[str] = []
    error_event: dict[str, Any] | None = None
    model_name: str | None = None
    final_envelope: Any = None

    async for event in engine.stream_content(envelope, context=context):
        event_counts[event.type.value] += 1

        if event.type == StreamEventType.TOKEN:
            if sum(len(part) for part in token_preview_parts) < 320:
                token_preview_parts.append(str(event.data))
            content_buffer_parts.append(str(event.data))
            continue

        if event.type == StreamEventType.META:
            payload = event.data if isinstance(event.data, dict) else {"value": str(event.data)}
            meta_events.append(payload)
            model_name = str(payload.get("model") or model_name or "") or model_name
            continue

        if event.type == StreamEventType.USAGE and hasattr(event.data, "to_dict"):
            usage_events.append(event.data.to_dict())
            continue

        if event.type == StreamEventType.ERROR:
            error_event = event.data if isinstance(event.data, dict) else {"status": 500, "error": str(event.data)}
            break

        if event.type == StreamEventType.DONE:
            final_envelope = event.data

    if final_envelope is None and error_event is not None:
        final_envelope = CompletionResult(
            content="".join(content_buffer_parts).strip() or None,
            usage=None,
            model=model_name or envelope.model,
            status=int(error_event.get("status", 500) or 500),
            error=str(error_event.get("error") or "Content stream failed"),
        )

    if final_envelope is None and content_buffer_parts:
        final_envelope = CompletionResult(
            content="".join(content_buffer_parts).strip() or None,
            usage=None,
            model=model_name or envelope.model,
            status=200,
            error=None,
        )

    if final_envelope is None:
        fallback_response = await engine.complete_content(replace(envelope, stream=False), context=context)
        final_envelope = fallback_response
        meta_events.append(
            {
                "fallback_mode": "non_stream_complete_content",
                "reason": "stream ended without terminal event",
            }
        )
        event_counts["fallback_complete"] += 1

    content_envelope = ensure_content_response_envelope(final_envelope)
    result = ensure_completion_result(content_envelope)
    return (
        {
            "content_envelope": content_envelope,
            "completion_result": result,
        },
        {
            "event_type_counts": dict(event_counts),
            "token_preview": "".join(token_preview_parts).strip(),
            "meta_events": meta_events,
            "usage_events": usage_events,
        },
    )


def _assembled_summary(structured_data: dict[str, Any] | None) -> str:
    if not structured_data:
        return ""
    evidence = "\n".join(
        f"- {item['source_id']} ({item['modality']}): {item['finding']}"
        for item in structured_data.get("evidence_items", [])
    )
    actions = "\n".join(f"- {item}" for item in structured_data.get("immediate_actions", []))
    missing = "\n".join(f"- {item}" for item in structured_data.get("missing_data", []))
    return (
        f"Situation\n- {structured_data.get('situation_summary', '')}\n\n"
        f"Risk\n- severity={structured_data.get('severity', '')}; owner={structured_data.get('likely_owner', '')}; "
        f"initial_risk={structured_data.get('initial_risk', '')}\n\n"
        f"Evidence\n{evidence}\n\n"
        f"Immediate Actions\n{actions}\n\n"
        f"Missing Data\n{missing}\n\n"
        f"Customer-Safe Summary\n- {structured_data.get('customer_safe_summary', '')}"
    ).strip()


async def main() -> None:
    handle = build_live_provider()
    try:
        redaction_policy = RedactionPolicy(
            sensitive_keys=("customer_email", "session_token", "callback_url", "auth_token"),
            preview_mode=PayloadPreviewMode.SUMMARY,
        )
        raw_metadata = {
            "source_id": "META-1",
            "incident_id": INCIDENT_PACKET["incident_id"],
            "severity_hint": INCIDENT_PACKET["severity_hint"],
            "owner_hint": INCIDENT_PACKET["suspected_owner"],
            "change_window": "routing config changed 12 minutes ago",
            "customer_email": "ops-lead@acme.example",
            "session_token": "tok_demo_sensitive_checkout",
            "callback_url": "https://api.acme.example/callbacks/checkout/secret",
        }
        safe_metadata = sanitize_payload(raw_metadata, policy=redaction_policy)
        metadata_preview = preview_payload(raw_metadata, policy=redaction_policy)
        redaction_audit = _redaction_audit(raw_metadata, safe_metadata)

        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)
        memory_notes = await memory.retrieve(MemoryQuery(scope=INTAKE_SCOPE, limit=3))

        source_blocks = [
            TextBlock(
                "SYS-1 Create an incident-intake brief with sections: Situation, Evidence, Initial Risk, Immediate Action, Missing Data, Customer-Safe Summary. Cite source IDs exactly."
            ),
            TextBlock(
                "TXT-1 Service: checkout-api. Business impact: payment completion is degraded for a subset of users after a routing change."
            ),
            TextBlock(
                "TXT-2 Handoff excerpt: support is receiving reports of delayed payment confirmations and retry storms from enterprise merchants."
            ),
            TextBlock(
                "FILE-1 Handoff bundle attached: incident-handoff.pdf. Treat this as supporting evidence only; do not invent contents beyond the supplied notes."
            ),
            AudioBlock(
                transcript=(
                    "AUD-1 On-call audio: error rate spiked within ten minutes of the routing change, queue lag is climbing, "
                    "and support is hearing about delayed webhook confirmations. Rollback status is not yet confirmed."
                )
            ),
            TextBlock(
                "IMG-1 Dashboard screenshot attached below. "
                "Screenshot-derived observations: error rate 18.2%, webhook confirmation lag 5m 12s, "
                "queue depth 14,220, and a top-note that the routing change was deployed 12 minutes before the spike. "
                "Cite IMG-1 for these dashboard metrics or timing details only."
            ),
            ImageBlock(
                image_url=_dashboard_image_data_url(),
                detail="low",
                mime_type="image/png",
            ),
            FileBlock(
                name="incident-handoff.pdf",
                mime_type="application/pdf",
            ),
            TextBlock(
                "MEM-1 Triage preferences: "
                + " | ".join(record.content for record in memory_notes)
            ),
            MetadataBlock(safe_metadata),
        ]

        active_projection = _projection_summary(handle.name, source_blocks)

        lifecycle = LifecycleRecorder(redaction_policy=redaction_policy)
        diagnostics = EngineDiagnosticsRecorder(redaction_policy=redaction_policy)
        engine = ExecutionEngine(provider=handle.provider, hooks=HookManager([lifecycle, diagnostics]))

        request_envelope = ContentRequestEnvelope(
            provider=handle.name,
            model=handle.model,
            stream=True,
            messages=(
                ContentMessage(
                    role=Role.SYSTEM,
                    blocks=(
                        TextBlock(
                            "You are a multimodal intake analyst. Use only the supplied content bundle. "
                            "Preserve evidence boundaries, cite source IDs exactly, respect metadata redaction, and call out missing data explicitly."
                        ),
                    ),
                ),
                ContentMessage(
                    role=Role.USER,
                    blocks=tuple(source_blocks),
                ),
            ),
        )
        request_context = RequestContext(
            session_id="cookbook-multimodal-intake",
            job_id=INCIDENT_PACKET["incident_id"],
            tags={"service": INCIDENT_PACKET["service"], "incident_id": INCIDENT_PACKET["incident_id"]},
        )

        stream_result, stream_summary = await _run_content_stream(engine, request_envelope, request_context)
        response_envelope = stream_result["content_envelope"]
        result = stream_result["completion_result"]

        structured = await extract_structured(
            handle.provider,
            [
                Message.system(
                    (
                        "Convert the intake brief into a structured intake packet. "
                        "Keep evidence grounded in the cited bundle. Use only the safe metadata and cited sources. "
                        "Use source modalities exactly as follows: TXT-1/TXT-2=text, AUD-1=audio, IMG-1=image, "
                        "FILE-1=file, META-1=metadata, MEM-1=memory. "
                        "If a finding references dashboard metrics or screenshot timing details, cite IMG-1, not AUD-1. "
                        "Only include citations that are actually used."
                    )
                ),
                Message.user(
                    json.dumps(
                        {
                            "incident_packet": INCIDENT_PACKET,
                            "redaction_audit": redaction_audit,
                            "brief": result.content,
                            "source_index": {
                                "TXT-1": "Service and impact note.",
                                "TXT-2": "Support handoff excerpt.",
                                "AUD-1": "On-call audio transcript about spike timing, queue lag climb, and rollback uncertainty.",
                                "IMG-1": "Dashboard screenshot with error rate, webhook lag, queue depth, and routing-change timing note.",
                                "FILE-1": "Attached incident handoff PDF reference only.",
                                "META-1": "Safe redacted incident metadata.",
                                "MEM-1": "Triage preferences and privacy rules.",
                            },
                            "available_citations": ["TXT-1", "TXT-2", "FILE-1", "AUD-1", "IMG-1", "META-1", "MEM-1"],
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "incident_id": {"type": "string"},
                        "severity": {"type": "string"},
                        "likely_owner": {"type": "string"},
                        "situation_summary": {"type": "string"},
                        "initial_risk": {"type": "string"},
                        "evidence_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "modality": {"type": "string"},
                                    "finding": {"type": "string"},
                                },
                                "required": ["source_id", "modality", "finding"],
                                "additionalProperties": False,
                            },
                        },
                        "immediate_actions": {"type": "array", "items": {"type": "string"}},
                        "missing_data": {"type": "array", "items": {"type": "string"}},
                        "customer_safe_summary": {"type": "string"},
                        "citations_used": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "incident_id",
                        "severity",
                        "likely_owner",
                        "situation_summary",
                        "initial_risk",
                        "evidence_items",
                        "immediate_actions",
                        "missing_data",
                        "customer_safe_summary",
                        "citations_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )

        normalized_structured_data = _normalize_structured_packet(structured.data)
        citation_audit = _citation_audit(
            ["TXT-1", "TXT-2", "FILE-1", "AUD-1", "IMG-1", "META-1", "MEM-1"],
            normalized_structured_data,
        )
        assembled_summary = _assembled_summary(normalized_structured_data)
        await memory.write(
            MemoryWrite(
                scope=INTAKE_SCOPE,
                content=json.dumps(normalized_structured_data, ensure_ascii=True, sort_keys=True),
                relevance=0.95,
                metadata={"kind": "intake_packet"},
            )
        )
        memory_after = await memory.retrieve(MemoryQuery(scope=INTAKE_SCOPE, limit=5))

        latest_request_report = list(lifecycle.requests.values())[-1] if lifecycle.requests else None
        latest_session_report = lifecycle.sessions.get(request_context.session_id or "")

        print_heading("Multimodal Intake Pipeline")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "incident_packet": INCIDENT_PACKET,
                "raw_metadata": raw_metadata,
                "safe_metadata": safe_metadata,
                "metadata_preview": metadata_preview,
                "redaction_audit": redaction_audit,
                "source_bundle": [block.to_dict() for block in source_blocks],
                "active_projection": {
                    "provider": handle.name,
                    **active_projection,
                },
                "provider_projection_matrix": {
                    provider_name: _projection_summary(provider_name, source_blocks)
                    for provider_name in ("openai", "anthropic", "google")
                },
                "memory_bootstrap": memory_bootstrap,
                "stream_summary": stream_summary,
                "content_result": {
                    "status": result.status,
                    "usage": summarize_usage(result.usage),
                    "response_blocks": [block.to_dict() for block in response_envelope.message.blocks],
                    "content": result.content,
                },
                "structured_packet": {
                    "valid": structured.valid,
                    "repair_attempts": structured.repair_attempts,
                    "usage": summarize_usage(getattr(structured, "usage", None)),
                    "data": normalized_structured_data,
                },
                "citation_audit": citation_audit,
                "assembled_summary": assembled_summary,
                "observability": {
                    "hook_event_counts": dict(Counter(event for event, _, _ in diagnostics.events)),
                    "lifecycle_event_counts": dict(Counter(event.type.value for event in lifecycle.events)),
                    "latest_request_report": latest_request_report.to_dict() if latest_request_report else None,
                    "latest_session_report": latest_session_report.to_dict() if latest_session_report else None,
                },
                "memory_after_action": [
                    {"kind": record.metadata.get("kind"), "content": record.content}
                    for record in memory_after
                ],
                "showcase_verdict": {
                    "multimodal_bundle_used": len(source_blocks) >= 6,
                    "redaction_applied": redaction_audit["safe_for_model"],
                    "streamed_content_run": bool(stream_summary["event_type_counts"]),
                    "citation_grounded": citation_audit["all_citations_resolved"] and "IMG-1" in citation_audit["citations_used"],
                    "operator_ready": structured.valid and bool(normalized_structured_data.get("immediate_actions")),
                },
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
