from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from tempfile import gettempdir
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.engine import ExecutionEngine
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import CompletionResult, Message, StreamEventType
from llm_client.redaction import (
    PayloadPreviewMode,
    ProviderPayloadCaptureMode,
    RedactionPolicy,
    ToolOutputPolicy,
    capture_provider_payload,
    preview_payload,
    sanitize_payload,
    sanitize_tool_output,
)
from llm_client.spec import RequestContext, RequestSpec
from llm_client.structured import StructuredOutputConfig, extract_structured


COMPLIANCE_SCOPE = "compliance-redaction-pipeline"
CASE_PACKET = {
    "case_id": "CMP-5512",
    "workflow": "compliance escalation for delayed settlement exports after logging rollout",
    "objective": "produce a compliance-safe internal summary and audit artifact without leaking secrets or sensitive customer details",
    "audience": ["trust-and-safety", "compliance-ops", "support-lead"],
}


def _field_delta_audit(raw_packet: dict[str, Any], safe_packet: dict[str, Any]) -> dict[str, Any]:
    transformed_fields: list[str] = []
    redacted_fields: list[str] = []
    for key, raw_value in raw_packet.items():
        safe_value = safe_packet.get(key)
        if safe_value != raw_value:
            transformed_fields.append(key)
            if safe_value in {"[REDACTED]", "<redacted>", None}:
                redacted_fields.append(key)
    return {
        "fields_seen": sorted(raw_packet.keys()),
        "transformed_fields": sorted(transformed_fields),
        "redacted_fields": sorted(redacted_fields),
        "safe_for_model": safe_packet != raw_packet,
    }


def _contains_sensitive_values(text: str | None, sensitive_values: list[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(value.lower() in lowered for value in sensitive_values if value)


def _artifact_safety_checks(
    *,
    summary_text: str | None,
    structured_data: dict[str, Any] | None,
    artifact_payload: dict[str, Any],
    sensitive_values: list[str],
) -> dict[str, Any]:
    structured_text = json.dumps(structured_data or {}, ensure_ascii=True, sort_keys=True)
    artifact_text = json.dumps(artifact_payload, ensure_ascii=True, sort_keys=True)
    return {
        "summary_safe": not _contains_sensitive_values(summary_text, sensitive_values),
        "structured_packet_safe": not _contains_sensitive_values(structured_text, sensitive_values),
        "artifact_safe": not _contains_sensitive_values(artifact_text, sensitive_values),
    }


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    entries = [
        MemoryWrite(
            scope=COMPLIANCE_SCOPE,
            content="Customer email, tokens, callback URLs, and internal commercial notes must not reach model prompts or audit artifacts.",
            relevance=0.98,
            metadata={"kind": "privacy_rule"},
        ),
        MemoryWrite(
            scope=COMPLIANCE_SCOPE,
            content="Provider payload capture must stay metadata-only for compliance audit logging unless explicit approval exists.",
            relevance=0.95,
            metadata={"kind": "provider_capture_rule"},
        ),
        MemoryWrite(
            scope=COMPLIANCE_SCOPE,
            content="Tool outputs written to compliance artifacts must be sanitized before persistence.",
            relevance=0.94,
            metadata={"kind": "tool_rule"},
        ),
        MemoryWrite(
            scope=COMPLIANCE_SCOPE,
            content="Compliance-safe summaries may describe impact and next actions but must avoid secrets and raw personal contact details.",
            relevance=0.93,
            metadata={"kind": "summary_rule"},
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
            if sum(len(part) for part in token_preview_parts) < 320:
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


def _deterministic_compliance_summary(
    *,
    redaction_audit: dict[str, Any],
    tool_output_audit: dict[str, Any],
    artifact_checks: dict[str, Any],
    provider_capture_mode: str,
) -> dict[str, Any]:
    blocking_findings: list[str] = []
    required_actions: list[str] = []

    if not redaction_audit["safe_for_model"]:
        blocking_findings.append("Model packet was not sanitized before dispatch.")
    if not artifact_checks["summary_safe"]:
        blocking_findings.append("Compliance-safe summary leaked a sensitive value.")
    if not artifact_checks["structured_packet_safe"]:
        blocking_findings.append("Structured packet leaked a sensitive value.")
    if not artifact_checks["artifact_safe"]:
        blocking_findings.append("Persisted audit artifact leaked a sensitive value.")
    if not tool_output_audit["safe_for_artifact"]:
        blocking_findings.append("Tool output was not sanitized before artifact persistence.")
    if provider_capture_mode != ProviderPayloadCaptureMode.METADATA_ONLY.value:
        blocking_findings.append("Provider payload capture mode exceeded metadata-only policy.")

    required_actions.append("Keep metadata-only provider payload capture for compliance logging.")
    required_actions.append("Persist only sanitized packets, previews, and tool outputs in audit artifacts.")
    required_actions.append("Treat customer contact details, tokens, and internal commercial notes as forbidden in summaries.")

    return {
        "overall_status": "compliant_redaction_pipeline_ready" if not blocking_findings else "compliance_blocked_by_redaction_gap",
        "redacted_fields": list(redaction_audit["redacted_fields"]),
        "tool_redacted_fields": list(tool_output_audit["redacted_fields"]),
        "provider_capture_mode": provider_capture_mode,
        "blocking_findings": blocking_findings,
        "required_actions": required_actions,
        "evidence_used": [
            "safe_packet",
            "packet_preview",
            "sanitized_tool_output",
            "provider_capture",
            "artifact_checks",
        ],
    }


def _normalize_structured_packet(
    structured_data: dict[str, Any] | None,
    *,
    deterministic_summary: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(structured_data or {})
    for key in ("redacted_fields", "tool_redacted_fields", "blocking_findings", "required_actions", "evidence_used"):
        normalized[key] = [str(item).strip() for item in list(normalized.get(key) or []) if str(item).strip()]
    for key in ("overall_status", "provider_capture_mode"):
        normalized[key] = deterministic_summary[key]
    normalized["redacted_fields"] = list(deterministic_summary["redacted_fields"])
    normalized["tool_redacted_fields"] = list(deterministic_summary["tool_redacted_fields"])
    normalized["blocking_findings"] = list(deterministic_summary["blocking_findings"])
    if not normalized["required_actions"]:
        normalized["required_actions"] = list(deterministic_summary["required_actions"])
    else:
        normalized["required_actions"] = list(deterministic_summary["required_actions"])
    normalized["evidence_used"] = list(deterministic_summary["evidence_used"])
    return normalized


def _assembled_summary(structured_data: dict[str, Any] | None) -> str:
    if not structured_data:
        return ""
    redacted = "\n".join(f"- {item}" for item in structured_data.get("redacted_fields", []))
    tool_redacted = "\n".join(f"- {item}" for item in structured_data.get("tool_redacted_fields", []))
    findings = "\n".join(f"- {item}" for item in structured_data.get("blocking_findings", []))
    actions = "\n".join(f"- {item}" for item in structured_data.get("required_actions", []))
    evidence = "\n".join(f"- {item}" for item in structured_data.get("evidence_used", []))
    return (
        f"Overall Status\n- {structured_data.get('overall_status')}\n\n"
        f"Redacted Fields\n{redacted}\n\n"
        f"Tool Redactions\n{tool_redacted}\n\n"
        f"Blocking Findings\n{findings if findings else '- none'}\n\n"
        f"Required Actions\n{actions}\n\n"
        f"Evidence Used\n{evidence}"
    ).strip()


async def main() -> None:
    handle = build_live_provider()
    try:
        raw_packet = {
            "case_id": "CMP-5512",
            "customer_email": "security-lead@acme.example",
            "session_token": "tok_demo_compliance_secret",
            "customer_tier": "enterprise",
            "issue_summary": "Audit exports include delayed settlement records after a logging rollout.",
            "business_impact": "Finance and compliance cannot complete month-end checks.",
            "internal_notes": "Potential renewal risk; keep internal priority high.",
            "callback_url": "https://api.acme.example/compliance/callback/private",
            "symptoms": [
                "exports exceed the 5-minute SLA",
                "queue backlog grows after large reconciliation runs",
                "support suspects audit fanout overhead",
            ],
        }

        redaction_policy = RedactionPolicy(
            sensitive_keys=("customer_email", "session_token", "internal_notes", "callback_url", "secret"),
            preview_mode=PayloadPreviewMode.SUMMARY,
            provider_payload_capture=ProviderPayloadCaptureMode.METADATA_ONLY,
            preview_max_chars=96,
        )

        safe_packet = sanitize_payload(raw_packet, policy=redaction_policy)
        packet_preview = preview_payload(raw_packet, policy=redaction_policy)
        redaction_audit = _field_delta_audit(raw_packet, safe_packet)

        simulated_tool_output = {
            "owner_email": "security-lead@acme.example",
            "meeting_token": "tok_followup_secret",
            "status": "priority follow-up requested before renewal review",
            "callback_url": "https://api.acme.example/compliance/callback/private",
        }
        sanitized_tool_output = sanitize_tool_output(
            simulated_tool_output,
            policy=ToolOutputPolicy(
                patterns=(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    r"\btok_[A-Za-z0-9_]+\b",
                    r"https?://[^\s]+",
                )
            ),
        )
        tool_output_audit = _field_delta_audit(simulated_tool_output, sanitized_tool_output)
        tool_output_audit["safe_for_artifact"] = not _contains_sensitive_values(
            json.dumps(sanitized_tool_output, ensure_ascii=True, sort_keys=True),
            [
                simulated_tool_output["owner_email"],
                simulated_tool_output["meeting_token"],
                simulated_tool_output["callback_url"],
            ],
        )

        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)
        memory_notes = await memory.retrieve(MemoryQuery(scope=COMPLIANCE_SCOPE, limit=4))

        lifecycle = LifecycleRecorder(redaction_policy=redaction_policy)
        diagnostics = EngineDiagnosticsRecorder(redaction_policy=redaction_policy)
        engine = ExecutionEngine(provider=handle.provider, hooks=HookManager([diagnostics, lifecycle]))

        session_id = "cookbook-compliance-redaction"
        summary_spec = RequestSpec(
            provider=handle.name,
            model=handle.model,
            messages=[
                Message.system(
                    "Write a compliance-safe internal summary with sections: Impact, Confirmed Facts, Next Action. "
                    "Use only sanitized facts and do not reproduce secrets, emails, callback URLs, or internal commercial notes."
                ),
                Message.user(
                    json.dumps(
                        {
                            "case_packet": CASE_PACKET,
                            "safe_packet": safe_packet,
                            "memory_notes": [record.content for record in memory_notes],
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                ),
            ],
        )
        summary_result, stream_summary = await _stream_completion(
            engine,
            summary_spec,
            context=RequestContext(session_id=session_id, job_id="compliance-safe-summary"),
        )

        provider_capture = capture_provider_payload(
            {
                "id": "resp_demo_5512",
                "model": summary_result.model,
                "status": summary_result.status,
                "raw_response": {"omitted": True, "secret": "should-not-leak"},
                "usage": summarize_usage(summary_result.usage),
            },
            policy=redaction_policy,
        )

        artifact_payload = {
            "case_packet": CASE_PACKET,
            "safe_packet": safe_packet,
            "packet_preview": packet_preview,
            "redaction_audit": redaction_audit,
            "sanitized_tool_output": sanitized_tool_output,
            "tool_output_audit": tool_output_audit,
            "provider_capture": provider_capture,
            "request_report": lifecycle.requests.get(next(reversed(lifecycle.requests))).to_dict() if lifecycle.requests else None,
            "summary_excerpt": (summary_result.content or "")[:260],
            "usage": summarize_usage(summary_result.usage),
        }

        sensitive_values = [
            raw_packet["customer_email"],
            raw_packet["session_token"],
            raw_packet["internal_notes"],
            raw_packet["callback_url"],
            simulated_tool_output["owner_email"],
            simulated_tool_output["meeting_token"],
        ]

        artifact_path = Path(gettempdir()) / "cookbook-compliance-redaction-audit.json"
        artifact_path.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True), encoding="utf-8")

        artifact_checks = _artifact_safety_checks(
            summary_text=summary_result.content,
            structured_data=None,
            artifact_payload=artifact_payload,
            sensitive_values=sensitive_values,
        )
        deterministic_summary = _deterministic_compliance_summary(
            redaction_audit=redaction_audit,
            tool_output_audit=tool_output_audit,
            artifact_checks=artifact_checks,
            provider_capture_mode=redaction_policy.provider_payload_capture.value,
        )

        structured = await extract_structured(
            handle.provider,
            [
                Message.system(
                    "Convert the compliance redaction run into a structured compliance packet. "
                    "Use only the sanitized packet, audits, provider capture summary, and deterministic compliance summary. "
                    "Do not claim compliance if any sensitive value leaked."
                ),
                Message.user(
                    json.dumps(
                        {
                            "case_packet": CASE_PACKET,
                            "safe_packet": safe_packet,
                            "redaction_audit": redaction_audit,
                            "tool_output_audit": tool_output_audit,
                            "provider_capture": provider_capture,
                            "artifact_checks": artifact_checks,
                            "deterministic_summary": deterministic_summary,
                            "summary_text": summary_result.content,
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
                        "overall_status": {"type": "string"},
                        "redacted_fields": {"type": "array", "items": {"type": "string"}},
                        "tool_redacted_fields": {"type": "array", "items": {"type": "string"}},
                        "provider_capture_mode": {"type": "string"},
                        "blocking_findings": {"type": "array", "items": {"type": "string"}},
                        "required_actions": {"type": "array", "items": {"type": "string"}},
                        "evidence_used": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "overall_status",
                        "redacted_fields",
                        "tool_redacted_fields",
                        "provider_capture_mode",
                        "blocking_findings",
                        "required_actions",
                        "evidence_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
            engine=engine,
            context=RequestContext(session_id=session_id, job_id="structured-compliance-packet"),
            model=handle.model,
        )

        artifact_checks = _artifact_safety_checks(
            summary_text=summary_result.content,
            structured_data=structured.data,
            artifact_payload=artifact_payload,
            sensitive_values=sensitive_values,
        )
        deterministic_summary = _deterministic_compliance_summary(
            redaction_audit=redaction_audit,
            tool_output_audit=tool_output_audit,
            artifact_checks=artifact_checks,
            provider_capture_mode=redaction_policy.provider_payload_capture.value,
        )
        normalized_structured_data = _normalize_structured_packet(
            structured.data,
            deterministic_summary=deterministic_summary,
        )
        assembled_summary = _assembled_summary(normalized_structured_data)

        await memory.write(
            MemoryWrite(
                scope=COMPLIANCE_SCOPE,
                content=json.dumps(normalized_structured_data, ensure_ascii=True, sort_keys=True),
                relevance=0.96,
                metadata={"kind": "compliance_packet"},
            )
        )
        memory_after = await memory.retrieve(MemoryQuery(scope=COMPLIANCE_SCOPE, limit=6))

        latest_request_report = list(lifecycle.requests.values())[-1] if lifecycle.requests else None
        latest_session_report = lifecycle.sessions.get(session_id)

        print_heading("Compliance Redaction Pipeline")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "case_packet": CASE_PACKET,
                "raw_packet_fields": sorted(raw_packet.keys()),
                "memory_bootstrap": memory_bootstrap,
                "safe_packet": safe_packet,
                "packet_preview": packet_preview,
                "redaction_audit": redaction_audit,
                "sanitized_tool_output": sanitized_tool_output,
                "tool_output_audit": tool_output_audit,
                "provider_capture": provider_capture,
                "artifact_checks": artifact_checks,
                "stream_summary": stream_summary,
                "summary_result": {
                    "status": summary_result.status,
                    "usage": summarize_usage(summary_result.usage),
                    "content": summary_result.content,
                },
                "structured_packet": {
                    "valid": structured.valid,
                    "repair_attempts": structured.repair_attempts,
                    "usage": summarize_usage(getattr(structured, "usage", None)),
                    "data": normalized_structured_data,
                },
                "assembled_summary": assembled_summary,
                "audit_artifact_path": str(artifact_path),
                "audit_artifact_preview": artifact_payload,
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
                    "sanitized_before_model": redaction_audit["safe_for_model"],
                    "tool_output_sanitized": tool_output_audit["safe_for_artifact"],
                    "artifact_safe": artifact_checks["artifact_safe"],
                    "structured_packet_ready": structured.valid and not normalized_structured_data["blocking_findings"],
                    "operator_ready": bool(assembled_summary),
                },
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
