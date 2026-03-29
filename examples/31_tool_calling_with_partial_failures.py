from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.agent.definition import ToolExecutionMode
from llm_client.engine import ExecutionEngine
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import CompletionResult, Message, StreamEventType, ToolCall
from llm_client.spec import RequestContext, RequestSpec
from llm_client.structured import StructuredOutputConfig, extract_structured
from llm_client.tools.base import Tool, ToolExecutionMetadata, ToolRegistry, ToolResult
from llm_client.tools.execution_engine import ToolExecutionBatch, ToolExecutionEngine


PARTIAL_SCOPE = "tool-calling-partial-failures"
INCIDENT_PACKET = {
    "incident_id": "INC-5024",
    "service": "checkout-api",
    "severity_hint": "sev-1",
    "objective": "prepare an incident commander update even when some tooling is degraded or unavailable",
    "audience": ["incident-commander", "payments-ops", "support-lead"],
}


async def _metrics_snapshot(service: str) -> dict[str, object]:
    await asyncio.sleep(0.05)
    return {
        "service": service,
        "error_rate": "18.4%",
        "queue_lag": "4m12s",
        "checkout_status": "degrading",
        "note": "checkout 5xx rate and queue lag remain elevated",
    }


async def _deployment_rollout(service: str) -> ToolResult:
    await asyncio.sleep(0.08)
    return ToolResult(
        content={
            "service": service,
            "latest_change": "payment routing config changed 12 minutes ago",
            "confidence": "partial",
            "note": "rollout metadata is delayed on one control-plane replica",
        },
        metadata={
            "partial": True,
            "warning": "rollout metadata is delayed by one control-plane replica",
            "known_gap": "timestamp may lag by one replica heartbeat",
        },
    )


async def _dependency_probe(service: str) -> ToolResult:
    _ = service
    await asyncio.sleep(0.35)
    return ToolResult.success_result(
        {
            "dependency_status": "healthy",
            "note": "this payload should never be returned because the timeout is intentionally lower",
        }
    )


async def _support_pressure(service: str) -> dict[str, object]:
    await asyncio.sleep(0.04)
    return {
        "service": service,
        "open_cases": 19,
        "merchant_reports": 7,
        "top_issue": "delayed payment confirmations",
    }


def _build_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(
                name="metrics_snapshot",
                description="Get current error and queue metrics for a service.",
                parameters={
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                    "additionalProperties": False,
                },
                handler=_metrics_snapshot,
                execution=ToolExecutionMetadata(concurrency_limit=2, trust_level="high", timeout_seconds=5.0),
            ),
            Tool(
                name="deployment_rollout",
                description="Inspect the latest deployment or routing change.",
                parameters={
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                    "additionalProperties": False,
                },
                handler=_deployment_rollout,
                execution=ToolExecutionMetadata(concurrency_limit=2, trust_level="medium", timeout_seconds=5.0),
            ),
            Tool(
                name="dependency_probe",
                description="Check downstream dependency health.",
                parameters={
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                    "additionalProperties": False,
                },
                handler=_dependency_probe,
                execution=ToolExecutionMetadata(
                    concurrency_limit=1,
                    trust_level="low",
                    timeout_seconds=0.15,
                    retry_attempts=1,
                ),
            ),
            Tool(
                name="support_pressure",
                description="Summarize support case pressure tied to the incident.",
                parameters={
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                    "additionalProperties": False,
                },
                handler=_support_pressure,
                execution=ToolExecutionMetadata(concurrency_limit=2, trust_level="high", timeout_seconds=5.0),
            ),
        ]
    )


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    entries = [
        MemoryWrite(
            scope=PARTIAL_SCOPE,
            content="Incident commanders want confirmed signal first, then degraded evidence, then unavailable evidence.",
            relevance=0.96,
            metadata={"kind": "commander_rule"},
        ),
        MemoryWrite(
            scope=PARTIAL_SCOPE,
            content="Partial rollout metadata must never be treated as confirmed causality.",
            relevance=0.93,
            metadata={"kind": "causality_rule"},
        ),
        MemoryWrite(
            scope=PARTIAL_SCOPE,
            content="If dependency health is unavailable, recommend fallback verification instead of root-cause claims.",
            relevance=0.94,
            metadata={"kind": "fallback_rule"},
        ),
        MemoryWrite(
            scope=PARTIAL_SCOPE,
            content="Support pressure should be tied to delayed confirmations when merchant reports are rising.",
            relevance=0.9,
            metadata={"kind": "support_rule"},
        ),
    ]
    written: list[dict[str, Any]] = []
    for entry in entries:
        record = await memory.write(entry)
        written.append({"kind": record.metadata.get("kind"), "content": record.content})
    return written


def _serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
        "parsed_arguments": tool_call.parse_arguments(),
    }


def _serialize_batch(batch: ToolExecutionBatch) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for envelope in batch.results:
        results.append(
            {
                "tool_name": envelope.tool_name,
                "tool_call_id": envelope.tool_call_id,
                "status": envelope.status,
                "success": envelope.success,
                "attempts": envelope.attempts,
                "duration_ms": round(envelope.duration_ms or 0.0, 2),
                "result": {
                    "content": envelope.result.content,
                    "error": envelope.result.error,
                    "metadata": envelope.result.metadata,
                },
                "execution": {
                    "timeout_seconds": envelope.timeout_seconds,
                    "retry_attempts": envelope.retry_attempts,
                    "concurrency_limit": envelope.concurrency_limit,
                    "trust_level": envelope.trust_level,
                    "safety_tags": list(envelope.safety_tags),
                    "metadata": envelope.metadata,
                },
            }
        )
    return {
        "mode": batch.mode.value,
        "has_partial": batch.has_partial,
        "has_errors": batch.has_errors,
        "status_counts": dict(Counter(result["status"] for result in results)),
        "results": results,
    }


def _deterministic_batch_summary(batch: ToolExecutionBatch) -> dict[str, Any]:
    confirmed_signal: list[str] = []
    degraded_evidence: list[str] = []
    unavailable_evidence: list[str] = []
    immediate_actions: list[str] = []
    evidence_used: list[str] = []

    for envelope in batch.results:
        evidence_used.append(envelope.tool_name)
        if envelope.status == "success":
            if envelope.tool_name == "metrics_snapshot" and isinstance(envelope.result.content, dict):
                confirmed_signal.append(
                    "metrics_snapshot confirms checkout degradation with error_rate=18.4% and queue_lag=4m12s."
                )
            elif envelope.tool_name == "support_pressure" and isinstance(envelope.result.content, dict):
                confirmed_signal.append(
                    "support_pressure confirms rising support load with 19 open cases and 7 merchant reports."
                )
        elif envelope.status == "partial":
            degraded_evidence.append(
                "deployment_rollout is partial: payment routing config changed 12 minutes ago, but rollout metadata is delayed on one control-plane replica."
            )
        elif envelope.status == "error":
            unavailable_evidence.append(
                f"{envelope.tool_name} unavailable: {envelope.result.error or 'tool execution failed'}"
            )

    if confirmed_signal:
        immediate_actions.append("Treat metrics_snapshot and support_pressure as the confirmed signal base for the incident commander.")
    if degraded_evidence:
        immediate_actions.append("Use deployment_rollout as degraded evidence only; do not treat it as confirmed causality.")
    if unavailable_evidence:
        immediate_actions.append("Run fallback dependency verification out of band because dependency_probe is unavailable.")
    immediate_actions.append("Prepare rollback evaluation and merchant-facing coordination while dependency health remains unknown.")

    return {
        "overall_status": "partial_failure_with_actionable_signal" if unavailable_evidence or degraded_evidence else "all_tools_healthy",
        "confirmed_signal": confirmed_signal,
        "degraded_evidence": degraded_evidence,
        "unavailable_evidence": unavailable_evidence,
        "immediate_actions": immediate_actions,
        "evidence_used": evidence_used,
    }


async def _stream_completion(
    engine: ExecutionEngine,
    handle: Any,
    messages: list[Message],
    *,
    session_id: str,
    job_id: str,
) -> tuple[CompletionResult, dict[str, Any]]:
    started = time.perf_counter()
    event_counts: Counter[str] = Counter()
    token_preview_parts: list[str] = []
    meta_events: list[dict[str, Any]] = []
    usage_events: list[dict[str, Any]] = []
    content_parts: list[str] = []
    final_result: CompletionResult | None = None
    error_payload: dict[str, Any] | None = None

    spec = dict(
        provider=handle.name,
        model=handle.model,
        messages=messages,
    )
    context = RequestContext(session_id=session_id, job_id=job_id)

    async for event in engine.stream(RequestSpec(**spec), context=context):
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
                    model=handle.model,
                    usage=None,
                )

    if final_result is None:
        final_result = CompletionResult(
            content="".join(content_parts).strip() or None,
            status=int((error_payload or {}).get("status", 500) or 500),
            error=str((error_payload or {}).get("error") or "Stream ended without a terminal completion result."),
            model=handle.model,
            usage=None,
        )

    return final_result, {
        "event_type_counts": dict(event_counts),
        "token_preview": "".join(token_preview_parts).strip(),
        "meta_events": meta_events,
        "usage_events": usage_events,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _normalize_structured_packet(
    structured_data: dict[str, Any] | None,
    *,
    batch_summary: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(structured_data, dict):
        structured_data = {}

    normalized = dict(structured_data)
    for key in ("confirmed_signal", "degraded_evidence", "unavailable_evidence", "immediate_actions", "evidence_used"):
        normalized[key] = [
            str(item).strip()
            for item in list(normalized.get(key) or [])
            if str(item).strip()
        ]

    normalized["overall_status"] = batch_summary["overall_status"]
    normalized["confirmed_signal"] = list(batch_summary["confirmed_signal"])
    normalized["degraded_evidence"] = list(batch_summary["degraded_evidence"])
    normalized["unavailable_evidence"] = list(batch_summary["unavailable_evidence"])
    normalized["evidence_used"] = list(batch_summary["evidence_used"])
    if not normalized["immediate_actions"]:
        normalized["immediate_actions"] = list(batch_summary["immediate_actions"])
    return normalized


def _assembled_summary(structured_data: dict[str, Any] | None) -> str:
    if not structured_data:
        return ""
    confirmed = "\n".join(f"- {item}" for item in structured_data.get("confirmed_signal", []))
    degraded = "\n".join(f"- {item}" for item in structured_data.get("degraded_evidence", []))
    unavailable = "\n".join(f"- {item}" for item in structured_data.get("unavailable_evidence", []))
    actions = "\n".join(f"- {item}" for item in structured_data.get("immediate_actions", []))
    evidence = "\n".join(f"- {item}" for item in structured_data.get("evidence_used", []))
    return (
        f"Overall Status\n- {structured_data.get('overall_status', '')}\n\n"
        f"Confirmed Signal\n{confirmed}\n\n"
        f"Degraded Evidence\n{degraded}\n\n"
        f"Unavailable Evidence\n{unavailable}\n\n"
        f"Immediate Actions\n{actions}\n\n"
        f"Evidence Used\n{evidence}"
    ).strip()


async def main() -> None:
    handle = build_live_provider()
    try:
        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)
        memory_notes = await memory.retrieve(MemoryQuery(scope=PARTIAL_SCOPE, limit=4))

        tool_calls = [
            ToolCall(id="call_metrics", name="metrics_snapshot", arguments=json.dumps({"service": "checkout-api"})),
            ToolCall(id="call_rollout", name="deployment_rollout", arguments=json.dumps({"service": "checkout-api"})),
            ToolCall(id="call_deps", name="dependency_probe", arguments=json.dumps({"service": "checkout-api"})),
            ToolCall(id="call_support", name="support_pressure", arguments=json.dumps({"service": "checkout-api"})),
        ]

        registry = _build_registry()
        tool_engine = ToolExecutionEngine(registry)
        batch_started = time.perf_counter()
        batch = await tool_engine.execute_calls(tool_calls, mode=ToolExecutionMode.PARALLEL)
        batch_duration_ms = round((time.perf_counter() - batch_started) * 1000.0, 2)
        tool_batch = _serialize_batch(batch)
        tool_batch["batch_duration_ms"] = batch_duration_ms
        deterministic_summary = _deterministic_batch_summary(batch)

        lifecycle = LifecycleRecorder()
        diagnostics = EngineDiagnosticsRecorder()
        engine = ExecutionEngine(provider=handle.provider, hooks=HookManager([lifecycle, diagnostics]))
        session_id = "cookbook-tool-partial-failures"

        summary_result, stream_summary = await _stream_completion(
            engine,
            handle,
            [
                Message.system(
                    "Write an incident commander update with sections: Confirmed Signal, Degraded Evidence, Unavailable Evidence, Immediate Action. "
                    "Cite tool names exactly. Do not present degraded or unavailable evidence as confirmed."
                ),
                Message.user(
                    json.dumps(
                        {
                            "incident_packet": INCIDENT_PACKET,
                            "tool_batch": tool_batch,
                            "deterministic_summary": deterministic_summary,
                            "memory_notes": [record.content for record in memory_notes],
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                ),
            ],
            session_id=session_id,
            job_id="commander-update",
        )

        structured = await extract_structured(
            handle.provider,
            [
                Message.system(
                    "Convert the commander update and tool batch into a structured packet. "
                    "Use only confirmed tool outputs and deterministic batch summary. "
                    "Do not invent dependency health when dependency_probe is unavailable."
                ),
                Message.user(
                    json.dumps(
                        {
                            "incident_packet": INCIDENT_PACKET,
                            "tool_batch": tool_batch,
                            "deterministic_summary": deterministic_summary,
                            "commander_update": summary_result.content,
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
                        "confirmed_signal": {"type": "array", "items": {"type": "string"}},
                        "degraded_evidence": {"type": "array", "items": {"type": "string"}},
                        "unavailable_evidence": {"type": "array", "items": {"type": "string"}},
                        "immediate_actions": {"type": "array", "items": {"type": "string"}},
                        "evidence_used": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "overall_status",
                        "confirmed_signal",
                        "degraded_evidence",
                        "unavailable_evidence",
                        "immediate_actions",
                        "evidence_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
            engine=engine,
            context=RequestContext(session_id=session_id, job_id="structured-partial-failure-packet"),
            model=handle.model,
        )

        normalized_structured_data = _normalize_structured_packet(structured.data, batch_summary=deterministic_summary)
        assembled_summary = _assembled_summary(normalized_structured_data)
        await memory.write(
            MemoryWrite(
                scope=PARTIAL_SCOPE,
                content=json.dumps(normalized_structured_data, ensure_ascii=True, sort_keys=True),
                relevance=0.95,
                metadata={"kind": "partial_failure_packet"},
            )
        )
        memory_after = await memory.retrieve(MemoryQuery(scope=PARTIAL_SCOPE, limit=6))

        latest_request_report = list(lifecycle.requests.values())[-1] if lifecycle.requests else None
        latest_session_report = lifecycle.sessions.get(session_id)

        print_heading("Tool Calling With Partial Failures")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "incident_packet": INCIDENT_PACKET,
                "memory_bootstrap": memory_bootstrap,
                "tool_calls": [_serialize_tool_call(tool_call) for tool_call in tool_calls],
                "tool_batch": tool_batch,
                "deterministic_summary": deterministic_summary,
                "stream_summary": stream_summary,
                "commander_update": {
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
                    "partial_failure_demonstrated": tool_batch["has_partial"] and tool_batch["has_errors"],
                    "streamed_summary_run": bool(stream_summary["event_type_counts"]),
                    "deterministic_summary_present": bool(deterministic_summary["confirmed_signal"]),
                    "structured_packet_ready": structured.valid and bool(normalized_structured_data.get("immediate_actions")),
                    "operator_ready": bool(assembled_summary),
                },
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
