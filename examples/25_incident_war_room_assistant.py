from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.agent import Agent, AgentDefinition, AgentExecutionPolicy, ToolExecutionMode
from llm_client.engine import ExecutionEngine
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import Message, StreamEventType, ToolCall, ToolCallDelta
from llm_client.spec import RequestContext
from llm_client.structured import StructuredOutputConfig, extract_structured
from llm_client.tools import Tool, ToolResult


WAR_ROOM_SCOPE = "incident-war-room"
SERVICE_NAME = "checkout-api"
INCIDENT_PACKET = {
    "incident_id": "INC-2403",
    "service": SERVICE_NAME,
    "severity": "sev-1",
    "impact": "18-22% of checkout transactions are failing during a month-end finance window.",
    "started_at": "2026-03-23T19:40:00Z",
    "change_context": "payment routing configuration changed 12 minutes before impact spike",
    "customer_impact": "Enterprise customers report delayed exports and intermittent payment confirmation gaps.",
}


def _truncate(value: Any, max_chars: int = 220) -> str:
    text = str(value)
    return text if len(text) <= max_chars else f"{text[:max_chars].rstrip()}..."


async def _incident_snapshot(service: str) -> dict[str, Any]:
    _ = service
    return {
        "incident_id": INCIDENT_PACKET["incident_id"],
        "service": SERVICE_NAME,
        "severity": "sev-1",
        "status": "active stabilization",
        "impact": INCIDENT_PACKET["impact"],
        "started_at": INCIDENT_PACKET["started_at"],
        "incident_commander": "sre-oncall-17",
    }


async def _service_metrics(service: str) -> dict[str, Any]:
    _ = service
    return {
        "checkout_success_rate_pct": 79.8,
        "error_rate_pct": 20.2,
        "p95_latency_ms": 4810,
        "payment_queue_depth": 18240,
        "webhook_backlog": 3910,
        "reconciliation_jobs_blocked": 14,
    }


async def _dependency_health(service: str) -> dict[str, Any]:
    _ = service
    return {
        "payment_gateway": "degraded after routing config change",
        "webhook_fanout": "lagging with backlog growth",
        "audit_log_pipeline": "healthy",
        "order_db": "healthy",
        "export_workers": "saturated",
    }


async def _change_log(service: str) -> dict[str, Any]:
    _ = service
    return {
        "latest_change": {
            "change_id": "chg_98231",
            "owner": "payments-platform",
            "rolled_out_at": "2026-03-23T19:28:00Z",
            "summary": "Adjusted payment routing weights and webhook dispatch batching thresholds.",
        },
        "risk_note": "Rollback is available, but webhook in-flight duplication must be checked first.",
    }


async def _timeline_events(service: str) -> dict[str, Any]:
    _ = service
    return {
        "events": [
            {"at": "19:28Z", "event": "payment routing config changed"},
            {"at": "19:40Z", "event": "5xx and timeout alerts triggered"},
            {"at": "19:44Z", "event": "support reports enterprise export delays"},
            {"at": "19:48Z", "event": "finance flags month-end reconciliation risk"},
            {"at": "19:51Z", "event": "rollback evaluation opened"},
        ]
    }


async def _mitigation_queue(service: str) -> dict[str, Any]:
    _ = service
    return {
        "items": [
            {"owner": "payments-platform", "action": "complete rollback feasibility check", "status": "in_progress"},
            {"owner": "sre-oncall-17", "action": "pin queue depth and webhook backlog dashboards", "status": "done"},
            {"owner": "data-platform", "action": "assess export worker scaling and replay safety", "status": "queued"},
            {"owner": "support-lead", "action": "prepare internal-only finance update", "status": "in_progress"},
        ]
    }


async def _customer_impact(service: str) -> dict[str, Any]:
    _ = service
    return {
        "affected_segments": ["enterprise checkout", "finance export users", "compliance reporting workflows"],
        "blocked_workflows": ["month-end reconciliation", "export delivery confirmations"],
        "customer_risk": "High: revenue-impacting payment failures plus reporting delays for high-value accounts.",
    }


def _build_tools(memory: ShortTermMemoryStore) -> list[Tool]:
    async def war_room_memory(service: str) -> dict[str, Any]:
        records = await memory.retrieve(MemoryQuery(scope=WAR_ROOM_SCOPE, limit=4))
        return {
            "service": service,
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
            name="incident_snapshot",
            description="Get the current incident snapshot for the affected service.",
            parameters={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            handler=_incident_snapshot,
        ),
        Tool(
            name="service_metrics",
            description="Get the current service metrics for checkout reliability and backlog pressure.",
            parameters={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            handler=_service_metrics,
        ),
        Tool(
            name="dependency_health",
            description="Get dependency health for the affected service.",
            parameters={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            handler=_dependency_health,
        ),
        Tool(
            name="change_log",
            description="Get the latest production change and rollback notes.",
            parameters={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            handler=_change_log,
        ),
        Tool(
            name="timeline_events",
            description="Get the most important incident timeline events for the war room.",
            parameters={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            handler=_timeline_events,
        ),
        Tool(
            name="mitigation_queue",
            description="Get the current mitigation queue and owner assignments.",
            parameters={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            handler=_mitigation_queue,
        ),
        Tool(
            name="customer_impact",
            description="Get the current customer-facing impact and blocked workflows.",
            parameters={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            handler=_customer_impact,
        ),
        Tool(
            name="war_room_memory",
            description="Retrieve recent memory notes, stakeholder constraints, and prior hypotheses for the incident war room.",
            parameters={
                "type": "object",
                "properties": {"service": {"type": "string"}},
                "required": ["service"],
                "additionalProperties": False,
            },
            handler=war_room_memory,
        ),
    ]


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    seed_entries = [
        MemoryWrite(
            scope=WAR_ROOM_SCOPE,
            content="Leadership wants proactive internal updates before any external customer escalation.",
            relevance=0.95,
            metadata={"kind": "stakeholder_preference"},
        ),
        MemoryWrite(
            scope=WAR_ROOM_SCOPE,
            content="Previous payment routing incident showed queue backlog can keep growing for 10-15 minutes after rollback unless webhook workers are drained deliberately.",
            relevance=0.92,
            metadata={"kind": "prior_incident"},
        ),
        MemoryWrite(
            scope=WAR_ROOM_SCOPE,
            content="Finance asked for a no-surprises internal update if reconciliation workflows remain blocked beyond 30 minutes.",
            relevance=0.91,
            metadata={"kind": "stakeholder_risk"},
        ),
        MemoryWrite(
            scope=WAR_ROOM_SCOPE,
            content="Executive audience wants confirmed facts separated from hypotheses and named owners for the next 15 minutes.",
            relevance=0.9,
            metadata={"kind": "exec_expectation"},
        ),
    ]
    written: list[dict[str, Any]] = []
    for entry in seed_entries:
        record = await memory.write(entry)
        written.append({"kind": record.metadata.get("kind"), "content": record.content})
    return written


def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "tool_name": call.name,
            "arguments": call.parse_arguments(),
        }
        for call in tool_calls
    ]


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
            if sum(len(part) for part in token_preview_parts) < 260:
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
    handle = build_live_provider()
    try:
        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)

        lifecycle = LifecycleRecorder()
        diagnostics = EngineDiagnosticsRecorder()
        hooks = HookManager([lifecycle, diagnostics])
        engine = ExecutionEngine(provider=handle.provider, hooks=hooks)

        tools = _build_tools(memory)
        agent = Agent(
            engine=engine,
            definition=AgentDefinition(
                name="incident-war-room-assistant",
                system_message=(
                    "You are an incident war-room assistant for executive and operator audiences. "
                    "Before finalizing, gather evidence with at least five distinct tools. "
                    "Separate confirmed facts from hypotheses. Name owners, immediate mitigations, and the next 15-minute plan. "
                    "Return sections: Current State, Incident Timeline, Hypothesis Board, Mitigation Queue, Executive Update, Risks and Unknowns, Next 15 Minutes."
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

        prompt = (
            f"Incident packet: {INCIDENT_PACKET}\n\n"
            "An executive standup starts in 10 minutes. Build a war-room update for the current incident. "
            "Use tool evidence, reflect memory-backed stakeholder constraints, and make the output operationally actionable."
        )
        request_context = RequestContext(
            session_id="cookbook-incident-war-room",
            job_id="war-room-brief",
            tags={"incident_id": INCIDENT_PACKET["incident_id"], "service": SERVICE_NAME},
        )
        result, stream_summary = await _run_agent_stream(agent, prompt, request_context)

        structured = await extract_structured(
            handle.provider,
            [
                Message.system(
                    (
                        "Convert the war-room update into a structured executive incident packet. "
                        "Keep confirmed facts and hypotheses separate."
                    )
                ),
                Message.user(result.content or ""),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "current_state": {"type": "string"},
                        "confirmed_facts": {"type": "array", "items": {"type": "string"}},
                        "top_hypotheses": {"type": "array", "items": {"type": "string"}},
                        "mitigation_actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "owner": {"type": "string"},
                                    "action": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                                "required": ["owner", "action", "status"],
                                "additionalProperties": False,
                            },
                        },
                        "executive_update": {"type": "string"},
                        "next_15_minutes": {"type": "array", "items": {"type": "string"}},
                        "customer_risk": {"type": "string"},
                    },
                    "required": [
                        "current_state",
                        "confirmed_facts",
                        "top_hypotheses",
                        "mitigation_actions",
                        "executive_update",
                        "next_15_minutes",
                        "customer_risk",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )

        await memory.write(
            MemoryWrite(
                scope=WAR_ROOM_SCOPE,
                content=f"War-room executive packet: {structured.data}",
                relevance=0.96,
                metadata={"kind": "executive_packet"},
            )
        )
        memory_after = await memory.retrieve(MemoryQuery(scope=WAR_ROOM_SCOPE, limit=6))

        distinct_tool_names = sorted({call.name for call in result.all_tool_calls})
        latest_request_report = list(lifecycle.requests.values())[-1] if lifecycle.requests else None
        latest_session_report = lifecycle.sessions.get(request_context.session_id or "")

        print_heading("Incident War Room Assistant")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "incident_packet": INCIDENT_PACKET,
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
                "structured_packet": {
                    "valid": structured.valid,
                    "repair_attempts": structured.repair_attempts,
                    "usage": summarize_usage(getattr(structured, "usage", None)),
                    "data": structured.data,
                },
                "observability": {
                    "hook_event_counts": dict(Counter(event for event, _, _ in diagnostics.events)),
                    "lifecycle_event_counts": dict(Counter(event.type.value for event in lifecycle.events)),
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
                    "memory_backed": any(record.metadata.get("kind") == "executive_packet" for record in memory_after),
                    "executive_packet_ready": structured.valid,
                },
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
