from __future__ import annotations

import asyncio
import json
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


RELEASE_SCOPE = "release-readiness-control-plane"
RELEASE_PACKET = {
    "release_id": "REL-2026-03B",
    "release_train": "2026.03B",
    "owner": "platform-foundations",
    "target_scope": "standalone package rollout plus hosted API consumers",
    "release_notes": [
        "Standalone package path is ready for wider adoption.",
        "Cache policy upgrades reduce cross-tenant cache bleed risk and improve hit-rate consistency.",
        "Cookbook expansion and llm_client example upgrades are queued for release.",
        "Phase-13 engine and observability fixes are included in the train.",
    ],
    "business_context": {
        "month_end_hours_until_freeze": 36,
        "leadership_expectation": "no-surprises internal update before any external launch communication",
        "finance_sensitivity": "billing and reconciliation workflows are sensitive near month-end close",
    },
    "proposed_window": {
        "starts_at": "2026-03-24T17:00:00Z",
        "ends_at": "2026-03-24T19:00:00Z",
        "change_type": "progressive rollout",
    },
}


def _truncate(value: Any, max_chars: int = 220) -> str:
    text = str(value)
    return text if len(text) <= max_chars else f"{text[:max_chars].rstrip()}..."


def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [{"tool_name": call.name, "arguments": call.parse_arguments()} for call in tool_calls]


def _serialize_tool_results(tool_results: list[ToolResult]) -> list[dict[str, Any]]:
    return [
        {
            "success": result.success,
            "error": result.error,
            "content_preview": _truncate(result.to_string(), 280),
        }
        for result in tool_results
    ]


def _serialize_turns(turns: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "turn_number": turn.turn_number + 1,
            "assistant_preview": _truncate(turn.content or "", 320),
            "tool_calls": _serialize_tool_calls(turn.tool_calls),
            "tool_results": _serialize_tool_results(turn.tool_results),
        }
        for turn in turns
    ]


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    seed_entries = [
        MemoryWrite(
            scope=RELEASE_SCOPE,
            content="Leadership accepts a conditional go only if the audience, rollback trigger, and owner checklist are explicit.",
            relevance=0.95,
            metadata={"kind": "leadership_rule"},
        ),
        MemoryWrite(
            scope=RELEASE_SCOPE,
            content="Previous package rollout showed skipped rollback rehearsal plus partial consumer migration created same-day hotfix pressure.",
            relevance=0.93,
            metadata={"kind": "prior_incident"},
        ),
        MemoryWrite(
            scope=RELEASE_SCOPE,
            content="Support wants a maintenance flag and escalation script ready if package consumers see migration regressions after launch.",
            relevance=0.91,
            metadata={"kind": "support_constraint"},
        ),
        MemoryWrite(
            scope=RELEASE_SCOPE,
            content="Finance expects a go/no-go recommendation that explicitly mentions month-end workflow risk.",
            relevance=0.92,
            metadata={"kind": "finance_constraint"},
        ),
    ]
    written: list[dict[str, Any]] = []
    for entry in seed_entries:
        record = await memory.write(entry)
        written.append({"kind": record.metadata.get("kind"), "content": record.content})
    return written


def _deterministic_policy_snapshot() -> dict[str, Any]:
    broad_release_ready = False
    internal_canary_ready = True
    decision = "conditional_go" if internal_canary_ready and not broad_release_ready else "go"
    return {
        "policy_version": "release-control-plane-v2",
        "rules": [
            "Hold broad rollout if rollback rehearsal is not completed within the current train.",
            "Hold broad rollout if critical consumer migrations remain incomplete.",
            "Conditional go is allowed for internal/canary scope if quality signals are healthy and rollback runbook exists.",
            "Finance-sensitive windows require explicit communication and rollback trigger ownership.",
        ],
        "evaluated_facts": {
            "rollback_rehearsal_completed": False,
            "critical_consumer_migrations_pending": 2,
            "dashboard_validation_complete": False,
            "quality_signal_regressions_blocking": False,
            "change_window_before_month_end_freeze_hours": 36,
        },
        "recommended_decision": decision,
        "recommended_scope": "internal canary only" if decision == "conditional_go" else "full rollout",
        "policy_risk_level": "high" if decision != "go" else "medium",
        "blocking_reasons": [
            "Rollback rehearsal skipped this week.",
            "Two critical package consumers have not completed migration validation.",
            "Release dashboards still need final validation for launch observability.",
        ],
    }


def _build_tools(memory: ShortTermMemoryStore) -> list[Tool]:
    async def release_manifest() -> dict[str, Any]:
        return {
            "release_id": RELEASE_PACKET["release_id"],
            "release_train": RELEASE_PACKET["release_train"],
            "owner": RELEASE_PACKET["owner"],
            "target_scope": RELEASE_PACKET["target_scope"],
            "release_notes": RELEASE_PACKET["release_notes"],
        }

    async def blocker_board() -> dict[str, Any]:
        return {
            "blockers": [
                {
                    "id": "BLK-201",
                    "title": "Final consumer migration pass not complete",
                    "severity": "high",
                    "owner": "platform-foundations",
                    "status": "in_progress",
                    "detail": "2 critical package consumers still need validation on the standalone package path.",
                },
                {
                    "id": "BLK-202",
                    "title": "Dashboard validation pending",
                    "severity": "medium",
                    "owner": "sre-observability",
                    "status": "queued",
                    "detail": "Launch dashboard still needs final validation for cache hit ratio and error-budget burn views.",
                },
                {
                    "id": "BLK-203",
                    "title": "Rollback rehearsal skipped this week",
                    "severity": "high",
                    "owner": "release-manager",
                    "status": "not_started",
                    "detail": "Runbook exists, but timed rehearsal and owner acknowledgment are incomplete for this train.",
                },
            ]
        }

    async def quality_signals() -> dict[str, Any]:
        return {
            "unit_tests": {"passed": 412, "failed": 0},
            "integration_tests": {"passed": 58, "failed": 0},
            "release_candidate_smoke": {"status": "passed", "notes": "Hosted API and packaging smoke checks are green."},
            "perf_regression": {"status": "improved", "latency_delta_pct": -8.4},
            "known_quality_risks": ["consumer migration variance not yet validated on two critical clients"],
        }

    async def dependency_health() -> dict[str, Any]:
        return {
            "package_registry": {"status": "healthy", "detail": "Artifacts propagated to primary mirror."},
            "hosted_api": {"status": "healthy", "detail": "No blocker-level regressions in staging."},
            "consumer_adoption": {
                "status": "partial",
                "critical_pending": ["billing-reconciler", "support-ops-exporter"],
            },
            "observability_stack": {"status": "partial", "detail": "Launch dashboard validation still pending."},
        }

    async def rollback_posture() -> dict[str, Any]:
        return {
            "runbook_present": True,
            "rehearsal_status": "skipped_this_week",
            "rollback_eta_minutes": 18,
            "owner_on_call": "release-manager",
            "gaps": [
                "No recorded rehearsal artifact for this train.",
                "Critical consumer rollback acknowledgements missing.",
            ],
        }

    async def change_window() -> dict[str, Any]:
        return {
            "window": RELEASE_PACKET["proposed_window"],
            "month_end_hours_until_freeze": RELEASE_PACKET["business_context"]["month_end_hours_until_freeze"],
            "constraints": [
                "Finance workflows become sensitive inside the next 36 hours.",
                "Leadership expects proactive internal communication before launch.",
            ],
        }

    async def stakeholder_state() -> dict[str, Any]:
        return {
            "leadership": "Needs explicit no-surprises update with launch scope and rollback trigger.",
            "finance": "Wants month-end workflow risk called out in the launch recommendation.",
            "support": "Needs maintenance flag and escalation script if conditional go proceeds.",
            "sre": "Wants dashboard validation complete before broad rollout.",
        }

    async def gate_policy() -> dict[str, Any]:
        return _deterministic_policy_snapshot()

    async def release_memory(topic: str) -> dict[str, Any]:
        records = await memory.retrieve(MemoryQuery(scope=RELEASE_SCOPE, limit=5))
        return {
            "topic": topic,
            "notes": [{"kind": record.metadata.get("kind"), "content": record.content} for record in records],
        }

    return [
        Tool(
            name="release_manifest",
            description="Return release train metadata, ownership, and the main release notes.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=release_manifest,
        ),
        Tool(
            name="blocker_board",
            description="Return the current release blockers, owners, severity, and status.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=blocker_board,
        ),
        Tool(
            name="quality_signals",
            description="Return test, smoke, and performance signals for the current release candidate.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=quality_signals,
        ),
        Tool(
            name="dependency_health",
            description="Return readiness of release dependencies such as registries, APIs, observability, and downstream consumers.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=dependency_health,
        ),
        Tool(
            name="rollback_posture",
            description="Return rollback readiness, rehearsal status, ETA, and known gaps.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=rollback_posture,
        ),
        Tool(
            name="change_window",
            description="Return the proposed launch window and timing constraints around month-end and stakeholder expectations.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=change_window,
        ),
        Tool(
            name="stakeholder_state",
            description="Return current stakeholder expectations for leadership, finance, support, and SRE.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=stakeholder_state,
        ),
        Tool(
            name="gate_policy",
            description="Return deterministic release gate rules and the current policy recommendation based on known facts.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=gate_policy,
        ),
        Tool(
            name="release_memory",
            description="Return memory-backed lessons and stakeholder constraints relevant to the release decision.",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            handler=release_memory,
        ),
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
            if sum(len(part) for part in token_preview_parts) < 300:
                token_preview_parts.append(str(event.data))
            continue

        if event.type in {StreamEventType.TOOL_CALL_START, StreamEventType.TOOL_CALL_DELTA, StreamEventType.TOOL_CALL_END}:
            payload = event.data
            if isinstance(payload, ToolCallDelta):
                tool_call_events.append(
                    {
                        "event": event.type.value,
                        "tool_name": payload.name,
                        "arguments_delta": _truncate(payload.arguments_delta, 180),
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
                        "content_preview": _truncate(payload.get("content"), 240),
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


def _assembled_summary(structured_data: dict[str, Any] | None) -> str:
    if not structured_data:
        return ""
    blockers = "\n".join(
        f"- {item['title']} ({item['severity']}; owner={item['owner']})"
        for item in structured_data.get("top_blockers", [])
    )
    actions = "\n".join(f"- {item}" for item in structured_data.get("required_actions", []))
    return (
        f"Gate Decision\n- {structured_data.get('go_no_go', '')} | risk={structured_data.get('risk_level', '')} | "
        f"scope={structured_data.get('launch_scope', '')}\n\n"
        f"Executive Summary\n- {structured_data.get('overall_status', '')}\n\n"
        f"Top Blockers\n{blockers}\n\n"
        f"Rollback Plan\n- {structured_data.get('rollback_plan', '')}\n\n"
        f"Required Actions\n{actions}\n\n"
        f"Comms Plan\n- {structured_data.get('comms_plan', '')}"
    ).strip()


def _normalize_structured_packet(structured_data: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(structured_data or {})
    required_actions = [str(item).strip() for item in data.get("required_actions", []) if str(item).strip()]
    if not required_actions:
        required_actions = [
            "Complete rollback rehearsal, dashboard validation, and critical consumer migration checks before broad rollout."
        ]
    evidence_used = [str(item).strip() for item in data.get("evidence_used", []) if str(item).strip()]
    top_blockers = [
        {
            "title": str(item.get("title", "")).strip(),
            "severity": str(item.get("severity", "")).strip(),
            "owner": str(item.get("owner", "")).strip(),
            "status": str(item.get("status", "")).strip(),
        }
        for item in data.get("top_blockers", [])
        if str(item.get("title", "")).strip()
    ]
    data["required_actions"] = required_actions
    data["evidence_used"] = evidence_used
    data["top_blockers"] = top_blockers
    return data


async def main() -> None:
    handle = build_live_provider()
    try:
        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)
        policy_snapshot = _deterministic_policy_snapshot()

        lifecycle = LifecycleRecorder()
        diagnostics = EngineDiagnosticsRecorder()
        hooks = HookManager([lifecycle, diagnostics])
        engine = ExecutionEngine(provider=handle.provider, hooks=hooks)

        tools = _build_tools(memory)
        agent = Agent(
            engine=engine,
            definition=AgentDefinition(
                name="release-readiness-control-plane",
                system_message=(
                    "You are a release-readiness control plane for engineering leadership. "
                    "Before finalizing, gather evidence with at least six distinct tools, including gate_policy and rollback_posture. "
                    "Separate confirmed facts from launch assumptions. "
                    "Make the launch scope explicit: full rollout, conditional canary, or hold. "
                    "Name blocking conditions, rollback expectations, communications, and the next decision point. "
                    "Return sections: Current Release State, Gate Decision, Blocking Conditions, Conditional Launch Plan, Rollback & Safety Net, Comms Plan, Next 24 Hours."
                ),
                execution_policy=AgentExecutionPolicy(
                    max_turns=6,
                    max_tool_calls_per_turn=9,
                    tool_execution_mode=ToolExecutionMode.PARALLEL,
                    stop_on_tool_error=False,
                ),
            ),
            tools=tools,
        )

        prompt = (
            f"Release packet: {RELEASE_PACKET}\n\n"
            f"Deterministic gate baseline: {policy_snapshot}\n\n"
            "Build a release-readiness recommendation for the current train. Use tool evidence, respect stakeholder and month-end constraints, "
            "and produce an operator-ready go/hold recommendation with launch scope, blockers, rollback posture, and communications."
        )
        request_context = RequestContext(
            session_id="cookbook-release-control-plane",
            job_id="release-readiness",
            tags={"release_id": RELEASE_PACKET["release_id"], "release_train": RELEASE_PACKET["release_train"]},
        )
        result, stream_summary = await _run_agent_stream(agent, prompt, request_context)

        structured = await extract_structured(
            handle.provider,
            [
                Message.system(
                    (
                        "Convert the release-readiness briefing into a structured release control packet. "
                        "Keep the decision grounded in blocker status, rollback posture, policy evidence, and stakeholder constraints."
                    )
                ),
                Message.user(
                    json.dumps(
                        {
                            "release_packet": RELEASE_PACKET,
                            "policy_snapshot": policy_snapshot,
                            "agent_output": result.content,
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
                        "go_no_go": {"type": "string", "enum": ["go", "hold", "conditional_go"]},
                        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                        "launch_scope": {"type": "string"},
                        "release_readiness_score": {"type": "integer"},
                        "top_blockers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "severity": {"type": "string"},
                                    "owner": {"type": "string"},
                                    "status": {"type": "string"},
                                },
                                "required": ["title", "severity", "owner", "status"],
                                "additionalProperties": False,
                            },
                        },
                        "required_actions": {"type": "array", "items": {"type": "string"}},
                        "rollback_plan": {"type": "string"},
                        "comms_plan": {"type": "string"},
                        "next_decision_point": {"type": "string"},
                        "evidence_used": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "overall_status",
                        "go_no_go",
                        "risk_level",
                        "launch_scope",
                        "release_readiness_score",
                        "top_blockers",
                        "required_actions",
                        "rollback_plan",
                        "comms_plan",
                        "next_decision_point",
                        "evidence_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )
        normalized_structured_data = _normalize_structured_packet(structured.data)

        assembled_summary = _assembled_summary(normalized_structured_data)
        await memory.write(
            MemoryWrite(
                scope=RELEASE_SCOPE,
                content=json.dumps(normalized_structured_data, ensure_ascii=True, sort_keys=True),
                relevance=0.96,
                metadata={"kind": "release_packet"},
            )
        )
        memory_after = await memory.retrieve(MemoryQuery(scope=RELEASE_SCOPE, limit=6))

        latest_request_report = list(lifecycle.requests.values())[-1] if lifecycle.requests else None
        latest_session_report = lifecycle.sessions.get(request_context.session_id or "")
        distinct_tool_names = sorted({call.name for call in result.all_tool_calls})

        print_heading("Release Readiness Control Plane")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "release_packet": RELEASE_PACKET,
                "policy_snapshot": policy_snapshot,
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
                    "streamed_agent_run": bool(stream_summary["event_type_counts"]),
                    "used_six_plus_tools": len(distinct_tool_names) >= 6,
                    "policy_backed": bool(policy_snapshot.get("recommended_decision")),
                    "memory_backed": any(record.metadata.get("kind") == "release_packet" for record in memory_after),
                    "operator_ready": structured.valid and bool(normalized_structured_data.get("required_actions")),
                },
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
