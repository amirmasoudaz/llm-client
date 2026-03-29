from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any

from cookbook_support import build_live_provider, close_provider, fail_or_skip, print_heading, print_json

from llm_client.agent import ToolExecutionMode
from llm_client.providers.types import Message, ToolCall
from llm_client.tools import Tool, ToolExecutionEngine, ToolExecutionMetadata, ToolRegistry, ToolResult


RELEASE_PACKET = {
    "release_notes": (
        "Release 2026.03 introduces the llm_client standalone package path, cache policy upgrades, "
        "security defaults, and cookbook examples. Remaining work includes final extraction cleanup "
        "and one consumer migration pass."
    ),
    "blockers_text": (
        "Dashboards are not yet fully validated. Consumer migration notes need sign-off. "
        "Rollback runbook was updated but not rehearsed this week."
    ),
    "dashboards_status": "not_validated",
    "migration_status": "pending_signoff",
    "owner": "platform-foundations",
}


def _keyword_list(text: str) -> list[str]:
    lowered = text.lower()
    findings: list[str] = []
    if "dashboard" in lowered:
        findings.append("dashboards")
    if "migration" in lowered:
        findings.append("consumer migration")
    if "rollback" in lowered or "runbook" in lowered:
        findings.append("rollback readiness")
    if "sign-off" in lowered or "signoff" in lowered:
        findings.append("stakeholder sign-off")
    return findings


async def _summarize_release_notes(release_notes: str) -> dict[str, Any]:
    await asyncio.sleep(0.05)
    return {
        "summary": "Standalone package path is largely ready, but release completion still depends on final cleanup and consumer migration.",
        "highlights": _keyword_list(release_notes) or ["package modernization"],
        "source_excerpt": release_notes[:180],
    }


async def _identify_operational_risks(
    blockers_text: str,
    dashboards_status: str,
    migration_status: str,
) -> dict[str, Any]:
    await asyncio.sleep(0.12)
    risks: list[str] = []
    if dashboards_status != "validated":
        risks.append("Monitoring dashboards are not validated for launch.")
    if migration_status != "complete":
        risks.append("Consumer migration still needs sign-off before release.")
    if "rollback" in blockers_text.lower():
        risks.append("Rollback runbook has not been rehearsed recently.")
    severity = "high" if len(risks) >= 2 else "medium"
    return {
        "severity": severity,
        "risk_count": len(risks),
        "risks": risks,
    }


async def _compute_readiness_score(
    dashboards_status: str,
    migration_status: str,
    blocker_count: int,
) -> dict[str, Any]:
    await asyncio.sleep(0.08)
    score = 92
    if dashboards_status != "validated":
        score -= 18
    if migration_status != "complete":
        score -= 14
    score -= blocker_count * 4
    score = max(score, 25)
    recommendation = "hold" if score < 75 else "ship_with_caution"
    return {
        "score": score,
        "recommendation": recommendation,
        "ready": score >= 75,
    }


async def _draft_next_actions(
    owner: str,
    blockers_text: str,
    migration_status: str,
) -> ToolResult:
    await asyncio.sleep(0.04)
    actions = [
        f"Ask {owner} to validate dashboards before the release window.",
        "Get explicit sign-off on the consumer migration notes.",
    ]
    if migration_status != "complete":
        actions.append("Schedule a final migration checkpoint and capture rollback criteria.")
    if "rollback" in blockers_text.lower():
        actions.append("Run a quick rollback rehearsal before approving launch.")
    return ToolResult(
        content={
            "actions": actions,
            "owner": owner,
        },
        success=True,
        metadata={
            "partial": True,
            "partial_reason": "Action list is tactical and should be reviewed by release ops before execution.",
        },
    )


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="summarize_release_notes",
            description="Summarize the release notes into the most important readiness themes.",
            parameters={
                "type": "object",
                "properties": {
                    "release_notes": {"type": "string"},
                },
                "required": ["release_notes"],
                "additionalProperties": False,
            },
            handler=_summarize_release_notes,
            execution=ToolExecutionMetadata(
                timeout_seconds=10.0,
                retry_attempts=0,
                concurrency_limit=2,
                safety_tags=("read-only",),
                trust_level="high",
            ),
        ),
        Tool(
            name="identify_operational_risks",
            description="Turn the blocker details into a concrete list of operational release risks.",
            parameters={
                "type": "object",
                "properties": {
                    "blockers_text": {"type": "string"},
                    "dashboards_status": {"type": "string"},
                    "migration_status": {"type": "string"},
                },
                "required": ["blockers_text", "dashboards_status", "migration_status"],
                "additionalProperties": False,
            },
            handler=_identify_operational_risks,
            execution=ToolExecutionMetadata(
                timeout_seconds=12.0,
                retry_attempts=1,
                concurrency_limit=1,
                safety_tags=("ops-risk",),
                trust_level="high",
            ),
        ),
        Tool(
            name="compute_readiness_score",
            description="Compute a launch-readiness score from the release gate statuses.",
            parameters={
                "type": "object",
                "properties": {
                    "dashboards_status": {"type": "string"},
                    "migration_status": {"type": "string"},
                    "blocker_count": {"type": "integer"},
                },
                "required": ["dashboards_status", "migration_status", "blocker_count"],
                "additionalProperties": False,
            },
            handler=_compute_readiness_score,
            execution=ToolExecutionMetadata(
                timeout_seconds=8.0,
                retry_attempts=0,
                concurrency_limit=2,
                safety_tags=("scoring",),
                trust_level="medium",
            ),
        ),
        Tool(
            name="draft_next_actions",
            description="Draft the next operational actions needed before approving the release.",
            parameters={
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "blockers_text": {"type": "string"},
                    "migration_status": {"type": "string"},
                },
                "required": ["owner", "blockers_text", "migration_status"],
                "additionalProperties": False,
            },
            handler=_draft_next_actions,
            execution=ToolExecutionMetadata(
                timeout_seconds=8.0,
                retry_attempts=0,
                concurrency_limit=2,
                safety_tags=("advisory",),
                trust_level="medium",
            ),
        ),
    ]


def summarize_batch(mode: ToolExecutionMode, batch, *, batch_duration_ms: float) -> dict[str, Any]:
    status_counts = {"success": 0, "partial": 0, "error": 0, "skipped": 0}
    results: list[dict[str, Any]] = []
    for envelope in batch.results:
        status_counts[envelope.status] += 1
        result_content = envelope.result.content
        if envelope.status == "skipped":
            preview = None
        elif isinstance(result_content, dict):
            preview = result_content
        else:
            preview = str(result_content)
        results.append(
            {
                "tool": envelope.tool_name,
                "status": envelope.status,
                "attempts": envelope.attempts,
                "duration_ms": round(envelope.duration_ms or 0.0, 3),
                "timeout_seconds": envelope.timeout_seconds,
                "retry_attempts": envelope.retry_attempts,
                "concurrency_limit": envelope.concurrency_limit,
                "safety_tags": list(envelope.safety_tags),
                "trust_level": envelope.trust_level,
                "metadata": envelope.metadata,
                "result": preview,
            }
        )
    return {
        "mode": mode.value,
        "batch_duration_ms": round(batch_duration_ms, 3),
        "has_errors": batch.has_errors,
        "has_partial": batch.has_partial,
        "status_counts": status_counts,
        "results": results,
    }


async def run_mode(mode: ToolExecutionMode, tool_calls: Sequence[ToolCall]) -> dict[str, Any]:
    registry = ToolRegistry(build_tools())
    engine = ToolExecutionEngine(registry)
    start = time.monotonic()
    batch = await engine.execute_calls(list(tool_calls), mode=mode)
    batch_duration_ms = (time.monotonic() - start) * 1000
    return summarize_batch(mode, batch, batch_duration_ms=batch_duration_ms)


def serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    try:
        parsed_arguments = tool_call.parse_arguments()
    except Exception:
        parsed_arguments = tool_call.arguments
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
        "parsed_arguments": parsed_arguments,
    }


async def main() -> None:
    handle = build_live_provider()
    try:
        tools = build_tools()
        provider_tools = tools
        planning_result = await handle.provider.complete(
            [
                Message.system(
                    "You are orchestrating a release-readiness tool workflow. "
                    "You must call every available tool exactly once. "
                    "Use only the values from the provided release packet. "
                    "Do not answer directly."
                ),
                Message.user(
                    "Plan the tool calls for this release packet:\n"
                    + json.dumps(RELEASE_PACKET, indent=2)
                    + "\n\n"
                    + "Call these tools exactly once with valid arguments:\n"
                    + "- summarize_release_notes(release_notes)\n"
                    + "- identify_operational_risks(blockers_text, dashboards_status, migration_status)\n"
                    + "- compute_readiness_score(dashboards_status, migration_status, blocker_count)\n"
                    + "- draft_next_actions(owner, blockers_text, migration_status)\n"
                    + "Use blocker_count=3."
                ),
            ],
            tools=provider_tools,
        )
        if not planning_result.tool_calls:
            fail_or_skip("The live model did not emit tool calls for the tool execution showcase.")

        print_heading("Release Packet")
        print_json(RELEASE_PACKET)

        print_heading("Tool Inventory")
        print_json(
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "execution": {
                        "timeout_seconds": tool.execution.timeout_seconds,
                        "retry_attempts": tool.execution.retry_attempts,
                        "concurrency_limit": tool.execution.concurrency_limit,
                        "safety_tags": list(tool.execution.safety_tags),
                        "trust_level": tool.execution.trust_level,
                    },
                }
                for tool in tools
            ]
        )

        print_heading("Live Tool Plan")
        print_json([serialize_tool_call(tool_call) for tool_call in planning_result.tool_calls])

        for mode in (
            ToolExecutionMode.SINGLE,
            ToolExecutionMode.SEQUENTIAL,
            ToolExecutionMode.PARALLEL,
            ToolExecutionMode.PLANNER,
        ):
            print_heading(f"Execution Mode: {mode.value}")
            print_json(await run_mode(mode, planning_result.tool_calls))
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
