from __future__ import annotations

import asyncio
from typing import Any

from cookbook_support import (
    build_live_provider,
    close_provider,
    fail_or_skip,
    print_heading,
    print_json,
    summarize_usage,
)

from llm_client.agent import Agent, AgentDefinition, AgentExecutionPolicy, ToolExecutionMode
from llm_client.tools import tool


INCIDENT_CONTEXT = {
    "service": "checkout-api",
    "requested_briefing": (
        "Prepare a 15-minute incident commander briefing with severity, likely cause, "
        "immediate actions, deployment-freeze recommendation, owners, and evidence."
    ),
}


@tool
async def get_incident_snapshot(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "severity": "SEV-1",
        "status": "active",
        "customer_impact": "Checkout failures affecting 18-22% of transactions in us-east-1 and eu-west-1.",
        "error_rate": "21.4%",
        "started_at": "2026-03-23T13:42:00Z",
    }


@tool
async def get_recent_alerts(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "alerts": [
            "5xx rate above threshold for 17 minutes",
            "payment authorization timeout saturation",
            "checkout queue depth rising",
        ],
        "paging_state": "incident channel active, page acknowledged",
    }


@tool
async def get_dependency_health(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "dependencies": [
            {
                "name": "payment-gateway",
                "status": "degraded",
                "notes": "timeouts increased sharply after a config rollout",
            },
            {
                "name": "fraud-check",
                "status": "healthy",
                "notes": "latency normal",
            },
            {
                "name": "order-db",
                "status": "healthy",
                "notes": "replication stable",
            },
        ],
    }


@tool
async def get_runbook_guidance(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "runbook": "checkout-api-major-incident",
        "recommended_steps": [
            "freeze new deploys to checkout and payment-edge",
            "shift traffic away from the degraded payment gateway pool if error budget allows",
            "confirm whether the last payment configuration rollout can be safely rolled back",
            "post a customer-facing status update if impact persists beyond 15 minutes",
        ],
        "rollback_ready": True,
    }


@tool
async def get_oncall_ownership(service: str) -> dict[str, Any]:
    return {
        "service": service,
        "incident_commander": "Maya Chen",
        "service_owner": "Checkout Platform",
        "slack_channel": "#inc-checkout-sev1",
        "executive_escalation": "VP, Commerce Infrastructure",
    }


def serialize_turn(turn) -> dict[str, Any]:
    return {
        "turn_number": turn.turn_number,
        "assistant_content": turn.content,
        "tool_calls": [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "parsed_arguments": tool_call.parse_arguments(),
            }
            for tool_call in turn.tool_calls
        ],
        "tool_results": [
            {
                "success": tool_result.success,
                "error": tool_result.error,
                "content": tool_result.content,
                "metadata": tool_result.metadata,
            }
            for tool_result in turn.tool_results
        ],
    }


async def main() -> None:
    handle = build_live_provider()
    try:
        agent = Agent(
            provider=handle.provider,
            tools=[
                get_incident_snapshot,
                get_recent_alerts,
                get_dependency_health,
                get_runbook_guidance,
                get_oncall_ownership,
            ],
            definition=AgentDefinition(
                name="incident_commander_agent",
                system_message=(
                    "You are a senior incident commander. "
                    "You must call tools before answering. Gather evidence first, then produce an executive-quality "
                    "incident briefing with severity, likely cause, immediate actions, deployment-freeze recommendation, "
                    "owners, and evidence bullets. Cite the tool names you used in parentheses."
                ),
                execution_policy=AgentExecutionPolicy(
                    max_turns=4,
                    tool_execution_mode=ToolExecutionMode.PARALLEL,
                    max_tool_calls_per_turn=8,
                ),
                metadata={"scenario": "incident-briefing"},
            ),
        )

        result = await agent.run(
            "Prepare the incident briefing for checkout-api. "
            "Use the tools to gather the current incident snapshot, recent alerts, dependency health, runbook guidance, "
            "and on-call ownership before you answer."
        )

        if not result.all_tool_calls:
            fail_or_skip("The live model did not use tools for the tool-calling agent showcase.")

        print_heading("Incident Context")
        print_json(INCIDENT_CONTEXT)

        print_heading("Tool-Calling Agent")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "status": result.status,
                "turn_count": len(result.turns),
                "tool_call_count": len(result.all_tool_calls),
                "tool_names_used": [tool_call.name for tool_call in result.all_tool_calls],
                "usage": summarize_usage(result.total_usage),
                "final_content": result.content,
            }
        )

        print_heading("Agent Turns")
        print_json([serialize_turn(turn) for turn in result.turns])
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
