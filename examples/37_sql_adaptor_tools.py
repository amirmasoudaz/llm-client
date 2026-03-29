from __future__ import annotations

import asyncio
from typing import Any

from cookbook_support import (
    build_live_provider,
    close_provider,
    fail_or_skip,
    print_heading,
    print_json,
    require_database_dsn,
    require_optional_module,
    summarize_usage,
)

if not require_optional_module("asyncpg", "Install it with: pip install llm-client[postgres]"):
    raise SystemExit(0)

import asyncpg

from llm_client.adapters import PostgresSQLAdaptor, SQLMutationRequest, build_sql_query_tool
from llm_client.agent import Agent, AgentDefinition, AgentExecutionPolicy, ToolExecutionMode


def _serialize_turn(turn: Any) -> dict[str, Any]:
    return {
        "turn_number": turn.turn_number,
        "assistant_content": turn.content,
        "tool_calls": [
            {
                "name": tool_call.name,
                "arguments": tool_call.parse_arguments(),
            }
            for tool_call in turn.tool_calls
        ],
        "tool_results": [
            {
                "success": tool_result.success,
                "error": tool_result.error,
                "content_preview": str(tool_result.content)[:240],
            }
            for tool_result in turn.tool_results
        ],
    }


async def main() -> None:
    dsn = require_database_dsn()
    handle = build_live_provider()
    try:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    except Exception as exc:
        await close_provider(handle.provider)
        fail_or_skip(
            "Could not connect to PostgreSQL for the SQL adaptor tool example. "
            f"Check LLM_CLIENT_EXAMPLE_PG_DSN and database availability. {type(exc).__name__}: {exc}"
        )
        return
    try:
        async with pool.acquire() as conn:
            writable = PostgresSQLAdaptor(conn, read_only=False, default_timeout_seconds=10.0)
            await writable.execute(
                SQLMutationRequest(
                    statement=(
                        "create temporary table incident_actions ("
                        "action_id integer primary key, owner text, severity text, state text, action_name text)"
                    ),
                    allow_write=True,
                )
            )
            for action_id, owner, severity, state, action_name in [
                (1, "platform", "sev-1", "open", "freeze deploys"),
                (2, "payments", "sev-1", "open", "roll back gateway config"),
                (3, "platform", "sev-1", "open", "shift traffic away from degraded shard"),
                (4, "support", "sev-2", "closed", "prepare customer status note"),
            ]:
                await writable.execute(
                    SQLMutationRequest(
                        statement=(
                            "insert into incident_actions(action_id, owner, severity, state, action_name) "
                            "values (:action_id, :owner, :severity, :state, :action_name)"
                        ),
                        parameters={
                            "action_id": action_id,
                            "owner": owner,
                            "severity": severity,
                            "state": state,
                            "action_name": action_name,
                        },
                        allow_write=True,
                    )
                )

            readonly = PostgresSQLAdaptor(conn, read_only=True, default_timeout_seconds=10.0)
            sql_tool = build_sql_query_tool(
                readonly,
                name="query_incident_actions_sql",
                description=(
                    "Run read-only SQL against the incident_actions table. "
                    "Only use SELECT statements. The table columns are: action_id, owner, severity, state, action_name."
                ),
            )

            agent = Agent(
                provider=handle.provider,
                tools=[sql_tool],
                definition=AgentDefinition(
                    name="sql_adaptor_tool_agent",
                    system_message=(
                        "You are an incident operations analyst. "
                        "You must call query_incident_actions_sql before answering. "
                        "Use only the tool data. Answer with the owner who has the most open actions, "
                        "the open action names, and a short operator summary."
                    ),
                    execution_policy=AgentExecutionPolicy(
                        max_turns=3,
                        tool_execution_mode=ToolExecutionMode.SINGLE,
                        max_tool_calls_per_turn=4,
                    ),
                ),
            )

            result = await agent.run(
                "Inspect the incident_actions table and tell me which owner currently has the most open actions. "
                "List the open action names grouped by owner."
            )

            if not result.all_tool_calls:
                fail_or_skip("The live model did not use the SQL adaptor tool example as expected.")

            print_heading("SQL Adaptor Tool Agent")
            print_json(
                {
                    "provider": handle.name,
                    "model": handle.model,
                    "status": result.status,
                    "tool_call_count": len(result.all_tool_calls),
                    "tool_names_used": [tool_call.name for tool_call in result.all_tool_calls],
                    "usage": summarize_usage(result.total_usage),
                    "final_content": result.content,
                }
            )

            print_heading("Agent Turns")
            print_json([_serialize_turn(turn) for turn in result.turns])
    finally:
        await pool.close()
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
