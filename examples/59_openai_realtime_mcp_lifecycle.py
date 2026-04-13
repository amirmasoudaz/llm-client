from __future__ import annotations

import asyncio

from cookbook_support import (
    build_provider_handle,
    close_provider,
    example_env,
    fail_or_skip,
    print_heading,
    print_json,
)

from llm_client.engine import ExecutionEngine
from llm_client.tools import ResponsesMCPTool


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_REALTIME_MODEL", "gpt-realtime") or "gpt-realtime"
    server_url = example_env("LLM_CLIENT_EXAMPLE_MCP_SERVER_URL")
    if not server_url:
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_MCP_SERVER_URL to run the realtime MCP lifecycle example.")

    handle = build_provider_handle("openai", model_name)
    try:
        engine = ExecutionEngine(provider=handle.provider)
        try:
            connection = await asyncio.wait_for(
                engine.connect_realtime(
                    provider_name="openai",
                    model=handle.model,
                ),
                timeout=10.0,
            )
        except TimeoutError:
            fail_or_skip("Timed out while connecting to the OpenAI realtime websocket.")

        try:
            tool = ResponsesMCPTool.remote_server(
                server_url,
                server_label=example_env("LLM_CLIENT_EXAMPLE_MCP_SERVER_LABEL", "Remote MCP"),
                authorization=example_env("LLM_CLIENT_EXAMPLE_MCP_AUTHORIZATION"),
                require_approval=example_env("LLM_CLIENT_EXAMPLE_MCP_REQUIRE_APPROVAL", "never"),
            )
            await connection.update_session_tools(
                [tool],
                session={"modalities": ["text"]},
                event_id="evt_session_mcp",
            )

            try:
                listing = await connection.wait_for_mcp_tool_listing(
                    server_label=tool.server_label,
                    timeout=10.0,
                )
            except TimeoutError:
                fail_or_skip("Timed out while waiting for realtime MCP tools to finish loading.")

            print_heading("OpenAI Realtime MCP Lifecycle")
            print_json(
                {
                    "provider": handle.name,
                    "model": handle.model,
                    "connection": connection.to_dict(),
                    "tool": tool.to_dict(),
                    "listing": listing.to_dict(),
                }
            )
        finally:
            await connection.close()
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
