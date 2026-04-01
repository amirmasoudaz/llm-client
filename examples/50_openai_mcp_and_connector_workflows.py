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


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_OPENAI_TOOLS_MODEL", "gpt-5-mini") or "gpt-5-mini"
    connector_id = example_env("LLM_CLIENT_EXAMPLE_CONNECTOR_ID")
    connector_authorization = example_env("LLM_CLIENT_EXAMPLE_CONNECTOR_AUTHORIZATION")
    mcp_server_url = example_env("LLM_CLIENT_EXAMPLE_MCP_SERVER_URL")
    mcp_authorization = example_env("LLM_CLIENT_EXAMPLE_MCP_AUTHORIZATION")
    if not connector_id and not mcp_server_url:
        fail_or_skip(
            "Set LLM_CLIENT_EXAMPLE_CONNECTOR_ID and/or LLM_CLIENT_EXAMPLE_MCP_SERVER_URL "
            "to run the connector/MCP workflow example."
        )

    handle = build_provider_handle("openai", model_name, use_responses_api=True)

    try:
        engine = ExecutionEngine(provider=handle.provider)
        results: dict[str, object] = {}

        web_search = await engine.respond_with_web_search(
            "Find the latest public OpenAI guidance about using hosted tools.",
            provider_name="openai",
            model=handle.model,
            tool_config={"search_context_size": "low"},
        )
        results["web_search"] = {
            "content": web_search.content,
            "usage": web_search.usage.to_dict() if web_search.usage else None,
        }

        if mcp_server_url:
            remote_mcp = await engine.respond_with_remote_mcp(
                "Inspect the available remote MCP capabilities and summarize them briefly.",
                provider_name="openai",
                model=handle.model,
                server_url=mcp_server_url,
                server_label=example_env("LLM_CLIENT_EXAMPLE_MCP_SERVER_LABEL", "Remote MCP"),
                authorization=mcp_authorization,
                require_approval=example_env("LLM_CLIENT_EXAMPLE_MCP_REQUIRE_APPROVAL", "never"),
            )
            results["remote_mcp"] = {
                "content": remote_mcp.content,
                "tool_calls": [tool.to_dict() for tool in (remote_mcp.tool_calls or [])],
            }

        if connector_id:
            connector = await engine.respond_with_connector(
                "Use the connector to inspect what data sources are available.",
                provider_name="openai",
                model=handle.model,
                connector_id=connector_id,
                server_label=example_env("LLM_CLIENT_EXAMPLE_CONNECTOR_LABEL", connector_id),
                authorization=connector_authorization,
                require_approval=example_env("LLM_CLIENT_EXAMPLE_CONNECTOR_REQUIRE_APPROVAL", "always"),
            )
            results["connector"] = {
                "content": connector.content,
                "tool_calls": [tool.to_dict() for tool in (connector.tool_calls or [])],
            }

        print_heading("OpenAI MCP And Connector Workflows")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "connector_id": connector_id,
                "connector_authorization_configured": bool(connector_authorization),
                "mcp_server_url": mcp_server_url,
                "mcp_authorization_configured": bool(mcp_authorization),
                "results": results,
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
