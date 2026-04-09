from __future__ import annotations

import asyncio

from cookbook_support import (
    build_live_provider,
    close_provider,
    example_env,
    fail_or_skip,
    print_heading,
    print_json,
    summarize_usage,
)

from llm_client.engine import ExecutionEngine


async def main() -> None:
    handle = build_live_provider(use_responses_api=True)
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the OpenAI MCP approval continuation example.")

    previous_response_id = example_env("LLM_CLIENT_EXAMPLE_MCP_PREVIOUS_RESPONSE_ID")
    approval_request_id = example_env("LLM_CLIENT_EXAMPLE_MCP_APPROVAL_REQUEST_ID")
    approve = (example_env("LLM_CLIENT_EXAMPLE_MCP_APPROVE", "1") or "1").strip().lower() not in {"0", "false", "no", "deny"}
    if not previous_response_id or not approval_request_id:
        fail_or_skip(
            "Set LLM_CLIENT_EXAMPLE_MCP_PREVIOUS_RESPONSE_ID and "
            "LLM_CLIENT_EXAMPLE_MCP_APPROVAL_REQUEST_ID to continue an MCP approval loop."
        )
    connector_id = example_env("LLM_CLIENT_EXAMPLE_CONNECTOR_ID")
    connector_authorization = example_env("LLM_CLIENT_EXAMPLE_CONNECTOR_AUTHORIZATION")
    mcp_server_url = example_env("LLM_CLIENT_EXAMPLE_MCP_SERVER_URL")
    mcp_authorization = example_env("LLM_CLIENT_EXAMPLE_MCP_AUTHORIZATION")

    continuation_kwargs: dict[str, object] = {}
    if connector_id:
        continuation_kwargs.update(
            {
                "connector_id": connector_id,
                "server_label": example_env("LLM_CLIENT_EXAMPLE_CONNECTOR_LABEL", connector_id),
                "authorization": connector_authorization,
                "require_approval": example_env("LLM_CLIENT_EXAMPLE_CONNECTOR_REQUIRE_APPROVAL", "always"),
            }
        )
    elif mcp_server_url:
        continuation_kwargs.update(
            {
                "server_url": mcp_server_url,
                "server_label": example_env("LLM_CLIENT_EXAMPLE_MCP_SERVER_LABEL", "Remote MCP"),
                "authorization": mcp_authorization,
                "require_approval": example_env("LLM_CLIENT_EXAMPLE_MCP_REQUIRE_APPROVAL", "never"),
            }
        )

    try:
        engine = ExecutionEngine(provider=handle.provider)
        result = await engine.submit_mcp_approval_response(
            previous_response_id=previous_response_id,
            approval_request_id=approval_request_id,
            approve=approve,
            provider_name="openai",
            model=handle.model,
            **continuation_kwargs,
        )

        print_heading("OpenAI MCP Approval Continuation")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "previous_response_id": previous_response_id,
                "approval_request_id": approval_request_id,
                "approve": approve,
                "continuation_kwargs": continuation_kwargs,
                "content": result.content,
                "usage": summarize_usage(result.usage),
                "output_items": [item.to_dict() for item in (result.output_items or [])],
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
