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

from llm_client.tools import (
    ResponsesFunctionTool,
    ResponsesToolNamespace,
    ResponsesToolSearch,
    tool,
)


@tool
async def billing_lookup_invoice(invoice_id: str) -> dict[str, str]:
    """Lookup invoice status for a customer billing workflow."""
    return {"invoice_id": invoice_id, "status": "paid"}


@tool
async def billing_quote_refund(invoice_id: str) -> dict[str, str]:
    """Estimate the refund outcome for an invoice dispute workflow."""
    return {"invoice_id": invoice_id, "recommended_action": "review_refund_policy"}


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_OPENAI_TOOLS_MODEL", "gpt-5-nano") or "gpt-5-nano"
    handle = build_provider_handle("openai", model_name, use_responses_api=True)
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the OpenAI tool-search example.")

    try:
        provider = handle.provider
        try:
            initial = await asyncio.wait_for(
                provider.complete(
                    "Search the billing tools and pick the refund-related one.",
                    model=handle.model,
                    max_tokens=32,
                    reasoning_effort="low",
                    tools=[
                        ResponsesToolSearch.hosted(),
                        ResponsesToolNamespace(
                            name="billing",
                            description="Deferred billing tools for invoices and refunds.",
                            tools=(
                                ResponsesFunctionTool.from_tool(billing_lookup_invoice, defer_loading=True),
                                ResponsesFunctionTool.from_tool(billing_quote_refund, defer_loading=True),
                            ),
                        ),
                    ],
                ),
                timeout=15.0,
            )
        except TimeoutError:
            fail_or_skip("Timed out while waiting for the OpenAI tool-search workflow to produce an initial response.")

        response_id = str(getattr(initial.raw_response, "id", "") or "")
        tool_search_calls = [item for item in (initial.output_items or []) if item.type == "tool_search_call"]
        loaded = None
        if response_id and tool_search_calls and tool_search_calls[0].call_id:
            try:
                loaded = await asyncio.wait_for(
                    provider.submit_tool_search_output(
                        previous_response_id=response_id,
                        call_id=str(tool_search_calls[0].call_id),
                        tools=[
                            ResponsesFunctionTool.from_tool(billing_quote_refund, defer_loading=True),
                        ],
                    ),
                    timeout=15.0,
                )
            except TimeoutError:
                fail_or_skip("Timed out while waiting for the OpenAI tool-search continuation response.")

        print_heading("OpenAI Tool Search And Namespaces")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "initial_content": initial.content,
                "initial_output_items": [item.to_dict() for item in (initial.output_items or [])],
                "loaded_follow_up": (
                    {
                        "content": loaded.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            }
                            for tool_call in (loaded.tool_calls or [])
                        ],
                        "output_items": [item.to_dict() for item in (loaded.output_items or [])],
                    }
                    if loaded is not None
                    else None
                ),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
