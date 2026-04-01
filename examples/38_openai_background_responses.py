from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, fail_or_skip, print_heading, print_json, summarize_usage

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec


async def main() -> None:
    handle = build_live_provider(use_responses_api=True)
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the OpenAI background Responses example.")

    try:
        engine = ExecutionEngine(provider=handle.provider)
        queued = await engine.complete(
            RequestSpec(
                provider="openai",
                model=handle.model,
                messages=[
                    Message.user(
                        "Write five concise bullets about why background orchestration matters for long-running LLM work."
                    )
                ],
                extra={"background": True, "store": True},
            )
        )
        response_id = str(getattr(queued.raw_response, "id", "") or "")
        if not response_id:
            fail_or_skip("The provider did not return a stored response id for the background workflow.")

        state = await engine.wait_background_response(
            response_id,
            provider_name="openai",
            model=handle.model,
            poll_interval=0.5,
            timeout=60.0,
        )
        deleted = await engine.delete_response(
            response_id,
            provider_name="openai",
            model=handle.model,
        )

        print_heading("OpenAI Background Responses")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "queued_status": getattr(queued.raw_response, "status", None),
                "response_id": response_id,
                "final_lifecycle_status": state.lifecycle_status,
                "final_content": state.completion.content if state.completion else None,
                "final_usage": summarize_usage(state.completion.usage if state.completion else None),
                "deleted": deleted.to_dict(),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
