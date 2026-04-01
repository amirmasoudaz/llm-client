from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, fail_or_skip, print_heading, print_json, summarize_usage

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec


def _count_reasoning_items_with_encryption(provider_items: list[dict[str, object]] | None) -> int:
    if not provider_items:
        return 0
    return sum(
        1
        for item in provider_items
        if item.get("type") in {"reasoning", "reasoning_summary"}
        and isinstance(item.get("encrypted_content"), str)
        and str(item.get("encrypted_content")).strip()
    )


async def main() -> None:
    handle = build_live_provider(use_responses_api=True)
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the OpenAI prompt-cache example.")

    try:
        engine = ExecutionEngine(provider=handle.provider)
        prompt = (
            "Think carefully about prompt caching tradeoffs, then explain in four short bullets "
            "how cache keys and encrypted reasoning continuity can help repeated operational prompts."
        )
        spec = RequestSpec(
            provider="openai",
            model=handle.model,
            messages=[Message.user(prompt)],
            reasoning={"effort": "medium"},
            include=["reasoning.encrypted_content"],
            prompt_cache_key="cookbook-openai-cache-demo",
            prompt_cache_retention="24h",
        )

        first = await engine.complete(spec)
        second = await engine.complete(spec)

        print_heading("OpenAI Prompt Cache And Encrypted Reasoning")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "first_content": first.content,
                "second_content": second.content,
                "first_usage": summarize_usage(first.usage),
                "second_usage": summarize_usage(second.usage),
                "second_input_tokens_cached": getattr(second.usage, "input_tokens_cached", 0) if second.usage else 0,
                "first_reasoning_encrypted_items": _count_reasoning_items_with_encryption(first.provider_items),
                "second_reasoning_encrypted_items": _count_reasoning_items_with_encryption(second.provider_items),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
