from __future__ import annotations

import asyncio

from cookbook_support import build_provider_handle, close_provider, example_env, fail_or_skip, print_heading, print_json

from llm_client.providers.types import Message


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_OPENAI_RESPONSES_MODEL", "gpt-5") or "gpt-5"
    handle = build_provider_handle("openai", model_name, use_responses_api=True)

    try:
        first = await handle.provider.complete(
            [
                Message.user(
                    "Explain in two short bullets why OpenAI Responses `output_items` are a stable surface while `provider_items` remain low-level replay data."
                )
            ],
            reasoning={"effort": "low"},
            include=["reasoning.encrypted_content"],
            temperature=0.0,
        )
        if not first.output_items and not first.provider_items:
            fail_or_skip("The provider did not return output_items/provider_items for the normalized-output-items example.")

        print_heading("OpenAI Normalized Output Items")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "content": first.content,
                "reasoning": first.reasoning,
                "output_items": [item.to_dict() for item in (first.output_items or [])],
                "provider_items_present": bool(first.provider_items),
                "provider_items_count": len(first.provider_items or []),
                "refusal": first.refusal,
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
