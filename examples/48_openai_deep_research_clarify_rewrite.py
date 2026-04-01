from __future__ import annotations

import asyncio

from cookbook_support import (
    build_provider_handle,
    close_provider,
    example_env,
    print_heading,
    print_json,
)

from llm_client.engine import ExecutionEngine


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_DEEP_RESEARCH_MODEL", "o4-mini-deep-research") or "o4-mini-deep-research"
    prompt = example_env(
        "LLM_CLIENT_EXAMPLE_DEEP_RESEARCH_PROMPT",
        (
            "Research the current enterprise tradeoffs between vector-store retrieval, "
            "remote MCP connectors, and direct function tools for internal knowledge workflows."
        ),
    ) or ""
    handle = build_provider_handle("openai", model_name, use_responses_api=True)

    try:
        engine = ExecutionEngine(provider=handle.provider)
        clarifications = await engine.clarify_deep_research_task(
            prompt,
            provider_name="openai",
            model=handle.model,
        )
        rewritten = await engine.rewrite_deep_research_prompt(
            prompt,
            provider_name="openai",
            model=handle.model,
            clarifications=clarifications.content,
        )
        queued = await engine.start_deep_research(
            rewritten.content or prompt,
            provider_name="openai",
            model=handle.model,
            web_search=True,
            rewrite_prompt=False,
            max_tool_calls=8,
        )

        print_heading("OpenAI Deep Research Clarify + Rewrite")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "original_prompt": prompt,
                "clarifications": clarifications.content,
                "rewritten_prompt": rewritten.content,
                "queued_content": queued.content,
                "queued_response_id": str(getattr(queued.raw_response, "id", "") or "") or None,
                "queued_lifecycle_status": str(getattr(queued.raw_response, "status", "") or "") or None,
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
