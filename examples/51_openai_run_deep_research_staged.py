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
            "Research the tradeoffs between hosted web search, vector-store retrieval, "
            "and remote MCP servers for enterprise knowledge assistants."
        ),
    ) or ""
    clarifications_raw = example_env(
        "LLM_CLIENT_EXAMPLE_DEEP_RESEARCH_CLARIFICATIONS",
        "Focus on enterprise internal knowledge workflows|Include operational risks|Prefer official sources",
    ) or ""
    clarifications = [item.strip() for item in clarifications_raw.split("|") if item.strip()]
    wait_for_completion = example_env("LLM_CLIENT_EXAMPLE_DEEP_RESEARCH_WAIT", "0") == "1"

    handle = build_provider_handle("openai", model_name, use_responses_api=True)

    try:
        engine = ExecutionEngine(provider=handle.provider)
        result = await engine.run_deep_research(
            prompt,
            provider_name="openai",
            model=handle.model,
            clarify_first=True,
            clarifications=clarifications,
            rewrite_prompt=True,
            wait_for_completion=wait_for_completion,
            web_search=True,
            max_tool_calls=8,
        )

        print_heading("OpenAI Staged Deep Research")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "result": result.to_dict(),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
