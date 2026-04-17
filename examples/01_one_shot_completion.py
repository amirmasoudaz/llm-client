from __future__ import annotations

import asyncio
import os

from llm_client import Message, OpenAIProvider, load_env

load_env()


async def main() -> None:
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    provider = OpenAIProvider(model=model_name)
    try:
        result = await provider.complete(
            messages=[
                Message.system("You are concise and answer in exactly one sentence."),
                Message.user("Introduce the llm_client cookbook and mention that this is a live provider call."),
            ],
            max_tokens=100,
            temperature=0.1,
            reasoning_effort="minimal"
        )

        print("\n=== One-Shot Completion ===\n")
        print(result.content)
        print("\n=== Usage ===")
        if result.usage is not None:
            usage = {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
                "total_cost": result.usage.total_cost,
            }
            print(f"provider=openai model={model_name} status={result.status} usage={usage}")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
