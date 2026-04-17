from __future__ import annotations

import asyncio
import os

from llm_client import Message, OpenAIProvider, StreamEventType, load_env

load_env()


async def main() -> None:
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    provider = OpenAIProvider(model=model_name)
    try:
        print("\n=== Streaming ===\n")
        async for event in provider.stream(
            messages=[
                Message.system("You are creative and poetic."),
                Message.user("Explain token streaming for an API user in a poetic way with 100 tokens or less."),
            ],
            max_tokens=120,
            temperature=0.1,
            reasoning_effort="minimal"
        ):
            if event.type is StreamEventType.TOKEN:
                print(event.data, end="", flush=True)
            elif event.type is StreamEventType.DONE:
                usage = (
                    {
                        "input_tokens": event.data.usage.input_tokens,
                        "output_tokens": event.data.usage.output_tokens,
                        "total_tokens": event.data.usage.total_tokens,
                        "total_cost": event.data.usage.total_cost,
                    }
                    if event.data.usage is not None
                    else {}
                )
                print(f"\n\nprovider=openai model={model_name} usage={usage}")
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
