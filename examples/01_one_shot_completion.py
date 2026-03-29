from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, print_heading, summarize_usage

from llm_client.providers.types import Message


async def main() -> None:
    handle = build_live_provider()
    try:
        messages = [
            Message.system("You are concise and answer in exactly one sentence."),
            Message.user("Introduce the llm_client cookbook and mention that this is a live provider call."),
        ]
        result = await handle.provider.complete(messages)

        print_heading("One-Shot Completion")
        print(result.content)
        print(
            f"provider={handle.name} model={handle.model} "
            f"status={result.status} usage={summarize_usage(result.usage)}"
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
