from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, print_heading, summarize_usage

from llm_client.providers.types import Message, StreamEventType


async def main() -> None:
    handle = build_live_provider()
    try:
        print_heading("Streaming")
        async for event in handle.provider.stream(
            [
                Message.system("You are concise and technical. You don't think too much."),
                Message.user("Explain token streaming for an API user in two sentences.")
            ]
        ):
            if event.type is StreamEventType.TOKEN:
                print(event.data, end="", flush=True)
            elif event.type is StreamEventType.DONE:
                print(f"\nprovider={handle.name} model={handle.model}")
                print(f"final={event.data.content}")
                print(f"usage={summarize_usage(event.data.usage)}")
            # else:
            #     print(f"\nevent={event}\n")
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
