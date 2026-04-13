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

from llm_client.engine import ExecutionEngine


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_REALTIME_MODEL", "gpt-realtime") or "gpt-realtime"
    handle = build_provider_handle("openai", model_name)

    try:
        engine = ExecutionEngine(provider=handle.provider)
        try:
            connection = await asyncio.wait_for(
                engine.connect_realtime(provider_name="openai", model=handle.model),
                timeout=10.0,
            )
        except TimeoutError:
            fail_or_skip("Timed out while connecting to the OpenAI realtime websocket.")

        try:
            await connection.update_session(
                {
                    "modalities": ["text"],
                    "instructions": "You are a concise realtime assistant for cookbook demonstrations.",
                },
                event_id="evt_session",
            )
            await connection.create_text_message(
                "Summarize why typed realtime output collectors are useful in one short sentence.",
                event_id="evt_text_turn",
            )
            await connection.create_response({"modalities": ["text"]}, event_id="evt_response")
            output = await connection.collect_response_output(timeout=8.0)

            print_heading("OpenAI Realtime Output Collection")
            print_json(
                {
                    "provider": handle.name,
                    "model": handle.model,
                    "connection": connection.to_dict(),
                    "output": output.to_dict(),
                }
            )
        finally:
            await connection.close()
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
