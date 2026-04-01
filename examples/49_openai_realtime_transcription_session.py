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
    model_name = (
        example_env("LLM_CLIENT_EXAMPLE_REALTIME_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
        or "gpt-4o-mini-transcribe"
    )
    handle = build_provider_handle("openai", model_name, use_responses_api=True)

    try:
        engine = ExecutionEngine(provider=handle.provider)
        session = await engine.create_realtime_transcription_session(
            provider_name="openai",
            model=handle.model,
            session={"type": "transcription", "model": handle.model},
        )
        connection = await engine.connect_realtime_transcription(
            provider_name="openai",
            model=handle.model,
        )
        try:
            first_event = None
            try:
                first_event = await asyncio.wait_for(connection.recv(), timeout=5.0)
            except TimeoutError:
                first_event = None
        finally:
            await connection.close()

        print_heading("OpenAI Realtime Transcription Session")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "session": session.to_dict(),
                "connection": connection.to_dict(),
                "first_event": first_event,
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
