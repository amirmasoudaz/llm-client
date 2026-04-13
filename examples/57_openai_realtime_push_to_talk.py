from __future__ import annotations

import asyncio
from pathlib import Path

from cookbook_support import (
    build_provider_handle,
    close_provider,
    example_env,
    fail_or_skip,
    print_heading,
    print_json,
)

from llm_client.engine import ExecutionEngine


def _chunk_bytes(data: bytes, *, size: int = 16_384) -> list[bytes]:
    return [data[index : index + size] for index in range(0, len(data), size)] or [b""]


async def main() -> None:
    model_name = example_env("LLM_CLIENT_EXAMPLE_REALTIME_MODEL", "gpt-realtime") or "gpt-realtime"
    audio_path = example_env("LLM_CLIENT_EXAMPLE_REALTIME_AUDIO_PATH")
    if not audio_path:
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_REALTIME_AUDIO_PATH to run the realtime push-to-talk example.")
    path = Path(audio_path)
    if not path.exists():
        fail_or_skip(f"Realtime audio file not found: {path}")

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

        audio_bytes = path.read_bytes()
        chunks = _chunk_bytes(audio_bytes)

        try:
            await connection.disable_vad(
                session={
                    "modalities": ["audio", "text"],
                    "instructions": "Respond briefly and acknowledge the audio turn.",
                },
                event_id="evt_disable_vad",
            )
            await connection.send_audio_turn(
                chunks,
                {"modalities": ["text"]},
                clear_input=True,
                clear_input_event_id="evt_clear_input",
                append_event_ids=[f"evt_append_{index}" for index in range(len(chunks))],
                commit_event_id="evt_commit",
                response_event_id="evt_response",
            )
            output = await connection.collect_response_output(timeout=12.0)

            print_heading("OpenAI Realtime Push To Talk")
            print_json(
                {
                    "provider": handle.name,
                    "model": handle.model,
                    "audio_path": str(path),
                    "chunk_count": len(chunks),
                    "output": output.to_dict(),
                }
            )
        finally:
            await connection.close()
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
