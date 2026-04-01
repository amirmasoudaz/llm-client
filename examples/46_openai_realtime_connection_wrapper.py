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
    model_name = example_env("LLM_CLIENT_EXAMPLE_REALTIME_MODEL", "gpt-realtime") or "gpt-realtime"
    handle = build_provider_handle("openai", model_name)

    try:
        engine = ExecutionEngine(provider=handle.provider)
        secret = await engine.create_realtime_client_secret(
            provider_name="openai",
            model=handle.model,
            session={"type": "realtime", "model": handle.model},
        )
        connection = await engine.connect_realtime(
            provider_name="openai",
            model=handle.model,
        )
        try:
            await connection.update_session(
                {
                    "modalities": ["text"],
                    "instructions": "You are a concise realtime status assistant.",
                }
            )
            await connection.create_response(
                {
                    "modalities": ["text"],
                    "instructions": "Acknowledge the session update in one sentence.",
                }
            )
            first_event = None
            try:
                first_event = await asyncio.wait_for(connection.recv(), timeout=5.0)
            except TimeoutError:
                first_event = None
        finally:
            await connection.close()

        print_heading("OpenAI Realtime Connection Wrapper")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "client_secret": secret.to_dict(),
                "connection": connection.to_dict(),
                "first_event": first_event,
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
