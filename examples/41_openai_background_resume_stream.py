from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, fail_or_skip, print_heading, print_json, summarize_usage

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message, StreamEventType
from llm_client.spec import RequestSpec


async def _consume_stream(
    stream,
    *,
    stop_after_first_token: bool,
    per_event_timeout: float,
) -> tuple[str, int | None, object | None, bool]:
    text = ""
    last_sequence: int | None = None
    completed = None
    timed_out = False

    while True:
        try:
            event = await asyncio.wait_for(anext(stream), timeout=per_event_timeout)
        except StopAsyncIteration:
            break
        except TimeoutError:
            timed_out = True
            break

        if event.sequence_number is not None:
            last_sequence = event.sequence_number
        if event.type == StreamEventType.TOKEN:
            text += str(event.data)
            if stop_after_first_token and last_sequence is not None:
                break
        elif event.type == StreamEventType.DONE:
            completed = event.data
            break

    return text, last_sequence, completed, timed_out


async def main() -> None:
    handle = build_live_provider(use_responses_api=True)
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the OpenAI background-resume-stream example.")

    try:
        engine = ExecutionEngine(provider=handle.provider)
        queued = await engine.complete(
            RequestSpec(
                provider="openai",
                model=handle.model,
                messages=[
                    Message.user(
                        "Write four concise bullets on how background response resumption helps systems recover after reconnects."
                    )
                ],
                extra={"background": True, "store": True},
            )
        )
        response_id = str(getattr(queued.raw_response, "id", "") or "")
        if not response_id:
            fail_or_skip("The provider did not return a stored response id for the background-resume-stream example.")

        completed = None
        first_chunk, resume_after, completed, initial_stream_timed_out = await _consume_stream(
            engine.stream_background_response(
                response_id,
                provider_name="openai",
                model=handle.model,
            ),
            stop_after_first_token=True,
            per_event_timeout=20.0,
        )

        resumed_text = ""
        resumed_sequence: int | None = None
        resumed_stream_timed_out = False
        if completed is None and resume_after is not None:
            resumed_text, resumed_sequence, completed, resumed_stream_timed_out = await _consume_stream(
                engine.stream_background_response(
                    response_id,
                    provider_name="openai",
                    model=handle.model,
                    starting_after=resume_after,
                ),
                stop_after_first_token=False,
                per_event_timeout=20.0,
            )

        wait_timed_out = False
        try:
            final_state = await engine.wait_background_response(
                response_id,
                provider_name="openai",
                model=handle.model,
                poll_interval=0.5,
                timeout=25.0,
            )
        except TimeoutError:
            wait_timed_out = True
            final_state = await engine.retrieve_background_response(
                response_id,
                provider_name="openai",
                model=handle.model,
            )
        deleted = await engine.delete_response(
            response_id,
            provider_name="openai",
            model=handle.model,
        )

        print_heading("OpenAI Background Resume Stream")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "response_id": response_id,
                "initial_stream_chunk": first_chunk or None,
                "resume_after_sequence": resume_after,
                "initial_stream_timed_out": initial_stream_timed_out,
                "resumed_stream_chunk": resumed_text or None,
                "last_resumed_sequence": resumed_sequence,
                "resumed_stream_timed_out": resumed_stream_timed_out,
                "wait_timed_out": wait_timed_out,
                "final_lifecycle_status": final_state.lifecycle_status,
                "final_content": (completed.content if completed else None) or (final_state.completion.content if final_state.completion else None),
                "final_usage": summarize_usage((completed.usage if completed else None) or (final_state.completion.usage if final_state.completion else None)),
                "deleted": deleted.to_dict(),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
