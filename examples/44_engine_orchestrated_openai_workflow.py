from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, fail_or_skip, print_heading, print_json, summarize_usage

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec


async def main() -> None:
    handle = build_live_provider(use_responses_api=True)
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the engine-orchestrated OpenAI workflow example.")

    try:
        engine = ExecutionEngine(provider=handle.provider)
        conversation = await engine.create_conversation(
            provider_name="openai",
            model=handle.model,
            items=[Message.user("Program update: migration cutover is scheduled for tomorrow morning.")],
            metadata={"workflow": "cookbook-engine-orchestrated-openai"},
        )
        background = await engine.complete(
            RequestSpec(
                provider="openai",
                model=handle.model,
                messages=[
                    Message.user(
                        "Using the current conversation, draft a launch-readiness memo with three bullets and one explicit blocker."
                    )
                ],
                extra={"conversation": conversation.conversation_id, "background": True, "store": True},
            )
        )
        response_id = str(getattr(background.raw_response, "id", "") or "")
        if not response_id:
            fail_or_skip("The provider did not return a stored response id for the engine-orchestrated workflow example.")

        final_state = await engine.wait_background_response(
            response_id,
            provider_name="openai",
            model=handle.model,
            poll_interval=0.5,
            timeout=90.0,
        )
        await engine.create_conversation_items(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
            items=[Message.user("Add a follow-up item with the owner for the final blocker and the next checkpoint.")],
        )
        followup = await engine.complete(
            RequestSpec(
                provider="openai",
                model=handle.model,
                messages=[Message.user("Using the conversation context, return one sentence naming the blocker owner and next checkpoint.")],
                extra={"conversation": conversation.conversation_id},
            )
        )
        items_page = await engine.list_conversation_items(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
            limit=50,
            order="asc",
        )
        compaction = await engine.compact_response_context(
            provider_name="openai",
            model=handle.model,
            messages=[Message.user("Compact this workflow into portable context for a later follow-up run.")],
            previous_response_id=response_id,
        )
        deleted_response = await engine.delete_response(
            response_id,
            provider_name="openai",
            model=handle.model,
        )
        deleted_conversation = await engine.delete_conversation(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
        )

        print_heading("Engine-Orchestrated OpenAI Workflow")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "conversation": conversation.to_dict(),
                "background_response_id": response_id,
                "background_lifecycle_status": final_state.lifecycle_status,
                "background_content": final_state.completion.content if final_state.completion else None,
                "background_usage": summarize_usage(final_state.completion.usage if final_state.completion else None),
                "followup_content": followup.content,
                "conversation_item_count": len(items_page.items),
                "compaction": compaction.to_dict(),
                "deleted_response": deleted_response.to_dict(),
                "deleted_conversation": deleted_conversation.to_dict(),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
