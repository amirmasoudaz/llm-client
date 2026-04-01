from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, fail_or_skip, print_heading, print_json

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec


async def main() -> None:
    handle = build_live_provider(use_responses_api=True)
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the OpenAI conversation-state example.")

    try:
        engine = ExecutionEngine(provider=handle.provider)
        conversation = await engine.create_conversation(
            provider_name="openai",
            model=handle.model,
            items=[Message.user("Mission update: launch readiness is blocked on one remaining rollback test.")],
            metadata={"workflow": "cookbook-conversation-state"},
        )
        await engine.create_conversation_items(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
            items=[Message.user("Add a concise launch-readiness summary with risks and next step.")],
        )
        listed_before = await engine.list_conversation_items(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
            limit=20,
            order="asc",
        )

        completion = await engine.complete(
            RequestSpec(
                provider="openai",
                model=handle.model,
                messages=[Message.user("Respond using the active conversation context in three short bullets.")],
                extra={"conversation": conversation.conversation_id, "store": True},
            )
        )

        response_id = str(getattr(completion.raw_response, "id", "") or "")
        compaction = None
        if response_id:
            compaction = await engine.compact_response_context(
                provider_name="openai",
                model=handle.model,
                messages=[Message.user("Compact the conversation context into a reusable state summary.")],
                previous_response_id=response_id,
            )

        listed_after = await engine.list_conversation_items(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
            limit=20,
            order="asc",
        )
        deleted = await engine.delete_conversation(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
        )

        print_heading("OpenAI Conversation State Workflow")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "conversation": conversation.to_dict(),
                "items_before_completion": listed_before.to_dict(),
                "completion_content": completion.content,
                "response_id": response_id,
                "compaction": compaction.to_dict() if compaction else None,
                "items_after_completion": listed_after.to_dict(),
                "deleted": deleted.to_dict(),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
