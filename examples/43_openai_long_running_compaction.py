from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, fail_or_skip, print_heading, print_json, summarize_usage

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec


async def main() -> None:
    handle = build_live_provider(use_responses_api=True)
    if handle.name != "openai":
        fail_or_skip("Set LLM_CLIENT_EXAMPLE_PROVIDER=openai to run the OpenAI long-running-compaction example.")

    try:
        engine = ExecutionEngine(provider=handle.provider)
        conversation = await engine.create_conversation(
            provider_name="openai",
            model=handle.model,
            items=[Message.user("Incident seed: a deployment rollback restored service but left observability gaps.")],
            metadata={"workflow": "cookbook-long-running-compaction"},
        )
        await engine.create_conversation_items(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
            items=[
                Message.user("Add note: customer impact was 14 minutes of degraded search results."),
                Message.user("Add note: follow-up actions include a rollback drill and alert threshold review."),
                Message.user("Add note: summarize open risks before the next deploy window."),
            ],
        )
        items_before = await engine.list_conversation_items(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
            limit=50,
            order="asc",
        )
        first_item_id = items_before.items[0].item_id if items_before.items else None
        retrieved_item = (
            await engine.retrieve_conversation_item(
                conversation.conversation_id,
                first_item_id,
                provider_name="openai",
                model=handle.model,
            )
            if first_item_id
            else None
        )

        completion = await engine.complete(
            RequestSpec(
                provider="openai",
                model=handle.model,
                messages=[
                    Message.user(
                        "Using the active conversation state, produce a concise incident handoff with risks, next action, and one watch metric."
                    )
                ],
                extra={"conversation": conversation.conversation_id, "store": True},
            )
        )

        response_id = str(getattr(completion.raw_response, "id", "") or "")
        compaction = await engine.compact_response_context(
            provider_name="openai",
            model=handle.model,
            messages=[Message.user("Compact the current incident thread into reusable compressed context.")],
            previous_response_id=response_id or None,
        )
        items_after = await engine.list_conversation_items(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
            limit=50,
            order="asc",
        )
        deleted = await engine.delete_conversation(
            conversation.conversation_id,
            provider_name="openai",
            model=handle.model,
        )

        print_heading("OpenAI Long-Running Compaction")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "conversation": conversation.to_dict(),
                "items_before_count": len(items_before.items),
                "retrieved_first_item": retrieved_item.to_dict() if retrieved_item else None,
                "completion_content": completion.content,
                "completion_usage": summarize_usage(completion.usage),
                "response_id": response_id or None,
                "compaction": compaction.to_dict(),
                "compaction_output_types": [item.type for item in (compaction.output_items or [])],
                "items_after_count": len(items_after.items),
                "deleted": deleted.to_dict(),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
