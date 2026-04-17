from __future__ import annotations

import json
import os
from typing import Any

from llm_client import Conversation, Message, ModelProfile, OpenAIProvider, load_env
from llm_client.summarization import LLMSummarizer, LLMSummarizerConfig
from llm_client.sync import get_messages_sync, run_async_sync, summarize_sync

load_env()


def _conversation_for_sync_showcase(summarizer: LLMSummarizer) -> Conversation:
    conversation = Conversation(
        system_message=(
            "You are supporting a release-operations lead. Preserve operational facts, "
            "explicit blockers, deadlines, rollback notes, and ownership handoffs."
        ),
        max_tokens=420,
        reserve_tokens=220,
        truncation_strategy="summarize",
        summarizer=summarizer,
    )
    conversation.add_user(
        "We are preparing the 2026.03 release handoff for platform-foundations. "
        "I need a crisp operator-ready digest."
    )
    conversation.add_assistant(
        "List the current readiness state, open blockers, deployment window, and rollback posture."
    )
    conversation.add_user(
        "Readiness is high, but extraction cleanup is still open and one consumer migration "
        "pass is unfinished."
    )
    conversation.add_assistant("What is the deployment window and who owns the release call?")
    conversation.add_user(
        "Deployment window is Thursday 22:00 to 23:30 ET. Maya runs the call and Omar owns "
        "the migration checklist."
    )
    conversation.add_assistant("What evidence do we have from stakeholders and dashboards?")
    conversation.add_user(
        "Leadership is comfortable shipping if dashboards are validated and consumer migration "
        "notes are confirmed."
    )
    conversation.add_assistant("Any rollback or operational caveats that the on-call lead needs?")
    conversation.add_user(
        "Rollback runbook was updated yesterday, but the team did not rehearse it this week. "
        "There is also concern about audit-log fanout on the export job path."
    )
    conversation.add_assistant("Call out concrete next actions and exact owners.")
    conversation.add_user(
        "Next actions: close extraction cleanup, finish the final consumer migration pass, "
        "validate dashboards, and get explicit sign-off on migration notes."
    )
    conversation.add_assistant("Who owns each action and what is the escalation condition?")
    conversation.add_user(
        "Data platform owns extraction cleanup. Omar owns migration. Priya owns dashboard "
        "validation. Maya escalates if sign-off is still missing by Thursday 18:00 ET."
    )
    conversation.add_assistant("Any customer or executive sensitivity attached to the release?")
    conversation.add_user(
        "Finance is waiting on month-end exports and leadership wants a no-surprises update "
        "before the deployment window opens."
    )
    return conversation


def _resolve_model_profile(model_key: str) -> type[ModelProfile]:
    return ModelProfile.get(model_key)


def _messages_preview(messages: list[Message]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            content_text = str(content)
        else:
            content_text = content
        preview.append(
            {
                "role": message.role.value,
                "content_excerpt": (content_text or "")[:220],
            }
        )
    return preview


async def _capture_guardrail_errors(
    conversation: Conversation,
    summarizer: LLMSummarizer,
    model_profile: type[ModelProfile],
    prepared_messages: list[Message],
) -> dict[str, str]:
    errors: dict[str, str] = {}
    try:
        get_messages_sync(conversation, model_profile)
    except RuntimeError as exc:
        errors["get_messages_sync"] = str(exc)
    try:
        summarize_sync(summarizer, prepared_messages, max_tokens=120)
    except RuntimeError as exc:
        errors["summarize_sync"] = str(exc)
    return errors


def main() -> None:
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    provider_name = "openai"
    provider = OpenAIProvider(model=model_name)
    summarizer = LLMSummarizer(
        provider=provider,
        config=LLMSummarizerConfig(
            max_summary_tokens=160,
            temperature=0.1,
        ),
    )
    conversation = _conversation_for_sync_showcase(summarizer)
    model_profile = _resolve_model_profile(model_name)

    try:
        raw_token_estimate = conversation.count_tokens(model_profile)
        prepared_messages = get_messages_sync(conversation, model_profile)
        prepared_without_system = get_messages_sync(
            conversation,
            model_profile,
            include_system=False,
        )
        operator_digest = summarize_sync(summarizer, prepared_messages, max_tokens=120)
        summary_message = next(
            (
                message
                for message in prepared_messages
                if message.role.value == "system"
                and isinstance(message.content, str)
                and message.content.startswith("[Earlier context]:")
            ),
            None,
        )
        guardrail_errors = run_async_sync(
            _capture_guardrail_errors(
                conversation,
                summarizer,
                model_profile,
                prepared_messages,
            )
        )

        print("\n=== Sync Wrapper Showcase ===\n")
        print(
            json.dumps(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "script_shape": "plain synchronous CLI flow with live summarization",
                    "conversation_message_count": len(conversation),
                    "raw_token_estimate": raw_token_estimate,
                    "prepared_message_count": len(prepared_messages),
                    "prepared_without_system_count": len(prepared_without_system),
                    "summary_injected": summary_message is not None,
                    "summary_message_excerpt": (
                        (summary_message.content or "")[:240] if summary_message else None
                    ),
                    "prepared_messages": _messages_preview(prepared_messages),
                    "operator_digest": operator_digest,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Async Guardrails ===\n")
        print(
            json.dumps(
                {
                    "get_messages_sync_error": guardrail_errors.get("get_messages_sync"),
                    "summarize_sync_error": guardrail_errors.get("summarize_sync"),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        run_async_sync(provider.close())


if __name__ == "__main__":
    main()
