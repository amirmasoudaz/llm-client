from __future__ import annotations

import pytest

from llm_client.providers.types import Message
from llm_client.structured import (
    StructuredOutputConfig,
    StructuredResponseMode,
    build_structured_response_format,
    extract_structured,
    select_structured_mode,
)
from tests.llm_client.fakes import ScriptedProvider, ok_result


class OpenAIProviderFake(ScriptedProvider):
    pass


class AnthropicProviderFake(ScriptedProvider):
    pass


def test_select_structured_mode_prefers_json_schema_for_openai() -> None:
    selection = select_structured_mode(
        provider=OpenAIProviderFake(model_name="gpt-5-mini"),
        model="gpt-5-mini",
        strict=True,
        schema={"type": "object"},
        name="response schema",
    )

    assert selection.provider_name == "openai"
    assert selection.response_mode is StructuredResponseMode.JSON_SCHEMA
    assert isinstance(selection.response_format, dict)
    assert selection.strict_applied is True


def test_select_structured_mode_uses_prompt_only_for_anthropic() -> None:
    selection = select_structured_mode(
        provider=AnthropicProviderFake(model_name="claude-4-5-sonnet"),
        model="claude-4-5-sonnet",
        strict=True,
        schema={"type": "object"},
    )

    assert selection.provider_name == "anthropic"
    assert selection.response_mode is StructuredResponseMode.PROMPT_ONLY
    assert selection.response_format is None
    assert selection.structured_outputs_supported is False


def test_select_structured_mode_can_infer_provider_from_model_without_provider_object() -> None:
    selection = select_structured_mode(
        provider=None,
        model="gpt-5-mini",
        strict=True,
        schema={"type": "object"},
    )

    assert selection.provider_name == "openai"
    assert selection.response_mode is StructuredResponseMode.JSON_SCHEMA


def test_build_structured_response_format_returns_none_for_prompt_only_known_provider() -> None:
    response_format = build_structured_response_format(
        {"type": "object"},
        provider=AnthropicProviderFake(model_name="claude-4-5-sonnet"),
        model="claude-4-5-sonnet",
        strict=True,
    )

    assert response_format is None


@pytest.mark.asyncio
async def test_extract_structured_skips_response_format_for_prompt_only_known_provider() -> None:
    provider = AnthropicProviderFake(
        complete_script=[ok_result('{"name":"Ada"}', model="claude-4-5-sonnet")],
        model_name="claude-4-5-sonnet",
    )

    result = await extract_structured(
        provider,
        [Message.user("extract a name")],
        StructuredOutputConfig(
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            max_repair_attempts=0,
        ),
        model="claude-4-5-sonnet",
    )

    assert result.valid is True
    assert "response_format" not in provider.complete_calls[0]["kwargs"]
