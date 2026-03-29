from __future__ import annotations

import pytest

from llm_client.providers.types import CompletionResult, Message, Usage
from llm_client.structured import StructuredOutputConfig, extract_structured, validate_and_parse


class _FakeProvider:
    model_name = "gpt-5-mini"
    model = type("Model", (), {"key": "gpt-5-mini"})()

    def __init__(self, *results: CompletionResult) -> None:
        self._results = list(results)

    async def complete(self, *args, **kwargs):
        if not self._results:
            raise AssertionError("unexpected provider.complete call")
        return self._results.pop(0)


@pytest.mark.asyncio
async def test_structured_extract_records_repair_attempt_diagnostics() -> None:
    provider = _FakeProvider(
        CompletionResult(content='{"wrong": 1}', usage=Usage(total_tokens=5), model="gpt-5-mini", status=200),
        CompletionResult(content='{"name": "Ada"}', usage=Usage(total_tokens=6), model="gpt-5-mini", status=200),
    )

    result = await extract_structured(
        provider,
        [Message.user("extract a name")],
        StructuredOutputConfig(
            schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            max_repair_attempts=1,
        ),
    )

    assert result.valid is True
    assert result.diagnostics is not None
    assert len(result.diagnostics.attempts) == 2
    assert result.diagnostics.successful_attempt == 1
    assert result.diagnostics.attempts[0].valid is False
    assert result.diagnostics.attempts[1].valid is True
    assert result.diagnostics.to_dict()["successful_attempt"] == 1


@pytest.mark.asyncio
async def test_validate_and_parse_returns_diagnostics_for_malformed_json() -> None:
    result = await validate_and_parse('{"name": ', {"type": "object"})

    assert result.valid is False
    assert result.validation_errors[0].startswith("JSON parse error:")
    assert result.diagnostics is not None
    assert result.diagnostics.attempts[0].parsed is False
    assert result.diagnostics.to_dict()["attempts"][0]["parsed"] is False


@pytest.mark.asyncio
async def test_validate_and_parse_returns_diagnostics_for_schema_mismatch() -> None:
    result = await validate_and_parse(
        '{"name": 1}',
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )

    assert result.valid is False
    assert any("Validation" in error or "string" in error for error in result.validation_errors)
    assert result.diagnostics is not None
    assert result.diagnostics.attempts[0].parsed is True
    assert result.diagnostics.attempts[0].valid is False


@pytest.mark.asyncio
async def test_validate_and_parse_extracts_json_wrapped_in_prose() -> None:
    result = await validate_and_parse(
        'Here is the routing result:\n{"name":"Ada"}\nThanks.',
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )

    assert result.valid is True
    assert result.data == {"name": "Ada"}
    assert result.diagnostics is not None
    assert result.diagnostics.attempts[0].parsed is True


@pytest.mark.asyncio
async def test_validate_and_parse_returns_diagnostics_for_missing_required_fields() -> None:
    result = await validate_and_parse(
        "{}",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )

    assert result.valid is False
    assert any("required property" in error.lower() or "name" in error.lower() for error in result.validation_errors)
    assert result.diagnostics is not None
    assert result.diagnostics.attempts[0].parsed is True
    assert result.diagnostics.attempts[0].valid is False
