"""Tests for structured output validation and repair."""

from __future__ import annotations

import json
import pytest

from llm_client.structured import (
    StructuredOutputConfig,
    StructuredResult,
    extract_structured,
    validate_and_parse,
    _extract_json,
)
from llm_client.providers.types import Message


class TestStructuredResult:
    """Test StructuredResult behavior."""

    def test_valid_result(self):
        """Valid result should have ok=True."""
        result = StructuredResult(
            data={"name": "test"},
            raw_content='{"name": "test"}',
            valid=True,
        )
        assert result.ok
        assert result.data == {"name": "test"}

    def test_invalid_result(self):
        """Invalid result should have ok=False."""
        result = StructuredResult(
            data=None,
            raw_content="invalid",
            valid=False,
            validation_errors=["Parse error"],
        )
        assert not result.ok
        assert result.data is None

    def test_raise_for_invalid_on_valid(self):
        """raise_for_invalid should return data when valid."""
        result = StructuredResult(
            data={"test": 1},
            raw_content="{}",
            valid=True,
        )
        assert result.raise_for_invalid() == {"test": 1}

    def test_raise_for_invalid_on_invalid(self):
        """raise_for_invalid should raise when invalid."""
        result = StructuredResult(
            data=None,
            raw_content="",
            valid=False,
            validation_errors=["Missing required field"],
        )
        with pytest.raises(ValueError, match="Missing required field"):
            result.raise_for_invalid()


class TestExtractJson:
    """Test JSON extraction from various formats."""

    def test_plain_json(self):
        """Should handle plain JSON."""
        assert _extract_json('{"key": "value"}') == '{"key": "value"}'

    def test_json_code_block(self):
        """Should extract from ```json blocks."""
        content = '```json\n{"key": "value"}\n```'
        assert _extract_json(content) == '{"key": "value"}'

    def test_plain_code_block(self):
        """Should extract from plain ``` blocks."""
        content = '```\n{"key": "value"}\n```'
        assert _extract_json(content) == '{"key": "value"}'

    def test_whitespace(self):
        """Should handle whitespace."""
        assert _extract_json('  {"key": "value"}  ') == '{"key": "value"}'


class TestValidateAndParse:
    """Test validate_and_parse function."""

    @pytest.mark.asyncio
    async def test_valid_json(self):
        """Should validate valid JSON against schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = await validate_and_parse('{"name": "test"}', schema)
        assert result.valid
        assert result.data == {"name": "test"}

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Should handle invalid JSON."""
        schema = {"type": "object"}
        result = await validate_and_parse("not json", schema)
        assert not result.valid
        assert "parse error" in result.validation_errors[0].lower()

    @pytest.mark.asyncio
    async def test_schema_mismatch(self):
        """Should detect schema mismatches."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        result = await validate_and_parse('{"other": 123}', schema)
        assert not result.valid
        assert len(result.validation_errors) > 0


class TestExtractStructured:
    """Test extract_structured with mock provider."""

    @pytest.mark.asyncio
    async def test_valid_first_try(self, mock_provider, mock_completion_result):
        """Should succeed on first valid response."""
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }
        
        responses = [mock_completion_result(content='{"result": "success"}')]
        provider = mock_provider(responses=responses)
        config = StructuredOutputConfig(schema=schema)
        
        result = await extract_structured(
            provider,
            [Message.user("test")],
            config,
        )
        
        assert result.valid
        assert result.data == {"result": "success"}
        assert result.repair_attempts == 0

    @pytest.mark.asyncio
    async def test_repair_on_invalid_json(self, mock_provider):
        """Should attempt repair on invalid JSON."""
        schema = {"type": "object"}
        
        # First response is invalid, second is valid
        call_count = 0
        original_complete = None
        
        async def mock_complete(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from llm_client.providers.types import CompletionResult
                return CompletionResult(content="not json")
            from llm_client.providers.types import CompletionResult
            return CompletionResult(content='{"fixed": true}')
        
        provider = mock_provider()
        provider.complete = mock_complete
        
        config = StructuredOutputConfig(schema=schema, max_repair_attempts=2)
        
        result = await extract_structured(
            provider,
            [Message.user("test")],
            config,
        )
        
        assert result.valid
        assert result.repair_attempts == 1


class TestStructuredOutputConfig:
    """Test StructuredOutputConfig defaults."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = StructuredOutputConfig(schema={"type": "object"})
        assert config.max_repair_attempts == 2
        assert config.repair_temperature == 0.0
        assert config.strict_mode is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = StructuredOutputConfig(
            schema={"type": "string"},
            max_repair_attempts=5,
            repair_temperature=0.5,
        )
        assert config.max_repair_attempts == 5
        assert config.repair_temperature == 0.5


class TestProviderCompleteStructured:
    """Test BaseProvider.complete_structured integration."""

    @pytest.mark.asyncio
    async def test_complete_structured_via_extract(self, mock_provider, mock_completion_result):
        """Should work via extract_structured with mock provider."""
        schema = {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        }
        
        responses = [mock_completion_result(content='{"value": 42}')]
        provider = mock_provider(responses=responses)
        
        # Test via extract_structured (which is what complete_structured calls)
        config = StructuredOutputConfig(schema=schema)
        result = await extract_structured(
            provider,
            [Message.user("give me a number")],
            config,
        )
        
        assert result.valid
        assert result.data == {"value": 42}
