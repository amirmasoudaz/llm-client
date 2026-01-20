"""
Tests for the validation module.
"""
import pytest

from llm_client.validation import (
    ValidationResult,
    MessageValidationConfig,
    ToolValidationConfig,
    validate_message,
    validate_messages,
    validate_tool_definition,
    validate_tool_arguments,
    validate_json_schema,
    validate_against_schema,
    validate_completion_response,
)
from llm_client.providers.types import Message, Role, ToolCall, CompletionResult, Usage


class TestValidationResult:
    """Test ValidationResult class."""
    
    def test_ok_result(self):
        """Test successful result."""
        result = ValidationResult.ok()
        
        assert result.valid
        assert bool(result)
        assert len(result.errors) == 0
    
    def test_error_result(self):
        """Test error result."""
        result = ValidationResult.error("Something wrong")
        
        assert not result.valid
        assert not bool(result)
        assert "Something wrong" in result.errors
    
    def test_merge_results(self):
        """Test merging results."""
        r1 = ValidationResult(valid=True, warnings=["warning1"])
        r2 = ValidationResult.error("error1")
        r3 = ValidationResult.ok()
        
        merged = ValidationResult.merge(r1, r2, r3)
        
        assert not merged.valid
        assert "error1" in merged.errors
        assert "warning1" in merged.warnings
    
    def test_raise_if_invalid(self):
        """Test raising on invalid."""
        result = ValidationResult.ok()
        result.raise_if_invalid()  # Should not raise
        
        result = ValidationResult.error("Test error")
        with pytest.raises(Exception, match="Validation failed"):
            result.raise_if_invalid()


class TestMessageValidation:
    """Test message validation."""
    
    def test_valid_user_message(self):
        """Test validating a user message."""
        msg = Message.user("Hello, world!")
        result = validate_message(msg)
        
        assert result.valid
    
    def test_valid_assistant_message(self):
        """Test validating an assistant message."""
        msg = Message.assistant("Hi there!")
        result = validate_message(msg)
        
        assert result.valid
    
    def test_valid_message_dict(self):
        """Test validating message as dict."""
        msg = {"role": "user", "content": "Hello"}
        result = validate_message(msg)
        
        assert result.valid
    
    def test_invalid_role(self):
        """Test detecting invalid role."""
        msg = {"role": "invalid_role", "content": "Hello"}
        result = validate_message(msg)
        
        assert not result.valid
        assert any("role" in e.lower() for e in result.errors)
    
    def test_missing_role(self):
        """Test detecting missing role."""
        msg = {"content": "Hello"}
        result = validate_message(msg)
        
        assert not result.valid
        assert any("role" in e.lower() for e in result.errors)
    
    def test_message_too_long(self):
        """Test detecting too-long messages."""
        config = MessageValidationConfig(max_message_length=10)
        msg = {"role": "user", "content": "A" * 100}
        
        result = validate_message(msg, config)
        
        assert not result.valid
        assert any("too long" in e.lower() for e in result.errors)
    
    def test_validate_messages_sequence(self):
        """Test validating message sequence."""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        
        result = validate_messages(messages)
        
        assert result.valid
    
    def test_too_many_messages(self):
        """Test detecting too many messages."""
        config = MessageValidationConfig(max_messages=3)
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(5)]
        
        result = validate_messages(messages, config)
        
        assert not result.valid
        assert any("too many" in e.lower() for e in result.errors)


class TestToolValidation:
    """Test tool validation."""
    
    def test_valid_tool(self):
        """Test validating a valid tool."""
        from llm_client import Tool
        from unittest.mock import AsyncMock
        
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"],
            },
            handler=AsyncMock(),
        )
        
        result = validate_tool_definition(tool)
        
        assert result.valid
    
    def test_valid_tool_dict(self):
        """Test validating tool as OpenAI format dict."""
        tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                },
            },
        }
        
        result = validate_tool_definition(tool)
        
        assert result.valid
    
    def test_missing_name(self):
        """Test detecting missing name."""
        tool = {"description": "A tool"}
        result = validate_tool_definition(tool)
        
        assert not result.valid
        assert any("name" in e.lower() for e in result.errors)
    
    def test_invalid_name_pattern(self):
        """Test detecting invalid name."""
        tool = {"name": "123-invalid!", "description": "Bad name"}
        result = validate_tool_definition(tool)
        
        assert not result.valid
        assert any("pattern" in e.lower() for e in result.errors)
    
    def test_missing_description(self):
        """Test detecting missing description."""
        config = ToolValidationConfig(require_description=True)
        tool = {"name": "my_tool", "parameters": {}}
        
        result = validate_tool_definition(tool, config)
        
        assert not result.valid
        assert any("description" in e.lower() for e in result.errors)


class TestToolArgumentValidation:
    """Test tool argument validation."""
    
    def test_valid_arguments(self):
        """Test valid JSON arguments."""
        result = validate_tool_arguments(
            "my_tool",
            '{"x": 1, "y": 2}',
        )
        
        assert result.valid
    
    def test_invalid_json(self):
        """Test invalid JSON detection."""
        result = validate_tool_arguments(
            "my_tool",
            'not valid json',
        )
        
        assert not result.valid
        assert any("invalid json" in e.lower() for e in result.errors)
    
    def test_non_object_arguments(self):
        """Test non-object arguments."""
        result = validate_tool_arguments(
            "my_tool",
            '"just a string"',
        )
        
        assert not result.valid
        assert any("object" in e.lower() for e in result.errors)
    
    def test_validate_against_schema(self):
        """Test validation against schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        
        result = validate_tool_arguments(
            "create_user",
            '{"age": 25}',  # Missing required 'name'
            schema,
        )
        
        assert not result.valid
        assert any("name" in e.lower() for e in result.errors)


class TestSchemaValidation:
    """Test JSON schema validation."""
    
    def test_valid_schema(self):
        """Test valid schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
            },
            "required": ["name"],
        }
        
        result = validate_json_schema(schema)
        
        assert result.valid
    
    def test_invalid_type(self):
        """Test invalid type."""
        schema = {"type": "banana"}
        result = validate_json_schema(schema)
        
        assert not result.valid
    
    def test_required_not_in_properties(self):
        """Test required property not defined."""
        schema = {
            "type": "object",
            "properties": {},
            "required": ["missing"],
        }
        
        result = validate_json_schema(schema)
        
        assert not result.valid
    
    def test_validate_data_against_schema(self):
        """Test validating data."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "minimum": 0},
            },
            "required": ["name"],
        }
        
        # Valid data
        result = validate_against_schema({"name": "test", "count": 5}, schema)
        assert result.valid
        
        # Missing required
        result = validate_against_schema({"count": 5}, schema)
        assert not result.valid
        
        # Wrong type
        result = validate_against_schema({"name": 123}, schema)
        assert not result.valid
        
        # Below minimum
        result = validate_against_schema({"name": "test", "count": -1}, schema)
        assert not result.valid


class TestResponseValidation:
    """Test completion response validation."""
    
    def test_valid_response(self):
        """Test validating good response."""
        response = CompletionResult(
            content="Hello!",
            status=200,
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        
        result = validate_completion_response(response)
        
        assert result.valid
    
    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        response = CompletionResult(
            content=None,
            status=200,
            tool_calls=[
                ToolCall(id="call_1", name="test", arguments='{"x": 1}'),
            ],
        )
        
        result = validate_completion_response(response)
        
        assert result.valid
    
    def test_tool_call_missing_id(self):
        """Test detecting tool call without id."""
        response = CompletionResult(
            content=None,
            status=200,
            tool_calls=[
                ToolCall(id="", name="test", arguments='{}'),
            ],
        )
        
        result = validate_completion_response(response)
        
        assert not result.valid
    
    def test_tool_call_invalid_json(self):
        """Test detecting invalid tool call arguments."""
        response = CompletionResult(
            content=None,
            status=200,
            tool_calls=[
                ToolCall(id="call_1", name="test", arguments='not json'),
            ],
        )
        
        result = validate_completion_response(response)
        
        assert not result.valid
