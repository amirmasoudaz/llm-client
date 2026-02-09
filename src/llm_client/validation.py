"""
Validation utilities for LLM Client.

Uses jsonschema for robust validation of data structures.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

import jsonschema
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

if TYPE_CHECKING:
    from .spec import RequestSpec

from .providers.types import CompletionResult, Message

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MessageValidationConfig:
    max_messages: int | None = None
    max_message_length: int | None = None


@dataclass
class ToolValidationConfig:
    require_description: bool = False


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls) -> ValidationResult:
        return cls(valid=True)

    @classmethod
    def error(cls, error: str) -> ValidationResult:
        return cls(valid=False, errors=[error])

    @classmethod
    def merge(cls, *results: ValidationResult) -> ValidationResult:
        valid = all(r.valid for r in results)
        errors = []
        warnings = []
        for r in results:
            errors.extend(r.errors)
            warnings.extend(r.warnings)
        return cls(valid=valid, errors=errors, warnings=warnings)

    def __bool__(self) -> bool:
        return self.valid

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ValidationError(f"Validation failed: {'; '.join(self.errors)}")


class ValidationError(ValueError):
    """Validation error exception with optional request correlation."""

    def __init__(self, message: str, request_id: str | None = None):
        self.request_id = request_id
        prefix = f"[{request_id}] " if request_id else ""
        super().__init__(f"{prefix}{message}")


def validate_or_raise(result: ValidationResult) -> None:
    """Helper to raise exception if result is invalid."""
    if not result.valid:
        raise ValidationError(f"Validation failed: {'; '.join(result.errors)}")


# Common schemas could be cached here
_TOOL_SCHEMA_STRICT = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
        "description": {"type": "string"},
        "parameters": {"type": "object"},
    },
    "required": ["name"],
}


def validate_message(
    message: Message | dict[str, Any], config: MessageValidationConfig | None = None
) -> ValidationResult:
    """
    Validate a single message.

    Checks role validity and content length.
    """
    config = config or MessageValidationConfig()

    # Normalization
    if isinstance(message, Message):
        role = str(message.role.value) if hasattr(message.role, "value") else str(message.role)
        content = message.content
    elif isinstance(message, dict):
        role = str(message.get("role", ""))
        content = message.get("content")
    else:
        return ValidationResult.error("Message must be a Message object or dict.")

    # Validate role
    valid_roles = ("system", "user", "assistant", "tool", "function", "developer")
    if role not in valid_roles:
        return ValidationResult.error(f"Invalid role: {role}")

    # Validate content length
    if config.max_message_length and content and isinstance(content, str):
        if len(content) > config.max_message_length:
            return ValidationResult.error(f"Message content too long ({len(content)} > {config.max_message_length})")

    return ValidationResult.ok()


def validate_messages(
    messages: Iterable[Message | dict[str, Any]], config: MessageValidationConfig | None = None
) -> ValidationResult:
    """Validate a sequence of messages."""
    config = config or MessageValidationConfig()
    msgs = list(messages)

    if config.max_messages and len(msgs) > config.max_messages:
        return ValidationResult.error(f"Too many messages ({len(msgs)} > {config.max_messages})")

    results = [validate_message(m, config) for m in msgs]
    return ValidationResult.merge(*results)


def validate_tool_definition(tool: Any, config: ToolValidationConfig | None = None) -> ValidationResult:
    """Validate tool definition (Tool object or dict)."""
    config = config or ToolValidationConfig()

    # Extract fields
    tool_data = {}
    if hasattr(tool, "to_openai_format"):
        # If it's a Tool object, use its export
        try:
            tool_dict = tool.to_openai_format()
            # tool_dict is {"type": "function", "function": {...}}
            if tool_dict.get("type") == "function" and "function" in tool_dict:
                tool_data = tool_dict["function"]
            else:
                tool_data = tool_dict
        except Exception as e:
            return ValidationResult.error(f"Failed to export tool: {e}")
    elif hasattr(tool, "name"):
        # Simple object with attributes
        tool_data = {
            "name": tool.name,
            "description": getattr(tool, "description", None),
            "parameters": getattr(tool, "parameters", {}),
        }
    elif isinstance(tool, dict):
        # OpenAI format: {"type": "function", "function": {...}} or direct dict
        if tool.get("type") == "function" and "function" in tool:
            tool_data = tool["function"]
        else:
            tool_data = tool
    else:
        return ValidationResult.error("Invalid tool definition format.")

    # Use schema validation for the structure
    schema_res = validate_against_schema(tool_data, _TOOL_SCHEMA_STRICT)
    if not schema_res.valid:
        return schema_res

    name = tool_data.get("name")
    description = tool_data.get("description")

    if config.require_description and not description:
        return ValidationResult.error(f"Tool missing description in {name} tool.")

    return ValidationResult.ok()


def validate_tool_arguments(name: str, arguments: str, schema: dict[str, Any] | None = None) -> ValidationResult:
    """
    Validate tool arguments JSON string against schema.
    """
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return ValidationResult.error(f"Invalid JSON in arguments in {name} tool.")

    if not isinstance(args, dict):
        return ValidationResult.error(f"Arguments must be a JSON object in {name} tool.")

    if schema:
        return validate_against_schema(args, schema)

    return ValidationResult.ok()


def validate_json_schema(schema: dict[str, Any]) -> ValidationResult:
    """Validate that a dict is a valid JSON schema."""
    try:
        Draft202012Validator.check_schema(schema)
        return ValidationResult.ok()
    except SchemaError as e:
        return ValidationResult.error(f"Invalid JSON schema: {e.message}")


def validate_against_schema(data: Any, schema: dict[str, Any]) -> ValidationResult:
    """Validate data against a JSON schema using jsonschema library."""
    try:
        jsonschema.validate(instance=data, schema=schema)
        return ValidationResult.ok()
    except JsonSchemaValidationError as e:
        # Format error message to be cleaner
        # e.path is a deque of keys/indices
        path = ".".join(str(p) for p in e.path)
        if path:
            msg = f"Validation failed at '{path}': {e.message}"
        else:
            msg = f"Validation error: {e.message}"

        return ValidationResult.error(msg)
    except Exception as e:
        return ValidationResult.error(f"Schema validation error: {str(e)}")


def validate_completion_response(response: CompletionResult) -> ValidationResult:
    """Validate a completion result structure."""
    if response.tool_calls:
        for tc in response.tool_calls:
            if not tc.id:
                return ValidationResult.error("Tool call missing ID")
            try:
                json.loads(tc.arguments)
            except json.JSONDecodeError:
                return ValidationResult.error("Tool call arguments invalid JSON")

    return ValidationResult.ok()


def validate_spec(spec: RequestSpec) -> None:
    """Validate RequestSpec using the above logic."""
    if not spec.messages and not spec.extra.get("input") and not spec.extra.get("prompt"):
        raise ValidationError(
            "RequestSpec must include `messages` or provider-specific `extra` inputs (e.g., `input`/`prompt`)."
        )

    if spec.messages:
        res = validate_messages(spec.messages)
        if not res.valid:
            raise ValidationError(f"Invalid messages: {res.errors}")

    if spec.temperature is not None and not (0.0 <= spec.temperature <= 2.0):
        raise ValidationError(f"Invalid temperature: {spec.temperature}")

    if spec.max_tokens is not None and spec.max_tokens <= 0:
        raise ValidationError(f"Invalid max_tokens: {spec.max_tokens}")

    if spec.tools:
        for tool in spec.tools:
            res = validate_tool_definition(tool)
            if not res.valid:
                raise ValidationError(f"Invalid tool: {res.errors}")


def validate_embedding_inputs(inputs: Any) -> None:
    if inputs is None:
        raise ValidationError("Embedding inputs cannot be empty.")

    # Accept a single string input.
    if isinstance(inputs, str):
        if not inputs.strip():
            raise ValidationError("Embedding input string cannot be empty.")
        return

    # Accept sequences/iterables of strings.
    if isinstance(inputs, (list, tuple)):
        if len(inputs) == 0:
            raise ValidationError("Embedding inputs cannot be empty.")
        for i, item in enumerate(inputs):
            if not isinstance(item, str):
                raise ValidationError(f"Embedding input at index {i} must be a string.")
            if not item.strip():
                raise ValidationError(f"Embedding input at index {i} cannot be empty.")
        return

    raise ValidationError("Embedding inputs must be a string or a list/tuple of strings.")
