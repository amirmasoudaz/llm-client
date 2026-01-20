"""
Request and Response Validation.

This module provides validation utilities for:
- Message content and structure validation
- Tool definition and argument validation  
- Provider response validation
- Schema validation for structured outputs
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from .errors import (
    ValidationError,
    InvalidMessageError,
    InvalidToolError,
    MessageTooLongError,
    TooManyMessagesError,
    InvalidSchemaError,
)
from .providers.types import Message, Role, ToolCall


# =============================================================================
# Validation Results
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.valid
    
    def raise_if_invalid(self):
        """Raise ValidationError if not valid."""
        if not self.valid:
            raise ValidationError(
                f"Validation failed: {'; '.join(self.errors)}"
            )
    
    @classmethod
    def ok(cls) -> "ValidationResult":
        return cls(valid=True)
    
    @classmethod
    def error(cls, message: str) -> "ValidationResult":
        return cls(valid=False, errors=[message])
    
    @classmethod
    def merge(cls, *results: "ValidationResult") -> "ValidationResult":
        """Merge multiple validation results."""
        errors = []
        warnings = []
        for r in results:
            errors.extend(r.errors)
            warnings.extend(r.warnings)
        return cls(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


# =============================================================================
# Message Validation
# =============================================================================

@dataclass
class MessageValidationConfig:
    """Configuration for message validation."""
    
    max_message_length: int = 100000
    max_messages: int = 100
    require_content_or_tool_calls: bool = True
    allow_empty_content: bool = True
    allowed_roles: List[Role] = field(
        default_factory=lambda: [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL]
    )


def validate_message(
    message: Union[Message, Dict[str, Any]],
    config: Optional[MessageValidationConfig] = None,
) -> ValidationResult:
    """
    Validate a single message.
    
    Args:
        message: Message to validate
        config: Validation configuration
        
    Returns:
        ValidationResult with any errors found
    """
    config = config or MessageValidationConfig()
    errors = []
    warnings = []
    
    # Convert dict to Message if needed
    if isinstance(message, dict):
        try:
            role = message.get("role")
            if role is None:
                errors.append("Message missing 'role' field")
                return ValidationResult(valid=False, errors=errors)
        except Exception as e:
            errors.append(f"Invalid message format: {e}")
            return ValidationResult(valid=False, errors=errors)
        
        content = message.get("content")
        tool_calls = message.get("tool_calls")
    else:
        role = message.role
        content = message.content
        tool_calls = message.tool_calls
    
    # Validate role
    role_value = role.value if isinstance(role, Role) else role
    valid_roles = [r.value if isinstance(r, Role) else r for r in config.allowed_roles]
    if role_value not in valid_roles:
        errors.append(f"Invalid role '{role_value}'. Allowed: {valid_roles}")
    
    # Validate content
    if content is not None:
        if isinstance(content, str):
            if len(content) > config.max_message_length:
                errors.append(
                    f"Message content too long: {len(content)} > {config.max_message_length}"
                )
        elif isinstance(content, list):
            # Multi-modal content
            for i, part in enumerate(content):
                if isinstance(part, dict):
                    if "type" not in part:
                        errors.append(f"Content part {i} missing 'type' field")
    
    # Validate content requirements
    if config.require_content_or_tool_calls:
        has_content = content is not None and (
            config.allow_empty_content or content
        )
        has_tools = tool_calls is not None and len(tool_calls) > 0
        
        if not has_content and not has_tools:
            if role_value == "assistant":
                # Assistant messages can be empty with tool calls
                if not has_tools:
                    warnings.append("Assistant message has no content or tool calls")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_messages(
    messages: Sequence[Union[Message, Dict[str, Any]]],
    config: Optional[MessageValidationConfig] = None,
) -> ValidationResult:
    """
    Validate a sequence of messages.
    
    Args:
        messages: Messages to validate
        config: Validation configuration
        
    Returns:
        ValidationResult with any errors found
    """
    config = config or MessageValidationConfig()
    results = []
    
    # Check message count
    if len(messages) > config.max_messages:
        results.append(ValidationResult.error(
            f"Too many messages: {len(messages)} > {config.max_messages}"
        ))
    
    # Validate each message
    for i, msg in enumerate(messages):
        result = validate_message(msg, config)
        if not result.valid:
            # Add context to errors
            result.errors = [f"Message {i}: {e}" for e in result.errors]
        results.append(result)
    
    return ValidationResult.merge(*results)


# =============================================================================
# Tool Validation
# =============================================================================

@dataclass
class ToolValidationConfig:
    """Configuration for tool validation."""
    
    max_tool_name_length: int = 64
    max_description_length: int = 1024
    max_parameters: int = 20
    require_description: bool = True
    name_pattern: str = r"^[a-zA-Z][a-zA-Z0-9_]*$"


def validate_tool_definition(
    tool: Any,
    config: Optional[ToolValidationConfig] = None,
) -> ValidationResult:
    """
    Validate a tool definition.
    
    Args:
        tool: Tool object or dict to validate
        config: Validation configuration
        
    Returns:
        ValidationResult with any errors found
    """
    config = config or ToolValidationConfig()
    errors = []
    warnings = []
    
    # Extract tool properties
    if hasattr(tool, "name"):
        name = tool.name
        description = getattr(tool, "description", None)
        parameters = getattr(tool, "parameters", {})
    elif isinstance(tool, dict):
        # Could be OpenAI function format
        if "function" in tool:
            func = tool["function"]
            name = func.get("name", "")
            description = func.get("description")
            parameters = func.get("parameters", {})
        else:
            name = tool.get("name", "")
            description = tool.get("description")
            parameters = tool.get("parameters", {})
    else:
        errors.append(f"Unknown tool format: {type(tool)}")
        return ValidationResult(valid=False, errors=errors)
    
    # Validate name
    if not name:
        errors.append("Tool name is required")
    elif len(name) > config.max_tool_name_length:
        errors.append(f"Tool name too long: {len(name)} > {config.max_tool_name_length}")
    elif not re.match(config.name_pattern, name):
        errors.append(f"Tool name '{name}' doesn't match pattern: {config.name_pattern}")
    
    # Validate description
    if config.require_description and not description:
        errors.append(f"Tool '{name}' missing description")
    elif description and len(description) > config.max_description_length:
        warnings.append(
            f"Tool '{name}' description is very long ({len(description)} chars)"
        )
    
    # Validate parameters schema
    if parameters:
        schema_result = validate_json_schema(parameters)
        if not schema_result.valid:
            errors.extend([f"Tool '{name}' parameters: {e}" for e in schema_result.errors])
        
        # Check parameter count
        properties = parameters.get("properties", {})
        if len(properties) > config.max_parameters:
            errors.append(
                f"Tool '{name}' has too many parameters: {len(properties)} > {config.max_parameters}"
            )
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_tool_arguments(
    tool_name: str,
    arguments: str,
    schema: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """
    Validate tool call arguments.
    
    Args:
        tool_name: Name of the tool
        arguments: JSON string of arguments
        schema: Optional JSON schema to validate against
        
    Returns:
        ValidationResult with any errors found
    """
    errors = []
    
    # Parse JSON
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as e:
        return ValidationResult.error(
            f"Tool '{tool_name}' has invalid JSON arguments: {e}"
        )
    
    if not isinstance(args, dict):
        return ValidationResult.error(
            f"Tool '{tool_name}' arguments must be an object, got {type(args).__name__}"
        )
    
    # Validate against schema if provided
    if schema:
        schema_result = validate_against_schema(args, schema)
        if not schema_result.valid:
            errors.extend([
                f"Tool '{tool_name}' argument error: {e}" 
                for e in schema_result.errors
            ])
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


# =============================================================================
# Schema Validation
# =============================================================================

def validate_json_schema(schema: Dict[str, Any]) -> ValidationResult:
    """
    Validate that a JSON schema is well-formed.
    
    Args:
        schema: JSON schema to validate
        
    Returns:
        ValidationResult with any errors found
    """
    errors = []
    
    if not isinstance(schema, dict):
        return ValidationResult.error("Schema must be an object")
    
    schema_type = schema.get("type")
    
    # Check for valid type
    valid_types = {"object", "array", "string", "number", "integer", "boolean", "null"}
    if schema_type and schema_type not in valid_types:
        errors.append(f"Invalid schema type: {schema_type}")
    
    # For object types, validate properties
    if schema_type == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            errors.append("'properties' must be an object")
        else:
            for prop_name, prop_schema in properties.items():
                if not isinstance(prop_schema, dict):
                    errors.append(f"Property '{prop_name}' must have a schema object")
                else:
                    sub_result = validate_json_schema(prop_schema)
                    errors.extend([f"Property '{prop_name}': {e}" for e in sub_result.errors])
        
        # Validate required field
        required = schema.get("required", [])
        if required and not isinstance(required, list):
            errors.append("'required' must be an array")
        elif required:
            for req in required:
                if req not in properties:
                    errors.append(f"Required property '{req}' not in properties")
    
    # For array types, validate items
    if schema_type == "array":
        items = schema.get("items")
        if items and isinstance(items, dict):
            sub_result = validate_json_schema(items)
            errors.extend([f"Array items: {e}" for e in sub_result.errors])
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_against_schema(
    data: Any,
    schema: Dict[str, Any],
    path: str = "",
) -> ValidationResult:
    """
    Validate data against a JSON schema.
    
    This is a simplified validator for common cases.
    For full JSON Schema support, use jsonschema library.
    
    Args:
        data: Data to validate
        schema: JSON schema to validate against
        path: Current path (for error messages)
        
    Returns:
        ValidationResult with any errors found
    """
    errors = []
    
    schema_type = schema.get("type")
    
    # Type checking
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    
    if schema_type:
        expected_types = type_map.get(schema_type)
        if expected_types and not isinstance(data, expected_types):
            return ValidationResult.error(
                f"{path or 'Value'}: expected {schema_type}, got {type(data).__name__}"
            )
    
    # Object validation
    if schema_type == "object" and isinstance(data, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Check required properties
        for req in required:
            if req not in data:
                errors.append(f"{path}.{req}: required property missing")
        
        # Validate each property
        for key, value in data.items():
            if key in properties:
                prop_path = f"{path}.{key}" if path else key
                sub_result = validate_against_schema(value, properties[key], prop_path)
                errors.extend(sub_result.errors)
    
    # Array validation
    if schema_type == "array" and isinstance(data, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                sub_result = validate_against_schema(item, items_schema, item_path)
                errors.extend(sub_result.errors)
        
        # Check min/max items
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items and len(data) < min_items:
            errors.append(f"{path or 'Array'}: minimum {min_items} items required")
        if max_items and len(data) > max_items:
            errors.append(f"{path or 'Array'}: maximum {max_items} items allowed")
    
    # String validation
    if schema_type == "string" and isinstance(data, str):
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")
        pattern = schema.get("pattern")
        enum = schema.get("enum")
        
        if min_length and len(data) < min_length:
            errors.append(f"{path or 'String'}: minimum length {min_length}")
        if max_length and len(data) > max_length:
            errors.append(f"{path or 'String'}: maximum length {max_length}")
        if pattern and not re.match(pattern, data):
            errors.append(f"{path or 'String'}: must match pattern {pattern}")
        if enum and data not in enum:
            errors.append(f"{path or 'String'}: must be one of {enum}")
    
    # Number validation
    if schema_type in ("number", "integer") and isinstance(data, (int, float)):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        
        if minimum is not None and data < minimum:
            errors.append(f"{path or 'Number'}: minimum value is {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path or 'Number'}: maximum value is {maximum}")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)


# =============================================================================
# Response Validation
# =============================================================================

def validate_completion_response(
    response: Any,
    expect_tool_calls: bool = False,
) -> ValidationResult:
    """
    Validate a completion response from a provider.
    
    Args:
        response: CompletionResult or similar response object
        expect_tool_calls: Whether tool calls are expected
        
    Returns:
        ValidationResult with any errors found
    """
    errors = []
    warnings = []
    
    # Check for basic response properties
    if hasattr(response, "ok"):
        if not response.ok:
            if not hasattr(response, "error") or not response.error:
                warnings.append("Response not OK but no error message provided")
    
    # Check content
    if hasattr(response, "content"):
        content = response.content
        if content is not None and not isinstance(content, str):
            errors.append(f"Response content must be string, got {type(content)}")
    
    # Check tool calls
    if hasattr(response, "tool_calls"):
        tool_calls = response.tool_calls
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                if not hasattr(tc, "id") or not tc.id:
                    errors.append(f"Tool call {i} missing id")
                if not hasattr(tc, "name") or not tc.name:
                    errors.append(f"Tool call {i} missing name")
                if hasattr(tc, "arguments"):
                    try:
                        json.loads(tc.arguments)
                    except (json.JSONDecodeError, TypeError):
                        errors.append(f"Tool call {i} has invalid arguments JSON")
        elif expect_tool_calls:
            warnings.append("Expected tool calls but none returned")
    
    # Check usage
    if hasattr(response, "usage") and response.usage is not None:
        usage = response.usage
        if hasattr(usage, "input_tokens") and usage.input_tokens < 0:
            errors.append("Usage input_tokens cannot be negative")
        if hasattr(usage, "output_tokens") and usage.output_tokens < 0:
            errors.append("Usage output_tokens cannot be negative")
    
    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


__all__ = [
    # Results
    "ValidationResult",
    # Config
    "MessageValidationConfig",
    "ToolValidationConfig",
    # Message validation
    "validate_message",
    "validate_messages",
    # Tool validation
    "validate_tool_definition",
    "validate_tool_arguments",
    # Schema validation
    "validate_json_schema",
    "validate_against_schema",
    # Response validation
    "validate_completion_response",
]
