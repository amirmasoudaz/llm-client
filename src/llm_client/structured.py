"""Structured output validation and repair.

This module provides utilities for extracting structured data from LLM
responses with automatic validation and repair loops.

Key features:
- JSON schema validation
- Automatic repair prompts for invalid output
- Configurable retry limits
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from .validation import validate_against_schema

if TYPE_CHECKING:
    from .providers.base import Provider
    from .providers.types import Message


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output handling."""
    schema: dict[str, Any]
    max_repair_attempts: int = 2
    repair_temperature: float = 0.0  # Lower temp for repairs
    strict_mode: bool = True  # Fail if repair doesn't help


@dataclass
class StructuredResult:
    """Result of structured output extraction.
    
    Attributes:
        data: Parsed and validated data, or None if failed.
        raw_content: Raw content from the LLM.
        valid: Whether the data passed schema validation.
        repair_attempts: Number of repair attempts made.
        validation_errors: List of validation errors if any.
    """
    data: dict[str, Any] | None
    raw_content: str
    valid: bool
    repair_attempts: int = 0
    validation_errors: list[str] = field(default_factory=list)
    
    @property
    def ok(self) -> bool:
        """Alias for valid. Returns True if data is valid."""
        return self.valid
    
    def raise_for_invalid(self) -> dict[str, Any]:
        """Return data or raise ValueError if invalid."""
        if not self.valid or self.data is None:
            errors = "; ".join(self.validation_errors) if self.validation_errors else "Unknown error"
            raise ValueError(f"Structured output validation failed: {errors}")
        return self.data


async def extract_structured(
    provider: Provider,
    messages: list[Message],
    config: StructuredOutputConfig,
    **kwargs,
) -> StructuredResult:
    """
    Extract structured data with validation and repair loop.
    
    Process:
    1. Ask LLM for JSON matching schema
    2. Parse and validate against schema
    3. If invalid, send repair prompt with error details
    4. Repeat up to max_repair_attempts
    
    Args:
        provider: LLM provider to use.
        messages: Input messages for the completion.
        config: Configuration for structured output.
        **kwargs: Additional arguments for provider.complete().
    
    Returns:
        StructuredResult with parsed data and validation status.
    """
    from .providers.types import Message as Msg
    
    schema = config.schema
    
    # Initial request with schema instruction
    schema_instruction = (
        f"Respond with valid JSON matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n"
        "Only output the JSON, no additional text."
    )
    
    # Build initial request
    request_messages = list(messages) + [Msg.system(schema_instruction)]
    
    for attempt in range(config.max_repair_attempts + 1):
        # Determine temperature
        temp = config.repair_temperature if attempt > 0 else kwargs.get("temperature")
        call_kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
        if temp is not None:
            call_kwargs["temperature"] = temp
        
        # Make completion request
        try:
            result = await provider.complete(
                request_messages,
                response_format="json_object",
                **call_kwargs,
            )
        except Exception as e:
            return StructuredResult(
                data=None,
                raw_content=str(e),
                valid=False,
                repair_attempts=attempt,
                validation_errors=[f"Provider error: {e}"],
            )
        
        raw = result.content or ""
        
        # Parse JSON
        try:
            # Extract JSON from potential markdown code blocks
            content = _extract_json(raw)
            data = json.loads(content)
        except json.JSONDecodeError as e:
            if attempt < config.max_repair_attempts:
                request_messages = list(messages) + [
                    Msg.assistant(raw),
                    Msg.user(f"Invalid JSON: {e}. Please respond with valid JSON only."),
                ]
                continue
            return StructuredResult(
                data=None,
                raw_content=raw,
                valid=False,
                repair_attempts=attempt,
                validation_errors=[f"JSON parse error: {e}"],
            )
        
        # Validate against schema
        validation = validate_against_schema(data, schema)
        if validation.valid:
            return StructuredResult(
                data=data,
                raw_content=raw,
                valid=True,
                repair_attempts=attempt,
            )
        
        # Repair loop
        if attempt < config.max_repair_attempts:
            error_msg = "; ".join(validation.errors)
            request_messages = list(messages) + [
                Msg.assistant(raw),
                Msg.user(
                    f"The JSON doesn't match the schema:\n{error_msg}\n\n"
                    f"Please fix and respond with valid JSON matching the schema."
                ),
            ]
            continue
        
        # Max attempts reached
        return StructuredResult(
            data=data,  # Return partial data even if invalid
            raw_content=raw,
            valid=False,
            repair_attempts=attempt,
            validation_errors=validation.errors,
        )
    
    # Should not reach here
    return StructuredResult(
        data=None,
        raw_content="",
        valid=False,
        repair_attempts=config.max_repair_attempts,
        validation_errors=["Max repair attempts reached"],
    )


def _extract_json(content: str) -> str:
    """Extract JSON from potential markdown code blocks.
    
    Handles:
    - Plain JSON
    - ```json ... ``` blocks
    - ``` ... ``` blocks
    """
    content = content.strip()
    
    # Check for markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        content = "\n".join(lines)
    
    return content.strip()


async def validate_and_parse(
    content: str,
    schema: dict[str, Any],
) -> StructuredResult:
    """Validate and parse JSON content against a schema.
    
    This is a simpler alternative when you already have content
    and just want to validate it.
    
    Args:
        content: JSON string to validate.
        schema: JSON schema to validate against.
    
    Returns:
        StructuredResult with validation status.
    """
    try:
        extracted = _extract_json(content)
        data = json.loads(extracted)
    except json.JSONDecodeError as e:
        return StructuredResult(
            data=None,
            raw_content=content,
            valid=False,
            validation_errors=[f"JSON parse error: {e}"],
        )
    
    validation = validate_against_schema(data, schema)
    return StructuredResult(
        data=data,
        raw_content=content,
        valid=validation.valid,
        validation_errors=validation.errors if not validation.valid else [],
    )


__all__ = [
    "StructuredOutputConfig",
    "StructuredResult",
    "extract_structured",
    "validate_and_parse",
]
