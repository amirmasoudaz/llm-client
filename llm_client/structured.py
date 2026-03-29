"""Structured output validation and repair.

This module provides utilities for extracting structured data from LLM
responses with automatic validation and repair loops.

Key features:
- JSON schema validation
- Automatic repair prompts for invalid output
- Configurable retry limits
"""

from __future__ import annotations

import copy
import inspect
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

from .content import content_blocks_to_text, ensure_completion_result
from .errors import normalize_structured_failure
from .request_builders import build_content_request_envelope, infer_model_name
from .retry_policy import is_retryable_status
from .tools.runtime import StructuredToolLoopError, StructuredToolRuntime, complete_with_tools
from .validation import validate_against_schema

if TYPE_CHECKING:
    from .engine import ExecutionEngine
    from .providers.base import Provider
    from .providers.types import Message
    from .spec import RequestContext


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
    response_kind: str = "text"
    diagnostics: StructuredDiagnostics | None = None
    usage: Any = None
    
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


@dataclass
class StructuredCompletionLoopResult:
    completion: Any | None
    completion_messages: list[dict[str, Any]]
    content: Any = None
    raw_content: Any = None
    validation_errors: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    tool_error: StructuredToolLoopError | None = None

    @property
    def ok(self) -> bool:
        return self.completion is not None and self.tool_error is None and not self.validation_errors


@dataclass(frozen=True)
class StructuredExecutionFailure:
    code: str
    message: str
    category: str
    retryable: bool = False
    retry_after_ms: int | None = None
    details: dict[str, Any] | None = None

    def to_normalized_failure(self, *, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        return normalize_structured_failure(
            self,
            provider=provider,
            model=model,
            operation="structured",
        ).to_dict()


@dataclass(frozen=True)
class StructuredExecutionOutcome:
    envelope: StructuredResultEnvelope | None
    failure: StructuredExecutionFailure | None
    completion: Any | None
    raw_content: Any = None
    validation_errors: list[str] = field(default_factory=list)
    repair_attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.envelope is not None and self.failure is None


@dataclass(frozen=True)
class StructuredEnvelopeError:
    code: str
    message: str
    category: str
    retryable: bool = False
    retry_after_ms: int | None = None
    details: dict[str, Any] | None = None

    def to_normalized_failure(self, *, provider: str | None = None, model: str | None = None) -> dict[str, Any]:
        return normalize_structured_failure(
            self,
            provider=provider,
            model=model,
            operation="structured",
        ).to_dict()


@dataclass(frozen=True)
class StructuredResultEnvelope:
    status: str
    result: dict[str, Any] | None
    artifacts: list[Any]
    nondeterminism: dict[str, Any]
    error: StructuredEnvelopeError | None = None


@dataclass
class StructuredAttemptTrace:
    attempt: int
    response_kind: str
    raw_content: str
    extracted_json: str | None = None
    parsed: bool = False
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "response_kind": self.response_kind,
            "raw_content": self.raw_content,
            "extracted_json": self.extracted_json,
            "parsed": self.parsed,
            "valid": self.valid,
            "validation_errors": list(self.validation_errors),
        }


@dataclass
class StructuredDiagnostics:
    attempts: list[StructuredAttemptTrace] = field(default_factory=list)
    final_response_kind: str = "text"
    successful_attempt: int | None = None

    def record(self, trace: StructuredAttemptTrace) -> None:
        self.attempts.append(trace)
        self.final_response_kind = trace.response_kind
        if trace.valid:
            self.successful_attempt = trace.attempt

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "final_response_kind": self.final_response_kind,
            "successful_attempt": self.successful_attempt,
        }


class StructuredStreamEventType(str, Enum):
    CONTENT_DELTA = "content_delta"
    RAW_EVENT = "raw_event"
    RESULT = "result"
    ERROR = "error"
    DONE = "done"


class StructuredExecutionMode(str, Enum):
    AUTO = "auto"
    COMPLETE = "complete"
    STREAM = "stream"
    VALIDATE = "validate"


class StructuredResponseMode(str, Enum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    PROMPT_ONLY = "prompt_only"


@dataclass
class StructuredStreamEvent:
    type: StructuredStreamEventType
    data: Any
    raw_event: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        raw = self.raw_event.to_dict() if hasattr(self.raw_event, "to_dict") else self.raw_event
        payload = self.data.to_dict() if hasattr(self.data, "to_dict") else self.data
        return {
            "type": self.type.value,
            "data": payload,
            "raw_event": raw,
        }


@dataclass(frozen=True)
class StructuredModeSelection:
    provider_name: str
    model_name: str | None
    response_mode: StructuredResponseMode
    response_format: str | dict[str, Any] | None
    structured_outputs_supported: bool
    strict_requested: bool
    strict_applied: bool
    streaming_requested: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "response_mode": self.response_mode.value,
            "response_format": self.response_format,
            "structured_outputs_supported": self.structured_outputs_supported,
            "strict_requested": self.strict_requested,
            "strict_applied": self.strict_applied,
            "streaming_requested": self.streaming_requested,
        }


def structured(
    *,
    provider: Provider | None = None,
    messages: list[Message] | None = None,
    config: StructuredOutputConfig | None = None,
    engine: ExecutionEngine | None = None,
    context: RequestContext | None = None,
    content: Any | None = None,
    mode: StructuredExecutionMode | str = StructuredExecutionMode.AUTO,
    stream: bool = False,
    **kwargs: Any,
) -> Any:
    """Unified structured-output entry point.

    Dispatches to one of:
    - `extract_structured(...)`
    - `stream_structured(...)`
    - `validate_and_parse(...)`
    """
    selected_mode = _normalize_structured_mode(mode, content=content, stream=stream)
    if config is None:
        raise ValueError("config is required for structured execution")

    if selected_mode == StructuredExecutionMode.VALIDATE:
        if content is None:
            raise ValueError("content is required for validate mode")
        return validate_and_parse(content, config.schema)

    if provider is None:
        raise ValueError("provider is required for complete/stream structured execution")
    if messages is None:
        raise ValueError("messages are required for complete/stream structured execution")

    if selected_mode == StructuredExecutionMode.STREAM:
        return stream_structured(
            provider,
            messages,
            config,
            engine=engine,
            context=context,
            **kwargs,
        )

    return extract_structured(
        provider,
        messages,
        config,
        engine=engine,
        context=context,
        **kwargs,
    )


def normalize_structured_schema(
    schema: dict[str, Any],
    *,
    provider: Any | None = None,
) -> dict[str, Any]:
    """Normalize a JSON schema for provider-specific structured-output APIs."""
    provider_name = _structured_provider_name(provider)
    if provider_name == "openai":
        return _sanitize_json_schema_for_openai(schema)
    return copy.deepcopy(schema)


def build_structured_response_format(
    schema: dict[str, Any],
    *,
    provider: Any | None = None,
    model: str | None = None,
    name: str = "structured_output",
    strict: bool = True,
) -> str | dict[str, Any]:
    """Build the provider-appropriate response_format for structured output."""
    selection = select_structured_mode(
        provider=provider,
        model=model,
        stream=False,
        strict=strict,
        name=name,
        schema=schema,
    )
    if selection.response_mode == StructuredResponseMode.JSON_SCHEMA:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": _openai_safe_schema_name(name),
                "strict": bool(strict),
                "schema": normalize_structured_schema(schema, provider=selection.provider_name),
            },
        }
    return selection.response_format


def build_structured_repair_messages(
    *,
    base_messages: list[Any],
    raw_content: Any,
    validation_errors: list[str],
    schema: dict[str, Any],
    assistant_role: str = "assistant",
    user_role: str = "user",
) -> list[dict[str, str]]:
    """Build a generic repair-turn message sequence for structured retries."""
    errors_text = "\n".join(f"- {message}" for message in validation_errors) if validation_errors else "- invalid output"
    repaired_messages: list[dict[str, str]] = []
    for message in base_messages:
        if isinstance(message, dict):
            repaired_messages.append(dict(message))
        elif hasattr(message, "to_dict"):
            repaired_messages.append(dict(message.to_dict()))
        else:
            repaired_messages.append(dict(message))
    repaired_messages.append(
        {
            "role": assistant_role,
            "content": _serialize_prompt_payload(raw_content),
        }
    )
    repaired_messages.append(
        {
            "role": user_role,
            "content": (
                "Your previous JSON failed validation.\n"
                f"Validation errors:\n{errors_text}\n\n"
                "Return corrected JSON only, with no prose. It must satisfy this schema exactly:\n"
                f"{_serialize_prompt_payload(schema)}"
            ),
        }
    )
    return repaired_messages


async def parse_json_object_content(value: Any) -> tuple[dict[str, Any] | str, list[str]]:
    if isinstance(value, dict):
        return dict(value), []
    text = str(value or "").strip()
    if not text:
        return "", ["Structured completion did not return a JSON object."]
    parsed = await structured(
        content=text,
        config=StructuredOutputConfig(schema={"type": "object"}),
        mode=StructuredExecutionMode.VALIDATE,
    )
    if parsed.valid and isinstance(parsed.data, dict):
        return parsed.data, []
    return text, list(parsed.validation_errors)


def normalize_structured_result_envelope(
    content: dict[str, Any],
    *,
    promote_in_progress_with_result: bool = False,
    default_error_code: str = "structured_output_failed",
    default_error_message: str = "Structured execution failed",
    default_error_category: str = "validation",
    default_nondeterminism: dict[str, Any] | None = None,
) -> StructuredResultEnvelope:
    normalized = dict(content)

    raw_nondeterminism = normalized.get("nondeterminism")
    if isinstance(raw_nondeterminism, dict):
        nondeterminism = dict(raw_nondeterminism)
        for key in ("fallback_mode", "llm_failure_code"):
            value = nondeterminism.get(key)
            if isinstance(value, str) and not value.strip():
                nondeterminism.pop(key, None)
    else:
        nondeterminism = dict(
            default_nondeterminism
            or {"is_nondeterministic": True, "reasons": ["llm_structured"], "stability": "medium"}
        )

    status = str(normalized.get("status") or "succeeded").strip().lower()
    if status not in {"succeeded", "failed", "in_progress"}:
        status = "succeeded"

    envelope_keys = {"status", "result", "error", "artifacts", "nondeterminism"}
    if "result" in normalized and isinstance(normalized.get("result"), (dict, type(None))):
        result_payload = normalized.get("result")
    elif envelope_keys.intersection(set(normalized.keys())):
        result_payload = normalized.get("result")
    else:
        result_payload = normalized
    if not isinstance(result_payload, (dict, type(None))):
        result_payload = None

    artifacts = normalized.get("artifacts") if isinstance(normalized.get("artifacts"), list) else []

    error: StructuredEnvelopeError | None = None
    raw_error = normalized.get("error")
    if status == "failed":
        if isinstance(raw_error, dict):
            details = raw_error.get("details") if isinstance(raw_error.get("details"), dict) else None
            error = StructuredEnvelopeError(
                code=str(raw_error.get("code") or default_error_code).strip() or default_error_code,
                message=str(raw_error.get("message") or default_error_message).strip() or default_error_message,
                category=str(raw_error.get("category") or default_error_category).strip() or default_error_category,
                retryable=bool(raw_error.get("retryable", False)),
                retry_after_ms=_coerce_int_or_none(raw_error.get("retry_after_ms")),
                details=details,
            )
        else:
            error = StructuredEnvelopeError(
                code=default_error_code,
                message=default_error_message,
                category=default_error_category,
                retryable=False,
            )

    if status == "in_progress" and promote_in_progress_with_result and isinstance(result_payload, dict):
        status = "succeeded"

    return StructuredResultEnvelope(
        status=status,
        result=result_payload,
        artifacts=list(artifacts),
        nondeterminism=nondeterminism,
        error=error,
    )


async def execute_structured_completion_loop(
    *,
    engine: ExecutionEngine | None,
    provider: Provider,
    messages: list[dict[str, Any]],
    schema: dict[str, Any],
    completion_kwargs: dict[str, Any],
    max_repairs: int,
    runtime: StructuredToolRuntime | None = None,
    disable_tools_during_repairs: bool = True,
    tool_timeout_ms: int = 5000,
    soft_limit: int = 5,
    token_delta_callback: Any | None = None,
    token_delta_mode: str | None = None,
    progress_callback: Any | None = None,
    content_preparer: Any | None = None,
) -> StructuredCompletionLoopResult:
    base_messages = list(messages)
    current_messages = list(messages)
    declared_runtime = runtime or StructuredToolRuntime(
        tools_by_name={},
        provider_tools=[],
        max_tool_calls=0,
        max_tool_call_depth=0,
    )

    for repair_attempt in range(max(0, int(max_repairs)) + 1):
        active_runtime = declared_runtime
        if (
            repair_attempt > 0
            and disable_tools_during_repairs
            and declared_runtime.provider_tools
        ):
            active_runtime = StructuredToolRuntime(
                tools_by_name={},
                provider_tools=[],
                max_tool_calls=0,
                max_tool_call_depth=0,
            )

        completion, tool_error, completion_messages = await complete_with_tools(
            engine=engine,
            provider=provider,
            messages=current_messages,
            completion_kwargs=completion_kwargs,
            runtime=active_runtime,
            tool_timeout_ms=max(1, int(tool_timeout_ms)),
            soft_limit=max(1, int(soft_limit)),
            token_delta_callback=(token_delta_callback if repair_attempt == 0 else None),
            token_delta_mode=token_delta_mode if repair_attempt == 0 else None,
            progress_callback=progress_callback,
        )
        raw_content = getattr(completion, "content", None) if completion is not None else None
        output_messages = completion_messages or list(current_messages)

        if tool_error is not None:
            return StructuredCompletionLoopResult(
                completion=completion,
                completion_messages=output_messages,
                raw_content=raw_content,
                repair_attempts=repair_attempt,
                tool_error=tool_error,
            )

        if completion is None:
            return StructuredCompletionLoopResult(
                completion=None,
                completion_messages=output_messages,
                raw_content=raw_content,
                validation_errors=["Structured completion returned no output."],
                repair_attempts=repair_attempt,
            )

        prepared_content, validation_errors = await parse_json_object_content(raw_content)
        if content_preparer is not None and isinstance(prepared_content, dict):
            prepared = content_preparer(prepared_content, completion)
            if inspect.isawaitable(prepared):
                prepared = await prepared
            prepared_content, validation_errors = prepared

        if isinstance(prepared_content, dict) and not validation_errors:
            return StructuredCompletionLoopResult(
                completion=completion,
                completion_messages=output_messages,
                content=prepared_content,
                raw_content=raw_content,
                repair_attempts=repair_attempt,
            )

        if repair_attempt >= max(0, int(max_repairs)):
            return StructuredCompletionLoopResult(
                completion=completion,
                completion_messages=output_messages,
                content=prepared_content,
                raw_content=raw_content,
                validation_errors=list(validation_errors) or ["Structured completion did not return a JSON object."],
                repair_attempts=repair_attempt,
            )

        current_messages = build_structured_repair_messages(
            base_messages=output_messages or base_messages,
            raw_content=raw_content,
            validation_errors=list(validation_errors) or ["Structured completion did not return a JSON object."],
            schema=schema,
        )

    return StructuredCompletionLoopResult(
        completion=None,
        completion_messages=list(current_messages),
        content=None,
        raw_content=None,
        validation_errors=["Max repair attempts reached."],
        repair_attempts=max(0, int(max_repairs)),
    )


def finalize_structured_completion_loop(
    loop_result: StructuredCompletionLoopResult,
    *,
    promote_in_progress_with_result: bool = False,
    default_error_code: str = "structured_output_failed",
    default_error_message: str = "Structured execution failed",
    default_error_category: str = "validation",
    default_nondeterminism: dict[str, Any] | None = None,
) -> StructuredExecutionOutcome:
    completion = loop_result.completion

    if loop_result.tool_error is not None:
        error = loop_result.tool_error
        details = dict(error.details) if isinstance(error.details, dict) else {}
        details.setdefault("normalized_failure", error.to_normalized_failure())
        return StructuredExecutionOutcome(
            envelope=None,
            failure=StructuredExecutionFailure(
                code=error.code,
                message=error.message,
                category=error.category,
                retryable=error.retryable,
                details=details or None,
            ),
            completion=completion,
            raw_content=loop_result.raw_content,
            validation_errors=list(loop_result.validation_errors),
            repair_attempts=loop_result.repair_attempts,
        )

    if completion is None:
        failure = StructuredExecutionFailure(
            code="provider_no_output",
            message="structured completion returned no output",
            category="dependency",
            retryable=True,
        )
        return StructuredExecutionOutcome(
            envelope=None,
            failure=StructuredExecutionFailure(
                code=failure.code,
                message=failure.message,
                category=failure.category,
                retryable=failure.retryable,
                details={"normalized_failure": failure.to_normalized_failure()},
            ),
            completion=None,
            raw_content=loop_result.raw_content,
            validation_errors=list(loop_result.validation_errors),
            repair_attempts=loop_result.repair_attempts,
        )

    completion_status = _coerce_int_or_none(getattr(completion, "status", None)) or 0
    if not bool(getattr(completion, "ok", False)):
        failure = StructuredExecutionFailure(
            code="provider_error",
            message=str(getattr(completion, "error", "") or f"status:{completion_status}"),
            category="dependency",
            retryable=is_retryable_status(completion_status),
        )
        return StructuredExecutionOutcome(
            envelope=None,
            failure=StructuredExecutionFailure(
                code=failure.code,
                message=failure.message,
                category=failure.category,
                retryable=failure.retryable,
                details={"normalized_failure": failure.to_normalized_failure(model=getattr(completion, "model", None))},
            ),
            completion=completion,
            raw_content=loop_result.raw_content,
            validation_errors=list(loop_result.validation_errors),
            repair_attempts=loop_result.repair_attempts,
        )

    if isinstance(loop_result.content, dict) and not loop_result.validation_errors:
        return StructuredExecutionOutcome(
            envelope=normalize_structured_result_envelope(
                loop_result.content,
                promote_in_progress_with_result=promote_in_progress_with_result,
                default_error_code=default_error_code,
                default_error_message=default_error_message,
                default_error_category=default_error_category,
                default_nondeterminism=default_nondeterminism,
            ),
            failure=None,
            completion=completion,
            raw_content=loop_result.raw_content,
            validation_errors=[],
            repair_attempts=loop_result.repair_attempts,
        )

    details: dict[str, Any] = {
        "validation_errors": list(loop_result.validation_errors),
        "repair_attempts": loop_result.repair_attempts,
    }
    if loop_result.raw_content is not None:
        details["raw_type"] = type(loop_result.raw_content).__name__
    failure = StructuredExecutionFailure(
        code=default_error_code,
        message=default_error_message,
        category=default_error_category,
        retryable=False,
        details=details,
    )
    failure_details = dict(details)
    failure_details["normalized_failure"] = failure.to_normalized_failure(model=getattr(completion, "model", None))
    return StructuredExecutionOutcome(
        envelope=None,
        failure=StructuredExecutionFailure(
            code=failure.code,
            message=failure.message,
            category=failure.category,
            retryable=failure.retryable,
            details=failure_details,
        ),
        completion=completion,
        raw_content=loop_result.raw_content,
        validation_errors=list(loop_result.validation_errors),
        repair_attempts=loop_result.repair_attempts,
    )


def select_structured_mode(
    *,
    provider: Any | None = None,
    model: str | None = None,
    stream: bool = False,
    strict: bool = True,
    name: str = "structured_output",
    schema: dict[str, Any] | None = None,
) -> StructuredModeSelection:
    """Select the best structured-output transport for the provider/model."""
    from .model_catalog import get_default_model_catalog, infer_provider_for_model
    from .provider_registry import get_default_provider_registry

    provider_name = _structured_provider_name(provider)
    model_name = str(model or infer_model_name(provider) or "").strip() or None

    if not provider_name and provider is None and model_name:
        provider_name = infer_provider_for_model(model_name)

    registry = get_default_provider_registry()
    catalog = get_default_model_catalog()

    descriptor = None
    if provider_name:
        try:
            descriptor = registry.get(provider_name)
            provider_name = descriptor.name
        except Exception:
            descriptor = None

    metadata = None
    if model_name:
        try:
            metadata = catalog.get(model_name)
        except Exception:
            metadata = None

    structured_outputs_supported = bool(descriptor.capabilities.structured_outputs) if descriptor else False
    if descriptor is None and provider is None and metadata is not None:
        try:
            descriptor = registry.get(metadata.provider)
            provider_name = descriptor.name
            structured_outputs_supported = bool(descriptor.capabilities.structured_outputs)
        except Exception:
            descriptor = None

    if provider_name == "openai" and structured_outputs_supported:
        response_mode = StructuredResponseMode.JSON_SCHEMA
        response_format: str | dict[str, Any] | None = {
            "type": "json_schema",
            "json_schema": {
                "name": _openai_safe_schema_name(name),
                "strict": bool(strict),
                "schema": normalize_structured_schema(schema or {}, provider=provider_name),
            },
        }
        strict_applied = bool(strict)
    elif structured_outputs_supported:
        response_mode = StructuredResponseMode.JSON_OBJECT
        response_format = "json_object"
        strict_applied = False
    elif provider_name:
        response_mode = StructuredResponseMode.PROMPT_ONLY
        response_format = None
        strict_applied = False
    else:
        response_mode = StructuredResponseMode.JSON_OBJECT
        response_format = "json_object"
        strict_applied = False

    return StructuredModeSelection(
        provider_name=provider_name,
        model_name=model_name,
        response_mode=response_mode,
        response_format=response_format,
        structured_outputs_supported=structured_outputs_supported,
        strict_requested=bool(strict),
        strict_applied=strict_applied,
        streaming_requested=bool(stream),
    )


async def extract_structured(
    provider: Provider,
    messages: list[Message],
    config: StructuredOutputConfig,
    *,
    engine: ExecutionEngine | None = None,
    context: RequestContext | None = None,
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
        engine: Optional execution engine. When provided, requests go through
            the engine instead of calling the provider directly.
        context: Optional request context for engine-backed execution.
        **kwargs: Additional arguments for provider.complete() / engine.complete().
    
    Returns:
        StructuredResult with parsed data and validation status.
    """
    from .providers.types import Message as Msg
    
    schema = config.schema
    structured_provider = engine.provider if engine is not None else provider
    mode_selection = select_structured_mode(
        provider=structured_provider,
        model=str(kwargs.get("model") or infer_model_name(structured_provider) or infer_model_name(provider) or "").strip() or None,
        stream=False,
        strict=config.strict_mode,
        name=f"{_structured_provider_name(structured_provider) or 'structured'}_output",
        schema=schema,
    )
    response_format = mode_selection.response_format
    
    # Initial request with schema instruction
    schema_instruction = (
        f"Respond with valid JSON matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n"
        "Only output the JSON, no additional text."
    )
    
    # Build initial request
    request_messages = list(messages) + [Msg.system(schema_instruction)]
    diagnostics = StructuredDiagnostics()
    
    for attempt in range(config.max_repair_attempts + 1):
        # Determine temperature
        temp = config.repair_temperature if attempt > 0 else kwargs.get("temperature")
        call_kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
        if temp is not None:
            call_kwargs["temperature"] = temp
        
        # Make completion request
        try:
            if engine is not None:
                result = await engine.complete_content(
                    build_content_request_envelope(
                        engine=engine,
                        provider=provider,
                        messages=request_messages,
                        request_kwargs={
                            **call_kwargs,
                            "temperature": temp,
                            **({"response_format": response_format} if response_format is not None else {}),
                        },
                        model=str(call_kwargs.get("model") or infer_model_name(provider) or "unknown"),
                    ),
                    context=context,
                )
            else:
                complete_kwargs = dict(call_kwargs)
                if response_format is not None:
                    complete_kwargs["response_format"] = response_format
                result = await provider.complete(request_messages, **complete_kwargs)
        except Exception as e:
            diagnostics.record(
                StructuredAttemptTrace(
                    attempt=attempt,
                    response_kind="provider_error",
                    raw_content=str(e),
                    validation_errors=[f"Provider error: {e}"],
                )
            )
            return StructuredResult(
                data=None,
                raw_content=str(e),
                valid=False,
                repair_attempts=attempt,
                validation_errors=[f"Provider error: {e}"],
                response_kind="provider_error",
                diagnostics=diagnostics,
                usage=getattr(result, "usage", None) if "result" in locals() else None,
            )

        raw, response_kind, pre_validation_errors = _coerce_structured_response(result)
        if pre_validation_errors:
            diagnostics.record(
                StructuredAttemptTrace(
                    attempt=attempt,
                    response_kind=response_kind,
                    raw_content=raw,
                    validation_errors=list(pre_validation_errors),
                )
            )
            return StructuredResult(
                data=None,
                raw_content=raw,
                valid=False,
                repair_attempts=attempt,
                validation_errors=pre_validation_errors,
                response_kind=response_kind,
                diagnostics=diagnostics,
                usage=getattr(result, "usage", None),
            )
        
        validation_result = await validate_and_parse(raw, schema)
        extracted_json = None
        parsed_flag = False
        if validation_result.diagnostics is not None and validation_result.diagnostics.attempts:
            first_attempt = validation_result.diagnostics.attempts[0]
            extracted_json = first_attempt.extracted_json
            parsed_flag = bool(first_attempt.parsed)

        if validation_result.valid and validation_result.data is not None:
            diagnostics.record(
                StructuredAttemptTrace(
                    attempt=attempt,
                    response_kind=response_kind,
                    raw_content=raw,
                    extracted_json=extracted_json,
                    parsed=parsed_flag,
                    valid=True,
                )
            )
            return StructuredResult(
                data=validation_result.data,
                raw_content=raw,
                valid=True,
                repair_attempts=attempt,
                response_kind=response_kind,
                diagnostics=diagnostics,
                usage=getattr(result, "usage", None),
            )
        
        # Repair loop
        diagnostics.record(
            StructuredAttemptTrace(
                attempt=attempt,
                response_kind=response_kind,
                raw_content=raw,
                extracted_json=extracted_json,
                parsed=parsed_flag,
                valid=False,
                validation_errors=list(validation_result.validation_errors),
            )
        )
        if attempt < config.max_repair_attempts:
            request_messages = build_structured_repair_messages(
                base_messages=list(messages),
                raw_content=raw,
                validation_errors=validation_result.validation_errors,
                schema=schema,
            )
            continue
        
        # Max attempts reached
        return StructuredResult(
            data=validation_result.data,
            raw_content=raw,
            valid=False,
            repair_attempts=attempt,
            validation_errors=validation_result.validation_errors,
            response_kind=response_kind,
            diagnostics=diagnostics,
            usage=getattr(result, "usage", None),
        )
    
    # Should not reach here
    return StructuredResult(
        data=None,
        raw_content="",
        valid=False,
        repair_attempts=config.max_repair_attempts,
        validation_errors=["Max repair attempts reached"],
        response_kind="empty",
        diagnostics=diagnostics,
    )


async def stream_structured(
    provider: Provider,
    messages: list[Message],
    config: StructuredOutputConfig,
    *,
    engine: ExecutionEngine | None = None,
    context: RequestContext | None = None,
    **kwargs,
) -> AsyncIterator[StructuredStreamEvent]:
    """Stream structured output and emit a final validated StructuredResult."""
    from .providers.types import CompletionResult, Message as Msg, StreamEventType

    schema = config.schema
    structured_provider = engine.provider if engine is not None else provider
    mode_selection = select_structured_mode(
        provider=structured_provider,
        model=str(kwargs.get("model") or infer_model_name(structured_provider) or infer_model_name(provider) or "").strip() or None,
        stream=True,
        strict=config.strict_mode,
        name=f"{_structured_provider_name(structured_provider) or 'structured'}_output",
        schema=schema,
    )
    response_format = mode_selection.response_format
    schema_instruction = (
        f"Respond with valid JSON matching this schema:\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n"
        "Only output the JSON, no additional text."
    )
    request_messages = list(messages) + [Msg.system(schema_instruction)]

    if engine is not None:
        stream = engine.stream_content(
            build_content_request_envelope(
                engine=engine,
                provider=provider,
                messages=request_messages,
                request_kwargs={
                    **kwargs,
                    **({"response_format": response_format} if response_format is not None else {}),
                },
                model=str(kwargs.get("model") or infer_model_name(provider) or "unknown"),
                stream=True,
            ),
            context=context,
        )
    else:
        stream_kwargs = dict(kwargs)
        if response_format is not None:
            stream_kwargs["response_format"] = response_format
        stream = provider.stream(request_messages, **stream_kwargs)

    buffered_content = ""
    async for event in stream:
        if event.type == StreamEventType.TOKEN:
            token = str(event.data or "")
            buffered_content += token
            yield StructuredStreamEvent(
                type=StructuredStreamEventType.CONTENT_DELTA,
                data=token,
                raw_event=event,
            )
            continue

        if event.type == StreamEventType.ERROR:
            payload = event.data if isinstance(event.data, dict) else {"status": 500, "error": str(event.data)}
            yield StructuredStreamEvent(
                type=StructuredStreamEventType.ERROR,
                data=payload,
                raw_event=event,
            )
            return

        if event.type == StreamEventType.DONE:
            try:
                done_result = ensure_completion_result(event.data)
            except TypeError:
                done_result = event.data if isinstance(event.data, CompletionResult) else None
            structured_result = await _structured_result_from_stream(
                buffered_content=buffered_content,
                result=done_result,
                schema=schema,
            )
            yield StructuredStreamEvent(
                type=StructuredStreamEventType.RESULT,
                data=structured_result,
                raw_event=event,
            )
            yield StructuredStreamEvent(
                type=StructuredStreamEventType.DONE,
                data=structured_result,
                raw_event=event,
            )
            return

        yield StructuredStreamEvent(
            type=StructuredStreamEventType.RAW_EVENT,
            data=event.data,
            raw_event=event,
        )


def _coerce_structured_response(result: Any) -> tuple[str, str, list[str]]:
    if hasattr(result, "message") and hasattr(result, "to_completion_result"):
        tool_calls = [block for block in getattr(result.message, "blocks", ()) if getattr(block, "type", None) == "tool_call"]
        text = content_blocks_to_text(list(getattr(result.message, "blocks", ())))
        response_kind = "content_blocks"
        if tool_calls and text.strip():
            return text, "mixed_content_and_tools", ["Structured output cannot mix tool calls with JSON content."]
        if tool_calls:
            return text, "tool_calls", ["Structured output returned tool calls instead of JSON content."]
        if not text.strip():
            return text, response_kind, ["Structured output response was empty."]
        return text, response_kind, []

    tool_calls = getattr(result, "tool_calls", None) or []
    raw_content = getattr(result, "content", None)
    if isinstance(raw_content, str) or raw_content is None:
        text = str(raw_content or "")
        response_kind = "text"
    elif isinstance(raw_content, list):
        text = content_blocks_to_text(raw_content)
        response_kind = "content_blocks"
    else:
        text = str(raw_content)
        response_kind = "unknown_content"

    if tool_calls and text.strip():
        return text, "mixed_content_and_tools", ["Structured output cannot mix tool calls with JSON content."]
    if tool_calls:
        return text, "tool_calls", ["Structured output returned tool calls instead of JSON content."]
    if not text.strip():
        return text, response_kind, ["Structured output response was empty."]
    return text, response_kind, []


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
    
    content = content.strip()

    candidate = _extract_first_json_value(content)
    if candidate is not None:
        return candidate

    return content


def _extract_first_json_value(content: str) -> str | None:
    start: int | None = None
    stack: list[str] = []
    in_string = False
    escape = False

    for index, char in enumerate(content):
        if start is None:
            if char in "{[":
                start = index
                stack.append("}" if char == "{" else "]")
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "{[":
            stack.append("}" if char == "{" else "]")
            continue

        if char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return content[start : index + 1].strip()

    return None


async def validate_and_parse(
    content: Any,
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
    diagnostics = StructuredDiagnostics()
    raw_content = _normalize_structured_text_input(content)
    try:
        extracted = _extract_json(raw_content)
        data = json.loads(extracted)
    except json.JSONDecodeError as e:
        diagnostics.record(
            StructuredAttemptTrace(
                attempt=0,
                response_kind="text",
                raw_content=raw_content,
                extracted_json=_extract_json(raw_content),
                validation_errors=[f"JSON parse error: {e}"],
            )
        )
        return StructuredResult(
            data=None,
            raw_content=raw_content,
            valid=False,
            validation_errors=[f"JSON parse error: {e}"],
            diagnostics=diagnostics,
        )
    
    validation = validate_against_schema(data, schema)
    diagnostics.record(
        StructuredAttemptTrace(
            attempt=0,
            response_kind="text",
            raw_content=raw_content,
            extracted_json=extracted,
            parsed=True,
            valid=validation.valid,
            validation_errors=validation.errors if not validation.valid else [],
        )
    )
    return StructuredResult(
        data=data,
        raw_content=raw_content,
        valid=validation.valid,
        validation_errors=validation.errors if not validation.valid else [],
        diagnostics=diagnostics,
    )


def _serialize_prompt_payload(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


async def _structured_result_from_stream(
    *,
    buffered_content: str,
    result: Any,
    schema: dict[str, Any],
) -> StructuredResult:
    if result is None:
        return StructuredResult(
            data=None,
            raw_content=buffered_content,
            valid=False,
            validation_errors=["Structured streaming completed without a final result."],
            response_kind="empty",
            diagnostics=StructuredDiagnostics(
                attempts=[
                    StructuredAttemptTrace(
                        attempt=0,
                        response_kind="empty",
                        raw_content=buffered_content,
                        validation_errors=["Structured streaming completed without a final result."],
                    )
                ],
                final_response_kind="empty",
            ),
            usage=getattr(result, "usage", None),
        )

    raw_content, response_kind, pre_validation_errors = _coerce_structured_response(result)
    if not buffered_content.strip():
        buffered_content = raw_content
    if pre_validation_errors:
        diagnostics = StructuredDiagnostics()
        diagnostics.record(
            StructuredAttemptTrace(
                attempt=0,
                response_kind=response_kind,
                raw_content=buffered_content,
                validation_errors=list(pre_validation_errors),
            )
        )
        return StructuredResult(
            data=None,
            raw_content=buffered_content,
            valid=False,
            validation_errors=pre_validation_errors,
            response_kind=response_kind,
            diagnostics=diagnostics,
            usage=getattr(result, "usage", None),
        )

    parsed = await validate_and_parse(buffered_content, schema)
    parsed.raw_content = buffered_content
    parsed.response_kind = response_kind
    parsed.usage = getattr(result, "usage", None)
    if parsed.diagnostics is not None:
        parsed.diagnostics.final_response_kind = response_kind
    return parsed


def _structured_provider_name(provider: Any | None) -> str:
    if provider is None:
        return ""
    if isinstance(provider, str):
        value = provider.lower()
    else:
        explicit_name = getattr(provider, "provider_name", None)
        if isinstance(explicit_name, str) and explicit_name.strip():
            value = explicit_name.strip().lower()
        else:
            value = provider.__class__.__name__.lower()
    if "openai" in value or value.startswith("gpt"):
        return "openai"
    if "anthropic" in value or "claude" in value:
        return "anthropic"
    if "google" in value or "gemini" in value:
        return "google"
    return ""


def _normalize_structured_mode(
    mode: StructuredExecutionMode | str,
    *,
    content: Any | None,
    stream: bool,
) -> StructuredExecutionMode:
    if isinstance(mode, StructuredExecutionMode):
        selected = mode
    else:
        selected = StructuredExecutionMode(str(mode or StructuredExecutionMode.AUTO.value).lower())
    if selected == StructuredExecutionMode.AUTO:
        if content is not None:
            return StructuredExecutionMode.VALIDATE
        if stream:
            return StructuredExecutionMode.STREAM
        return StructuredExecutionMode.COMPLETE
    return selected


def _normalize_structured_text_input(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content_blocks_to_text(content)
    return str(content or "")


def _openai_safe_schema_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip())
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "structured_output"
    if len(name) > 64:
        name = name[:64].rstrip("_") or "structured_output"
    return name


def _sanitize_json_schema_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if str(key) in {"$schema", "$id"}:
                    continue
                out[str(key)] = _walk(item)
            if "$ref" in out:
                # OpenAI strict json_schema rejects sibling keywords next to $ref.
                # Pydantic v2 commonly emits metadata like description/title/default
                # alongside references, so collapse those nodes to a bare reference.
                return {"$ref": out["$ref"]}
            if out.get("type") == "object" or isinstance(out.get("properties"), dict):
                properties = out.get("properties")
                if isinstance(properties, dict):
                    out["required"] = list(properties.keys())
                out["additionalProperties"] = False
            return out
        if isinstance(value, list):
            return [_walk(item) for item in value]
        return value

    sanitized = _walk(copy.deepcopy(schema))
    return sanitized if isinstance(sanitized, dict) else dict(schema)


def _coerce_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "StructuredExecutionMode",
    "StructuredResponseMode",
    "StructuredModeSelection",
    "StructuredOutputConfig",
    "StructuredAttemptTrace",
    "StructuredCompletionLoopResult",
    "StructuredDiagnostics",
    "StructuredEnvelopeError",
    "StructuredResultEnvelope",
    "StructuredStreamEventType",
    "StructuredStreamEvent",
    "StructuredResult",
    "build_structured_response_format",
    "build_structured_repair_messages",
    "execute_structured_completion_loop",
    "extract_structured",
    "finalize_structured_completion_loop",
    "normalize_structured_schema",
    "normalize_structured_result_envelope",
    "parse_json_object_content",
    "select_structured_mode",
    "stream_structured",
    "structured",
    "validate_and_parse",
    "StructuredExecutionFailure",
    "StructuredExecutionOutcome",
]
