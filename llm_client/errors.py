"""
Error taxonomy for llm-client.

This module provides a hierarchical exception system with:
- Error codes for programmatic handling
- Retryable vs non-retryable classification
- Structured context for debugging
- Provider-specific error mapping
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standardized error codes for the LLM client."""

    # Provider errors (1xxx)
    PROVIDER_ERROR = "ERR_1000"
    RATE_LIMIT = "ERR_1001"
    AUTHENTICATION = "ERR_1002"
    QUOTA_EXCEEDED = "ERR_1003"
    MODEL_NOT_FOUND = "ERR_1004"
    CONTEXT_LENGTH = "ERR_1005"
    CONTENT_FILTER = "ERR_1006"
    PROVIDER_UNAVAILABLE = "ERR_1007"
    PROVIDER_TIMEOUT = "ERR_1008"
    INVALID_RESPONSE = "ERR_1009"

    # Validation errors (2xxx)
    VALIDATION_ERROR = "ERR_2000"
    INVALID_MESSAGE = "ERR_2001"
    INVALID_TOOL = "ERR_2002"
    MESSAGE_TOO_LONG = "ERR_2003"
    TOO_MANY_MESSAGES = "ERR_2004"
    INVALID_SCHEMA = "ERR_2005"

    # Cache errors (3xxx)
    CACHE_ERROR = "ERR_3000"
    CACHE_READ_ERROR = "ERR_3001"
    CACHE_WRITE_ERROR = "ERR_3002"
    CACHE_CONNECTION_ERROR = "ERR_3003"

    # Tool errors (4xxx)
    TOOL_ERROR = "ERR_4000"
    TOOL_NOT_FOUND = "ERR_4001"
    TOOL_EXECUTION_ERROR = "ERR_4002"
    TOOL_TIMEOUT = "ERR_4003"
    TOOL_VALIDATION_ERROR = "ERR_4004"

    # Agent errors (5xxx)
    AGENT_ERROR = "ERR_5000"
    MAX_TURNS_EXCEEDED = "ERR_5001"
    AGENT_TIMEOUT = "ERR_5002"

    # Configuration errors (6xxx)
    CONFIG_ERROR = "ERR_6000"
    MISSING_API_KEY = "ERR_6001"
    INVALID_CONFIG = "ERR_6002"

    # Internal errors (9xxx)
    INTERNAL_ERROR = "ERR_9000"
    UNKNOWN_ERROR = "ERR_9999"


@dataclass
class ErrorContext:
    """Structured context for error debugging."""

    request_id: str | None = None
    trace_id: str | None = None
    provider: str | None = None
    model: str | None = None
    attempt: int = 1
    operation: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "provider": self.provider,
            "model": self.model,
            "attempt": self.attempt,
            "operation": self.operation,
            **self.extra,
        }


class FailureCategory(str, Enum):
    PROVIDER = "provider"
    REQUEST = "request"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    QUOTA = "quota"
    TIMEOUT = "timeout"
    AVAILABILITY = "availability"
    MODEL = "model"
    CONTENT_FILTER = "content_filter"
    VALIDATION = "validation"
    TOOL = "tool"
    TOOL_POLICY = "tool_policy"
    STRUCTURED_OUTPUT = "structured_output"
    CONFIGURATION = "configuration"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NormalizedFailure:
    code: str
    category: FailureCategory
    message: str
    retryable: bool = False
    status: int | None = None
    provider: str | None = None
    model: str | None = None
    operation: str | None = None
    request_id: str | None = None
    remediation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    cause_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "category": self.category.value,
            "message": self.message,
            "retryable": self.retryable,
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "request_id": self.request_id,
            "remediation": self.remediation,
            "details": dict(self.details),
            "cause_type": self.cause_type,
        }


class LLMClientError(Exception):
    """
    Base exception for all LLM client errors.

    Attributes:
        code: Standardized error code for programmatic handling
        message: Human-readable error message
        retryable: Whether the operation can be retried
        context: Structured debugging context
        cause: Original exception that caused this error
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        self.context = context or ErrorContext()
        self.cause = cause

    def __str__(self) -> str:
        parts = [f"[{self.code.value}] {self.message}"]
        if self.context.request_id:
            parts.append(f"(request_id={self.context.request_id})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
        }

    def to_failure(self) -> NormalizedFailure:
        status = getattr(self, "http_status", None)
        return NormalizedFailure(
            code=self.code.value,
            category=_category_for_error_code(self.code),
            message=self.message,
            retryable=self.retryable,
            status=status if isinstance(status, int) else None,
            provider=self.context.provider,
            model=self.context.model,
            operation=self.context.operation,
            request_id=self.context.request_id,
            remediation=_remediation_hint_for_error_code(self.code),
            details=dict(self.context.extra),
            cause_type=type(self.cause).__name__ if self.cause else type(self).__name__,
        )


# =============================================================================
# Provider Errors
# =============================================================================


class ProviderError(LLMClientError):
    """Base class for errors from LLM providers."""

    code = ErrorCode.PROVIDER_ERROR
    retryable = False
    http_status: int | None = None

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.http_status = http_status


class RateLimitError(ProviderError):
    """Rate limit exceeded. Operation can be retried after a delay."""

    code = ErrorCode.RATE_LIMIT
    retryable = True

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: float | None = None,
        **kwargs,
    ):
        super().__init__(message, http_status=429, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Invalid or missing API key. Not retryable."""

    code = ErrorCode.AUTHENTICATION
    retryable = False

    def __init__(
        self,
        message: str = "Authentication failed. Check your API key.",
        **kwargs,
    ):
        super().__init__(message, http_status=401, **kwargs)


class QuotaExceededError(ProviderError):
    """API quota/spending limit exceeded. Not immediately retryable."""

    code = ErrorCode.QUOTA_EXCEEDED
    retryable = False

    def __init__(
        self,
        message: str = "API quota exceeded",
        **kwargs,
    ):
        super().__init__(message, http_status=402, **kwargs)


class ModelNotFoundError(ProviderError):
    """Requested model does not exist or is not accessible."""

    code = ErrorCode.MODEL_NOT_FOUND
    retryable = False

    def __init__(
        self,
        message: str = "Model not found",
        *,
        model: str | None = None,
        **kwargs,
    ):
        if model:
            message = f"Model not found: {model}"
        super().__init__(message, http_status=404, **kwargs)


class ContextLengthError(ProviderError):
    """Input exceeds the model's context window."""

    code = ErrorCode.CONTEXT_LENGTH
    retryable = False

    def __init__(
        self,
        message: str = "Context length exceeded",
        *,
        max_tokens: int | None = None,
        actual_tokens: int | None = None,
        **kwargs,
    ):
        super().__init__(message, http_status=400, **kwargs)
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


class ContentFilterError(ProviderError):
    """Content was rejected by safety filters."""

    code = ErrorCode.CONTENT_FILTER
    retryable = False

    def __init__(
        self,
        message: str = "Content rejected by safety filters",
        **kwargs,
    ):
        super().__init__(message, http_status=400, **kwargs)


class ProviderUnavailableError(ProviderError):
    """Provider service is temporarily unavailable. Retryable."""

    code = ErrorCode.PROVIDER_UNAVAILABLE
    retryable = True

    def __init__(
        self,
        message: str = "Provider service unavailable",
        **kwargs,
    ):
        super().__init__(message, http_status=503, **kwargs)


class ProviderTimeoutError(ProviderError):
    """Request to provider timed out. Retryable."""

    code = ErrorCode.PROVIDER_TIMEOUT
    retryable = True

    def __init__(
        self,
        message: str = "Request timed out",
        *,
        timeout: float | None = None,
        **kwargs,
    ):
        super().__init__(message, http_status=504, **kwargs)
        self.timeout = timeout


class InvalidResponseError(ProviderError):
    """Provider returned an invalid or unexpected response."""

    code = ErrorCode.INVALID_RESPONSE
    retryable = True  # Might work on retry

    def __init__(
        self,
        message: str = "Invalid response from provider",
        **kwargs,
    ):
        super().__init__(message, http_status=500, **kwargs)


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(LLMClientError):
    """Base class for input/output validation errors."""

    code = ErrorCode.VALIDATION_ERROR
    retryable = False


class InvalidMessageError(ValidationError):
    """Message format is invalid."""

    code = ErrorCode.INVALID_MESSAGE


class InvalidToolError(ValidationError):
    """Tool definition or call is invalid."""

    code = ErrorCode.INVALID_TOOL


class MessageTooLongError(ValidationError):
    """Individual message exceeds length limit."""

    code = ErrorCode.MESSAGE_TOO_LONG

    def __init__(
        self,
        message: str = "Message too long",
        *,
        max_length: int | None = None,
        actual_length: int | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.max_length = max_length
        self.actual_length = actual_length


class TooManyMessagesError(ValidationError):
    """Too many messages in conversation."""

    code = ErrorCode.TOO_MANY_MESSAGES


class InvalidSchemaError(ValidationError):
    """JSON schema validation failed."""

    code = ErrorCode.INVALID_SCHEMA


# =============================================================================
# Cache Errors
# =============================================================================


class CacheError(LLMClientError):
    """Base class for cache-related errors."""

    code = ErrorCode.CACHE_ERROR
    retryable = False  # Cache failures are often not retryable


class CacheReadError(CacheError):
    """Failed to read from cache."""

    code = ErrorCode.CACHE_READ_ERROR


class CacheWriteError(CacheError):
    """Failed to write to cache."""

    code = ErrorCode.CACHE_WRITE_ERROR


class CacheConnectionError(CacheError):
    """Failed to connect to cache backend."""

    code = ErrorCode.CACHE_CONNECTION_ERROR
    retryable = True  # Connection issues might resolve


# =============================================================================
# Tool Errors
# =============================================================================


class ToolError(LLMClientError):
    """Base class for tool-related errors."""

    code = ErrorCode.TOOL_ERROR
    retryable = False


class ToolNotFoundError(ToolError):
    """Requested tool is not registered."""

    code = ErrorCode.TOOL_NOT_FOUND

    def __init__(
        self,
        message: str = "Tool not found",
        *,
        tool_name: str | None = None,
        **kwargs,
    ):
        if tool_name:
            message = f"Tool not found: {tool_name}"
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    code = ErrorCode.TOOL_EXECUTION_ERROR

    def __init__(
        self,
        message: str = "Tool execution failed",
        *,
        tool_name: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.tool_name = tool_name


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""

    code = ErrorCode.TOOL_TIMEOUT
    retryable = True

    def __init__(
        self,
        message: str = "Tool execution timed out",
        *,
        timeout: float | None = None,
        tool_name: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.timeout = timeout
        self.tool_name = tool_name


class ToolValidationError(ToolError):
    """Tool arguments failed validation."""

    code = ErrorCode.TOOL_VALIDATION_ERROR


# =============================================================================
# Agent Errors
# =============================================================================


class AgentError(LLMClientError):
    """Base class for agent-related errors."""

    code = ErrorCode.AGENT_ERROR
    retryable = False


class MaxTurnsExceededError(AgentError):
    """Agent reached maximum number of turns."""

    code = ErrorCode.MAX_TURNS_EXCEEDED

    def __init__(
        self,
        message: str = "Maximum turns exceeded",
        *,
        max_turns: int | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.max_turns = max_turns


class AgentTimeoutError(AgentError):
    """Agent execution timed out."""

    code = ErrorCode.AGENT_TIMEOUT
    retryable = True


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigError(LLMClientError):
    """Base class for configuration errors."""

    code = ErrorCode.CONFIG_ERROR
    retryable = False


class MissingAPIKeyError(ConfigError):
    """Required API key is not set."""

    code = ErrorCode.MISSING_API_KEY

    def __init__(
        self,
        message: str = "API key not found",
        *,
        provider: str | None = None,
        env_var: str | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.provider = provider
        self.env_var = env_var


class InvalidConfigError(ConfigError):
    """Configuration is invalid."""

    code = ErrorCode.INVALID_CONFIG


# =============================================================================
# Error Mapping from HTTP Status Codes
# =============================================================================


def error_from_status(
    status: int,
    message: str,
    *,
    provider: str | None = None,
    context: ErrorContext | None = None,
) -> LLMClientError:
    """
    Create an appropriate ProviderError from an HTTP status code.

    Args:
        status: HTTP status code
        message: Error message from the provider
        provider: Provider name for context
        context: Additional error context

    Returns:
        Appropriate ProviderError subclass
    """
    ctx = context or ErrorContext(provider=provider)
    if status == 400:
        inferred = _error_from_bad_request_message(message, context=ctx)
        if inferred is not None:
            return inferred
    if status == 403:
        inferred = _error_from_forbidden_message(message, context=ctx)
        if inferred is not None:
            return inferred

    error_map: dict[int, type[ProviderError]] = {
        400: InvalidResponseError,  # Bad request - could be many things
        401: AuthenticationError,
        402: QuotaExceededError,
        403: AuthenticationError,  # Forbidden usually means auth issue
        404: ModelNotFoundError,
        408: ProviderTimeoutError,
        429: RateLimitError,
        500: InvalidResponseError,
        502: ProviderUnavailableError,
        503: ProviderUnavailableError,
        504: ProviderTimeoutError,
    }

    error_class = error_map.get(status, ProviderError)
    if error_class is ProviderError:
        return error_class(message, context=ctx, http_status=status)
    return error_class(message, context=ctx)


def normalize_provider_failure(
    *,
    status: int | None,
    message: str,
    provider: str | None = None,
    model: str | None = None,
    operation: str | None = None,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> NormalizedFailure:
    normalized_status = int(status) if status is not None else None
    if normalized_status == 408:
        return NormalizedFailure(
            code=ErrorCode.PROVIDER_TIMEOUT.value,
            category=FailureCategory.TIMEOUT,
            message=message,
            retryable=True,
            status=408,
            provider=provider,
            model=model,
            operation=operation,
            request_id=request_id,
            remediation=_remediation_hint_for_error_code(ErrorCode.PROVIDER_TIMEOUT),
            details=dict(details or {}),
            cause_type="ProviderTimeoutError",
        )
    context = ErrorContext(
        request_id=request_id,
        provider=provider,
        model=model,
        operation=operation,
        extra=dict(details or {}),
    )
    if normalized_status is not None:
        error = error_from_status(normalized_status, message, provider=provider, context=context)
        return error.to_failure()
    fallback = ProviderError(message, context=context, retryable=False)
    return fallback.to_failure()


def normalize_exception(
    error: Exception,
    *,
    provider: str | None = None,
    model: str | None = None,
    operation: str | None = None,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> NormalizedFailure:
    if isinstance(error, LLMClientError):
        return error.to_failure()

    context = ErrorContext(
        request_id=request_id,
        provider=provider,
        model=model,
        operation=operation,
        extra=dict(details or {}),
    )

    import asyncio

    if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return ProviderTimeoutError(context=context, cause=error).to_failure()
    if isinstance(error, ConnectionError):
        return ProviderUnavailableError(message=str(error) or "Connection error", context=context, cause=error).to_failure()

    status = _extract_status_code(error)
    message = str(error) or type(error).__name__
    if status is not None:
        failure = normalize_provider_failure(
            status=status,
            message=message,
            provider=provider,
            model=model,
            operation=operation,
            request_id=request_id,
            details=details,
        )
        return NormalizedFailure(
            code=failure.code,
            category=failure.category,
            message=failure.message,
            retryable=failure.retryable,
            status=failure.status,
            provider=failure.provider,
            model=failure.model,
            operation=failure.operation,
            request_id=failure.request_id,
            remediation=failure.remediation,
            details=dict(failure.details),
            cause_type=type(error).__name__,
        )

    generic = LLMClientError(
        message,
        code=ErrorCode.INTERNAL_ERROR,
        retryable=False,
        context=context,
        cause=error,
    )
    return generic.to_failure()


def normalize_tool_failure(
    error: Any,
    *,
    provider: str | None = None,
    model: str | None = None,
    operation: str | None = None,
    request_id: str | None = None,
) -> NormalizedFailure:
    code = str(getattr(error, "code", "ERR_4000") or "ERR_4000")
    message = str(getattr(error, "message", error) or "Tool execution failed")
    category_name = str(getattr(error, "category", "") or "")
    retryable = bool(getattr(error, "retryable", False))
    details = dict(getattr(error, "details", {}) or {})
    category = FailureCategory.TOOL_POLICY if "policy" in category_name else FailureCategory.TOOL
    remediation = "Fix the tool declaration or tool-call arguments."
    if retryable:
        remediation = "Retry the tool execution or allow the model to repair the tool call."
    return NormalizedFailure(
        code=code,
        category=category,
        message=message,
        retryable=retryable,
        provider=provider,
        model=model,
        operation=operation,
        request_id=request_id,
        remediation=remediation,
        details=details,
        cause_type=type(error).__name__,
    )


def normalize_structured_failure(
    error: Any,
    *,
    provider: str | None = None,
    model: str | None = None,
    operation: str | None = None,
    request_id: str | None = None,
    validation_errors: list[str] | None = None,
) -> NormalizedFailure:
    code = str(getattr(error, "code", "structured_output_failed") or "structured_output_failed")
    message = str(getattr(error, "message", error) or "Structured output failed")
    retryable = bool(getattr(error, "retryable", False))
    details = dict(getattr(error, "details", {}) or {})
    if validation_errors:
        details.setdefault("validation_errors", list(validation_errors))
    remediation = "Repair the response to match the schema or simplify the schema/prompt."
    return NormalizedFailure(
        code=code,
        category=FailureCategory.STRUCTURED_OUTPUT,
        message=message,
        retryable=retryable,
        provider=provider,
        model=model,
        operation=operation,
        request_id=request_id,
        remediation=remediation,
        details=details,
        cause_type=type(error).__name__,
    )


def failure_to_completion_result(failure: NormalizedFailure, *, model: str | None = None) -> Any:
    from .providers.types import CompletionResult

    details = failure.to_dict()
    status = failure.status if failure.status is not None else _default_status_for_failure_category(failure.category)
    return CompletionResult(
        status=status,
        error=failure.message,
        model=model or failure.model or "unknown",
        raw_response={"normalized_failure": details},
    )


def failure_to_embedding_result(failure: NormalizedFailure, *, model: str | None = None) -> Any:
    from .providers.types import EmbeddingResult

    return EmbeddingResult(
        embeddings=[],
        status=failure.status if failure.status is not None else _default_status_for_failure_category(failure.category),
        error=failure.message,
        model=model or failure.model or "unknown",
        raw_response={"normalized_failure": failure.to_dict()},
    )


def failure_to_stream_error_data(failure: NormalizedFailure) -> dict[str, Any]:
    return {
        "status": failure.status if failure.status is not None else _default_status_for_failure_category(failure.category),
        "error": failure.message,
        "normalized_failure": failure.to_dict(),
    }


def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if the error is retryable
    """
    if isinstance(error, LLMClientError):
        return error.retryable

    # Check for common retryable exceptions
    import asyncio

    retryable_types = (
        asyncio.TimeoutError,
        ConnectionError,
        TimeoutError,
    )
    return isinstance(error, retryable_types)


def _category_for_error_code(code: ErrorCode) -> FailureCategory:
    mapping = {
        ErrorCode.RATE_LIMIT: FailureCategory.RATE_LIMIT,
        ErrorCode.AUTHENTICATION: FailureCategory.AUTHENTICATION,
        ErrorCode.QUOTA_EXCEEDED: FailureCategory.QUOTA,
        ErrorCode.MODEL_NOT_FOUND: FailureCategory.MODEL,
        ErrorCode.CONTEXT_LENGTH: FailureCategory.REQUEST,
        ErrorCode.CONTENT_FILTER: FailureCategory.CONTENT_FILTER,
        ErrorCode.PROVIDER_UNAVAILABLE: FailureCategory.AVAILABILITY,
        ErrorCode.PROVIDER_TIMEOUT: FailureCategory.TIMEOUT,
        ErrorCode.INVALID_RESPONSE: FailureCategory.PROVIDER,
        ErrorCode.VALIDATION_ERROR: FailureCategory.VALIDATION,
        ErrorCode.INVALID_MESSAGE: FailureCategory.VALIDATION,
        ErrorCode.INVALID_TOOL: FailureCategory.TOOL,
        ErrorCode.MESSAGE_TOO_LONG: FailureCategory.REQUEST,
        ErrorCode.TOO_MANY_MESSAGES: FailureCategory.REQUEST,
        ErrorCode.INVALID_SCHEMA: FailureCategory.STRUCTURED_OUTPUT,
        ErrorCode.TOOL_ERROR: FailureCategory.TOOL,
        ErrorCode.TOOL_NOT_FOUND: FailureCategory.TOOL,
        ErrorCode.TOOL_EXECUTION_ERROR: FailureCategory.TOOL,
        ErrorCode.TOOL_TIMEOUT: FailureCategory.TOOL,
        ErrorCode.TOOL_VALIDATION_ERROR: FailureCategory.TOOL,
        ErrorCode.CONFIG_ERROR: FailureCategory.CONFIGURATION,
        ErrorCode.MISSING_API_KEY: FailureCategory.CONFIGURATION,
        ErrorCode.INVALID_CONFIG: FailureCategory.CONFIGURATION,
        ErrorCode.INTERNAL_ERROR: FailureCategory.INTERNAL,
        ErrorCode.UNKNOWN_ERROR: FailureCategory.UNKNOWN,
    }
    return mapping.get(code, FailureCategory.PROVIDER)


def _remediation_hint_for_error_code(code: ErrorCode) -> str | None:
    mapping = {
        ErrorCode.RATE_LIMIT: "Retry after backoff or lower concurrency.",
        ErrorCode.AUTHENTICATION: "Check API credentials and provider configuration.",
        ErrorCode.QUOTA_EXCEEDED: "Increase quota or switch to another provider/model.",
        ErrorCode.MODEL_NOT_FOUND: "Use a supported model key for the selected provider.",
        ErrorCode.CONTEXT_LENGTH: "Trim or summarize input context.",
        ErrorCode.CONTENT_FILTER: "Adjust the prompt or content to satisfy safety filters.",
        ErrorCode.PROVIDER_UNAVAILABLE: "Retry later or fail over to another provider.",
        ErrorCode.PROVIDER_TIMEOUT: "Retry with a longer timeout or a faster provider.",
        ErrorCode.INVALID_RESPONSE: "Retry or inspect provider compatibility and response parsing.",
        ErrorCode.INVALID_SCHEMA: "Repair the schema or response to satisfy structured-output validation.",
        ErrorCode.TOOL_VALIDATION_ERROR: "Repair the tool-call arguments before retrying.",
        ErrorCode.TOOL_TIMEOUT: "Increase tool timeout or reduce tool workload.",
        ErrorCode.MISSING_API_KEY: "Set the required provider API key.",
        ErrorCode.INVALID_CONFIG: "Fix the llm-client configuration before retrying.",
    }
    return mapping.get(code)


def _extract_status_code(error: Exception) -> int | None:
    for attr in ("status", "status_code", "http_status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    response = getattr(error, "response", None)
    for attr in ("status_code", "status"):
        value = getattr(response, attr, None)
        if isinstance(value, int):
            return value
    return None


def _error_from_bad_request_message(message: str, *, context: ErrorContext) -> ProviderError | None:
    normalized = message.lower()
    if any(needle in normalized for needle in ("content filter", "safety filter", "blocked by safety", "content policy")):
        return ContentFilterError(message, context=context)
    if any(
        needle in normalized
        for needle in (
            "context length",
            "maximum context length",
            "too many tokens",
            "prompt is too long",
            "context window",
            "input is too long",
        )
    ):
        return ContextLengthError(message, context=context)
    if any(needle in normalized for needle in ("invalid request", "bad request", "malformed", "invalid input", "validation")):
        return ValidationError(message, context=context, code=ErrorCode.VALIDATION_ERROR)
    return None


def _error_from_forbidden_message(message: str, *, context: ErrorContext) -> ProviderError | None:
    normalized = message.lower()
    if any(needle in normalized for needle in ("content filter", "safety filter", "blocked by safety", "content policy")):
        return ContentFilterError(message, context=context)
    return AuthenticationError(message, context=context)


def _default_status_for_failure_category(category: FailureCategory) -> int:
    mapping = {
        FailureCategory.AUTHENTICATION: 401,
        FailureCategory.AUTHORIZATION: 403,
        FailureCategory.QUOTA: 402,
        FailureCategory.RATE_LIMIT: 429,
        FailureCategory.MODEL: 404,
        FailureCategory.TIMEOUT: 408,
        FailureCategory.AVAILABILITY: 503,
        FailureCategory.CONTENT_FILTER: 400,
        FailureCategory.REQUEST: 400,
        FailureCategory.VALIDATION: 400,
        FailureCategory.TOOL: 400,
        FailureCategory.TOOL_POLICY: 403,
        FailureCategory.STRUCTURED_OUTPUT: 400,
        FailureCategory.CONFIGURATION: 500,
        FailureCategory.INTERNAL: 500,
        FailureCategory.UNKNOWN: 500,
        FailureCategory.PROVIDER: 500,
    }
    return mapping.get(category, 500)


__all__ = [
    # Base
    "ErrorCode",
    "ErrorContext",
    "FailureCategory",
    "NormalizedFailure",
    "LLMClientError",
    # Provider errors
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "QuotaExceededError",
    "ModelNotFoundError",
    "ContextLengthError",
    "ContentFilterError",
    "ProviderUnavailableError",
    "ProviderTimeoutError",
    "InvalidResponseError",
    # Validation errors
    "ValidationError",
    "InvalidMessageError",
    "InvalidToolError",
    "MessageTooLongError",
    "TooManyMessagesError",
    "InvalidSchemaError",
    # Cache errors
    "CacheError",
    "CacheReadError",
    "CacheWriteError",
    "CacheConnectionError",
    # Tool errors
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolValidationError",
    # Agent errors
    "AgentError",
    "MaxTurnsExceededError",
    "AgentTimeoutError",
    # Config errors
    "ConfigError",
    "MissingAPIKeyError",
    "InvalidConfigError",
    # Utilities
    "error_from_status",
    "normalize_provider_failure",
    "normalize_exception",
    "normalize_tool_failure",
    "normalize_structured_failure",
    "failure_to_completion_result",
    "failure_to_embedding_result",
    "failure_to_stream_error_data",
    "is_retryable",
]
