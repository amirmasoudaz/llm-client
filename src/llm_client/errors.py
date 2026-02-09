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
) -> ProviderError:
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

    error_map: dict[int, type[ProviderError]] = {
        400: InvalidResponseError,  # Bad request - could be many things
        401: AuthenticationError,
        402: QuotaExceededError,
        403: AuthenticationError,  # Forbidden usually means auth issue
        404: ModelNotFoundError,
        429: RateLimitError,
        500: InvalidResponseError,
        502: ProviderUnavailableError,
        503: ProviderUnavailableError,
        504: ProviderTimeoutError,
    }

    error_class = error_map.get(status, ProviderError)
    return error_class(message, context=ctx)


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


__all__ = [
    # Base
    "ErrorCode",
    "ErrorContext",
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
    "is_retryable",
]
