"""
Provider protocol and base classes.

This module defines the abstract interface that all LLM providers must implement,
enabling provider-agnostic agent and application code.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .types import (
    CompletionResult,
    EmbeddingResult,
    Message,
    MessageInput,
    StreamEvent,
    Usage,
    normalize_messages,
)

T = TypeVar("T")

if TYPE_CHECKING:
    from ..models import ModelProfile
    from ..tools.base import Tool


@runtime_checkable
class Provider(Protocol):
    """
    Protocol defining the interface for LLM providers.

    All providers must implement these core methods to be compatible
    with the agent framework.
    """

    @property
    def model(self) -> type[ModelProfile]:
        """Get the model profile for this provider."""
        ...

    @property
    def model_name(self) -> str:
        """Get the model name string."""
        ...

    async def complete(
        self,
        messages: MessageInput,
        *,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a completion for the given messages.

        Args:
            messages: Input messages (str, dict, Message, or list of these)
            tools: Optional list of tools the model can call
            tool_choice: How to handle tool selection ("auto", "none", "required", or specific tool)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            response_format: Output format ("text", "json_object", or JSON schema)
            **kwargs: Provider-specific parameters

        Returns:
            CompletionResult with the model's response
        """
        ...

    def stream(
        self,
        messages: MessageInput,
        *,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream a completion as a series of events.

        Args:
            messages: Input messages
            tools: Optional list of tools
            tool_choice: Tool selection mode
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Provider-specific parameters

        Yields:
            StreamEvent objects for tokens, tool calls, usage, etc.
        """
        ...

    async def embed(
        self,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings for the given inputs.

        Args:
            inputs: Text or list of texts to embed
            **kwargs: Provider-specific parameters

        Returns:
            EmbeddingResult with embedding vectors
        """
        ...

    def count_tokens(self, content: Any) -> int:
        """Count tokens in the given content."""
        ...

    def parse_usage(self, raw_usage: dict[str, Any]) -> Usage:
        """Parse raw API usage into Usage object."""
        ...

    async def close(self) -> None:
        """Clean up provider resources."""
        ...

    async def __aenter__(self) -> Provider:
        """Async context manager entry."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        ...


class BaseProvider(Provider, ABC):
    """
    Abstract base class for provider implementations.

    Provides common functionality and default implementations while
    requiring subclasses to implement provider-specific methods.
    """

    def __init__(
        self,
        model: type[ModelProfile] | str,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the provider.

        Args:
            model: ModelProfile class or model key string
            **kwargs: Provider-specific configuration
        """
        from ..models import ModelProfile

        if isinstance(model, str):
            self._model = ModelProfile.get(model)
        elif isinstance(model, type) and issubclass(model, ModelProfile):
            self._model = model
        else:
            raise ValueError("Model must be a ModelProfile subclass or a model key string")

    @property
    def model(self) -> type[ModelProfile]:
        """Get the model profile."""
        return self._model

    @property
    def model_name(self) -> str:
        """Get the model name string."""
        return self._model.model_name

    def count_tokens(self, content: Any) -> int:
        """Count tokens in the given content."""
        return self._model.count_tokens(content)

    def parse_usage(self, raw_usage: dict[str, Any]) -> Usage:
        """Parse raw API usage into Usage object."""
        parsed = self._model.parse_usage(raw_usage)
        return Usage(
            input_tokens=int(parsed.get("input_tokens", 0) or 0),
            output_tokens=int(parsed.get("output_tokens", 0) or 0),
            total_tokens=int(parsed.get("total_tokens", 0) or 0),
            input_tokens_cached=int(parsed.get("input_tokens_cached", 0) or 0),
            input_cost=float(parsed.get("input_cost", 0.0) or 0.0),
            output_cost=float(parsed.get("output_cost", 0.0) or 0.0),
            total_cost=float(parsed.get("total_cost", 0.0) or 0.0),
        )

    @staticmethod
    def _normalize_messages(messages: MessageInput) -> list[Message]:
        """Normalize message input to list of Message objects."""
        return normalize_messages(messages)

    @staticmethod
    def _messages_to_api_format(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to API-compatible format."""
        return [msg.to_dict() for msg in messages]

    @staticmethod
    def _tools_to_api_format(tools: list[Tool] | None) -> list[dict[str, Any]] | None:
        """Convert Tool objects to API-compatible format."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @staticmethod
    async def _with_retry(
        operation: Callable[[], Any],
        *,
        attempts: int = 3,
        backoff: float = 1.0,
        retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
    ) -> Any:
        """
        Execute an operation with retry, exponential backoff, and Retry-After support.

        Args:
            operation: Async callable to execute
            attempts: Maximum number of attempts
            backoff: Initial backoff delay in seconds (doubles each retry)
            retryable_statuses: HTTP status codes that should trigger retry

        Returns:
            Result of the operation

        Note:
            The operation should return a result with a 'status' attribute
            or raise an exception on failure.
        """
        last_result = None
        last_error = None
        current_backoff = backoff

        for attempt in range(attempts):
            try:
                result = await operation()

                # Check if result indicates a retryable error
                if hasattr(result, "status"):
                    if result.status in retryable_statuses and attempt < attempts - 1:
                        last_result = result

                        # Check response headers for Retry-After (if available)
                        wait_time = current_backoff * random.uniform(0.8, 1.2)

                        # Try to extract headers from raw_response if present
                        if hasattr(result, "raw_response") and hasattr(result.raw_response, "headers"):
                            retry_after = result.raw_response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    wait_time = float(retry_after)
                                except (ValueError, TypeError):
                                    # Could be HTTP-date, ignoring for simplicity and falling back to backoff
                                    pass

                        await asyncio.sleep(wait_time)
                        current_backoff *= 2
                        continue

                return result

            except Exception as e:
                last_error = e
                # Check for OpenAI-style RateLimitError with retry_after
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            # If explicitly told to wait, ensure we do
                            wait_time = float(retry_after)
                            if attempt < attempts - 1:
                                await asyncio.sleep(wait_time)
                                # Don't increment backoff if we obeyed the server
                                continue
                        except ValueError:
                            pass

                if attempt < attempts - 1:
                    await asyncio.sleep(current_backoff * random.uniform(0.8, 1.2))
                    current_backoff *= 2
                    continue
                raise

        # Return last result if we have one, otherwise raise last error
        if last_result is not None:
            return last_result
        if last_error is not None:
            raise last_error

        raise RuntimeError("Retry logic failed unexpectedly")

    @abstractmethod
    async def complete(
        self,
        messages: MessageInput,
        *,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Generate a completion. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def stream(
        self,
        messages: MessageInput,
        *,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion. Must be implemented by subclasses."""
        ...

    async def complete_structured(
        self,
        messages: MessageInput,
        schema: dict[str, Any],
        *,
        max_repair_attempts: int = 2,
        **kwargs: Any,
    ) -> Any:  # Returns StructuredResult
        """
        Complete with structured output validation.
        
        Automatically validates output against the schema and attempts
        repair if validation fails.
        
        Args:
            messages: Input messages
            schema: JSON schema for the expected output
            max_repair_attempts: Max repair attempts (default: 2)
            **kwargs: Additional arguments for complete()
        
        Returns:
            StructuredResult with validated data
        """
        from ..structured import extract_structured, StructuredOutputConfig
        
        normalized = self._normalize_messages(messages)
        config = StructuredOutputConfig(
            schema=schema,
            max_repair_attempts=max_repair_attempts,
        )
        return await extract_structured(self, normalized, config, **kwargs)

    async def embed(
        self,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings.

        Default implementation raises NotImplementedError.
        Override in subclasses that support embeddings.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    async def close(self) -> None:
        """
        Clean up provider resources.

        Override in subclasses that need cleanup.
        """
        pass

    async def __aenter__(self) -> Provider:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


__all__ = [
    "Provider",
    "BaseProvider",
]
