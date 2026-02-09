"""
Anthropic (Claude) provider implementation.

This module implements the Provider protocol for Anthropic's Claude API,
supporting chat completions with tool calling and streaming.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

from ..cache import CacheSettings, build_cache_core
from ..cache.serializers import cache_dict_to_result, result_to_cache_dict
from ..hashing import cache_key as compute_cache_key
from ..rate_limit import Limiter
from .base import BaseProvider
from .types import (
    CompletionResult,
    EmbeddingResult,
    Message,
    MessageInput,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallDelta,
    Usage,
)

if TYPE_CHECKING:
    from ..models import ModelProfile
    from ..tools.base import Tool

try:
    import anthropic
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    anthropic = None  # type: ignore[assignment]
    AsyncAnthropic = None  # type: ignore[assignment, misc]
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude API provider implementation.

    Supports:
    - Chat completions (with and without streaming)
    - Tool/function calling
    - Extended thinking (for supported models)
    - Rate limiting

    Note: Anthropic does not support embeddings natively.

    Example:
        ```python
        provider = AnthropicProvider(model="claude-3-5-sonnet")
        result = await provider.complete("Hello, world!")
        print(result.content)
        ```

    Requires:
        - anthropic package: `pip install anthropic`
        - ANTHROPIC_API_KEY environment variable or api_key parameter
    """

    # Map our Role enum to Anthropic's role strings
    ROLE_MAP = {
        Role.USER: "user",
        Role.ASSISTANT: "assistant",
        # SYSTEM is handled separately in Anthropic API
        # TOOL results use "user" role with tool_result content
    }

    def __init__(
        self,
        model: type[ModelProfile] | str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        default_temperature: float | None = None,
        max_retries: int = 2,
        timeout: float | None = None,
        # Cache settings
        cache_dir: str | Path | None = None,
        cache_backend: Literal["qdrant", "pg_redis", "fs"] | None = None,
        cache_collection: str | None = None,
        pg_dsn: str | None = None,
        redis_url: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        redis_ttl_seconds: int = 60 * 60 * 24,
        compress_pg: bool = True,
    ) -> None:
        """
        Initialize the Anthropic provider.

        Args:
            model: ModelProfile class or model key string (e.g., "claude-3-5-sonnet")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Custom API base URL
            max_tokens: Default max tokens for completions (Anthropic requires this)
            default_temperature: Default temperature for completions
            max_retries: Number of retries for transient failures (SDK built-in, default: 2)
            timeout: Request timeout in seconds (SDK default: 600s)
            cache_dir: Directory for file-based caching
            cache_backend: Cache backend type ("fs", "qdrant", "pg_redis", or None)
            cache_collection: Collection/table name for caching
            pg_dsn: PostgreSQL connection string
            redis_url: Redis connection URL
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            redis_ttl_seconds: Redis TTL for cached items
            compress_pg: Whether to compress PostgreSQL cache entries
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is not installed. Install it with: pip install anthropic")

        super().__init__(model)

        self.max_tokens = max_tokens
        self.default_temperature = default_temperature

        # Cache setup
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.default_cache_collection = cache_collection

        if self.cache_dir and cache_backend == "fs":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Anthropic client with retry and timeout config
        client_kwargs: dict[str, Any] = {
            "max_retries": max_retries,
        }

        if timeout is not None:
            client_kwargs["timeout"] = timeout

        if api_key:
            client_kwargs["api_key"] = api_key
        elif os.environ.get("ANTHROPIC_API_KEY"):
            client_kwargs["api_key"] = os.environ["ANTHROPIC_API_KEY"]

        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = AsyncAnthropic(**client_kwargs)

        # Initialize rate limiter (uses model's rate_limits if available)
        self.limiter = Limiter(self._model)

        # Initialize cache
        backend_name = cache_backend or "none"
        self.cache = build_cache_core(
            CacheSettings(
                backend=backend_name,
                client_type=self._model.category if hasattr(self._model, "category") else "chat",
                default_collection=cache_collection,
                cache_dir=self.cache_dir,
                pg_dsn=pg_dsn,
                redis_url=redis_url,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                redis_ttl_seconds=redis_ttl_seconds,
                compress=compress_pg,
            )
        )

    async def warm_cache(self) -> None:
        """Pre-warm the cache (for backends that support it)."""
        await self.cache.warm()

    @staticmethod
    def _cache_key(api: str, params: dict[str, Any]) -> str:
        """Generate a cache key from API endpoint and parameters."""
        return compute_cache_key(api, params)

    # Cache serialization methods imported from cache.serializers
    _cached_to_result = staticmethod(cache_dict_to_result)
    _result_to_cache = staticmethod(result_to_cache_dict)

    def _convert_messages_for_anthropic(
        self,
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert our Message format to Anthropic's format.

        Returns:
            Tuple of (system_message, messages_list)

        Note: Anthropic handles system message separately from messages.
        """
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Anthropic requires system as a separate parameter
                system_message = msg.content
                continue

            if msg.role == Role.TOOL:
                # Tool results in Anthropic use a special format
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content or "",
                            }
                        ],
                    }
                )
                continue

            if msg.role == Role.ASSISTANT and msg.tool_calls:
                # Assistant message with tool calls
                content_blocks: list[dict[str, Any]] = []

                if msg.content:
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": msg.content,
                        }
                    )

                for tc in msg.tool_calls:
                    # Parse arguments from JSON string
                    try:
                        parsed_args = json.loads(tc.arguments) if tc.arguments else {}
                    except json.JSONDecodeError:
                        parsed_args = {}
                    input_data: dict[str, Any] = parsed_args if isinstance(parsed_args, dict) else {}

                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": input_data,
                        }
                    )

                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )
                continue

            # Regular user/assistant message
            role = self.ROLE_MAP.get(msg.role, "user")
            anthropic_messages.append(
                {
                    "role": role,
                    "content": msg.content or "",
                }
            )

        return system_message, anthropic_messages

    @staticmethod
    def _convert_tools_for_anthropic(
        tools: list[Tool] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert our Tool format to Anthropic's format."""
        if not tools:
            return None

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    @staticmethod
    def _extract_tool_calls_from_response(
        content_blocks: list[Any],
    ) -> tuple[str | None, list[ToolCall] | None]:
        """
        Extract text content and tool calls from Anthropic response.

        Returns:
            Tuple of (text_content, tool_calls)
        """
        text_parts = []
        tool_calls = []

        for block in content_blocks:
            if hasattr(block, "type"):
                block_type = block.type
            elif isinstance(block, dict):
                block_type = block.get("type")
            else:
                continue

            if block_type == "text":
                text = block.text if hasattr(block, "text") else block.get("text", "")
                text_parts.append(text)
            elif block_type == "tool_use":
                tool_id = block.id if hasattr(block, "id") else block.get("id", "")
                tool_name = block.name if hasattr(block, "name") else block.get("name", "")
                tool_input = block.input if hasattr(block, "input") else block.get("input", {})

                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        arguments=json.dumps(tool_input) if tool_input else "{}",
                    )
                )

        text_content = "\n".join(text_parts) if text_parts else None
        return text_content, tool_calls if tool_calls else None

    @staticmethod
    def _parse_anthropic_usage(usage: Any) -> Usage:
        """Parse Anthropic usage into our Usage format."""
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)

        # Anthropic doesn't provide cost info, we'd need to calculate based on model
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    async def complete(
        self,
        messages: MessageInput,
        *,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        cache_response: bool = False,
        cache_collection: str | None = None,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a completion using Claude.

        Args:
            messages: Input messages
            tools: Available tools for the model
            tool_choice: Tool selection mode ("auto", "any", "tool" or specific tool name)
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            response_format: Not fully supported by Anthropic (JSON mode via prompting)
            cache_response: Whether to cache the response
            cache_collection: Cache collection name
            rewrite_cache: Create new cache entry even if one exists
            regen_cache: Regenerate cache (ignore existing)
            **kwargs: Additional API parameters

        Returns:
            CompletionResult with the model's response
        """
        # Normalize and convert messages
        msg_objects = self._normalize_messages(messages)
        system_message, anthropic_messages = self._convert_messages_for_anthropic(msg_objects)

        # Build params
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if system_message:
            params["system"] = system_message

        if temperature is not None:
            params["temperature"] = temperature
        elif self.default_temperature is not None:
            params["temperature"] = self.default_temperature

        # Add tools
        anthropic_tools = self._convert_tools_for_anthropic(tools)
        if anthropic_tools:
            params["tools"] = anthropic_tools

            # Handle tool_choice
            if tool_choice:
                if tool_choice == "auto":
                    params["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    # Remove tools to prevent tool use
                    params.pop("tools", None)
                elif tool_choice == "required" or tool_choice == "any":
                    params["tool_choice"] = {"type": "any"}
                else:
                    # Specific tool name
                    params["tool_choice"] = {"type": "tool", "name": tool_choice}

        # Add extra kwargs
        params.update(kwargs)

        # Build cache params dict for cache key
        cache_params = {
            "model": self.model_name,
            "messages": [str(m) for m in msg_objects],
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "response_format": response_format,
        }
        if tools:
            cache_params["tools"] = [t.name for t in tools]
        if system_message:
            cache_params["system"] = system_message

        # Check cache before making request
        if cache_response:
            identifier = self._cache_key("anthropic.messages.create", cache_params)

            effective_collection = cache_collection or self.default_cache_collection
            cached, _ = await self.cache.get_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                only_ok=True,
                collection=effective_collection,
            )
            if cached:
                return self._cached_to_result(cached)
        else:
            identifier = None

        # Count input tokens for rate limiting
        input_tokens = self.count_tokens(anthropic_messages)

        async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_ctx:
            try:
                response = await self.client.messages.create(**params)

                # Extract content and tool calls
                text_content, tool_calls = self._extract_tool_calls_from_response(response.content)

                # Parse usage
                usage = self._parse_anthropic_usage(response.usage)

                # Track output tokens for rate limiting
                limit_ctx.output_tokens = usage.output_tokens

                result = CompletionResult(
                    content=text_content,
                    tool_calls=tool_calls,
                    usage=usage,
                    model=self.model_name,
                    finish_reason=response.stop_reason,
                    status=200,
                    raw_response=response,
                )

            except anthropic.APIConnectionError as e:
                result = CompletionResult(
                    status=500,
                    error=f"Connection error: {e}",
                )
            except anthropic.RateLimitError as e:
                result = CompletionResult(
                    status=429,
                    error=f"Rate limit exceeded: {e}",
                )
            except anthropic.APIStatusError as e:
                result = CompletionResult(
                    status=e.status_code,
                    error=str(e.message),
                )
            except Exception as e:
                result = CompletionResult(
                    status=500,
                    error=str(e),
                )

        # Cache successful responses
        if cache_response and identifier and result.ok:
            effective_collection = cache_collection or self.default_cache_collection
            await self.cache.put_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                response=self._result_to_cache(result, cache_params),
                model_name=self.model_name,
                log_errors=True,
                collection=effective_collection,
            )

        return result

    async def stream(
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
        Stream a completion as events.

        Anthropic uses Server-Sent Events with different event types:
        - message_start: Initial message metadata
        - content_block_start: Start of a content block (text or tool_use)
        - content_block_delta: Content chunk
        - content_block_stop: End of content block
        - message_delta: Final message updates (stop reason, usage)
        - message_stop: Stream complete

        Args:
            messages: Input messages
            tools: Available tools
            tool_choice: Tool selection mode
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            **kwargs: Additional API parameters

        Yields:
            StreamEvent objects for each chunk
        """
        # Normalize and convert messages
        msg_objects = self._normalize_messages(messages)
        system_message, anthropic_messages = self._convert_messages_for_anthropic(msg_objects)

        # Build params
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if system_message:
            params["system"] = system_message

        if temperature is not None:
            params["temperature"] = temperature
        elif self.default_temperature is not None:
            params["temperature"] = self.default_temperature

        # Add tools
        anthropic_tools = self._convert_tools_for_anthropic(tools)
        if anthropic_tools:
            params["tools"] = anthropic_tools

            if tool_choice:
                if tool_choice == "auto":
                    params["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    params.pop("tools", None)
                elif tool_choice == "required" or tool_choice == "any":
                    params["tool_choice"] = {"type": "any"}
                else:
                    params["tool_choice"] = {"type": "tool", "name": tool_choice}

        params.update(kwargs)

        # Emit metadata event
        yield StreamEvent(
            type=StreamEventType.META, data={"model": self.model_name, "stream": True, "provider": "anthropic"}
        )

        # Track state
        content_buffer = ""
        tool_calls_buffer: dict[int, dict[str, Any]] = {}
        current_block_index = 0
        usage = None
        finish_reason = None

        # Count input tokens for rate limiting
        input_tokens = self.count_tokens(anthropic_messages)

        async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_ctx:
            try:
                async with self.client.messages.stream(**params) as stream:
                    async for raw_event in stream:
                        event = cast(Any, raw_event)
                        event_type = event.type

                        if event_type == "content_block_start":
                            block = event.content_block
                            block_type = block.type if hasattr(block, "type") else None

                            if block_type == "tool_use":
                                # Tool use block starting
                                tool_calls_buffer[current_block_index] = {
                                    "id": block.id,
                                    "name": block.name,
                                    "arguments": "",
                                }

                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    data=ToolCallDelta(
                                        id=block.id,
                                        index=current_block_index,
                                        name=block.name,
                                    ),
                                )

                            current_block_index = event.index

                        elif event_type == "content_block_delta":
                            delta = event.delta
                            delta_type = delta.type if hasattr(delta, "type") else None

                            if delta_type == "text_delta":
                                text = delta.text
                                content_buffer += text
                                yield StreamEvent(type=StreamEventType.TOKEN, data=text)

                            elif delta_type == "thinking_delta":
                                # Extended thinking content (for models that support it)
                                thinking = delta.thinking
                                yield StreamEvent(type=StreamEventType.REASONING, data=thinking)

                            elif delta_type == "input_json_delta":
                                # Tool input being streamed
                                partial_json = delta.partial_json
                                if event.index in tool_calls_buffer:
                                    tool_calls_buffer[event.index]["arguments"] += partial_json

                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_DELTA,
                                        data=ToolCallDelta(
                                            id=tool_calls_buffer[event.index]["id"],
                                            index=event.index,
                                            arguments_delta=partial_json,
                                        ),
                                    )

                        elif event_type == "content_block_stop":
                            # Check if this was a tool use block
                            if event.index in tool_calls_buffer:
                                tc_data = tool_calls_buffer[event.index]
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_END,
                                    data=ToolCall(
                                        id=tc_data["id"],
                                        name=tc_data["name"],
                                        arguments=tc_data["arguments"],
                                    ),
                                )

                        elif event_type == "message_delta":
                            # Final message updates
                            if hasattr(event, "usage") and event.usage:
                                output_tokens = event.usage.output_tokens
                                if usage:
                                    usage.output_tokens = output_tokens
                                    usage.total_tokens = usage.input_tokens + output_tokens
                                else:
                                    usage = Usage(output_tokens=output_tokens)
                                # Track output tokens for rate limiting
                                limit_ctx.output_tokens = output_tokens

                            if hasattr(event.delta, "stop_reason"):
                                finish_reason = event.delta.stop_reason

                        elif event_type == "message_start":
                            # Initial message with input token count
                            if hasattr(event.message, "usage") and event.message.usage:
                                actual_input_tokens = event.message.usage.input_tokens
                                if usage:
                                    usage.input_tokens = actual_input_tokens
                                    usage.total_tokens = actual_input_tokens + usage.output_tokens
                                else:
                                    usage = Usage(input_tokens=actual_input_tokens)

                # Build final tool calls list
                tool_calls = None
                if tool_calls_buffer:
                    tool_calls = [
                        ToolCall(
                            id=tc["id"],
                            name=tc["name"],
                            arguments=tc["arguments"],
                        )
                        for tc in tool_calls_buffer.values()
                    ]

                # Emit usage event
                if usage:
                    yield StreamEvent(type=StreamEventType.USAGE, data=usage)

                # Emit final result
                final_result = CompletionResult(
                    content=content_buffer if content_buffer else None,
                    tool_calls=tool_calls,
                    usage=usage,
                    model=self.model_name,
                    finish_reason=finish_reason,
                    status=200,
                )

                yield StreamEvent(type=StreamEventType.DONE, data=final_result)

            except anthropic.APIConnectionError as e:
                yield StreamEvent(type=StreamEventType.ERROR, data={"status": 500, "error": f"Connection error: {e}"})
            except anthropic.RateLimitError as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR, data={"status": 429, "error": f"Rate limit exceeded: {e}"}
                )
            except anthropic.APIStatusError as e:
                yield StreamEvent(type=StreamEventType.ERROR, data={"status": e.status_code, "error": str(e.message)})
            except Exception as e:
                yield StreamEvent(type=StreamEventType.ERROR, data={"status": 500, "error": str(e)})

    async def embed(
        self,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Anthropic does not support embeddings natively.

        Raises:
            NotImplementedError: Always, as Anthropic doesn't have an embeddings API
        """
        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. "
            "Consider using OpenAI's text-embedding models or a dedicated embedding service."
        )

    async def close(self) -> None:
        """
        Clean up provider resources.

        Closes the underlying AsyncAnthropic client connection and cache.
        """
        # Close cache first
        await self.cache.close()

        if hasattr(self, "client") and self.client:
            await self.client.close()

    async def count_tokens_api(
        self,
        messages: MessageInput,
        *,
        tools: list[Tool] | None = None,
    ) -> int:
        """
        Count tokens using Anthropic's token counting API.

        This is more accurate than local estimation as it uses the same
        tokenizer that would be used for the actual API call.

        Args:
            messages: Input messages to count tokens for
            tools: Optional tools to include in token count

        Returns:
            Number of input tokens

        Note:
            Token counting is free but subject to rate limits.
        """
        msg_objects = self._normalize_messages(messages)
        system_message, anthropic_messages = self._convert_messages_for_anthropic(msg_objects)

        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
        }

        if system_message:
            params["system"] = system_message

        if tools:
            anthropic_tools = self._convert_tools_for_anthropic(tools)
            if anthropic_tools:
                params["tools"] = anthropic_tools

        response = await self.client.messages.count_tokens(**params)
        return response.input_tokens


__all__ = ["AnthropicProvider", "ANTHROPIC_AVAILABLE"]
