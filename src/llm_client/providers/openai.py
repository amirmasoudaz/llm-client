"""
OpenAI provider implementation.

This module implements the Provider protocol for OpenAI's API,
supporting both chat completions and embeddings.
"""

from __future__ import annotations

import base64
import inspect
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np
import openai
from openai import AsyncOpenAI

from ..cache import CacheSettings, build_cache_core
from ..cache.serializers import cache_dict_to_result, result_to_cache_dict
from ..hashing import cache_key as compute_cache_key
from ..rate_limit import Limiter
from .base import BaseProvider
from .types import (
    CompletionResult,
    EmbeddingResult,
    MessageInput,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallDelta,
    Usage,
)

if TYPE_CHECKING:
    from ..models import ModelProfile
    from ..tools.base import Tool


class OpenAIProvider(BaseProvider):
    """
    OpenAI API provider implementation.

    Supports:
    - Chat completions (with and without streaming)
    - Tool/function calling
    - Reasoning models with configurable effort
    - Embeddings
    - Response caching
    - Rate limiting

    Example:
        ```python
        provider = OpenAIProvider(model="gpt-5")
        result = await provider.complete("Hello, world!")
        print(result.content)
        ```
    """

    def __init__(
        self,
        model: type[ModelProfile] | str,
        *,
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
        # API settings
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        # Other settings
        use_responses_api: bool = False,
    ) -> None:
        """
        Initialize the OpenAI provider.

        Args:
            model: ModelProfile class or model key string
            cache_dir: Directory for file-based caching
            cache_backend: Cache backend type ("fs", "qdrant", "pg_redis", or None)
            cache_collection: Collection/table name for caching
            pg_dsn: PostgreSQL connection string
            redis_url: Redis connection URL
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            redis_ttl_seconds: Redis TTL for cached items
            compress_pg: Whether to compress PostgreSQL cache entries
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL
            organization: OpenAI organization ID
            use_responses_api: Use the responses API instead of chat completions
        """
        super().__init__(model)

        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        self.cache_dir = cache_dir
        self.use_responses_api = use_responses_api
        self.default_cache_collection = cache_collection

        if self.cache_dir and cache_backend == "fs":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize rate limiter
        self.limiter = Limiter(self._model)

        # Initialize OpenAI client
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization

        self.client = AsyncOpenAI(**client_kwargs)

        # Initialize cache
        backend_name = cache_backend or "none"
        self.cache = build_cache_core(
            CacheSettings(
                backend=backend_name,
                client_type=self._model.category,
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

    async def close(self) -> None:
        """Close provider resources."""
        await self.cache.close()
        close_fn = getattr(self.client, "close", None)
        if close_fn:
            res = close_fn()
            if inspect.isawaitable(res):
                await res

    def _check_reasoning_params(
        self,
        params: dict[str, Any],
        api_type: Literal["completions", "responses"],
    ) -> dict[str, Any]:
        """Validate and normalize reasoning parameters."""
        has_reasoning = "reasoning" in params
        has_reasoning_effort = "reasoning_effort" in params

        if not (has_reasoning or has_reasoning_effort):
            return params

        if not self._model.reasoning_model:
            raise ValueError("Model does not support reasoning, but reasoning parameters were provided.")

        effort = None

        if has_reasoning:
            reasoning_val = params["reasoning"]
            if not isinstance(reasoning_val, dict):
                raise ValueError("`reasoning` must be an object like {'effort': '<level>'}.")
            effort = reasoning_val.get("effort")

        if has_reasoning_effort:
            reff = params.get("reasoning_effort")
            if effort is not None and reff is not None and reff != effort:
                raise ValueError("Provide only one of `reasoning` or `reasoning_effort`, or ensure they match.")
            effort = reff if effort is None else effort

        if effort not in self._model.reasoning_efforts:
            raise ValueError(f"Invalid reasoning effort. Choose from: {self._model.reasoning_efforts}")

        if api_type == "responses":
            params.pop("reasoning_effort", None)
            params["reasoning"] = {"effort": effort}
        elif api_type == "completions":
            params.pop("reasoning", None)
            params["reasoning_effort"] = effort

        return params

    @staticmethod
    def _normalize_response_format(
        response_format: str | dict[str, Any] | type | None,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any] | type | None:
        """Normalize response format parameter."""
        if response_format is None:
            return {"type": "text"}

        if response_format == "text":
            return {"type": "text"}

        if response_format == "json_object":
            # Validate that 'json' appears in the context
            if "json" not in str(messages).lower():
                raise ValueError("Context doesn't contain 'json' keyword which is required for JSON mode.")
            return {"type": "json_object"}

        if isinstance(response_format, dict):
            return response_format

        # Assume it's a Pydantic model or similar for structured output
        if isinstance(response_format, str):
            raise ValueError(f"Unsupported response_format: {response_format!r}")
        return response_format

    @staticmethod
    def _cache_key(api: str, params: dict[str, Any]) -> str:
        return compute_cache_key(api, params)

    async def complete(
        self,
        messages: MessageInput,
        *,
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        cache_response: bool = False,
        cache_collection: str | None = None,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        attempts: int = 3,
        backoff: float = 1.0,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a completion.

        Args:
            messages: Input messages
            tools: Available tools for the model
            tool_choice: Tool selection mode
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            response_format: Output format
            reasoning_effort: Reasoning effort level (for reasoning models)
            reasoning: Reasoning configuration dict
            cache_response: Whether to cache the response
            cache_collection: Cache collection name
            rewrite_cache: Create new cache entry even if one exists
            regen_cache: Regenerate cache (ignore existing)
            attempts: Number of retry attempts for transient errors
            backoff: Initial backoff delay in seconds (doubles each retry)
            **kwargs: Additional API parameters

        Returns:
            CompletionResult with the model's response
        """
        # Normalize messages
        msg_objects = self._normalize_messages(messages)
        api_messages = self._messages_to_api_format(msg_objects)

        # Build params
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
        }

        # Add optional parameters
        if tools:
            params["tools"] = self._tools_to_api_format(tools)
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = tool_choice
                else:
                    params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Handle response format
        rf = self._normalize_response_format(response_format, api_messages)
        if rf:
            params["response_format"] = rf

        # Handle reasoning params
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        if reasoning:
            params["reasoning"] = reasoning
        params = self._check_reasoning_params(params, "completions")

        # Add any extra kwargs
        params.update(kwargs)

        # Use responses API if enabled
        if self.use_responses_api:
            return await self._complete_responses(
                api_messages,
                params,
                cache_response=cache_response,
                cache_collection=cache_collection,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                attempts=attempts,
                backoff=backoff,
            )

        # Check cache
        if cache_response:
            identifier = self._cache_key("chat.completions", params)

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

        # Make the request with rate limiting and retry
        input_tokens = self.count_tokens(api_messages)

        async def _do_completion() -> CompletionResult:
            async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_ctx:
                try:
                    if isinstance(rf, dict):
                        response = await self.client.chat.completions.create(**params)
                        content = response.choices[0].message.content

                        # Parse JSON if requested
                        if rf.get("type") in ("json_object", "json_schema"):
                            try:
                                content = json.loads(content)
                            except json.JSONDecodeError:
                                pass
                    else:
                        # Structured output with Pydantic model
                        response = await self.client.beta.chat.completions.parse(**params)
                        parsed = response.choices[0].message.parsed
                        if parsed is None:
                            content = None
                        elif hasattr(parsed, "model_dump"):
                            content = parsed.model_dump()
                        elif hasattr(parsed, "dict"):
                            content = parsed.dict()
                        else:
                            content = parsed

                    # Extract tool calls
                    tool_calls = None
                    msg = response.choices[0].message
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_calls = [
                            ToolCall(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            )
                            for tc in msg.tool_calls
                        ]

                    # Parse usage
                    usage = self.parse_usage(response.usage.to_dict())
                    limit_ctx.output_tokens = usage.output_tokens

                    return CompletionResult(
                        content=content if isinstance(content, str) else json.dumps(content),
                        tool_calls=tool_calls,
                        usage=usage,
                        model=self.model_name,
                        finish_reason=response.choices[0].finish_reason,
                        status=200,
                        raw_response=response,
                    )

                except openai.APIConnectionError as e:
                    return CompletionResult(
                        status=500,
                        error=str(e.__cause__),
                    )
                except openai.RateLimitError as e:
                    return CompletionResult(
                        status=429,
                        error=f"Rate limit exceeded: {e}",
                    )
                except openai.APIStatusError as e:
                    return CompletionResult(
                        status=e.status_code,
                        error=str(e),
                    )

        result = await self._with_retry(_do_completion, attempts=attempts, backoff=backoff)

        # Cache successful responses
        if cache_response and identifier and result.ok:
            effective_collection = cache_collection or self.default_cache_collection
            await self.cache.put_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                response=self._result_to_cache(result, params),
                model_name=self.model_name,
                log_errors=True,
                collection=effective_collection,
            )

        return result

    async def _complete_responses(
        self,
        api_messages: list[dict[str, Any]],
        params: dict[str, Any],
        *,
        cache_response: bool = False,
        cache_collection: str | None = None,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        attempts: int = 3,
        backoff: float = 1.0,
    ) -> CompletionResult:
        """
        Complete using the OpenAI Responses API.

        The Responses API is a newer OpenAI endpoint that supports
        extended features like file inputs and different output formats.
        """
        # Convert messages format for responses API
        # Responses API uses 'input' instead of 'messages'
        responses_params: dict[str, Any] = {
            "model": params["model"],
            "input": api_messages,
        }

        # Copy over reasoning params if present
        if "reasoning_effort" in params:
            responses_params["reasoning_effort"] = params["reasoning_effort"]
        if "reasoning" in params:
            responses_params["reasoning"] = params["reasoning"]

        # Validate params
        responses_params = self._check_reasoning_params(responses_params, "responses")

        # Check cache
        if cache_response:
            identifier = self._cache_key("responses", responses_params)

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

        input_tokens = self.count_tokens(api_messages)

        async def _do_responses() -> CompletionResult:
            async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_ctx:
                try:
                    response = await self.client.responses.create(**responses_params)

                    # Extract output text
                    content = response.output_text

                    # Parse usage
                    raw_usage = response.usage.to_dict() if response.usage else {}
                    usage = self.parse_usage(raw_usage)
                    limit_ctx.output_tokens = usage.output_tokens

                    return CompletionResult(
                        content=content,
                        usage=usage,
                        model=self.model_name,
                        status=200,
                        raw_response=response,
                    )

                except openai.APIConnectionError as e:
                    return CompletionResult(
                        status=500,
                        error=str(e.__cause__),
                    )
                except openai.RateLimitError as e:
                    return CompletionResult(
                        status=429,
                        error=f"Rate limit exceeded: {e}",
                    )
                except openai.APIStatusError as e:
                    return CompletionResult(
                        status=e.status_code,
                        error=str(e),
                    )

        result = await self._with_retry(_do_responses, attempts=attempts, backoff=backoff)

        # Cache successful responses
        if cache_response and identifier and result.ok:
            effective_collection = cache_collection or self.default_cache_collection
            await self.cache.put_cached(
                identifier,
                rewrite_cache=rewrite_cache,
                regen_cache=regen_cache,
                response=self._result_to_cache(result, responses_params),
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
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream a completion as events.

        Args:
            messages: Input messages
            tools: Available tools
            tool_choice: Tool selection mode
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            reasoning_effort: Reasoning effort level
            reasoning: Reasoning configuration
            **kwargs: Additional API parameters

        Yields:
            StreamEvent objects for each chunk
        """
        # Normalize messages
        msg_objects = self._normalize_messages(messages)
        api_messages = self._messages_to_api_format(msg_objects)

        # Build params
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        # Add optional parameters
        if tools:
            params["tools"] = self._tools_to_api_format(tools)
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = tool_choice
                else:
                    params["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        # Handle reasoning params
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        if reasoning:
            params["reasoning"] = reasoning
        params = self._check_reasoning_params(params, "completions")

        params.update(kwargs)

        # Emit metadata event
        yield StreamEvent(type=StreamEventType.META, data={"model": self.model_name, "stream": True})

        # Track accumulated content and tool calls
        content_buffer = ""
        reasoning_buffer = ""
        tool_calls_buffer: dict[int, dict[str, Any]] = {}  # index -> {id, name, arguments}
        usage = None

        input_tokens = self.count_tokens(api_messages)

        async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_ctx:
            try:
                stream = await self.client.chat.completions.create(**params)

                async for chunk in stream:
                    # Handle usage in final chunk
                    if not chunk.choices and chunk.usage:
                        raw_usage = chunk.usage.to_dict()
                        if hasattr(chunk.usage, "completion_tokens_details"):
                            raw_usage["completion_tokens_details"] = (
                                chunk.usage.completion_tokens_details.dict()
                                if hasattr(chunk.usage.completion_tokens_details, "dict")
                                else {}
                            )
                        if hasattr(chunk.usage, "prompt_tokens_details"):
                            raw_usage["prompt_tokens_details"] = (
                                chunk.usage.prompt_tokens_details.dict()
                                if hasattr(chunk.usage.prompt_tokens_details, "dict")
                                else {}
                            )
                        usage = self.parse_usage(raw_usage)
                        limit_ctx.output_tokens = usage.output_tokens

                        yield StreamEvent(type=StreamEventType.USAGE, data=usage)
                        continue

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # Handle reasoning tokens (for o1/GPT-5 reasoning models)
                    # OpenAI emits reasoning content via delta.reasoning_content
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_buffer += delta.reasoning_content
                        yield StreamEvent(type=StreamEventType.REASONING, data=delta.reasoning_content)

                    # Handle content tokens
                    if delta.content:
                        content_buffer += delta.content
                        yield StreamEvent(type=StreamEventType.TOKEN, data=delta.content)

                    # Handle tool call deltas
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index

                            if idx not in tool_calls_buffer:
                                # New tool call starting
                                tool_calls_buffer[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": tc_delta.function.name if tc_delta.function else "",
                                    "arguments": "",
                                }

                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    data=ToolCallDelta(
                                        id=tool_calls_buffer[idx]["id"],
                                        index=idx,
                                        name=tool_calls_buffer[idx]["name"],
                                    ),
                                )

                            # Update with delta
                            if tc_delta.id:
                                tool_calls_buffer[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_buffer[idx]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_buffer[idx]["arguments"] += tc_delta.function.arguments

                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_DELTA,
                                        data=ToolCallDelta(
                                            id=tool_calls_buffer[idx]["id"],
                                            index=idx,
                                            arguments_delta=tc_delta.function.arguments,
                                        ),
                                    )

                    # Check for finish
                    if chunk.choices[0].finish_reason:
                        # Emit tool call end events
                        for _idx, tc_data in tool_calls_buffer.items():
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_END,
                                data=ToolCall(
                                    id=tc_data["id"],
                                    name=tc_data["name"],
                                    arguments=tc_data["arguments"],
                                ),
                            )

                # Build final result
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

                final_result = CompletionResult(
                    content=content_buffer if content_buffer else None,
                    tool_calls=tool_calls,
                    usage=usage,
                    model=self.model_name,
                    status=200,
                    reasoning=reasoning_buffer if reasoning_buffer else None,
                )

                yield StreamEvent(type=StreamEventType.DONE, data=final_result)

            except openai.APIConnectionError as e:
                yield StreamEvent(type=StreamEventType.ERROR, data={"status": 500, "error": str(e.__cause__)})
            except openai.RateLimitError as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR, data={"status": 429, "error": f"Rate limit exceeded: {e}"}
                )
            except openai.APIStatusError as e:
                yield StreamEvent(type=StreamEventType.ERROR, data={"status": e.status_code, "error": str(e)})
            except Exception as e:
                yield StreamEvent(type=StreamEventType.ERROR, data={"status": 500, "error": str(e)})

    async def embed(
        self,
        inputs: str | list[str],
        *,
        encoding_format: Literal["float", "base64"] = "base64",
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings.

        Args:
            inputs: Text or list of texts to embed
            encoding_format: Output format ("float" or "base64")
            dimensions: Output dimensionality (if model supports)
            **kwargs: Additional API parameters

        Returns:
            EmbeddingResult with embedding vectors
        """
        if self._model.category != "embeddings":
            raise ValueError(
                f"Model {self.model_name} does not support embeddings. "
                "Use an embedding model like text-embedding-3-large."
            )

        if isinstance(inputs, str):
            inputs = [inputs]

        params: dict[str, Any] = {
            "model": self.model_name,
            "input": inputs,
            "encoding_format": encoding_format,
        }

        if dimensions is not None:
            params["dimensions"] = dimensions

        params.update(kwargs)

        input_tokens = self.count_tokens(inputs)

        async with self.limiter.limit(tokens=input_tokens, requests=1):
            try:
                response = await self.client.embeddings.create(**params)

                embeddings = [d.embedding for d in response.data]

                # Decode base64 if needed
                if encoding_format == "base64":
                    embeddings = [
                        np.frombuffer(base64.b64decode(emb), dtype=np.float32).tolist() if isinstance(emb, str) else emb
                        for emb in embeddings
                    ]

                usage = self.parse_usage(response.usage.to_dict())

                return EmbeddingResult(
                    embeddings=embeddings,
                    usage=usage,
                    model=self.model_name,
                    status=200,
                )

            except openai.APIConnectionError as e:
                return EmbeddingResult(
                    embeddings=[],
                    status=500,
                    error=str(e.__cause__),
                )
            except openai.RateLimitError as e:
                return EmbeddingResult(
                    embeddings=[],
                    status=429,
                    error=f"Rate limit exceeded: {e}",
                )
            except openai.APIStatusError as e:
                return EmbeddingResult(
                    embeddings=[],
                    status=e.status_code,
                    error=str(e.response),
                )

    # Cache serialization methods now imported from cache.serializers
    _result_to_cache = staticmethod(result_to_cache_dict)
    _cached_to_result = staticmethod(cache_dict_to_result)


__all__ = ["OpenAIProvider"]
