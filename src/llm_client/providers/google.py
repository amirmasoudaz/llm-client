"""
Google Gemini provider implementation.

Uses the new google-genai SDK (replaces legacy google-generativeai).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from blake3 import blake3

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
    from ..tools.base import Tool

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None
    types = None
    genai_errors = None


class GoogleProvider(BaseProvider):
    """
    Provider for Google Gemini models via google-genai SDK.

    Setup:
        pip install llm-client[google]
        export GEMINI_API_KEY=...  (or GOOGLE_API_KEY)
    """

    def __init__(
        self,
        model: str | Any = "gemini-2.0-flash",
        api_key: str | None = None,
        base_url: str | None = None,
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
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Google Gemini provider.

        Args:
            model: Model name or ModelProfile
            api_key: API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            base_url: Custom API base URL (e.g., for proxy or gateway)
            cache_dir: Directory for file-based caching
            cache_backend: Cache backend type ("fs", "qdrant", "pg_redis", or None)
            cache_collection: Collection/table name for caching
            pg_dsn: PostgreSQL connection string
            redis_url: Redis connection URL
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            redis_ttl_seconds: Redis TTL for cached items
            compress_pg: Whether to compress PostgreSQL cache entries
            **kwargs: Additional configuration options
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-genai is not installed. Install with `pip install llm-client[google]`")
        super().__init__(model=model, **kwargs)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required")

        # Cache setup
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.default_cache_collection = cache_collection

        if self.cache_dir and cache_backend == "fs":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build client with optional http_options for custom base_url
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["http_options"] = types.HttpOptions(base_url=base_url)

        self._client = genai.Client(**client_kwargs)
        self.generation_config = kwargs.get("generation_config", {})

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

    @staticmethod
    def _convert_messages(messages: list[Message]) -> tuple[str | None, list[Any]]:
        """
        Convert messages to Gemini format.

        Returns:
            (system_instruction, history)
        """
        system_instruction = None
        history: list[Any] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Concatenate multiple system messages if present
                if system_instruction:
                    system_instruction += "\n" + (msg.content or "")
                else:
                    system_instruction = msg.content
                continue

            if msg.role == Role.USER:
                role = "user"
            elif msg.role == Role.ASSISTANT:
                role = "model"
            elif msg.role == Role.TOOL:
                # Tool/function responses are provided back to the model as "user" content.
                role = "user"
            else:
                role = "user"
            parts: list[Any] = []

            # For tool result messages, prefer function_response blocks over plain text parts.
            if msg.role != Role.TOOL and msg.content:
                parts.append(types.Part.from_text(text=msg.content))

            if msg.tool_calls:
                # Add function calls
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    parts.append(types.Part.from_function_call(name=tc.name, args=args))

            if msg.tool_call_id:
                # Function response
                tool_name = msg.name or "unknown_tool"

                # Try to parse content as JSON if it looks like it, otherwise string
                try:
                    response_content = json.loads(msg.content) if msg.content else {}
                except Exception:
                    response_content = {"result": msg.content}

                parts.append(types.Part.from_function_response(name=tool_name, response=response_content))

            if parts:
                history.append(types.Content(role=role, parts=parts))

        return system_instruction, history

    def _convert_tools(self, tools: list[Tool]) -> list[Any] | None:
        """Convert tools to Gemini format."""
        if not tools:
            return None

        function_declarations = []
        for tool in tools:
            function_declarations.append(
                types.FunctionDeclaration(
                    name=tool.name, description=tool.description, parameters_json_schema=tool.parameters
                )
            )

        return [types.Tool(function_declarations=function_declarations)]

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
        attempts: int = 3,
        backoff: float = 1.0,
        **kwargs: Any,
    ) -> CompletionResult:
        """
        Generate a completion using Google Gemini.

        Args:
            messages: Input messages
            tools: Available tools for the model
            tool_choice: Tool selection mode
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            response_format: Response format ("json_object" for JSON mode)
            cache_response: Whether to cache the response
            cache_collection: Cache collection name
            rewrite_cache: Create new cache entry even if one exists
            regen_cache: Regenerate cache (ignore existing)
            attempts: Maximum number of retry attempts (default: 3)
            backoff: Initial backoff delay in seconds (default: 1.0)
            **kwargs: Additional API parameters

        Returns:
            CompletionResult with the model's response
        """
        msgs = self._normalize_messages(messages)
        system_instruction, history = self._convert_messages(msgs)

        # Build config
        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        converted_tools = self._convert_tools(tools) if tools else None
        if converted_tools:
            config_kwargs["tools"] = converted_tools
            # Disable automatic function calling - we handle it ourselves
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

        if response_format == "json_object":
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        # Build cache params dict for cache key
        cache_params = {
            "model": self.model_name,
            "messages": [m.to_dict() if hasattr(m, "to_dict") else str(m) for m in msgs],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        if tools:
            cache_params["tools"] = [t.name for t in tools]

        # Check cache before making request
        if cache_response:
            identifier = self._cache_key("gemini.generate_content", cache_params)

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
        input_tokens = self.count_tokens(history)

        async def _make_request() -> CompletionResult:
            """Inner function to make the API request (with rate limiting and retry)."""
            async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_ctx:
                try:
                    response = await self._client.aio.models.generate_content(
                        model=self.model_name,
                        contents=history,
                        config=config,
                    )
                except genai_errors.APIError as e:
                    return CompletionResult(
                        status=e.code if hasattr(e, "code") else 500,
                        error=e.message if hasattr(e, "message") else str(e),
                    )
                except Exception as e:
                    return CompletionResult(
                        status=500,
                        error=str(e),
                    )

                # Parse response
                content = None
                tool_calls = []

                if response.parts:
                    texts = []
                    for part in response.parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
                        if hasattr(part, "function_call") and part.function_call:
                            args_source = part.function_call.args
                            args_dict = args_source if isinstance(args_source, dict) else dict(args_source)
                            # Use UUID for unique tool call IDs
                            tool_calls.append(
                                ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:16]}",
                                    name=part.function_call.name,
                                    arguments=json.dumps(args_dict),
                                )
                            )
                    if texts:
                        content = "\n".join(texts)

                # Parse usage metadata
                usage_data = Usage(total_tokens=0)
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage_data = Usage(
                        input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
                        total_tokens=getattr(response.usage_metadata, "total_token_count", 0) or 0,
                    )
                    # Track output tokens for rate limiting
                    limit_ctx.output_tokens = usage_data.output_tokens

                return CompletionResult(
                    content=content,
                    tool_calls=tool_calls or None,
                    usage=usage_data,
                    model=self.model_name,
                    status=200,
                )

        # Use retry wrapper for transient failures
        result = await self._with_retry(
            _make_request,
            attempts=attempts,
            backoff=backoff,
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
        msgs = self._normalize_messages(messages)
        system_instruction, history = self._convert_messages(msgs)

        # Build config
        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        converted_tools = self._convert_tools(tools) if tools else None
        if converted_tools:
            config_kwargs["tools"] = converted_tools
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

        config = types.GenerateContentConfig(**config_kwargs)

        # Emit META event at start
        yield StreamEvent(
            type=StreamEventType.META, data={"model": self.model_name, "stream": True, "provider": "google"}
        )

        try:
            stream_response = await self._client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=history,
                config=config,
            )

            content_buffer = ""
            tool_calls_buffer: dict[str, ToolCall] = {}

            async for chunk in stream_response:
                if chunk.candidates and chunk.candidates[0].content:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            content_buffer += part.text
                            yield StreamEvent(type=StreamEventType.TOKEN, data=part.text)
                        if hasattr(part, "function_call") and part.function_call:
                            args_source = part.function_call.args
                            args = args_source if isinstance(args_source, dict) else dict(args_source)
                            # Use UUID for unique tool call IDs
                            tool_id = f"call_{uuid.uuid4().hex[:16]}"
                            tc = ToolCall(
                                id=tool_id,
                                name=part.function_call.name,
                                arguments=json.dumps(args),
                            )
                            tool_calls_buffer[tool_id] = tc
                            tool_index = len(tool_calls_buffer) - 1

                            # Emit full sequence of tool call events
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                data=ToolCallDelta(id=tool_id, index=tool_index, name=tc.name),
                            )
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                data=ToolCallDelta(
                                    id=tool_id,
                                    index=tool_index,
                                    arguments_delta=tc.arguments,
                                ),
                            )
                            yield StreamEvent(type=StreamEventType.TOOL_CALL_END, data=tc)

                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = Usage(
                        input_tokens=getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0,
                        total_tokens=getattr(chunk.usage_metadata, "total_token_count", 0) or 0,
                    )
                    yield StreamEvent(type=StreamEventType.USAGE, data=usage)

            # Build final result
            final_tool_calls = list(tool_calls_buffer.values()) if tool_calls_buffer else None
            final_result = CompletionResult(
                content=content_buffer if content_buffer else None,
                tool_calls=final_tool_calls,
                model=self.model_name,
                status=200,
            )
            yield StreamEvent(type=StreamEventType.DONE, data=final_result)

        except genai_errors.APIError as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={
                    "status": e.code if hasattr(e, "code") else 500,
                    "error": e.message if hasattr(e, "message") else str(e),
                },
            )
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"status": 500, "error": str(e)},
            )

    async def embed(
        self,
        inputs: str | list[str],
        *,
        dimensions: int | None = None,
        model: str | None = None,
        attempts: int = 3,
        backoff: float = 1.0,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """
        Generate embeddings for the given inputs.

        Args:
            inputs: Text or list of texts to embed
            dimensions: Output dimensionality (optional, uses model default if not specified)
            model: Override embedding model (default: gemini-embedding-001)
            attempts: Maximum number of retry attempts (default: 3)
            backoff: Initial backoff delay in seconds (default: 1.0)
            **kwargs: Additional options

        Returns:
            EmbeddingResult with embeddings and usage data
        """
        if isinstance(inputs, str):
            inputs_list = [inputs]
        else:
            inputs_list = inputs

        embed_model = model or kwargs.get("model", "gemini-embedding-001")

        # Build config with optional dimensions
        config = None
        if dimensions is not None:
            config = types.EmbedContentConfig(output_dimensionality=dimensions)

        # Estimate input tokens for rate limiting (rough estimate based on text length)
        input_tokens = sum(len(text) // 4 for text in inputs_list)  # ~4 chars per token

        async def _make_request() -> EmbeddingResult:
            """Inner function to make the API request (with rate limiting and retry)."""
            async with self.limiter.limit(tokens=input_tokens, requests=1):
                try:
                    response = await self._client.aio.models.embed_content(
                        model=embed_model,
                        contents=inputs_list,
                        config=config,
                    )

                    # Handle response - can be a list of embeddings or single embedding
                    if hasattr(response, "embeddings") and response.embeddings:
                        embeddings = [emb.values for emb in response.embeddings]
                    elif hasattr(response, "embedding"):
                        embeddings = [response.embedding]
                    else:
                        embeddings = []

                    return EmbeddingResult(
                        embeddings=embeddings,
                        model=embed_model,
                        usage=Usage(total_tokens=0),
                        status=200,
                    )
                except genai_errors.APIError as e:
                    return EmbeddingResult(
                        embeddings=[],
                        status=e.code if hasattr(e, "code") else 500,
                        error=e.message if hasattr(e, "message") else str(e),
                    )
                except Exception as e:
                    return EmbeddingResult(
                        embeddings=[],
                        status=500,
                        error=str(e),
                    )

        # Use retry wrapper for transient failures
        return await self._with_retry(
            _make_request,
            attempts=attempts,
            backoff=backoff,
        )

    async def close(self) -> None:
        """
        Clean up provider resources.

        Uses aclose() for the async client as per google-genai SDK docs.
        """
        # Close cache first
        await self.cache.close()

        if hasattr(self, "_client") and self._client:
            # The async client uses aclose()
            if hasattr(self._client, "aio") and hasattr(self._client.aio, "aclose"):
                await self._client.aio.aclose()
            elif hasattr(self._client, "close"):
                self._client.close()

    async def count_tokens_api(self, messages: MessageInput) -> int:
        """
        Count tokens using Google's token counting API.

        Args:
            messages: Input messages to count tokens for

        Returns:
            Total number of tokens
        """
        msgs = self._normalize_messages(messages)
        _, history = self._convert_messages(msgs)

        response = await self._client.aio.models.count_tokens(
            model=self.model_name,
            contents=history,
        )
        return response.total_tokens
