"""
OpenAI provider implementation.

This module implements the Provider protocol for OpenAI's API,
supporting both chat completions and embeddings.
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import re
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
from ..content import (
    ContentHandlingMode,
    content_blocks_to_openai_responses_content,
    message_to_content_blocks,
    message_to_openai_chat_dict_with_mode,
)
from ..errors import (
    failure_to_completion_result,
    failure_to_embedding_result,
    failure_to_stream_error_data,
    normalize_exception,
    normalize_provider_failure,
)
from ..hashing import cache_key as compute_cache_key
from ..rate_limit import Limiter
from ..structured import build_structured_response_format
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


logger = logging.getLogger("llm_client.providers.openai")


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

    def _failure(
        self,
        *,
        message: str | None = None,
        status: int | None = None,
        error: Exception | None = None,
        operation: str,
    ):
        if error is not None:
            return normalize_exception(
                error,
                provider="openai",
                model=self.model_name,
                operation=operation,
            )
        return normalize_provider_failure(
            status=status,
            message=message or "Provider error",
            provider="openai",
            model=self.model_name,
            operation=operation,
        )

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
            if response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                if isinstance(json_schema, dict) and isinstance(json_schema.get("schema"), dict):
                    normalized = build_structured_response_format(
                        json_schema["schema"],
                        provider="openai",
                        name=str(json_schema.get("name") or "structured_output"),
                        strict=bool(json_schema.get("strict", True)),
                    )
                    if isinstance(normalized, dict):
                        return normalized
            return response_format

        # Assume it's a Pydantic model or similar for structured output
        if isinstance(response_format, str):
            raise ValueError(f"Unsupported response_format: {response_format!r}")
        return response_format

    @staticmethod
    def _cache_key(api: str, params: dict[str, Any]) -> str:
        return compute_cache_key(api, params)

    @staticmethod
    def _sanitize_openai_function_name(name: str, *, used: set[str]) -> str:
        """
        OpenAI chat-completions function names must match ^[a-zA-Z0-9_-]+$.
        """
        base = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name or "")).strip("_")
        if not base:
            base = "tool"
        # Keep room for suffixes if we need to resolve collisions.
        base = base[:64]
        candidate = base
        suffix = 2
        while candidate in used:
            suffix_str = f"_{suffix}"
            candidate = f"{base[: max(1, 64 - len(suffix_str))]}{suffix_str}"
            suffix += 1
        used.add(candidate)
        return candidate

    def _prepare_openai_tools(
        self,
        tools: list[Tool] | None,
    ) -> tuple[list[dict[str, Any]] | None, dict[str, str], dict[str, str]]:
        """
        Convert tools and alias names to OpenAI-safe function names when needed.

        Returns:
            (provider_tools, alias_to_original, original_to_alias)
        """
        api_tools = self._tools_to_api_format(tools)
        if not api_tools:
            return None, {}, {}

        alias_to_original: dict[str, str] = {}
        original_to_alias: dict[str, str] = {}
        used_aliases: set[str] = set()
        rewritten: list[dict[str, Any]] = []

        for item in api_tools:
            if not isinstance(item, dict):
                rewritten.append(item)  # defensive passthrough
                continue

            item_copy = dict(item)
            fn = item_copy.get("function")
            if not isinstance(fn, dict):
                rewritten.append(item_copy)
                continue

            fn_copy = dict(fn)
            original_name = str(fn_copy.get("name") or "")
            alias_name = self._sanitize_openai_function_name(original_name, used=used_aliases)
            fn_copy["name"] = alias_name
            parameters = fn_copy.get("parameters")
            if isinstance(parameters, dict):
                fn_copy["parameters"] = self._sanitize_openai_function_parameters_schema(parameters)
            item_copy["function"] = fn_copy
            rewritten.append(item_copy)

            if original_name:
                alias_to_original[alias_name] = original_name
                original_to_alias[original_name] = alias_name

        return rewritten, alias_to_original, original_to_alias

    @staticmethod
    def _messages_to_api_format(
        messages: list[Any],
        *,
        content_mode: ContentHandlingMode = ContentHandlingMode.LOSSY,
        responses_api: bool = False,
    ) -> list[dict[str, Any]]:
        if not responses_api:
            return [message_to_openai_chat_dict_with_mode(msg, mode=content_mode) for msg in messages]
        payloads: list[dict[str, Any]] = []
        for msg in messages:
            payload = msg.to_dict()
            payload["content"] = content_blocks_to_openai_responses_content(message_to_content_blocks(msg), mode=content_mode)
            if payload.get("content") is None:
                payload.pop("content", None)
            payloads.append(payload)
        return payloads

    def _collect_message_tool_name_aliases(
        self,
        messages: list[dict[str, Any]],
        *,
        seed_alias_to_original: dict[str, str] | None = None,
        seed_original_to_alias: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Collect alias mappings for tool names embedded in message history."""
        alias_to_original = dict(seed_alias_to_original or {})
        original_to_alias = dict(seed_original_to_alias or {})
        used_aliases: set[str] = set(alias_to_original.keys())

        def _register(name: Any) -> None:
            original_name = str(name or "").strip()
            if not original_name:
                return
            if original_name in original_to_alias:
                used_aliases.add(original_to_alias[original_name])
                return
            alias_name = self._sanitize_openai_function_name(original_name, used=used_aliases)
            alias_to_original[alias_name] = original_name
            original_to_alias[original_name] = alias_name

        for message in messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    fn = tool_call.get("function")
                    if isinstance(fn, dict):
                        _register(fn.get("name"))
            if str(message.get("role") or "") == "tool":
                _register(message.get("name"))

        return alias_to_original, original_to_alias

    @staticmethod
    def _sanitize_openai_function_parameters_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """
        OpenAI function parameters require a strict top-level object schema and
        reject top-level combinators like oneOf/anyOf/allOf.

        We preserve core object-shape information while removing unsupported
        combinators/refs/metadata that frequently appear in JSON Schema drafts.
        """
        def _sanitize(node: Any, *, top_level: bool = False) -> Any:
            if isinstance(node, list):
                return [_sanitize(item, top_level=False) for item in node]
            if not isinstance(node, dict):
                return node

            # External refs regularly appear in contract-derived schemas; replace
            # them with permissive sub-schemas for OpenAI function validation.
            if "$ref" in node and len(node) == 1:
                return {}

            out: dict[str, Any] = {}
            for key, value in node.items():
                if key in {"$schema", "$id", "$defs", "$ref"}:
                    continue
                if key in {"oneOf", "anyOf", "allOf", "not"}:
                    continue
                if top_level and key == "enum":
                    continue
                if key in {"title", "examples", "deprecated", "readOnly", "writeOnly"}:
                    continue
                out[key] = _sanitize(value, top_level=False)

            is_object_schema = (
                out.get("type") == "object"
                or "properties" in out
                or "additionalProperties" in out
                or "required" in out
            )
            if is_object_schema:
                out["type"] = "object"
                props = out.get("properties")
                if not isinstance(props, dict) or not props:
                    # OpenAI rejects object schemas without properties. Keep
                    # pass-through behavior by allowing additional properties.
                    out["properties"] = {
                        "_payload": {
                            "type": "string",
                            "description": "Optional placeholder. Additional properties may be provided.",
                        }
                    }
                    if "additionalProperties" not in out:
                        out["additionalProperties"] = True
                    required = out.get("required")
                    if isinstance(required, list):
                        out["required"] = [item for item in required if item in out["properties"]]

            return out

        sanitized = _sanitize(dict(schema), top_level=True)
        if not isinstance(sanitized, dict):
            return {"type": "object", "properties": {"_payload": {"type": "string"}}}
        sanitized.setdefault("type", "object")
        if sanitized.get("type") != "object":
            sanitized["type"] = "object"
        sanitized.pop("enum", None)
        if not isinstance(sanitized.get("properties"), dict) or not sanitized.get("properties"):
            sanitized["properties"] = {"_payload": {"type": "string"}}
        return sanitized

    @staticmethod
    def _rewrite_messages_for_openai_tool_aliases(
        messages: list[dict[str, Any]],
        *,
        original_to_alias: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not original_to_alias:
            return messages

        rewritten_messages: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                rewritten_messages.append(message)
                continue

            msg_copy = dict(message)

            tool_calls = msg_copy.get("tool_calls")
            if isinstance(tool_calls, list):
                rewritten_tool_calls: list[Any] = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        rewritten_tool_calls.append(tool_call)
                        continue
                    tc_copy = dict(tool_call)
                    fn = tc_copy.get("function")
                    if isinstance(fn, dict):
                        fn_copy = dict(fn)
                        fn_name = str(fn_copy.get("name") or "")
                        if fn_name:
                            fn_copy["name"] = original_to_alias.get(fn_name, fn_name)
                        tc_copy["function"] = fn_copy
                    rewritten_tool_calls.append(tc_copy)
                msg_copy["tool_calls"] = rewritten_tool_calls

            if str(msg_copy.get("role") or "") == "tool":
                tool_name = str(msg_copy.get("name") or "")
                if tool_name:
                    msg_copy["name"] = original_to_alias.get(tool_name, tool_name)

            rewritten_messages.append(msg_copy)

        return rewritten_messages

    def _completion_token_limit_param(self) -> str:
        """
        Return the correct output-token parameter for chat completions.

        GPT-5 chat completions reject `max_tokens` and require
        `max_completion_tokens`.
        """
        model_key = getattr(self._model, "key", "")
        if str(model_key).startswith("gpt-5") or str(self.model_name).startswith("gpt-5"):
            return "max_completion_tokens"
        return "max_tokens"

    def _set_completion_token_limit(
        self,
        params: dict[str, Any],
        max_tokens: int | None,
    ) -> None:
        if max_tokens is None:
            return
        params[self._completion_token_limit_param()] = max_tokens

    def _set_temperature(
        self,
        params: dict[str, Any],
        temperature: float | None,
    ) -> None:
        """
        GPT-5 chat-completions currently only support the default temperature (1).
        Omit non-default values instead of sending unsupported parameters.
        """
        if temperature is None:
            return
        model_key = getattr(self._model, "key", "")
        is_gpt5_family = str(model_key).startswith("gpt-5") or str(self.model_name).startswith("gpt-5")
        if is_gpt5_family and float(temperature) != 1.0:
            return
        params["temperature"] = temperature

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
        api_messages = self._messages_to_api_format(msg_objects, responses_api=self.use_responses_api)

        # Build params
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
        }
        alias_to_original: dict[str, str] = {}
        original_to_alias: dict[str, str] = {}

        # Add optional parameters
        if tools:
            provider_tools, alias_to_original, original_to_alias = self._prepare_openai_tools(tools)
            if provider_tools:
                params["tools"] = provider_tools
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = tool_choice
                else:
                    params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": original_to_alias.get(tool_choice, tool_choice)},
                    }

        alias_to_original, original_to_alias = self._collect_message_tool_name_aliases(
            api_messages,
            seed_alias_to_original=alias_to_original,
            seed_original_to_alias=original_to_alias,
        )
        if original_to_alias:
            api_messages = self._rewrite_messages_for_openai_tool_aliases(
                api_messages,
                original_to_alias=original_to_alias,
            )
            params["messages"] = api_messages

        self._set_temperature(params, temperature)
        self._set_completion_token_limit(params, max_tokens)

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
                        msg = response.choices[0].message
                        content = self._coerce_chat_message_content(getattr(msg, "content", None))

                        # Parse JSON if requested
                        if rf.get("type") in ("json_object", "json_schema"):
                            if (content is None or (isinstance(content, str) and not content.strip())) and hasattr(
                                msg, "parsed"
                            ):
                                parsed = getattr(msg, "parsed", None)
                                if parsed is not None:
                                    if hasattr(parsed, "model_dump"):
                                        content = parsed.model_dump()
                                    elif hasattr(parsed, "dict"):
                                        content = parsed.dict()
                                    else:
                                        content = parsed
                            if isinstance(content, str):
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
                                name=alias_to_original.get(tc.function.name, tc.function.name),
                                arguments=tc.function.arguments,
                            )
                            for tc in msg.tool_calls
                        ]

                    # Parse usage
                    usage = self.parse_usage(response.usage.to_dict())
                    limit_ctx.output_tokens = usage.output_tokens

                    return CompletionResult(
                        content=(
                            content
                            if isinstance(content, str)
                            else (None if content is None else json.dumps(content))
                        ),
                        tool_calls=tool_calls,
                        usage=usage,
                        model=str(getattr(response, "model", None) or params.get("model") or self.model_name),
                        finish_reason=response.choices[0].finish_reason,
                        status=200,
                        raw_response=response,
                    )

                except openai.APIConnectionError as e:
                    return failure_to_completion_result(
                        normalize_provider_failure(
                            status=503,
                            message=str(e.__cause__ or e),
                            provider="openai",
                            model=self.model_name,
                            operation="complete",
                        ),
                        model=self.model_name,
                    )
                except openai.RateLimitError as e:
                    return failure_to_completion_result(self._failure(error=e, operation="complete"), model=self.model_name)
                except openai.APIStatusError as e:
                    return failure_to_completion_result(self._failure(error=e, operation="complete"), model=self.model_name)
                except Exception as e:
                    return failure_to_completion_result(self._failure(error=e, operation="complete"), model=self.model_name)

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

    @staticmethod
    def _coerce_chat_message_content(value: Any) -> Any:
        """
        Normalize chat-completions message.content across SDK/model variants.

        Some models/SDK versions return content as a list of content-part objects
        instead of a plain string.
        """
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                    if isinstance(text, dict):
                        nested_text = text.get("value")
                        if isinstance(nested_text, str):
                            parts.append(nested_text)
                            continue
                    inner = item.get("content")
                    if isinstance(inner, str):
                        parts.append(inner)
                        continue
                    if isinstance(inner, dict):
                        nested_inner = inner.get("value")
                        if isinstance(nested_inner, str):
                            parts.append(nested_inner)
                            continue
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
                    continue
                nested_text_attr = getattr(text_attr, "value", None)
                if isinstance(nested_text_attr, str):
                    parts.append(nested_text_attr)
                    continue
                inner_attr = getattr(item, "content", None)
                if isinstance(inner_attr, str):
                    parts.append(inner_attr)
                    continue
                nested_inner_attr = getattr(inner_attr, "value", None)
                if isinstance(nested_inner_attr, str):
                    parts.append(nested_inner_attr)
                    continue
            if parts:
                return "".join(parts)
            return ""
        return value

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
                        model=str(getattr(response, "model", None) or responses_params.get("model") or self.model_name),
                        status=200,
                        raw_response=response,
                    )

                except openai.APIConnectionError as e:
                    return failure_to_completion_result(
                        normalize_provider_failure(
                            status=503,
                            message=str(e.__cause__ or e),
                            provider="openai",
                            model=self.model_name,
                            operation="complete",
                        ),
                        model=self.model_name,
                    )
                except openai.RateLimitError as e:
                    return failure_to_completion_result(self._failure(error=e, operation="complete"), model=self.model_name)
                except openai.APIStatusError as e:
                    return failure_to_completion_result(self._failure(error=e, operation="complete"), model=self.model_name)
                except Exception as e:
                    return failure_to_completion_result(self._failure(error=e, operation="complete"), model=self.model_name)

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
        response_format: str | dict[str, Any] | type | None = None,
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
        api_messages = self._messages_to_api_format(msg_objects, responses_api=self.use_responses_api)

        # Build params
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        alias_to_original: dict[str, str] = {}
        original_to_alias: dict[str, str] = {}

        # Add optional parameters
        if tools:
            provider_tools, alias_to_original, original_to_alias = self._prepare_openai_tools(tools)
            if provider_tools:
                params["tools"] = provider_tools
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    params["tool_choice"] = tool_choice
                else:
                    params["tool_choice"] = {
                        "type": "function",
                        "function": {"name": original_to_alias.get(tool_choice, tool_choice)},
                    }

        alias_to_original, original_to_alias = self._collect_message_tool_name_aliases(
            api_messages,
            seed_alias_to_original=alias_to_original,
            seed_original_to_alias=original_to_alias,
        )
        if original_to_alias:
            api_messages = self._rewrite_messages_for_openai_tool_aliases(
                api_messages,
                original_to_alias=original_to_alias,
            )
            params["messages"] = api_messages

        self._set_temperature(params, temperature)
        self._set_completion_token_limit(params, max_tokens)

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

        params.update(kwargs)

        # Emit metadata event
        yield StreamEvent(type=StreamEventType.META, data={"model": str(params.get("model") or self.model_name), "stream": True})

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
                    delta_content = self._coerce_chat_message_content(getattr(delta, "content", None))
                    if isinstance(delta_content, str) and delta_content:
                        content_buffer += delta_content
                        yield StreamEvent(type=StreamEventType.TOKEN, data=delta_content)

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
                                        name=alias_to_original.get(
                                            tool_calls_buffer[idx]["name"], tool_calls_buffer[idx]["name"]
                                        ),
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
                                    name=alias_to_original.get(tc_data["name"], tc_data["name"]),
                                    arguments=tc_data["arguments"],
                                ),
                            )

                # Build final result
                tool_calls = None
                if tool_calls_buffer:
                    tool_calls = [
                        ToolCall(
                            id=tc["id"],
                            name=alias_to_original.get(tc["name"], tc["name"]),
                            arguments=tc["arguments"],
                        )
                        for tc in tool_calls_buffer.values()
                    ]

                final_result = CompletionResult(
                    content=content_buffer if content_buffer else None,
                    tool_calls=tool_calls,
                    usage=usage,
                    model=str(params.get("model") or self.model_name),
                    status=200,
                    reasoning=reasoning_buffer if reasoning_buffer else None,
                )

                yield StreamEvent(type=StreamEventType.DONE, data=final_result)

            except openai.APIConnectionError as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data=failure_to_stream_error_data(
                        normalize_provider_failure(
                            status=503,
                            message=str(e.__cause__ or e),
                            provider="openai",
                            model=self.model_name,
                            operation="stream",
                        )
                    ),
                )
            except openai.RateLimitError as e:
                yield StreamEvent(type=StreamEventType.ERROR, data=failure_to_stream_error_data(self._failure(error=e, operation="stream")))
            except openai.APIStatusError as e:
                yield StreamEvent(type=StreamEventType.ERROR, data=failure_to_stream_error_data(self._failure(error=e, operation="stream")))
            except Exception as e:
                yield StreamEvent(type=StreamEventType.ERROR, data=failure_to_stream_error_data(self._failure(error=e, operation="stream")))

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
                return failure_to_embedding_result(
                    normalize_provider_failure(
                        status=503,
                        message=str(e.__cause__ or e),
                        provider="openai",
                        model=self.model_name,
                        operation="embed",
                    ),
                    model=self.model_name,
                )
            except openai.RateLimitError as e:
                return failure_to_embedding_result(self._failure(error=e, operation="embed"), model=self.model_name)
            except openai.APIStatusError as e:
                return failure_to_embedding_result(self._failure(error=e, operation="embed"), model=self.model_name)

    # Cache serialization methods now imported from cache.serializers
    _result_to_cache = staticmethod(result_to_cache_dict)
    _cached_to_result = staticmethod(cache_dict_to_result)


__all__ = ["OpenAIProvider"]
