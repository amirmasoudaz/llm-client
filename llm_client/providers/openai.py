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
    MetadataBlock,
    content_blocks_to_text,
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
    AudioSpeechResult,
    AudioTranscriptionResult,
    BackgroundResponseResult,
    CompactionResult,
    CompletionResult,
    DeepResearchRunResult,
    ConversationItemResource,
    ConversationItemsPage,
    ConversationResource,
    DeletionResult,
    EmbeddingResult,
    FileContentResult,
    FileResource,
    FilesPage,
    FineTuningJobEventsPage,
    FineTuningJobResult,
    FineTuningJobsPage,
    GeneratedImage,
    ImageGenerationResult,
    MessageInput,
    ModerationResult,
    NormalizedOutputItem,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallDelta,
    Usage,
    RealtimeCallResult,
    RealtimeConnection,
    RealtimeClientSecretResult,
    RealtimeTranscriptionSessionResult,
    VectorStoreFileBatchResource,
    VectorStoreFileContentResult,
    VectorStoreFileResource,
    VectorStoreFilesPage,
    VectorStoreResource,
    VectorStoreSearchResult,
    VectorStoresPage,
    WebhookEventResult,
)

if TYPE_CHECKING:
    from ..models import ModelProfile
    from ..tools.base import ToolDefinition
from ..tools.base import (
    ResponsesAttributeFilter,
    ResponsesBuiltinTool,
    ResponsesFileSearchRankingOptions,
    ResponsesMCPTool,
    is_provider_native_tool,
)


logger = logging.getLogger("llm_client.providers.openai")

_OPENAI_FILE_SEARCH_RESULTS_INCLUDE = "file_search_call.results"


_DEEP_RESEARCH_CLARIFY_INSTRUCTIONS = """
You are talking to a user who is asking for a research task to be conducted. Your job is to gather
more information from the user to successfully complete the task.

GUIDELINES:
- Be concise while gathering all necessary information.
- Gather only the information needed to carry out the research task.
- Use bullet points or numbered lists when they improve clarity.
- Do not conduct research yourself.
"""

_DEEP_RESEARCH_REWRITE_INSTRUCTIONS = """
You will be given a research task by a user. Your job is to produce a set of instructions for a
researcher that will complete the task. Do not complete the task yourself.

GUIDELINES:
1. Include all known user preferences and constraints.
2. Fill in necessary but unstated dimensions as open-ended instead of inventing details.
3. Use the first person to preserve the user's intent.
4. Ask for structured output and tables when they would improve the final report.
5. Prefer primary and official sources when the task implies source quality requirements.
"""


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
    def _serialize_openai_request_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(key): OpenAIProvider._serialize_openai_request_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [OpenAIProvider._serialize_openai_request_value(item) for item in value]
        if hasattr(value, "to_dict"):
            return OpenAIProvider._serialize_openai_request_value(value.to_dict())
        if hasattr(value, "model_dump"):
            return OpenAIProvider._serialize_openai_request_value(value.model_dump())
        return value

    @staticmethod
    def _merge_openai_include(
        include: list[str] | tuple[str, ...] | None,
        *extra_values: str,
    ) -> list[str] | None:
        merged: list[str] = [str(item) for item in (include or []) if str(item)]
        for item in extra_values:
            if item and item not in merged:
                merged.append(item)
        return merged or None

    @staticmethod
    def _apply_openai_param_alias(
        params: dict[str, Any],
        *,
        canonical_key: str,
        alias_value: Any,
        alias_name: str,
    ) -> None:
        if alias_value is None:
            return
        if canonical_key in params:
            raise ValueError(f"Provide only one of `{alias_name}` or `{canonical_key}`.")
        params[canonical_key] = OpenAIProvider._serialize_openai_request_value(alias_value)

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
        tools: list[ToolDefinition] | None,
        *,
        responses_api: bool = False,
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

        def _rewrite_tool_dict(item: dict[str, Any], *, nested_in_namespace: bool = False) -> dict[str, Any]:
            item_copy = dict(item)
            item_type = str(item_copy.get("type") or "")

            if item_type == "namespace":
                namespace_tools = item_copy.get("tools")
                if isinstance(namespace_tools, list):
                    item_copy["tools"] = [
                        _rewrite_tool_dict(tool_item, nested_in_namespace=True)
                        if isinstance(tool_item, dict)
                        else tool_item
                        for tool_item in namespace_tools
                    ]
                return item_copy

            flatten_function = responses_api or nested_in_namespace
            if flatten_function and item_type == "function":
                fn = item_copy.pop("function", None)
                if isinstance(fn, dict):
                    for key in ("name", "description", "parameters", "strict", "defer_loading"):
                        if key in fn and key not in item_copy:
                            item_copy[key] = fn[key]
                fn = item_copy
            else:
                fn = item_copy.get("function")
                if item_type == "function" and not isinstance(fn, dict):
                    flattened = {
                        key: item_copy.pop(key)
                        for key in ("name", "description", "parameters", "strict", "defer_loading")
                        if key in item_copy
                    }
                    if flattened:
                        fn = flattened
                        item_copy["function"] = fn
            if not isinstance(fn, dict):
                return item_copy

            fn_copy = dict(fn)
            original_name = str(fn_copy.get("name") or "")
            alias_name = self._sanitize_openai_function_name(original_name, used=used_aliases)
            fn_copy["name"] = alias_name
            parameters = fn_copy.get("parameters")
            if isinstance(parameters, dict):
                fn_copy["parameters"] = self._sanitize_openai_function_parameters_schema(parameters)
            if flatten_function and item_copy.get("type") == "function" and "strict" not in fn_copy:
                fn_copy["strict"] = True
            if flatten_function:
                item_copy.update(fn_copy)
            else:
                item_copy["function"] = fn_copy

            if original_name:
                alias_to_original[alias_name] = original_name
                original_to_alias[original_name] = alias_name

            return item_copy

        for item in api_tools:
            if not isinstance(item, dict):
                rewritten.append(item)  # defensive passthrough
                continue
            rewritten.append(_rewrite_tool_dict(item))

        return rewritten, alias_to_original, original_to_alias

    @staticmethod
    def _validate_tool_configuration(
        *,
        tools: list[Any] | None,
        use_responses_api: bool,
    ) -> None:
        if not tools:
            return

        if not use_responses_api:
            invalid_native = [type(tool).__name__ for tool in tools if is_provider_native_tool(tool)]
            invalid_dicts = [
                type(tool).__name__
                for tool in tools
                if isinstance(tool, dict) and str(tool.get("type") or "") not in {"", "function"}
            ]
            invalid = sorted(set(invalid_native + invalid_dicts))
            if invalid:
                rendered = ", ".join(invalid)
                raise ValueError(
                    "OpenAI built-in Responses tools and custom grammar tools require "
                    "`OpenAIProvider(..., use_responses_api=True)`; got unsupported tool descriptors: "
                    f"{rendered}"
                )

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
        for index, msg in enumerate(messages):
            message_dict = msg.to_dict() if hasattr(msg, "to_dict") else dict(msg)
            role = str(message_dict.get("role") or "")
            content = message_dict.get("content")

            if role in {"system", "user", "developer"}:
                rendered = content_blocks_to_openai_responses_content(content, mode=content_mode)
                if rendered is None:
                    rendered = []
                elif isinstance(rendered, str):
                    rendered = [{"type": "input_text", "text": rendered}]
                payloads.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": rendered,
                    }
                )
                continue

            if role == "assistant":
                preserved_output = OpenAIProvider._extract_preserved_openai_response_output(msg)
                if preserved_output:
                    payloads.extend(preserved_output)
                    continue
                assistant_text = content_blocks_to_text(content)
                if assistant_text:
                    payloads.append(
                        {
                            "type": "message",
                            "id": f"msg_assistant_{index}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": assistant_text,
                                    "annotations": [],
                                    "logprobs": [],
                                }
                            ],
                        }
                    )
                for tool_index, tool_call in enumerate(message_dict.get("tool_calls") or []):
                    if not isinstance(tool_call, dict):
                        continue
                    function_data = tool_call.get("function") or {}
                    call_id = str(tool_call.get("id") or f"call_{index}_{tool_index}")
                    payloads.append(
                        {
                            "type": "function_call",
                            "id": call_id,
                            "call_id": call_id,
                            "name": str(function_data.get("name") or ""),
                            "arguments": str(function_data.get("arguments") or ""),
                            "status": "completed",
                        }
                    )
                continue

            if role == "tool":
                rendered = content_blocks_to_openai_responses_content(content, mode=content_mode)
                if rendered is None:
                    rendered = ""
                payloads.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message_dict.get("tool_call_id") or f"call_{index}"),
                        "output": rendered,
                    }
                )
                continue

        return payloads

    @staticmethod
    def _extract_preserved_openai_response_output(message: Any) -> list[dict[str, Any]] | None:
        for block in message_to_content_blocks(message):
            if not isinstance(block, MetadataBlock):
                continue
            data = block.data
            if data.get("provider") != "openai":
                continue
            output = data.get("responses_output")
            if isinstance(output, list):
                return [dict(item) for item in output if isinstance(item, dict)] or None
        return None

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

    @staticmethod
    def _rewrite_responses_input_items_for_openai_tool_aliases(
        items: list[dict[str, Any]],
        *,
        original_to_alias: dict[str, str],
    ) -> list[dict[str, Any]]:
        if not original_to_alias:
            return items

        rewritten_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                rewritten_items.append(item)
                continue
            item_copy = dict(item)
            if item_copy.get("type") == "function_call":
                name = str(item_copy.get("name") or "")
                if name:
                    item_copy["name"] = original_to_alias.get(name, name)
            elif item_copy.get("type") == "tool_search_output":
                tools = item_copy.get("tools")
                if isinstance(tools, list):
                    item_copy["tools"] = [
                        OpenAIProvider._rewrite_tool_definition_aliases(tool, original_to_alias=original_to_alias)
                        if isinstance(tool, dict)
                        else tool
                        for tool in tools
                    ]
            rewritten_items.append(item_copy)
        return rewritten_items

    @staticmethod
    def _rewrite_tool_definition_aliases(
        tool: dict[str, Any],
        *,
        original_to_alias: dict[str, str] | None = None,
        alias_to_original: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        tool_copy = dict(tool)
        item_type = str(tool_copy.get("type") or "")

        if item_type == "namespace":
            namespace_tools = tool_copy.get("tools")
            if isinstance(namespace_tools, list):
                tool_copy["tools"] = [
                    OpenAIProvider._rewrite_tool_definition_aliases(
                        namespace_tool,
                        original_to_alias=original_to_alias,
                        alias_to_original=alias_to_original,
                    )
                    if isinstance(namespace_tool, dict)
                    else namespace_tool
                    for namespace_tool in namespace_tools
                ]
            return tool_copy

        if item_type != "function":
            return tool_copy

        if isinstance(tool_copy.get("function"), dict):
            fn_copy = dict(tool_copy["function"])
            name = str(fn_copy.get("name") or "")
            if name:
                if original_to_alias is not None:
                    fn_copy["name"] = original_to_alias.get(name, name)
                elif alias_to_original is not None:
                    fn_copy["name"] = alias_to_original.get(name, name)
            tool_copy["function"] = fn_copy
            return tool_copy

        name = str(tool_copy.get("name") or "")
        if name:
            if original_to_alias is not None:
                tool_copy["name"] = original_to_alias.get(name, name)
            elif alias_to_original is not None:
                tool_copy["name"] = alias_to_original.get(name, name)
        return tool_copy

    @staticmethod
    def _responses_text_config_and_parser(
        response_format: str | dict[str, Any] | type | None,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, type | None]:
        normalized = OpenAIProvider._normalize_response_format(response_format, messages)
        if normalized is None:
            return None, None
        if isinstance(normalized, dict):
            if normalized.get("type") == "text":
                return None, None
            return {"format": normalized}, None
        return None, normalized

    @staticmethod
    def _serialize_responses_item(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): OpenAIProvider._serialize_responses_item(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [OpenAIProvider._serialize_responses_item(item) for item in value]
        if hasattr(value, "model_dump"):
            return OpenAIProvider._serialize_responses_item(value.model_dump())
        if hasattr(value, "to_dict"):
            return OpenAIProvider._serialize_responses_item(value.to_dict())
        if hasattr(value, "dict"):
            return OpenAIProvider._serialize_responses_item(value.dict())
        if hasattr(value, "__dict__"):
            return {
                str(key): OpenAIProvider._serialize_responses_item(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return str(value)

    @staticmethod
    def _serialize_responses_output_items(response: Any) -> list[dict[str, Any]] | None:
        output = getattr(response, "output", None)
        if not output:
            return None
        serialized: list[dict[str, Any]] = []
        for item in output:
            rendered = OpenAIProvider._serialize_responses_item(item)
            if isinstance(rendered, dict):
                serialized.append(rendered)
        return serialized or None

    @staticmethod
    def _is_raw_responses_input_item(value: Any) -> bool:
        return isinstance(value, dict) and "type" in value and "role" not in value

    def _coerce_responses_input_items(
        self,
        items: MessageInput | list[dict[str, Any]] | None,
        *,
        content_mode: ContentHandlingMode = ContentHandlingMode.LOSSY,
    ) -> list[dict[str, Any]] | None:
        if items is None:
            return None
        if self._is_raw_responses_input_item(items):
            rendered = self._serialize_responses_item(items)
            return [rendered] if isinstance(rendered, dict) else None
        if isinstance(items, list) and all(self._is_raw_responses_input_item(item) for item in items):
            return [
                rendered
                for item in items
                if isinstance((rendered := self._serialize_responses_item(item)), dict)
            ]
        normalized = self._normalize_messages(items)
        return self._messages_to_api_format(normalized, responses_api=True, content_mode=content_mode)

    @staticmethod
    def _conversation_resource_from_response(response: Any) -> ConversationResource:
        metadata = getattr(response, "metadata", None)
        metadata_dict: dict[str, Any] | None = None
        if isinstance(metadata, dict):
            metadata_dict = dict(metadata)
        elif metadata is not None:
            try:
                metadata_dict = dict(metadata)
            except Exception:
                metadata_dict = None
        return ConversationResource(
            conversation_id=str(getattr(response, "id", "") or ""),
            created_at=getattr(response, "created_at", None),
            metadata=metadata_dict,
            deleted=getattr(response, "deleted", None),
            raw_response=response,
        )

    def _compaction_result_from_response(
        self,
        response: Any,
        *,
        alias_to_original: dict[str, str] | None = None,
    ) -> CompactionResult:
        raw_usage = response.usage.to_dict() if getattr(response, "usage", None) else {}
        usage = self.parse_usage(raw_usage) if raw_usage else None
        provider_items = self._serialize_responses_output_items(response)
        output_items = self._normalize_serialized_responses_output_items(
            provider_items,
            alias_to_original=alias_to_original or {},
        )
        return CompactionResult(
            compaction_id=str(getattr(response, "id", "") or ""),
            created_at=getattr(response, "created_at", None),
            usage=usage,
            output_items=output_items,
            provider_items=provider_items,
            raw_response=response,
        )

    def _conversation_item_resource_from_item(
        self,
        item: Any,
        *,
        alias_to_original: dict[str, str] | None = None,
    ) -> ConversationItemResource:
        serialized = self._serialize_responses_item(item)
        payload = serialized if isinstance(serialized, dict) else {"type": str(getattr(item, "type", "") or "")}
        output_items = self._normalize_serialized_responses_output_items(
            [payload],
            alias_to_original=alias_to_original or {},
        )
        content = payload.get("content")
        if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
            part = content[0]
            if part.get("type") in {"input_text", "output_text"} and isinstance(part.get("text"), str):
                content = part["text"]
        return ConversationItemResource(
            item_id=str(payload.get("id") or "") or None,
            item_type=str(payload.get("type") or ""),
            role=str(payload.get("role") or "") or None,
            status=str(payload.get("status") or "") or None,
            content=content,
            output_items=output_items,
            raw_item=payload,
        )

    def _conversation_items_page_from_response(
        self,
        response: Any,
        *,
        alias_to_original: dict[str, str] | None = None,
    ) -> ConversationItemsPage:
        serialized = self._serialize_responses_item(response)
        payload = serialized if isinstance(serialized, dict) else {}
        raw_items = payload.get("data")
        items: list[ConversationItemResource] = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                items.append(
                    self._conversation_item_resource_from_item(
                        raw_item,
                        alias_to_original=alias_to_original,
                    )
                )
        return ConversationItemsPage(
            items=items,
            first_id=str(payload.get("first_id") or "") or None,
            last_id=str(payload.get("last_id") or "") or None,
            has_more=bool(payload.get("has_more")),
            raw_response=response,
        )

    def _moderation_result_from_response(self, response: Any) -> ModerationResult:
        payload = self._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        serialized_results = response_dict.get("results")
        results = [dict(item) for item in serialized_results] if isinstance(serialized_results, list) else []
        flagged = any(bool(item.get("flagged")) for item in results if isinstance(item, dict))
        return ModerationResult(
            flagged=flagged,
            model=str(response_dict.get("model") or "") or None,
            results=results,
            status=200,
            raw_response=response,
        )

    @staticmethod
    def _image_generation_result_from_response(response: Any, *, model_name: str) -> ImageGenerationResult:
        payload = OpenAIProvider._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        images: list[GeneratedImage] = []
        for item in response_dict.get("data") or []:
            if isinstance(item, dict):
                images.append(
                    GeneratedImage(
                        url=str(item.get("url") or "") or None,
                        b64_json=str(item.get("b64_json") or "") or None,
                        revised_prompt=str(item.get("revised_prompt") or "") or None,
                        raw_item=dict(item),
                    )
                )
        usage_payload = response_dict.get("usage")
        usage = None
        if isinstance(usage_payload, dict):
            usage = Usage.from_dict(
                {
                    "input_tokens": int(usage_payload.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage_payload.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage_payload.get("total_tokens", 0) or 0),
                }
            )
        return ImageGenerationResult(
            images=images,
            created_at=response_dict.get("created"),
            usage=usage,
            model=str(response_dict.get("model") or model_name) or model_name,
            status=200,
            raw_response=response,
        )

    @staticmethod
    def _audio_transcription_result_from_response(response: Any, *, model_name: str) -> AudioTranscriptionResult:
        payload = OpenAIProvider._serialize_responses_item(response)
        if isinstance(payload, str):
            return AudioTranscriptionResult(text=payload, model=model_name, status=200, raw_response=response)
        response_dict = payload if isinstance(payload, dict) else {}
        return AudioTranscriptionResult(
            text=str(response_dict.get("text") or ""),
            language=str(response_dict.get("language") or "") or None,
            duration_seconds=float(response_dict["duration"]) if response_dict.get("duration") is not None else None,
            segments=[dict(item) for item in response_dict.get("segments") or []] or None,
            words=[dict(item) for item in response_dict.get("words") or []] or None,
            model=model_name,
            status=200,
            raw_response=response,
        )

    @staticmethod
    def _vector_store_resource_from_response(response: Any) -> VectorStoreResource:
        payload = OpenAIProvider._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        file_counts = response_dict.get("file_counts")
        metadata = response_dict.get("metadata")
        return VectorStoreResource(
            vector_store_id=str(response_dict.get("id") or ""),
            name=str(response_dict.get("name") or "") or None,
            status=str(response_dict.get("status") or "") or None,
            file_counts=dict(file_counts) if isinstance(file_counts, dict) else None,
            metadata=dict(metadata) if isinstance(metadata, dict) else None,
            usage_bytes=int(response_dict.get("usage_bytes", 0)) if response_dict.get("usage_bytes") is not None else None,
            expires_at=response_dict.get("expires_at"),
            last_active_at=response_dict.get("last_active_at"),
            raw_response=response,
        )

    def _vector_stores_page_from_response(self, response: Any) -> VectorStoresPage:
        payload = self._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        items = [
            self._vector_store_resource_from_response(item)
            for item in response_dict.get("data") or []
            if item is not None
        ]
        return VectorStoresPage(
            items=items,
            first_id=str(response_dict.get("first_id") or "") or None,
            last_id=str(response_dict.get("last_id") or "") or None,
            has_more=bool(response_dict.get("has_more")),
            raw_response=response,
        )

    def _vector_store_search_result_from_response(
        self,
        response: Any,
        *,
        vector_store_id: str,
        query: str | list[str],
    ) -> VectorStoreSearchResult:
        payload = self._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        results = [dict(item) for item in response_dict.get("data") or [] if isinstance(item, dict)]
        return VectorStoreSearchResult(
            vector_store_id=vector_store_id,
            query=query,
            results=results,
            raw_response=response,
        )

    @staticmethod
    def _vector_store_file_resource_from_response(response: Any, *, vector_store_id: str) -> VectorStoreFileResource:
        payload = OpenAIProvider._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        attributes = response_dict.get("attributes")
        chunking_strategy = response_dict.get("chunking_strategy")
        resolved_vector_store_id = str(response_dict.get("vector_store_id") or vector_store_id)
        return VectorStoreFileResource(
            file_id=str(response_dict.get("id") or response_dict.get("file_id") or ""),
            vector_store_id=resolved_vector_store_id,
            status=str(response_dict.get("status") or "") or None,
            attributes=dict(attributes) if isinstance(attributes, dict) else None,
            usage_bytes=int(response_dict.get("usage_bytes", 0)) if response_dict.get("usage_bytes") is not None else None,
            chunking_strategy=dict(chunking_strategy) if isinstance(chunking_strategy, dict) else None,
            raw_response=response,
        )

    def _vector_store_files_page_from_response(self, response: Any, *, vector_store_id: str) -> VectorStoreFilesPage:
        payload = self._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        items = [
            self._vector_store_file_resource_from_response(item, vector_store_id=vector_store_id)
            for item in response_dict.get("data") or []
            if item is not None
        ]
        return VectorStoreFilesPage(
            items=items,
            first_id=str(response_dict.get("first_id") or "") or None,
            last_id=str(response_dict.get("last_id") or "") or None,
            has_more=bool(response_dict.get("has_more")),
            raw_response=response,
        )

    def _vector_store_file_content_result_from_response(
        self,
        response: Any,
        *,
        vector_store_id: str,
        file_id: str,
    ) -> VectorStoreFileContentResult:
        payload = self._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        return VectorStoreFileContentResult(
            file_id=file_id,
            vector_store_id=vector_store_id,
            chunks=[dict(item) for item in response_dict.get("data") or [] if isinstance(item, dict)],
            raw_response=response,
        )

    @staticmethod
    def _vector_store_file_batch_resource_from_response(
        response: Any,
        *,
        vector_store_id: str,
    ) -> VectorStoreFileBatchResource:
        payload = OpenAIProvider._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        file_counts = response_dict.get("file_counts")
        resolved_vector_store_id = str(response_dict.get("vector_store_id") or vector_store_id)
        return VectorStoreFileBatchResource(
            batch_id=str(response_dict.get("id") or response_dict.get("batch_id") or ""),
            vector_store_id=resolved_vector_store_id,
            status=str(response_dict.get("status") or "") or None,
            file_counts=dict(file_counts) if isinstance(file_counts, dict) else None,
            raw_response=response,
        )

    @staticmethod
    def _fine_tuning_job_result_from_response(response: Any) -> FineTuningJobResult:
        payload = OpenAIProvider._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        metadata = response_dict.get("metadata")
        return FineTuningJobResult(
            job_id=str(response_dict.get("id") or ""),
            status=str(response_dict.get("status") or ""),
            base_model=str(response_dict.get("model") or "") or None,
            fine_tuned_model=str(response_dict.get("fine_tuned_model") or "") or None,
            created_at=response_dict.get("created_at"),
            finished_at=response_dict.get("finished_at"),
            trained_tokens=response_dict.get("trained_tokens"),
            training_file=str(response_dict.get("training_file") or "") or None,
            validation_file=str(response_dict.get("validation_file") or "") or None,
            result_files=[str(item) for item in response_dict.get("result_files") or []] or None,
            metadata=dict(metadata) if isinstance(metadata, dict) else None,
            raw_response=response,
        )

    def _fine_tuning_jobs_page_from_response(self, response: Any) -> FineTuningJobsPage:
        payload = self._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        items = [
            self._fine_tuning_job_result_from_response(item)
            for item in response_dict.get("data") or []
            if item is not None
        ]
        return FineTuningJobsPage(
            items=items,
            first_id=str(response_dict.get("first_id") or "") or None,
            last_id=str(response_dict.get("last_id") or "") or None,
            has_more=bool(response_dict.get("has_more")),
            raw_response=response,
        )

    def _fine_tuning_job_events_page_from_response(self, response: Any, *, job_id: str) -> FineTuningJobEventsPage:
        payload = self._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        return FineTuningJobEventsPage(
            job_id=job_id,
            events=[dict(item) for item in response_dict.get("data") or [] if isinstance(item, dict)],
            has_more=bool(response_dict.get("has_more")),
            raw_response=response,
        )

    @staticmethod
    def _realtime_client_secret_result_from_response(response: Any) -> RealtimeClientSecretResult:
        payload = OpenAIProvider._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        client_secret = response_dict.get("client_secret")
        session = response_dict.get("session")
        value = ""
        expires_at = response_dict.get("expires_at")
        if isinstance(client_secret, dict):
            value = str(client_secret.get("value") or "")
            if expires_at is None:
                expires_at = client_secret.get("expires_at")
        return RealtimeClientSecretResult(
            value=value,
            expires_at=expires_at,
            session=dict(session) if isinstance(session, dict) else None,
            raw_response=response,
        )

    @staticmethod
    def _realtime_transcription_session_result_from_response(response: Any) -> RealtimeTranscriptionSessionResult:
        payload = OpenAIProvider._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        client_secret = response_dict.get("client_secret")
        session = response_dict.get("session")
        value = ""
        expires_at = response_dict.get("expires_at")
        if isinstance(client_secret, dict):
            value = str(client_secret.get("value") or "")
            if expires_at is None:
                expires_at = client_secret.get("expires_at")
        elif isinstance(response_dict.get("value"), str):
            value = str(response_dict.get("value") or "")
        return RealtimeTranscriptionSessionResult(
            value=value,
            expires_at=expires_at,
            session=dict(session) if isinstance(session, dict) else None,
            raw_response=response,
        )

    @staticmethod
    def _realtime_call_result_from_response(
        response: Any,
        *,
        action: str,
        call_id: str | None = None,
    ) -> RealtimeCallResult:
        payload = response
        response_call_id = call_id
        sdp: str | None = None
        status = 200
        if hasattr(response, "response"):
            http_response = getattr(response, "response")
            response_call_id = response_call_id or str(http_response.headers.get("Location", "")).rstrip("/").split("/")[-1] or None
            sdp = getattr(response, "text", None)
            status = int(getattr(http_response, "status_code", 200) or 200)
            payload = response
        elif isinstance(response, dict):
            response_call_id = response_call_id or str(response.get("call_id") or response.get("id") or "") or None
            sdp = str(response.get("sdp") or "") or None
        return RealtimeCallResult(
            call_id=response_call_id,
            sdp=sdp,
            action=action,
            status=status,
            raw_response=payload,
        )

    @staticmethod
    def _webhook_event_result_from_unwrapped(event: Any) -> WebhookEventResult:
        payload = OpenAIProvider._serialize_responses_item(event)
        event_dict = payload if isinstance(payload, dict) else {}
        data = event_dict.get("data")
        return WebhookEventResult(
            event_id=str(event_dict.get("id") or "") or None,
            event_type=str(event_dict.get("type") or "") or None,
            data=dict(data) if isinstance(data, dict) else event_dict,
            raw_event=event,
        )

    @staticmethod
    def _file_resource_from_response(response: Any) -> FileResource:
        payload = OpenAIProvider._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        return FileResource(
            file_id=str(response_dict.get("id") or response_dict.get("file_id") or ""),
            filename=str(response_dict.get("filename") or "") or None,
            purpose=str(response_dict.get("purpose") or "") or None,
            bytes=int(response_dict.get("bytes", 0)) if response_dict.get("bytes") is not None else None,
            status=str(response_dict.get("status") or "") or None,
            media_type=str(response_dict.get("mime_type") or response_dict.get("media_type") or "") or None,
            created_at=response_dict.get("created_at"),
            raw_response=response,
        )

    def _files_page_from_response(self, response: Any) -> FilesPage:
        payload = self._serialize_responses_item(response)
        response_dict = payload if isinstance(payload, dict) else {}
        items = [
            self._file_resource_from_response(item)
            for item in response_dict.get("data") or []
            if item is not None
        ]
        return FilesPage(
            items=items,
            first_id=str(response_dict.get("first_id") or "") or None,
            last_id=str(response_dict.get("last_id") or "") or None,
            has_more=bool(response_dict.get("has_more")),
            raw_response=response,
        )

    @staticmethod
    def _format_deep_research_rewrite_input(
        prompt: str,
        *,
        clarifications: str | list[str] | None = None,
    ) -> str:
        sections = [f"User research request:\n{prompt.strip()}"]
        if clarifications:
            if isinstance(clarifications, str):
                clarifications_text = clarifications.strip()
            else:
                clarifications_text = "\n".join(f"- {item}" for item in clarifications if str(item).strip())
            if clarifications_text:
                sections.append(f"Clarifications and constraints:\n{clarifications_text}")
        sections.append("Return the rewritten deep-research instructions only.")
        return "\n\n".join(section for section in sections if section)

    def _require_responses_api(self, feature: str) -> None:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError(f"{feature} requires use_responses_api=True")

    @staticmethod
    def _coerce_mcp_tool(tool: ResponsesMCPTool | dict[str, Any]) -> dict[str, Any]:
        if isinstance(tool, ResponsesMCPTool):
            return tool.to_dict()
        return dict(tool)

    def _normalize_deep_research_mcp_tool(self, tool: Any) -> Any:
        if isinstance(tool, ResponsesMCPTool):
            normalized_tool = tool.to_dict()
        elif isinstance(tool, dict) and str(tool.get("type") or "") == "mcp":
            normalized_tool = dict(tool)
        else:
            return tool

        require_approval = normalized_tool.get("require_approval")
        if require_approval not in (None, "never"):
            raise ValueError("Deep research MCP tools must set require_approval='never'.")
        normalized_tool["require_approval"] = "never"
        return normalized_tool

    async def _complete_with_responses_tools(
        self,
        prompt: str,
        *,
        tools: list[Any],
        model: str | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        self._require_responses_api("Hosted OpenAI tool workflows")
        resolved_tools = list(tools)
        extra_tools = list(kwargs.pop("tools", []) or [])
        resolved_tools.extend(extra_tools)
        if not resolved_tools:
            raise ValueError("At least one Responses tool descriptor is required.")
        return await self.complete(
            prompt,
            tools=resolved_tools,
            model=str(model or self.model_name),
            **kwargs,
        )

    @staticmethod
    def _normalized_refusal_from_output_items(items: list[NormalizedOutputItem] | None) -> str | None:
        if not items:
            return None
        refusals = [item.text for item in items if item.type == "refusal" and item.text]
        return "\n".join(refusals) if refusals else None

    @staticmethod
    def _normalize_serialized_responses_output_items(
        items: list[dict[str, Any]] | None,
        *,
        alias_to_original: dict[str, str],
    ) -> list[NormalizedOutputItem] | None:
        if not items:
            return None

        normalized: list[NormalizedOutputItem] = []
        for item in items:
            item_type = str(item.get("type") or "")
            item_id = str(item.get("id") or "") or None
            item_status = str(item.get("status") or "") or None

            if item_type == "message":
                for part in item.get("content") or []:
                    if not isinstance(part, dict):
                        continue
                    part_type = str(part.get("type") or "")
                    if part_type == "output_text":
                        details: dict[str, Any] = {}
                        annotations = part.get("annotations")
                        if isinstance(annotations, list) and annotations:
                            details["annotations"] = annotations
                        normalized.append(
                            NormalizedOutputItem(
                                type="output_text",
                                id=item_id,
                                status=item_status,
                                text=str(part.get("text") or ""),
                                details=details,
                            )
                        )
                    elif part_type == "refusal":
                        normalized.append(
                            NormalizedOutputItem(
                                type="refusal",
                                id=item_id,
                                status=item_status,
                                text=str(part.get("refusal") or ""),
                            )
                        )
                continue

            if item_type == "function_call":
                original_name = str(item.get("name") or "")
                normalized.append(
                    NormalizedOutputItem(
                        type="function_call",
                        id=item_id,
                        call_id=str(item.get("call_id") or item.get("id") or "") or None,
                        status=item_status,
                        name=alias_to_original.get(original_name, original_name) or None,
                        details={"arguments": str(item.get("arguments") or "")},
                    )
                )
                continue

            if item_type == "reasoning":
                summary_texts = [
                    str(summary.get("text") or "")
                    for summary in item.get("summary") or []
                    if isinstance(summary, dict) and summary.get("text")
                ]
                content_texts = [
                    str(part.get("text") or "")
                    for part in item.get("content") or []
                    if isinstance(part, dict) and part.get("text")
                ]
                text = "\n".join(part for part in (summary_texts or content_texts) if part).strip() or None
                details: dict[str, Any] = {}
                encrypted_content = item.get("encrypted_content")
                if encrypted_content:
                    details["encrypted_content"] = str(encrypted_content)
                normalized.append(
                    NormalizedOutputItem(
                        type="reasoning",
                        id=item_id,
                        status=item_status,
                        text=text,
                        details=details,
                    )
                )
                continue

            if item_type == "compaction":
                details: dict[str, Any] = {}
                encrypted_content = item.get("encrypted_content")
                if encrypted_content:
                    details["encrypted_content"] = str(encrypted_content)
                created_by = item.get("created_by")
                if created_by:
                    details["created_by"] = str(created_by)
                normalized.append(
                    NormalizedOutputItem(
                        type="compaction",
                        id=item_id,
                        details=details,
                    )
                )
                continue

            if item_type == "file_search_call":
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        status=item_status,
                        details={
                            "queries": list(item.get("queries") or []),
                            "results": list(item.get("results") or []),
                        },
                    )
                )
                continue

            if item_type == "tool_search_call":
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        call_id=str(item.get("call_id") or "") or None,
                        status=item_status,
                        details={k: v for k, v in item.items() if k not in {"type", "id", "call_id", "status"}},
                    )
                )
                continue

            if item_type == "tool_search_output":
                loaded_tools = item.get("tools")
                if isinstance(loaded_tools, list):
                    loaded_tools = [
                        OpenAIProvider._rewrite_tool_definition_aliases(
                            tool,
                            alias_to_original=alias_to_original,
                        )
                        if isinstance(tool, dict)
                        else tool
                        for tool in loaded_tools
                    ]
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        call_id=str(item.get("call_id") or "") or None,
                        status=item_status,
                        details={
                            **{
                                k: v
                                for k, v in item.items()
                                if k not in {"type", "id", "call_id", "status", "tools"}
                            },
                            "tools": loaded_tools if isinstance(loaded_tools, list) else list(item.get("tools") or []),
                        },
                    )
                )
                continue

            if item_type == "web_search_call":
                action = item.get("action")
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        status=item_status,
                        details={"action": action if isinstance(action, dict) else {"value": action}},
                    )
                )
                continue

            if item_type == "computer_call":
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        call_id=str(item.get("call_id") or "") or None,
                        status=item_status,
                        details={
                            "action": item.get("action"),
                            "pending_safety_checks": list(item.get("pending_safety_checks") or []),
                        },
                    )
                )
                continue

            if item_type == "code_interpreter_call":
                logs: list[str] = []
                image_urls: list[str] = []
                for output in item.get("outputs") or []:
                    if not isinstance(output, dict):
                        continue
                    output_type = str(output.get("type") or "")
                    if output_type == "logs" and output.get("logs"):
                        logs.append(str(output.get("logs")))
                    elif output_type == "image" and output.get("url"):
                        image_urls.append(str(output.get("url")))
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        status=item_status,
                        text="\n".join(logs) if logs else None,
                        url=image_urls[0] if image_urls else None,
                        details={
                            "code": item.get("code"),
                            "container_id": item.get("container_id"),
                            "image_urls": image_urls,
                        },
                    )
                )
                continue

            if item_type == "image_generation_call":
                result = item.get("result")
                result_text = str(result or "") or None
                result_url = result_text if isinstance(result_text, str) and result_text.startswith(("http://", "https://")) else None
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        status=item_status,
                        text=None if result_url else result_text,
                        url=result_url,
                    )
                )
                continue

            if item_type in {"shell_call", "apply_patch_call", "custom_tool_call"}:
                text: str | None = None
                if item_type == "custom_tool_call":
                    text = str(item.get("input") or "") or None
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        call_id=str(item.get("call_id") or "") or None,
                        status=item_status,
                        name=str(item.get("name") or "") or None,
                        text=text,
                        details={k: v for k, v in item.items() if k not in {"type", "id", "call_id", "status", "name", "input"}},
                    )
                )
                continue

            if item_type in {
                "function_call_output",
                "shell_call_output",
                "apply_patch_call_output",
                "custom_tool_call_output",
                "computer_call_output",
                "mcp_call",
                "mcp_list_tools",
                "mcp_approval_request",
                "mcp_approval_response",
            }:
                output_value = item.get("output")
                text = output_value or item.get("error")
                if isinstance(text, list):
                    text = json.dumps(text)
                details = {k: v for k, v in item.items() if k not in {"type", "id", "call_id", "status", "name", "output", "error"}}
                url: str | None = None
                if isinstance(output_value, dict):
                    if isinstance(output_value.get("image_url"), str):
                        url = output_value["image_url"]
                    elif isinstance(output_value.get("url"), str):
                        url = output_value["url"]
                    details["output"] = dict(output_value)
                elif output_value is not None and not isinstance(output_value, (str, int, float, list)):
                    details["output"] = output_value
                normalized.append(
                    NormalizedOutputItem(
                        type=item_type,
                        id=item_id,
                        call_id=str(item.get("call_id") or "") or None,
                        status=item_status,
                        name=str(item.get("name") or "") or None,
                        text=str(text) if isinstance(text, (str, int, float)) and text is not None else None,
                        url=url,
                        details=details,
                    )
                )
                continue

        return normalized or None

    @staticmethod
    def _normalize_tool_choice(
        tool_choice: str | dict[str, Any] | None,
        *,
        use_responses_api: bool,
        original_to_alias: dict[str, str],
    ) -> str | dict[str, Any] | None:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            if tool_choice in ("auto", "none", "required"):
                return tool_choice
            aliased_name = original_to_alias.get(tool_choice, tool_choice)
            if use_responses_api:
                return {"type": "function", "name": aliased_name}
            return {"type": "function", "function": {"name": aliased_name}}
        if not isinstance(tool_choice, dict):
            return tool_choice

        def _rewrite_tool_descriptor(descriptor: Any) -> Any:
            if not isinstance(descriptor, dict):
                return descriptor
            descriptor_copy = dict(descriptor)
            if descriptor_copy.get("type") == "function":
                name = descriptor_copy.get("name")
                if isinstance(name, str) and name:
                    descriptor_copy["name"] = original_to_alias.get(name, name)
                function_payload = descriptor_copy.get("function")
                if isinstance(function_payload, dict):
                    function_copy = dict(function_payload)
                    function_name = function_copy.get("name")
                    if isinstance(function_name, str) and function_name:
                        function_copy["name"] = original_to_alias.get(function_name, function_name)
                    descriptor_copy["function"] = function_copy
            return descriptor_copy

        choice = dict(tool_choice)
        choice_type = str(choice.get("type") or "")
        if choice_type == "function":
            name = choice.get("name")
            if isinstance(name, str) and name:
                choice["name"] = original_to_alias.get(name, name)
            function_payload = choice.get("function")
            if isinstance(function_payload, dict):
                function_copy = dict(function_payload)
                function_name = function_copy.get("name")
                if isinstance(function_name, str) and function_name:
                    function_copy["name"] = original_to_alias.get(function_name, function_name)
                choice["function"] = function_copy
        elif choice_type == "allowed_tools" and isinstance(choice.get("tools"), list):
            choice["tools"] = [_rewrite_tool_descriptor(item) for item in choice["tools"]]
        return choice

    @staticmethod
    def _extract_responses_output(
        response: Any,
        *,
        alias_to_original: dict[str, str],
        parsed_text_format: type | None = None,
    ) -> tuple[str | None, list[ToolCall] | None, str | None, str | None]:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        output = list(getattr(response, "output", []) or [])
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", []) or []:
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text:
                            content_parts.append(text)
                    elif part_type == "refusal":
                        refusal = getattr(part, "refusal", None)
                        if isinstance(refusal, str) and refusal:
                            content_parts.append(refusal)
            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(item, "call_id", None) or getattr(item, "id", None) or ""),
                        name=alias_to_original.get(str(getattr(item, "name", None) or ""), str(getattr(item, "name", None) or "")),
                        arguments=str(getattr(item, "arguments", None) or ""),
                    )
                )
            elif item_type == "reasoning":
                for summary in getattr(item, "summary", []) or []:
                    text = getattr(summary, "text", None)
                    if isinstance(text, str) and text:
                        reasoning_parts.append(text)
                if not reasoning_parts:
                    for part in getattr(item, "content", []) or []:
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text:
                            reasoning_parts.append(text)

        parsed_output = getattr(response, "output_parsed", None) if parsed_text_format is not None else None
        if parsed_output is not None and not content_parts:
            if hasattr(parsed_output, "model_dump"):
                content_parts.append(json.dumps(parsed_output.model_dump()))
            elif hasattr(parsed_output, "dict"):
                content_parts.append(json.dumps(parsed_output.dict()))
            else:
                content_parts.append(json.dumps(parsed_output))

        content: str | None
        helper_text = getattr(response, "output_text", None)
        if isinstance(helper_text, str) and helper_text:
            content = helper_text
        elif content_parts:
            content = "".join(content_parts)
        else:
            content = None

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
        finish_reason: str | None = None
        incomplete_details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete_details, "reason", None) if incomplete_details is not None else None
        status = str(getattr(response, "status", "") or "")
        if tool_calls:
            finish_reason = "tool_calls"
        elif incomplete_reason == "max_output_tokens":
            finish_reason = "length"
        elif incomplete_reason:
            finish_reason = str(incomplete_reason)
        elif status == "completed":
            finish_reason = "stop"
        elif status in {"queued", "in_progress"}:
            finish_reason = None
        elif status:
            finish_reason = status

        return content, (tool_calls or None), reasoning, finish_reason

    @staticmethod
    def _response_error_message(response: Any) -> str | None:
        error = getattr(response, "error", None)
        if isinstance(error, dict):
            message = error.get("message")
            return str(message) if message else None
        message = getattr(error, "message", None)
        if isinstance(message, str) and message:
            return message
        if isinstance(error, str) and error:
            return error
        return None

    def _background_response_result_from_response(
        self,
        response: Any,
        *,
        alias_to_original: dict[str, str] | None = None,
        parsed_text_format: type | None = None,
    ) -> BackgroundResponseResult:
        alias_map = alias_to_original or {}
        content, tool_calls, reasoning, finish_reason = self._extract_responses_output(
            response,
            alias_to_original=alias_map,
            parsed_text_format=parsed_text_format,
        )
        raw_usage = response.usage.to_dict() if getattr(response, "usage", None) else {}
        usage = self.parse_usage(raw_usage) if raw_usage else None
        provider_items = self._serialize_responses_output_items(response)
        output_items = self._normalize_serialized_responses_output_items(provider_items, alias_to_original=alias_map)
        refusal = self._normalized_refusal_from_output_items(output_items)
        completion: CompletionResult | None = None
        if any(
            [
                content is not None,
                tool_calls,
                reasoning,
                refusal is not None,
                usage is not None,
                finish_reason is not None,
                output_items,
                provider_items,
            ]
        ):
            completion = CompletionResult(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                model=str(getattr(response, "model", None) or self.model_name),
                finish_reason=finish_reason,
                status=200,
                raw_response=response,
                reasoning=reasoning,
                refusal=refusal,
                output_items=output_items,
                provider_items=provider_items,
            )

        return BackgroundResponseResult(
            response_id=str(getattr(response, "id", None) or ""),
            lifecycle_status=str(getattr(response, "status", None) or "unknown"),
            completion=completion,
            error=self._response_error_message(response),
            raw_response=response,
        )

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

    @staticmethod
    def _normalize_realtime_connection_model(model_name: str | None) -> str | None:
        if model_name is None:
            return None
        normalized = str(model_name or "").strip()
        lowered = normalized.lower()
        if not normalized:
            return None
        if "transcribe" in lowered or lowered == "whisper-1":
            return "gpt-realtime"
        return normalized

    async def complete(
        self,
        messages: MessageInput,
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        include: list[str] | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
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
            include: Responses include fields such as ``["reasoning.encrypted_content"]``
            prompt_cache_key: Explicit prompt-cache routing key for OpenAI prompt caching
            prompt_cache_retention: Prompt cache retention policy such as ``"in_memory"`` or ``"24h"``
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
        use_responses_api = bool(getattr(self, "use_responses_api", False))
        api_messages = self._messages_to_api_format(msg_objects, responses_api=use_responses_api)
        source_messages = [msg.to_dict() for msg in msg_objects]
        self._validate_tool_configuration(tools=tools, use_responses_api=use_responses_api)

        # Build params
        params: dict[str, Any] = {"model": self.model_name}
        params["input" if use_responses_api else "messages"] = api_messages
        alias_to_original: dict[str, str] = {}
        original_to_alias: dict[str, str] = {}

        # Add optional parameters
        if tools:
            provider_tools, alias_to_original, original_to_alias = self._prepare_openai_tools(
                tools,
                responses_api=use_responses_api,
            )
            if provider_tools:
                params["tools"] = provider_tools
        normalized_tool_choice = self._normalize_tool_choice(
            tool_choice,
            use_responses_api=use_responses_api,
            original_to_alias=original_to_alias,
        )
        if normalized_tool_choice is not None:
            params["tool_choice"] = normalized_tool_choice

        alias_to_original, original_to_alias = self._collect_message_tool_name_aliases(
            source_messages,
            seed_alias_to_original=alias_to_original,
            seed_original_to_alias=original_to_alias,
        )
        if original_to_alias:
            if use_responses_api:
                api_messages = self._rewrite_responses_input_items_for_openai_tool_aliases(
                    api_messages,
                    original_to_alias=original_to_alias,
                )
                params["input"] = api_messages
            else:
                api_messages = self._rewrite_messages_for_openai_tool_aliases(
                    api_messages,
                    original_to_alias=original_to_alias,
                )
                params["messages"] = api_messages

        parsed_text_format: type | None = None
        if use_responses_api:
            self._set_temperature(params, temperature)
            if max_tokens is not None:
                params["max_output_tokens"] = max_tokens
            text_config, parsed_text_format = self._responses_text_config_and_parser(response_format, source_messages)
            if text_config:
                params["text"] = text_config
        else:
            self._set_temperature(params, temperature)
            self._set_completion_token_limit(params, max_tokens)

            # Handle response format
            rf = self._normalize_response_format(response_format, source_messages)
            if rf:
                params["response_format"] = rf

        # Handle reasoning params
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        if reasoning:
            params["reasoning"] = reasoning
        if include is not None:
            params["include"] = list(include)
        if prompt_cache_key is not None:
            params["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            params["prompt_cache_retention"] = prompt_cache_retention
        params = self._check_reasoning_params(params, "responses" if use_responses_api else "completions")

        # Add any extra kwargs
        params.update(kwargs)

        # Use responses API if enabled
        if use_responses_api:
            return await self._complete_responses(
                api_messages,
                params,
                alias_to_original=alias_to_original,
                parsed_text_format=parsed_text_format,
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
                    response_format_payload = params.get("response_format")
                    if isinstance(response_format_payload, dict):
                        response = await self.client.chat.completions.create(**params)
                        msg = response.choices[0].message
                        content = self._coerce_chat_message_content(getattr(msg, "content", None))

                        # Parse JSON if requested
                        if response_format_payload.get("type") in ("json_object", "json_schema"):
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
        alias_to_original: dict[str, str] | None = None,
        parsed_text_format: type | None = None,
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
        responses_params = dict(params)
        responses_params["input"] = responses_params.pop("input", api_messages)
        responses_params.pop("messages", None)
        alias_map = alias_to_original or {}

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
                    if parsed_text_format is not None:
                        response = await self.client.responses.parse(
                            **responses_params,
                            text_format=parsed_text_format,
                        )
                    else:
                        response = await self.client.responses.create(**responses_params)

                    content, tool_calls, reasoning, finish_reason = self._extract_responses_output(
                        response,
                        alias_to_original=alias_map,
                        parsed_text_format=parsed_text_format,
                    )

                    # Parse usage
                    raw_usage = response.usage.to_dict() if response.usage else {}
                    usage = self.parse_usage(raw_usage)
                    limit_ctx.output_tokens = usage.output_tokens
                    provider_items = self._serialize_responses_output_items(response)
                    output_items = self._normalize_serialized_responses_output_items(provider_items, alias_to_original=alias_map)

                    return CompletionResult(
                        content=content,
                        tool_calls=tool_calls,
                        usage=usage,
                        model=str(getattr(response, "model", None) or responses_params.get("model") or self.model_name),
                        finish_reason=finish_reason,
                        status=200,
                        raw_response=response,
                        reasoning=reasoning,
                        refusal=self._normalized_refusal_from_output_items(output_items),
                        output_items=output_items,
                        provider_items=provider_items,
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

    async def _iter_responses_stream_events(
        self,
        event_stream: AsyncIterator[Any],
        *,
        model_name_hint: str | None = None,
        alias_to_original: dict[str, str] | None = None,
        parsed_text_format: type | None = None,
        limit_ctx: Any | None = None,
    ) -> AsyncIterator[StreamEvent]:
        alias_map = alias_to_original or {}
        content_buffer = ""
        reasoning_buffer = ""
        tool_calls_buffer: dict[int, dict[str, Any]] = {}

        async for event in event_stream:
            event_type = getattr(event, "type", None)
            sequence_number = getattr(event, "sequence_number", None)

            if event_type == "response.output_text.delta":
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    content_buffer += delta
                    yield StreamEvent(type=StreamEventType.TOKEN, data=delta, sequence_number=sequence_number)
                continue

            if event_type in {"response.reasoning_text.delta", "response.reasoning_summary_text.delta"}:
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    reasoning_buffer += delta
                    yield StreamEvent(type=StreamEventType.REASONING, data=delta, sequence_number=sequence_number)
                continue

            if event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    index = int(getattr(event, "output_index", len(tool_calls_buffer)))
                    call_id = str(getattr(item, "call_id", None) or getattr(item, "id", None) or "")
                    name = str(getattr(item, "name", None) or "")
                    tool_calls_buffer[index] = {
                        "id": call_id,
                        "name": name,
                        "arguments": str(getattr(item, "arguments", "") or ""),
                    }
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_START,
                        data=ToolCallDelta(
                            id=call_id,
                            index=index,
                            name=alias_map.get(name, name),
                        ),
                        sequence_number=sequence_number,
                    )
                continue

            if event_type == "response.function_call_arguments.delta":
                index = int(getattr(event, "output_index", len(tool_calls_buffer)))
                delta = str(getattr(event, "delta", "") or "")
                current = tool_calls_buffer.setdefault(index, {"id": "", "name": "", "arguments": ""})
                current["arguments"] += delta
                yield StreamEvent(
                    type=StreamEventType.TOOL_CALL_DELTA,
                    data=ToolCallDelta(
                        id=str(current.get("id") or ""),
                        index=index,
                        arguments_delta=delta,
                    ),
                    sequence_number=sequence_number,
                )
                continue

            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    index = int(getattr(event, "output_index", len(tool_calls_buffer)))
                    current = tool_calls_buffer.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    current["id"] = str(getattr(item, "call_id", None) or getattr(item, "id", None) or current["id"])
                    current["name"] = str(getattr(item, "name", None) or current["name"])
                    current["arguments"] = str(getattr(item, "arguments", None) or current["arguments"])
                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_END,
                        data=ToolCall(
                            id=str(current["id"]),
                            name=alias_map.get(str(current["name"]), str(current["name"])),
                            arguments=str(current["arguments"]),
                        ),
                        sequence_number=sequence_number,
                    )
                continue

            if event_type == "response.completed":
                response = getattr(event, "response", None)
                raw_usage = response.usage.to_dict() if response is not None and getattr(response, "usage", None) else {}
                usage = self.parse_usage(raw_usage)
                if limit_ctx is not None:
                    limit_ctx.output_tokens = usage.output_tokens
                yield StreamEvent(type=StreamEventType.USAGE, data=usage, sequence_number=sequence_number)

                content, tool_calls, final_reasoning, finish_reason = self._extract_responses_output(
                    response,
                    alias_to_original=alias_map,
                    parsed_text_format=parsed_text_format,
                )
                provider_items = self._serialize_responses_output_items(response)
                output_items = self._normalize_serialized_responses_output_items(provider_items, alias_to_original=alias_map)
                final_result = CompletionResult(
                    content=content if content is not None else (content_buffer or None),
                    tool_calls=tool_calls,
                    usage=usage,
                    model=str(getattr(response, "model", None) or model_name_hint or self.model_name),
                    finish_reason=finish_reason,
                    status=200,
                    raw_response=response,
                    reasoning=final_reasoning or (reasoning_buffer or None),
                    refusal=self._normalized_refusal_from_output_items(output_items),
                    output_items=output_items,
                    provider_items=provider_items,
                )
                yield StreamEvent(type=StreamEventType.DONE, data=final_result, sequence_number=sequence_number)
                return

            if event_type in {"response.failed", "response.error"}:
                response = getattr(event, "response", None)
                failure = normalize_provider_failure(
                    status=500,
                    message=str(self._response_error_message(response) or "Responses stream failed"),
                    provider="openai",
                    model=self.model_name,
                    operation="stream",
                )
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data=failure_to_stream_error_data(failure),
                    sequence_number=sequence_number,
                )
                return

    async def _stream_responses(
        self,
        params: dict[str, Any],
        api_messages: list[dict[str, Any]],
        *,
        alias_to_original: dict[str, str] | None = None,
        parsed_text_format: type | None = None,
    ) -> AsyncIterator[StreamEvent]:
        input_tokens = self.count_tokens(api_messages)

        async with self.limiter.limit(tokens=input_tokens, requests=1) as limit_ctx:
            try:
                stream_manager_kwargs = dict(params)
                if parsed_text_format is not None:
                    stream_manager_kwargs["text_format"] = parsed_text_format

                async with self.client.responses.stream(**stream_manager_kwargs) as stream:
                    async for emitted_event in self._iter_responses_stream_events(
                        stream,
                        model_name_hint=str(params.get("model") or self.model_name),
                        alias_to_original=alias_to_original,
                        parsed_text_format=parsed_text_format,
                        limit_ctx=limit_ctx,
                    ):
                        yield emitted_event
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

    async def stream(
        self,
        messages: MessageInput,
        *,
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: str | dict[str, Any] | type | None = None,
        reasoning_effort: str | None = None,
        reasoning: dict[str, Any] | None = None,
        include: list[str] | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
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
            include: Responses include fields such as ``["reasoning.encrypted_content"]``
            prompt_cache_key: Explicit prompt-cache routing key for OpenAI prompt caching
            prompt_cache_retention: Prompt cache retention policy such as ``"in_memory"`` or ``"24h"``
            **kwargs: Additional API parameters

        Yields:
            StreamEvent objects for each chunk
        """
        # Normalize messages
        msg_objects = self._normalize_messages(messages)
        use_responses_api = bool(getattr(self, "use_responses_api", False))
        api_messages = self._messages_to_api_format(msg_objects, responses_api=use_responses_api)
        source_messages = [msg.to_dict() for msg in msg_objects]
        self._validate_tool_configuration(tools=tools, use_responses_api=use_responses_api)

        # Build params
        params: dict[str, Any] = {
            "model": self.model_name,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        params["input" if use_responses_api else "messages"] = api_messages
        alias_to_original: dict[str, str] = {}
        original_to_alias: dict[str, str] = {}

        # Add optional parameters
        if tools:
            provider_tools, alias_to_original, original_to_alias = self._prepare_openai_tools(
                tools,
                responses_api=use_responses_api,
            )
            if provider_tools:
                params["tools"] = provider_tools
        normalized_tool_choice = self._normalize_tool_choice(
            tool_choice,
            use_responses_api=use_responses_api,
            original_to_alias=original_to_alias,
        )
        if normalized_tool_choice is not None:
            params["tool_choice"] = normalized_tool_choice

        alias_to_original, original_to_alias = self._collect_message_tool_name_aliases(
            source_messages,
            seed_alias_to_original=alias_to_original,
            seed_original_to_alias=original_to_alias,
        )
        if original_to_alias:
            if use_responses_api:
                api_messages = self._rewrite_responses_input_items_for_openai_tool_aliases(
                    api_messages,
                    original_to_alias=original_to_alias,
                )
                params["input"] = api_messages
            else:
                api_messages = self._rewrite_messages_for_openai_tool_aliases(
                    api_messages,
                    original_to_alias=original_to_alias,
                )
                params["messages"] = api_messages

        parsed_text_format: type | None = None
        if use_responses_api:
            if temperature is not None:
                params["temperature"] = temperature
            if max_tokens is not None:
                params["max_output_tokens"] = max_tokens
            text_config, parsed_text_format = self._responses_text_config_and_parser(response_format, source_messages)
            if text_config:
                params["text"] = text_config
        else:
            self._set_temperature(params, temperature)
            self._set_completion_token_limit(params, max_tokens)

            # Handle response format
            rf = self._normalize_response_format(response_format, source_messages)
            if rf:
                params["response_format"] = rf

        # Handle reasoning params
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        if reasoning:
            params["reasoning"] = reasoning
        if include is not None:
            params["include"] = list(include)
        if prompt_cache_key is not None:
            params["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            params["prompt_cache_retention"] = prompt_cache_retention
        params = self._check_reasoning_params(params, "responses" if use_responses_api else "completions")

        params.update(kwargs)

        # Emit metadata event
        yield StreamEvent(type=StreamEventType.META, data={"model": str(params.get("model") or self.model_name), "stream": True})

        if use_responses_api:
            async for event in self._stream_responses(
                params,
                api_messages,
                alias_to_original=alias_to_original,
                parsed_text_format=parsed_text_format,
            ):
                yield event
            return

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

    async def moderate(
        self,
        inputs: str | list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModerationResult:
        params: dict[str, Any] = {"input": inputs}
        if "model" not in kwargs:
            params["model"] = self.model_name
        params.update(kwargs)

        async with self.limiter.limit(tokens=self.count_tokens(inputs), requests=1):
            try:
                response = await self.client.moderations.create(**params)
                return self._moderation_result_from_response(response)
            except openai.APIConnectionError as e:
                failure = normalize_provider_failure(
                    status=503,
                    message=str(e.__cause__ or e),
                    provider="openai",
                    model=self.model_name,
                    operation="moderate",
                )
                return ModerationResult(flagged=False, model=self.model_name, status=503, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.RateLimitError as e:
                failure = self._failure(error=e, operation="moderate")
                return ModerationResult(flagged=False, model=self.model_name, status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.APIStatusError as e:
                failure = self._failure(error=e, operation="moderate")
                return ModerationResult(flagged=False, model=self.model_name, status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})

    async def generate_image(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        params: dict[str, Any] = {"prompt": prompt}
        if "model" not in kwargs:
            params["model"] = self.model_name
        params.update(kwargs)

        async with self.limiter.limit(tokens=self.count_tokens(prompt), requests=1):
            try:
                response = await self.client.images.generate(**params)
                return self._image_generation_result_from_response(response, model_name=str(params.get("model") or self.model_name))
            except openai.APIConnectionError as e:
                failure = normalize_provider_failure(
                    status=503,
                    message=str(e.__cause__ or e),
                    provider="openai",
                    model=str(params.get("model") or self.model_name),
                    operation="generate_image",
                )
                return ImageGenerationResult(images=[], model=str(params.get("model") or self.model_name), status=503, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.RateLimitError as e:
                failure = self._failure(error=e, operation="generate_image")
                return ImageGenerationResult(images=[], model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.APIStatusError as e:
                failure = self._failure(error=e, operation="generate_image")
                return ImageGenerationResult(images=[], model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})

    async def edit_image(
        self,
        image: Any,
        prompt: str,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        params: dict[str, Any] = {"image": image, "prompt": prompt}
        if "model" not in kwargs:
            params["model"] = self.model_name
        params.update(kwargs)

        async with self.limiter.limit(tokens=self.count_tokens(prompt), requests=1):
            try:
                response = await self.client.images.edit(**params)
                return self._image_generation_result_from_response(response, model_name=str(params.get("model") or self.model_name))
            except openai.APIConnectionError as e:
                failure = normalize_provider_failure(
                    status=503,
                    message=str(e.__cause__ or e),
                    provider="openai",
                    model=str(params.get("model") or self.model_name),
                    operation="edit_image",
                )
                return ImageGenerationResult(images=[], model=str(params.get("model") or self.model_name), status=503, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.RateLimitError as e:
                failure = self._failure(error=e, operation="edit_image")
                return ImageGenerationResult(images=[], model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.APIStatusError as e:
                failure = self._failure(error=e, operation="edit_image")
                return ImageGenerationResult(images=[], model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})

    async def transcribe_audio(
        self,
        file: Any,
        **kwargs: Any,
    ) -> AudioTranscriptionResult:
        params: dict[str, Any] = {"file": file}
        if "model" not in kwargs:
            params["model"] = self.model_name
        params.update(kwargs)

        async with self.limiter.limit(tokens=0, requests=1):
            try:
                response = await self.client.audio.transcriptions.create(**params)
                return self._audio_transcription_result_from_response(response, model_name=str(params.get("model") or self.model_name))
            except openai.APIConnectionError as e:
                failure = normalize_provider_failure(
                    status=503,
                    message=str(e.__cause__ or e),
                    provider="openai",
                    model=str(params.get("model") or self.model_name),
                    operation="transcribe_audio",
                )
                return AudioTranscriptionResult(text="", model=str(params.get("model") or self.model_name), status=503, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.RateLimitError as e:
                failure = self._failure(error=e, operation="transcribe_audio")
                return AudioTranscriptionResult(text="", model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.APIStatusError as e:
                failure = self._failure(error=e, operation="transcribe_audio")
                return AudioTranscriptionResult(text="", model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})

    async def translate_audio(
        self,
        file: Any,
        **kwargs: Any,
    ) -> AudioTranscriptionResult:
        params: dict[str, Any] = {"file": file}
        if "model" not in kwargs:
            params["model"] = self.model_name
        params.update(kwargs)

        async with self.limiter.limit(tokens=0, requests=1):
            try:
                response = await self.client.audio.translations.create(**params)
                return self._audio_transcription_result_from_response(response, model_name=str(params.get("model") or self.model_name))
            except openai.APIConnectionError as e:
                failure = normalize_provider_failure(
                    status=503,
                    message=str(e.__cause__ or e),
                    provider="openai",
                    model=str(params.get("model") or self.model_name),
                    operation="translate_audio",
                )
                return AudioTranscriptionResult(text="", model=str(params.get("model") or self.model_name), status=503, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.RateLimitError as e:
                failure = self._failure(error=e, operation="translate_audio")
                return AudioTranscriptionResult(text="", model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.APIStatusError as e:
                failure = self._failure(error=e, operation="translate_audio")
                return AudioTranscriptionResult(text="", model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})

    async def synthesize_speech(
        self,
        text: str,
        *,
        voice: str,
        **kwargs: Any,
    ) -> AudioSpeechResult:
        params: dict[str, Any] = {"input": text, "voice": voice}
        if "model" not in kwargs:
            params["model"] = self.model_name
        params.update(kwargs)

        async with self.limiter.limit(tokens=self.count_tokens(text), requests=1):
            try:
                response = await self.client.audio.speech.create(**params)
                audio_bytes = bytes(getattr(response, "content", b"") or b"")
                if not audio_bytes and hasattr(response, "aread"):
                    audio_bytes = await response.aread()
                return AudioSpeechResult(
                    audio=audio_bytes,
                    format=str(params.get("response_format") or "mp3"),
                    model=str(params.get("model") or self.model_name),
                    status=200,
                    raw_response=response,
                )
            except openai.APIConnectionError as e:
                failure = normalize_provider_failure(
                    status=503,
                    message=str(e.__cause__ or e),
                    provider="openai",
                    model=str(params.get("model") or self.model_name),
                    operation="synthesize_speech",
                )
                return AudioSpeechResult(audio=b"", format=str(params.get("response_format") or "mp3"), model=str(params.get("model") or self.model_name), status=503, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.RateLimitError as e:
                failure = self._failure(error=e, operation="synthesize_speech")
                return AudioSpeechResult(audio=b"", format=str(params.get("response_format") or "mp3"), model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.APIStatusError as e:
                failure = self._failure(error=e, operation="synthesize_speech")
                return AudioSpeechResult(audio=b"", format=str(params.get("response_format") or "mp3"), model=str(params.get("model") or self.model_name), status=failure.status, error=failure.message, raw_response={"normalized_failure": failure.to_dict()})

    async def create_file(
        self,
        *,
        file: Any,
        purpose: str,
        **kwargs: Any,
    ) -> FileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.files.create(file=file, purpose=purpose, **kwargs)
        return self._file_resource_from_response(response)

    async def retrieve_file(self, file_id: str, **kwargs: Any) -> FileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.files.retrieve(file_id, **kwargs)
        return self._file_resource_from_response(response)

    async def list_files(self, **kwargs: Any) -> FilesPage:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.files.list(**kwargs)
        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._files_page_from_response(page)
        items = [item async for item in response]
        synthetic = {
            "data": [self._serialize_responses_item(item) for item in items],
            "first_id": str(getattr(items[0], "id", "") or "") if items else None,
            "last_id": str(getattr(items[-1], "id", "") or "") if items else None,
            "has_more": False,
        }
        return self._files_page_from_response(synthetic)

    async def delete_file(self, file_id: str, **kwargs: Any) -> DeletionResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.files.delete(file_id, **kwargs)
        deleted = bool(getattr(response, "deleted", True))
        return DeletionResult(resource_id=file_id, deleted=deleted, raw_response=response)

    async def get_file_content(self, file_id: str, **kwargs: Any) -> FileContentResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.files.content(file_id, **kwargs)
        media_type = None
        content_bytes = b""
        if hasattr(response, "content"):
            content_bytes = bytes(getattr(response, "content", b"") or b"")
            headers = getattr(response, "headers", None)
            if headers is not None:
                media_type = headers.get("content-type")
        if not content_bytes and hasattr(response, "read"):
            read_result = response.read()
            content_bytes = await read_result if inspect.isawaitable(read_result) else bytes(read_result)
        if not content_bytes and hasattr(response, "aread"):
            content_bytes = await response.aread()
        return FileContentResult(
            file_id=file_id,
            content=content_bytes,
            media_type=str(media_type or "") or None,
            raw_response=response,
        )

    async def create_vector_store(self, **kwargs: Any) -> VectorStoreResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.create(**kwargs)
        return self._vector_store_resource_from_response(response)

    async def retrieve_vector_store(self, vector_store_id: str, **kwargs: Any) -> VectorStoreResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.retrieve(vector_store_id, **kwargs)
        return self._vector_store_resource_from_response(response)

    async def update_vector_store(self, vector_store_id: str, **kwargs: Any) -> VectorStoreResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.update(vector_store_id, **kwargs)
        return self._vector_store_resource_from_response(response)

    async def delete_vector_store(self, vector_store_id: str, **kwargs: Any) -> DeletionResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.delete(vector_store_id, **kwargs)
        deleted = bool(getattr(response, "deleted", True))
        return DeletionResult(resource_id=vector_store_id, deleted=deleted, raw_response=response)

    async def list_vector_stores(self, **kwargs: Any) -> VectorStoresPage:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.list(**kwargs)
        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._vector_stores_page_from_response(page)
        items = [item async for item in response]
        synthetic = {
            "data": [self._serialize_responses_item(item) for item in items],
            "first_id": str(getattr(items[0], "id", "") or "") if items else None,
            "last_id": str(getattr(items[-1], "id", "") or "") if items else None,
            "has_more": False,
        }
        return self._vector_stores_page_from_response(synthetic)

    async def search_vector_store(
        self,
        vector_store_id: str,
        *,
        query: str | list[str],
        max_num_results: int | None = None,
        attribute_filter: ResponsesAttributeFilter | dict[str, Any] | None = None,
        ranking_options: ResponsesFileSearchRankingOptions | dict[str, Any] | None = None,
        rewrite_query: bool | None = None,
        **kwargs: Any,
    ) -> VectorStoreSearchResult:
        params = dict(kwargs)
        self._apply_openai_param_alias(
            params,
            canonical_key="filters",
            alias_value=attribute_filter,
            alias_name="attribute_filter",
        )
        self._apply_openai_param_alias(
            params,
            canonical_key="ranking_options",
            alias_value=ranking_options,
            alias_name="ranking_options",
        )
        self._apply_openai_param_alias(
            params,
            canonical_key="max_num_results",
            alias_value=max_num_results,
            alias_name="max_num_results",
        )
        self._apply_openai_param_alias(
            params,
            canonical_key="rewrite_query",
            alias_value=rewrite_query,
            alias_name="rewrite_query",
        )
        async with self.limiter.limit(tokens=self.count_tokens(query), requests=1):
            response = await self.client.vector_stores.search(vector_store_id, query=query, **params)
        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._vector_store_search_result_from_response(page, vector_store_id=vector_store_id, query=query)
        items = [item async for item in response]
        synthetic = {"data": [self._serialize_responses_item(item) for item in items]}
        return self._vector_store_search_result_from_response(synthetic, vector_store_id=vector_store_id, query=query)

    async def create_fine_tuning_job(self, **kwargs: Any) -> FineTuningJobResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.fine_tuning.jobs.create(**kwargs)
        return self._fine_tuning_job_result_from_response(response)

    async def retrieve_fine_tuning_job(self, job_id: str, **kwargs: Any) -> FineTuningJobResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.fine_tuning.jobs.retrieve(job_id, **kwargs)
        return self._fine_tuning_job_result_from_response(response)

    async def cancel_fine_tuning_job(self, job_id: str, **kwargs: Any) -> FineTuningJobResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.fine_tuning.jobs.cancel(job_id, **kwargs)
        return self._fine_tuning_job_result_from_response(response)

    async def list_fine_tuning_jobs(self, **kwargs: Any) -> FineTuningJobsPage:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.fine_tuning.jobs.list(**kwargs)
        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._fine_tuning_jobs_page_from_response(page)
        items = [item async for item in response]
        synthetic = {
            "data": [self._serialize_responses_item(item) for item in items],
            "first_id": str(getattr(items[0], "id", "") or "") if items else None,
            "last_id": str(getattr(items[-1], "id", "") or "") if items else None,
            "has_more": False,
        }
        return self._fine_tuning_jobs_page_from_response(synthetic)

    async def list_fine_tuning_events(self, job_id: str, **kwargs: Any) -> FineTuningJobEventsPage:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.fine_tuning.jobs.list_events(job_id, **kwargs)
        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._fine_tuning_job_events_page_from_response(page, job_id=job_id)
        items = [item async for item in response]
        synthetic = {"data": [self._serialize_responses_item(item) for item in items], "has_more": False}
        return self._fine_tuning_job_events_page_from_response(synthetic, job_id=job_id)

    async def create_realtime_client_secret(self, **kwargs: Any) -> RealtimeClientSecretResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.realtime.client_secrets.create(**kwargs)
        return self._realtime_client_secret_result_from_response(response)

    async def create_realtime_transcription_session(self, **kwargs: Any) -> RealtimeTranscriptionSessionResult:
        params = dict(kwargs)
        session = params.get("session")
        if isinstance(session, dict) and str(session.get("type") or "") == "transcription":
            session_payload = dict(session)
            transcription_model = str(session_payload.pop("model", "") or self.model_name or "")
            audio_payload = dict(session_payload.get("audio") or {})
            input_payload = dict(audio_payload.get("input") or {})
            transcription_payload = dict(input_payload.get("transcription") or {})
            if transcription_model and not transcription_payload.get("model"):
                transcription_payload["model"] = transcription_model
            if transcription_payload:
                input_payload["transcription"] = transcription_payload
            if input_payload:
                audio_payload["input"] = input_payload
            if audio_payload:
                session_payload["audio"] = audio_payload
            params["session"] = session_payload
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.realtime.client_secrets.create(**params)
        return self._realtime_transcription_session_result_from_response(response)

    async def connect_realtime(self, **kwargs: Any) -> RealtimeConnection:
        connect_kwargs = dict(kwargs)
        connect_kwargs["model"] = self._normalize_realtime_connection_model(
            str(connect_kwargs.get("model") or self.model_name or "")
        )
        async with self.limiter.limit(tokens=0, requests=1):
            manager = self.client.realtime.connect(**connect_kwargs)
            connection = await manager.__aenter__()

        async def _close() -> None:
            await manager.__aexit__(None, None, None)

        return RealtimeConnection(
            connection,
            model=str(connect_kwargs.get("model") or self.model_name),
            call_id=str(connect_kwargs.get("call_id") or "") or None,
            close_callback=_close,
            raw_manager=manager,
        )

    async def connect_realtime_transcription(self, **kwargs: Any) -> RealtimeConnection:
        connect_kwargs = dict(kwargs)
        connect_kwargs["model"] = self._normalize_realtime_connection_model(
            str(connect_kwargs.get("model") or self.model_name or "")
        )
        async with self.limiter.limit(tokens=0, requests=1):
            manager = self.client.realtime.connect(**connect_kwargs)
            connection = await manager.__aenter__()

        async def _close() -> None:
            await manager.__aexit__(None, None, None)

        return RealtimeConnection(
            connection,
            model=str(connect_kwargs.get("model") or self.model_name),
            call_id=str(connect_kwargs.get("call_id") or "") or None,
            close_callback=_close,
            raw_manager=manager,
        )

    async def create_realtime_call(self, sdp: str, **kwargs: Any) -> RealtimeCallResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.realtime.calls.create(sdp=sdp, **kwargs)
        return self._realtime_call_result_from_response(response, action="create")

    async def accept_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        async with self.limiter.limit(tokens=0, requests=1):
            await self.client.realtime.calls.accept(call_id, **kwargs)
        return RealtimeCallResult(call_id=call_id, action="accept", status=200)

    async def reject_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        async with self.limiter.limit(tokens=0, requests=1):
            await self.client.realtime.calls.reject(call_id, **kwargs)
        return RealtimeCallResult(call_id=call_id, action="reject", status=200)

    async def hangup_realtime_call(self, call_id: str, **kwargs: Any) -> RealtimeCallResult:
        async with self.limiter.limit(tokens=0, requests=1):
            await self.client.realtime.calls.hangup(call_id, **kwargs)
        return RealtimeCallResult(call_id=call_id, action="hangup", status=200)

    async def refer_realtime_call(self, call_id: str, *, target_uri: str, **kwargs: Any) -> RealtimeCallResult:
        async with self.limiter.limit(tokens=0, requests=1):
            await self.client.realtime.calls.refer(call_id, target_uri=target_uri, **kwargs)
        return RealtimeCallResult(call_id=call_id, action="refer", status=200)

    async def unwrap_webhook(
        self,
        payload: str | bytes,
        headers: Any,
        *,
        secret: str | None = None,
    ) -> WebhookEventResult:
        event = self.client.webhooks.unwrap(payload, headers, secret=secret)
        return self._webhook_event_result_from_unwrapped(event)

    async def verify_webhook_signature(
        self,
        payload: str | bytes,
        headers: Any,
        *,
        secret: str | None = None,
        tolerance: int = 300,
    ) -> bool:
        self.client.webhooks.verify_signature(payload, headers, secret=secret, tolerance=tolerance)
        return True

    async def create_vector_store_file(
        self,
        vector_store_id: str,
        *,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.create(vector_store_id, file_id=file_id, **kwargs)
        return self._vector_store_file_resource_from_response(response, vector_store_id=vector_store_id)

    async def upload_vector_store_file(
        self,
        vector_store_id: str,
        *,
        file: Any,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.upload(vector_store_id=vector_store_id, file=file, **kwargs)
        return self._vector_store_file_resource_from_response(response, vector_store_id=vector_store_id)

    async def list_vector_store_files(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFilesPage:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.list(vector_store_id, **kwargs)
        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._vector_store_files_page_from_response(page, vector_store_id=vector_store_id)
        items = [item async for item in response]
        synthetic = {
            "data": [self._serialize_responses_item(item) for item in items],
            "first_id": str(getattr(items[0], "id", "") or "") if items else None,
            "last_id": str(getattr(items[-1], "id", "") or "") if items else None,
            "has_more": False,
        }
        return self._vector_store_files_page_from_response(synthetic, vector_store_id=vector_store_id)

    async def retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.retrieve(file_id, vector_store_id=vector_store_id, **kwargs)
        return self._vector_store_file_resource_from_response(response, vector_store_id=vector_store_id)

    async def update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.update(file_id, vector_store_id=vector_store_id, **kwargs)
        return self._vector_store_file_resource_from_response(response, vector_store_id=vector_store_id)

    async def delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> DeletionResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.delete(file_id, vector_store_id=vector_store_id, **kwargs)
        deleted = bool(getattr(response, "deleted", True))
        return DeletionResult(resource_id=file_id, deleted=deleted, raw_response=response)

    async def get_vector_store_file_content(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileContentResult:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.content(file_id, vector_store_id=vector_store_id, **kwargs)
        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._vector_store_file_content_result_from_response(page, vector_store_id=vector_store_id, file_id=file_id)
        items = [item async for item in response]
        synthetic = {"data": [self._serialize_responses_item(item) for item in items]}
        return self._vector_store_file_content_result_from_response(synthetic, vector_store_id=vector_store_id, file_id=file_id)

    async def poll_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.poll(file_id, vector_store_id=vector_store_id, **kwargs)
        return self._vector_store_file_resource_from_response(response, vector_store_id=vector_store_id)

    async def create_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        *,
        file_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.create_and_poll(file_id, vector_store_id=vector_store_id, **kwargs)
        return self._vector_store_file_resource_from_response(response, vector_store_id=vector_store_id)

    async def upload_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        *,
        file: Any,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.files.upload_and_poll(vector_store_id=vector_store_id, file=file, **kwargs)
        return self._vector_store_file_resource_from_response(response, vector_store_id=vector_store_id)

    async def create_vector_store_file_batch(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.file_batches.create(vector_store_id, **kwargs)
        return self._vector_store_file_batch_resource_from_response(response, vector_store_id=vector_store_id)

    async def retrieve_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.file_batches.retrieve(batch_id, vector_store_id=vector_store_id, **kwargs)
        return self._vector_store_file_batch_resource_from_response(response, vector_store_id=vector_store_id)

    async def cancel_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.file_batches.cancel(batch_id, vector_store_id=vector_store_id, **kwargs)
        return self._vector_store_file_batch_resource_from_response(response, vector_store_id=vector_store_id)

    async def poll_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.file_batches.poll(batch_id, vector_store_id=vector_store_id, **kwargs)
        return self._vector_store_file_batch_resource_from_response(response, vector_store_id=vector_store_id)

    async def list_vector_store_file_batch_files(
        self,
        vector_store_id: str,
        batch_id: str,
        **kwargs: Any,
    ) -> VectorStoreFilesPage:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.file_batches.list_files(batch_id, vector_store_id=vector_store_id, **kwargs)
        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._vector_store_files_page_from_response(page, vector_store_id=vector_store_id)
        items = [item async for item in response]
        synthetic = {
            "data": [self._serialize_responses_item(item) for item in items],
            "first_id": str(getattr(items[0], "id", "") or "") if items else None,
            "last_id": str(getattr(items[-1], "id", "") or "") if items else None,
            "has_more": False,
        }
        return self._vector_store_files_page_from_response(synthetic, vector_store_id=vector_store_id)

    async def create_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.file_batches.create_and_poll(vector_store_id, **kwargs)
        return self._vector_store_file_batch_resource_from_response(response, vector_store_id=vector_store_id)

    async def upload_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        *,
        files: list[Any] | tuple[Any, ...],
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.vector_stores.file_batches.upload_and_poll(vector_store_id, files=files, **kwargs)
        return self._vector_store_file_batch_resource_from_response(response, vector_store_id=vector_store_id)

    async def clarify_deep_research_task(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        model_name = str(kwargs.pop("model", "gpt-4.1"))
        instructions = str(kwargs.pop("instructions", _DEEP_RESEARCH_CLARIFY_INSTRUCTIONS))
        return await self.complete(prompt, model=model_name, instructions=instructions, **kwargs)

    async def rewrite_deep_research_prompt(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        model_name = str(kwargs.pop("model", "gpt-4.1"))
        clarifications = kwargs.pop("clarifications", None)
        instructions = str(kwargs.pop("instructions", _DEEP_RESEARCH_REWRITE_INSTRUCTIONS))
        rewrite_input = self._format_deep_research_rewrite_input(prompt, clarifications=clarifications)
        return await self.complete(rewrite_input, model=model_name, instructions=instructions, **kwargs)

    async def respond_with_web_search(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        model_name = str(kwargs.pop("model", self.model_name))
        preview = bool(kwargs.pop("preview", True))
        tool = kwargs.pop("tool", None)
        tool_config = dict(kwargs.pop("tool_config", {}) or {})
        if tool is None:
            tool = (
                ResponsesBuiltinTool.web_search_preview(**tool_config)
                if preview
                else ResponsesBuiltinTool.web_search(**tool_config)
            )
        return await self._complete_with_responses_tools(prompt, tools=[tool], model=model_name, **kwargs)

    async def respond_with_file_search(
        self,
        prompt: str,
        *,
        vector_store_ids: list[str] | tuple[str, ...],
        max_num_results: int | None = None,
        attribute_filter: ResponsesAttributeFilter | dict[str, Any] | None = None,
        ranking_options: ResponsesFileSearchRankingOptions | dict[str, Any] | None = None,
        include_search_results: bool = False,
        **kwargs: Any,
    ) -> CompletionResult:
        model_name = str(kwargs.pop("model", self.model_name))
        tool = kwargs.pop("tool", None)
        tool_config = dict(kwargs.pop("tool_config", {}) or {})
        include = kwargs.pop("include", None)
        if tool is not None and any(value is not None for value in (max_num_results, attribute_filter, ranking_options)):
            raise ValueError(
                "Provide file-search tuning controls on the explicit `tool` object, or omit `tool` and use helper kwargs."
            )
        self._apply_openai_param_alias(
            tool_config,
            canonical_key="filters",
            alias_value=attribute_filter,
            alias_name="attribute_filter",
        )
        self._apply_openai_param_alias(
            tool_config,
            canonical_key="ranking_options",
            alias_value=ranking_options,
            alias_name="ranking_options",
        )
        self._apply_openai_param_alias(
            tool_config,
            canonical_key="max_num_results",
            alias_value=max_num_results,
            alias_name="max_num_results",
        )
        if tool is None:
            tool = ResponsesBuiltinTool.file_search(
                vector_store_ids=list(vector_store_ids),
                **tool_config,
            )
        if include_search_results:
            include = self._merge_openai_include(include, _OPENAI_FILE_SEARCH_RESULTS_INCLUDE)
        if include is not None:
            kwargs["include"] = include
        return await self._complete_with_responses_tools(prompt, tools=[tool], model=model_name, **kwargs)

    async def respond_with_tool_search(
        self,
        prompt: str,
        *,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> CompletionResult:
        model_name = str(kwargs.pop("model", self.model_name))
        tool_search = kwargs.pop("tool_search", None)
        tool_search_config = dict(kwargs.pop("tool_search_config", {}) or {})
        if tool_search is None:
            tool_search = ResponsesBuiltinTool.of("tool_search", **tool_search_config)
        return await self._complete_with_responses_tools(
            prompt,
            tools=[tool_search, *list(tools)],
            model=model_name,
            **kwargs,
        )

    async def respond_with_code_interpreter(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        model_name = str(kwargs.pop("model", self.model_name))
        tool = kwargs.pop("tool", None)
        tool_config = dict(kwargs.pop("tool_config", {}) or {})
        if tool is None:
            tool = ResponsesBuiltinTool.code_interpreter(
                container=tool_config.pop("container", {"type": "auto"}),
                **tool_config,
            )
        return await self._complete_with_responses_tools(prompt, tools=[tool], model=model_name, **kwargs)

    async def respond_with_remote_mcp(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        model_name = str(kwargs.pop("model", self.model_name))
        tool = kwargs.pop("tool", None)
        if tool is None:
            server_url = kwargs.pop("server_url")
            tool = ResponsesMCPTool.remote_server(
                server_url,
                server_label=kwargs.pop("server_label", None),
                server_description=kwargs.pop("server_description", None),
                authorization=kwargs.pop("authorization", None),
                headers=kwargs.pop("headers", None),
                allowed_tools=kwargs.pop("allowed_tools", None),
                require_approval=kwargs.pop("require_approval", None),
                **dict(kwargs.pop("tool_metadata", {}) or {}),
            )
        return await self._complete_with_responses_tools(prompt, tools=[tool], model=model_name, **kwargs)

    async def respond_with_connector(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        model_name = str(kwargs.pop("model", self.model_name))
        tool = kwargs.pop("tool", None)
        if tool is None:
            connector_id = str(kwargs.pop("connector_id"))
            tool = ResponsesMCPTool.connector(
                connector_id,
                server_label=kwargs.pop("server_label", None),
                authorization=kwargs.pop("authorization", None),
                allowed_tools=kwargs.pop("allowed_tools", None),
                require_approval=kwargs.pop("require_approval", None),
                **dict(kwargs.pop("tool_metadata", {}) or {}),
            )
        return await self._complete_with_responses_tools(prompt, tools=[tool], model=model_name, **kwargs)

    async def start_deep_research(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> CompletionResult:
        self._require_responses_api("Deep research orchestration")

        model_name = str(kwargs.pop("model", self.model_name))
        vector_store_ids = list(kwargs.pop("vector_store_ids", []) or [])
        web_search = bool(kwargs.pop("web_search", False))
        mcp_tools = list(kwargs.pop("mcp_tools", []) or [])
        include_code_interpreter = bool(kwargs.pop("include_code_interpreter", False))
        extra_tools = list(kwargs.pop("tools", []) or [])
        rewrite_prompt = bool(kwargs.pop("rewrite_prompt", False))
        clarifications = kwargs.pop("clarifications", None)
        rewrite_model = str(kwargs.pop("rewrite_model", "gpt-4.1"))
        rewrite_instructions = kwargs.pop("rewrite_instructions", _DEEP_RESEARCH_REWRITE_INSTRUCTIONS)
        if rewrite_prompt or clarifications is not None:
            rewritten = await self.rewrite_deep_research_prompt(
                prompt,
                model=rewrite_model,
                clarifications=clarifications,
                instructions=rewrite_instructions,
            )
            if isinstance(rewritten.content, str) and rewritten.content.strip():
                prompt = rewritten.content

        tools: list[Any] = list(extra_tools)
        if web_search:
            tools.append(ResponsesBuiltinTool.web_search_preview())
        if vector_store_ids:
            tools.append(ResponsesBuiltinTool.file_search(vector_store_ids=vector_store_ids))
        tools.extend(self._normalize_deep_research_mcp_tool(tool) for tool in mcp_tools)
        if include_code_interpreter:
            tools.append(ResponsesBuiltinTool.code_interpreter(container={"type": "auto"}))

        if not tools:
            raise ValueError(
                "Deep research requires at least one data source tool: web search, file search/vector stores, or MCP."
            )

        kwargs.setdefault("background", True)
        return await self.complete(
            prompt,
            tools=tools,
            model=model_name,
            **kwargs,
        )

    async def run_deep_research(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> DeepResearchRunResult:
        self._require_responses_api("Deep research orchestration")

        prompt_text = str(prompt)
        effective_prompt = prompt_text
        clarification: CompletionResult | None = None
        rewrite: CompletionResult | None = None
        background: BackgroundResponseResult | None = None

        clarify_first = bool(kwargs.pop("clarify_first", False))
        clarifications = kwargs.pop("clarifications", None)
        clarify_model = str(kwargs.pop("clarify_model", "gpt-4.1"))
        clarify_instructions = kwargs.pop("clarify_instructions", _DEEP_RESEARCH_CLARIFY_INSTRUCTIONS)
        rewrite_prompt = bool(kwargs.pop("rewrite_prompt", False) or clarifications is not None)
        rewrite_model = str(kwargs.pop("rewrite_model", "gpt-4.1"))
        rewrite_instructions = kwargs.pop("rewrite_instructions", _DEEP_RESEARCH_REWRITE_INSTRUCTIONS)
        wait_for_completion = bool(kwargs.pop("wait_for_completion", False))
        poll_interval = float(kwargs.pop("poll_interval", 2.0))
        wait_timeout = kwargs.pop("wait_timeout", None)
        wait_kwargs = dict(kwargs.pop("wait_kwargs", {}) or {})

        if clarify_first:
            clarification = await self.clarify_deep_research_task(
                prompt_text,
                model=clarify_model,
                instructions=clarify_instructions,
            )

        if rewrite_prompt:
            rewrite = await self.rewrite_deep_research_prompt(
                prompt_text,
                model=rewrite_model,
                clarifications=clarifications,
                instructions=rewrite_instructions,
            )
            if isinstance(rewrite.content, str) and rewrite.content.strip():
                effective_prompt = rewrite.content

        queued = await self.start_deep_research(
            effective_prompt,
            rewrite_prompt=False,
            clarifications=None,
            **kwargs,
        )
        response_id = str(getattr(queued.raw_response, "id", "") or "") or None

        if wait_for_completion and response_id:
            background = await self.wait_background_response(
                response_id,
                poll_interval=poll_interval,
                timeout=wait_timeout,
                **wait_kwargs,
            )

        return DeepResearchRunResult(
            prompt=prompt_text,
            effective_prompt=effective_prompt,
            clarification=clarification,
            rewrite=rewrite,
            queued=queued,
            response_id=response_id,
            background=background,
        )

    async def retrieve_background_response(self, response_id: str, **kwargs: Any) -> BackgroundResponseResult:
        """Retrieve a background Responses API object by id."""
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Background response lifecycle requires use_responses_api=True")

        async with self.limiter.limit(tokens=0, requests=1):
            try:
                response = await self.client.responses.retrieve(response_id, **kwargs)
                return self._background_response_result_from_response(response)
            except openai.APIConnectionError as e:
                failure = normalize_provider_failure(
                    status=503,
                    message=str(e.__cause__ or e),
                    provider="openai",
                    model=self.model_name,
                    operation="retrieve_background_response",
                )
                return BackgroundResponseResult(response_id=response_id, lifecycle_status="failed", error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.RateLimitError as e:
                failure = self._failure(error=e, operation="retrieve_background_response")
                return BackgroundResponseResult(response_id=response_id, lifecycle_status="failed", error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.APIStatusError as e:
                failure = self._failure(error=e, operation="retrieve_background_response")
                return BackgroundResponseResult(response_id=response_id, lifecycle_status="failed", error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except Exception as e:
                failure = self._failure(error=e, operation="retrieve_background_response")
                return BackgroundResponseResult(response_id=response_id, lifecycle_status="failed", error=failure.message, raw_response={"normalized_failure": failure.to_dict()})

    async def cancel_background_response(self, response_id: str, **kwargs: Any) -> BackgroundResponseResult:
        """Cancel a background Responses API object by id."""
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Background response lifecycle requires use_responses_api=True")

        async with self.limiter.limit(tokens=0, requests=1):
            try:
                response = await self.client.responses.cancel(response_id, **kwargs)
                return self._background_response_result_from_response(response)
            except openai.APIConnectionError as e:
                failure = normalize_provider_failure(
                    status=503,
                    message=str(e.__cause__ or e),
                    provider="openai",
                    model=self.model_name,
                    operation="cancel_background_response",
                )
                return BackgroundResponseResult(response_id=response_id, lifecycle_status="failed", error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.RateLimitError as e:
                failure = self._failure(error=e, operation="cancel_background_response")
                return BackgroundResponseResult(response_id=response_id, lifecycle_status="failed", error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except openai.APIStatusError as e:
                failure = self._failure(error=e, operation="cancel_background_response")
                return BackgroundResponseResult(response_id=response_id, lifecycle_status="failed", error=failure.message, raw_response={"normalized_failure": failure.to_dict()})
            except Exception as e:
                failure = self._failure(error=e, operation="cancel_background_response")
                return BackgroundResponseResult(response_id=response_id, lifecycle_status="failed", error=failure.message, raw_response={"normalized_failure": failure.to_dict()})

    async def wait_background_response(
        self,
        response_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> BackgroundResponseResult:
        return await super().wait_background_response(
            response_id,
            poll_interval=poll_interval,
            timeout=timeout,
            **kwargs,
        )

    async def stream_background_response(
        self,
        response_id: str,
        *,
        starting_after: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Resume or attach to a background Responses stream."""
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Background response lifecycle requires use_responses_api=True")

        yield StreamEvent(
            type=StreamEventType.META,
            data={"model": self.model_name, "stream": True, "background_response_id": response_id, "starting_after": starting_after},
        )

        async with self.limiter.limit(tokens=0, requests=1) as limit_ctx:
            try:
                stream = await self.client.responses.retrieve(
                    response_id,
                    stream=True,
                    starting_after=starting_after,
                    **kwargs,
                )
                async for emitted_event in self._iter_responses_stream_events(
                    stream,
                    model_name_hint=self.model_name,
                    parsed_text_format=None,
                    limit_ctx=limit_ctx,
                ):
                    yield emitted_event
            except openai.APIConnectionError as e:
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data=failure_to_stream_error_data(
                        normalize_provider_failure(
                            status=503,
                            message=str(e.__cause__ or e),
                            provider="openai",
                            model=self.model_name,
                            operation="stream_background_response",
                        )
                    ),
                )
            except openai.RateLimitError as e:
                yield StreamEvent(type=StreamEventType.ERROR, data=failure_to_stream_error_data(self._failure(error=e, operation="stream_background_response")))
            except openai.APIStatusError as e:
                yield StreamEvent(type=StreamEventType.ERROR, data=failure_to_stream_error_data(self._failure(error=e, operation="stream_background_response")))
            except Exception as e:
                yield StreamEvent(type=StreamEventType.ERROR, data=failure_to_stream_error_data(self._failure(error=e, operation="stream_background_response")))

    async def create_conversation(
        self,
        *,
        items: MessageInput | list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Conversation lifecycle requires use_responses_api=True")

        params: dict[str, Any] = {}
        if items is not None:
            params["items"] = self._coerce_responses_input_items(items)
        if metadata is not None:
            params["metadata"] = dict(metadata)
        params.update(kwargs)

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.conversations.create(**params)
        return self._conversation_resource_from_response(response)

    async def retrieve_conversation(self, conversation_id: str, **kwargs: Any) -> ConversationResource:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Conversation lifecycle requires use_responses_api=True")

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.conversations.retrieve(conversation_id, **kwargs)
        return self._conversation_resource_from_response(response)

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Conversation lifecycle requires use_responses_api=True")

        params: dict[str, Any] = {}
        if metadata is not None:
            params["metadata"] = dict(metadata)
        params.update(kwargs)

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.conversations.update(conversation_id, **params)
        return self._conversation_resource_from_response(response)

    async def delete_conversation(self, conversation_id: str, **kwargs: Any) -> ConversationResource:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Conversation lifecycle requires use_responses_api=True")

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.conversations.delete(conversation_id, **kwargs)
        return self._conversation_resource_from_response(response)

    async def compact_response_context(
        self,
        *,
        messages: MessageInput | list[dict[str, Any]] | None = None,
        model: str | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        **kwargs: Any,
    ) -> CompactionResult:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Response compaction requires use_responses_api=True")

        params: dict[str, Any] = {
            "model": str(model or self.model_name),
        }
        if messages is not None:
            params["input"] = self._coerce_responses_input_items(messages)
        if instructions is not None:
            params["instructions"] = instructions
        if previous_response_id is not None:
            params["previous_response_id"] = previous_response_id
        params.update(kwargs)

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.responses.compact(**params)
        return self._compaction_result_from_response(response)

    async def submit_mcp_approval_response(
        self,
        *,
        previous_response_id: str,
        approval_request_id: str,
        approve: bool,
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("MCP approval flows require use_responses_api=True")

        params: dict[str, Any] = {
            "model": self.model_name,
            "previous_response_id": previous_response_id,
            "input": [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval_request_id,
                    "approve": bool(approve),
                }
            ],
        }
        if tools:
            provider_tools, _, _ = self._prepare_openai_tools(tools, responses_api=True)
            if provider_tools:
                params["tools"] = provider_tools
        params.update(kwargs)

        return await self._complete_responses(
            [],
            params,
            alias_to_original={},
        )

    async def submit_tool_search_output(
        self,
        *,
        previous_response_id: str,
        call_id: str,
        tools: list[ToolDefinition],
        **kwargs: Any,
    ) -> CompletionResult:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Tool-search continuation requires use_responses_api=True")

        provider_tools, alias_to_original, _ = self._prepare_openai_tools(tools, responses_api=True)
        params: dict[str, Any] = {
            "model": self.model_name,
            "previous_response_id": previous_response_id,
            "input": [
                {
                    "type": "tool_search_output",
                    "call_id": call_id,
                    "tools": provider_tools or [],
                }
            ],
        }
        params.update(kwargs)

        return await self._complete_responses(
            [],
            params,
            alias_to_original=alias_to_original,
        )

    async def delete_response(self, response_id: str, **kwargs: Any) -> DeletionResult:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Stored response deletion requires use_responses_api=True")

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.responses.delete(response_id, **kwargs)
        return DeletionResult(resource_id=response_id, deleted=True, raw_response=response)

    async def create_conversation_items(
        self,
        conversation_id: str,
        *,
        items: MessageInput | list[dict[str, Any]],
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> ConversationItemsPage:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Conversation items require use_responses_api=True")

        params: dict[str, Any] = {
            "items": self._coerce_responses_input_items(items),
        }
        if include is not None:
            params["include"] = list(include)
        params.update(kwargs)

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.conversations.items.create(conversation_id, **params)
        return self._conversation_items_page_from_response(response)

    async def list_conversation_items(
        self,
        conversation_id: str,
        *,
        after: str | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        order: str | None = None,
        **kwargs: Any,
    ) -> ConversationItemsPage:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Conversation items require use_responses_api=True")

        params: dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if include is not None:
            params["include"] = list(include)
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        params.update(kwargs)

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.conversations.items.list(conversation_id, **params)

        if hasattr(response, "_get_page"):
            page = await response._get_page()
            return self._conversation_items_page_from_response(page)

        items: list[Any] = [item async for item in response]
        synthetic = {
            "data": [self._serialize_responses_item(item) for item in items],
            "first_id": str(getattr(items[0], "id", "") or "") if items else None,
            "last_id": str(getattr(items[-1], "id", "") or "") if items else None,
            "has_more": False,
        }
        return self._conversation_items_page_from_response(synthetic)

    async def retrieve_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        *,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> ConversationItemResource:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Conversation items require use_responses_api=True")

        params: dict[str, Any] = {"conversation_id": conversation_id}
        if include is not None:
            params["include"] = list(include)
        params.update(kwargs)

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.conversations.items.retrieve(item_id, **params)
        return self._conversation_item_resource_from_item(response)

    async def delete_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        **kwargs: Any,
    ) -> ConversationResource:
        if not bool(getattr(self, "use_responses_api", False)):
            raise NotImplementedError("Conversation items require use_responses_api=True")

        async with self.limiter.limit(tokens=0, requests=1):
            response = await self.client.conversations.items.delete(item_id, conversation_id=conversation_id, **kwargs)
        return self._conversation_resource_from_response(response)

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
