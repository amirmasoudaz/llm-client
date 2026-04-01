"""
Shared request construction helpers for engine-backed execution.

These helpers centralize provider/model inference and `RequestSpec` building so
engine-first code paths do not have to duplicate request-shaping logic.
"""

from __future__ import annotations

from typing import Any

from .content import ContentRequestEnvelope, ContentMessage
from .content import FileBlock, prepare_content_blocks_for_transport, message_from_content_blocks, message_to_content_blocks
from .providers.types import Message, normalize_messages
from .spec import RequestSpec

_REQUEST_SPEC_RESERVED_KEYS = {
    "model",
    "tools",
    "tool_choice",
    "temperature",
    "max_tokens",
    "response_format",
    "reasoning_effort",
    "reasoning",
    "include",
    "prompt_cache_key",
    "prompt_cache_retention",
}


def infer_provider_name(provider: Any) -> str:
    if provider is None:
        return "unknown"
    class_name = type(provider).__name__.strip().lstrip("_")
    if not class_name:
        return "unknown"
    if class_name.endswith("Provider"):
        class_name = class_name[:-8]
    return class_name.lower() or "unknown"


def infer_model_name(provider: Any) -> str | None:
    if provider is None:
        return None
    model_name = getattr(provider, "model_name", None)
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    model = getattr(provider, "model", None)
    key = getattr(model, "key", None)
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def build_request_spec(
    *,
    messages: list[Message] | list[dict[str, Any]],
    provider: Any = None,
    engine: Any = None,
    request_kwargs: dict[str, Any] | None = None,
    model: str | None = None,
    stream: bool = False,
) -> RequestSpec:
    request_kwargs = dict(request_kwargs or {})
    runtime_provider = provider if provider is not None else getattr(engine, "provider", None)
    provider_name = infer_provider_name(runtime_provider)
    model_name = (
        str(model or request_kwargs.get("model") or "").strip()
        or infer_model_name(runtime_provider)
        or "unknown"
    )
    extra = {key: value for key, value in request_kwargs.items() if key not in _REQUEST_SPEC_RESERVED_KEYS}
    normalized_messages = normalize_messages(messages)
    prepared_messages = []
    for message in normalized_messages:
        blocks = message_to_content_blocks(message)
        if not any(isinstance(block, FileBlock) for block in blocks):
            prepared_messages.append(message)
            continue
        prepared_messages.append(
            message_from_content_blocks(
                role=message.role,
                blocks=prepare_content_blocks_for_transport(blocks),
                name=message.name,
            )
        )
    return RequestSpec(
        provider=provider_name,
        model=model_name,
        messages=prepared_messages,
        tools=request_kwargs.get("tools"),
        tool_choice=request_kwargs.get("tool_choice"),
        temperature=request_kwargs.get("temperature"),
        max_tokens=request_kwargs.get("max_tokens"),
        response_format=request_kwargs.get("response_format"),
        reasoning_effort=request_kwargs.get("reasoning_effort"),
        reasoning=request_kwargs.get("reasoning"),
        include=list(request_kwargs.get("include")) if request_kwargs.get("include") is not None else None,
        prompt_cache_key=request_kwargs.get("prompt_cache_key"),
        prompt_cache_retention=request_kwargs.get("prompt_cache_retention"),
        extra=extra,
        stream=stream,
    )


def build_content_request_envelope(
    *,
    messages: list[Message] | list[dict[str, Any]] | list[ContentMessage],
    provider: Any = None,
    engine: Any = None,
    request_kwargs: dict[str, Any] | None = None,
    model: str | None = None,
    stream: bool = False,
) -> ContentRequestEnvelope:
    request_kwargs = dict(request_kwargs or {})
    if messages and isinstance(messages[0], ContentMessage):
        runtime_provider = provider if provider is not None else getattr(engine, "provider", None)
        provider_name = infer_provider_name(runtime_provider)
        model_name = (
            str(model or request_kwargs.get("model") or "").strip()
            or infer_model_name(runtime_provider)
            or "unknown"
        )
        extra = {key: value for key, value in request_kwargs.items() if key not in _REQUEST_SPEC_RESERVED_KEYS}
        return ContentRequestEnvelope(
            provider=provider_name,
            model=model_name,
            messages=tuple(messages),
            tools=tuple(request_kwargs.get("tools")) if request_kwargs.get("tools") is not None else None,
            tool_choice=request_kwargs.get("tool_choice"),
            temperature=request_kwargs.get("temperature"),
            max_tokens=request_kwargs.get("max_tokens"),
            response_format=request_kwargs.get("response_format"),
            reasoning_effort=request_kwargs.get("reasoning_effort"),
            reasoning=request_kwargs.get("reasoning"),
            include=tuple(request_kwargs.get("include")) if request_kwargs.get("include") is not None else None,
            prompt_cache_key=request_kwargs.get("prompt_cache_key"),
            prompt_cache_retention=request_kwargs.get("prompt_cache_retention"),
            extra=extra,
            stream=stream,
        )
    return ContentRequestEnvelope.from_request_spec(
        build_request_spec(
            messages=messages,
            provider=provider,
            engine=engine,
            request_kwargs=request_kwargs,
            model=model,
            stream=stream,
        )
    )


__all__ = [
    "build_content_request_envelope",
    "build_request_spec",
    "infer_provider_name",
    "infer_model_name",
]
