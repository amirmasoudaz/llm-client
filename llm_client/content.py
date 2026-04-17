"""
Canonical content model and message adapters for llm-client.

This module provides typed content blocks that can represent text, multimodal
inputs, reasoning traces, tool calls, and tool results while remaining
compatible with the current `Message` abstraction.
"""

from __future__ import annotations

import json
import mimetypes
from base64 import b64decode, b64encode
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

from .providers.types import (
    CompletionResult,
    Message,
    MessageInput,
    Role,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolCallDelta,
    normalize_messages,
)


class ContentBlockType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    FILE = "file"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    METADATA = "metadata"


class ContentHandlingMode(str, Enum):
    LOSSY = "lossy"
    STRICT = "strict"


@dataclass(frozen=True)
class ContentDegradation:
    provider: str
    block_type: ContentBlockType
    reason: str
    replacement_text: str | None = None


@dataclass(frozen=True)
class ContentProjection:
    blocks: tuple[ContentBlock, ...]
    degradations: tuple[ContentDegradation, ...] = ()


class UnsupportedContentError(ValueError):
    def __init__(
        self,
        *,
        provider: str,
        block_type: ContentBlockType,
        reason: str,
    ) -> None:
        self.provider = provider
        self.block_type = block_type
        self.reason = reason
        super().__init__(f"{provider} cannot represent {block_type.value} content: {reason}")


@dataclass(frozen=True)
class TextBlock:
    text: str
    type: ContentBlockType = ContentBlockType.TEXT

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "text": self.text}


@dataclass(frozen=True)
class ImageBlock:
    image_url: str
    detail: str | None = None
    mime_type: str | None = None
    type: ContentBlockType = ContentBlockType.IMAGE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type.value, "image_url": self.image_url}
        if self.detail:
            payload["detail"] = self.detail
        if self.mime_type:
            payload["mime_type"] = self.mime_type
        return payload


@dataclass(frozen=True)
class AudioBlock:
    audio_url: str | None = None
    data: str | None = None
    mime_type: str | None = None
    transcript: str | None = None
    type: ContentBlockType = ContentBlockType.AUDIO

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type.value}
        if self.audio_url is not None:
            payload["audio_url"] = self.audio_url
        if self.data is not None:
            payload["data"] = self.data
        if self.mime_type is not None:
            payload["mime_type"] = self.mime_type
        if self.transcript is not None:
            payload["transcript"] = self.transcript
        return payload


@dataclass(frozen=True)
class FileBlock:
    file_url: str | None = None
    file_id: str | None = None
    file_path: str | None = None
    data: str | bytes | None = None
    name: str | None = None
    mime_type: str | None = None
    extracted_text: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    type: ContentBlockType = ContentBlockType.FILE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type.value}
        if self.file_url is not None:
            payload["file_url"] = self.file_url
        if self.file_id is not None:
            payload["file_id"] = self.file_id
        if self.file_path is not None:
            payload["file_path"] = self.file_path
        if self.data is not None:
            payload["data"] = self.data.decode("utf-8") if isinstance(self.data, bytes) else self.data
        if self.name is not None:
            payload["name"] = self.name
        if self.mime_type is not None:
            payload["mime_type"] = self.mime_type
        if self.extracted_text is not None:
            payload["extracted_text"] = self.extracted_text
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        return payload


@dataclass(frozen=True)
class ReasoningBlock:
    text: str
    signature: str | None = None
    type: ContentBlockType = ContentBlockType.REASONING

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type.value, "text": self.text}
        if self.signature is not None:
            payload["signature"] = self.signature
        return payload


@dataclass(frozen=True)
class ToolCallBlock:
    id: str
    name: str
    arguments: str
    type: ContentBlockType = ContentBlockType.TOOL_CALL

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    def to_tool_call(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments=self.arguments)


@dataclass(frozen=True)
class ToolResultBlock:
    tool_call_id: str
    content: str
    name: str | None = None
    is_error: bool = False
    type: ContentBlockType = ContentBlockType.TOOL_RESULT

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type.value,
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class MetadataBlock:
    data: dict[str, Any] = field(default_factory=dict)
    type: ContentBlockType = ContentBlockType.METADATA

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "data": dict(self.data)}


ContentBlock = (
    TextBlock
    | ImageBlock
    | AudioBlock
    | FileBlock
    | ReasoningBlock
    | ToolCallBlock
    | ToolResultBlock
    | MetadataBlock
)
SerializedContentBlock = dict[str, Any]
MessageContent = (
    str
    | dict[str, Any]
    | list[ContentBlock]
    | list[SerializedContentBlock]
    | tuple[ContentBlock, ...]
    | tuple[SerializedContentBlock, ...]
    | None
)


@dataclass(frozen=True)
class ContentMessage:
    role: Role
    blocks: tuple[ContentBlock, ...] = ()
    name: str | None = None
    tool_call_id: str | None = None

    def to_message(self) -> Message:
        return message_from_content_blocks(
            role=self.role,
            blocks=list(self.blocks),
            name=self.name,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "role": self.role.value,
            "content": [content_block_to_dict(block) for block in self.blocks],
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    @classmethod
    def from_message(cls, message: Message | dict[str, Any] | str) -> ContentMessage:
        if isinstance(message, str):
            return cls(role=Role.USER, blocks=(TextBlock(message),))
        if isinstance(message, dict):
            message = normalize_messages([message])[0]
        return cls(
            role=message.role,
            blocks=tuple(message_to_content_blocks(message)),
            name=message.name,
            tool_call_id=message.tool_call_id,
        )


@dataclass(frozen=True)
class ContentRequestEnvelope:
    provider: str
    model: str
    messages: tuple[ContentMessage, ...]
    tools: tuple[Any, ...] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str | dict[str, Any] | type | None = None
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None
    include: tuple[str, ...] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    stream: bool = False

    def to_request_spec(self) -> Any:
        from .spec import RequestSpec

        return RequestSpec(
            provider=self.provider,
            model=self.model,
            messages=[message.to_message() for message in self.messages],
            tools=list(self.tools) if self.tools is not None else None,
            tool_choice=self.tool_choice,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=self.response_format,
            reasoning_effort=self.reasoning_effort,
            reasoning=self.reasoning,
            include=list(self.include) if self.include is not None else None,
            prompt_cache_key=self.prompt_cache_key,
            prompt_cache_retention=self.prompt_cache_retention,
            extra=dict(self.extra),
            stream=self.stream,
        )

    @classmethod
    def from_request_spec(cls, spec: Any) -> ContentRequestEnvelope:
        return cls(
            provider=spec.provider,
            model=spec.model,
            messages=tuple(ContentMessage.from_message(message) for message in spec.messages),
            tools=tuple(spec.tools) if spec.tools is not None else None,
            tool_choice=spec.tool_choice,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            response_format=spec.response_format,
            reasoning_effort=spec.reasoning_effort,
            reasoning=dict(spec.reasoning) if isinstance(spec.reasoning, dict) else spec.reasoning,
            include=tuple(spec.include) if spec.include is not None else None,
            prompt_cache_key=spec.prompt_cache_key,
            prompt_cache_retention=spec.prompt_cache_retention,
            extra=dict(spec.extra),
            stream=bool(spec.stream),
        )


@dataclass(frozen=True)
class ContentResponseEnvelope:
    message: ContentMessage
    usage: Any | None = None
    model: str | None = None
    finish_reason: str | None = None
    status: int = 200
    error: str | None = None
    reasoning: str | None = None
    raw_response: Any | None = None

    def to_completion_result(self) -> Any:
        from .providers.types import CompletionResult, NormalizedOutputItem

        assistant_message = self.message.to_message()
        content_blocks = [block for block in self.message.blocks if not isinstance(block, ToolCallBlock)]
        content_text = content_blocks_to_text(content_blocks)
        provider_items: list[dict[str, Any]] | None = None
        normalized_output_items: list[Any] | None = None
        refusal: str | None = None
        for block in self.message.blocks:
            if not isinstance(block, MetadataBlock):
                continue
            data = block.data
            if data.get("provider") != "openai":
                continue
            if isinstance(data.get("responses_output"), list):
                provider_items = [dict(item) for item in data["responses_output"] if isinstance(item, dict)]
            if isinstance(data.get("normalized_output_items"), list):
                normalized_output_items = [
                    NormalizedOutputItem.from_dict(item)
                    for item in data["normalized_output_items"]
                    if isinstance(item, dict)
                ] or None
            if isinstance(data.get("refusal"), str):
                refusal = data["refusal"]
        return CompletionResult(
            content=content_text or None,
            tool_calls=assistant_message.tool_calls,
            usage=self.usage,
            reasoning=self.reasoning,
            refusal=refusal,
            model=self.model,
            finish_reason=self.finish_reason,
            status=self.status,
            error=self.error,
            raw_response=self.raw_response,
            output_items=normalized_output_items,
            provider_items=provider_items,
        )

    @classmethod
    def from_completion_result(cls, result: Any) -> ContentResponseEnvelope:
        base_message = result.to_message() if hasattr(result, "to_message") else Message.assistant(content=getattr(result, "content", None))
        return cls(
            message=ContentMessage.from_message(base_message),
            usage=getattr(result, "usage", None),
            model=getattr(result, "model", None),
            finish_reason=getattr(result, "finish_reason", None),
            status=int(getattr(result, "status", 200) or 200),
            error=getattr(result, "error", None),
            reasoning=getattr(result, "reasoning", None),
            raw_response=getattr(result, "raw_response", None),
        )


def content_block_to_dict(block: ContentBlock | SerializedContentBlock) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)
    if hasattr(block, "to_dict"):
        return block.to_dict()
    raise TypeError(f"Unsupported content block type: {type(block)}")


def content_block_from_dict(data: dict[str, Any]) -> ContentBlock:
    block_type = str(data.get("type") or "").strip().lower()
    if block_type in {"text", "input_text", "output_text"}:
        return TextBlock(text=str(data.get("text") or data.get("content") or ""))
    if block_type in {"image", "image_url", "input_image"}:
        image_url = data.get("image_url")
        if isinstance(image_url, dict):
            return ImageBlock(
                image_url=str(image_url.get("url") or ""),
                detail=image_url.get("detail"),
                mime_type=data.get("mime_type"),
            )
        return ImageBlock(
            image_url=str(image_url or data.get("url") or ""),
            detail=data.get("detail"),
            mime_type=data.get("mime_type"),
        )
    if block_type in {"audio", "input_audio"}:
        input_audio = data.get("input_audio")
        if isinstance(input_audio, dict):
            return AudioBlock(
                data=input_audio.get("data"),
                mime_type=input_audio.get("format") or data.get("mime_type"),
                transcript=data.get("transcript"),
            )
        return AudioBlock(
            audio_url=data.get("audio_url"),
            data=data.get("data"),
            mime_type=data.get("mime_type"),
            transcript=data.get("transcript"),
        )
    if block_type in {"file", "input_file"}:
        return FileBlock(
            file_url=data.get("file_url"),
            file_id=data.get("file_id"),
            file_path=data.get("file_path"),
            data=data.get("data"),
            name=data.get("name"),
            mime_type=data.get("mime_type"),
            extracted_text=data.get("extracted_text"),
            sha256=data.get("sha256"),
            size_bytes=data.get("size_bytes"),
        )
    if block_type == "reasoning":
        return ReasoningBlock(text=str(data.get("text") or ""), signature=data.get("signature"))
    if block_type in {"tool_call", "function_call"}:
        function_payload = data.get("function")
        if isinstance(function_payload, dict):
            return ToolCallBlock(
                id=str(data.get("id") or ""),
                name=str(function_payload.get("name") or ""),
                arguments=str(function_payload.get("arguments") or ""),
            )
        return ToolCallBlock(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            arguments=str(data.get("arguments") or ""),
        )
    if block_type in {"tool_result", "function_result"}:
        return ToolResultBlock(
            tool_call_id=str(data.get("tool_call_id") or data.get("id") or ""),
            content=str(data.get("content") or ""),
            name=data.get("name"),
            is_error=bool(data.get("is_error", False)),
        )
    if block_type == "metadata":
        payload = data.get("data")
        return MetadataBlock(data=dict(payload) if isinstance(payload, dict) else dict(data))
    raise ValueError(f"Unsupported content block type: {block_type!r}")


def normalize_content_blocks(content: MessageContent) -> list[ContentBlock]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextBlock(content)]
    if isinstance(content, dict):
        return [TextBlock(json.dumps(content, ensure_ascii=False, sort_keys=True, default=str))]
    if isinstance(content, (list, tuple)):
        blocks: list[ContentBlock] = []
        for item in content:
            if isinstance(
                item,
                (TextBlock, ImageBlock, AudioBlock, FileBlock, ReasoningBlock, ToolCallBlock, ToolResultBlock, MetadataBlock),
            ):
                blocks.append(item)
            elif isinstance(item, dict):
                blocks.append(content_block_from_dict(item))
            else:
                raise TypeError(f"Unsupported content item: {type(item)}")
        return blocks
    raise TypeError(f"Unsupported content value: {type(content)}")


def serialize_message_content(content: MessageContent) -> str | list[dict[str, Any]] | None:
    if content is None or isinstance(content, str):
        return content
    return [content_block_to_dict(block) for block in normalize_content_blocks(content)]


def content_blocks_to_text(blocks: list[ContentBlock] | list[dict[str, Any]] | str | None) -> str:
    normalized = normalize_content_blocks(blocks)
    parts: list[str] = []
    for block in normalized:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ReasoningBlock):
            parts.append(block.text)
        elif isinstance(block, ToolResultBlock):
            parts.append(block.content)
    return "\n".join(part for part in parts if part).strip()


def _content_placeholder_text(block: ContentBlock) -> str | None:
    if isinstance(block, ImageBlock):
        return f"[image] {block.image_url}" if block.image_url else "[image]"
    if isinstance(block, AudioBlock):
        if block.transcript:
            return block.transcript
        if block.audio_url:
            return f"[audio] {block.audio_url}"
        return "[audio]"
    if isinstance(block, FileBlock):
        if block.extracted_text:
            return block.extracted_text
        ref = block.file_id or block.file_url or block.name or "file"
        return f"[file] {ref}"
    if isinstance(block, ToolCallBlock):
        return f"[tool_call] {block.name}"
    return None


def _metadata_block_to_text(block: MetadataBlock) -> str:
    if not block.data:
        return "Metadata: {}"
    pairs: list[str] = []
    for key, value in block.data.items():
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            rendered = str(value)
        pairs.append(f"{key}={rendered}")
    return "Metadata: " + "; ".join(pairs)


def project_content_blocks(
    blocks: list[ContentBlock] | list[dict[str, Any]] | str | None,
    *,
    provider: str,
    mode: ContentHandlingMode = ContentHandlingMode.LOSSY,
    include_metadata: bool = True,
    supports_images: bool = False,
    supports_audio_data: bool = False,
    supports_audio_url: bool = False,
    supports_files: bool = False,
    allow_tool_calls: bool = False,
) -> ContentProjection:
    normalized = normalize_content_blocks(blocks)
    projected: list[ContentBlock] = []
    degradations: list[ContentDegradation] = []

    def _degrade(block: ContentBlock, *, reason: str) -> None:
        replacement_text = _content_placeholder_text(block)
        if mode is ContentHandlingMode.STRICT:
            raise UnsupportedContentError(provider=provider, block_type=block.type, reason=reason)
        degradations.append(
            ContentDegradation(
                provider=provider,
                block_type=block.type,
                reason=reason,
                replacement_text=replacement_text,
            )
        )
        if replacement_text:
            projected.append(TextBlock(replacement_text))

    for block in normalized:
        if isinstance(block, (TextBlock, ReasoningBlock, ToolResultBlock)):
            projected.append(block)
            continue
        if isinstance(block, MetadataBlock):
            if include_metadata:
                projected.append(TextBlock(_metadata_block_to_text(block)))
            continue
        if isinstance(block, ImageBlock):
            if supports_images:
                projected.append(block)
            else:
                _degrade(block, reason="provider does not support image content")
            continue
        if isinstance(block, AudioBlock):
            if block.data and supports_audio_data:
                projected.append(block)
            elif block.audio_url and supports_audio_url:
                projected.append(block)
            else:
                _degrade(block, reason="provider does not support this audio content form")
            continue
        if isinstance(block, FileBlock):
            if supports_files:
                projected.append(block)
            else:
                _degrade(block, reason="provider does not support file content")
            continue
        if isinstance(block, ToolCallBlock):
            if allow_tool_calls:
                projected.append(block)
            else:
                _degrade(block, reason="tool call blocks are not allowed in message content")
            continue
        projected.append(block)

    return ContentProjection(blocks=tuple(projected), degradations=tuple(degradations))


def message_to_content_blocks(message: Message | dict[str, Any] | str) -> list[ContentBlock]:
    if isinstance(message, str):
        return [TextBlock(message)]
    if isinstance(message, dict):
        message = normalize_messages([message])[0]
    tool_result_id = getattr(message, "tool_call_id", None)
    if tool_result_id:
        blocks: list[ContentBlock] = []
    else:
        blocks = normalize_content_blocks(getattr(message, "content", None))
    if getattr(message, "tool_calls", None):
        blocks.extend(ToolCallBlock(id=tc.id, name=tc.name, arguments=tc.arguments) for tc in (message.tool_calls or []))
    if tool_result_id and getattr(message, "content", None):
        blocks.append(
            ToolResultBlock(
                tool_call_id=str(tool_result_id),
                content=content_blocks_to_text(getattr(message, "content", None)),
                name=getattr(message, "name", None),
            )
        )
    return blocks


def message_from_content_blocks(
    *,
    role: Role,
    blocks: list[ContentBlock] | list[dict[str, Any]],
    name: str | None = None,
) -> Message:
    normalized = normalize_content_blocks(blocks)
    tool_calls = [block.to_tool_call() for block in normalized if isinstance(block, ToolCallBlock)] or None
    tool_results = [block for block in normalized if isinstance(block, ToolResultBlock)]
    content_blocks = [block for block in normalized if not isinstance(block, (ToolCallBlock, ToolResultBlock))]
    tool_call_id = tool_results[0].tool_call_id if tool_results else None
    content: str | list[dict[str, Any]] | None
    if content_blocks:
        content = serialize_message_content(content_blocks)
    elif tool_results:
        content = tool_results[0].content
    else:
        content = None
    return Message(role=role, content=content, name=name, tool_calls=tool_calls, tool_call_id=tool_call_id)


def content_blocks_to_openai_chat_content(
    blocks: list[ContentBlock] | list[dict[str, Any]] | str | None,
    *,
    mode: ContentHandlingMode = ContentHandlingMode.LOSSY,
) -> str | list[dict[str, Any]] | None:
    if blocks is None or isinstance(blocks, str):
        return blocks
    projection = project_content_blocks(
        blocks,
        provider="openai",
        mode=mode,
        supports_images=True,
        supports_audio_data=True,
        supports_audio_url=False,
        supports_files=False,
        allow_tool_calls=False,
    )
    parts: list[dict[str, Any]] = []
    for block in projection.blocks:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            image_payload: dict[str, Any] = {"url": block.image_url}
            if block.detail:
                image_payload["detail"] = block.detail
            parts.append({"type": "image_url", "image_url": image_payload})
        elif isinstance(block, AudioBlock):
            if block.data:
                audio_payload: dict[str, Any] = {"data": block.data}
                if block.mime_type:
                    audio_payload["format"] = block.mime_type.split("/")[-1]
                parts.append({"type": "input_audio", "input_audio": audio_payload})
            elif block.audio_url:
                parts.append({"type": "text", "text": f"[audio] {block.audio_url}"})
        elif isinstance(block, ReasoningBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolResultBlock):
            parts.append({"type": "text", "text": block.content})
        elif isinstance(block, MetadataBlock):
            continue
    return parts


def _normalize_file_data(data: str | bytes) -> tuple[str, bytes]:
    if isinstance(data, bytes):
        return b64encode(data).decode("utf-8"), data
    encoded = str(data)
    if encoded.startswith("data:") and ";base64," in encoded:
        try:
            _, base64_payload = encoded.split(";base64,", 1)
            decoded = b64decode(base64_payload, validate=True)
            return encoded, decoded
        except Exception:
            pass
    try:
        decoded = b64decode(encoded, validate=True)
        return encoded, decoded
    except Exception:
        raw = encoded.encode("utf-8")
        return b64encode(raw).decode("utf-8"), raw


def _openai_file_data_uri(block: FileBlock) -> str:
    data = block.data or ""
    if data.startswith("data:"):
        return data
    mime_type = block.mime_type or "application/octet-stream"
    return f"data:{mime_type};base64,{data}"


def prepare_file_block(block: FileBlock) -> FileBlock:
    if not any((block.file_id, block.file_url, block.file_path, block.data)):
        raise ValueError("FileBlock requires one of: file_id, file_url, file_path, or data")

    name = block.name
    mime_type = block.mime_type
    data = block.data
    sha256_value = block.sha256
    size_bytes = block.size_bytes

    if block.file_path:
        file_path = Path(block.file_path).expanduser().resolve()
        raw_bytes = file_path.read_bytes()
        encoded, normalized_bytes = _normalize_file_data(raw_bytes)
        data = encoded
        name = name or file_path.name
        mime_type = mime_type or mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        sha256_value = sha256_value or sha256(normalized_bytes).hexdigest()
        size_bytes = size_bytes or len(normalized_bytes)
    elif data is not None:
        encoded, normalized_bytes = _normalize_file_data(data)
        data = encoded
        sha256_value = sha256_value or sha256(normalized_bytes).hexdigest()
        size_bytes = size_bytes or len(normalized_bytes)

    return FileBlock(
        file_url=block.file_url,
        file_id=block.file_id,
        file_path=block.file_path,
        data=data,
        name=name,
        mime_type=mime_type,
        extracted_text=block.extracted_text,
        sha256=sha256_value,
        size_bytes=size_bytes,
    )


def prepare_content_blocks_for_transport(
    blocks: list[ContentBlock] | list[dict[str, Any]] | str | None,
) -> list[ContentBlock]:
    normalized = normalize_content_blocks(blocks)
    prepared: list[ContentBlock] = []
    for block in normalized:
        if isinstance(block, FileBlock):
            prepared.append(prepare_file_block(block))
        else:
            prepared.append(block)
    return prepared


def content_blocks_to_openai_responses_content(
    blocks: list[ContentBlock] | list[dict[str, Any]] | str | None,
    *,
    mode: ContentHandlingMode = ContentHandlingMode.LOSSY,
) -> str | list[dict[str, Any]] | None:
    if blocks is None or isinstance(blocks, str):
        return blocks
    prepared_blocks = prepare_content_blocks_for_transport(blocks)
    parts: list[dict[str, Any]] = []
    for block in prepared_blocks:
        if isinstance(block, TextBlock):
            parts.append({"type": "input_text", "text": block.text})
            continue
        if isinstance(block, ImageBlock):
            image_payload: dict[str, Any] = {"type": "input_image", "image_url": block.image_url}
            if block.detail:
                image_payload["detail"] = block.detail
            parts.append(image_payload)
            continue
        if isinstance(block, AudioBlock):
            if block.data:
                audio_payload: dict[str, Any] = {"data": block.data}
                if block.mime_type:
                    audio_payload["format"] = block.mime_type.split("/")[-1]
                parts.append({"type": "input_audio", "input_audio": audio_payload})
            elif block.audio_url:
                if mode is ContentHandlingMode.STRICT:
                    raise UnsupportedContentError(
                        provider="openai",
                        block_type=block.type,
                        reason="responses api requires inline audio data for native audio transport",
                    )
                parts.append({"type": "input_text", "text": _content_placeholder_text(block) or "[audio]"})
            continue
        if isinstance(block, FileBlock):
            if block.file_id:
                parts.append({"type": "input_file", "file_id": block.file_id})
                continue
            if block.file_url:
                parts.append({"type": "input_file", "file_url": block.file_url})
                continue
            if block.data:
                filename = block.name or "file"
                parts.append(
                    {
                        "type": "input_file",
                        "filename": filename,
                        "file_data": _openai_file_data_uri(block),
                    }
                )
                continue
            fallback_text = _content_placeholder_text(block)
            if mode is ContentHandlingMode.STRICT:
                raise UnsupportedContentError(
                    provider="openai",
                    block_type=block.type,
                    reason="responses api requires file_id or inline file data for native file transport",
                )
            if fallback_text:
                parts.append({"type": "input_text", "text": fallback_text})
            continue
        if isinstance(block, ReasoningBlock):
            parts.append({"type": "input_text", "text": block.text})
            continue
        if isinstance(block, ToolResultBlock):
            parts.append({"type": "input_text", "text": block.content})
            continue
        if isinstance(block, MetadataBlock):
            parts.append({"type": "input_text", "text": _metadata_block_to_text(block)})
            continue
    return parts


def content_blocks_to_anthropic_content(
    blocks: list[ContentBlock] | list[dict[str, Any]] | str | None,
    *,
    mode: ContentHandlingMode = ContentHandlingMode.LOSSY,
    allow_tool_use: bool = False,
) -> str | list[dict[str, Any]]:
    projection = project_content_blocks(
        blocks,
        provider="anthropic",
        mode=mode,
        supports_images=False,
        supports_audio_data=False,
        supports_audio_url=False,
        supports_files=False,
        allow_tool_calls=allow_tool_use,
    )
    content: list[dict[str, Any]] = []
    for block in projection.blocks:
        if isinstance(block, TextBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ReasoningBlock):
            content.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolResultBlock):
            content.append({"type": "text", "text": block.content})
        elif isinstance(block, ToolCallBlock) and allow_tool_use:
            try:
                parsed_args = json.loads(block.arguments) if block.arguments else {}
            except json.JSONDecodeError:
                parsed_args = {}
            input_data: dict[str, Any] = parsed_args if isinstance(parsed_args, dict) else {}
            content.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": input_data,
                }
            )
    if len(content) == 1 and content[0].get("type") == "text":
        return str(content[0].get("text") or "")
    return content


def content_blocks_to_google_parts(
    blocks: list[ContentBlock] | list[dict[str, Any]] | str | None,
    *,
    types_module: Any,
    mode: ContentHandlingMode = ContentHandlingMode.LOSSY,
) -> list[Any]:
    projection = project_content_blocks(
        blocks,
        provider="google",
        mode=mode,
        supports_images=False,
        supports_audio_data=False,
        supports_audio_url=False,
        supports_files=False,
        allow_tool_calls=False,
    )
    parts: list[Any] = []
    for block in projection.blocks:
        if isinstance(block, TextBlock):
            parts.append(types_module.Part.from_text(text=block.text))
        elif isinstance(block, ReasoningBlock):
            parts.append(types_module.Part.from_text(text=block.text))
        elif isinstance(block, ToolResultBlock):
            parts.append(types_module.Part.from_text(text=block.content))
    return parts


def message_to_openai_chat_dict(message: Message) -> dict[str, Any]:
    return message_to_openai_chat_dict_with_mode(message)


def message_to_openai_chat_dict_with_mode(
    message: Message,
    *,
    mode: ContentHandlingMode = ContentHandlingMode.LOSSY,
) -> dict[str, Any]:
    payload = message.to_dict()
    payload["content"] = content_blocks_to_openai_chat_content(getattr(message, "content", None), mode=mode)
    if payload.get("content") is None:
        payload.pop("content", None)
    return payload


def ensure_content_response_envelope(value: Any) -> ContentResponseEnvelope:
    if isinstance(value, ContentResponseEnvelope):
        return value
    if isinstance(value, CompletionResult):
        return ContentResponseEnvelope.from_completion_result(value)
    if hasattr(value, "to_message") or any(hasattr(value, attr) for attr in ("content", "tool_calls", "usage", "status", "error")):
        return ContentResponseEnvelope.from_completion_result(value)
    raise TypeError(f"Unsupported content response value: {type(value)}")


def ensure_completion_result(value: Any) -> CompletionResult:
    if isinstance(value, CompletionResult):
        return value
    if isinstance(value, ContentResponseEnvelope):
        return value.to_completion_result()
    raise TypeError(f"Unsupported completion value: {type(value)}")


def completion_stream_event_to_content_event(event: StreamEvent) -> StreamEvent:
    if event.type != StreamEventType.DONE:
        return event
    return StreamEvent(
        type=event.type,
        data=ensure_content_response_envelope(event.data),
        timestamp=event.timestamp,
    )


def content_stream_event_to_completion_event(event: StreamEvent) -> StreamEvent:
    if event.type != StreamEventType.DONE:
        return event
    return StreamEvent(
        type=event.type,
        data=ensure_completion_result(event.data),
        timestamp=event.timestamp,
    )


__all__ = [
    "Role",
    "Message",
    "MessageInput",
    "ToolCall",
    "ToolCallDelta",
    "normalize_messages",
    "ContentBlockType",
    "ContentHandlingMode",
    "ContentDegradation",
    "ContentProjection",
    "UnsupportedContentError",
    "TextBlock",
    "ImageBlock",
    "AudioBlock",
    "FileBlock",
    "ReasoningBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "MetadataBlock",
    "ContentBlock",
    "MessageContent",
    "ContentMessage",
    "ContentRequestEnvelope",
    "ContentResponseEnvelope",
    "content_block_to_dict",
    "content_block_from_dict",
    "normalize_content_blocks",
    "serialize_message_content",
    "content_blocks_to_text",
    "project_content_blocks",
    "message_to_content_blocks",
    "message_from_content_blocks",
    "content_blocks_to_anthropic_content",
    "content_blocks_to_google_parts",
    "content_blocks_to_openai_chat_content",
    "content_blocks_to_openai_responses_content",
    "message_to_openai_chat_dict_with_mode",
    "message_to_openai_chat_dict",
    "prepare_file_block",
    "prepare_content_blocks_for_transport",
    "ensure_content_response_envelope",
    "ensure_completion_result",
    "completion_stream_event_to_content_event",
    "content_stream_event_to_completion_event",
]
