"""
Core types for the provider abstraction layer.

These types provide a unified interface for LLM responses, streaming events,
and messages across different providers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StreamEventType(str, Enum):
    """Types of events emitted during streaming."""

    # Content events
    TOKEN = "token"
    REASONING = "reasoning"

    # Tool calling events
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"

    # Metadata events
    META = "meta"
    USAGE = "usage"

    # Terminal events
    DONE = "done"
    ERROR = "error"


@dataclass
class ToolCall:
    """Represents a tool/function call made by the model."""

    id: str
    name: str
    arguments: str  # JSON string of arguments

    def parse_arguments(self) -> dict[str, Any]:
        """Parse the JSON arguments string."""
        return json.loads(self.arguments) if self.arguments else {}


@dataclass
class ToolCallDelta:
    """Partial tool call data during streaming."""

    id: str
    index: int
    name: str | None = None
    arguments_delta: str = ""


@dataclass
class Message:
    """A message in a conversation."""

    role: Role
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool response messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-compatible dictionary format."""
        d: dict[str, Any] = {"role": self.role.value}

        if self.content is not None:
            d["content"] = self._serialize_content(self.content)
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id

        return d

    @staticmethod
    def _serialize_content(content: str | list[Any]) -> str | list[Any]:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            serialized: list[Any] = []
            for item in content:
                if isinstance(item, dict):
                    serialized.append(dict(item))
                elif hasattr(item, "to_dict"):
                    serialized.append(item.to_dict())
                else:
                    serialized.append(item)
            return serialized
        return content

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create a Message from a dictionary."""
        role = Role(data["role"])
        content = data.get("content")
        name = data.get("name")
        tool_call_id = data.get("tool_call_id")

        tool_calls = None
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=tc["function"]["arguments"])
                for tc in data["tool_calls"]
            ]

        return cls(role=role, content=content, name=name, tool_calls=tool_calls, tool_call_id=tool_call_id)

    @classmethod
    def user(cls, content: str | list[Any]) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str | list[Any] | None = None, tool_calls: list[ToolCall] | None = None) -> Message:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def system(cls, content: str | list[Any]) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str | list[Any], name: str | None = None) -> Message:
        """Create a tool result message."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_cached: int = 0
    output_tokens_reasoning: int = 0

    # Cost tracking
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_tokens_cached": self.input_tokens_cached,
            "output_tokens_reasoning": self.output_tokens_reasoning,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Usage:
        """Create Usage from a dictionary."""
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            input_tokens_cached=data.get("input_tokens_cached", 0),
            output_tokens_reasoning=data.get("output_tokens_reasoning", 0),
            input_cost=data.get("input_cost", 0.0),
            output_cost=data.get("output_cost", 0.0),
            total_cost=data.get("total_cost", 0.0),
        )


@dataclass
class StreamEvent:
    """
    A unified streaming event emitted by providers.

    Event types and their data:
    - TOKEN: str (the token text)
    - REASONING: str (reasoning/thinking content)
    - TOOL_CALL_START: ToolCallDelta (initial tool call info)
    - TOOL_CALL_DELTA: ToolCallDelta (argument chunks)
    - TOOL_CALL_END: ToolCall (complete tool call)
    - META: dict (model info, stream metadata)
    - USAGE: Usage (token counts and costs)
    - DONE: CompletionResult (final result)
    - ERROR: dict (error info with status and message)
    """

    type: StreamEventType
    data: Any
    sequence_number: int | None = None
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as a Server-Sent Event string."""
        event_name = self.type.value

        if isinstance(self.data, str):
            data_str = self.data
        elif hasattr(self.data, "to_dict"):
            data_str = json.dumps(self.data.to_dict())
        elif isinstance(self.data, (dict, list)):
            data_str = json.dumps(self.data)
        else:
            data_str = str(self.data)

        return f"event: {event_name}\ndata: {data_str}\n\n"


@dataclass(frozen=True)
class NormalizedOutputItem:
    """Stable normalized output-item view for provider-specific rich outputs."""

    type: str
    id: str | None = None
    call_id: str | None = None
    status: str | None = None
    name: str | None = None
    text: str | None = None
    url: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.id is not None:
            payload["id"] = self.id
        if self.call_id is not None:
            payload["call_id"] = self.call_id
        if self.status is not None:
            payload["status"] = self.status
        if self.name is not None:
            payload["name"] = self.name
        if self.text is not None:
            payload["text"] = self.text
        if self.url is not None:
            payload["url"] = self.url
        if self.details:
            payload["details"] = dict(self.details)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedOutputItem:
        return cls(
            type=str(data.get("type") or ""),
            id=data.get("id"),
            call_id=data.get("call_id"),
            status=data.get("status"),
            name=data.get("name"),
            text=data.get("text"),
            url=data.get("url"),
            details=dict(data.get("details") or {}),
        )


@dataclass
class CompletionResult:
    """
    Result of a completion request.

    This unified result type works for both streaming and non-streaming completions.
    """

    content: str | list[Any] | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    reasoning: str | None = None
    refusal: str | None = None
    output_items: list[NormalizedOutputItem] | None = None

    # Request metadata
    model: str | None = None
    finish_reason: str | None = None

    # Status tracking
    status: int = 200
    error: str | None = None

    # Original response for debugging
    raw_response: Any | None = field(default=None, repr=False)
    provider_items: list[dict[str, Any]] | None = None

    @property
    def ok(self) -> bool:
        """Check if the request was successful."""
        return self.status == 200 and self.error is None

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return bool(self.tool_calls)

    def to_message(self) -> Message:
        """Convert this result to an assistant message."""
        content = Message._serialize_content(self.content) if self.content is not None else None
        if self.provider_items or self.output_items or self.refusal is not None:
            content_items: list[Any] = []
            if isinstance(content, str):
                content_items.append({"type": "text", "text": content})
            elif isinstance(content, list):
                content_items.extend(content)
            elif content is not None:
                content_items.append({"type": "text", "text": str(content)})
            metadata_payload: dict[str, Any] = {"provider": "openai"}
            if self.provider_items:
                metadata_payload["responses_output"] = [dict(item) for item in self.provider_items]
            if self.output_items:
                metadata_payload["normalized_output_items"] = [item.to_dict() for item in self.output_items]
            if self.refusal is not None:
                metadata_payload["refusal"] = self.refusal
            content_items.append(
                {
                    "type": "metadata",
                    "data": metadata_payload,
                }
            )
            content = content_items
        return Message.assistant(content=content, tool_calls=self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": Message._serialize_content(self.content) if self.content is not None else None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in (self.tool_calls or [])],
            "usage": self.usage.to_dict() if self.usage else None,
            "reasoning": self.reasoning,
            "refusal": self.refusal,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "status": self.status,
            "error": self.error,
            "output_items": [item.to_dict() for item in (self.output_items or [])] or None,
            "provider_items": [dict(item) for item in (self.provider_items or [])] or None,
        }


@dataclass
class BackgroundResponseResult:
    """Lifecycle state for a background provider response."""

    response_id: str
    lifecycle_status: str
    completion: CompletionResult | None = None
    error: str | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None and self.lifecycle_status != "failed"

    @property
    def is_terminal(self) -> bool:
        return self.lifecycle_status in {"completed", "failed", "incomplete", "cancelled", "canceled"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "lifecycle_status": self.lifecycle_status,
            "completion": self.completion.to_dict() if self.completion else None,
            "error": self.error,
        }


@dataclass
class DeepResearchRunResult:
    """Result of a staged deep-research orchestration workflow."""

    prompt: str
    effective_prompt: str
    clarification: CompletionResult | None = None
    rewrite: CompletionResult | None = None
    queued: CompletionResult | None = None
    response_id: str | None = None
    background: BackgroundResponseResult | None = None

    @property
    def ok(self) -> bool:
        if self.background is not None:
            return self.background.ok
        return self.queued.ok if self.queued is not None else False

    @property
    def final_completion(self) -> CompletionResult | None:
        if self.background and self.background.completion is not None:
            return self.background.completion
        return self.queued

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "effective_prompt": self.effective_prompt,
            "clarification": self.clarification.to_dict() if self.clarification else None,
            "rewrite": self.rewrite.to_dict() if self.rewrite else None,
            "queued": self.queued.to_dict() if self.queued else None,
            "response_id": self.response_id,
            "background": self.background.to_dict() if self.background else None,
        }


@dataclass
class ConversationResource:
    """Provider-level conversation lifecycle resource."""

    conversation_id: str
    created_at: int | None = None
    metadata: dict[str, Any] | None = None
    deleted: bool | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.deleted is not False

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "metadata": dict(self.metadata or {}) if self.metadata is not None else None,
            "deleted": self.deleted,
        }


@dataclass
class CompactionResult:
    """Result of `/responses/compact` style context compaction."""

    compaction_id: str
    created_at: int | None = None
    usage: Usage | None = None
    output_items: list[NormalizedOutputItem] | None = None
    provider_items: list[dict[str, Any]] | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "compaction_id": self.compaction_id,
            "created_at": self.created_at,
            "usage": self.usage.to_dict() if self.usage else None,
            "output_items": [item.to_dict() for item in (self.output_items or [])] or None,
            "provider_items": [dict(item) for item in (self.provider_items or [])] or None,
        }


@dataclass
class DeletionResult:
    """Provider-level deletion result for stored resources."""

    resource_id: str
    deleted: bool = True
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.deleted

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "deleted": self.deleted,
        }


@dataclass
class ConversationItemResource:
    """Provider-level representation of a conversation item."""

    item_id: str | None
    item_type: str
    role: str | None = None
    status: str | None = None
    content: Any | None = None
    output_items: list[NormalizedOutputItem] | None = None
    raw_item: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "role": self.role,
            "status": self.status,
            "content": self.content,
            "output_items": [item.to_dict() for item in (self.output_items or [])] or None,
            "raw_item": dict(self.raw_item or {}) if self.raw_item is not None else None,
        }


@dataclass
class ConversationItemsPage:
    """A page of provider conversation items."""

    items: list[ConversationItemResource]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "first_id": self.first_id,
            "last_id": self.last_id,
            "has_more": self.has_more,
        }


@dataclass
class ModerationResult:
    """Result of a moderation request."""

    flagged: bool
    model: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    status: int = 200
    error: str | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flagged": self.flagged,
            "model": self.model,
            "results": [dict(item) for item in self.results],
            "status": self.status,
            "error": self.error,
        }


@dataclass
class GeneratedImage:
    """A single generated or edited image artifact."""

    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None
    raw_item: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "url": self.url,
            "b64_json": self.b64_json,
            "revised_prompt": self.revised_prompt,
        }
        if self.raw_item is not None:
            payload["raw_item"] = dict(self.raw_item)
        return payload


@dataclass
class ImageGenerationResult:
    """Result of an image generation or editing request."""

    images: list[GeneratedImage]
    created_at: int | None = None
    usage: Usage | None = None
    model: str | None = None
    status: int = 200
    error: str | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "images": [image.to_dict() for image in self.images],
            "created_at": self.created_at,
            "usage": self.usage.to_dict() if self.usage else None,
            "model": self.model,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class AudioTranscriptionResult:
    """Result of a speech-to-text or speech-translation request."""

    text: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[dict[str, Any]] | None = None
    words: list[dict[str, Any]] | None = None
    model: str | None = None
    status: int = 200
    error: str | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "segments": [dict(item) for item in (self.segments or [])] or None,
            "words": [dict(item) for item in (self.words or [])] or None,
            "model": self.model,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class AudioSpeechResult:
    """Result of a text-to-speech request."""

    audio: bytes
    format: str
    model: str | None = None
    status: int = 200
    error: str | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def byte_length(self) -> int:
        return len(self.audio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "model": self.model,
            "status": self.status,
            "error": self.error,
            "byte_length": self.byte_length,
        }


@dataclass
class VectorStoreResource:
    """Provider-level representation of a hosted vector store."""

    vector_store_id: str
    name: str | None = None
    status: str | None = None
    file_counts: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    usage_bytes: int | None = None
    expires_at: int | None = None
    last_active_at: int | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_store_id": self.vector_store_id,
            "name": self.name,
            "status": self.status,
            "file_counts": dict(self.file_counts or {}) if self.file_counts is not None else None,
            "metadata": dict(self.metadata or {}) if self.metadata is not None else None,
            "usage_bytes": self.usage_bytes,
            "expires_at": self.expires_at,
            "last_active_at": self.last_active_at,
        }


@dataclass
class VectorStoresPage:
    """A page of vector stores."""

    items: list[VectorStoreResource]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "first_id": self.first_id,
            "last_id": self.last_id,
            "has_more": self.has_more,
        }


@dataclass
class VectorStoreSearchResult:
    """Search results returned by a vector store."""

    vector_store_id: str
    query: str | list[str]
    results: list[dict[str, Any]]
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_store_id": self.vector_store_id,
            "query": self.query,
            "results": [dict(item) for item in self.results],
        }


@dataclass
class FineTuningJobResult:
    """Provider-level representation of a fine-tuning job."""

    job_id: str
    status: str
    base_model: str | None = None
    fine_tuned_model: str | None = None
    created_at: int | None = None
    finished_at: int | None = None
    trained_tokens: int | None = None
    training_file: str | None = None
    validation_file: str | None = None
    result_files: list[str] | None = None
    metadata: dict[str, Any] | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status not in {"failed", "cancelled"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "base_model": self.base_model,
            "fine_tuned_model": self.fine_tuned_model,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "trained_tokens": self.trained_tokens,
            "training_file": self.training_file,
            "validation_file": self.validation_file,
            "result_files": list(self.result_files or []) or None,
            "metadata": dict(self.metadata or {}) if self.metadata is not None else None,
        }


@dataclass
class FineTuningJobsPage:
    """A page of fine-tuning jobs."""

    items: list[FineTuningJobResult]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "first_id": self.first_id,
            "last_id": self.last_id,
            "has_more": self.has_more,
        }


@dataclass
class FineTuningJobEventsPage:
    """A page of fine-tuning job events."""

    job_id: str
    events: list[dict[str, Any]]
    has_more: bool = False
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "events": [dict(item) for item in self.events],
            "has_more": self.has_more,
        }


@dataclass
class RealtimeClientSecretResult:
    """Result of creating a realtime client secret."""

    value: str
    expires_at: int | None = None
    session: dict[str, Any] | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return bool(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "expires_at": self.expires_at,
            "session": dict(self.session or {}) if self.session is not None else None,
        }


@dataclass
class RealtimeCallResult:
    """Realtime call/session creation or control result."""

    call_id: str | None = None
    sdp: str | None = None
    action: str | None = None
    status: int = 200
    error: str | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "sdp": self.sdp,
            "action": self.action,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class RealtimeTranscriptionSessionResult:
    """Result of creating a realtime transcription session."""

    value: str
    expires_at: int | None = None
    session: dict[str, Any] | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return bool(self.value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "expires_at": self.expires_at,
            "session": dict(self.session or {}) if self.session is not None else None,
        }


@dataclass
class RealtimeEventResult:
    """Normalized wrapper for server events received from a realtime connection."""

    event_type: str | None = None
    event_id: str | None = None
    item_id: str | None = None
    response_id: str | None = None
    sequence_number: int | None = None
    previous_item_id: str | None = None
    delta: str | None = None
    transcript: str | None = None
    status: str | None = None
    item: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    session: dict[str, Any] | None = None
    rate_limits: list[dict[str, Any]] | None = None
    details: dict[str, Any] = field(default_factory=dict)
    raw_event: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return bool(self.event_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "item_id": self.item_id,
            "response_id": self.response_id,
            "sequence_number": self.sequence_number,
            "previous_item_id": self.previous_item_id,
            "delta": self.delta,
            "transcript": self.transcript,
            "status": self.status,
            "item": dict(self.item or {}) if self.item is not None else None,
            "response": dict(self.response or {}) if self.response is not None else None,
            "session": dict(self.session or {}) if self.session is not None else None,
            "rate_limits": [dict(limit) for limit in self.rate_limits] if self.rate_limits is not None else None,
            "details": dict(self.details),
        }


@dataclass
class RealtimeResponseOutput:
    """Collected output from a realtime response event stream."""

    response_id: str | None = None
    text: str = ""
    transcript: str = ""
    audio: bytes = b""
    status: str | None = None
    item_ids: list[str] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    final_event: RealtimeEventResult | None = None
    raw_events: list[RealtimeEventResult] = field(default_factory=list, repr=False)

    @property
    def ok(self) -> bool:
        return bool(self.response_id or self.final_event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "text": self.text,
            "transcript": self.transcript,
            "audio_b64": base64.b64encode(self.audio).decode("ascii") if self.audio else None,
            "audio_bytes": len(self.audio),
            "status": self.status,
            "item_ids": list(self.item_ids),
            "event_types": list(self.event_types),
            "final_event": self.final_event.to_dict() if self.final_event is not None else None,
        }


class RealtimeConnection:
    """Stable wrapper around a provider realtime connection."""

    def __init__(
        self,
        connection: Any,
        *,
        model: str | None = None,
        call_id: str | None = None,
        close_callback: Callable[[], Awaitable[Any] | Any] | None = None,
        raw_manager: Any | None = None,
    ) -> None:
        self._connection = connection
        self._close_callback = close_callback
        self.model = model
        self.call_id = call_id
        self.raw_manager = raw_manager
        self._closed = False

    @property
    def ok(self) -> bool:
        return self._connection is not None and not self._closed

    @property
    def raw_connection(self) -> Any:
        return self._connection

    @staticmethod
    def _serialize_event(event: Any) -> Any:
        if hasattr(event, "to_dict"):
            return event.to_dict()
        if hasattr(event, "model_dump"):
            return event.model_dump()
        if hasattr(event, "dict"):
            return event.dict()
        return event

    @classmethod
    def _normalize_event(cls, event: Any) -> RealtimeEventResult:
        serialized = cls._serialize_event(event)
        if not isinstance(serialized, dict):
            return RealtimeEventResult(details={"value": serialized}, raw_event=event)

        reserved = {
            "type",
            "event_id",
            "item_id",
            "response_id",
            "sequence_number",
            "previous_item_id",
            "delta",
            "transcript",
            "status",
            "item",
            "response",
            "session",
            "rate_limits",
        }
        return RealtimeEventResult(
            event_type=str(serialized.get("type") or "") or None,
            event_id=str(serialized.get("event_id") or "") or None,
            item_id=str(serialized.get("item_id") or "") or None,
            response_id=str(serialized.get("response_id") or "") or None,
            sequence_number=int(serialized["sequence_number"]) if isinstance(serialized.get("sequence_number"), int) else None,
            previous_item_id=str(serialized.get("previous_item_id") or "") or None,
            delta=str(serialized.get("delta") or "") or None,
            transcript=str(serialized.get("transcript") or "") or None,
            status=str(serialized.get("status") or "") or None,
            item=dict(serialized["item"]) if isinstance(serialized.get("item"), dict) else None,
            response=dict(serialized["response"]) if isinstance(serialized.get("response"), dict) else None,
            session=dict(serialized["session"]) if isinstance(serialized.get("session"), dict) else None,
            rate_limits=[dict(limit) for limit in serialized.get("rate_limits", []) if isinstance(limit, dict)]
            if isinstance(serialized.get("rate_limits"), list)
            else None,
            details={key: value for key, value in serialized.items() if key not in reserved},
            raw_event=event,
        )

    async def send(self, event: Any) -> None:
        result = self._connection.send(event)
        if hasattr(result, "__await__"):
            await result

    async def update_session(self, session: dict[str, Any], *, event_id: str | None = None) -> None:
        payload: dict[str, Any] = {"type": "session.update", "session": dict(session)}
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def disable_vad(
        self,
        *,
        session: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> None:
        payload = dict(session or {})
        payload["turn_detection"] = None
        await self.update_session(payload, event_id=event_id)

    async def create_response(
        self,
        response: dict[str, Any] | None = None,
        *,
        event_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "response.create"}
        if response:
            payload["response"] = dict(response)
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def create_text_message(
        self,
        text: str,
        *,
        role: str = "user",
        previous_item_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        await self.create_conversation_item(
            {
                "type": "message",
                "role": str(role),
                "content": [{"type": "input_text", "text": str(text)}],
            },
            previous_item_id=previous_item_id,
            event_id=event_id,
        )

    async def create_conversation_item(
        self,
        item: dict[str, Any],
        *,
        previous_item_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "conversation.item.create",
            "item": dict(item),
        }
        if previous_item_id:
            payload["previous_item_id"] = previous_item_id
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def retrieve_conversation_item(
        self,
        item_id: str,
        *,
        event_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "conversation.item.retrieve",
            "item_id": item_id,
        }
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def delete_conversation_item(
        self,
        item_id: str,
        *,
        event_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "conversation.item.delete",
            "item_id": item_id,
        }
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def truncate_conversation_item(
        self,
        item_id: str,
        *,
        audio_end_ms: int,
        content_index: int = 0,
        event_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": content_index,
            "audio_end_ms": int(audio_end_ms),
        }
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def append_input_audio_chunks(
        self,
        audio_chunks: Sequence[bytes],
        *,
        event_ids: Sequence[str | None] | None = None,
    ) -> None:
        if event_ids is not None and len(event_ids) != len(audio_chunks):
            raise ValueError("`event_ids` must match the number of audio chunks.")
        for index, chunk in enumerate(audio_chunks):
            event_id = event_ids[index] if event_ids is not None else None
            await self.append_input_audio(chunk, event_id=event_id)

    async def append_input_audio(self, audio: bytes, *, event_id: str | None = None) -> None:
        payload: dict[str, Any] = {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode("ascii"),
        }
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def cancel_response(
        self,
        *,
        response_id: str | None = None,
        event_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "response.cancel"}
        if response_id:
            payload["response_id"] = response_id
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def commit_input_audio(self, *, event_id: str | None = None) -> None:
        payload: dict[str, Any] = {"type": "input_audio_buffer.commit"}
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def commit_audio_and_create_response(
        self,
        response: dict[str, Any] | None = None,
        *,
        commit_event_id: str | None = None,
        response_event_id: str | None = None,
    ) -> None:
        await self.commit_input_audio(event_id=commit_event_id)
        await self.create_response(response, event_id=response_event_id)

    async def send_audio_turn(
        self,
        audio_chunks: Sequence[bytes],
        response: dict[str, Any] | None = None,
        *,
        clear_input: bool = False,
        clear_output: bool = False,
        cancel_response_id: str | None = None,
        clear_input_event_id: str | None = None,
        clear_output_event_id: str | None = None,
        cancel_event_id: str | None = None,
        append_event_ids: Sequence[str | None] | None = None,
        commit_event_id: str | None = None,
        response_event_id: str | None = None,
    ) -> None:
        if clear_input:
            await self.clear_input_audio(event_id=clear_input_event_id)
        if cancel_response_id is not None:
            await self.cancel_response(response_id=cancel_response_id, event_id=cancel_event_id)
        if clear_output:
            await self.clear_output_audio(event_id=clear_output_event_id)
        await self.append_input_audio_chunks(audio_chunks, event_ids=append_event_ids)
        await self.commit_audio_and_create_response(
            response,
            commit_event_id=commit_event_id,
            response_event_id=response_event_id,
        )

    async def clear_input_audio(self, *, event_id: str | None = None) -> None:
        payload: dict[str, Any] = {"type": "input_audio_buffer.clear"}
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def clear_output_audio(self, *, event_id: str | None = None) -> None:
        payload: dict[str, Any] = {"type": "output_audio_buffer.clear"}
        if event_id:
            payload["event_id"] = event_id
        await self.send(payload)

    async def recv(self) -> Any:
        result = self._connection.recv()
        if hasattr(result, "__await__"):
            result = await result
        return self._serialize_event(result)

    async def recv_event(self) -> RealtimeEventResult:
        return self._normalize_event(await self.recv_raw())

    async def recv_until_type(
        self,
        event_types: str | Sequence[str],
        *,
        timeout: float | None = None,
    ) -> RealtimeEventResult:
        expected = {event_types} if isinstance(event_types, str) else {str(item) for item in event_types}

        async def _wait() -> RealtimeEventResult:
            while True:
                event = await self.recv_event()
                if str(event.event_type or "") in expected:
                    return event

        if timeout is None:
            return await _wait()
        return await asyncio.wait_for(_wait(), timeout=timeout)

    async def collect_response_output(
        self,
        *,
        response_id: str | None = None,
        timeout: float | None = None,
        decode_audio: bool = True,
    ) -> RealtimeResponseOutput:
        terminal_types = {"response.done", "response.cancelled", "response.failed"}

        async def _collect() -> RealtimeResponseOutput:
            collected = RealtimeResponseOutput(response_id=response_id)
            while True:
                event = await self.recv_event()
                event_type = str(event.event_type or "")
                event_response_id = event.response_id
                if response_id is not None and event_response_id != response_id:
                    continue

                collected.raw_events.append(event)
                if event_response_id and not collected.response_id:
                    collected.response_id = event_response_id
                if event.item_id and event.item_id not in collected.item_ids:
                    collected.item_ids.append(event.item_id)
                if event_type:
                    collected.event_types.append(event_type)

                if event_type == "response.output_text.delta" and event.delta:
                    collected.text += event.delta
                elif event_type == "response.output_audio_transcript.delta" and event.transcript:
                    collected.transcript += event.transcript
                elif event_type == "response.output_audio.delta" and event.delta and decode_audio:
                    try:
                        collected.audio += base64.b64decode(event.delta)
                    except Exception:
                        pass

                if event_type in terminal_types:
                    collected.status = event.status or event_type.removeprefix("response.")
                    collected.final_event = event
                    return collected

        if timeout is None:
            return await _collect()
        return await asyncio.wait_for(_collect(), timeout=timeout)

    async def recv_raw(self) -> Any:
        result = self._connection.recv()
        if hasattr(result, "__await__"):
            return await result
        return result

    async def recv_bytes(self) -> bytes:
        result = self._connection.recv_bytes()
        if hasattr(result, "__await__"):
            result = await result
        return bytes(result)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close_callback is not None:
            result = self._close_callback()
            if hasattr(result, "__await__"):
                await result
            return
        close_fn = getattr(self._connection, "close", None)
        if close_fn is None:
            return
        result = close_fn()
        if hasattr(result, "__await__"):
            await result

    async def __aenter__(self) -> RealtimeConnection:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb
        await self.close()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "call_id": self.call_id,
            "connected": self.ok,
        }


@dataclass
class WebhookEventResult:
    """Parsed and verified webhook event payload."""

    event_id: str | None = None
    event_type: str | None = None
    data: dict[str, Any] | None = None
    raw_event: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.event_type is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": dict(self.data or {}) if self.data is not None else None,
        }


@dataclass
class FileResource:
    """Provider-level representation of an uploaded file."""

    file_id: str
    filename: str | None = None
    purpose: str | None = None
    bytes: int | None = None
    status: str | None = None
    media_type: str | None = None
    created_at: int | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "purpose": self.purpose,
            "bytes": self.bytes,
            "status": self.status,
            "media_type": self.media_type,
            "created_at": self.created_at,
        }


@dataclass
class FilesPage:
    """A page of uploaded files."""

    items: list[FileResource]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "first_id": self.first_id,
            "last_id": self.last_id,
            "has_more": self.has_more,
        }


@dataclass
class FileContentResult:
    """Binary content returned for an uploaded file."""

    file_id: str
    content: bytes
    media_type: str | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    @property
    def byte_length(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "media_type": self.media_type,
            "byte_length": self.byte_length,
        }


@dataclass
class UploadResource:
    """Provider-level representation of an OpenAI upload lifecycle object."""

    upload_id: str
    status: str | None = None
    filename: str | None = None
    purpose: str | None = None
    bytes: int | None = None
    created_at: int | None = None
    expires_at: int | None = None
    file: FileResource | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status not in {"cancelled", "expired"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "status": self.status,
            "filename": self.filename,
            "purpose": self.purpose,
            "bytes": self.bytes,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "file": self.file.to_dict() if self.file is not None else None,
        }


@dataclass
class UploadPartResource:
    """Provider-level representation of an upload part."""

    part_id: str
    upload_id: str
    created_at: int | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return bool(self.part_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "upload_id": self.upload_id,
            "created_at": self.created_at,
        }


@dataclass
class VectorStoreFileResource:
    """Provider-level representation of a vector-store file."""

    file_id: str
    vector_store_id: str
    status: str | None = None
    attributes: dict[str, Any] | None = None
    usage_bytes: int | None = None
    chunking_strategy: dict[str, Any] | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "vector_store_id": self.vector_store_id,
            "status": self.status,
            "attributes": dict(self.attributes or {}) if self.attributes is not None else None,
            "usage_bytes": self.usage_bytes,
            "chunking_strategy": dict(self.chunking_strategy or {}) if self.chunking_strategy is not None else None,
        }


@dataclass
class VectorStoreFilesPage:
    """A page of vector-store files."""

    items: list[VectorStoreFileResource]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "first_id": self.first_id,
            "last_id": self.last_id,
            "has_more": self.has_more,
        }


@dataclass
class VectorStoreFileContentResult:
    """Content chunks returned for a vector-store file."""

    file_id: str
    vector_store_id: str
    chunks: list[dict[str, Any]]
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "vector_store_id": self.vector_store_id,
            "chunks": [dict(item) for item in self.chunks],
        }


@dataclass
class VectorStoreFileBatchResource:
    """Provider-level representation of a vector-store file batch."""

    batch_id: str
    vector_store_id: str
    status: str | None = None
    file_counts: dict[str, Any] | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "vector_store_id": self.vector_store_id,
            "status": self.status,
            "file_counts": dict(self.file_counts or {}) if self.file_counts is not None else None,
        }


@dataclass
class EmbeddingResult:
    """Result of an embedding request."""

    embeddings: list[list[float]]
    usage: Usage | None = None
    model: str | None = None

    status: int = 200
    error: str | None = None
    raw_response: Any | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def embedding(self) -> list[float] | None:
        """Get single embedding (for single-input requests)."""
        return self.embeddings[0] if self.embeddings else None


# Type aliases for convenience
MessageInput = str | dict[str, Any] | Message | Sequence[str | dict[str, Any] | Message]


def normalize_messages(messages: MessageInput) -> list[Message]:
    """
    Normalize various message input formats to a list of Message objects.

    Accepts:
    - str: Converted to single user message
    - dict: Converted using Message.from_dict
    - Message: Used as-is
    - List of the above
    """
    if isinstance(messages, str):
        return [Message.user(messages)]

    if isinstance(messages, Message):
        return [messages]

    if isinstance(messages, dict):
        return [Message.from_dict(messages)]

    if isinstance(messages, list):
        result = []
        for msg in messages:
            if isinstance(msg, str):
                result.append(Message.user(msg))
            elif isinstance(msg, Message):
                result.append(msg)
            elif isinstance(msg, dict):
                result.append(Message.from_dict(msg))
            else:
                raise TypeError(f"Unsupported message type: {type(msg)}")
        return result

    raise TypeError(f"Unsupported messages type: {type(messages)}")


__all__ = [
    "Role",
    "StreamEventType",
    "ToolCall",
    "ToolCallDelta",
    "Message",
    "Usage",
    "StreamEvent",
    "NormalizedOutputItem",
    "CompletionResult",
    "BackgroundResponseResult",
    "DeepResearchRunResult",
    "ConversationResource",
    "CompactionResult",
    "DeletionResult",
    "ConversationItemResource",
    "ConversationItemsPage",
    "ModerationResult",
    "GeneratedImage",
    "ImageGenerationResult",
    "AudioTranscriptionResult",
    "AudioSpeechResult",
    "VectorStoreResource",
    "VectorStoresPage",
    "VectorStoreSearchResult",
    "FineTuningJobResult",
    "FineTuningJobsPage",
    "FineTuningJobEventsPage",
    "RealtimeClientSecretResult",
    "RealtimeCallResult",
    "RealtimeTranscriptionSessionResult",
    "RealtimeEventResult",
    "RealtimeResponseOutput",
    "RealtimeConnection",
    "WebhookEventResult",
    "FileResource",
    "FilesPage",
    "FileContentResult",
    "VectorStoreFileResource",
    "VectorStoreFilesPage",
    "VectorStoreFileContentResult",
    "VectorStoreFileBatchResource",
    "EmbeddingResult",
    "MessageInput",
    "normalize_messages",
]
