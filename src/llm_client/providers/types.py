"""
Core types for the provider abstraction layer.

These types provide a unified interface for LLM responses, streaming events,
and messages across different providers.
"""

from __future__ import annotations

import time, json
from collections.abc import Sequence
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
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool response messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to API-compatible dictionary format."""
        d: dict[str, Any] = {"role": self.role.value}

        if self.content is not None:
            d["content"] = self.content
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
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> Message:
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, name: str | None = None) -> Message:
        """Create a tool result message."""
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class Usage:
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_cached: int = 0

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


@dataclass
class CompletionResult:
    """
    Result of a completion request.

    This unified result type works for both streaming and non-streaming completions.
    """

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    reasoning: str | None = None

    # Request metadata
    model: str | None = None
    finish_reason: str | None = None

    # Status tracking
    status: int = 200
    error: str | None = None

    # Original response for debugging
    raw_response: Any | None = field(default=None, repr=False)

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
        return Message.assistant(content=self.content, tool_calls=self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in (self.tool_calls or [])],
            "usage": self.usage.to_dict() if self.usage else None,
            "reasoning": self.reasoning,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class EmbeddingResult:
    """Result of an embedding request."""

    embeddings: list[list[float]]
    usage: Usage | None = None
    model: str | None = None

    status: int = 200
    error: str | None = None

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
    "CompletionResult",
    "EmbeddingResult",
    "MessageInput",
    "normalize_messages",
]
