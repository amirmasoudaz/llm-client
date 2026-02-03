"""
Conversation management with context window handling.

This module provides:
- Conversation class for tracking message history
- Context window management with truncation strategies
- Session persistence and serialization
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from .providers.types import Message, Role, ToolCall
from .validation import validate_message, validate_or_raise

if TYPE_CHECKING:
    from .models import ModelProfile


TruncationStrategy = Literal["sliding", "drop_oldest", "drop_middle", "summarize"]


@dataclass
class ConversationConfig:
    """Configuration for conversation behavior."""

    # Context window management
    max_tokens: int | None = None
    truncation_strategy: TruncationStrategy = "sliding"
    reserve_tokens: int = 1000  # Reserve for model response

    # System message handling
    system_message: str | None = None
    preserve_system: bool = True  # Never truncate system message

    # Session settings
    session_id: str | None = None
    
    # Summarization (optional - for "summarize" strategy)
    summarizer: Any = None  # Summarizer protocol - Any to avoid import issues

    def __post_init__(self) -> None:
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())


class Conversation:
    """
    Manages a conversation with message history and context window handling.

    Features:
    - Automatic context window management
    - Multiple truncation strategies
    - Message history tracking
    - Session serialization
    - Conversation forking

    Example:
        ```python
        conv = Conversation(
            system_message="You are a helpful assistant.",
            max_tokens=8000,
        )

        conv.add_user("Hello!")
        conv.add_assistant("Hi there! How can I help?")
        conv.add_user("What's the weather?")

        # Get messages for API call
        messages = conv.get_messages()

        # After model response with tool calls
        conv.add_assistant_with_tools(
            content=None,
            tool_calls=[ToolCall(id="1", name="get_weather", arguments='{"city": "NYC"}')]
        )
        conv.add_tool_result("1", "Sunny, 72°F")
        ```
    """

    def __init__(
        self,
        messages: list[Message] | None = None,
        *,
        system_message: str | None = None,
        max_tokens: int | None = None,
        truncation_strategy: TruncationStrategy = "sliding",
        reserve_tokens: int = 1000,
        session_id: str | None = None,
        summarizer: Any = None,
    ) -> None:
        """
        Initialize conversation.

        Args:
            messages: Initial messages (excluding system)
            system_message: System instruction message
            max_tokens: Maximum context window size
            truncation_strategy: How to handle context overflow
            reserve_tokens: Tokens to reserve for response
            session_id: Unique session identifier
            summarizer: Optional Summarizer for "summarize" truncation strategy
        """
        self.config = ConversationConfig(
            max_tokens=max_tokens,
            truncation_strategy=truncation_strategy,
            reserve_tokens=reserve_tokens,
            system_message=system_message,
            session_id=session_id,
            summarizer=summarizer,
        )

        self._messages: list[Message] = []
        self._metadata: dict[str, Any] = {
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        # Add initial messages
        if messages:
            for msg in messages:
                self._messages.append(msg)

    @property
    def session_id(self) -> str:
        """Get session identifier."""
        return self.config.session_id or ""

    @property
    def system_message(self) -> str | None:
        """Get current system message."""
        return self.config.system_message

    @system_message.setter
    def system_message(self, content: str | None) -> None:
        """Set system message."""
        self.config.system_message = content
        self._metadata["updated_at"] = time.time()

    def __len__(self) -> int:
        """Get number of messages (excluding system)."""
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        """Iterate over messages."""
        return iter(self._messages)

    def __getitem__(self, index: int) -> Message:
        """Get message by index."""
        return self._messages[index]

    # === Message Addition ===

    def add_message(self, message: Message) -> Conversation:
        """
        Add a message to the conversation.

        Args:
            message: Message to add

        Returns:
            Self for chaining
        """
        validate_or_raise(validate_message(message))
        self._messages.append(message)
        self._metadata["updated_at"] = time.time()
        return self

    def add_user(self, content: str) -> Conversation:
        """Add a user message."""
        return self.add_message(Message.user(content))

    def add_assistant(self, content: str) -> Conversation:
        """Add an assistant message."""
        return self.add_message(Message.assistant(content))

    def add_assistant_with_tools(
        self,
        content: str | None,
        tool_calls: list[ToolCall],
    ) -> Conversation:
        """Add an assistant message with tool calls."""
        return self.add_message(Message.assistant(content or "", tool_calls=tool_calls))

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str,
        name: str | None = None,
    ) -> Conversation:
        """Add a tool result message."""
        return self.add_message(Message.tool_result(tool_call_id, content, name))

    def add_system(self, content: str) -> Conversation:
        """
        Set the system message.

        Note: This replaces any existing system message.
        """
        self.config.system_message = content
        self._metadata["updated_at"] = time.time()
        return self

    # === Message Retrieval ===

    def get_messages(
        self,
        model: type[ModelProfile] | None = None,
        include_system: bool = True,
    ) -> list[Message]:
        """
        Get messages for API calls, with optional truncation.

        Args:
            model: Model profile for token counting (enables truncation)
            include_system: Whether to include system message

        Returns:
            List of messages ready for API call
        """
        messages = list(self._messages)

        # Apply truncation if model provided
        if model and self.config.max_tokens:
            messages = self._truncate(
                messages,
                model,
                self.config.max_tokens - self.config.reserve_tokens,
            )

        # Prepend system message
        if include_system and self.config.system_message:
            messages = [Message.system(self.config.system_message)] + messages

        return messages

    async def get_messages_async(
        self,
        model: type[ModelProfile] | None = None,
        include_system: bool = True,
    ) -> list[Message]:
        """
        Async version of get_messages that supports summarization.

        Use this method when using truncation_strategy="summarize" to
        allow the summarizer to make async LLM calls.

        Args:
            model: Model profile for token counting (enables truncation)
            include_system: Whether to include system message

        Returns:
            List of messages ready for API call
        """
        messages = list(self._messages)

        # Apply truncation if model provided
        if model and self.config.max_tokens:
            target_tokens = self.config.max_tokens - self.config.reserve_tokens
            
            if self.config.truncation_strategy == "summarize" and self.config.summarizer:
                messages = await self._truncate_with_summary(messages, model, target_tokens)
            else:
                messages = self._truncate(messages, model, target_tokens)

        # Prepend system message
        if include_system and self.config.system_message:
            messages = [Message.system(self.config.system_message)] + messages

        return messages

    async def _truncate_with_summary(
        self,
        messages: list[Message],
        model: type[ModelProfile],
        max_tokens: int,
    ) -> list[Message]:
        """Truncate by summarizing old messages.
        
        Keeps recent messages and summarizes older ones.
        """
        if not messages:
            return messages
        
        # Count function for messages
        def count_tokens(msgs: list[Message]) -> int:
            total = 0
            for m in msgs:
                if m.content:
                    total += model.count_tokens(m.content)
                if m.tool_calls:
                    for tc in m.tool_calls:
                        total += model.count_tokens(tc.name)
                        total += model.count_tokens(tc.arguments)
            return total
        
        # If already fits, no need to truncate
        if count_tokens(messages) <= max_tokens:
            return messages
        
        # Find split point: keep recent messages that fit in 70% of budget
        recent_budget = int(max_tokens * 0.7)
        recent: list[Message] = []
        
        for msg in reversed(messages):
            test = [msg] + recent
            if count_tokens(test) <= recent_budget:
                recent = test
            else:
                break
        
        old = messages[:len(messages) - len(recent)]
        
        if not old:
            # Everything fits or we must keep all recent
            return recent
        
        # Summarize old messages
        summary_tokens = int(max_tokens * 0.2)
        summary = await self.config.summarizer.summarize(old, summary_tokens)
        
        # Return summary + recent
        summary_msg = Message.system(f"[Earlier context]: {summary}")
        return [summary_msg] + recent

    def get_messages_dict(
        self,
        model: type[ModelProfile] | None = None,
        include_system: bool = True,
    ) -> list[dict[str, Any]]:
        """Get messages as list of dicts for API calls."""
        return [m.to_dict() for m in self.get_messages(model, include_system)]

    def get_last_message(self) -> Message | None:
        """Get the most recent message."""
        return self._messages[-1] if self._messages else None

    def get_last_user_message(self) -> Message | None:
        """Get the most recent user message."""
        for msg in reversed(self._messages):
            if msg.role == Role.USER:
                return msg
        return None

    def get_last_assistant_message(self) -> Message | None:
        """Get the most recent assistant message."""
        for msg in reversed(self._messages):
            if msg.role == Role.ASSISTANT:
                return msg
        return None

    # === Token Counting ===

    def count_tokens(self, model: type[ModelProfile]) -> int:
        """
        Count total tokens in the conversation.

        Args:
            model: Model profile for tokenization

        Returns:
            Total token count
        """
        total = 0

        if self.config.system_message:
            total += model.count_tokens(self.config.system_message)

        for msg in self._messages:
            if msg.content:
                total += model.count_tokens(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total += model.count_tokens(tc.name)
                    total += model.count_tokens(tc.arguments)

        return total

    # === Truncation ===

    def _truncate(
        self,
        messages: list[Message],
        model: type[ModelProfile],
        max_tokens: int,
    ) -> list[Message]:
        """Apply a truncation strategy to fit within the token limit."""
        strategy = self.config.truncation_strategy

        # Count current tokens
        def count(msgs: list[Message]) -> int:
            total = 0
            if self.config.system_message:
                total += model.count_tokens(self.config.system_message)
            for m in msgs:
                if m.content:
                    total += model.count_tokens(m.content)
                if m.tool_calls:
                    for tc in m.tool_calls:
                        total += model.count_tokens(tc.name)
                        total += model.count_tokens(tc.arguments)
            return total

        if count(messages) <= max_tokens:
            return messages

        if strategy == "sliding":
            return self._truncate_sliding(messages, model, max_tokens, count)
        elif strategy == "drop_oldest":
            return self._truncate_drop_oldest(messages, model, max_tokens, count)
        elif strategy == "drop_middle":
            return self._truncate_drop_middle(messages, model, max_tokens, count)
        else:
            # Default to sliding
            return self._truncate_sliding(messages, model, max_tokens, count)

    @staticmethod
    def _truncate_sliding(
        messages: list[Message],
        model: type[ModelProfile],
        max_tokens: int,
        count_fn,
    ) -> list[Message]:
        """Keep most recent messages that fit."""
        result: list[Message] = []

        # Start from most recent and work backwards
        for msg in reversed(messages):
            candidate = [msg] + result
            if count_fn(candidate) <= max_tokens:
                result = candidate
            else:
                break

        return result

    @staticmethod
    def _truncate_drop_oldest(
        messages: list[Message],
        model: type[ModelProfile],
        max_tokens: int,
        count_fn,
    ) -> list[Message]:
        """Drop the oldest messages until within limit."""
        result = list(messages)

        while result and count_fn(result) > max_tokens:
            result.pop(0)

        return result

    @staticmethod
    def _truncate_drop_middle(
        messages: list[Message],
        model: type[ModelProfile],
        max_tokens: int,
        count_fn,
    ) -> list[Message]:
        """
        Keep first and last messages, drop middle.

        Preserves initial context and recent history.
        """
        if len(messages) <= 2:
            return messages

        # Always keep first message
        first = [messages[0]]

        # Find how many recent messages we can keep
        recent: list[Message] = []
        for msg in reversed(messages[1:]):
            candidate = first + recent + [msg]
            if count_fn(candidate) <= max_tokens:
                recent = [msg] + recent
            else:
                break

        return first + recent

    # === Conversation Management ===

    def clear(self) -> Conversation:
        """Clear all messages (preserves system message)."""
        self._messages.clear()
        self._metadata["updated_at"] = time.time()
        return self

    def fork(self) -> Conversation:
        """
        Create a copy of this conversation.

        The fork has the same messages but a new session ID.
        """
        new_conv = Conversation(
            messages=[deepcopy(m) for m in self._messages],
            system_message=self.config.system_message,
            max_tokens=self.config.max_tokens,
            truncation_strategy=self.config.truncation_strategy,
            reserve_tokens=self.config.reserve_tokens,
            session_id=str(uuid.uuid4()),  # New session ID
        )
        return new_conv

    def branch(self, from_index: int = 0) -> Conversation:
        """
        Create a branch from a specific point.

        Args:
            from_index: Message index to branch from (inclusive)
        """
        messages = self._messages[:from_index]
        return Conversation(
            messages=[deepcopy(m) for m in messages],
            system_message=self.config.system_message,
            max_tokens=self.config.max_tokens,
            truncation_strategy=self.config.truncation_strategy,
            reserve_tokens=self.config.reserve_tokens,
            session_id=str(uuid.uuid4()),
        )

    # === Serialization ===

    def to_dict(self) -> dict[str, Any]:
        """Serialize conversation to dictionary."""
        return {
            "session_id": self.session_id,
            "system_message": self.config.system_message,
            "messages": [m.to_dict() for m in self._messages],
            "config": {
                "max_tokens": self.config.max_tokens,
                "truncation_strategy": self.config.truncation_strategy,
                "reserve_tokens": self.config.reserve_tokens,
            },
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        """Deserialize conversation from dictionary."""
        messages = [Message.from_dict(m) for m in data.get("messages", [])]
        config = data.get("config", {})

        conv = cls(
            messages=messages,
            system_message=data.get("system_message"),
            max_tokens=config.get("max_tokens"),
            truncation_strategy=config.get("truncation_strategy", "sliding"),
            reserve_tokens=config.get("reserve_tokens", 1000),
            session_id=data.get("session_id"),
        )
        conv._metadata = data.get("metadata", conv._metadata)

        return conv

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Conversation:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, path: str | Path) -> None:
        """Save conversation to file."""
        path = Path(path)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> Conversation:
        """Load conversation from a file."""
        path = Path(path)
        return cls.from_json(path.read_text())

    # === String Representation ===

    def __repr__(self) -> str:
        return (
            f"Conversation(session_id={self.session_id!r}, "
            f"messages={len(self._messages)}, "
            f"system={'set' if self.config.system_message else 'unset'})"
        )

    def format_history(self, max_messages: int | None = None) -> str:
        """
        Format conversation history as a readable string.

        Args:
            max_messages: Limit the number of messages to show
        """
        lines = []

        if self.config.system_message:
            lines.append(f"[System] {self.config.system_message}\n")

        messages = self._messages
        if max_messages:
            messages = messages[-max_messages:]

        for msg in messages:
            role = msg.role.value.capitalize()

            if msg.tool_calls:
                tool_names = ", ".join(tc.name for tc in msg.tool_calls)
                lines.append(f"[{role}] (tools: {tool_names})")
                if msg.content:
                    lines.append(f"  {msg.content}")
            elif msg.tool_call_id:
                lines.append(f"[Tool:{msg.name or msg.tool_call_id}] {msg.content}")
            else:
                lines.append(f"[{role}] {msg.content}")

        return "\n".join(lines)


__all__ = [
    "Conversation",
    "ConversationConfig",
    "TruncationStrategy",
]
