"""
Provider abstraction layer.

This module provides a unified interface for interacting with different LLM providers.
"""
from .base import BaseProvider, Provider
from .openai import OpenAIProvider
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
    normalize_messages,
)

__all__ = [
    # Protocols and base classes
    "Provider",
    "BaseProvider",
    # Provider implementations
    "OpenAIProvider",
    # Types
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

