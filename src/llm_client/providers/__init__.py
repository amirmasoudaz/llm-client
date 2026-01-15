"""
Provider abstraction layer.

This module provides a unified interface for interacting with different LLM providers.
"""
from .base import BaseProvider, Provider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider, ANTHROPIC_AVAILABLE
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
    "AnthropicProvider",
    "ANTHROPIC_AVAILABLE",
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

