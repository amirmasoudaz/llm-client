"""
Provider abstraction layer.

This module provides a unified interface for interacting with different LLM providers.
"""

from .anthropic import ANTHROPIC_AVAILABLE, AnthropicProvider
from .base import BaseProvider, Provider
from .google import GOOGLE_AVAILABLE, GoogleProvider
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
    "AnthropicProvider",
    "ANTHROPIC_AVAILABLE",
    "GoogleProvider",
    "GOOGLE_AVAILABLE",
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
