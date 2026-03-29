"""
Stable shared type namespace for llm-client.

Use this module when you want a single import path for the package's common
request, response, streaming, and cancellation datatypes.
"""

from .cancellation import CancellationToken, CancelledError
from .providers.types import (
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
from .spec import RequestContext, RequestSpec

__all__ = [
    "Role",
    "Message",
    "MessageInput",
    "ToolCall",
    "ToolCallDelta",
    "Usage",
    "StreamEventType",
    "StreamEvent",
    "CompletionResult",
    "EmbeddingResult",
    "RequestContext",
    "RequestSpec",
    "CancellationToken",
    "CancelledError",
    "normalize_messages",
]
