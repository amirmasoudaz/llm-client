"""Thin sync wrappers for async-first APIs.

This module provides synchronous wrappers for the async-first APIs
in llm-client. Use with caution - these are primarily for scripting
and testing contexts where async is not available.

Key design:
- Uses asyncio.run() when no event loop is active
- Raises RuntimeError if called inside an existing event loop
- Clear error messages guide users to the async alternative
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation import Conversation
    from .models import ModelProfile
    from .providers.types import Message


def get_messages_sync(
    conversation: Conversation,
    model: type[ModelProfile] | None = None,
    include_system: bool = True,
) -> list[Message]:
    """Sync wrapper for Conversation.get_messages_async.
    
    This is a convenience wrapper for contexts where async is not available.
    Prefer using the async version when possible.
    
    Args:
        conversation: The Conversation instance.
        model: Optional model profile for token counting.
        include_system: Whether to include system message.
    
    Returns:
        List of messages, potentially truncated/summarized.
    
    Raises:
        RuntimeError: If called inside an existing async event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None:
        raise RuntimeError(
            "get_messages_sync() cannot be called inside an async context. "
            "Use 'await conversation.get_messages_async()' instead."
        )
    
    return asyncio.run(
        conversation.get_messages_async(model=model, include_system=include_system)
    )


def summarize_sync(
    summarizer: any,  # Summarizer protocol
    messages: list[Message],
    max_tokens: int,
) -> str:
    """Sync wrapper for Summarizer.summarize.
    
    Args:
        summarizer: A Summarizer implementation.
        messages: Messages to summarize.
        max_tokens: Maximum tokens for summary.
    
    Returns:
        Summary string.
    
    Raises:
        RuntimeError: If called inside an existing async event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None:
        raise RuntimeError(
            "summarize_sync() cannot be called inside an async context. "
            "Use 'await summarizer.summarize()' instead."
        )
    
    return asyncio.run(summarizer.summarize(messages, max_tokens))


__all__ = ["get_messages_sync", "summarize_sync"]
