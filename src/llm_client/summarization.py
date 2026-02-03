"""Pluggable conversation summarization interface.

This module provides a protocol for conversation summarizers and a
production-ready LLM-based implementation.

Key design:
- Summarizer is a protocol (duck typing) for easy testing/swapping
- LLMSummarizer is policy-neutral and fully configurable
- Core is async-only to reflect I/O-bound nature
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .providers.types import Message
    from .providers.base import Provider


class Summarizer(Protocol):
    """Protocol for conversation summarizers.
    
    Implementations should condense a list of messages into a summary
    string that preserves key facts and context for conversation continuity.
    """
    
    async def summarize(self, messages: list[Message], max_tokens: int) -> str:
        """Summarize messages into a condensed form.
        
        Args:
            messages: List of messages to summarize.
            max_tokens: Maximum tokens for the summary output.
        
        Returns:
            A summary string.
        """
        ...


class NoOpSummarizer:
    """Default summarizer for testing and fallback.
    
    Returns a placeholder summary without making any LLM calls.
    """
    
    async def summarize(self, messages: list[Message], max_tokens: int) -> str:
        return f"[Summary of {len(messages)} earlier messages]"


@dataclass
class LLMSummarizerConfig:
    """Configuration for LLM-based summarization.
    
    All parameters are exposed for external configuration to maintain
    policy neutrality in the kernel.
    """
    prompt_template: str = (
        "Summarize the following conversation concisely, preserving key facts, "
        "decisions, and context needed for continuity:\n\n{conversation}"
    )
    max_summary_tokens: int = 500
    temperature: float = 0.3
    model_override: str | None = None  # Use different model for summarization


class LLMSummarizer:
    """Policy-neutral LLM summarizer. Fully configurable from outside.
    
    Uses a Provider to generate summaries via LLM completion.
    Does not embed any tenant-specific logic or policies.
    """
    
    def __init__(
        self, 
        provider: Provider, 
        config: LLMSummarizerConfig | None = None,
    ):
        """Initialize the summarizer.
        
        Args:
            provider: LLM provider to use for summarization.
            config: Optional configuration. Uses defaults if not provided.
        """
        self.provider = provider
        self.config = config or LLMSummarizerConfig()
    
    async def summarize(self, messages: list[Message], max_tokens: int) -> str:
        """Summarize messages using LLM.
        
        Args:
            messages: List of messages to summarize.
            max_tokens: Maximum tokens for the summary output.
        
        Returns:
            LLM-generated summary string.
        """
        from .providers.types import Message as Msg
        
        # Format messages for summarization
        formatted = "\n".join(
            f"{m.role.value}: {m.content}" for m in messages if m.content
        )
        
        if not formatted.strip():
            return "[No content to summarize]"
        
        prompt = self.config.prompt_template.format(conversation=formatted)
        effective_tokens = min(max_tokens, self.config.max_summary_tokens)
        
        try:
            result = await self.provider.complete(
                [Msg.user(prompt)],
                max_tokens=effective_tokens,
                temperature=self.config.temperature,
            )
            return result.content or "[Unable to summarize]"
        except Exception as e:
            # Graceful degradation
            return f"[Summarization failed: {len(messages)} messages]"


__all__ = [
    "Summarizer",
    "NoOpSummarizer", 
    "LLMSummarizer",
    "LLMSummarizerConfig",
]
