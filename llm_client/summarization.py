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

from .content import ContentResponseEnvelope, content_blocks_to_text
from .request_builders import build_content_request_envelope, build_request_spec, infer_model_name

if TYPE_CHECKING:
    from .engine import ExecutionEngine
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


@dataclass(frozen=True)
class SummarizationRequest:
    """Request envelope for summarization strategies."""

    messages: list[Message]
    max_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SummarizationResult:
    """Structured summarization response."""

    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SummarizationStrategy(Protocol):
    """Strategy interface for conversation summarization."""

    async def summarize_request(self, request: SummarizationRequest) -> SummarizationResult:
        """Summarize a request envelope into a structured result."""
        ...


class NoOpSummarizer:
    """Default summarizer for testing and fallback.
    
    Returns a placeholder summary without making any LLM calls.
    """
    
    async def summarize(self, messages: list[Message], max_tokens: int) -> str:
        return f"[Summary of {len(messages)} earlier messages]"

    async def summarize_request(self, request: SummarizationRequest) -> SummarizationResult:
        return SummarizationResult(
            summary=await self.summarize(request.messages, request.max_tokens),
            metadata={"strategy": "noop"},
        )


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
        provider: Provider | None = None,
        *,
        engine: ExecutionEngine | None = None,
        config: LLMSummarizerConfig | None = None,
    ):
        """Initialize the summarizer.
        
        Args:
            provider: LLM provider to use for summarization.
            engine: Optional execution engine. When provided, summarization
                will prefer engine-backed execution.
            config: Optional configuration. Uses defaults if not provided.
        """
        if provider is None and engine is None:
            raise ValueError("LLMSummarizer requires a provider or an engine.")
        self.provider = provider
        if engine is not None:
            self.engine = engine
        elif provider is not None:
            from .engine import ExecutionEngine

            self.engine = ExecutionEngine(provider=provider)
        else:
            self.engine = None
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
        prompt_messages = [
            Msg.system(
                "You summarize prior conversation context for continuity. "
                "Return concise plain text only. Preserve key facts, constraints, "
                "decisions, open questions, and any user preferences."
            ),
            Msg.user(prompt),
        ]

        try:
            if self.engine is None:
                raise RuntimeError("LLMSummarizer engine is not configured")
            provider_ref = self.provider or getattr(self.engine, "provider", None)
            request_kwargs = {
                "max_tokens": effective_tokens,
                "temperature": self.config.temperature,
            }
            model_name = self.config.model_override or infer_model_name(provider_ref)
            summary_text = ""
            if hasattr(self.engine, "complete"):
                try:
                    direct_result = await self.engine.complete(
                        build_request_spec(
                            engine=self.engine,
                            provider=provider_ref,
                            messages=prompt_messages,
                            request_kwargs=request_kwargs,
                            model=model_name,
                        )
                    )
                    summary_text = _extract_summary_text(direct_result)
                except Exception:
                    summary_text = ""

            if not summary_text and hasattr(self.engine, "complete_content"):
                try:
                    result = await self.engine.complete_content(
                        build_content_request_envelope(
                            engine=self.engine,
                            provider=provider_ref,
                            messages=prompt_messages,
                            request_kwargs=request_kwargs,
                            model=model_name,
                        )
                    )
                    if isinstance(result, ContentResponseEnvelope):
                        summary_text = content_blocks_to_text(result.message.blocks)
                        if not summary_text:
                            summary_text = _extract_summary_text(result.to_completion_result())
                    else:
                        summary_text = _extract_summary_text(result)
                except Exception:
                    summary_text = ""

            if summary_text:
                return summary_text
            return _heuristic_summary_fallback(messages)
        except Exception:
            # Graceful degradation
            return _heuristic_summary_fallback(messages)

    async def summarize_request(self, request: SummarizationRequest) -> SummarizationResult:
        return SummarizationResult(
            summary=await self.summarize(request.messages, request.max_tokens),
            metadata=dict(request.metadata),
        )


def _extract_summary_text(result: Any) -> str:
    if result is None:
        return ""
    content = getattr(result, "content", None)
    reasoning = getattr(result, "reasoning", None)
    for value in (content, reasoning):
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            text = content_blocks_to_text(value).strip()
            if text:
                return text
    return ""


def _heuristic_summary_fallback(messages: list[Message]) -> str:
    user_points = [
        _clean_summary_item(message.content)
        for message in messages
        if _role_value(message) == "user"
    ]
    assistant_points = [
        _clean_summary_item(message.content)
        for message in messages
        if _role_value(message) == "assistant"
    ]
    user_points = [item for item in user_points if item]
    assistant_points = [item for item in assistant_points if item]

    lines: list[str] = []
    if user_points:
        lines.append("Earlier user context: " + "; ".join(user_points[-3:]))
    if assistant_points:
        lines.append("Earlier assistant guidance: " + "; ".join(assistant_points[-2:]))
    if lines:
        return "\n".join(lines)
    return "[No content to summarize]"


def _clean_summary_item(content: Any, *, max_len: int = 140) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        text = content_blocks_to_text(content)
    else:
        text = str(content)
    compact = " ".join(text.split()).strip()
    if not compact:
        return ""
    if len(compact) > max_len:
        return compact[: max_len - 1].rstrip() + "…"
    return compact


def _role_value(message: Message) -> str:
    role = getattr(message, "role", "")
    return getattr(role, "value", role) or ""
__all__ = [
    "Summarizer",
    "SummarizationRequest",
    "SummarizationResult",
    "SummarizationStrategy",
    "NoOpSummarizer", 
    "LLMSummarizer",
    "LLMSummarizerConfig",
]
