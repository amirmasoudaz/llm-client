"""
Provider configuration classes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    # API settings
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None

    # Request settings
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 1.0

    # Model defaults
    default_model: str | None = None
    default_temperature: float = 0.7
    default_max_tokens: int | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if self.retry_backoff < 0:
            raise ValueError("retry_backoff cannot be negative")


@dataclass
class OpenAIConfig(ProviderConfig):
    """OpenAI-specific configuration."""

    api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = "gpt-4o"

    # OpenAI-specific settings
    use_responses_api: bool = False


@dataclass
class AnthropicConfig(ProviderConfig):
    """Anthropic-specific configuration."""

    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    default_model: str = "claude-sonnet-4-20250514"

    # Anthropic-specific settings
    max_thinking_tokens: int | None = None


@dataclass
class GoogleConfig(ProviderConfig):
    """Google-specific configuration."""

    api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    default_model: str = "gemini-2.0-flash"


__all__ = ["ProviderConfig", "OpenAIConfig", "AnthropicConfig", "GoogleConfig"]
