"""
Logging configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import LogFormat, LogLevel


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: LogLevel = "INFO"
    format: LogFormat = "text"

    # Output settings
    log_file: Path | None = None
    include_timestamp: bool = True
    include_trace_id: bool = True

    # What to log
    log_requests: bool = True
    log_responses: bool = True
    log_tool_calls: bool = True
    log_usage: bool = True

    # Redaction
    redact_api_keys: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if self.level not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        valid_formats = ("text", "json")
        if self.format not in valid_formats:
            raise ValueError(f"Invalid log format: {self.format}. Must be one of {valid_formats}")
        if self.log_file and isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)


@dataclass
class MetricsConfig:
    """Configuration for metrics."""

    enabled: bool = False
    provider: str = "none"  # "none", "prometheus", "otel"

    # Prometheus settings
    prometheus_port: int = 8000

    # OpenTelemetry settings
    otel_endpoint: str | None = None
    otel_service_name: str = "llm-client"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.prometheus_port < 1 or self.prometheus_port > 65535:
            raise ValueError("prometheus_port must be between 1 and 65535")
        if self.provider not in ("none", "prometheus", "otel"):
            raise ValueError(f"Invalid metrics provider: {self.provider}")
        if self.otel_endpoint and not self.otel_endpoint.startswith(("http://", "https://")):
            raise ValueError("otel_endpoint must be a valid HTTP(S) URL")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    enabled: bool = True

    # Token bucket settings
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000

    # Behavior
    wait_on_limit: bool = True
    max_wait_seconds: float = 60.0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.tokens_per_minute <= 0:
            raise ValueError("tokens_per_minute must be positive")
        if self.max_wait_seconds < 0:
            raise ValueError("max_wait_seconds cannot be negative")


__all__ = ["LoggingConfig", "MetricsConfig", "RateLimitConfig"]
