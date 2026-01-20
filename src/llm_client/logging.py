"""
Structured Logging for LLM Client.

This module provides:
- Structured JSON logging with consistent fields
- Request/response logging with trace correlation
- Performance metrics logging
- Log level filtering and formatting options
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, Union

# =============================================================================
# Log Record Types
# =============================================================================

@dataclass
class LogContext:
    """Context information attached to log records."""
    
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if v is not None and k != "extra"}
        d.update(self.extra)
        return d
    
    def with_update(self, **kwargs) -> "LogContext":
        """Create a new context with updated values."""
        return LogContext(
            trace_id=kwargs.get("trace_id", self.trace_id),
            request_id=kwargs.get("request_id", self.request_id),
            provider=kwargs.get("provider", self.provider),
            model=kwargs.get("model", self.model),
            operation=kwargs.get("operation", self.operation),
            user_id=kwargs.get("user_id", self.user_id),
            extra={**self.extra, **kwargs.get("extra", {})},
        )


@dataclass
class RequestLog:
    """Log record for an LLM request."""
    
    request_id: str
    provider: str
    model: str
    operation: str
    
    # Timing
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Request details (sanitized)
    message_count: int = 0
    tool_count: int = 0
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    # Cache info
    cache_enabled: bool = False
    cache_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ResponseLog:
    """Log record for an LLM response."""
    
    request_id: str
    provider: str
    model: str
    operation: str
    
    # Status
    success: bool = True
    status_code: Optional[int] = None
    error: Optional[str] = None
    
    # Timing
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_ms: Optional[float] = None
    
    # Response details
    finish_reason: Optional[str] = None
    has_tool_calls: bool = False
    tool_call_count: int = 0
    
    # Cache info
    cache_hit: bool = False
    
    # Usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ToolCallLog:
    """Log record for a tool execution."""
    
    request_id: str
    tool_name: str
    tool_call_id: str
    
    # Timing
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_ms: Optional[float] = None
    
    # Status
    success: bool = True
    error: Optional[str] = None
    
    # Truncated output preview
    output_preview: Optional[str] = None
    output_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class UsageLog:
    """Log record for usage tracking."""
    
    request_id: str
    provider: str
    model: str
    
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# =============================================================================
# Structured Logger
# =============================================================================

class StructuredLogger:
    """
    Logger with structured JSON output and context tracking.
    
    Example:
        ```python
        logger = StructuredLogger("llm_client")
        
        with logger.trace_context(provider="openai", model="gpt-4"):
            logger.log_request(RequestLog(...))
            # ... make request ...
            logger.log_response(ResponseLog(...))
        ```
    """
    
    def __init__(
        self,
        name: str = "llm_client",
        level: str = "INFO",
        json_output: bool = True,
        include_timestamp: bool = True,
        redact_keys: bool = True,
    ):
        self.name = name
        self.json_output = json_output
        self.include_timestamp = include_timestamp
        self.redact_keys = redact_keys
        
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper()))
        
        # Context stack for trace correlation
        self._context: LogContext = LogContext()
        
        # Configure handler if not already configured
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            if json_output:
                handler.setFormatter(JSONFormatter())
            else:
                handler.setFormatter(TextFormatter())
            self._logger.addHandler(handler)
    
    @property
    def context(self) -> LogContext:
        return self._context
    
    def set_context(self, **kwargs) -> None:
        """Update the current log context."""
        self._context = self._context.with_update(**kwargs)
    
    @contextmanager
    def trace_context(
        self,
        trace_id: Optional[str] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Context manager for trace correlation.
        
        Args:
            trace_id: Trace ID (auto-generated if not provided)
            **kwargs: Additional context fields
            
        Yields:
            The trace ID
        """
        trace_id = trace_id or generate_trace_id()
        old_context = self._context
        
        try:
            self._context = old_context.with_update(trace_id=trace_id, **kwargs)
            yield trace_id
        finally:
            self._context = old_context
    
    @contextmanager
    def request_context(
        self,
        provider: str,
        model: str,
        operation: str = "complete",
    ) -> Iterator[str]:
        """
        Context manager for a single request.
        
        Args:
            provider: Provider name
            model: Model name
            operation: Operation type
            
        Yields:
            The request ID
        """
        request_id = generate_request_id()
        
        with self.trace_context(
            request_id=request_id,
            provider=provider,
            model=model,
            operation=operation,
        ):
            yield request_id
    
    def _log(
        self,
        level: int,
        message: str,
        event_type: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Internal logging method."""
        record_data = {
            "message": message,
            **self._context.to_dict(),
        }
        
        if event_type:
            record_data["event_type"] = event_type
        
        if data:
            record_data.update(data)
        
        if self.json_output:
            self._logger.log(level, json.dumps(record_data))
        else:
            extras = " ".join(f"{k}={v}" for k, v in record_data.items() if k != "message")
            self._logger.log(level, f"{message} {extras}")
    
    def debug(self, message: str, **kwargs) -> None:
        self._log(logging.DEBUG, message, data=kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self._log(logging.INFO, message, data=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self._log(logging.WARNING, message, data=kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self._log(logging.ERROR, message, data=kwargs)
    
    # Typed logging methods
    
    def log_request(self, request: RequestLog) -> None:
        """Log an LLM request."""
        self._log(
            logging.INFO,
            f"LLM request to {request.provider}/{request.model}",
            event_type="request",
            data=request.to_dict(),
        )
    
    def log_response(self, response: ResponseLog) -> None:
        """Log an LLM response."""
        level = logging.INFO if response.success else logging.WARNING
        message = (
            f"LLM response from {response.provider}/{response.model}"
            f" ({response.duration_ms:.0f}ms)" if response.duration_ms else ""
        )
        self._log(level, message, event_type="response", data=response.to_dict())
    
    def log_tool_call(self, tool_log: ToolCallLog) -> None:
        """Log a tool execution."""
        level = logging.INFO if tool_log.success else logging.WARNING
        message = f"Tool '{tool_log.tool_name}' executed"
        self._log(level, message, event_type="tool_call", data=tool_log.to_dict())
    
    def log_usage(self, usage: UsageLog) -> None:
        """Log usage statistics."""
        self._log(
            logging.INFO,
            f"Usage: {usage.total_tokens} tokens (${usage.total_cost:.4f})",
            event_type="usage",
            data=usage.to_dict(),
        )
    
    def log_cache_hit(self, cache_key: str) -> None:
        """Log a cache hit."""
        self._log(
            logging.DEBUG,
            f"Cache hit: {cache_key}",
            event_type="cache_hit",
            data={"cache_key": cache_key},
        )
    
    def log_cache_miss(self, cache_key: str) -> None:
        """Log a cache miss."""
        self._log(
            logging.DEBUG,
            f"Cache miss: {cache_key}",
            event_type="cache_miss",
            data={"cache_key": cache_key},
        )
    
    def log_error(
        self,
        error: Exception,
        message: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log an error with context."""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs,
        }
        
        # Extract additional info from LLMClientError
        if hasattr(error, "code"):
            error_data["error_code"] = str(error.code.value)
        if hasattr(error, "retryable"):
            error_data["retryable"] = error.retryable
        if hasattr(error, "context") and error.context:
            error_data["error_context"] = error.context.to_dict()
        
        self._log(
            logging.ERROR,
            message or f"Error: {error}",
            event_type="error",
            data=error_data,
        )


# =============================================================================
# Formatters
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        
        # Parse JSON message if present
        try:
            message_data = json.loads(record.getMessage())
            log_data.update(message_data)
        except (json.JSONDecodeError, TypeError):
            log_data["message"] = record.getMessage()
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter."""
    
    LEVEL_COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        color = self.LEVEL_COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""
        
        return f"{timestamp} {color}{record.levelname:8}{reset} {record.getMessage()}"


# =============================================================================
# Utilities
# =============================================================================

def generate_trace_id() -> str:
    """Generate a unique trace ID."""
    return f"trace_{uuid.uuid4().hex[:16]}"


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


def redact_api_key(key: Optional[str]) -> str:
    """Redact an API key for safe logging."""
    if not key:
        return "<not set>"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def truncate_for_log(text: str, max_length: int = 200) -> str:
    """Truncate text for logging."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... ({len(text)} chars total)"


# =============================================================================
# Timing Utilities
# =============================================================================

@dataclass
class Timer:
    """Simple timer for measuring durations."""
    
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    
    def stop(self) -> float:
        """Stop the timer and return duration in milliseconds."""
        self.end_time = time.perf_counter()
        return self.elapsed_ms
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000


@contextmanager
def timed() -> Iterator[Timer]:
    """Context manager for timing operations."""
    timer = Timer()
    try:
        yield timer
    finally:
        timer.stop()


def log_timing(logger: StructuredLogger, operation: str):
    """Decorator for logging function timing."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with timed() as timer:
                result = await func(*args, **kwargs)
            logger.debug(f"{operation} completed", duration_ms=timer.elapsed_ms)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with timed() as timer:
                result = func(*args, **kwargs)
            logger.debug(f"{operation} completed", duration_ms=timer.elapsed_ms)
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# Global Logger
# =============================================================================

_default_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "llm_client") -> StructuredLogger:
    """Get or create a structured logger."""
    global _default_logger
    if _default_logger is None or _default_logger.name != name:
        _default_logger = StructuredLogger(name)
    return _default_logger


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    **kwargs,
) -> StructuredLogger:
    """Configure the default logger."""
    global _default_logger
    _default_logger = StructuredLogger(
        level=level,
        json_output=json_output,
        **kwargs,
    )
    return _default_logger


__all__ = [
    # Context
    "LogContext",
    # Log records
    "RequestLog",
    "ResponseLog",
    "ToolCallLog",
    "UsageLog",
    # Logger
    "StructuredLogger",
    # Formatters
    "JSONFormatter",
    "TextFormatter",
    # Timing
    "Timer",
    "timed",
    "log_timing",
    # Utilities
    "generate_trace_id",
    "generate_request_id",
    "redact_api_key",
    "truncate_for_log",
    # Global
    "get_logger",
    "configure_logging",
]
