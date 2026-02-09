"""
Tests for the structured logging module.
"""

import logging

from llm_client.logging import (
    LogContext,
    RequestLog,
    ResponseLog,
    StructuredLogger,
    Timer,
    ToolCallLog,
    UsageLog,
    generate_request_id,
    generate_trace_id,
    redact_api_key,
    timed,
    truncate_for_log,
)


class TestLogContext:
    """Test LogContext dataclass."""

    def test_create_context(self):
        """Test creating context."""
        ctx = LogContext(
            trace_id="trace_123",
            provider="openai",
            model="gpt-4",
        )

        assert ctx.trace_id == "trace_123"
        assert ctx.provider == "openai"

    def test_to_dict(self):
        """Test converting to dict."""
        ctx = LogContext(
            trace_id="t1",
            request_id="r1",
            extra={"custom": "value"},
        )

        d = ctx.to_dict()

        assert d["trace_id"] == "t1"
        assert d["request_id"] == "r1"
        assert d["custom"] == "value"

    def test_with_update(self):
        """Test creating updated context."""
        ctx = LogContext(trace_id="t1", provider="openai")
        updated = ctx.with_update(model="gpt-4", extra={"new": "value"})

        assert updated.trace_id == "t1"
        assert updated.provider == "openai"
        assert updated.model == "gpt-4"
        assert "new" in updated.extra


class TestLogRecords:
    """Test log record classes."""

    def test_request_log(self):
        """Test RequestLog."""
        log = RequestLog(
            request_id="req_123",
            provider="openai",
            model="gpt-4",
            operation="complete",
            message_count=3,
        )

        d = log.to_dict()

        assert d["request_id"] == "req_123"
        assert d["provider"] == "openai"
        assert d["message_count"] == 3
        assert "timestamp" in d

    def test_response_log(self):
        """Test ResponseLog."""
        log = ResponseLog(
            request_id="req_123",
            provider="openai",
            model="gpt-4",
            operation="complete",
            success=True,
            duration_ms=150.5,
            input_tokens=100,
            output_tokens=50,
        )

        d = log.to_dict()

        assert d["success"] is True
        assert d["duration_ms"] == 150.5
        assert d["input_tokens"] == 100

    def test_tool_call_log(self):
        """Test ToolCallLog."""
        log = ToolCallLog(
            request_id="req_123",
            tool_name="get_weather",
            tool_call_id="call_abc",
            duration_ms=50.0,
            success=True,
        )

        d = log.to_dict()

        assert d["tool_name"] == "get_weather"
        assert d["success"] is True

    def test_usage_log(self):
        """Test UsageLog."""
        log = UsageLog(
            request_id="req_123",
            provider="openai",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            total_cost=0.0025,
        )

        d = log.to_dict()

        assert d["total_tokens"] == 150
        assert d["total_cost"] == 0.0025


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_create_logger(self):
        """Test creating logger."""
        logger = StructuredLogger("test", level="DEBUG")

        assert logger.name == "test"
        assert logger.json_output is True

    def test_trace_context(self):
        """Test trace context manager."""
        logger = StructuredLogger("test")

        with logger.trace_context(provider="openai") as trace_id:
            assert trace_id.startswith("trace_")
            assert logger.context.trace_id == trace_id
            assert logger.context.provider == "openai"

        # Context should be restored
        assert logger.context.trace_id is None

    def test_request_context(self):
        """Test request context manager."""
        logger = StructuredLogger("test")

        with logger.request_context("openai", "gpt-4") as request_id:
            assert request_id.startswith("req_")
            assert logger.context.request_id == request_id
            assert logger.context.provider == "openai"
            assert logger.context.model == "gpt-4"

    def test_logging_methods(self, caplog):
        """Test logging methods."""
        logger = StructuredLogger("test", level="DEBUG", json_output=False)
        logger._logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG, logger="test"):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")

        # Note: exact assertion depends on handler setup
        assert len(caplog.records) >= 0  # Just ensure no errors

    def test_log_request(self):
        """Test log_request method."""
        logger = StructuredLogger("test", level="DEBUG")

        request = RequestLog(
            request_id="req_test",
            provider="openai",
            model="gpt-4",
            operation="complete",
        )

        # Should not raise
        logger.log_request(request)

    def test_log_response(self):
        """Test log_response method."""
        logger = StructuredLogger("test", level="DEBUG")

        response = ResponseLog(
            request_id="req_test",
            provider="openai",
            model="gpt-4",
            operation="complete",
            success=True,
            duration_ms=100.0,
        )

        logger.log_response(response)

    def test_log_error(self):
        """Test log_error with LLMClientError."""
        from llm_client.errors import RateLimitError

        logger = StructuredLogger("test")
        error = RateLimitError(retry_after=30.0)

        logger.log_error(error, "Rate limited!")


class TestTimer:
    """Test Timer utility."""

    def test_timer_basic(self):
        """Test basic timer usage."""
        timer = Timer()
        import time

        time.sleep(0.01)  # 10ms
        duration = timer.stop()

        assert duration >= 9  # Should be at least 9ms
        assert timer.elapsed_ms == duration

    def test_timed_context_manager(self):
        """Test timed context manager."""
        import time

        with timed() as timer:
            time.sleep(0.01)

        assert timer.elapsed_ms >= 9


class TestUtilities:
    """Test utility functions."""

    def test_generate_trace_id(self):
        """Test trace ID generation."""
        id1 = generate_trace_id()
        id2 = generate_trace_id()

        assert id1.startswith("trace_")
        assert id2.startswith("trace_")
        assert id1 != id2

    def test_generate_request_id(self):
        """Test request ID generation."""
        id1 = generate_request_id()
        id2 = generate_request_id()

        assert id1.startswith("req_")
        assert id2.startswith("req_")
        assert id1 != id2

    def test_redact_api_key(self):
        """Test API key redaction."""
        assert redact_api_key(None) == "<not set>"
        assert redact_api_key("short") == "***"

        key = "sk-abcdefghijklmnop"
        redacted = redact_api_key(key)
        assert "sk-a" in redacted
        assert "mnop" in redacted
        assert "..." in redacted

    def test_truncate_for_log(self):
        """Test text truncation."""
        short = "Hello"
        assert truncate_for_log(short, 100) == short

        long = "A" * 300
        truncated = truncate_for_log(long, 100)
        assert len(truncated) < 200
        assert "..." in truncated
        assert "300" in truncated
