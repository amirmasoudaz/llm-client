"""
Tests for the error taxonomy.
"""
import pytest

from llm_client.errors import (
    # Base
    ErrorCode,
    ErrorContext,
    LLMClientError,
    # Provider errors
    ProviderError,
    RateLimitError,
    AuthenticationError,
    QuotaExceededError,
    ModelNotFoundError,
    ContextLengthError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    # Validation errors
    ValidationError,
    MessageTooLongError,
    # Cache errors
    CacheError,
    CacheConnectionError,
    # Tool errors
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
    # Agent errors
    AgentError,
    MaxTurnsExceededError,
    # Config errors
    ConfigError,
    MissingAPIKeyError,
    # Utilities
    error_from_status,
    is_retryable,
)


class TestErrorCodes:
    """Test error code enumeration."""
    
    def test_error_codes_are_strings(self):
        """Test that error codes are strings."""
        assert ErrorCode.RATE_LIMIT.value.startswith("ERR_")
        assert ErrorCode.AUTHENTICATION.value.startswith("ERR_")
    
    def test_error_codes_unique(self):
        """Test that all error codes are unique."""
        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values))


class TestErrorContext:
    """Test error context."""
    
    def test_create_context(self):
        """Test creating error context."""
        ctx = ErrorContext(
            request_id="req_123",
            provider="openai",
            model="gpt-5-nano",
            attempt=2,
        )
        
        assert ctx.request_id == "req_123"
        assert ctx.provider == "openai"
        assert ctx.attempt == 2
    
    def test_to_dict(self):
        """Test context serialization."""
        ctx = ErrorContext(
            request_id="req_456",
            provider="anthropic",
            extra={"custom": "data"},
        )
        
        d = ctx.to_dict()
        
        assert d["request_id"] == "req_456"
        assert d["provider"] == "anthropic"
        assert d["custom"] == "data"


class TestLLMClientError:
    """Test base LLM client error."""
    
    def test_create_error(self):
        """Test creating base error."""
        error = LLMClientError("Something went wrong")
        
        assert error.message == "Something went wrong"
        assert error.code == ErrorCode.INTERNAL_ERROR
        assert not error.retryable
    
    def test_error_with_context(self):
        """Test error with context."""
        ctx = ErrorContext(request_id="req_789")
        error = LLMClientError("Error", context=ctx)
        
        assert error.context.request_id == "req_789"
    
    def test_error_str_includes_code(self):
        """Test string representation includes error code."""
        error = LLMClientError("Test error")
        s = str(error)
        
        assert "ERR_" in s
        assert "Test error" in s
    
    def test_to_dict(self):
        """Test error serialization."""
        error = LLMClientError(
            "Test error",
            context=ErrorContext(provider="openai"),
        )
        
        d = error.to_dict()
        
        assert d["error_type"] == "LLMClientError"
        assert d["message"] == "Test error"
        assert d["context"]["provider"] == "openai"


class TestProviderErrors:
    """Test provider-specific errors."""
    
    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError(retry_after=30.0)
        
        assert error.code == ErrorCode.RATE_LIMIT
        assert error.retryable is True
        assert error.http_status == 429
        assert error.retry_after == 30.0
    
    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError()
        
        assert error.code == ErrorCode.AUTHENTICATION
        assert error.retryable is False
        assert error.http_status == 401
    
    def test_model_not_found_error(self):
        """Test model not found error."""
        error = ModelNotFoundError(model="gpt-6")
        
        assert "gpt-6" in error.message
        assert error.code == ErrorCode.MODEL_NOT_FOUND
    
    def test_context_length_error(self):
        """Test context length error."""
        error = ContextLengthError(max_tokens=4096, actual_tokens=5000)
        
        assert error.max_tokens == 4096
        assert error.actual_tokens == 5000
    
    def test_provider_timeout_error(self):
        """Test provider timeout error."""
        error = ProviderTimeoutError(timeout=30.0)
        
        assert error.retryable is True
        assert error.timeout == 30.0
    
    def test_provider_unavailable_error(self):
        """Test provider unavailable error."""
        error = ProviderUnavailableError()
        
        assert error.retryable is True
        assert error.http_status == 503


class TestValidationErrors:
    """Test validation errors."""
    
    def test_validation_error(self):
        """Test base validation error."""
        error = ValidationError("Invalid input")
        
        assert error.code == ErrorCode.VALIDATION_ERROR
        assert error.retryable is False
    
    def test_message_too_long_error(self):
        """Test message too long error."""
        error = MessageTooLongError(
            max_length=10000,
            actual_length=15000,
        )
        
        assert error.max_length == 10000
        assert error.actual_length == 15000


class TestToolErrors:
    """Test tool-related errors."""
    
    def test_tool_not_found_error(self):
        """Test tool not found error."""
        error = ToolNotFoundError(tool_name="missing_tool")
        
        assert "missing_tool" in error.message
        assert error.tool_name == "missing_tool"
    
    def test_tool_execution_error(self):
        """Test tool execution error."""
        error = ToolExecutionError(
            "Tool failed",
            tool_name="failing_tool",
            cause=ValueError("Inner error"),
        )
        
        assert error.tool_name == "failing_tool"
        assert error.cause is not None
    
    def test_tool_timeout_error(self):
        """Test tool timeout error."""
        error = ToolTimeoutError(timeout=10.0, tool_name="slow_tool")
        
        assert error.retryable is True
        assert error.timeout == 10.0
        assert error.tool_name == "slow_tool"


class TestAgentErrors:
    """Test agent-related errors."""
    
    def test_max_turns_exceeded_error(self):
        """Test max turns exceeded error."""
        error = MaxTurnsExceededError(max_turns=10)
        
        assert error.max_turns == 10
        assert error.code == ErrorCode.MAX_TURNS_EXCEEDED


class TestConfigErrors:
    """Test configuration errors."""
    
    def test_missing_api_key_error(self):
        """Test missing API key error."""
        error = MissingAPIKeyError(
            provider="openai",
            env_var="OPENAI_API_KEY",
        )
        
        assert error.provider == "openai"
        assert error.env_var == "OPENAI_API_KEY"


class TestErrorFromStatus:
    """Test error_from_status utility."""
    
    def test_401_returns_authentication_error(self):
        """Test 401 maps to AuthenticationError."""
        error = error_from_status(401, "Unauthorized")
        
        assert isinstance(error, AuthenticationError)
    
    def test_429_returns_rate_limit_error(self):
        """Test 429 maps to RateLimitError."""
        error = error_from_status(429, "Too Many Requests")
        
        assert isinstance(error, RateLimitError)
    
    def test_503_returns_unavailable_error(self):
        """Test 503 maps to ProviderUnavailableError."""
        error = error_from_status(503, "Service Unavailable")
        
        assert isinstance(error, ProviderUnavailableError)
    
    def test_unknown_status_returns_provider_error(self):
        """Test unknown status returns base ProviderError."""
        error = error_from_status(418, "I'm a teapot")
        
        assert isinstance(error, ProviderError)


class TestIsRetryable:
    """Test is_retryable utility."""
    
    def test_retryable_llm_error(self):
        """Test retryable LLM error."""
        error = RateLimitError()
        assert is_retryable(error) is True
    
    def test_non_retryable_llm_error(self):
        """Test non-retryable LLM error."""
        error = AuthenticationError()
        assert is_retryable(error) is False
    
    def test_timeout_error(self):
        """Test asyncio.TimeoutError is retryable."""
        import asyncio
        error = asyncio.TimeoutError()
        assert is_retryable(error) is True
    
    def test_connection_error(self):
        """Test ConnectionError is retryable."""
        error = ConnectionError()
        assert is_retryable(error) is True
    
    def test_value_error_not_retryable(self):
        """Test ValueError is not retryable."""
        error = ValueError("Bad value")
        assert is_retryable(error) is False
