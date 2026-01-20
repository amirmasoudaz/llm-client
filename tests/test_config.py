"""
Tests for the configuration system.
"""
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from llm_client.config import (
    ProviderConfig,
    OpenAIConfig,
    AnthropicConfig,
    CacheConfig,
    FSCacheConfig,
    RedisPGCacheConfig,
    AgentConfig,
    LoggingConfig,
    RateLimitConfig,
    Settings,
    get_settings,
    configure,
    load_env,
)


class TestProviderConfig:
    """Test provider configuration classes."""
    
    def test_default_provider_config(self):
        """Test default values."""
        config = ProviderConfig()
        
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_backoff == 1.0
        assert config.default_temperature == 0.7
    
    def test_provider_validation(self):
        """Test validation in __post_init__."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            ProviderConfig(timeout=-1)
        
        with pytest.raises(ValueError, match="max_retries cannot be negative"):
            ProviderConfig(max_retries=-1)
    
    def test_openai_config_defaults(self):
        """Test OpenAI-specific defaults."""
        config = OpenAIConfig()
        
        assert config.default_model == "gpt-4o"
        assert config.use_responses_api is False
    
    def test_anthropic_config_defaults(self):
        """Test Anthropic-specific defaults."""
        config = AnthropicConfig()
        
        assert "claude" in config.default_model.lower()
        assert config.max_thinking_tokens is None


class TestCacheConfig:
    """Test cache configuration."""
    
    def test_default_cache_config(self):
        """Test defaults."""
        config = CacheConfig()
        
        assert config.backend == "none"
        assert config.enabled is True
        assert config.cache_errors is False
    
    def test_fs_cache_config(self):
        """Test FS cache config."""
        config = FSCacheConfig(cache_dir=Path("/tmp/cache"))
        
        assert config.backend == "fs"
        assert config.cache_dir == Path("/tmp/cache")
    
    def test_fs_cache_path_conversion(self):
        """Test path string conversion."""
        config = FSCacheConfig(cache_dir="/tmp/test")
        
        assert isinstance(config.cache_dir, Path)
    
    def test_redis_pg_config(self):
        """Test Redis+PG cache config."""
        config = RedisPGCacheConfig()
        
        assert config.backend == "pg_redis"
        assert config.compress is True
        assert config.redis_ttl_seconds == 86400


class TestAgentConfig:
    """Test agent configuration."""
    
    def test_defaults(self):
        """Test default values."""
        config = AgentConfig()
        
        assert config.max_turns == 10
        assert config.parallel_tool_execution is True
        assert config.tool_timeout == 30.0
    
    def test_validation(self):
        """Test validation."""
        with pytest.raises(ValueError, match="max_turns must be at least 1"):
            AgentConfig(max_turns=0)
        
        with pytest.raises(ValueError, match="tool_timeout must be positive"):
            AgentConfig(tool_timeout=0)


class TestLoggingConfig:
    """Test logging configuration."""
    
    def test_defaults(self):
        """Test defaults."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format == "text"
        assert config.redact_api_keys is True


class TestSettings:
    """Test the master Settings class."""
    
    def test_default_settings(self):
        """Test creating default settings."""
        settings = Settings()
        
        assert isinstance(settings.openai, OpenAIConfig)
        assert isinstance(settings.anthropic, AnthropicConfig)
        assert isinstance(settings.cache, CacheConfig)
        assert isinstance(settings.agent, AgentConfig)
        assert isinstance(settings.logging, LoggingConfig)
        assert isinstance(settings.rate_limit, RateLimitConfig)
    
    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("LLM_OPENAI_API_KEY", "sk-test123")
        monkeypatch.setenv("LLM_AGENT_MAX_TURNS", "20")
        monkeypatch.setenv("LLM_LOG_LEVEL", "DEBUG")
        
        settings = Settings.from_env()
        
        assert settings.openai.api_key == "sk-test123"
        assert settings.agent.max_turns == 20
        assert settings.logging.level == "DEBUG"
    
    def test_from_yaml_file(self):
        """Test loading from YAML file."""
        pytest.importorskip("yaml")
        
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("""
openai:
  api_key: sk-yaml-test
  default_model: gpt-5

agent:
  max_turns: 15
  tool_timeout: 45.0

logging:
  level: DEBUG
""")
            
            settings = Settings.from_file(config_path)
            
            assert settings.openai.api_key == "sk-yaml-test"
            assert settings.openai.default_model == "gpt-5"
            assert settings.agent.max_turns == 15
            assert settings.logging.level == "DEBUG"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        settings = Settings()
        d = settings.to_dict()
        
        assert "openai" in d
        assert "anthropic" in d
        assert "cache" in d
        assert "agent" in d
    
    def test_file_not_found(self):
        """Test error for missing file."""
        with pytest.raises(FileNotFoundError):
            Settings.from_file("/nonexistent/config.yaml")


class TestGlobalSettings:
    """Test global settings functions."""
    
    def test_get_settings(self, monkeypatch):
        """Test getting global settings."""
        # Clear any existing
        import llm_client.config as cfg
        cfg._global_settings = None
        
        settings = get_settings()
        assert isinstance(settings, Settings)
        
        # Should return same instance
        assert get_settings() is settings
    
    def test_configure(self):
        """Test configuration helper."""
        import llm_client.config as cfg
        cfg._global_settings = None
        
        settings = configure()
        assert isinstance(settings, Settings)
        
        # Configure with overrides
        new_agent = AgentConfig(max_turns=5)
        settings = configure(agent=new_agent)
        assert settings.agent.max_turns == 5
