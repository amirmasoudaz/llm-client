"""
Configuration system for llm-client.

This package provides typed configuration classes with:
- Dataclass-based settings with validation
- Environment variable loading
- YAML/TOML file loading
- Sensible defaults with override capability
"""

from .agent import AgentConfig
from .base import CacheBackendType, LogFormat, LogLevel
from .cache import CacheConfig, FSCacheConfig, QdrantCacheConfig, RedisPGCacheConfig
from .logging import LoggingConfig, MetricsConfig, RateLimitConfig
from .provider import AnthropicConfig, GoogleConfig, OpenAIConfig, ProviderConfig
from .settings import Settings, configure, get_settings, load_env

__all__ = [
    # Types
    "CacheBackendType",
    "LogLevel",
    "LogFormat",
    # Provider configs
    "ProviderConfig",
    "OpenAIConfig",
    "AnthropicConfig",
    "GoogleConfig",
    # Cache configs
    "CacheConfig",
    "FSCacheConfig",
    "RedisPGCacheConfig",
    "QdrantCacheConfig",
    # Other configs
    "AgentConfig",
    "LoggingConfig",
    "MetricsConfig",
    "RateLimitConfig",
    # Master config
    "Settings",
    # Global functions
    "get_settings",
    "configure",
    "load_env",
]
