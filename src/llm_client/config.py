"""
Configuration system for llm-client.

This module provides typed configuration classes with:
- Dataclass-based settings with validation
- Environment variable loading
- YAML/TOML file loading
- Sensible defaults with override capability
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from dotenv import find_dotenv, load_dotenv


# =============================================================================
# Provider Configuration
# =============================================================================

@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    
    # API settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    
    # Request settings
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    
    # Model defaults
    default_model: Optional[str] = None
    default_temperature: float = 0.7
    default_max_tokens: Optional[int] = None
    
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
    
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    default_model: str = "gpt-4o"
    
    # OpenAI-specific settings
    use_responses_api: bool = False


@dataclass
class AnthropicConfig(ProviderConfig):
    """Anthropic-specific configuration."""
    
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    default_model: str = "claude-sonnet-4-20250514"
    
    # Anthropic-specific settings  
    max_thinking_tokens: Optional[int] = None


# =============================================================================
# Cache Configuration
# =============================================================================

CacheBackendType = Literal["none", "fs", "pg_redis", "qdrant"]


@dataclass
class CacheConfig:
    """Configuration for caching."""
    
    # Backend selection
    backend: CacheBackendType = "none"
    enabled: bool = True
    
    # Collection/namespace
    default_collection: Optional[str] = None
    
    # TTL settings
    ttl_seconds: Optional[int] = None
    
    # Behavior
    cache_errors: bool = False
    only_cache_ok: bool = True


@dataclass
class FSCacheConfig(CacheConfig):
    """Filesystem cache configuration."""
    
    backend: CacheBackendType = "fs"
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    
    def __post_init__(self):
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)


@dataclass
class RedisPGCacheConfig(CacheConfig):
    """PostgreSQL + Redis hybrid cache configuration."""
    
    backend: CacheBackendType = "pg_redis"
    
    # PostgreSQL settings
    pg_dsn: str = field(
        default_factory=lambda: os.getenv(
            "POSTGRES_DSN", 
            "postgresql://postgres:postgres@localhost:5432/postgres"
        )
    )
    
    # Redis settings
    redis_url: str = field(
        default_factory=lambda: os.getenv(
            "REDIS_URL", 
            "redis://localhost:6379/0"
        )
    )
    redis_ttl_seconds: int = 86400  # 24 hours
    
    # Compression
    compress: bool = True
    compression_level: int = 6


@dataclass
class QdrantCacheConfig(CacheConfig):
    """Qdrant vector cache configuration."""
    
    backend: CacheBackendType = "qdrant"
    
    qdrant_url: str = field(
        default_factory=lambda: os.getenv(
            "QDRANT_URL", 
            "http://localhost:6333"
        )
    )
    qdrant_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY")
    )


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    
    # Turn limits
    max_turns: int = 10
    max_tool_calls_per_turn: int = 10
    
    # Tool execution
    parallel_tool_execution: bool = True
    tool_timeout: float = 30.0
    max_tool_output_chars: Optional[int] = None
    
    # Context management
    max_tokens: Optional[int] = None
    reserve_tokens: int = 2000
    
    # Behavior
    stop_on_tool_error: bool = False
    include_tool_errors_in_context: bool = True
    stream_tool_calls: bool = True
    
    def __post_init__(self):
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.tool_timeout <= 0:
            raise ValueError("tool_timeout must be positive")


# =============================================================================
# Logging Configuration
# =============================================================================

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["text", "json"]


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: LogLevel = "INFO"
    format: LogFormat = "text"
    
    # Output settings
    log_file: Optional[Path] = None
    include_timestamp: bool = True
    include_trace_id: bool = True
    
    # What to log
    log_requests: bool = True
    log_responses: bool = True
    log_tool_calls: bool = True
    log_usage: bool = True
    
    # Redaction
    redact_api_keys: bool = True


# =============================================================================
# Rate Limiting Configuration
# =============================================================================

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


# =============================================================================
# Master Configuration
# =============================================================================

@dataclass
class Settings:
    """
    Master configuration for the LLM client.
    
    This aggregates all configuration sections into a single object
    that can be loaded from environment variables, files, or constructed
    programmatically.
    """
    
    # Provider configurations
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    
    # Cache configuration
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Agent configuration
    agent: AgentConfig = field(default_factory=AgentConfig)
    
    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    
    @classmethod
    def from_env(cls, prefix: str = "LLM_") -> "Settings":
        """
        Load settings from environment variables.
        
        Environment variables are prefixed (default: LLM_) and use
        underscore-separated paths for nested settings.
        
        Example:
            LLM_OPENAI_API_KEY=sk-...
            LLM_CACHE_BACKEND=fs
            LLM_AGENT_MAX_TURNS=20
        """
        settings = cls()
        
        # OpenAI settings
        if key := os.getenv(f"{prefix}OPENAI_API_KEY"):
            settings.openai.api_key = key
        if url := os.getenv(f"{prefix}OPENAI_BASE_URL"):
            settings.openai.base_url = url
        if model := os.getenv(f"{prefix}OPENAI_MODEL"):
            settings.openai.default_model = model
        
        # Anthropic settings
        if key := os.getenv(f"{prefix}ANTHROPIC_API_KEY"):
            settings.anthropic.api_key = key
        if model := os.getenv(f"{prefix}ANTHROPIC_MODEL"):
            settings.anthropic.default_model = model
        
        # Cache settings
        if backend := os.getenv(f"{prefix}CACHE_BACKEND"):
            settings.cache.backend = backend  # type: ignore
        if cache_dir := os.getenv(f"{prefix}CACHE_DIR"):
            settings.cache = FSCacheConfig(cache_dir=Path(cache_dir))
        
        # Agent settings
        if max_turns := os.getenv(f"{prefix}AGENT_MAX_TURNS"):
            settings.agent.max_turns = int(max_turns)
        if tool_timeout := os.getenv(f"{prefix}AGENT_TOOL_TIMEOUT"):
            settings.agent.tool_timeout = float(tool_timeout)
        
        # Logging settings
        if level := os.getenv(f"{prefix}LOG_LEVEL"):
            settings.logging.level = level.upper()  # type: ignore
        if log_format := os.getenv(f"{prefix}LOG_FORMAT"):
            settings.logging.format = log_format.lower()  # type: ignore
        
        return settings
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Settings":
        """
        Load settings from a YAML or TOML file.
        
        Args:
            path: Path to configuration file (.yaml, .yml, or .toml)
            
        Returns:
            Settings object with values from file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
        elif suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore
                except ImportError:
                    raise ImportError("tomli is required for TOML config files: pip install tomli")
            with open(path, "rb") as f:
                data = tomllib.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create Settings from a dictionary."""
        settings = cls()
        
        if "openai" in data:
            for key, value in data["openai"].items():
                if hasattr(settings.openai, key):
                    setattr(settings.openai, key, value)
        
        if "anthropic" in data:
            for key, value in data["anthropic"].items():
                if hasattr(settings.anthropic, key):
                    setattr(settings.anthropic, key, value)
        
        if "cache" in data:
            cache_data = data["cache"]
            backend = cache_data.get("backend", "none")
            if backend == "fs":
                settings.cache = FSCacheConfig(**{
                    k: v for k, v in cache_data.items() 
                    if hasattr(FSCacheConfig, k)
                })
            elif backend == "pg_redis":
                settings.cache = RedisPGCacheConfig(**{
                    k: v for k, v in cache_data.items() 
                    if hasattr(RedisPGCacheConfig, k)
                })
            else:
                for key, value in cache_data.items():
                    if hasattr(settings.cache, key):
                        setattr(settings.cache, key, value)
        
        if "agent" in data:
            for key, value in data["agent"].items():
                if hasattr(settings.agent, key):
                    setattr(settings.agent, key, value)
        
        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(settings.logging, key):
                    setattr(settings.logging, key, value)
        
        if "rate_limit" in data:
            for key, value in data["rate_limit"].items():
                if hasattr(settings.rate_limit, key):
                    setattr(settings.rate_limit, key, value)
        
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        import dataclasses
        
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        return convert(self)


# =============================================================================
# Global Settings & Helpers
# =============================================================================

_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance, creating with defaults if needed."""
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings.from_env()
    return _global_settings


def configure(settings: Optional[Settings] = None, **kwargs) -> Settings:
    """
    Configure global settings.
    
    Args:
        settings: Settings object to use globally
        **kwargs: Override specific settings
        
    Returns:
        The configured Settings object
    """
    global _global_settings
    
    if settings is not None:
        _global_settings = settings
    elif _global_settings is None:
        _global_settings = Settings.from_env()
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(_global_settings, key):
            setattr(_global_settings, key, value)
    
    return _global_settings


def load_env(path: Optional[str] = None, *, override: bool = False) -> bool:
    """
    Load environment variables from a .env file.
    
    Args:
        path: Optional path to a .env file. If not provided, uses find_dotenv().
        override: Whether to override existing environment variables.
        
    Returns:
        True if a .env file was found and loaded, False otherwise.
    """
    env_path = path or find_dotenv(usecwd=True)
    if not env_path:
        return False
    return load_dotenv(env_path, override=override)


__all__ = [
    # Provider configs
    "ProviderConfig",
    "OpenAIConfig",
    "AnthropicConfig",
    # Cache configs
    "CacheConfig",
    "FSCacheConfig",
    "RedisPGCacheConfig",
    "QdrantCacheConfig",
    # Other configs
    "AgentConfig",
    "LoggingConfig",
    "RateLimitConfig",
    # Master config
    "Settings",
    # Global functions
    "get_settings",
    "configure",
    "load_env",
]
