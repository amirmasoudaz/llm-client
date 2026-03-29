"""
Settings master configuration and global helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from dotenv import find_dotenv, load_dotenv

from ..config_schema import CONFIG_SCHEMA
from .agent import AgentConfig
from .cache import CacheConfig, FSCacheConfig, RedisPGCacheConfig
from .logging import LoggingConfig, MetricsConfig, RateLimitConfig
from .provider import AnthropicConfig, GoogleConfig, OpenAIConfig


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
    google: GoogleConfig = field(default_factory=GoogleConfig)

    # Cache configuration
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Agent configuration
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Metrics configuration
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    @classmethod
    def from_env(cls, prefix: str = "LLM_") -> Settings:
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

        # Google settings
        if key := os.getenv(f"{prefix}GOOGLE_API_KEY"):
            settings.google.api_key = key
        if model := os.getenv(f"{prefix}GOOGLE_MODEL"):
            settings.google.default_model = model

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

        if batch_concurrency := os.getenv(f"{prefix}AGENT_BATCH_CONCURRENCY"):
            settings.agent.batch_concurrency = int(batch_concurrency)

        # Logging settings
        if level := os.getenv(f"{prefix}LOG_LEVEL"):
            settings.logging.level = level.upper()  # type: ignore
        if log_format := os.getenv(f"{prefix}LOG_FORMAT"):
            settings.logging.format = log_format.lower()  # type: ignore

        # Metrics settings
        if metrics_enabled := os.getenv(f"{prefix}METRICS_ENABLED"):
            settings.metrics.enabled = metrics_enabled.lower() == "true"
        if metrics_provider := os.getenv(f"{prefix}METRICS_PROVIDER"):
            settings.metrics.provider = metrics_provider.lower()
        if prom_port := os.getenv(f"{prefix}METRICS_PROMETHEUS_PORT"):
            settings.metrics.prometheus_port = int(prom_port)
        if otel_endpoint := os.getenv(f"{prefix}METRICS_OTEL_ENDPOINT"):
            settings.metrics.otel_endpoint = otel_endpoint

        return settings

    @classmethod
    def from_file(cls, path: str | Path) -> Settings:
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
                import yaml  # type: ignore[import-untyped]

                with open(path) as f:
                    data = yaml.safe_load(f)
            except ImportError as exc:
                raise ImportError("PyYAML is required for YAML config files: pip install pyyaml") from exc
        elif suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError as exc:
                    raise ImportError("tomli is required for TOML config files: pip install tomli") from exc
            with open(path, "rb") as f:
                data = tomllib.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")

        return cls._from_dict(data)

    @classmethod
    def default(cls) -> Settings:
        """
        Create default configuration.

        Returns:
            Settings object with default values
        """
        return cls()

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Settings:
        """
        Create Settings from a dictionary.

        This method validates the input dictionary against the configuration schema
        before creating the Settings object.
        """
        try:
            jsonschema.validate(instance=data, schema=CONFIG_SCHEMA)
        except ImportError:
            # Skip schema validation if jsonschema is not installed (runtime optional)
            pass
        except Exception as e:
            # Wrap schema validation errors
            raise ValueError(f"Configuration validation failed: {str(e)}") from e

        settings = cls()

        if "openai" in data:
            for key, value in data["openai"].items():
                if hasattr(settings.openai, key):
                    setattr(settings.openai, key, value)

        if "anthropic" in data:
            for key, value in data["anthropic"].items():
                if hasattr(settings.anthropic, key):
                    setattr(settings.anthropic, key, value)

        if "google" in data:
            for key, value in data["google"].items():
                if hasattr(settings.google, key):
                    setattr(settings.google, key, value)

        if "cache" in data:
            cache_data = data["cache"]
            backend = cache_data.get("backend", "none")
            if backend == "fs":
                settings.cache = FSCacheConfig(**{k: v for k, v in cache_data.items() if hasattr(FSCacheConfig, k)})
            elif backend == "pg_redis":
                settings.cache = RedisPGCacheConfig(
                    **{k: v for k, v in cache_data.items() if hasattr(RedisPGCacheConfig, k)}
                )
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

        if "metrics" in data:
            for key, value in data["metrics"].items():
                if hasattr(settings.metrics, key):
                    setattr(settings.metrics, key, value)

        if "rate_limit" in data:
            for key, value in data["rate_limit"].items():
                if hasattr(settings.rate_limit, key):
                    setattr(settings.rate_limit, key, value)

        return settings

    def to_dict(self) -> dict[str, Any]:
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

_global_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance, creating with defaults if needed."""
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings.from_env()
    return _global_settings


def configure(settings: Settings | None = None, **kwargs) -> Settings:
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


def load_env(path: str | None = None, *, override: bool = False) -> bool:
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


__all__ = ["Settings", "get_settings", "configure", "load_env"]
