"""
Dependency Injection Container and Factories.

This module provides:
- A lightweight DI container for managing dependencies
- Factory functions for creating configured providers
- Service locator pattern for optional dependencies
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")


# =============================================================================
# Service Registry
# =============================================================================

class ServiceRegistry:
    """
    Lightweight dependency injection container.
    
    Supports:
    - Singleton and transient lifetimes
    - Factory functions for lazy instantiation
    - Type-safe service resolution
    
    Example:
        ```python
        registry = ServiceRegistry()
        
        # Register singleton
        registry.register_singleton(OpenAIConfig, OpenAIConfig())
        
        # Register factory
        registry.register_factory(OpenAIProvider, lambda r: OpenAIProvider(
            config=r.resolve(OpenAIConfig)
        ))
        
        # Resolve
        provider = registry.resolve(OpenAIProvider)
        ```
    """
    
    def __init__(self):
        self._singletons: Dict[type, Any] = {}
        self._factories: Dict[type, Callable[["ServiceRegistry"], Any]] = {}
        self._instances: Dict[type, Any] = {}
    
    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """Register a singleton instance."""
        self._singletons[service_type] = instance
        self._instances[service_type] = instance
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[["ServiceRegistry"], T],
        singleton: bool = True,
    ) -> None:
        """
        Register a factory function.
        
        Args:
            service_type: Type to register
            factory: Factory function that receives the registry
            singleton: If True, cache the first instance
        """
        self._factories[service_type] = (factory, singleton)  # type: ignore
    
    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """Register an instance (alias for register_singleton)."""
        self.register_singleton(service_type, instance)
    
    @overload
    def resolve(self, service_type: Type[T]) -> T: ...
    
    @overload
    def resolve(self, service_type: Type[T], default: T) -> T: ...
    
    def resolve(self, service_type: Type[T], default: Any = None) -> Any:
        """
        Resolve a service by type.
        
        Args:
            service_type: Type to resolve
            default: Default value if not registered
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not found and no default provided
        """
        # Check cached instances first
        if service_type in self._instances:
            return self._instances[service_type]
        
        # Check singletons
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # Check factories
        if service_type in self._factories:
            factory, is_singleton = self._factories[service_type]
            instance = factory(self)
            if is_singleton:
                self._instances[service_type] = instance
            return instance
        
        # Return default or raise
        if default is not None:
            return default
        
        raise KeyError(f"Service not registered: {service_type.__name__}")
    
    def try_resolve(self, service_type: Type[T]) -> Optional[T]:
        """Try to resolve a service, returning None if not found."""
        try:
            return self.resolve(service_type)
        except KeyError:
            return None
    
    def has(self, service_type: type) -> bool:
        """Check if a service is registered."""
        return (
            service_type in self._singletons or
            service_type in self._factories or
            service_type in self._instances
        )
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._singletons.clear()
        self._factories.clear()
        self._instances.clear()


# =============================================================================
# Provider Factories
# =============================================================================

def create_openai_provider(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    **kwargs,
):
    """
    Factory function to create an OpenAI provider.
    
    Args:
        api_key: API key (defaults to OPENAI_API_KEY env var)
        model: Model name
        **kwargs: Additional provider options
        
    Returns:
        Configured OpenAIProvider
    """
    from .providers.openai import OpenAIProvider
    from .config import OpenAIConfig
    
    config = OpenAIConfig(api_key=api_key, default_model=model)
    
    return OpenAIProvider(
        model=model,
        api_key=api_key or config.api_key,
        **kwargs,
    )


def create_anthropic_provider(
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
    **kwargs,
):
    """
    Factory function to create an Anthropic provider.
    
    Args:
        api_key: API key (defaults to ANTHROPIC_API_KEY env var)
        model: Model name
        **kwargs: Additional provider options
        
    Returns:
        Configured AnthropicProvider
    """
    from .providers.anthropic import AnthropicProvider
    from .config import AnthropicConfig
    
    config = AnthropicConfig(api_key=api_key, default_model=model)
    
    return AnthropicProvider(
        model=model,
        api_key=api_key or config.api_key,
        **kwargs,
    )


def create_provider(
    provider_name: str,
    model: Optional[str] = None,
    **kwargs,
):
    """
    Generic factory to create a provider by name.
    
    Args:
        provider_name: "openai" or "anthropic"
        model: Model name (uses default if not provided)
        **kwargs: Provider options
        
    Returns:
        Configured provider instance
    """
    provider_name = provider_name.lower()
    
    if provider_name == "openai":
        return create_openai_provider(model=model or "gpt-4o", **kwargs)
    elif provider_name == "anthropic":
        return create_anthropic_provider(model=model or "claude-3-5-sonnet-latest", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


# =============================================================================
# Cache Factories
# =============================================================================

def create_cache(
    backend: str = "none",
    **kwargs,
):
    """
    Factory to create a cache backend.
    
    Args:
        backend: "none", "fs", "pg_redis", "qdrant"
        **kwargs: Backend-specific options
        
    Returns:
        CacheCore instance
    """
    from .cache import (
        CacheCore,
        FSCache,
        FSCacheConfig,
        QdrantCache,
        HybridRedisPostgreSQLCache,
        HybridCacheConfig,
    )
    from pathlib import Path
    
    if backend == "none":
        return CacheCore(backend=None)
    
    if backend == "fs":
        cache_dir = kwargs.get("cache_dir", Path("./cache"))
        config = FSCacheConfig(
            dir=Path(cache_dir),
            client_type=kwargs.get("client_type", "completions"),
            default_collection=kwargs.get("collection", "default"),
        )
        return CacheCore(backend=FSCache(config))
    
    if backend == "pg_redis":
        config = HybridCacheConfig(
            default_table=kwargs.get("collection", "llm_cache"),
            client_type=kwargs.get("client_type", "completions"),
            pg_dsn=kwargs.get("pg_dsn", ""),
            redis_url=kwargs.get("redis_url", ""),
        )
        return CacheCore(backend=HybridRedisPostgreSQLCache(config))
    
    if backend == "qdrant":
        return CacheCore(backend=QdrantCache(
            default_collection=kwargs.get("collection", "llm_cache"),
            client_type=kwargs.get("client_type", "completions"),
            base_url=kwargs.get("qdrant_url"),
            api_key=kwargs.get("qdrant_api_key"),
        ))
    
    raise ValueError(f"Unknown cache backend: {backend}")


# =============================================================================
# Agent Factory
# =============================================================================

def create_agent(
    provider: Optional[Any] = None,
    provider_name: str = "openai",
    model: Optional[str] = None,
    tools: Optional[list] = None,
    system_message: Optional[str] = None,
    **kwargs,
):
    """
    Factory to create a fully configured agent.
    
    Args:
        provider: Pre-configured provider (optional)
        provider_name: "openai" or "anthropic" (if no provider given)
        model: Model name
        tools: List of tools
        system_message: System instruction
        **kwargs: Additional agent config
        
    Returns:
        Configured Agent instance
    """
    from .agent import Agent, AgentConfig
    
    if provider is None:
        provider = create_provider(provider_name, model=model)
    
    config = AgentConfig(
        max_turns=kwargs.pop("max_turns", 10),
        tool_timeout=kwargs.pop("tool_timeout", 30.0),
        parallel_tool_execution=kwargs.pop("parallel_tool_execution", True),
    )
    
    return Agent(
        provider=provider,
        tools=tools,
        system_message=system_message,
        config=config,
        **kwargs,
    )


# =============================================================================
# Application Container
# =============================================================================

@dataclass
class Container:
    """
    Application-level dependency container.
    
    Provides pre-configured factories for common use cases
    with easy access to all services.
    
    Example:
        ```python
        container = Container.from_config(settings)
        
        # Get services
        provider = container.openai_provider()
        cache = container.cache()
        agent = container.agent(tools=[my_tool])
        ```
    """
    
    registry: ServiceRegistry = field(default_factory=ServiceRegistry)
    
    # Lazy-loaded services
    _openai_provider: Optional[Any] = field(default=None, repr=False)
    _anthropic_provider: Optional[Any] = field(default=None, repr=False)
    _cache: Optional[Any] = field(default=None, repr=False)
    
    @classmethod
    def from_config(cls, settings: Optional[Any] = None) -> "Container":
        """
        Create a container from Settings.
        
        Args:
            settings: Settings object (uses global if not provided)
            
        Returns:
            Configured Container
        """
        from .config import Settings, get_settings
        
        settings = settings or get_settings()
        container = cls()
        
        # Register settings
        container.registry.register_singleton(Settings, settings)
        
        return container
    
    @classmethod
    def default(cls) -> "Container":
        """Create a container with default configuration."""
        return cls.from_config()
    
    def openai_provider(self, **kwargs):
        """Get or create OpenAI provider."""
        if self._openai_provider is None or kwargs:
            from .config import Settings
            settings = self.registry.try_resolve(Settings)
            config = settings.openai if settings else None
            
            self._openai_provider = create_openai_provider(
                api_key=kwargs.get("api_key") or (config.api_key if config else None),
                model=kwargs.get("model") or (config.default_model if config else "gpt-4o"),
                **{k: v for k, v in kwargs.items() if k not in ("api_key", "model")},
            )
        return self._openai_provider
    
    def anthropic_provider(self, **kwargs):
        """Get or create Anthropic provider."""
        if self._anthropic_provider is None or kwargs:
            from .config import Settings
            settings = self.registry.try_resolve(Settings)
            config = settings.anthropic if settings else None
            
            self._anthropic_provider = create_anthropic_provider(
                api_key=kwargs.get("api_key") or (config.api_key if config else None),
                model=kwargs.get("model") or (config.default_model if config else "claude-sonnet-4-20250514"),
                **{k: v for k, v in kwargs.items() if k not in ("api_key", "model")},
            )
        return self._anthropic_provider
    
    def provider(self, name: str = "openai", **kwargs):
        """Get or create a provider by name."""
        if name == "openai":
            return self.openai_provider(**kwargs)
        elif name == "anthropic":
            return self.anthropic_provider(**kwargs)
        else:
            return create_provider(name, **kwargs)
    
    def cache(self, **kwargs):
        """Get or create cache."""
        if self._cache is None or kwargs:
            from .config import Settings
            settings = self.registry.try_resolve(Settings)
            config = settings.cache if settings else None
            
            self._cache = create_cache(
                backend=kwargs.get("backend") or (config.backend if config else "none"),
                **{k: v for k, v in kwargs.items() if k != "backend"},
            )
        return self._cache
    
    def agent(
        self,
        provider_name: str = "openai",
        **kwargs,
    ):
        """Create a new agent."""
        provider = kwargs.pop("provider", None)
        if provider is None:
            provider = self.provider(provider_name)
        
        return create_agent(provider=provider, **kwargs)


# =============================================================================
# Global Container
# =============================================================================

_default_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container."""
    global _default_container
    if _default_container is None:
        _default_container = Container.default()
    return _default_container


def set_container(container: Container) -> None:
    """Set the global container."""
    global _default_container
    _default_container = container


__all__ = [
    # Registry
    "ServiceRegistry",
    # Factories
    "create_openai_provider",
    "create_anthropic_provider",
    "create_provider",
    "create_cache",
    "create_agent",
    # Container
    "Container",
    "get_container",
    "set_container",
]
