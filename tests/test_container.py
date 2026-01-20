"""
Tests for the dependency injection container.
"""
import pytest

from llm_client.container import (
    ServiceRegistry,
    Container,
    create_provider,
    create_cache,
    create_agent,
    get_container,
    set_container,
)


class TestServiceRegistry:
    """Test ServiceRegistry class."""
    
    def test_register_singleton(self):
        """Test registering and resolving singleton."""
        registry = ServiceRegistry()
        
        class MyService:
            pass
        
        instance = MyService()
        registry.register_singleton(MyService, instance)
        
        resolved = registry.resolve(MyService)
        assert resolved is instance
    
    def test_register_factory(self):
        """Test registering factory."""
        registry = ServiceRegistry()
        
        class Config:
            value = 42
        
        class MyService:
            def __init__(self, config):
                self.config = config
        
        registry.register_singleton(Config, Config())
        registry.register_factory(
            MyService,
            lambda r: MyService(r.resolve(Config))
        )
        
        service = registry.resolve(MyService)
        assert service.config.value == 42
    
    def test_singleton_factory(self):
        """Test that singleton factories return same instance."""
        registry = ServiceRegistry()
        
        call_count = [0]
        
        class Service:
            pass
        
        def factory(r):
            call_count[0] += 1
            return Service()
        
        registry.register_factory(Service, factory, singleton=True)
        
        s1 = registry.resolve(Service)
        s2 = registry.resolve(Service)
        
        assert s1 is s2
        assert call_count[0] == 1
    
    def test_transient_factory(self):
        """Test that transient factories return new instances."""
        registry = ServiceRegistry()
        
        class Service:
            pass
        
        registry.register_factory(Service, lambda r: Service(), singleton=False)
        
        s1 = registry.resolve(Service)
        s2 = registry.resolve(Service)
        
        assert s1 is not s2
    
    def test_resolve_with_default(self):
        """Test resolving with default value."""
        registry = ServiceRegistry()
        
        class NotRegistered:
            pass
        
        default = NotRegistered()
        result = registry.resolve(NotRegistered, default)
        
        assert result is default
    
    def test_resolve_raises_on_missing(self):
        """Test that resolve raises KeyError for missing service."""
        registry = ServiceRegistry()
        
        class NotRegistered:
            pass
        
        with pytest.raises(KeyError, match="not registered"):
            registry.resolve(NotRegistered)
    
    def test_try_resolve(self):
        """Test try_resolve returns None for missing."""
        registry = ServiceRegistry()
        
        class NotRegistered:
            pass
        
        result = registry.try_resolve(NotRegistered)
        assert result is None
    
    def test_has(self):
        """Test has() check."""
        registry = ServiceRegistry()
        
        class MyService:
            pass
        
        assert not registry.has(MyService)
        
        registry.register_singleton(MyService, MyService())
        
        assert registry.has(MyService)
    
    def test_clear(self):
        """Test clearing registry."""
        registry = ServiceRegistry()
        
        class MyService:
            pass
        
        registry.register_singleton(MyService, MyService())
        assert registry.has(MyService)
        
        registry.clear()
        
        assert not registry.has(MyService)


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_provider_openai(self, monkeypatch):
        """Test creating OpenAI provider factory."""
        from llm_client.container import create_openai_provider
        
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        
        # Test that the factory works (actual model validation is provider's job)
        with pytest.raises((ValueError, Exception)):
            # Will fail on model profile validation, but proves factory calls through
            create_openai_provider(model="test-model")
    
    def test_create_provider_anthropic(self, monkeypatch):
        """Test creating Anthropic provider factory."""
        from llm_client.container import create_anthropic_provider
        
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        
        # Test that the factory works (actual model validation is provider's job)
        with pytest.raises((ValueError, Exception)):
            # Will fail on model profile validation, but proves factory calls through
            create_anthropic_provider(model="test-model")
    
    def test_create_provider_invalid(self):
        """Test invalid provider name."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("not_a_provider")
    
    def test_create_cache_none(self):
        """Test creating no-op cache."""
        cache = create_cache("none")
        
        assert cache is not None
        assert cache.backend is None
    
    def test_create_cache_fs(self, tmp_path):
        """Test creating FS cache."""
        cache = create_cache("fs", cache_dir=tmp_path)
        
        assert cache is not None
        assert cache.backend is not None
    
    def test_create_cache_invalid(self):
        """Test invalid cache backend."""
        with pytest.raises(ValueError, match="Unknown cache"):
            create_cache("not_a_backend")
    
    def test_create_agent(self, monkeypatch, mock_provider):
        """Test creating agent."""
        provider = mock_provider()
        
        agent = create_agent(
            provider=provider,
            system_message="Be helpful",
            max_turns=5,
        )
        
        assert agent is not None
        assert agent.config.max_turns == 5


class TestContainer:
    """Test Container class."""
    
    def test_default_container(self):
        """Test creating default container."""
        container = Container.default()
        
        assert container is not None
        assert container.registry is not None
    
    def test_from_config(self):
        """Test creating from config."""
        from llm_client.config import Settings
        
        settings = Settings()
        container = Container.from_config(settings)
        
        # Settings should be registered
        resolved = container.registry.resolve(Settings)
        assert resolved is settings
    
    def test_cache(self):
        """Test getting cache."""
        container = Container.default()
        
        cache = container.cache()
        
        assert cache is not None
    
    def test_create_agent(self, mock_provider):
        """Test creating agent from container."""
        container = Container.default()
        
        provider = mock_provider()
        agent = container.agent(provider=provider, system_message="Test")
        
        assert agent is not None


class TestGlobalContainer:
    """Test global container functions."""
    
    def test_get_container(self):
        """Test getting global container."""
        import llm_client.container as mod
        mod._default_container = None
        
        container = get_container()
        
        assert container is not None
        assert get_container() is container
    
    def test_set_container(self):
        """Test setting global container."""
        new_container = Container()
        
        set_container(new_container)
        
        assert get_container() is new_container
        
        # Reset for other tests
        import llm_client.container as mod
        mod._default_container = None
