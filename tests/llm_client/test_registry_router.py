from __future__ import annotations

import pytest

from llm_client.provider_registry import ProviderCapabilities, ProviderDescriptor, ProviderRegistry
from llm_client.routing import ProviderHealthTracker, RegistryRouter, RoutingPreferences, RoutingRequirements
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec
from llm_client.validation import ValidationError, validate_spec


class _DummyProvider:
    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs


def _spec(**kwargs) -> RequestSpec:
    base = dict(
        provider="unknown",
        model="gpt-5",
        messages=[Message.user("hello")],
    )
    base.update(kwargs)
    return RequestSpec(**base)


def test_registry_router_honors_explicit_provider_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod

    monkeypatch.setattr(google_mod, "GoogleProvider", _DummyProvider)

    router = RegistryRouter()

    providers = list(router.select(_spec(provider="gemini", model="gemini-2.0-flash")))

    assert len(providers) == 1
    assert isinstance(providers[0], _DummyProvider)
    assert providers[0].model == "gemini-2.0-flash"


def test_registry_router_filters_by_structured_tool_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.anthropic as anthropic_mod
    import llm_client.providers.google as google_mod
    import llm_client.providers.openai as openai_mod

    monkeypatch.setattr(openai_mod, "OpenAIProvider", _DummyProvider)
    monkeypatch.setattr(google_mod, "GoogleProvider", _DummyProvider)
    monkeypatch.setattr(anthropic_mod, "AnthropicProvider", _DummyProvider)

    router = RegistryRouter()

    providers = list(
        router.select(
            _spec(
                response_format="json_object",
                tools=[object()],
            )
        )
    )

    assert len(providers) == 1
    assert isinstance(providers[0], _DummyProvider)
    assert providers[0].model == "gpt-5"


def test_registry_router_respects_allowed_provider_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.google as google_mod
    import llm_client.providers.openai as openai_mod

    monkeypatch.setattr(openai_mod, "OpenAIProvider", _DummyProvider)
    monkeypatch.setattr(google_mod, "GoogleProvider", _DummyProvider)

    router = RegistryRouter(
        allowed_providers=["google"],
        provider_kwargs={"google": {"api_key": "key-1"}},
        requirements=RoutingRequirements(streaming=True),
    )

    providers = list(router.select(_spec(provider="unknown", model="gemini-2.0-flash", stream=True)))

    assert len(providers) == 1
    assert isinstance(providers[0], _DummyProvider)
    assert providers[0].api_key == "key-1"
    assert providers[0].model == "gemini-2.0-flash"


def test_registry_router_auto_route_filters_candidates_by_model_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    import llm_client.providers.anthropic as anthropic_mod
    import llm_client.providers.google as google_mod
    import llm_client.providers.openai as openai_mod

    monkeypatch.setattr(openai_mod, "OpenAIProvider", _DummyProvider)
    monkeypatch.setattr(google_mod, "GoogleProvider", _DummyProvider)
    monkeypatch.setattr(anthropic_mod, "AnthropicProvider", _DummyProvider)

    router = RegistryRouter()

    providers = list(router.select(_spec(provider="unknown", model="claude-4-5-sonnet")))

    assert len(providers) == 1
    assert isinstance(providers[0], _DummyProvider)
    assert providers[0].model == "claude-4-5-sonnet"


def test_validate_spec_rejects_explicit_provider_model_mismatch() -> None:
    with pytest.raises(ValidationError, match="not compatible"):
        validate_spec(_spec(provider="google", model="gpt-5"))


def test_registry_router_demotes_degraded_providers() -> None:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="openai",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="openai", **kwargs),
            default_model="gpt-5",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=10,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="google",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="google", **kwargs),
            default_model="gemini-2.0-flash",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=20,
        )
    )
    router = RegistryRouter(
        registry=registry,
        health_tracker=ProviderHealthTracker(unhealthy_after=2),
    )

    router.record_provider_failure("openai", status=503)
    router.record_provider_failure("openai", status=503)

    providers = list(router.select(_spec(provider="unknown", model=None)))

    assert len(providers) == 1
    assert providers[0].kwargs["provider_name"] == "google"

    router.record_provider_success("openai")
    providers = list(router.select(_spec(provider="unknown", model=None)))

    assert [provider.kwargs["provider_name"] for provider in providers] == ["openai", "google"]


def test_registry_router_honors_ordered_provider_overrides() -> None:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="openai",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="openai", **kwargs),
            default_model="gpt-5",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=10,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="google",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="google", **kwargs),
            default_model="gemini-2.0-flash",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=20,
        )
    )
    router = RegistryRouter(registry=registry)

    providers = list(
        router.select(
            _spec(
                provider="unknown",
                model=None,
                extra={"provider_overrides": ["google", "openai"]},
            )
        )
    )

    assert [provider.kwargs["provider_name"] for provider in providers] == ["google", "openai"]


def test_registry_router_uses_latency_cost_and_compliance_preferences() -> None:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="openai",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="openai", **kwargs),
            default_model="gpt-5",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=20,
            latency_tier="standard",
            cost_tier="standard",
            compliance_tags=("hosted", "soc2"),
        )
    )
    registry.register(
        ProviderDescriptor(
            name="google",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="google", **kwargs),
            default_model="gemini-2.0-flash",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=30,
            latency_tier="low",
            cost_tier="low",
            compliance_tags=("hosted",),
        )
    )
    registry.register(
        ProviderDescriptor(
            name="anthropic",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="anthropic", **kwargs),
            default_model="claude-4-5-sonnet",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=10,
            latency_tier="standard",
            cost_tier="premium",
            compliance_tags=("hosted", "soc2"),
        )
    )
    router = RegistryRouter(
        registry=registry,
        preferences=RoutingPreferences(
            preferred_latency_tier="low",
            preferred_cost_tier="low",
            required_compliance_tags=("hosted",),
        ),
    )

    providers = list(router.select(_spec(provider="unknown", model=None)))

    assert [provider.kwargs["provider_name"] for provider in providers] == ["google", "openai", "anthropic"]


def test_registry_router_spec_preferences_override_router_preferences() -> None:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="openai",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="openai", **kwargs),
            default_model="gpt-5",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=10,
            latency_tier="standard",
            cost_tier="standard",
            compliance_tags=("hosted", "soc2"),
        )
    )
    registry.register(
        ProviderDescriptor(
            name="google",
            factory=lambda model, **kwargs: _DummyProvider(model=model, provider_name="google", **kwargs),
            default_model="gemini-2.0-flash",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=20,
            latency_tier="low",
            cost_tier="low",
            compliance_tags=("hosted",),
        )
    )
    router = RegistryRouter(
        registry=registry,
        preferences=RoutingPreferences(preferred_latency_tier="low"),
    )

    providers = list(
        router.select(
            _spec(
                provider="unknown",
                model=None,
                extra={
                    "preferred_latency_tier": "standard",
                    "required_compliance_tags": ["soc2"],
                },
            )
        )
    )

    assert [provider.kwargs["provider_name"] for provider in providers] == ["openai"]
