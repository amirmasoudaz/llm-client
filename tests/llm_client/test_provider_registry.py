from __future__ import annotations

import pytest

from llm_client.provider_registry import (
    ProviderCapabilities,
    ProviderDescriptor,
    ProviderRegistry,
    get_default_provider_registry,
)


class _DummyProvider:
    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs


def test_provider_registry_resolves_aliases_and_creates_instances() -> None:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="example",
            aliases=("sample",),
            factory=_DummyProvider,
            default_model="example-model",
            capabilities=ProviderCapabilities(completions=True, streaming=False, embeddings=True),
            priority=5,
        )
    )

    created = registry.create("sample", api_key="secret", foo="bar")

    assert isinstance(created, _DummyProvider)
    assert created.model == "example-model"
    assert created.api_key == "secret"
    assert created.kwargs["foo"] == "bar"
    assert registry.get("example").aliases == ("sample",)


def test_provider_registry_can_filter_by_capability_and_priority() -> None:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="slow",
            factory=_DummyProvider,
            default_model="slow-model",
            capabilities=ProviderCapabilities(completions=True, streaming=False),
            priority=50,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="fast",
            factory=_DummyProvider,
            default_model="fast-model",
            capabilities=ProviderCapabilities(completions=True, streaming=True),
            priority=10,
        )
    )

    matches = registry.find_capable(completions=True, streaming=True)

    assert [item.name for item in matches] == ["fast"]
    assert [item.name for item in registry.list()] == ["fast", "slow"]


def test_default_provider_registry_exposes_common_providers() -> None:
    registry = get_default_provider_registry()

    assert registry.get("openai").default_model == "gpt-5"
    assert registry.get("openai").capabilities.responses_api is True
    assert registry.get("openai").capabilities.background_responses is True
    assert registry.get("openai").capabilities.responses_native_tools is True
    assert registry.get("openai").capabilities.normalized_output_items is True
    assert registry.get("anthropic").capabilities.reasoning is True
    assert registry.get("anthropic").capabilities.responses_api is False
    assert registry.get("gemini").name == "google"
    assert registry.get("google").capabilities.embeddings is True


def test_provider_registry_can_filter_by_responses_capabilities() -> None:
    registry = get_default_provider_registry()

    matches = registry.find_capable(responses_api=True, background_responses=True, responses_native_tools=True)

    assert [item.name for item in matches] == ["openai"]


def test_provider_registry_rejects_duplicate_names_and_aliases() -> None:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="primary",
            aliases=("alias-one",),
            factory=_DummyProvider,
            default_model="m1",
        )
    )

    with pytest.raises(ValueError):
        registry.register(
            ProviderDescriptor(
                name="primary",
                factory=_DummyProvider,
                default_model="m2",
            )
        )

    with pytest.raises(ValueError):
        registry.register(
            ProviderDescriptor(
                name="secondary",
                aliases=("alias-one",),
                factory=_DummyProvider,
                default_model="m3",
            )
        )


def test_provider_registry_supports_models_for_custom_entries_via_provider_family_metadata() -> None:
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="primary_live",
            factory=_DummyProvider,
            default_model="gpt-5-nano",
            metadata={"provider_family": "openai"},
        )
    )

    assert registry.supports_model("primary_live", "gpt-5-mini") is True
    assert registry.supports_model("primary_live", "claude-4-5-sonnet") is False
    assert registry.resolve_model("primary_live", "gpt-5-mini") == "gpt-5-mini"
