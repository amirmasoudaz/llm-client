"""
Provider registry and capability metadata.

This module centralizes provider registration, capability metadata, aliases,
and factory-based provider creation so provider resolution is no longer ad hoc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable


ProviderFactory = Callable[..., Any]


@dataclass(frozen=True)
class ProviderCapabilities:
    completions: bool = True
    streaming: bool = True
    embeddings: bool = False
    tool_calling: bool = False
    structured_outputs: bool = False
    reasoning: bool = False
    vision_input: bool = False
    audio_input: bool = False
    file_input: bool = False


@dataclass(frozen=True)
class ProviderDescriptor:
    name: str
    factory: ProviderFactory
    default_model: str
    aliases: tuple[str, ...] = ()
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    priority: int = 100
    latency_tier: str = "standard"
    cost_tier: str = "standard"
    compliance_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderRegistry:
    def __init__(self) -> None:
        self._descriptors: dict[str, ProviderDescriptor] = {}
        self._aliases: dict[str, str] = {}

    def register(self, descriptor: ProviderDescriptor) -> None:
        name = descriptor.name.strip().lower()
        if not name:
            raise ValueError("Provider name cannot be empty")
        if name in self._descriptors:
            raise ValueError(f"Provider already registered: {name}")
        self._descriptors[name] = ProviderDescriptor(
            name=name,
            factory=descriptor.factory,
            default_model=descriptor.default_model,
            aliases=tuple(alias.strip().lower() for alias in descriptor.aliases if alias.strip()),
            capabilities=descriptor.capabilities,
            priority=descriptor.priority,
            latency_tier=descriptor.latency_tier,
            cost_tier=descriptor.cost_tier,
            compliance_tags=descriptor.compliance_tags,
            metadata=dict(descriptor.metadata),
        )
        for alias in self._descriptors[name].aliases:
            if alias == name:
                continue
            if alias in self._aliases or alias in self._descriptors:
                raise ValueError(f"Provider alias already registered: {alias}")
            self._aliases[alias] = name

    def get(self, name: str) -> ProviderDescriptor:
        normalized = self.resolve_name(name)
        try:
            return self._descriptors[normalized]
        except KeyError:
            raise ValueError(f"Unknown provider: {name}") from None

    def resolve_name(self, name: str) -> str:
        normalized = str(name or "").strip().lower()
        if not normalized:
            raise ValueError("Provider name cannot be empty")
        return self._aliases.get(normalized, normalized)

    def create(self, name: str, *, model: str | None = None, **kwargs: Any) -> Any:
        descriptor = self.get(name)
        return descriptor.factory(model=self.resolve_model(name, model), **kwargs)

    def list(self) -> list[ProviderDescriptor]:
        return sorted(self._descriptors.values(), key=lambda item: (item.priority, item.name))

    def find_capable(
        self,
        *,
        model: str | None = None,
        completions: bool | None = None,
        streaming: bool | None = None,
        embeddings: bool | None = None,
        tool_calling: bool | None = None,
        structured_outputs: bool | None = None,
        reasoning: bool | None = None,
        vision_input: bool | None = None,
        audio_input: bool | None = None,
        file_input: bool | None = None,
    ) -> list[ProviderDescriptor]:
        matches: list[ProviderDescriptor] = []
        for descriptor in self.list():
            caps = descriptor.capabilities
            if model is not None and not self.supports_model(descriptor.name, model):
                continue
            if completions is not None and caps.completions != completions:
                continue
            if streaming is not None and caps.streaming != streaming:
                continue
            if embeddings is not None and caps.embeddings != embeddings:
                continue
            if tool_calling is not None and caps.tool_calling != tool_calling:
                continue
            if structured_outputs is not None and caps.structured_outputs != structured_outputs:
                continue
            if reasoning is not None and caps.reasoning != reasoning:
                continue
            if vision_input is not None and caps.vision_input != vision_input:
                continue
            if audio_input is not None and caps.audio_input != audio_input:
                continue
            if file_input is not None and caps.file_input != file_input:
                continue
            matches.append(descriptor)
        return matches

    def supports_model(self, name: str, model: str | None) -> bool:
        if model is None or not str(model).strip():
            return True
        from .model_catalog import infer_provider_for_model

        provider_name = self.effective_provider_name(name)
        inferred_provider = infer_provider_for_model(str(model).strip())
        if inferred_provider == "unknown":
            return True
        return inferred_provider == provider_name

    def effective_provider_name(self, name: str) -> str:
        descriptor = self.get(name)
        provider_family = str(descriptor.metadata.get("provider_family", "")).strip().lower()
        if provider_family:
            return self.resolve_name(provider_family)
        return descriptor.name

    def resolve_model(self, name: str, model: str | None) -> str:
        descriptor = self.get(name)
        if model is None or not str(model).strip():
            return descriptor.default_model
        requested = str(model).strip()
        if self.supports_model(name, requested):
            return requested
        return descriptor.default_model


def _create_openai_provider(*, model: str, api_key: str | None = None, **kwargs: Any) -> Any:
    from .config import OpenAIConfig
    from .providers.openai import OpenAIProvider

    config = OpenAIConfig(api_key=api_key, default_model=model)
    return OpenAIProvider(model=model, api_key=api_key or config.api_key, **kwargs)


def _create_anthropic_provider(*, model: str, api_key: str | None = None, **kwargs: Any) -> Any:
    from .config import AnthropicConfig
    from .providers.anthropic import AnthropicProvider

    config = AnthropicConfig(api_key=api_key, default_model=model)
    return AnthropicProvider(model=model, api_key=api_key or config.api_key, **kwargs)


def _create_google_provider(*, model: str, api_key: str | None = None, **kwargs: Any) -> Any:
    from .config import GoogleConfig
    from .providers.google import GoogleProvider

    config = GoogleConfig(api_key=api_key, default_model=model)
    return GoogleProvider(model=model, api_key=api_key or config.api_key, **kwargs)


@lru_cache(maxsize=1)
def get_default_provider_registry() -> ProviderRegistry:
    try:
        from .model_catalog import get_default_model_catalog

        catalog = get_default_model_catalog()
        openai_default = catalog.default_key_for_provider("openai") or "gpt-5"
        anthropic_default = catalog.default_key_for_provider("anthropic") or "claude-sonnet-4"
        google_default = catalog.default_key_for_provider("google") or "gemini-2.0-flash"
    except Exception:
        openai_default = "gpt-5"
        anthropic_default = "claude-sonnet-4"
        google_default = "gemini-2.0-flash"
    registry = ProviderRegistry()
    registry.register(
        ProviderDescriptor(
            name="openai",
            factory=_create_openai_provider,
            default_model=openai_default,
            capabilities=ProviderCapabilities(
                completions=True,
                streaming=True,
                embeddings=True,
                tool_calling=True,
                structured_outputs=True,
                reasoning=True,
                vision_input=True,
                audio_input=True,
                file_input=True,
            ),
            priority=10,
            latency_tier="standard",
            cost_tier="standard",
            compliance_tags=("hosted",),
        )
    )
    registry.register(
        ProviderDescriptor(
            name="anthropic",
            factory=_create_anthropic_provider,
            default_model=anthropic_default,
            capabilities=ProviderCapabilities(
                completions=True,
                streaming=True,
                embeddings=False,
                tool_calling=True,
                structured_outputs=False,
                reasoning=True,
                vision_input=True,
                file_input=True,
            ),
            priority=20,
            latency_tier="standard",
            cost_tier="standard",
            compliance_tags=("hosted",),
        )
    )
    registry.register(
        ProviderDescriptor(
            name="google",
            factory=_create_google_provider,
            default_model=google_default,
            aliases=("gemini",),
            capabilities=ProviderCapabilities(
                completions=True,
                streaming=True,
                embeddings=True,
                tool_calling=True,
                structured_outputs=True,
                reasoning=False,
                vision_input=True,
                audio_input=True,
                file_input=True,
            ),
            priority=15,
            latency_tier="low",
            cost_tier="low",
            compliance_tags=("hosted",),
        )
    )
    return registry


__all__ = [
    "ProviderCapabilities",
    "ProviderDescriptor",
    "ProviderRegistry",
    "get_default_provider_registry",
]
