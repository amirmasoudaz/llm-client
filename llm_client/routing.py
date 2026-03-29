"""
Provider routing utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import time
from typing import Any
from typing import Protocol

from .provider_registry import ProviderCapabilities, ProviderRegistry, get_default_provider_registry
from .providers.base import Provider
from .spec import RequestSpec

_LATENCY_TIER_ORDER = {"low": 0, "standard": 1, "high": 2}
_COST_TIER_ORDER = {"low": 0, "standard": 1, "premium": 2}


class ProviderRouter(Protocol):
    def select(self, spec: RequestSpec) -> Iterable[Provider]: ...


class StaticRouter:
    """
    Simple ordered fallback router.
    """

    def __init__(self, providers: list[Provider]) -> None:
        if not providers:
            raise ValueError("StaticRouter requires at least one provider.")
        self._providers = providers

    def select(self, spec: RequestSpec) -> Iterable[Provider]:
        return list(self._providers)


@dataclass(frozen=True)
class RoutingRequirements:
    completions: bool = True
    streaming: bool | None = None
    embeddings: bool | None = None
    tool_calling: bool | None = None
    structured_outputs: bool | None = None
    reasoning: bool | None = None
    vision_input: bool | None = None
    audio_input: bool | None = None
    file_input: bool | None = None


@dataclass(frozen=True)
class RoutingPreferences:
    preferred_latency_tier: str | None = None
    preferred_cost_tier: str | None = None
    required_compliance_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderHealthStatus:
    provider: str
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    degraded: bool = False
    last_status: int | None = None
    last_error_at: float | None = None
    last_success_at: float | None = None


class ProviderHealthTracker:
    def __init__(self, *, unhealthy_after: int = 3) -> None:
        self._unhealthy_after = max(1, int(unhealthy_after))
        self._entries: dict[str, ProviderHealthStatus] = {}

    def get(self, provider: str) -> ProviderHealthStatus:
        provider_name = str(provider or "").strip().lower()
        if not provider_name:
            raise ValueError("Provider name cannot be empty")
        return self._entries.get(provider_name, ProviderHealthStatus(provider=provider_name))

    def record_success(self, provider: str) -> ProviderHealthStatus:
        current = self.get(provider)
        updated = ProviderHealthStatus(
            provider=current.provider,
            successes=current.successes + 1,
            failures=current.failures,
            consecutive_failures=0,
            degraded=False,
            last_status=None,
            last_error_at=current.last_error_at,
            last_success_at=time.time(),
        )
        self._entries[current.provider] = updated
        return updated

    def record_failure(self, provider: str, *, status: int | None = None) -> ProviderHealthStatus:
        current = self.get(provider)
        consecutive = current.consecutive_failures + 1
        updated = ProviderHealthStatus(
            provider=current.provider,
            successes=current.successes,
            failures=current.failures + 1,
            consecutive_failures=consecutive,
            degraded=consecutive >= self._unhealthy_after,
            last_status=status,
            last_error_at=time.time(),
            last_success_at=current.last_success_at,
        )
        self._entries[current.provider] = updated
        return updated

    def list(self) -> list[ProviderHealthStatus]:
        return [self.get(name) for name in sorted(self._entries)]


class RegistryRouter:
    """
    Registry-backed router with simple capability-aware selection.

    Selection rules:
    - explicit `spec.provider` wins when present and registered
    - otherwise match providers by required capabilities
    - sort by registry priority
    """

    def __init__(
        self,
        *,
        registry: ProviderRegistry | None = None,
        allowed_providers: list[str] | None = None,
        provider_kwargs: dict[str, dict[str, Any]] | None = None,
        requirements: RoutingRequirements | None = None,
        preferences: RoutingPreferences | None = None,
        health_tracker: ProviderHealthTracker | None = None,
        unhealthy_after: int = 3,
        skip_degraded_if_possible: bool = True,
    ) -> None:
        self._registry = registry or get_default_provider_registry()
        self._allowed = {self._registry.resolve_name(name) for name in allowed_providers} if allowed_providers else None
        self._provider_kwargs = {
            self._registry.resolve_name(name): dict(values) for name, values in (provider_kwargs or {}).items()
        }
        self._requirements = requirements or RoutingRequirements()
        self._preferences = preferences or RoutingPreferences()
        self._health = health_tracker or ProviderHealthTracker(unhealthy_after=unhealthy_after)
        self._skip_degraded_if_possible = bool(skip_degraded_if_possible)

    def select(self, spec: RequestSpec) -> Iterable[Provider]:
        explicit_provider = str(spec.provider or "").strip().lower()
        if explicit_provider and explicit_provider not in {"auto", "any", "unknown"}:
            resolved = self._registry.resolve_name(explicit_provider)
            if self._allowed is not None and resolved not in self._allowed:
                return []
            if not self._registry.supports_model(resolved, spec.model):
                raise ValueError(f"Model {spec.model!r} is not compatible with provider {resolved!r}")
            return [self._create(resolved, spec)]

        required = self._merge_requirements(spec)
        descriptors = self._registry.find_capable(
            model=spec.model,
            completions=required.completions,
            streaming=required.streaming,
            embeddings=required.embeddings,
            tool_calling=required.tool_calling,
            structured_outputs=required.structured_outputs,
            reasoning=required.reasoning,
            vision_input=required.vision_input,
            audio_input=required.audio_input,
            file_input=required.file_input,
        )
        if self._allowed is not None:
            descriptors = [descriptor for descriptor in descriptors if descriptor.name in self._allowed]
        override_names = self._resolve_provider_overrides(spec)
        if override_names:
            descriptors = [descriptor for descriptor in descriptors if descriptor.name in override_names]
            order = {name: index for index, name in enumerate(override_names)}
            descriptors = sorted(descriptors, key=lambda descriptor: (order.get(descriptor.name, len(order)), descriptor.priority))
        descriptors = self._rank_descriptors(descriptors, self._merge_preferences(spec), preserve_order=bool(override_names))
        return [self._create(descriptor.name, spec) for descriptor in descriptors]

    def record_provider_success(self, provider: Provider | str) -> ProviderHealthStatus:
        return self._health.record_success(self._provider_name(provider))

    def record_provider_failure(self, provider: Provider | str, *, status: int | None = None) -> ProviderHealthStatus:
        return self._health.record_failure(self._provider_name(provider), status=status)

    def get_provider_health(self, provider: str) -> ProviderHealthStatus:
        return self._health.get(self._provider_name(provider))

    def _merge_requirements(self, spec: RequestSpec) -> RoutingRequirements:
        response_format = spec.response_format
        structured = None
        if response_format is not None:
            structured = response_format == "json_object" or isinstance(response_format, (dict, type))
        tool_calling = self._requirements.tool_calling
        if spec.tools:
            tool_calling = True
        streaming = self._requirements.streaming
        if spec.stream:
            streaming = True
        return RoutingRequirements(
            completions=self._requirements.completions,
            streaming=streaming,
            embeddings=self._requirements.embeddings,
            tool_calling=tool_calling,
            structured_outputs=structured if structured is not None else self._requirements.structured_outputs,
            reasoning=self._requirements.reasoning,
            vision_input=self._requirements.vision_input,
            audio_input=self._requirements.audio_input,
            file_input=self._requirements.file_input,
        )

    def _create(self, provider_name: str, spec: RequestSpec) -> Provider:
        kwargs = dict(self._provider_kwargs.get(provider_name, {}))
        provider = self._registry.create(provider_name, model=self._registry.resolve_model(provider_name, spec.model), **kwargs)
        try:
            setattr(provider, "_llm_provider_name", provider_name)
        except Exception:
            pass
        return provider

    def _provider_name(self, provider: Provider | str) -> str:
        if isinstance(provider, str):
            return self._registry.resolve_name(provider)
        tagged = str(getattr(provider, "_llm_provider_name", "")).strip().lower()
        if tagged:
            return self._registry.resolve_name(tagged)
        name = provider.__class__.__name__.removesuffix("Provider").lower()
        return self._registry.resolve_name(name)

    def _rank_descriptors(
        self,
        descriptors: list[Any],
        preferences: RoutingPreferences,
        *,
        preserve_order: bool = False,
    ) -> list[Any]:
        if not descriptors:
            return []
        health_map = {descriptor.name: self._health.get(descriptor.name) for descriptor in descriptors}
        required_tags = {tag for tag in preferences.required_compliance_tags if str(tag).strip()}
        if required_tags:
            descriptors = [
                descriptor
                for descriptor in descriptors
                if required_tags.issubset(set(descriptor.compliance_tags))
            ]
            if not descriptors:
                return []
            health_map = {descriptor.name: self._health.get(descriptor.name) for descriptor in descriptors}
        healthy = [descriptor for descriptor in descriptors if not health_map[descriptor.name].degraded]
        if healthy and self._skip_degraded_if_possible:
            descriptors = healthy
            health_map = {descriptor.name: self._health.get(descriptor.name) for descriptor in descriptors}
        if preserve_order:
            return descriptors
        return sorted(descriptors, key=lambda descriptor: self._descriptor_sort_key(descriptor, health_map, preferences))

    def _descriptor_sort_key(
        self,
        descriptor: Any,
        health_map: dict[str, ProviderHealthStatus],
        preferences: RoutingPreferences,
    ) -> tuple[Any, ...]:
        health = health_map[descriptor.name]
        latency_penalty = self._tier_distance(
            descriptor.latency_tier,
            preferences.preferred_latency_tier,
            _LATENCY_TIER_ORDER,
        )
        cost_penalty = self._tier_distance(
            descriptor.cost_tier,
            preferences.preferred_cost_tier,
            _COST_TIER_ORDER,
        )
        return (
            1 if health.degraded else 0,
            health.consecutive_failures,
            latency_penalty,
            cost_penalty,
            descriptor.priority,
            descriptor.name,
        )

    def _tier_distance(self, current: str, preferred: str | None, ordering: dict[str, int]) -> int:
        if not preferred:
            return 0
        current_norm = str(current or "").strip().lower()
        preferred_norm = str(preferred or "").strip().lower()
        if current_norm not in ordering or preferred_norm not in ordering:
            return int(current_norm != preferred_norm)
        return abs(ordering[current_norm] - ordering[preferred_norm])

    def _merge_preferences(self, spec: RequestSpec) -> RoutingPreferences:
        extra = spec.extra if isinstance(spec.extra, dict) else {}
        latency = str(extra.get("preferred_latency_tier") or self._preferences.preferred_latency_tier or "").strip() or None
        cost = str(extra.get("preferred_cost_tier") or self._preferences.preferred_cost_tier or "").strip() or None
        compliance_raw = extra.get("required_compliance_tags")
        if compliance_raw is None:
            compliance_tags = self._preferences.required_compliance_tags
        else:
            compliance_tags = tuple(str(tag).strip() for tag in compliance_raw if str(tag).strip())
        return RoutingPreferences(
            preferred_latency_tier=latency,
            preferred_cost_tier=cost,
            required_compliance_tags=tuple(compliance_tags),
        )

    def _resolve_provider_overrides(self, spec: RequestSpec) -> list[str]:
        extra = spec.extra if isinstance(spec.extra, dict) else {}
        raw = extra.get("provider_overrides")
        if not raw:
            return []
        resolved: list[str] = []
        for name in raw:
            candidate = str(name or "").strip().lower()
            if not candidate:
                continue
            normalized = self._registry.resolve_name(candidate)
            if self._allowed is not None and normalized not in self._allowed:
                continue
            if normalized not in resolved:
                resolved.append(normalized)
        return resolved


__all__ = [
    "ProviderRouter",
    "StaticRouter",
    "RoutingRequirements",
    "RoutingPreferences",
    "ProviderHealthStatus",
    "ProviderHealthTracker",
    "RegistryRouter",
]
