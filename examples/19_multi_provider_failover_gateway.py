from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from cookbook_support import (
    build_live_provider,
    build_provider_handle,
    close_provider,
    fail_or_skip,
    print_heading,
    print_json,
    summarize_usage,
)

from llm_client.engine import ExecutionEngine, RetryConfig
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.provider_registry import ProviderCapabilities, ProviderDescriptor, ProviderRegistry
from llm_client.providers.types import CompletionResult, Message
from llm_client.routing import RegistryRouter
from llm_client.spec import RequestContext, RequestSpec


@dataclass
class _InjectedFailureState:
    failures_remaining: int = 1
    last_failure: dict[str, Any] | None = None


class _FailureInjectedProvider:
    def __init__(self, inner: Any, *, label: str, state: _InjectedFailureState) -> None:
        self._inner = inner
        self._label = label
        self._state = state
        self.name = getattr(inner, "name", label)
        self.model_name = getattr(inner, "model_name", None)
        self.model = getattr(inner, "model", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def complete(self, *args: Any, **kwargs: Any) -> CompletionResult:
        if self._state.failures_remaining > 0:
            self._state.failures_remaining -= 1
            self._state.last_failure = {
                "provider": self._label,
                "status": 503,
                "error": "Injected gateway demo failure: primary provider unavailable",
            }
            return CompletionResult(
                status=503,
                error="Injected gateway demo failure: primary provider unavailable",
                model=self.model_name or self.model,
            )
        return await self._inner.complete(*args, **kwargs)


def _build_registry(primary: Any, secondary: Any, *, primary_family: str, secondary_family: str) -> ProviderRegistry:
    registry = ProviderRegistry()
    capabilities = ProviderCapabilities(completions=True, streaming=True, embeddings=False, tool_calling=True)
    registry.register(
        ProviderDescriptor(
            name="gateway_primary",
            default_model=primary.model_name or primary.model,
            priority=10,
            capabilities=capabilities,
            metadata={"provider_family": primary_family},
            factory=lambda **_: primary,
        )
    )
    registry.register(
        ProviderDescriptor(
            name="gateway_secondary",
            default_model=secondary.model,
            priority=20,
            capabilities=capabilities,
            metadata={"provider_family": secondary_family},
            factory=lambda **_: secondary.provider,
        )
    )
    return registry


def _observed_failover_summary(
    *,
    primary_provider: str,
    primary_model: str,
    secondary_provider: str,
    secondary_model: str,
    request_report: Any,
    secondary_strategy: dict[str, Any],
    injected_failure: dict[str, Any] | None,
) -> str:
    if request_report is None:
        return "Failover summary unavailable."
    fallback_mode = secondary_strategy.get("mode")
    fallback_kind = "cross-provider fallback"
    if fallback_mode == "same_provider_backup":
        fallback_kind = "same-provider backup fallback"
    return (
        f"Primary {primary_provider}:{primary_model} was forced to fail with "
        f"{(injected_failure or {}).get('status', 'unknown')} and the gateway performed one fallback. "
        f"It routed to {secondary_provider}:{secondary_model} via {fallback_kind}, "
        f"completed in {request_report.latency_ms:.0f}ms with status {request_report.status}, "
        f"and used {request_report.usage.total_tokens} total tokens."
    )


async def _probe_handle(handle: Any) -> dict[str, Any]:
    engine = ExecutionEngine(provider=handle.provider)
    result = await engine.complete(
        RequestSpec(
            provider=handle.name,
            model=handle.model,
            messages=[
                Message.system("Reply with exactly: ok"),
                Message.user("ok"),
            ],
            max_tokens=8,
        )
    )
    return {
        "provider": handle.name,
        "model": handle.model,
        "status": result.status,
        "ok": result.ok,
        "error": result.error,
    }


async def _resolve_secondary(primary: Any, configured_secondary: Any) -> tuple[Any, dict[str, Any]]:
    configured_probe = await _probe_handle(configured_secondary)
    if configured_probe["ok"]:
        return configured_secondary, {
            "mode": "configured_secondary",
            "probe": configured_probe,
        }

    backup_model = {
        "openai": "gpt-5-mini",
        "anthropic": "claude-sonnet-4",
        "google": "gemini-2.5-flash",
    }[primary.name]
    backup_handle = build_provider_handle(primary.name, backup_model)
    backup_probe = await _probe_handle(backup_handle)
    if backup_probe["ok"]:
        return backup_handle, {
            "mode": "same_provider_backup",
            "configured_probe": configured_probe,
            "backup_probe": backup_probe,
        }

    await close_provider(backup_handle.provider)
    fail_or_skip(
        "Failover example could not find a working secondary path. "
        f"Configured secondary failed: {configured_probe}. "
        f"Same-provider backup failed: {backup_probe}."
    )
    raise AssertionError("unreachable")


async def main() -> None:
    primary = build_live_provider()
    configured_secondary = build_live_provider(secondary=True)
    state = _InjectedFailureState(failures_remaining=1)
    injected_primary = _FailureInjectedProvider(primary.provider, label="gateway_primary", state=state)
    secondary = configured_secondary
    try:
        secondary, secondary_strategy = await _resolve_secondary(primary, configured_secondary)
        diagnostics = EngineDiagnosticsRecorder()
        lifecycle = LifecycleRecorder()
        registry = _build_registry(
            injected_primary,
            secondary,
            primary_family=primary.name,
            secondary_family=secondary.name,
        )
        router = RegistryRouter(registry=registry)
        engine = ExecutionEngine(
            router=router,
            retry=RetryConfig(attempts=1, backoff=0.0, max_backoff=0.0),
            hooks=HookManager([diagnostics, lifecycle]),
        )
        context = RequestContext(session_id="cookbook-failover-gateway")
        spec = RequestSpec(
            provider="auto",
            model="",
            messages=[
                Message.system(
                    "You are a gateway response assistant. Answer in 3 bullets with labels: Request Path, Failover Behavior, Final Outcome."
                ),
                Message.user(
                    "Explain what happened when the primary provider failed and the gateway routed the request to the fallback model."
                ),
            ],
        )
        result = await engine.complete(spec, context=context)
        diagnostic_snapshot = diagnostics.latest_request(context.request_id)
        request_report = lifecycle.requests.get(context.request_id)
        session_report = lifecycle.sessions.get(context.session_id or "")

        output = {
            "configured_gateway": {
                "providers": [
                    {
                        "name": descriptor.name,
                        "priority": descriptor.priority,
                        "default_model": descriptor.default_model,
                        "provider_family": descriptor.metadata.get("provider_family"),
                    }
                    for descriptor in registry.find_capable(completions=True)
                ],
            },
            "primary": {
                "provider": primary.name,
                "model": primary.model,
                "failure_injected": True,
            },
            "secondary": {
                "provider": secondary.name,
                "model": secondary.model,
            },
            "secondary_strategy": secondary_strategy,
            "gateway_story": {
                "primary_failure": state.last_failure,
                "fallback_provider_selected": request_report.provider if request_report else None,
                "fallback_model_selected": request_report.model if request_report else None,
                "final_status": result.status,
                "request_succeeded": result.ok,
            },
            "observed_failover_summary": _observed_failover_summary(
                primary_provider=primary.name,
                primary_model=primary.model,
                secondary_provider=secondary.name,
                secondary_model=secondary.model,
                request_report=request_report,
                secondary_strategy=secondary_strategy,
                injected_failure=state.last_failure,
            ),
            "request_report": request_report.to_dict() if request_report else None,
            "diagnostics": diagnostic_snapshot.payload if diagnostic_snapshot else None,
            "session_report": session_report.to_dict() if session_report else None,
            "result": {
                "status": result.status,
                "model": result.model,
                "usage": summarize_usage(result.usage),
                "model_explanation": result.content,
            },
        }

        print_heading("Failover Gateway")
        print_json(output)
        if not result.ok:
            fail_or_skip("Gateway fallback request failed. Check the fallback provider path or the provider normalization layer.")
    finally:
        await close_provider(primary.provider)
        if configured_secondary.provider is not primary.provider and configured_secondary.provider is not secondary.provider:
            await close_provider(configured_secondary.provider)
        if secondary.provider is not primary.provider:
            await close_provider(secondary.provider)


if __name__ == "__main__":
    asyncio.run(main())
