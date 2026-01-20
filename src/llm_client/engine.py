"""
Execution engine for request orchestration, caching, and hooks.
"""
from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
import inspect
import time
from typing import Any, AsyncIterator, Dict, Iterable, Optional, Tuple

from .cache import CacheCore
from .serialization import stable_json_dumps
from .hooks import HookManager
from .providers.base import Provider
from .providers.types import CompletionResult, StreamEvent, StreamEventType, ToolCall, Usage
from .resilience import CircuitBreaker, CircuitBreakerConfig
from .routing import ProviderRouter, StaticRouter
from .spec import RequestContext, RequestSpec


@dataclass
class RetryConfig:
    attempts: int = 3
    backoff: float = 1.0
    max_backoff: float = 20.0
    retryable_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504)


class ExecutionEngine:
    def __init__(
        self,
        provider: Optional[Provider] = None,
        *,
        router: Optional[ProviderRouter] = None,
        cache: Optional[CacheCore] = None,
        hooks: Optional[HookManager] = None,
        retry: Optional[RetryConfig] = None,
        breaker_config: Optional[CircuitBreakerConfig] = None,
        fallback_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504),
    ) -> None:
        if provider is None and router is None:
            raise ValueError("ExecutionEngine requires a provider or a router.")
        self.provider = provider
        self.router = router
        if cache is not None:
            self.cache = cache
        elif provider is not None and hasattr(provider, "cache"):
            self.cache = getattr(provider, "cache")
        else:
            self.cache = None
        self.hooks = hooks or HookManager()
        self.retry = retry or RetryConfig()
        self.breaker_config = breaker_config or CircuitBreakerConfig()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_statuses = fallback_statuses

    async def complete(
        self,
        spec: RequestSpec,
        *,
        context: Optional[RequestContext] = None,
        cache_response: bool = False,
        cache_collection: Optional[str] = None,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        cache_key: Optional[str] = None,
        retry: Optional[RetryConfig] = None,
    ) -> CompletionResult:
        ctx = context or RequestContext()
        start_time = time.monotonic()
        await self.hooks.emit("request.start", {"spec": spec.to_dict()}, ctx)

        providers = self._select_providers(spec)
        last_result: Optional[CompletionResult] = None

        for provider in providers:
            provider_id = self._provider_id(provider)
            breaker = self._get_breaker(provider_id)

            if not await breaker.allow():
                await self.hooks.emit("circuit.open", {"provider": provider_id}, ctx)
                last_result = CompletionResult(status=503, error="Circuit open")
                continue

            effective_cache_key = cache_key or self._cache_key(spec, provider)

            if cache_response and self.cache:
                cached, _ = await self.cache.get_cached(
                    effective_cache_key,
                    rewrite_cache=rewrite_cache,
                    regen_cache=regen_cache,
                    only_ok=True,
                    collection=cache_collection,
                )
                if cached:
                    await self.hooks.emit("cache.hit", {"key": effective_cache_key}, ctx)
                    result = self._cached_to_result(cached)
                    await breaker.on_success()
                    await self.hooks.emit(
                        "request.end",
                        {
                            "status": result.status,
                            "latency_ms": int((time.monotonic() - start_time) * 1000),
                        },
                        ctx,
                    )
                    return result
                await self.hooks.emit("cache.miss", {"key": effective_cache_key}, ctx)

            use_retry = retry or self.retry
            current_backoff = use_retry.backoff

            for attempt in range(use_retry.attempts):
                await self.hooks.emit(
                    "request.attempt",
                    {"attempt": attempt + 1, "provider": provider_id},
                    ctx,
                )

                result = await self._call_provider(provider, spec)
                last_result = result

                if result.ok:
                    await breaker.on_success()
                    break

                await breaker.on_failure()

                if result.status not in use_retry.retryable_statuses:
                    break

                if attempt < use_retry.attempts - 1:
                    await asyncio.sleep(current_backoff * random.uniform(0.8, 1.2))
                    current_backoff = min(current_backoff * 2, use_retry.max_backoff)

            if last_result and last_result.ok:
                if cache_response and self.cache:
                    await self.cache.put_cached(
                        effective_cache_key,
                        rewrite_cache=rewrite_cache,
                        regen_cache=regen_cache,
                        response=self._result_to_cache(last_result, spec.to_dict()),
                        model_name=spec.model,
                        log_errors=True,
                        collection=cache_collection,
                    )
                await self.hooks.emit(
                    "provider.success", {"provider": provider_id}, ctx
                )
                await self.hooks.emit(
                    "request.end",
                    {
                        "status": last_result.status,
                        "latency_ms": int((time.monotonic() - start_time) * 1000),
                    },
                    ctx,
                )
                return last_result

            if last_result:
                await self.hooks.emit(
                    "provider.error",
                    {"provider": provider_id, "status": last_result.status},
                    ctx,
                )
                if last_result.status in self.fallback_statuses:
                    await self.hooks.emit(
                        "router.fallback",
                        {"provider": provider_id, "status": last_result.status},
                        ctx,
                    )
                    continue
                break

        final = last_result or CompletionResult(status=500, error="No result")
        await self.hooks.emit(
            "request.end",
            {
                "status": final.status,
                "latency_ms": int((time.monotonic() - start_time) * 1000),
            },
            ctx,
        )
        return final

    async def stream(
        self,
        spec: RequestSpec,
        *,
        context: Optional[RequestContext] = None,
    ) -> AsyncIterator[StreamEvent]:
        ctx = context or RequestContext()
        await self.hooks.emit("stream.start", {"spec": spec.to_dict()}, ctx)

        providers = list(self._select_providers(spec))
        token_seen = False

        for provider in providers:
            provider_id = self._provider_id(provider)
            breaker = self._get_breaker(provider_id)

            if not await breaker.allow():
                await self.hooks.emit("circuit.open", {"provider": provider_id}, ctx)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"status": 503, "error": "Circuit open"},
                )
                continue

            try:
                async for event in provider.stream(
                    spec.messages,
                    tools=spec.tools,
                    tool_choice=spec.tool_choice,
                    temperature=spec.temperature,
                    max_tokens=spec.max_tokens,
                    reasoning_effort=spec.reasoning_effort,
                    reasoning=spec.reasoning,
                    **spec.extra,
                ):
                    if event.type in (
                        StreamEventType.TOKEN,
                        StreamEventType.REASONING,
                        StreamEventType.TOOL_CALL_START,
                        StreamEventType.TOOL_CALL_DELTA,
                        StreamEventType.TOOL_CALL_END,
                    ):
                        token_seen = True

                    await self.hooks.emit(
                        "stream.event",
                        {"provider": provider_id, "type": event.type.value},
                        ctx,
                    )

                    if event.type == StreamEventType.ERROR:
                        await breaker.on_failure()
                        await self.hooks.emit(
                            "stream.error",
                            {"provider": provider_id, "data": event.data},
                            ctx,
                        )
                        status = event.data.get("status", 500) if isinstance(event.data, dict) else 500
                        if not token_seen and status in self.fallback_statuses:
                            await self.hooks.emit(
                                "router.fallback",
                                {"provider": provider_id, "status": status},
                                ctx,
                            )
                            break
                        yield event
                        await self.hooks.emit("stream.end", {"status": status}, ctx)
                        return

                    if event.type == StreamEventType.DONE:
                        await breaker.on_success()
                        yield event
                        await self.hooks.emit("stream.end", {"status": 200}, ctx)
                        return

                    yield event
            except Exception as exc:
                await breaker.on_failure()
                await self.hooks.emit(
                    "stream.error",
                    {"provider": provider_id, "error": str(exc)},
                    ctx,
                )
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"status": 500, "error": str(exc)},
                )
                await self.hooks.emit("stream.end", {"status": 500}, ctx)
                return

        await self.hooks.emit("stream.end", {"status": 500}, ctx)

    def _select_providers(self, spec: RequestSpec) -> Iterable[Provider]:
        if self.router:
            return self.router.select(spec)
        if self.provider is None:
            return []
        return [self.provider]

    def _provider_id(self, provider: Provider) -> str:
        model_name = getattr(provider, "model_name", "")
        return f"{provider.__class__.__name__}:{model_name}"

    def _get_breaker(self, provider_id: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider_id)
        if breaker is None:
            breaker = CircuitBreaker(self.breaker_config)
            self._breakers[provider_id] = breaker
        return breaker

    def _cache_key(self, spec: RequestSpec, provider: Provider) -> str:
        payload = spec.to_dict()
        payload["provider"] = self._provider_id(provider)
        return blake3(stable_json_dumps(payload).encode("utf-8")).hexdigest()

    async def _call_provider(self, provider: Provider, spec: RequestSpec) -> CompletionResult:
        provider_kwargs = dict(spec.extra)
        for key in ("cache_response", "cache_collection", "rewrite_cache", "regen_cache"):
            provider_kwargs.pop(key, None)

        signature = inspect.signature(provider.complete)
        if "attempts" in signature.parameters and "attempts" not in provider_kwargs:
            provider_kwargs["attempts"] = 1
        if "backoff" in signature.parameters and "backoff" not in provider_kwargs:
            provider_kwargs["backoff"] = 0.0

        return await provider.complete(
            spec.messages,
            tools=spec.tools,
            tool_choice=spec.tool_choice,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            response_format=spec.response_format,
            reasoning_effort=spec.reasoning_effort,
            reasoning=spec.reasoning,
            **provider_kwargs,
        )

    def _result_to_cache(self, result: CompletionResult, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "params": params,
            "output": result.content,
            "usage": result.usage.to_dict() if result.usage else {},
            "status": result.status,
            "error": result.error or "OK",
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in (result.tool_calls or [])
            ],
        }

    def _cached_to_result(self, cached: Dict[str, Any]) -> CompletionResult:
        tool_calls = None
        if cached.get("tool_calls"):
            tool_calls = [
                ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
                for tc in cached["tool_calls"]
            ]
        return CompletionResult(
            content=cached.get("output"),
            tool_calls=tool_calls,
            usage=Usage.from_dict(cached.get("usage", {})),
            status=cached.get("status", 200),
            error=cached.get("error") if cached.get("error") != "OK" else None,
        )


__all__ = ["ExecutionEngine", "RetryConfig"]
