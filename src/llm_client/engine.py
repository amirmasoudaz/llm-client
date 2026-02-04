"""
Execution engine for request orchestration, caching, and hooks.
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from typing import Any

from blake3 import blake3

from .cache import CacheCore
from .cache.serializers import cache_dict_to_result, result_to_cache_dict
from .cancellation import CancelledError
from .hashing import content_hash
from .hooks import HookManager
from .idempotency import IdempotencyTracker
from .providers.base import Provider
from .providers.types import CompletionResult, StreamEvent, StreamEventType
from .resilience import CircuitBreaker, CircuitBreakerConfig
from .routing import ProviderRouter
from .spec import RequestContext, RequestSpec


@dataclass
class RetryConfig:
    attempts: int = 3
    backoff: float = 1.0
    max_backoff: float = 20.0
    retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)


class ExecutionEngine:
    def __init__(
        self,
        provider: Provider | None = None,
        *,
        router: ProviderRouter | None = None,
        cache: CacheCore | None = None,
        hooks: HookManager | None = None,
        retry: RetryConfig | None = None,
        breaker_config: CircuitBreakerConfig | None = None,
        fallback_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
        max_concurrency: int = 20,
        idempotency_tracker: IdempotencyTracker | None = None,
    ) -> None:
        if provider is None and router is None:
            raise ValueError("ExecutionEngine requires a provider or a router.")
        self.provider = provider
        self.router = router
        self.cache: CacheCore | None
        if cache is not None:
            self.cache = cache
        elif provider is not None and hasattr(provider, "cache"):
            self.cache = provider.cache
        else:
            self.cache = None
        self.hooks = hooks or HookManager()
        self.retry = retry or RetryConfig()
        self.breaker_config = breaker_config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self.fallback_statuses = fallback_statuses
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._idempotency = idempotency_tracker

    async def complete(
        self,
        spec: RequestSpec,
        *,
        context: RequestContext | None = None,
        cache_response: bool = False,
        cache_collection: str | None = None,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        cache_key: str | None = None,
        retry: RetryConfig | None = None,
        idempotency_key: str | None = None,
    ) -> CompletionResult:
        ctx = RequestContext.ensure(context)
        start_time = time.monotonic()
        await self.hooks.emit("request.start", {"spec": spec.to_dict()}, ctx)

        # Validate request
        from .validation import validate_spec

        validate_spec(spec)

        # Handle idempotency
        idem_key = idempotency_key or spec.extra.get("idempotency_key") or ctx.tags.get("idempotency_key")
        if idem_key and self._idempotency:
            # Check for existing completed result
            if self._idempotency.has_result(idem_key):
                await self.hooks.emit("idempotency.hit", {"key": idem_key}, ctx)
                result = self._idempotency.get_result(idem_key)
                await self.hooks.emit(
                    "request.end",
                    {
                        "status": result.status,
                        "latency_ms": int((time.monotonic() - start_time) * 1000),
                        "idempotent": True,
                    },
                    ctx,
                )
                return result
            
            # Check if request is in-flight
            if not self._idempotency.can_start(idem_key):
                await self.hooks.emit("idempotency.conflict", {"key": idem_key}, ctx)
                return CompletionResult(
                    status=409,
                    error=f"Request with idempotency key '{idem_key}' is already in flight",
                    model=spec.model or "unknown",
                )
            
            # Start tracking this request
            self._idempotency.start_request(idem_key)
            await self.hooks.emit("idempotency.start", {"key": idem_key}, ctx)

        providers = self._select_providers(spec)
        last_result: CompletionResult | None = None

        for provider in providers:
            provider_id = self._provider_id(provider)
            breaker = self._get_breaker(provider_id)

            if not await breaker.allow():
                await self.hooks.emit("circuit.open", {"provider": provider_id}, ctx)
                last_result = CompletionResult(status=503, error="Circuit open")
                continue

            effective_cache_key = cache_key or self._cache_key(spec, provider, ctx)

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
                # Check cancellation before each attempt
                ctx.cancellation_token.raise_if_cancelled()
                
                await self.hooks.emit(
                    "request.attempt",
                    {"attempt": attempt + 1, "provider": provider_id},
                    ctx,
                )

                try:
                    result = await self._call_provider(provider, spec)
                except Exception as e:
                    result = CompletionResult(
                        status=500,
                        error=f"Internal provider error: {str(e)}",
                        model=spec.model or "unknown",
                    )
                last_result = result

                if result.ok:
                    await breaker.on_success()
                    break

                await breaker.on_failure()

                if result.status not in use_retry.retryable_statuses:
                    break

                if attempt < use_retry.attempts - 1:
                    # Check cancellation before sleeping
                    ctx.cancellation_token.raise_if_cancelled()
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
                # Complete idempotency tracking on success
                if idem_key and self._idempotency:
                    self._idempotency.complete_request(idem_key, last_result)
                    await self.hooks.emit("idempotency.complete", {"key": idem_key}, ctx)
                
                await self.hooks.emit("provider.success", {"provider": provider_id}, ctx)
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
        
        # Fail idempotency tracking on error
        if idem_key and self._idempotency:
            self._idempotency.fail_request(idem_key)
            await self.hooks.emit("idempotency.fail", {"key": idem_key}, ctx)
        
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
        context: RequestContext | None = None,
        idempotency_key: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        ctx = RequestContext.ensure(context)
        await self.hooks.emit("stream.start", {"spec": spec.to_dict()}, ctx)

        # Validate request
        from .validation import validate_spec

        validate_spec(spec)

        # Handle idempotency for streaming (prevent duplicate streams)
        idem_key = idempotency_key or spec.extra.get("idempotency_key") or ctx.tags.get("idempotency_key")
        if idem_key and self._idempotency:
            # Check if stream is already in-flight (can't return cached results for streams)
            if self._idempotency.is_pending(idem_key):
                await self.hooks.emit("idempotency.conflict", {"key": idem_key, "type": "stream"}, ctx)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"status": 409, "error": f"Stream with idempotency key '{idem_key}' is already in flight"},
                )
                return
            
            # Start tracking this stream
            self._idempotency.start_request(idem_key)
            await self.hooks.emit("idempotency.start", {"key": idem_key, "type": "stream"}, ctx)

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
                    # Check cancellation between stream events
                    ctx.cancellation_token.raise_if_cancelled()
                    
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
                            # Break inner loop to try next provider
                            break
                        # Fail idempotency tracking on stream error
                        if idem_key and self._idempotency:
                            self._idempotency.fail_request(idem_key)
                            await self.hooks.emit("idempotency.fail", {"key": idem_key, "type": "stream"}, ctx)
                        yield StreamEvent(
                            type=StreamEventType.ERROR,
                            data={"status": status, "error": event.data.get("error", "Provider error")},
                        )
                        await self.hooks.emit("stream.end", {"status": status}, ctx)
                        return

                    if event.type == StreamEventType.DONE:
                        await breaker.on_success()
                        # Complete idempotency tracking on stream success
                        if idem_key and self._idempotency:
                            self._idempotency.complete_request(idem_key)
                            await self.hooks.emit("idempotency.complete", {"key": idem_key, "type": "stream"}, ctx)
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
                if not token_seen:
                    # Fallback for connection errors etc
                    await self.hooks.emit("router.fallback", {"provider": provider_id, "error": str(exc)}, ctx)
                    continue

                # Fail idempotency tracking on exception
                if idem_key and self._idempotency:
                    self._idempotency.fail_request(idem_key)
                    await self.hooks.emit("idempotency.fail", {"key": idem_key, "type": "stream"}, ctx)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"status": 500, "error": str(exc)},
                )
                await self.hooks.emit("stream.end", {"status": 500}, ctx)
                return

        # Fail idempotency tracking if all providers exhausted
        if idem_key and self._idempotency:
            self._idempotency.fail_request(idem_key)
            await self.hooks.emit("idempotency.fail", {"key": idem_key, "type": "stream"}, ctx)
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

    def _cache_key(self, spec: RequestSpec, provider: Provider, ctx: RequestContext | None = None) -> str:
        payload = spec.to_dict()
        payload["provider"] = self._provider_id(provider)
        # Include tenant_id for tenant isolation
        if ctx and ctx.tenant_id:
            payload["tenant_id"] = ctx.tenant_id
        return content_hash(payload)

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

    # Cache serialization methods imported from cache.serializers
    _result_to_cache = staticmethod(result_to_cache_dict)
    _cached_to_result = staticmethod(cache_dict_to_result)

    async def embed(
        self,
        inputs: str | Iterable[str],
        *,
        context: RequestContext | None = None,
        timeout: float | None = None,
        cache_response: bool = False,
        cache_collection: str | None = None,
        **kwargs: Any,
    ) -> Any:  # Returns EmbeddingResult (avoiding circular import issues if possible, or use Any)
        """
        Generate embeddings for the given inputs.

        Args:
            inputs: List of strings to embed
            context: Request context
            timeout: Request timeout
            cache_response: Whether to cache the response
            cache_collection: Optional cache collection name
            **kwargs: Additional provider-specific arguments

        Returns:
            The embedding result from the provider
        """
        ctx = RequestContext.ensure(context)
        if isinstance(inputs, str):
            inputs_list = [inputs]
        else:
            inputs_list = list(inputs)

        await self.hooks.emit("embed.start", {"count": len(inputs_list)}, ctx)

        # Validate inputs
        from .validation import validate_embedding_inputs

        validate_embedding_inputs(inputs_list)

        # Handle cache lookup
        cache_key = None
        if cache_response and self.cache:
            cache_key = self._embed_cache_key(inputs_list)
            cached, _ = await self.cache.get_cached(
                cache_key,
                only_ok=True,
                collection=cache_collection,
            )
            if cached:
                await self.hooks.emit("cache.hit", {"key": cache_key, "type": "embed"}, ctx)
                return self._cached_to_embedding_result(cached)
            await self.hooks.emit("cache.miss", {"key": cache_key, "type": "embed"}, ctx)

        # Select provider (default to self.provider if no router logic for embeddings yet)
        provider = self.provider
        if not provider and self.router:
            # Select first available provider for embeddings
            # Embedding routing could be enhanced in the future
            provider = self.router.select(RequestSpec(messages=[]), context=ctx)

        if not provider:
            raise ValueError("No provider available for embeddings")

        provider_id = self._provider_id(provider)
        breaker = self._get_breaker(provider_id)

        if not await breaker.allow():
            await self.hooks.emit("circuit.open", {"provider": provider_id}, ctx)
            raise RuntimeError("Circuit open for provider")

        try:
            call = provider.embed(inputs_list, **kwargs)
            result = await asyncio.wait_for(call, timeout=timeout) if timeout else await call
            await breaker.on_success()
            await self.hooks.emit("embed.end", {"status": 200}, ctx)

            # Cache the result
            if cache_response and self.cache and cache_key:
                await self.cache.put_cached(
                    cache_key,
                    response=self._embedding_to_cache(result),
                    collection=cache_collection,
                )

            return result

        except Exception as exc:
            await breaker.on_failure()
            await self.hooks.emit("embed.error", {"error": str(exc)}, ctx)
            raise

    def _embed_cache_key(self, inputs: list[str]) -> str:
        """Generate cache key for embedding inputs."""
        payload = {
            "type": "embedding",
            "inputs": inputs,
            "model": self.provider.model_name if self.provider else "default",
        }
        return content_hash(payload)

    @staticmethod
    def _embedding_to_cache(result: Any) -> dict[str, Any]:
        """Convert EmbeddingResult to cache-friendly dict."""
        return {
            "embeddings": result.embeddings,
            "usage": {
                "input_tokens": result.usage.input_tokens if result.usage else 0,
                "output_tokens": result.usage.output_tokens if result.usage else 0,
                "total_tokens": result.usage.total_tokens if result.usage else 0,
            } if result.usage else None,
            "model": result.model,
        }

    @staticmethod
    def _cached_to_embedding_result(cached: dict[str, Any]) -> Any:
        """Convert cached dict back to EmbeddingResult."""
        from .providers.types import EmbeddingResult, Usage

        usage = None
        if cached.get("usage"):
            usage = Usage(
                input_tokens=cached["usage"].get("input_tokens", 0),
                output_tokens=cached["usage"].get("output_tokens", 0),
                total_tokens=cached["usage"].get("total_tokens", 0),
            )

        return EmbeddingResult(
            embeddings=cached["embeddings"],
            usage=usage,
            model=cached.get("model"),
        )

    async def batch_complete(
        self,
        specs: Iterable[RequestSpec],
        *,
        max_concurrency: int | None = None,
        **kwargs: Any,
    ) -> list[CompletionResult]:
        """
        Execute a batch of requests concurrently.

        Args:
            specs: List of request specifications
            max_concurrency: Override default concurrency limit for this batch (lower only effectively)
            **kwargs: Arguments passed to complete() (e.g. cache settings)

        Returns:
            List of CompletionResults in the same order as specs.
        """
        semaphore = self._semaphore
        if max_concurrency is not None:
            # Create a new local semaphore if specific concurrency requested
            semaphore = asyncio.Semaphore(max_concurrency)

        async def _wrapped(spec: RequestSpec) -> CompletionResult:
            async with semaphore:
                try:
                    return await self.complete(spec, **kwargs)
                except Exception as e:
                    return CompletionResult(status=500, error=str(e), model=spec.model or "unknown")

        tasks = [asyncio.create_task(_wrapped(s)) for s in specs]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results


__all__ = ["ExecutionEngine", "RetryConfig"]
