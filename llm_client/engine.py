"""
Execution engine for request orchestration, caching, and hooks.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, replace
from typing import Any

from .cache import CacheCore
from .cache.serializers import cache_dict_to_result, result_to_cache_dict
from .cache_keys import CACHE_KEY_SCHEMA_VERSION, embedding_cache_key, request_cache_key
from .cache.policy import CacheInvalidationMode, CachePolicy
from .cancellation import CancelledError
from .errors import failure_to_completion_result, normalize_exception, normalize_provider_failure
from .hooks import HookManager
from .idempotency import IdempotencyTracker
from .providers.base import Provider
from .providers.types import (
    AudioSpeechResult,
    AudioTranscriptionResult,
    BackgroundResponseResult,
    CompactionResult,
    CompletionResult,
    DeepResearchRunResult,
    ConversationItemResource,
    ConversationItemsPage,
    ConversationResource,
    DeletionResult,
    FileContentResult,
    FileResource,
    FilesPage,
    FineTuningJobEventsPage,
    FineTuningJobResult,
    FineTuningJobsPage,
    ImageGenerationResult,
    ModerationResult,
    RealtimeCallResult,
    RealtimeConnection,
    RealtimeClientSecretResult,
    RealtimeTranscriptionSessionResult,
    VectorStoreFileBatchResource,
    StreamEvent,
    StreamEventType,
    VectorStoreFileContentResult,
    VectorStoreFileResource,
    VectorStoreFilesPage,
    VectorStoreResource,
    VectorStoreSearchResult,
    VectorStoresPage,
    WebhookEventResult,
)
from .retry_policy import DEFAULT_RETRYABLE_STATUSES, compute_backoff_delay, is_retryable_status
from .resilience import CircuitBreaker, CircuitBreakerConfig
from .routing import ProviderRouter
from .spec import RequestContext, RequestSpec
from .tools.base import (
    ResponsesChunkingStrategy,
    ResponsesExpirationPolicy,
    ResponsesVectorStoreFileSpec,
)


@dataclass
class RetryConfig:
    attempts: int = 3
    backoff: float = 1.0
    max_backoff: float = 20.0
    retryable_statuses: tuple[int, ...] = DEFAULT_RETRYABLE_STATUSES


@dataclass(frozen=True)
class FailoverPolicy:
    fallback_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    fallback_on_exceptions: bool = True
    fallback_on_circuit_open: bool = True
    max_providers: int | None = None


class ExecutionEngine:
    def __init__(
        self,
        provider: Provider | None = None,
        *,
        router: ProviderRouter | None = None,
        cache: CacheCore | None = None,
        hooks: HookManager | None = None,
        retry: RetryConfig | None = None,
        failover_policy: FailoverPolicy | None = None,
        breaker_config: CircuitBreakerConfig | None = None,
        fallback_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504),
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
        resolved_failover_policy = failover_policy or FailoverPolicy()
        default_fallback_statuses = FailoverPolicy().fallback_statuses
        if fallback_statuses != default_fallback_statuses:
            resolved_failover_policy = replace(
                resolved_failover_policy,
                fallback_statuses=fallback_statuses,
            )
        self.failover_policy = resolved_failover_policy
        self.breaker_config = breaker_config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self.fallback_statuses = self.failover_policy.fallback_statuses
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._idempotency = idempotency_tracker

    async def complete(
        self,
        spec: RequestSpec,
        *,
        context: RequestContext | None = None,
        timeout: float | None = None,
        cache_response: bool = False,
        cache_collection: str | None = None,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        cache_key: str | None = None,
        retry: RetryConfig | None = None,
        idempotency_key: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> CompletionResult:
        ctx = RequestContext.ensure(context)
        start_time = time.monotonic()
        request_timeout = self._resolve_timeout(spec, timeout)
        resolved_cache_policy = self._resolve_cache_policy(
            cache_policy,
            enabled=cache_response,
            collection=cache_collection,
            rewrite_cache=rewrite_cache,
            regen_cache=regen_cache,
        )
        cache_response = resolved_cache_policy.enabled
        cache_collection = resolved_cache_policy.collection
        rewrite_cache = resolved_cache_policy.rewrite_cache
        regen_cache = resolved_cache_policy.regen_cache
        await self.hooks.emit("request.start", {"spec": spec.to_dict()}, ctx)
        diagnostics: dict[str, Any] = {
            "attempts": 0,
            "fallbacks": 0,
            "providers_selected": [],
            "providers_dispatched": [],
            "providers_tried": [],
            "cache_hit": False,
            "cache_key_version": CACHE_KEY_SCHEMA_VERSION,
            "idempotent": False,
            "idempotency_key": None,
            "final_provider": None,
            "final_status": None,
            "final_error": None,
        }

        # Validate request
        from .validation import validate_spec

        validate_spec(spec)

        # Handle idempotency
        idem_key = idempotency_key or spec.extra.get("idempotency_key") or ctx.tags.get("idempotency_key")
        diagnostics["idempotency_key"] = idem_key
        if idem_key and self._idempotency:
            # Check for existing completed result
            if self._idempotency.has_result(idem_key):
                await self.hooks.emit("idempotency.hit", {"key": idem_key}, ctx)
                result = self._idempotency.get_result(idem_key)
                diagnostics.update(
                    {
                        "idempotent": True,
                        "final_provider": diagnostics.get("final_provider") or self._lifecycle_provider_name(spec),
                        "final_status": result.status,
                        "final_error": result.error,
                    }
                )
                await self.hooks.emit("request.diagnostics", dict(diagnostics), ctx)
                await self.hooks.emit(
                    "request.end",
                    self._request_terminal_payload(
                        spec=spec,
                        result=result,
                        diagnostics=diagnostics,
                        start_time=start_time,
                        idempotent=True,
                    ),
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

        providers = self._resolve_providers(spec)
        selected_provider_ids = [self._provider_id(provider) for provider in providers]
        diagnostics["providers_selected"] = list(selected_provider_ids)
        await self.hooks.emit(
            "router.selection",
            {
                "requested_provider": spec.provider,
                "requested_model": spec.model,
                "selected_count": len(providers),
                "selected_providers": selected_provider_ids,
            },
            ctx,
        )
        if not providers:
            await self.hooks.emit(
                "router.empty",
                {
                    "requested_provider": spec.provider,
                    "requested_model": spec.model,
                },
                ctx,
            )
        last_result: CompletionResult | None = None

        for provider in providers:
            provider_id = self._provider_id(provider)
            breaker = self._get_breaker(provider_id)

            if not await breaker.allow():
                await self.hooks.emit("circuit.open", {"provider": provider_id}, ctx)
                self._record_router_failure(provider, status=503)
                circuit_failure = normalize_provider_failure(
                    status=503,
                    message="Circuit open",
                    provider=provider_id,
                    model=spec.model,
                    operation="complete",
                    request_id=ctx.request_id,
                )
                last_result = failure_to_completion_result(circuit_failure, model=spec.model)
                continue

            effective_cache_key = cache_key or self._cache_key(spec, provider, ctx)

            if cache_response and self.cache:
                lookup = await self.cache.lookup(
                    effective_cache_key,
                    rewrite_cache=rewrite_cache,
                    regen_cache=regen_cache,
                    only_ok=resolved_cache_policy.only_ok,
                    collection=cache_collection,
                )
                cache_payload = {
                    "key": effective_cache_key,
                    "effective_key": lookup.effective_key,
                    "collection": lookup.collection,
                    "latency_ms": lookup.latency_ms,
                    "backend": lookup.backend,
                    "cache_key_version": CACHE_KEY_SCHEMA_VERSION,
                    "type": "complete",
                }
                if lookup.error:
                    await self.hooks.emit(
                        "cache.error",
                        {
                            **cache_payload,
                            "error": lookup.error,
                            "can_read_existing": lookup.can_read_existing,
                            "operation": "lookup",
                        },
                        ctx,
                    )
                if lookup.hit and lookup.response:
                    await self.hooks.emit("cache.hit", cache_payload, ctx)
                    diagnostics.update(
                        {
                            "cache_hit": True,
                            "final_provider": provider_id,
                        }
                    )
                    result = self._cached_to_result(lookup.response)
                    await breaker.on_success()
                    diagnostics.update(
                        {
                            "final_status": result.status,
                            "final_error": result.error,
                        }
                    )
                    await self.hooks.emit("request.diagnostics", dict(diagnostics), ctx)
                    await self.hooks.emit(
                        "request.end",
                        self._request_terminal_payload(
                            spec=spec,
                            result=result,
                            diagnostics=diagnostics,
                            start_time=start_time,
                            provider=provider,
                        ),
                        ctx,
                    )
                    return result
                await self.hooks.emit(
                    "cache.miss",
                    {
                        **cache_payload,
                        "can_read_existing": lookup.can_read_existing,
                    },
                    ctx,
                )

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
                diagnostics["attempts"] += 1
                await self.hooks.emit(
                    "request.pre_dispatch",
                    {
                        "attempt": attempt + 1,
                        "provider": provider_id,
                        "timeout": request_timeout,
                    },
                    ctx,
                )
                diagnostics["providers_dispatched"].append(provider_id)
                diagnostics["providers_tried"] = list(diagnostics["providers_dispatched"])

                try:
                    result = await self._call_provider_with_timeout(provider, spec, timeout=request_timeout)
                except Exception as e:
                    failure = normalize_exception(
                        e,
                        provider=provider_id,
                        model=spec.model,
                        operation="complete",
                        request_id=ctx.request_id,
                    )
                    result = failure_to_completion_result(failure, model=spec.model)
                await self.hooks.emit(
                    "request.post_response",
                    {
                        "attempt": attempt + 1,
                        "provider": provider_id,
                        "status": result.status,
                        "ok": result.ok,
                    },
                    ctx,
                )
                last_result = result

                if result.ok:
                    await breaker.on_success()
                    self._record_router_success(provider)
                    break

                await breaker.on_failure(status=last_result.status)
                self._record_router_failure(provider, status=result.status)

                if not is_retryable_status(result.status, retryable_statuses=use_retry.retryable_statuses):
                    break

                if attempt < use_retry.attempts - 1:
                    # Check cancellation before sleeping
                    ctx.cancellation_token.raise_if_cancelled()
                    await asyncio.sleep(
                        compute_backoff_delay(
                            attempt=attempt,
                            base_backoff=current_backoff,
                            max_backoff=use_retry.max_backoff,
                        )
                    )
                    current_backoff = min(current_backoff * 2, use_retry.max_backoff)

            if last_result and last_result.ok:
                if cache_response and self.cache:
                    write_result = await self.cache.store(
                        effective_cache_key,
                        rewrite_cache=rewrite_cache,
                        regen_cache=regen_cache,
                        response=self._result_to_cache(last_result, spec.to_dict()),
                        model_name=spec.model,
                        log_errors=resolved_cache_policy.cache_errors,
                        collection=cache_collection,
                    )
                    cache_payload = {
                        "key": effective_cache_key,
                        "effective_key": write_result.effective_key,
                        "collection": write_result.collection,
                        "latency_ms": write_result.latency_ms,
                        "backend": write_result.backend,
                        "cache_key_version": CACHE_KEY_SCHEMA_VERSION,
                        "type": "complete",
                    }
                    if write_result.error:
                        await self.hooks.emit(
                            "cache.error",
                            {
                                **cache_payload,
                                "error": write_result.error,
                                "operation": "write",
                            },
                            ctx,
                        )
                    elif write_result.written:
                        await self.hooks.emit("cache.write", cache_payload, ctx)
                # Complete idempotency tracking on success
                if idem_key and self._idempotency:
                    self._idempotency.complete_request(idem_key, last_result)
                    await self.hooks.emit("idempotency.complete", {"key": idem_key}, ctx)
                
                await self.hooks.emit("provider.success", {"provider": provider_id}, ctx)
                diagnostics.update(
                    {
                        "final_provider": provider_id,
                        "final_status": last_result.status,
                        "final_error": last_result.error,
                    }
                )
                await self.hooks.emit("request.diagnostics", dict(diagnostics), ctx)
                await self.hooks.emit(
                    "request.end",
                    self._request_terminal_payload(
                        spec=spec,
                        result=last_result,
                        diagnostics=diagnostics,
                        start_time=start_time,
                        provider=provider,
                    ),
                    ctx,
                )
                return last_result

            if last_result:
                failure_payload = _normalized_failure_payload(last_result)
                diagnostics.update(
                    {
                        "final_provider": provider_id,
                        "final_status": last_result.status,
                        "final_error": last_result.error,
                    }
                )
                await self.hooks.emit(
                    "provider.error",
                    {"provider": provider_id, "status": last_result.status, **failure_payload},
                    ctx,
                )
                if self._should_fallback_status(last_result.status):
                    diagnostics["fallbacks"] += 1
                    await self.hooks.emit(
                        "router.fallback",
                        {"provider": provider_id, "status": last_result.status},
                        ctx,
                    )
                    continue
                break

        final = last_result or CompletionResult(status=500, error="No result")
        diagnostics.update(
            {
                "final_status": final.status,
                "final_error": final.error,
            }
        )
        
        # Fail idempotency tracking on error
        if idem_key and self._idempotency:
            self._idempotency.fail_request(idem_key)
            await self.hooks.emit("idempotency.fail", {"key": idem_key}, ctx)
        
        await self.hooks.emit(
            "request.error",
            {"status": final.status, "error": final.error, **_normalized_failure_payload(final), **diagnostics},
            ctx,
        )
        await self.hooks.emit("request.diagnostics", dict(diagnostics), ctx)
        
        await self.hooks.emit(
            "request.end",
            self._request_terminal_payload(
                spec=spec,
                result=final,
                diagnostics=diagnostics,
                start_time=start_time,
            ),
            ctx,
        )
        return final

    async def complete_content(
        self,
        envelope: Any,
        *,
        context: RequestContext | None = None,
        timeout: float | None = None,
        cache_response: bool = False,
        cache_collection: str | None = None,
        rewrite_cache: bool = False,
        regen_cache: bool = False,
        cache_key: str | None = None,
        retry: RetryConfig | None = None,
        idempotency_key: str | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> Any:
        from .content import ContentRequestEnvelope, ContentResponseEnvelope

        if isinstance(envelope, RequestSpec):
            request_spec = envelope
        elif isinstance(envelope, ContentRequestEnvelope):
            request_spec = envelope.to_request_spec()
        elif hasattr(envelope, "to_request_spec"):
            request_spec = envelope.to_request_spec()
        else:
            raise TypeError(f"Unsupported content envelope type: {type(envelope)}")
        request_spec = self._normalize_content_request_spec(request_spec)

        result = await self.complete(
            request_spec,
            context=context,
            timeout=timeout,
            cache_response=cache_response,
            cache_collection=cache_collection,
            rewrite_cache=rewrite_cache,
            regen_cache=regen_cache,
            cache_key=cache_key,
            retry=retry,
            idempotency_key=idempotency_key,
            cache_policy=cache_policy,
        )
        return ContentResponseEnvelope.from_completion_result(result)

    async def stream(
        self,
        spec: RequestSpec,
        *,
        context: RequestContext | None = None,
        timeout: float | None = None,
        idempotency_key: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        ctx = RequestContext.ensure(context)
        start_time = time.monotonic()
        request_timeout = self._resolve_timeout(spec, timeout)
        await self.hooks.emit("stream.start", {"spec": spec.to_dict()}, ctx)
        diagnostics: dict[str, Any] = {
            "providers_selected": [],
            "providers_dispatched": [],
            "providers_tried": [],
            "fallbacks": 0,
            "token_seen": False,
            "idempotent": False,
            "idempotency_key": None,
            "final_provider": None,
            "final_status": None,
            "final_error": None,
        }

        # Validate request
        from .validation import validate_spec

        validate_spec(spec)

        # Handle idempotency for streaming (prevent duplicate streams)
        idem_key = idempotency_key or spec.extra.get("idempotency_key") or ctx.tags.get("idempotency_key")
        diagnostics["idempotency_key"] = idem_key
        if idem_key and self._idempotency:
            if self._idempotency.has_result(idem_key):
                from .content import ensure_completion_result

                await self.hooks.emit("idempotency.hit", {"key": idem_key, "type": "stream"}, ctx)
                cached_result = ensure_completion_result(self._idempotency.get_result(idem_key))
                diagnostics.update(
                    {
                        "idempotent": True,
                        "final_status": cached_result.status,
                        "final_error": cached_result.error,
                    }
                )
                await self.hooks.emit("stream.diagnostics", dict(diagnostics), ctx)
                yield StreamEvent(type=StreamEventType.DONE, data=cached_result)
                await self.hooks.emit(
                    "stream.end",
                    {"status": cached_result.status, "idempotent": True, **diagnostics},
                    ctx,
                )
                return

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

        providers = self._resolve_providers(spec)
        selected_provider_ids = [self._provider_id(provider) for provider in providers]
        diagnostics["providers_selected"] = list(selected_provider_ids)
        await self.hooks.emit(
            "router.selection",
            {
                "requested_provider": spec.provider,
                "requested_model": spec.model,
                "selected_count": len(providers),
                "selected_providers": selected_provider_ids,
                "stream": True,
            },
            ctx,
        )
        if not providers:
            await self.hooks.emit(
                "router.empty",
                {
                    "requested_provider": spec.provider,
                    "requested_model": spec.model,
                    "stream": True,
                },
                ctx,
            )
        token_seen = False

        for provider in providers:
            provider_id = self._provider_id(provider)
            breaker = self._get_breaker(provider_id)

            if not await breaker.allow():
                await self.hooks.emit("circuit.open", {"provider": provider_id}, ctx)
                self._record_router_failure(provider, status=503)
                if self.failover_policy.fallback_on_circuit_open:
                    diagnostics["fallbacks"] += 1
                    continue
                circuit_failure = normalize_provider_failure(
                    status=503,
                    message="Circuit open",
                    provider=provider_id,
                    model=spec.model,
                    operation="stream",
                    request_id=ctx.request_id,
                )
                diagnostics.update({"final_provider": provider_id, "final_status": 503, "final_error": "Circuit open"})
                await self.hooks.emit("stream.error", {"provider": provider_id, **circuit_failure.to_dict()}, ctx)
                await self.hooks.emit("stream.diagnostics", dict(diagnostics), ctx)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"status": 503, "error": "Circuit open", "normalized_failure": circuit_failure.to_dict()},
                )
                await self.hooks.emit(
                    "stream.end",
                    {
                        "status": 503,
                        "error": "Circuit open",
                        "provider": provider_id,
                        "model": spec.model,
                        "latency_ms": round((time.monotonic() - start_time) * 1000.0, 3),
                        **diagnostics,
                    },
                    ctx,
                )
                return

            try:
                await self.hooks.emit(
                    "stream.pre_dispatch",
                    {
                        "provider": provider_id,
                        "timeout": request_timeout,
                    },
                    ctx,
                )
                diagnostics["providers_dispatched"].append(provider_id)
                diagnostics["providers_tried"] = list(diagnostics["providers_dispatched"])
                async for event in self._stream_with_timeout(
                    provider.stream(
                    spec.messages,
                    tools=spec.tools,
                    tool_choice=spec.tool_choice,
                    temperature=spec.temperature,
                    max_tokens=spec.max_tokens,
                    response_format=spec.response_format,
                    reasoning_effort=spec.reasoning_effort,
                    reasoning=spec.reasoning,
                    **spec.extra,
                    ),
                    timeout=request_timeout,
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
                        diagnostics["token_seen"] = True

                    await self.hooks.emit(
                        "stream.event",
                        {"provider": provider_id, "type": event.type.value},
                        ctx,
                    )

                    if event.type == StreamEventType.ERROR:
                        await self.hooks.emit(
                            "stream.error",
                            {
                                "provider": provider_id,
                                "data": event.data,
                                **_normalized_failure_payload_from_stream_error(
                                    event.data,
                                    provider=provider_id,
                                    model=spec.model,
                                    operation="stream",
                                    request_id=ctx.request_id,
                                ),
                            },
                            ctx,
                        )
                        status = event.data.get("status", 500) if isinstance(event.data, dict) else 500
                        await breaker.on_failure(status=status)
                        self._record_router_failure(provider, status=status)
                        if not token_seen and self._should_fallback_status(status):
                            diagnostics["fallbacks"] += 1
                            await self.hooks.emit(
                                "router.fallback",
                                {"provider": provider_id, "status": status},
                                ctx,
                            )
                            # Break inner loop to try next provider
                            break
                        diagnostics.update({"final_provider": provider_id, "final_status": status, "final_error": event.data.get("error", "Provider error")})
                        # Fail idempotency tracking on stream error
                        if idem_key and self._idempotency:
                            self._idempotency.fail_request(idem_key)
                            await self.hooks.emit("idempotency.fail", {"key": idem_key, "type": "stream"}, ctx)
                        await self.hooks.emit("stream.diagnostics", dict(diagnostics), ctx)
                        yield StreamEvent(
                            type=StreamEventType.ERROR,
                            data={
                                "status": status,
                                "error": event.data.get("error", "Provider error"),
                                **_normalized_failure_payload_from_stream_error(
                                    event.data,
                                    provider=provider_id,
                                    model=spec.model,
                                    operation="stream",
                                    request_id=ctx.request_id,
                                ),
                            },
                        )
                        await self.hooks.emit(
                            "stream.end",
                            {
                                "status": status,
                                "error": event.data.get("error", "Provider error"),
                                "provider": provider_id,
                                "model": spec.model,
                                "latency_ms": round((time.monotonic() - start_time) * 1000.0, 3),
                                **diagnostics,
                            },
                            ctx,
                        )
                        return

                    if event.type == StreamEventType.DONE:
                        from .content import ensure_completion_result

                        await breaker.on_success()
                        self._record_router_success(provider)
                        final_result = ensure_completion_result(event.data)
                        await self.hooks.emit(
                            "stream.post_response",
                            {
                                "provider": provider_id,
                                "status": final_result.status,
                                "ok": final_result.ok,
                            },
                            ctx,
                        )
                        # Complete idempotency tracking on stream success
                        if idem_key and self._idempotency:
                            self._idempotency.complete_request(idem_key, final_result)
                            await self.hooks.emit("idempotency.complete", {"key": idem_key, "type": "stream"}, ctx)
                        diagnostics.update(
                            {
                                "final_provider": provider_id,
                                "final_status": final_result.status,
                                "final_error": final_result.error,
                            }
                        )
                        await self.hooks.emit("stream.diagnostics", dict(diagnostics), ctx)
                        yield StreamEvent(type=StreamEventType.DONE, data=final_result)
                        await self.hooks.emit(
                            "stream.end",
                            {
                                "status": final_result.status,
                                "provider": provider_id,
                                "model": final_result.model or spec.model,
                                "usage": final_result.usage.to_dict() if final_result.usage else None,
                                "latency_ms": round((time.monotonic() - start_time) * 1000.0, 3),
                                **diagnostics,
                            },
                            ctx,
                        )
                        return

                    yield event
            except Exception as exc:
                await breaker.on_failure()
                self._record_router_failure(provider, status=500)
                failure = normalize_exception(
                    exc,
                    provider=provider_id,
                    model=spec.model,
                    operation="stream",
                    request_id=ctx.request_id,
                )
                await self.hooks.emit(
                    "stream.error",
                    {"provider": provider_id, **failure.to_dict()},
                    ctx,
                )
                if not token_seen and self.failover_policy.fallback_on_exceptions:
                    # Fallback for connection errors etc
                    diagnostics["fallbacks"] += 1
                    await self.hooks.emit("router.fallback", {"provider": provider_id, "error": str(exc)}, ctx)
                    continue

                # Fail idempotency tracking on exception
                if idem_key and self._idempotency:
                    self._idempotency.fail_request(idem_key)
                    await self.hooks.emit("idempotency.fail", {"key": idem_key, "type": "stream"}, ctx)
                diagnostics.update({"final_provider": provider_id, "final_status": 500, "final_error": str(exc)})
                await self.hooks.emit("stream.diagnostics", dict(diagnostics), ctx)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"status": failure.status or 500, "error": failure.message, "normalized_failure": failure.to_dict()},
                )
                await self.hooks.emit(
                    "stream.end",
                    {
                        "status": failure.status or 500,
                        "error": failure.message,
                        "provider": provider_id,
                        "model": spec.model,
                        "latency_ms": round((time.monotonic() - start_time) * 1000.0, 3),
                        **diagnostics,
                    },
                    ctx,
                )
                return

        # Fail idempotency tracking if all providers exhausted
        if idem_key and self._idempotency:
            self._idempotency.fail_request(idem_key)
            await self.hooks.emit("idempotency.fail", {"key": idem_key, "type": "stream"}, ctx)
        diagnostics.update({"final_status": 500, "final_error": "All providers exhausted"})
        await self.hooks.emit("stream.diagnostics", dict(diagnostics), ctx)
        await self.hooks.emit(
            "stream.end",
            {
                "status": 500,
                "error": "All providers exhausted",
                "model": spec.model,
                "latency_ms": round((time.monotonic() - start_time) * 1000.0, 3),
                **diagnostics,
            },
            ctx,
        )

    async def stream_content(
        self,
        envelope: Any,
        *,
        context: RequestContext | None = None,
        timeout: float | None = None,
        idempotency_key: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        from .content import ContentRequestEnvelope, completion_stream_event_to_content_event

        if isinstance(envelope, RequestSpec):
            request_spec = envelope
        elif isinstance(envelope, ContentRequestEnvelope):
            request_spec = envelope.to_request_spec()
        elif hasattr(envelope, "to_request_spec"):
            request_spec = envelope.to_request_spec()
        else:
            raise TypeError(f"Unsupported content envelope type: {type(envelope)}")
        request_spec = self._normalize_content_request_spec(request_spec)

        async for event in self.stream(request_spec, context=context, timeout=timeout, idempotency_key=idempotency_key):
            yield completion_stream_event_to_content_event(event)

    async def moderate(
        self,
        inputs: str | list[str] | list[dict[str, Any]],
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ModerationResult:
        return await self._run_workflow_operation(
            "moderate",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.moderate(inputs, **kwargs),
        )

    async def generate_image(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        return await self._run_workflow_operation(
            "generate_image",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.generate_image(prompt, **kwargs),
        )

    async def edit_image(
        self,
        image: Any,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ImageGenerationResult:
        return await self._run_workflow_operation(
            "edit_image",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.edit_image(image, prompt, **kwargs),
        )

    async def transcribe_audio(
        self,
        file: Any,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AudioTranscriptionResult:
        return await self._run_workflow_operation(
            "transcribe_audio",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.transcribe_audio(file, **kwargs),
        )

    async def translate_audio(
        self,
        file: Any,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AudioTranscriptionResult:
        return await self._run_workflow_operation(
            "translate_audio",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.translate_audio(file, **kwargs),
        )

    async def synthesize_speech(
        self,
        text: str,
        *,
        voice: str,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AudioSpeechResult:
        return await self._run_workflow_operation(
            "synthesize_speech",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.synthesize_speech(text, voice=voice, **kwargs),
        )

    async def create_file(
        self,
        *,
        file: Any,
        purpose: str,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FileResource:
        return await self._run_workflow_operation(
            "create_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_file(file=file, purpose=purpose, **kwargs),
        )

    async def retrieve_file(
        self,
        file_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FileResource:
        return await self._run_workflow_operation(
            "retrieve_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.retrieve_file(file_id, **kwargs),
        )

    async def list_files(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FilesPage:
        return await self._run_workflow_operation(
            "list_files",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.list_files(**kwargs),
        )

    async def delete_file(
        self,
        file_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> DeletionResult:
        return await self._run_workflow_operation(
            "delete_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.delete_file(file_id, **kwargs),
        )

    async def get_file_content(
        self,
        file_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FileContentResult:
        return await self._run_workflow_operation(
            "get_file_content",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.get_file_content(file_id, **kwargs),
        )

    async def create_vector_store(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        file_ids: list[str] | tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        expiration_policy: ResponsesExpirationPolicy | dict[str, Any] | None = None,
        chunking_strategy: ResponsesChunkingStrategy | dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreResource:
        if name is not None:
            kwargs["name"] = name
        if description is not None:
            kwargs["description"] = description
        if file_ids is not None:
            kwargs["file_ids"] = list(file_ids)
        if metadata is not None:
            kwargs["metadata"] = metadata
        if expiration_policy is not None:
            kwargs["expiration_policy"] = expiration_policy
        if chunking_strategy is not None:
            kwargs["chunking_strategy"] = chunking_strategy
        return await self._run_workflow_operation(
            "create_vector_store",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_vector_store(**kwargs),
        )

    async def retrieve_vector_store(
        self,
        vector_store_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreResource:
        return await self._run_workflow_operation(
            "retrieve_vector_store",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.retrieve_vector_store(vector_store_id, **kwargs),
        )

    async def update_vector_store(
        self,
        vector_store_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreResource:
        return await self._run_workflow_operation(
            "update_vector_store",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.update_vector_store(vector_store_id, **kwargs),
        )

    async def delete_vector_store(
        self,
        vector_store_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> DeletionResult:
        return await self._run_workflow_operation(
            "delete_vector_store",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.delete_vector_store(vector_store_id, **kwargs),
        )

    async def list_vector_stores(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoresPage:
        return await self._run_workflow_operation(
            "list_vector_stores",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.list_vector_stores(**kwargs),
        )

    async def search_vector_store(
        self,
        vector_store_id: str,
        *,
        query: str | list[str],
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreSearchResult:
        return await self._run_workflow_operation(
            "search_vector_store",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.search_vector_store(vector_store_id, query=query, **kwargs),
        )

    async def create_fine_tuning_job(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FineTuningJobResult:
        return await self._run_workflow_operation(
            "create_fine_tuning_job",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_fine_tuning_job(**kwargs),
        )

    async def retrieve_fine_tuning_job(
        self,
        job_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FineTuningJobResult:
        return await self._run_workflow_operation(
            "retrieve_fine_tuning_job",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.retrieve_fine_tuning_job(job_id, **kwargs),
        )

    async def cancel_fine_tuning_job(
        self,
        job_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FineTuningJobResult:
        return await self._run_workflow_operation(
            "cancel_fine_tuning_job",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.cancel_fine_tuning_job(job_id, **kwargs),
        )

    async def list_fine_tuning_jobs(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FineTuningJobsPage:
        return await self._run_workflow_operation(
            "list_fine_tuning_jobs",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.list_fine_tuning_jobs(**kwargs),
        )

    async def list_fine_tuning_events(
        self,
        job_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> FineTuningJobEventsPage:
        return await self._run_workflow_operation(
            "list_fine_tuning_events",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.list_fine_tuning_events(job_id, **kwargs),
        )

    async def create_realtime_client_secret(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeClientSecretResult:
        return await self._run_workflow_operation(
            "create_realtime_client_secret",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_realtime_client_secret(**kwargs),
        )

    async def connect_realtime(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeConnection:
        return await self._run_workflow_operation(
            "connect_realtime",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.connect_realtime(**kwargs),
        )

    async def create_realtime_transcription_session(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeTranscriptionSessionResult:
        return await self._run_workflow_operation(
            "create_realtime_transcription_session",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_realtime_transcription_session(**kwargs),
        )

    async def connect_realtime_transcription(
        self,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeConnection:
        return await self._run_workflow_operation(
            "connect_realtime_transcription",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.connect_realtime_transcription(**kwargs),
        )

    async def create_realtime_call(
        self,
        sdp: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeCallResult:
        return await self._run_workflow_operation(
            "create_realtime_call",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_realtime_call(sdp, **kwargs),
        )

    async def accept_realtime_call(
        self,
        call_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeCallResult:
        return await self._run_workflow_operation(
            "accept_realtime_call",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.accept_realtime_call(call_id, **kwargs),
        )

    async def reject_realtime_call(
        self,
        call_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeCallResult:
        return await self._run_workflow_operation(
            "reject_realtime_call",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.reject_realtime_call(call_id, **kwargs),
        )

    async def hangup_realtime_call(
        self,
        call_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeCallResult:
        return await self._run_workflow_operation(
            "hangup_realtime_call",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.hangup_realtime_call(call_id, **kwargs),
        )

    async def refer_realtime_call(
        self,
        call_id: str,
        *,
        target_uri: str,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> RealtimeCallResult:
        return await self._run_workflow_operation(
            "refer_realtime_call",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.refer_realtime_call(call_id, target_uri=target_uri, **kwargs),
        )

    async def unwrap_webhook(
        self,
        payload: str | bytes,
        headers: Any,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        secret: str | None = None,
    ) -> WebhookEventResult:
        return await self._run_workflow_operation(
            "unwrap_webhook",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.unwrap_webhook(payload, headers, secret=secret),
        )

    async def verify_webhook_signature(
        self,
        payload: str | bytes,
        headers: Any,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        secret: str | None = None,
        tolerance: int = 300,
    ) -> bool:
        return await self._run_workflow_operation(
            "verify_webhook_signature",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.verify_webhook_signature(
                payload,
                headers,
                secret=secret,
                tolerance=tolerance,
            ),
        )

    async def create_vector_store_file(
        self,
        vector_store_id: str,
        *,
        file_id: str,
        attributes: dict[str, str | float | bool] | None = None,
        chunking_strategy: ResponsesChunkingStrategy | dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        if attributes is not None:
            kwargs["attributes"] = attributes
        if chunking_strategy is not None:
            kwargs["chunking_strategy"] = chunking_strategy
        return await self._run_workflow_operation(
            "create_vector_store_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_vector_store_file(vector_store_id, file_id=file_id, **kwargs),
        )

    async def upload_vector_store_file(
        self,
        vector_store_id: str,
        *,
        file: Any,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        return await self._run_workflow_operation(
            "upload_vector_store_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.upload_vector_store_file(vector_store_id, file=file, **kwargs),
        )

    async def list_vector_store_files(
        self,
        vector_store_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFilesPage:
        return await self._run_workflow_operation(
            "list_vector_store_files",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.list_vector_store_files(vector_store_id, **kwargs),
        )

    async def retrieve_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        return await self._run_workflow_operation(
            "retrieve_vector_store_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.retrieve_vector_store_file(vector_store_id, file_id, **kwargs),
        )

    async def update_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        return await self._run_workflow_operation(
            "update_vector_store_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.update_vector_store_file(vector_store_id, file_id, **kwargs),
        )

    async def delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> DeletionResult:
        return await self._run_workflow_operation(
            "delete_vector_store_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.delete_vector_store_file(vector_store_id, file_id, **kwargs),
        )

    async def get_vector_store_file_content(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileContentResult:
        return await self._run_workflow_operation(
            "get_vector_store_file_content",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.get_vector_store_file_content(vector_store_id, file_id, **kwargs),
        )

    async def poll_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        return await self._run_workflow_operation(
            "poll_vector_store_file",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.poll_vector_store_file(vector_store_id, file_id, **kwargs),
        )

    async def create_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        *,
        file_id: str,
        attributes: dict[str, str | float | bool] | None = None,
        chunking_strategy: ResponsesChunkingStrategy | dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        if attributes is not None:
            kwargs["attributes"] = attributes
        if chunking_strategy is not None:
            kwargs["chunking_strategy"] = chunking_strategy
        return await self._run_workflow_operation(
            "create_vector_store_file_and_poll",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_vector_store_file_and_poll(vector_store_id, file_id=file_id, **kwargs),
        )

    async def upload_vector_store_file_and_poll(
        self,
        vector_store_id: str,
        *,
        file: Any,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileResource:
        return await self._run_workflow_operation(
            "upload_vector_store_file_and_poll",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.upload_vector_store_file_and_poll(vector_store_id, file=file, **kwargs),
        )

    async def create_vector_store_file_batch(
        self,
        vector_store_id: str,
        *,
        file_ids: list[str] | tuple[str, ...] | None = None,
        files: list[ResponsesVectorStoreFileSpec | dict[str, Any]] | tuple[ResponsesVectorStoreFileSpec | dict[str, Any], ...] | None = None,
        attributes: dict[str, str | float | bool] | None = None,
        chunking_strategy: ResponsesChunkingStrategy | dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        if file_ids is not None:
            kwargs["file_ids"] = list(file_ids)
        if files is not None:
            kwargs["files"] = list(files)
        if attributes is not None:
            kwargs["attributes"] = attributes
        if chunking_strategy is not None:
            kwargs["chunking_strategy"] = chunking_strategy
        return await self._run_workflow_operation(
            "create_vector_store_file_batch",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_vector_store_file_batch(vector_store_id, **kwargs),
        )

    async def retrieve_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        return await self._run_workflow_operation(
            "retrieve_vector_store_file_batch",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.retrieve_vector_store_file_batch(vector_store_id, batch_id, **kwargs),
        )

    async def cancel_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        return await self._run_workflow_operation(
            "cancel_vector_store_file_batch",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.cancel_vector_store_file_batch(vector_store_id, batch_id, **kwargs),
        )

    async def poll_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        return await self._run_workflow_operation(
            "poll_vector_store_file_batch",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.poll_vector_store_file_batch(vector_store_id, batch_id, **kwargs),
        )

    async def list_vector_store_file_batch_files(
        self,
        vector_store_id: str,
        batch_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFilesPage:
        return await self._run_workflow_operation(
            "list_vector_store_file_batch_files",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.list_vector_store_file_batch_files(vector_store_id, batch_id, **kwargs),
        )

    async def create_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        *,
        file_ids: list[str] | tuple[str, ...] | None = None,
        files: list[ResponsesVectorStoreFileSpec | dict[str, Any]] | tuple[ResponsesVectorStoreFileSpec | dict[str, Any], ...] | None = None,
        attributes: dict[str, str | float | bool] | None = None,
        chunking_strategy: ResponsesChunkingStrategy | dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        if file_ids is not None:
            kwargs["file_ids"] = list(file_ids)
        if files is not None:
            kwargs["files"] = list(files)
        if attributes is not None:
            kwargs["attributes"] = attributes
        if chunking_strategy is not None:
            kwargs["chunking_strategy"] = chunking_strategy
        return await self._run_workflow_operation(
            "create_vector_store_file_batch_and_poll",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_vector_store_file_batch_and_poll(vector_store_id, **kwargs),
        )

    async def upload_vector_store_file_batch_and_poll(
        self,
        vector_store_id: str,
        *,
        files: list[Any] | tuple[Any, ...],
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> VectorStoreFileBatchResource:
        return await self._run_workflow_operation(
            "upload_vector_store_file_batch_and_poll",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.upload_vector_store_file_batch_and_poll(vector_store_id, files=files, **kwargs),
        )

    async def clarify_deep_research_task(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "clarify_deep_research_task",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.clarify_deep_research_task(prompt, **kwargs),
        )

    async def rewrite_deep_research_prompt(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "rewrite_deep_research_prompt",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.rewrite_deep_research_prompt(prompt, **kwargs),
        )

    async def respond_with_web_search(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "respond_with_web_search",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.respond_with_web_search(prompt, **kwargs),
        )

    async def respond_with_file_search(
        self,
        prompt: str,
        *,
        vector_store_ids: list[str] | tuple[str, ...],
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "respond_with_file_search",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.respond_with_file_search(
                prompt,
                vector_store_ids=vector_store_ids,
                **kwargs,
            ),
        )

    async def respond_with_code_interpreter(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "respond_with_code_interpreter",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.respond_with_code_interpreter(prompt, **kwargs),
        )

    async def respond_with_remote_mcp(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "respond_with_remote_mcp",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.respond_with_remote_mcp(prompt, **kwargs),
        )

    async def respond_with_connector(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "respond_with_connector",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.respond_with_connector(prompt, **kwargs),
        )

    async def start_deep_research(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "start_deep_research",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.start_deep_research(prompt, **kwargs),
        )

    async def run_deep_research(
        self,
        prompt: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> DeepResearchRunResult:
        return await self._run_workflow_operation(
            "run_deep_research",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.run_deep_research(prompt, **kwargs),
        )

    async def retrieve_background_response(
        self,
        response_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> BackgroundResponseResult:
        return await self._run_workflow_operation(
            "retrieve_background_response",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.retrieve_background_response(response_id, **kwargs),
        )

    async def cancel_background_response(
        self,
        response_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> BackgroundResponseResult:
        return await self._run_workflow_operation(
            "cancel_background_response",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.cancel_background_response(response_id, **kwargs),
        )

    async def wait_background_response(
        self,
        response_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        poll_interval: float = 2.0,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> BackgroundResponseResult:
        return await self._run_workflow_operation(
            "wait_background_response",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.wait_background_response(
                response_id,
                poll_interval=poll_interval,
                timeout=timeout,
                **kwargs,
            ),
        )

    async def stream_background_response(
        self,
        response_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        starting_after: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._run_stream_workflow_operation(
            "stream_background_response",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.stream_background_response(
                response_id,
                starting_after=starting_after,
                **kwargs,
            ),
        ):
            yield event

    async def create_conversation(
        self,
        *,
        items: Any = None,
        metadata: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        return await self._run_workflow_operation(
            "create_conversation",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_conversation(items=items, metadata=metadata, **kwargs),
        )

    async def retrieve_conversation(
        self,
        conversation_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        return await self._run_workflow_operation(
            "retrieve_conversation",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.retrieve_conversation(conversation_id, **kwargs),
        )

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        return await self._run_workflow_operation(
            "update_conversation",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.update_conversation(conversation_id, metadata=metadata, **kwargs),
        )

    async def delete_conversation(
        self,
        conversation_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        return await self._run_workflow_operation(
            "delete_conversation",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.delete_conversation(conversation_id, **kwargs),
        )

    async def create_conversation_items(
        self,
        conversation_id: str,
        *,
        items: Any,
        include: list[str] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ConversationItemsPage:
        return await self._run_workflow_operation(
            "create_conversation_items",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.create_conversation_items(
                conversation_id,
                items=items,
                include=include,
                **kwargs,
            ),
        )

    async def list_conversation_items(
        self,
        conversation_id: str,
        *,
        after: str | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        order: str | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ConversationItemsPage:
        return await self._run_workflow_operation(
            "list_conversation_items",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.list_conversation_items(
                conversation_id,
                after=after,
                include=include,
                limit=limit,
                order=order,
                **kwargs,
            ),
        )

    async def retrieve_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        *,
        include: list[str] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ConversationItemResource:
        return await self._run_workflow_operation(
            "retrieve_conversation_item",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.retrieve_conversation_item(
                conversation_id,
                item_id,
                include=include,
                **kwargs,
            ),
        )

    async def delete_conversation_item(
        self,
        conversation_id: str,
        item_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ConversationResource:
        return await self._run_workflow_operation(
            "delete_conversation_item",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.delete_conversation_item(conversation_id, item_id, **kwargs),
        )

    async def compact_response_context(
        self,
        *,
        messages: Any = None,
        model: str | None = None,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        provider_name: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompactionResult:
        return await self._run_workflow_operation(
            "compact_response_context",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.compact_response_context(
                messages=messages,
                model=model,
                instructions=instructions,
                previous_response_id=previous_response_id,
                **kwargs,
            ),
        )

    async def submit_mcp_approval_response(
        self,
        *,
        previous_response_id: str,
        approval_request_id: str,
        approve: bool,
        tools: list[Any] | None = None,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return await self._run_workflow_operation(
            "submit_mcp_approval_response",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.submit_mcp_approval_response(
                previous_response_id=previous_response_id,
                approval_request_id=approval_request_id,
                approve=approve,
                tools=tools,
                **kwargs,
            ),
        )

    async def delete_response(
        self,
        response_id: str,
        *,
        provider_name: str | None = None,
        model: str | None = None,
        context: RequestContext | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> DeletionResult:
        return await self._run_workflow_operation(
            "delete_response",
            provider_name=provider_name,
            model=model,
            context=context,
            timeout=timeout,
            call=lambda provider: provider.delete_response(response_id, **kwargs),
        )

    def _normalize_content_request_spec(self, spec: RequestSpec) -> RequestSpec:
        provider_name = str(spec.provider or "").strip().lower()
        if provider_name in {"", "unknown", "auto", "any"}:
            return spec
        try:
            from .provider_registry import get_default_provider_registry

            resolved = get_default_provider_registry().get(provider_name).name
        except Exception:
            resolved = "unknown"
        if resolved == provider_name:
            return spec
        return replace(spec, provider=resolved)

    def _select_providers(self, spec: RequestSpec) -> Iterable[Provider]:
        if self.router:
            return self.router.select(spec)
        if self.provider is None:
            return []
        return [self.provider]

    def _resolve_providers(self, spec: RequestSpec) -> list[Provider]:
        providers = list(self._select_providers(spec))
        max_providers = self.failover_policy.max_providers
        if max_providers is not None and max_providers >= 0:
            return providers[:max_providers]
        return providers

    def _record_router_success(self, provider: Provider) -> None:
        if self.router and hasattr(self.router, "record_provider_success"):
            self.router.record_provider_success(provider)

    def _record_router_failure(self, provider: Provider, *, status: int | None = None) -> None:
        if self.router and hasattr(self.router, "record_provider_failure"):
            self.router.record_provider_failure(provider, status=status)

    def _provider_id(self, provider: Provider) -> str:
        model_name = getattr(provider, "model_name", "")
        return f"{provider.__class__.__name__}:{model_name}"

    def _workflow_spec(self, *, provider_name: str | None, model: str | None) -> RequestSpec:
        return RequestSpec(
            provider=provider_name or "auto",
            model=model,
            messages=[],
        )

    async def _run_workflow_operation(
        self,
        operation: str,
        *,
        provider_name: str | None,
        model: str | None,
        context: RequestContext | None,
        timeout: float | None,
        call: Any,
    ) -> Any:
        ctx = RequestContext.ensure(context)
        spec = self._workflow_spec(provider_name=provider_name, model=model)
        providers = self._resolve_providers(spec)
        selected_provider_ids = [self._provider_id(provider) for provider in providers]
        await self.hooks.emit(
            "workflow.start",
            {
                "operation": operation,
                "requested_provider": provider_name,
                "requested_model": model,
                "selected_providers": selected_provider_ids,
            },
            ctx,
        )
        if not providers:
            error = ValueError(f"No provider available for workflow operation {operation!r}")
            await self.hooks.emit(
                "workflow.error",
                {"operation": operation, "error": str(error), "selected_providers": selected_provider_ids},
                ctx,
            )
            raise error

        last_error: Exception | None = None
        for provider in providers:
            try:
                result = await self._await_with_timeout(call(provider), timeout=timeout)
                await self.hooks.emit(
                    "workflow.end",
                    {
                        "operation": operation,
                        "provider": self._provider_id(provider),
                        "resolved_provider": self._lifecycle_provider_name(spec, provider),
                    },
                    ctx,
                )
                return result
            except NotImplementedError as exc:
                last_error = exc
                continue
            except Exception as exc:
                await self.hooks.emit(
                    "workflow.error",
                    {
                        "operation": operation,
                        "provider": self._provider_id(provider),
                        "resolved_provider": self._lifecycle_provider_name(spec, provider),
                        "error": str(exc),
                    },
                    ctx,
                )
                raise

        error = last_error or NotImplementedError(f"No provider supports workflow operation {operation!r}")
        await self.hooks.emit(
            "workflow.error",
            {"operation": operation, "error": str(error), "selected_providers": selected_provider_ids},
            ctx,
        )
        raise error

    async def _run_stream_workflow_operation(
        self,
        operation: str,
        *,
        provider_name: str | None,
        model: str | None,
        context: RequestContext | None,
        timeout: float | None,
        call: Any,
    ) -> AsyncIterator[StreamEvent]:
        ctx = RequestContext.ensure(context)
        spec = self._workflow_spec(provider_name=provider_name, model=model)
        providers = self._resolve_providers(spec)
        selected_provider_ids = [self._provider_id(provider) for provider in providers]
        await self.hooks.emit(
            "workflow.start",
            {
                "operation": operation,
                "requested_provider": provider_name,
                "requested_model": model,
                "selected_providers": selected_provider_ids,
                "stream": True,
            },
            ctx,
        )
        if not providers:
            error = ValueError(f"No provider available for workflow stream operation {operation!r}")
            await self.hooks.emit(
                "workflow.error",
                {"operation": operation, "error": str(error), "selected_providers": selected_provider_ids, "stream": True},
                ctx,
            )
            raise error

        last_error: Exception | None = None
        for provider in providers:
            try:
                async for event in self._stream_with_timeout(call(provider), timeout=timeout):
                    yield event
                await self.hooks.emit(
                    "workflow.end",
                    {
                        "operation": operation,
                        "provider": self._provider_id(provider),
                        "resolved_provider": self._lifecycle_provider_name(spec, provider),
                        "stream": True,
                    },
                    ctx,
                )
                return
            except NotImplementedError as exc:
                last_error = exc
                continue
            except Exception as exc:
                await self.hooks.emit(
                    "workflow.error",
                    {
                        "operation": operation,
                        "provider": self._provider_id(provider),
                        "resolved_provider": self._lifecycle_provider_name(spec, provider),
                        "error": str(exc),
                        "stream": True,
                    },
                    ctx,
                )
                raise

        error = last_error or NotImplementedError(f"No provider supports workflow stream operation {operation!r}")
        await self.hooks.emit(
            "workflow.error",
            {"operation": operation, "error": str(error), "selected_providers": selected_provider_ids, "stream": True},
            ctx,
        )
        raise error

    def _lifecycle_provider_name(self, spec: RequestSpec, provider: Provider | None = None) -> str | None:
        requested = str(spec.provider or "").strip()
        if requested and requested.lower() not in {"auto", "any", "unknown"}:
            return requested
        if provider is not None:
            from .request_builders import infer_provider_name

            inferred = infer_provider_name(provider)
            return inferred if inferred != "unknown" else None
        return requested or None

    @staticmethod
    def _serialize_usage(usage: Any) -> dict[str, Any] | None:
        if usage is None:
            return None
        if hasattr(usage, "to_dict"):
            return usage.to_dict()
        if isinstance(usage, dict):
            return dict(usage)
        return None

    def _request_terminal_payload(
        self,
        *,
        spec: RequestSpec,
        result: CompletionResult,
        diagnostics: dict[str, Any],
        start_time: float,
        provider: Provider | None = None,
        idempotent: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": result.status,
            "latency_ms": int((time.monotonic() - start_time) * 1000),
            "provider": self._lifecycle_provider_name(spec, provider),
            "model": result.model or spec.model,
            "usage": self._serialize_usage(getattr(result, "usage", None)),
            **diagnostics,
        }
        if idempotent is not None:
            payload["idempotent"] = idempotent
        return payload

    def _get_breaker(self, provider_id: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider_id)
        if breaker is None:
            breaker = CircuitBreaker(self.breaker_config)
            self._breakers[provider_id] = breaker
        return breaker

    def _should_fallback_status(self, status: int) -> bool:
        return status in self.failover_policy.fallback_statuses

    def _cache_key(self, spec: RequestSpec, provider: Provider, ctx: RequestContext | None = None) -> str:
        return request_cache_key(
            spec,
            provider=self._provider_id(provider),
            tenant_id=ctx.tenant_id if ctx else None,
        )

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

    async def _call_provider_with_timeout(
        self,
        provider: Provider,
        spec: RequestSpec,
        *,
        timeout: float | None,
    ) -> CompletionResult:
        if timeout is None:
            return await self._call_provider(provider, spec)
        try:
            return await asyncio.wait_for(self._call_provider(provider, spec), timeout=timeout)
        except asyncio.TimeoutError:
            failure = normalize_provider_failure(
                status=408,
                message=f"Request timed out after {timeout:.3f}s",
                provider=self._provider_id(provider),
                model=spec.model or getattr(provider, "model_name", "unknown"),
                operation="complete",
            )
            return failure_to_completion_result(
                failure,
                model=spec.model or getattr(provider, "model_name", "unknown"),
            )

    async def _await_with_timeout(self, awaitable: Any, *, timeout: float | None) -> Any:
        if timeout is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=timeout)

    async def _stream_with_timeout(
        self,
        stream: AsyncIterator[StreamEvent],
        *,
        timeout: float | None,
    ) -> AsyncIterator[StreamEvent]:
        if timeout is None:
            async for event in stream:
                yield event
            return

        iterator = stream.__aiter__()
        while True:
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                failure = normalize_provider_failure(
                    status=408,
                    message=f"Stream timed out after {timeout:.3f}s",
                    operation="stream",
                )
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    data={"status": 408, "error": f"Stream timed out after {timeout:.3f}s", "normalized_failure": failure.to_dict()},
                )
                return
            yield event

    def _resolve_timeout(self, spec: RequestSpec, explicit_timeout: float | None) -> float | None:
        if explicit_timeout is not None:
            return explicit_timeout
        extra = spec.extra if isinstance(spec.extra, dict) else {}
        timeout_value = extra.get("timeout_seconds", extra.get("timeout"))
        if timeout_value in (None, "", 0):
            return None
        try:
            resolved = float(timeout_value)
        except (TypeError, ValueError):
            return None
        return resolved if resolved > 0 else None

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
        cache_policy: CachePolicy | None = None,
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
        resolved_cache_policy = self._resolve_cache_policy(
            cache_policy,
            enabled=cache_response,
            collection=cache_collection,
            rewrite_cache=False,
            regen_cache=False,
        )
        cache_response = resolved_cache_policy.enabled
        cache_collection = resolved_cache_policy.collection
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
            cache_key = self._embed_cache_key(inputs_list, provider=self.provider, ctx=ctx, extra=kwargs)
            lookup = await self.cache.lookup(
                cache_key,
                rewrite_cache=resolved_cache_policy.rewrite_cache,
                regen_cache=resolved_cache_policy.regen_cache,
                only_ok=resolved_cache_policy.only_ok,
                collection=cache_collection,
            )
            cache_payload = {
                "key": cache_key,
                "effective_key": lookup.effective_key,
                "collection": lookup.collection,
                "latency_ms": lookup.latency_ms,
                "backend": lookup.backend,
                "cache_key_version": CACHE_KEY_SCHEMA_VERSION,
                "type": "embed",
            }
            if lookup.error:
                await self.hooks.emit(
                    "cache.error",
                    {
                        **cache_payload,
                        "error": lookup.error,
                        "can_read_existing": lookup.can_read_existing,
                        "operation": "lookup",
                    },
                    ctx,
                )
            if lookup.hit and lookup.response:
                await self.hooks.emit("cache.hit", cache_payload, ctx)
                return self._cached_to_embedding_result(lookup.response)
            await self.hooks.emit(
                "cache.miss",
                {
                    **cache_payload,
                    "can_read_existing": lookup.can_read_existing,
                },
                ctx,
            )

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
                write_result = await self.cache.store(
                    cache_key,
                    rewrite_cache=resolved_cache_policy.rewrite_cache,
                    regen_cache=resolved_cache_policy.regen_cache,
                    response=self._embedding_to_cache(result),
                    model_name=getattr(result, "model", None) or getattr(provider, "model_name", "unknown"),
                    log_errors=resolved_cache_policy.cache_errors,
                    collection=cache_collection,
                )
                cache_payload = {
                    "key": cache_key,
                    "effective_key": write_result.effective_key,
                    "collection": write_result.collection,
                    "latency_ms": write_result.latency_ms,
                    "backend": write_result.backend,
                    "cache_key_version": CACHE_KEY_SCHEMA_VERSION,
                    "type": "embed",
                }
                if write_result.error:
                    await self.hooks.emit(
                        "cache.error",
                        {
                            **cache_payload,
                            "error": write_result.error,
                            "operation": "write",
                        },
                        ctx,
                    )
                elif write_result.written:
                    await self.hooks.emit("cache.write", cache_payload, ctx)

            return result

        except Exception as exc:
            await breaker.on_failure()
            failure = normalize_exception(
                exc,
                provider=provider_id,
                model=getattr(provider, "model_name", None),
                operation="embed",
                request_id=ctx.request_id,
            )
            await self.hooks.emit("embed.error", failure.to_dict(), ctx)
            raise

    def _embed_cache_key(
        self,
        inputs: list[str],
        *,
        provider: Provider | None = None,
        ctx: RequestContext | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Generate cache key for embedding inputs."""
        resolved_provider = provider or self.provider
        model_name = getattr(resolved_provider, "model_name", None) or "default"
        return embedding_cache_key(
            model=model_name,
            inputs=inputs,
            provider=self._provider_id(resolved_provider) if resolved_provider else None,
            tenant_id=ctx.tenant_id if ctx else None,
            extra=dict(extra or {}),
        )

    @staticmethod
    def _resolve_cache_policy(
        cache_policy: CachePolicy | None,
        *,
        enabled: bool,
        collection: str | None,
        rewrite_cache: bool,
        regen_cache: bool,
    ) -> CachePolicy:
        if cache_policy is not None:
            return cache_policy
        if regen_cache:
            invalidation = CacheInvalidationMode.REGENERATE
        elif rewrite_cache:
            invalidation = CacheInvalidationMode.REWRITE
        else:
            invalidation = CacheInvalidationMode.USE_EXISTING
        return CachePolicy(
            enabled=enabled,
            collection=collection,
            invalidation=invalidation,
        )

    @staticmethod
    def _embedding_to_cache(result: Any) -> dict[str, Any]:
        """Convert EmbeddingResult to cache-friendly dict."""
        return {
            "error": "OK",
            "status": getattr(result, "status", 200),
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
                    failure = normalize_exception(
                        e,
                        provider=spec.provider,
                        model=spec.model,
                        operation="batch_complete",
                    )
                    return failure_to_completion_result(failure, model=spec.model)

        tasks = [asyncio.create_task(_wrapped(s)) for s in specs]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results


__all__ = ["ExecutionEngine", "FailoverPolicy", "RetryConfig"]


def _normalized_failure_payload(result: CompletionResult) -> dict[str, Any]:
    raw = getattr(result, "raw_response", None)
    if isinstance(raw, dict):
        failure = raw.get("normalized_failure")
        if isinstance(failure, dict):
            return {"normalized_failure": dict(failure)}
    if result.error:
        failure = normalize_provider_failure(status=result.status, message=result.error)
        return {"normalized_failure": failure.to_dict()}
    return {}


def _normalized_failure_payload_from_stream_error(
    data: Any,
    *,
    provider: str | None,
    model: str | None,
    operation: str,
    request_id: str | None,
) -> dict[str, Any]:
    if isinstance(data, dict):
        existing = data.get("normalized_failure")
        if isinstance(existing, dict):
            return {"normalized_failure": dict(existing)}
        status = data.get("status")
        message = str(data.get("error") or "Provider error")
    else:
        status = 500
        message = str(data or "Provider error")
    failure = normalize_provider_failure(
        status=int(status) if status is not None else None,
        message=message,
        provider=provider,
        model=model,
        operation=operation,
        request_id=request_id,
    )
    return {"normalized_failure": failure.to_dict()}
