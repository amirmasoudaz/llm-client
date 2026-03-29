from __future__ import annotations

import asyncio

from cookbook_support import build_live_provider, close_provider, print_heading, print_json

from llm_client.cache import CacheCore, CachePolicy
from llm_client.cache.base import BaseCacheBackend
from llm_client.engine import ExecutionEngine, RetryConfig
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager
from llm_client.idempotency import IdempotencyTracker
from llm_client.providers.types import Message
from llm_client.spec import RequestContext, RequestSpec


class _InMemoryCacheBackend(BaseCacheBackend):
    name = "fs"
    default_collection = "cookbook"

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], dict[str, object]] = {}

    async def ensure_ready(self) -> None:
        return None

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        return (effective_key, collection or self.default_collection) in self._entries

    async def read(self, effective_key: str, collection: str | None = None) -> dict[str, object] | None:
        return self._entries.get((effective_key, collection or self.default_collection))

    async def write(
        self,
        effective_key: str,
        response: dict[str, object],
        model_name: str,
        collection: str | None = None,
    ) -> None:
        _ = model_name
        self._entries[(effective_key, collection or self.default_collection)] = dict(response)


async def main() -> None:
    handle = build_live_provider()
    try:
        diagnostics = EngineDiagnosticsRecorder()
        engine = ExecutionEngine(
            provider=handle.provider,
            cache=CacheCore(_InMemoryCacheBackend()),
            retry=RetryConfig(attempts=2, backoff=0.25, max_backoff=0.5),
            idempotency_tracker=IdempotencyTracker(),
            hooks=HookManager([diagnostics]),
        )

        retry_context = RequestContext()
        retry_spec = RequestSpec(
            provider=handle.name,
            model=handle.model,
            messages=[Message.user("Explain why retries are useful in LLM infrastructure.")],
        )
        retry_result = await engine.complete(retry_spec, context=retry_context, idempotency_key="cookbook-live-retry")

        idem_first_context = RequestContext()
        idem_second_context = RequestContext()
        idem_spec = RequestSpec(
            provider=handle.name,
            model=handle.model,
            messages=[Message.user("Answer with the phrase: idempotency prevents duplicate work.")],
        )
        idem_first = await engine.complete(
            idem_spec,
            context=idem_first_context,
            idempotency_key="cookbook-live-idempotency",
        )
        idem_second = await engine.complete(
            idem_spec,
            context=idem_second_context,
            idempotency_key="cookbook-live-idempotency",
        )

        cache_spec = RequestSpec(
            provider=handle.name,
            model=handle.model,
            messages=[Message.user("Summarize cache hits in one sentence.")],
        )
        cold_context = RequestContext()
        warm_context = RequestContext()
        cold_result = await engine.complete(
            cache_spec,
            context=cold_context,
            cache_policy=CachePolicy.default_response(collection="cookbook-live"),
        )
        warm_result = await engine.complete(
            cache_spec,
            context=warm_context,
            cache_policy=CachePolicy.default_response(collection="cookbook-live"),
        )

        print_heading("Retry + Idempotency")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "retry_configured_attempts": 2,
                "retry_observed_attempts": (
                    diagnostics.latest_request(retry_context.request_id).payload.get("attempts")
                    if diagnostics.latest_request(retry_context.request_id)
                    else None
                ),
                "retry_result": retry_result.content,
                "idempotent_same_content": idem_first.content == idem_second.content,
                "idempotent_second_request": (
                    diagnostics.latest_request(idem_second_context.request_id).payload
                    if diagnostics.latest_request(idem_second_context.request_id)
                    else {}
                ),
            }
        )

        print_heading("Cache + Diagnostics")
        print_json(
            {
                "cold_result": cold_result.content,
                "warm_result": warm_result.content,
                "cache_stats": engine.cache.get_stats().to_dict() if engine.cache else {},
                "warm_request_diagnostics": (
                    diagnostics.latest_request(warm_context.request_id).payload
                    if diagnostics.latest_request(warm_context.request_id)
                    else {}
                ),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
