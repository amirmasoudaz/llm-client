from __future__ import annotations

import asyncio
import json
import os

from llm_client import (
    EngineDiagnosticsRecorder,
    ExecutionEngine,
    HookManager,
    Message,
    OpenAIProvider,
    RequestContext,
    RequestSpec,
    RetryConfig,
    load_env,
)
from llm_client.cache import CacheCore, CachePolicy
from llm_client.cache.base import BaseCacheBackend
from llm_client.idempotency import IdempotencyTracker

load_env()


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
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    provider_name = "openai"
    provider = OpenAIProvider(model=model_name)
    try:
        diagnostics = EngineDiagnosticsRecorder()
        engine = ExecutionEngine(
            provider=provider,
            cache=CacheCore(_InMemoryCacheBackend()),
            retry=RetryConfig(attempts=2, backoff=0.25, max_backoff=0.5),
            idempotency_tracker=IdempotencyTracker(),
            hooks=HookManager([diagnostics]),
        )

        retry_context = RequestContext()
        retry_spec = RequestSpec(
            provider=provider_name,
            model=model_name,
            messages=[Message.user("Explain why retries are useful in LLM infrastructure.")],
        )
        retry_result = await engine.complete(retry_spec, context=retry_context, idempotency_key="cookbook-live-retry")

        idem_first_context = RequestContext()
        idem_second_context = RequestContext()
        idem_spec = RequestSpec(
            provider=provider_name,
            model=model_name,
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
            provider=provider_name,
            model=model_name,
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

        print("\n=== Retry + Idempotency ===\n")
        print(
            json.dumps(
                {
                    "provider": provider_name,
                    "model": model_name,
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
                },
                indent=4,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Cache + Diagnostics ===\n")
        print(
            json.dumps(
                {
                    "cold_result": cold_result.content,
                    "warm_result": warm_result.content,
                    "cache_stats": engine.cache.get_stats().to_dict() if engine.cache else {},
                    "warm_request_diagnostics": (
                        diagnostics.latest_request(warm_context.request_id).payload
                        if diagnostics.latest_request(warm_context.request_id)
                        else {}
                    ),
                },
                indent=4,
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
