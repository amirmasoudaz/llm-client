from __future__ import annotations

from typing import Any

import pytest

from llm_client.cache import CacheCore, MetadataCacheStore, SummaryCacheStore
from llm_client.cache.base import BaseCacheBackend
from llm_client.cache.policy import CacheInvalidationMode, CachePolicy
from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec
from tests.llm_client.fakes import ScriptedProvider, ok_result


class _MemoryCacheBackend(BaseCacheBackend):
    name = "none"
    default_collection = "test"

    def __init__(self) -> None:
        self._data: dict[tuple[str | None, str], dict[str, Any]] = {}

    async def ensure_ready(self) -> None:
        return None

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        return (collection or self.default_collection, effective_key) in self._data

    async def read(self, effective_key: str, collection: str | None = None) -> dict[str, Any] | None:
        return self._data.get((collection or self.default_collection, effective_key))

    async def write(
        self,
        effective_key: str,
        response: dict[str, Any],
        model_name: str,
        collection: str | None = None,
    ) -> None:
        payload = dict(response)
        payload["_model_name"] = model_name
        self._data[(collection or self.default_collection, effective_key)] = payload


def _spec() -> RequestSpec:
    return RequestSpec(provider="openai", model="gpt-5-mini", messages=[Message.user("hello")])


def test_cache_policy_exposes_invalidation_semantics() -> None:
    default = CachePolicy.default_response(collection="responses")
    regen = CachePolicy(collection="responses", invalidation=CacheInvalidationMode.REGENERATE)
    rewrite = CachePolicy(collection="responses", invalidation=CacheInvalidationMode.REWRITE)

    assert default.rewrite_cache is False
    assert default.regen_cache is False
    assert regen.regen_cache is True
    assert regen.rewrite_cache is False
    assert rewrite.rewrite_cache is True
    assert rewrite.regen_cache is False
    assert default.should_cache_status(200) is True
    assert default.should_cache_status(500) is False
    assert CachePolicy(cache_errors=True).should_cache_status(500) is True


@pytest.mark.asyncio
async def test_engine_cache_policy_can_force_regeneration() -> None:
    backend = _MemoryCacheBackend()
    cache = CacheCore(backend=backend)
    provider = ScriptedProvider(complete_script=[ok_result("first"), ok_result("second")])
    engine = ExecutionEngine(provider=provider, cache=cache)

    first = await engine.complete(_spec(), cache_policy=CachePolicy.default_response(collection="responses"))
    second = await engine.complete(
        _spec(),
        cache_policy=CachePolicy(
            enabled=True,
            collection="responses",
            invalidation=CacheInvalidationMode.REGENERATE,
        ),
    )

    assert first.content == "first"
    assert second.content == "second"
    assert len(provider.complete_calls) == 2


@pytest.mark.asyncio
async def test_metadata_cache_store_round_trips_records() -> None:
    backend = _MemoryCacheBackend()
    cache = CacheCore(backend=backend)
    store = MetadataCacheStore(cache, default_collection="metadata")

    before = await store.get("model_profile", "gpt-5", scope="catalog:v1")
    assert before is None

    await store.put(
        "model_profile",
        "gpt-5",
        {"context_window": 272000, "provider": "openai"},
        scope="catalog:v1",
        metadata={"source": "unit-test"},
    )
    after = await store.get("model_profile", "gpt-5", scope="catalog:v1")

    assert after is not None
    assert after.kind == "model_profile"
    assert after.identifier == "gpt-5"
    assert after.scope == "catalog:v1"
    assert after.value["provider"] == "openai"
    assert after.metadata["source"] == "unit-test"


@pytest.mark.asyncio
async def test_summary_cache_store_round_trips_records() -> None:
    backend = _MemoryCacheBackend()
    cache = CacheCore(backend=backend)
    store = SummaryCacheStore(
        cache,
        model="gpt-5-mini",
        strategy="heuristic",
        scope="thread",
        default_collection="summaries",
    )

    before = await store.get("thread:1")
    assert before is None

    await store.put("thread:1", "Earlier context summary", metadata={"source": "planner"})
    after = await store.get("thread:1")

    assert after is not None
    assert after.scope == "thread:1"
    assert after.summary == "Earlier context summary"
    assert after.metadata["source"] == "planner"
