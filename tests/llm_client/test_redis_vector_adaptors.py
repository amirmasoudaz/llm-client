from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_client import ExecutionContext
from llm_client.adapters import (
    AdaptorExecutionOptions,
    AdaptorRuntime,
    AdaptorTimeoutError,
    QdrantVectorAdaptor,
    RedisGetRequest,
    RedisHashDeleteRequest,
    RedisHashGetRequest,
    RedisHashSetRequest,
    RedisKVAdaptor,
    RedisSafetyError,
    RedisSetRequest,
    VectorDeleteRequest,
    VectorPoint,
    VectorSearchRequest,
    VectorSafetyError,
    VectorUpsertRequest,
)
from llm_client.budgets import InMemoryLedgerWriter, Ledger, LedgerEventType


class _CollectingBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


class _FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, str | bytes] = {}
        self.hashes: dict[str, dict[str, str | bytes]] = {}
        self.fail_once = False
        self.delay_seconds = 0.0

    async def get(self, key: str) -> str | bytes | None:
        if self.fail_once:
            self.fail_once = False
            raise AdaptorTimeoutError("temporary timeout", backend="redis")
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return self.values.get(key)

    async def set(self, key: str, value: str | bytes, *, ex: int | None = None, nx: bool = False, xx: bool = False) -> bool:
        if nx and key in self.values:
            return False
        if xx and key not in self.values:
            return False
        _ = ex
        self.values[key] = value
        return True

    async def delete(self, key: str) -> int:
        return 1 if self.values.pop(key, None) is not None else 0

    async def hget(self, key: str, field_name: str) -> str | bytes | None:
        return self.hashes.get(key, {}).get(field_name)

    async def hset(self, key: str, field_name: str, value: str | bytes) -> int:
        self.hashes.setdefault(key, {})[field_name] = value
        return 1

    async def hdel(self, key: str, field_name: str) -> int:
        if field_name in self.hashes.get(key, {}):
            del self.hashes[key][field_name]
            return 1
        return 0


class _FakeQdrantSearchPoint:
    def __init__(self, point_id: str, score: float, payload: dict[str, Any]) -> None:
        self.id = point_id
        self.score = score
        self.payload = payload


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, Any]] = {}
        self.upsert_calls: list[dict[str, Any]] = []
        self.search_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []

    async def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    async def create_collection(self, *, collection_name: str, vectors_config: Any, timeout: float | None = None) -> None:
        self.collections[collection_name] = {"vectors_config": vectors_config, "points": []}
        _ = timeout

    async def upsert(self, *, collection_name: str, points: list[Any], wait: bool = True, timeout: float | None = None) -> None:
        self.upsert_calls.append(
            {
                "collection_name": collection_name,
                "points": points,
                "wait": wait,
                "timeout": timeout,
            }
        )
        self.collections.setdefault(collection_name, {"points": []})["points"] = list(points)

    async def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        query_filter: Any,
        timeout: float | None = None,
    ) -> list[_FakeQdrantSearchPoint]:
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "limit": limit,
                "with_payload": with_payload,
                "query_filter": query_filter,
                "timeout": timeout,
            }
        )
        return [
            _FakeQdrantSearchPoint("doc-1", 0.93, {"kind": "runbook"}),
            _FakeQdrantSearchPoint("doc-2", 0.81, {"kind": "note"}),
        ][:limit]

    async def delete(
        self,
        *,
        collection_name: str,
        points_selector: Any,
        wait: bool = True,
        timeout: float | None = None,
    ) -> None:
        self.delete_calls.append(
            {
                "collection_name": collection_name,
                "points_selector": points_selector,
                "wait": wait,
                "timeout": timeout,
            }
        )


def _runtime_with_tracking() -> tuple[AdaptorRuntime, _CollectingBus, Ledger, InMemoryLedgerWriter]:
    bus = _CollectingBus()
    writer = InMemoryLedgerWriter()
    ledger = Ledger(writer)
    ctx = ExecutionContext(scope_id="scope-1", principal_id="principal-1", session_id="session-1")
    runtime = AdaptorRuntime(
        execution_context=ctx,
        event_bus=bus,
        ledger=ledger,
        retry_attempts=1,
        base_backoff_seconds=0.0,
    )
    return runtime, bus, ledger, writer


@pytest.mark.asyncio
async def test_redis_adaptor_enforces_prefix_size_and_delete_controls() -> None:
    runtime, bus, ledger, writer = _runtime_with_tracking()
    _ = ledger
    client = _FakeRedisClient()
    adaptor = RedisKVAdaptor(client, key_prefix="tenant-a", allow_delete=True, runtime=runtime)

    set_result = await adaptor.set(RedisSetRequest(key="profile:1", value="active", ttl_seconds=30))
    get_result = await adaptor.get(RedisGetRequest(key="profile:1"))
    hash_set = await adaptor.hset(RedisHashSetRequest(key="profile:1", field_name="tier", value="gold"))
    hash_get = await adaptor.hget(RedisHashGetRequest(key="profile:1", field_name="tier"))
    hash_delete = await adaptor.hdel(RedisHashDeleteRequest(key="profile:1", field_name="tier"))

    assert set_result.key == "tenant-a:profile:1"
    assert get_result.value == "active"
    assert hash_set.written is True
    assert hash_get.value == "gold"
    assert hash_delete.deleted_count == 1
    assert any(event.data["kind"] == "adaptor.start" for event in bus.events)
    connector_events = await writer.list_events(event_type=LedgerEventType.CONNECTOR_USAGE)
    assert connector_events[0].connector_name == "redis"

    with pytest.raises(RedisSafetyError, match="max_value_bytes"):
        await adaptor.set(RedisSetRequest(key="profile:2", value="x" * 70000))

    readonly = RedisKVAdaptor(client, key_prefix="tenant-a", read_only=True)
    with pytest.raises(RedisSafetyError, match="read-only"):
        await readonly.set(RedisSetRequest(key="profile:3", value="blocked"))


@pytest.mark.asyncio
async def test_redis_adaptor_retries_retryable_failures_and_normalizes_timeouts() -> None:
    runtime, _bus, _ledger, _writer = _runtime_with_tracking()
    client = _FakeRedisClient()
    client.values["tenant-a:session:1"] = "warm"
    client.fail_once = True
    adaptor = RedisKVAdaptor(client, key_prefix="tenant-a", runtime=runtime)

    result = await adaptor.get(RedisGetRequest(key="session:1"))
    assert result.value == "warm"

    client.delay_seconds = 0.05
    with pytest.raises(AdaptorTimeoutError):
        await adaptor.get(
            RedisGetRequest(
                key="session:1",
                options=AdaptorExecutionOptions(timeout_seconds=0.001),
            )
        )


@pytest.mark.asyncio
async def test_qdrant_adaptor_supports_create_upsert_search_delete_and_tracking() -> None:
    runtime, bus, _ledger, writer = _runtime_with_tracking()
    client = _FakeQdrantClient()
    adaptor = QdrantVectorAdaptor(client, runtime=runtime, allow_delete=True, max_search_limit=5)

    upsert_result = await adaptor.upsert(
        VectorUpsertRequest(
            collection="mission_docs",
            points=[VectorPoint(point_id="doc-1", vector=[0.1, 0.2], payload={"kind": "runbook"})],
            create_if_missing=True,
        )
    )
    search_result = await adaptor.search(
        VectorSearchRequest(collection="mission_docs", query_vector=[0.1, 0.2], limit=2)
    )
    delete_result = await adaptor.delete(
        VectorDeleteRequest(collection="mission_docs", point_ids=["doc-1"])
    )

    assert upsert_result.upserted_count == 1
    assert search_result.matches[0].point_id == "doc-1"
    assert delete_result.deleted_count == 1
    assert client.collections["mission_docs"]["points"]
    assert any(event.data["backend"] == "qdrant" for event in bus.events)
    connector_events = await writer.list_events(event_type=LedgerEventType.CONNECTOR_USAGE)
    assert any(event.connector_name == "qdrant" for event in connector_events)


@pytest.mark.asyncio
async def test_qdrant_adaptor_enforces_search_limit_and_delete_guard() -> None:
    adaptor = QdrantVectorAdaptor(_FakeQdrantClient(), max_search_limit=3)

    with pytest.raises(VectorSafetyError, match="max_search_limit"):
        await adaptor.search(
            VectorSearchRequest(collection="docs", query_vector=[0.2, 0.4], limit=4)
        )

    with pytest.raises(VectorSafetyError, match="allow_delete=True"):
        await adaptor.delete(VectorDeleteRequest(collection="docs", point_ids=["doc-1"]))
