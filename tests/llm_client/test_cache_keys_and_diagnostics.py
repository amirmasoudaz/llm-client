from __future__ import annotations

from typing import Any

import pytest

from llm_client.cache import CACHE_KEY_SCHEMA_VERSION, CacheCore, request_cache_key
from llm_client.cache.base import BaseCacheBackend
from llm_client.engine import ExecutionEngine
from llm_client.hooks import HookManager
from llm_client.providers.types import Message
from llm_client.spec import RequestContext, RequestSpec
from llm_client.tools.base import Tool
from tests.llm_client.fakes import ScriptedProvider, ok_result


class _CollectingHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, event: str, payload: dict[str, Any], context: RequestContext) -> None:
        _ = context
        self.events.append((event, dict(payload)))


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


def _base_spec(**overrides: Any) -> RequestSpec:
    payload: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-5-mini",
        "messages": [Message.user("hello world")],
    }
    payload.update(overrides)
    return RequestSpec(**payload)


async def _echo_tool(query: str) -> str:
    return query


def _search_tool() -> Tool:
    return Tool(
        name="search",
        description="Search",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=_echo_tool,
    )


@pytest.mark.asyncio
async def test_request_cache_keys_change_when_request_shape_changes() -> None:
    base = _base_spec()
    model_changed = _base_spec(model="gpt-5")
    content_changed = _base_spec(messages=[Message.user("hello there")])
    schema_changed = _base_spec(
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }
    )
    tools_changed = _base_spec(tools=[_search_tool()])

    keys = {
        request_cache_key(base),
        request_cache_key(model_changed),
        request_cache_key(content_changed),
        request_cache_key(schema_changed),
        request_cache_key(tools_changed),
    }

    assert len(keys) == 5
    assert base.cache_key() == request_cache_key(base)


@pytest.mark.asyncio
async def test_engine_emits_cache_diagnostics_for_completion_paths() -> None:
    backend = _MemoryCacheBackend()
    cache = CacheCore(backend=backend)
    hook = _CollectingHook()
    provider = ScriptedProvider(complete_script=[ok_result("cached once")])
    engine = ExecutionEngine(provider=provider, cache=cache, hooks=HookManager([hook]))

    first = await engine.complete(_base_spec(), cache_response=True, cache_collection="responses")
    second = await engine.complete(_base_spec(), cache_response=True, cache_collection="responses")

    assert first.ok is True
    assert second.ok is True
    assert second.content == "cached once"
    assert len(provider.complete_calls) == 1

    miss_payload = next(payload for event, payload in hook.events if event == "cache.miss")
    write_payload = next(payload for event, payload in hook.events if event == "cache.write")
    hit_payload = next(payload for event, payload in hook.events if event == "cache.hit")

    assert miss_payload["cache_key_version"] == CACHE_KEY_SCHEMA_VERSION
    assert miss_payload["collection"] == "responses"
    assert miss_payload["type"] == "complete"
    assert "effective_key" in miss_payload
    assert "latency_ms" in miss_payload

    assert write_payload["cache_key_version"] == CACHE_KEY_SCHEMA_VERSION
    assert write_payload["backend"] == "none"
    assert write_payload["collection"] == "responses"

    assert hit_payload["cache_key_version"] == CACHE_KEY_SCHEMA_VERSION
    assert hit_payload["backend"] == "none"
    assert hit_payload["effective_key"] == write_payload["effective_key"]


@pytest.mark.asyncio
async def test_engine_emits_cache_diagnostics_for_embedding_paths() -> None:
    backend = _MemoryCacheBackend()
    cache = CacheCore(backend=backend)
    hook = _CollectingHook()
    provider = ScriptedProvider()
    engine = ExecutionEngine(provider=provider, cache=cache, hooks=HookManager([hook]))

    first = await engine.embed(["alpha"], cache_response=True, cache_collection="embeddings", dimensions=1536)
    second = await engine.embed(["alpha"], cache_response=True, cache_collection="embeddings", dimensions=1536)

    assert len(first.embeddings) == 1
    assert len(second.embeddings) == 1
    assert len(provider.embed_calls) == 1

    miss_payload = next(payload for event, payload in hook.events if event == "cache.miss" and payload["type"] == "embed")
    write_payload = next(payload for event, payload in hook.events if event == "cache.write" and payload["type"] == "embed")
    hit_payload = next(payload for event, payload in hook.events if event == "cache.hit" and payload["type"] == "embed")

    assert miss_payload["cache_key_version"] == CACHE_KEY_SCHEMA_VERSION
    assert miss_payload["collection"] == "embeddings"
    assert write_payload["collection"] == "embeddings"
    assert hit_payload["effective_key"] == write_payload["effective_key"]
