from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace

import pytest


def test_cache_namespace_import_does_not_eagerly_load_pg_redis_backend() -> None:
    sys.modules.pop("llm_client.cache", None)
    sys.modules.pop("llm_client.cache.postgres_redis", None)

    cache_module = importlib.import_module("llm_client.cache")

    assert cache_module.CacheCore is not None
    assert "llm_client.cache.postgres_redis" not in sys.modules


def test_persistence_module_imports_without_asyncpg(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("llm_client.persistence", None)
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "asyncpg":
            raise ModuleNotFoundError("No module named 'asyncpg'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    persistence_module = importlib.import_module("llm_client.persistence")

    assert persistence_module.PostgresRepository is not None


@pytest.mark.asyncio
async def test_pg_redis_backend_requires_asyncpg_only_when_used(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_module = importlib.import_module("llm_client.cache.postgres_redis")
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "asyncpg":
            raise ModuleNotFoundError("No module named 'asyncpg'")
        return real_import_module(name, package)

    monkeypatch.setattr(backend_module, "import_module", fake_import_module)

    cache = backend_module.HybridRedisPostgreSQLCache(
        backend_module.HybridCacheConfig(default_table="llm_cache", client_type="completions")
    )

    with pytest.raises(ModuleNotFoundError, match=r"llm-client\[postgres\]"):
        await cache.ensure_ready()


@pytest.mark.asyncio
async def test_pg_redis_backend_degrades_without_redis_package(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_module = importlib.import_module("llm_client.cache.postgres_redis")
    real_import_module = importlib.import_module

    async def fake_create_pool(**kwargs):
        return object()

    class DummyRepository:
        def __init__(self, pool, compress: bool = True, compression_level: int = 6):
            self.pool = pool
            self.compress = compress
            self.compression_level = compression_level
            self.tables: list[str] = []

        async def ensure_table(self, table_name: str) -> None:
            self.tables.append(table_name)

    def fake_import_module(name: str, package: str | None = None):
        if name == "asyncpg":
            return SimpleNamespace(create_pool=fake_create_pool)
        if name in {"redis.asyncio", "redis.exceptions"}:
            raise ModuleNotFoundError("No module named 'redis'")
        return real_import_module(name, package)

    monkeypatch.setattr(backend_module, "import_module", fake_import_module)
    monkeypatch.setattr(backend_module, "PostgresRepository", DummyRepository)

    cache = backend_module.HybridRedisPostgreSQLCache(
        backend_module.HybridCacheConfig(default_table="llm_cache", client_type="completions")
    )

    await cache.ensure_ready()

    assert cache._repo is not None
    assert cache._redis is None
