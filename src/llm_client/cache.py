from __future__ import annotations

import asyncio
import json
import os
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Literal

import aiofiles
import aiohttp
import asyncpg
import redis.asyncio as redis
from blake3 import blake3
from redis.exceptions import ConnectionError as RedisConnectionError


CacheBackendName = Literal["fs", "qdrant", "pg_redis", "none"]


class CacheBackend(Protocol):
    name: CacheBackendName

    async def ensure_ready(self) -> None: ...
    async def close(self) -> None: ...

    async def resolve_key(self, identifier: str, rewrite_cache: bool, regen_cache: bool) -> Tuple[str, bool]:
        """
        Returns (effective_key, can_read_existing).
        If rewrite_cache=True (and not regen_cache), should pick a new unused suffix and set can_read_existing=False.
        Otherwise, returns identifier and can_read_existing = not regen_cache.
        """
        ...

    async def read(self, effective_key: str) -> Optional[dict]: ...
    async def write(self, effective_key: str, response: dict, model_name: str) -> None: ...

    async def exists(self, effective_key: str) -> bool: ...
    async def warm(self) -> None: ...


def _u64_hash(s: str) -> int:
    return int.from_bytes(blake3(s.encode("utf-8")).digest()[:8], "big", signed=False)


class CacheCore:
    def __init__(self, backend: CacheBackend | None) -> None:
        self.backend = backend

    async def ensure_ready(self) -> None:
        if self.backend:
            await self.backend.ensure_ready()

    async def warm(self) -> None:
        if self.backend:
            await self.backend.warm()

    async def close(self) -> None:
        if self.backend:
            await self.backend.close()

    async def get_cached(
        self,
        identifier: str,
        *,
        rewrite_cache: bool,
        regen_cache: bool,
        only_ok: bool = True,
    ) -> Tuple[Optional[dict], str]:
        if not self.backend:
            return None, identifier

        eff, can_read = await self.backend.resolve_key(identifier, rewrite_cache, regen_cache)
        if not can_read:
            return None, eff

        resp = await self.backend.read(eff)
        if not resp:
            return None, eff

        if only_ok and resp.get("error") != "OK":
            return None, eff

        return resp, eff

    async def put_cached(
        self,
        identifier: str,
        *,
        rewrite_cache: bool,
        regen_cache: bool,
        response: dict,
        model_name: str,
        log_errors: bool,
    ) -> str:
        if not self.backend:
            return identifier

        eff, _ = await self.backend.resolve_key(identifier, rewrite_cache, regen_cache)
        if response.get("error") == "OK" or (response.get("error") != "OK" and log_errors):
            await self.backend.write(eff, response, model_name=model_name)
        return eff


@dataclass
class FSCacheConfig:
    dir: Path
    client_type: str  # kept for parity, not used
    name: CacheBackendName = "fs"


class FSCache:
    name: CacheBackendName = "fs"

    def __init__(self, cfg: FSCacheConfig) -> None:
        self.cfg = cfg
        self.cfg.dir.mkdir(parents=True, exist_ok=True)

    async def ensure_ready(self) -> None:
        return

    async def close(self) -> None:
        return

    async def warm(self) -> None:
        return

    def _path_for(self, key: str) -> Path:
        return self.cfg.dir / f"{key}.json"

    async def exists(self, effective_key: str) -> bool:
        return self._path_for(effective_key).exists()

    async def resolve_key(self, identifier: str, rewrite_cache: bool, regen_cache: bool) -> Tuple[str, bool]:
        if rewrite_cache and not regen_cache and self.cfg.client_type == "completions":
            for i in range(0, 1000):
                cand = f"{identifier}_{i}"
                if not await self.exists(cand):
                    return cand, False
            return f"{identifier}_{int(time.time())}", False
        return identifier, (not regen_cache)

    async def read(self, effective_key: str) -> Optional[dict]:
        path = self._path_for(effective_key)
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                return json.loads(await f.read())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    async def write(self, effective_key: str, response: dict, model_name: str) -> None:
        path = self._path_for(effective_key)
        target = path
        if response.get("error") != "OK":
            target = path.with_name(f"{path.stem}_error{path.suffix}")

        async with aiofiles.open(target, "w", encoding="utf-8") as f:
            await f.write(json.dumps(response, indent=2))


class QdrantCache:
    name: CacheBackendName = "qdrant"

    def __init__(
        self,
        *,
        collection: str,
        client_type: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("QDRANT_URL") or "http://localhost:6333").rstrip("/")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY") or None
        self.collection = collection
        self.client_type = client_type
        self._ensured = False
        self._ensure_lock = asyncio.Lock()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["api-key"] = self.api_key
        return h

    async def ensure_ready(self) -> None:
        await self._ensure_collection()

    async def close(self) -> None:
        return

    async def warm(self) -> None:
        return

    async def _ensure_collection(self) -> None:
        async with self._ensure_lock:
            if self._ensured:
                return
            async with aiohttp.ClientSession() as s:
                url = f"{self.base_url}/collections/{self.collection}"
                async with s.get(url, headers=self._headers()) as r:
                    if r.status == 200:
                        self._ensured = True
                        return
                body = {"vectors": {"size": 1, "distance": "Dot"}}
                async with s.put(url, headers=self._headers(), data=json.dumps(body)) as r:
                    if r.status in (200, 201, 409):
                        self._ensured = True
                        return
                    txt = await r.text()
                    raise RuntimeError(f"Failed to create Qdrant collection: {r.status} {txt}")

    async def exists(self, effective_key: str) -> bool:
        await self._ensure_collection()
        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{self.collection}/points/scroll"
            body = {
                "filter": {"must": [
                    {"key": "identifier", "match": {"value": effective_key}},
                    {"key": "client_type", "match": {"value": self.client_type}},
                ]},
                "limit": 1,
                "with_payload": False,
            }
            async with s.post(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status != 200:
                    return False
                data = await r.json()
                return bool(data.get("result", {}).get("points"))

    async def resolve_key(self, identifier: str, rewrite_cache: bool, regen_cache: bool) -> Tuple[str, bool]:
        if self.client_type == "completions" and rewrite_cache and not regen_cache:
            for i in range(0, 1000):
                eff = f"{identifier}_{i}"
                if not await self.exists(eff):
                    return eff, False
            return f"{identifier}_{int(time.time())}", False
        return identifier, (not regen_cache)

    async def read(self, effective_key: str) -> Optional[dict]:
        await self._ensure_collection()
        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{self.collection}/points/scroll"
            body = {
                "filter": {"must": [
                    {"key": "identifier", "match": {"value": effective_key}},
                    {"key": "client_type", "match": {"value": self.client_type}},
                ]},
                "limit": 1,
                "with_payload": True,
            }
            async with s.post(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                pts = data.get("result", {}).get("points", [])
                if not pts:
                    return None
                payload = pts[0].get("payload", {})
                return payload.get("cache")

    async def write(self, effective_key: str, response: dict, model_name: str) -> None:
        await self._ensure_collection()
        payload = {
            "identifier": effective_key,
            "client_type": self.client_type,
            "model": model_name,
            "error": response.get("error"),
            "status": response.get("status"),
            "cache": response,
            "created_at": int(time.time()),
        }
        point = {
            "id": _u64_hash(effective_key),
            "vector": [0.0],
            "payload": payload,
        }
        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{self.collection}/points?wait=true"
            body = {"points": [point]}
            async with s.put(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status not in (200, 202):
                    txt = await r.text()
                    print(f"[QdrantCache] upsert failed: {r.status} {txt}")


def _sanitize_table_name(name: str) -> str:
    if not name:
        raise ValueError("table_name cannot be empty")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
        raise ValueError(f"Invalid table name: {name!r}")
    return name


@dataclass
class HybridCacheConfig:
    table_name: str
    client_type: str

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_prefix: str = "resp_cache"
    redis_ttl_seconds: int = 60 * 60 * 24

    lock_ttl_seconds: int = 30

    pg_dsn: str = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")

    compress: bool = True
    compression_level: int = 6


class HybridRedisPostgreSQLCache:
    name: CacheBackendName = "pg_redis"

    def __init__(self, cfg: HybridCacheConfig) -> None:
        self.cfg = cfg
        self.table = _sanitize_table_name(cfg.table_name)
        self._ensured = False
        self._ensure_lock = asyncio.Lock()

        self._redis: Optional[redis.Redis] = None
        self._pg_pool: Optional[asyncpg.Pool] = None

    async def _redis_ok(self) -> bool:
        if self._redis is None:
            return False
        try:
            await self._redis.ping()
            return True
        except RedisConnectionError:
            return False
        except Exception:
            return False

    async def ensure_ready(self) -> None:
        # Postgres must work
        if self._pg_pool is None:
            # noinspection PyUnresolvedReferences
            self._pg_pool = await asyncpg.create_pool(
                dsn=self.cfg.pg_dsn, min_size=1, max_size=20
            )

        # Redis is optional
        if self._redis is None:
            self._redis = redis.from_url(self.cfg.redis_url, decode_responses=False)

        await self._ensure_table()

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
        if self._pg_pool is not None:
            await self._pg_pool.close()

    async def warm(self) -> None:
        return

    async def _ensure_connections(self) -> None:
        if self._redis is None:
            self._redis = redis.from_url(self.cfg.redis_url, decode_responses=False)
        if self._pg_pool is None:
            # noinspection PyUnresolvedReferences
            self._pg_pool = await asyncpg.create_pool(dsn=self.cfg.pg_dsn, min_size=1, max_size=20)

    async def _ensure_table(self) -> None:
        async with self._ensure_lock:
            if self._ensured:
                return

            if self.cfg.compress:
                ddl = f'''
                CREATE TABLE IF NOT EXISTS "{self.table}" (
                  cache_key     TEXT PRIMARY KEY,
                  client_type   TEXT NOT NULL,
                  model         TEXT NOT NULL,
                  status        INTEGER,
                  error         TEXT,
                  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                  response_blob BYTEA NOT NULL
                );
                CREATE INDEX IF NOT EXISTS "{self.table}_created_at_idx" ON "{self.table}" (created_at);
                '''
            else:
                ddl = f'''
                CREATE TABLE IF NOT EXISTS "{self.table}" (
                  cache_key     TEXT PRIMARY KEY,
                  client_type   TEXT NOT NULL,
                  model         TEXT NOT NULL,
                  status        INTEGER,
                  error         TEXT,
                  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                  response_json JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS "{self.table}_created_at_idx" ON "{self.table}" (created_at);
                '''

            assert self._pg_pool is not None
            async with self._pg_pool.acquire() as conn:
                for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                    await conn.execute(stmt)
            self._ensured = True

    def _encode(self, obj: Dict[str, Any]) -> bytes:
        raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if not self.cfg.compress:
            return raw
        return zlib.compress(raw, level=self.cfg.compression_level)

    def _decode(self, data: bytes) -> Dict[str, Any]:
        if self.cfg.compress:
            data = zlib.decompress(data)
        return json.loads(data.decode("utf-8"))

    def _rk(self, key: str) -> str:
        return f"{self.cfg.redis_prefix}:{self.cfg.client_type}:{key}"

    async def exists(self, effective_key: str) -> bool:
        await self.ensure_ready()

        if await self._redis_ok():
            try:
                if await self._redis.exists(self._rk(effective_key)):  # type: ignore[union-attr]
                    return True
            except Exception:
                pass

        return (await self._pg_read(effective_key)) is not None

    async def resolve_key(self, identifier: str, rewrite_cache: bool, regen_cache: bool) -> Tuple[str, bool]:
        if self.cfg.client_type == "completions" and rewrite_cache and not regen_cache:
            for i in range(0, 1000):
                eff = f"{identifier}_{i}"
                if not await self.exists(eff):
                    return eff, False
            return f"{identifier}_{int(time.time())}", False
        return identifier, (not regen_cache)

    async def read(self, effective_key: str) -> Optional[dict]:
        await self.ensure_ready()

        rk = self._rk(effective_key)

        # Try Redis first, but never die if it's down.
        if await self._redis_ok():
            try:
                b = await self._redis.get(rk)  # type: ignore[union-attr]
                if b:
                    try:
                        # print(f"[CACHE HIT] redis key={rk}")
                        return self._decode(b)
                    except Exception:
                        # corrupt entry
                        await self._redis.delete(rk)  # type: ignore[union-attr]
            except RedisConnectionError:
                pass
            except Exception:
                pass

        # Postgres fallback
        row = await self._pg_read(effective_key)
        if row is None:
            # print(f"[CACHE MISS] postgres table={self.table} key={effective_key}")
            return None
        # print(f"[CACHE HIT] postgres table={self.table} key={effective_key}")

        # Best-effort repopulate Redis
        if await self._redis_ok():
            try:
                await self._redis.set(rk, self._encode(row), ex=self.cfg.redis_ttl_seconds)  # type: ignore[union-attr]
            except Exception:
                pass

        return row

    async def write(self, effective_key: str, response: dict, model_name: str) -> None:
        await self.ensure_ready()

        # Durable always
        await self._pg_upsert(effective_key, response, model_name=model_name)

        # Best-effort hot cache
        if await self._redis_ok():
            try:
                await self._redis.set(self._rk(effective_key), self._encode(response),
                                      ex=self.cfg.redis_ttl_seconds)  # type: ignore[union-attr]
            except Exception:
                pass

    async def _pg_read(self, cache_key: str) -> Optional[dict]:
        assert self._pg_pool is not None
        async with self._pg_pool.acquire() as conn:
            if self.cfg.compress:
                q = f'''SELECT response_blob FROM "{self.table}" WHERE cache_key = $1 AND client_type = $2'''
                b = await conn.fetchval(q, cache_key, self.cfg.client_type)
                if b is None:
                    return None
                return self._decode(bytes(b))
            q = f'''SELECT response_json FROM "{self.table}" WHERE cache_key = $1 AND client_type = $2'''
            j = await conn.fetchval(q, cache_key, self.cfg.client_type)
            return dict(j) if j is not None else None

    async def _pg_upsert(self, cache_key: str, response: dict, model_name: str) -> None:
        assert self._pg_pool is not None
        status = response.get("status")
        error = response.get("error")

        async with self._pg_pool.acquire() as conn:
            if self.cfg.compress:
                blob = self._encode(response)
                q = f'''
                INSERT INTO "{self.table}" (cache_key, client_type, model, status, error, response_blob)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (cache_key)
                DO UPDATE SET
                  model = EXCLUDED.model,
                  status = EXCLUDED.status,
                  error = EXCLUDED.error,
                  response_blob = EXCLUDED.response_blob,
                  created_at = NOW()
                '''
                await conn.execute(q, cache_key, self.cfg.client_type, model_name, status, error, blob)
            else:
                q = f'''
                INSERT INTO "{self.table}" (cache_key, client_type, model, status, error, response_json)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (cache_key)
                DO UPDATE SET
                  model = EXCLUDED.model,
                  status = EXCLUDED.status,
                  error = EXCLUDED.error,
                  response_json = EXCLUDED.response_json,
                  created_at = NOW()
                '''
                await conn.execute(q, cache_key, self.cfg.client_type, model_name, status, error, json.dumps(response))


@dataclass
class CacheSettings:
    backend: CacheBackendName
    client_type: str
    collection: str | None = None
    cache_dir: Path | None = None

    # qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # pg_redis
    pg_dsn: str | None = None
    redis_url: str | None = None
    redis_ttl_seconds: int = 60 * 60 * 24
    compress: bool = True


def build_cache_core(settings: CacheSettings) -> CacheCore:
    backend = settings.backend
    if backend == "none" or backend is None:
        return CacheCore(None)

    if backend == "fs":
        if not settings.cache_dir:
            raise ValueError("cache_dir is required for fs cache backend")
        fs = FSCache(FSCacheConfig(dir=settings.cache_dir, client_type=settings.client_type))
        return CacheCore(fs)

    if backend == "qdrant":
        coll = settings.collection or f"{settings.client_type}_cache"
        q = QdrantCache(
            collection=coll,
            client_type=settings.client_type,
            base_url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        return CacheCore(q)

    if backend == "pg_redis":
        coll = settings.collection or f"{settings.client_type}_cache"
        h = HybridRedisPostgreSQLCache(
            HybridCacheConfig(
                table_name=coll,
                client_type=settings.client_type,
                pg_dsn=settings.pg_dsn or os.getenv("PG_DSN", ""),
                redis_url=settings.redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                redis_ttl_seconds=settings.redis_ttl_seconds,
                compress=settings.compress,
            )
        )
        return CacheCore(h)

    raise ValueError(f"Unknown cache backend: {backend!r}")
