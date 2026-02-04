from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Sequence

import aiomysql


@dataclass(frozen=True)
class PlatformDBConfig:
    host: str
    port: int
    user: str
    password: str
    db: str
    minsize: int = 1
    maxsize: int = 10


class PlatformDB:
    """Async MariaDB/MySQL connection pool (aiomysql).

    Modeled after `canapply-funding/src/db/session.py` for operational consistency.
    """

    _pool: aiomysql.Pool | None
    _pool_loop: asyncio.AbstractEventLoop | None
    _locks: dict[asyncio.AbstractEventLoop, asyncio.Lock]

    def __init__(self, config: PlatformDBConfig):
        self._config = config
        self._pool = None
        self._pool_loop = None
        self._locks = {}

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock

    async def _create_pool(self) -> aiomysql.Pool:
        return await aiomysql.create_pool(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password,
            db=self._config.db,
            minsize=self._config.minsize,
            maxsize=self._config.maxsize,
            autocommit=True,
            charset="utf8mb4",
        )

    async def pool(self) -> aiomysql.Pool:
        loop = asyncio.get_running_loop()
        if self._pool is None or self._pool_loop is not loop or getattr(self._pool, "_closed", False):
            async with self._get_lock():
                loop = asyncio.get_running_loop()
                need_new = (
                    self._pool is None
                    or self._pool_loop is not loop
                    or getattr(self._pool, "_closed", False)
                )
                if need_new:
                    if self._pool is not None:
                        try:
                            self._pool.close()
                            await self._pool.wait_closed()
                        except Exception:
                            pass
                    self._pool = await self._create_pool()
                    self._pool_loop = loop
        return self._pool

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        for attempt in (1, 2):
            pool = await self.pool()
            try:
                async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(sql, params)
                    return await cur.fetchone()
            except RuntimeError as exc:
                if "Cannot acquire connection after closing pool" in str(exc) and attempt == 1:
                    await self.close()
                    continue
                raise

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        for attempt in (1, 2):
            pool = await self.pool()
            try:
                async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(sql, params)
                    return list(await cur.fetchall())
            except RuntimeError as exc:
                if "Cannot acquire connection after closing pool" in str(exc) and attempt == 1:
                    await self.close()
                    continue
                raise

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        for attempt in (1, 2):
            pool = await self.pool()
            try:
                async with pool.acquire() as conn, conn.cursor() as cur:
                    await cur.execute(sql, params)
                    return int(cur.lastrowid or 0)
            except RuntimeError as exc:
                if "Cannot acquire connection after closing pool" in str(exc) and attempt == 1:
                    await self.close()
                    continue
                raise

    async def close(self) -> None:
        try:
            lock = self._get_lock()
            async with lock:
                if self._pool:
                    self._pool.close()
                    await self._pool.wait_closed()
                    self._pool = None
                    self._pool_loop = None
        except RuntimeError:
            # No running loop / loop closed.
            pass

