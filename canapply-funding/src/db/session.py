# src/db/session.py

import asyncio, aiomysql
from typing import Any, Sequence

from src.config import settings
from src.tools.logger import Logger


_LOG, _ = Logger().create(application="db")


class DB:
    _pool: aiomysql.Pool | None = None
    _pool_loop: asyncio.AbstractEventLoop | None = None
    # Use a dictionary to store locks per event loop to avoid binding a lock to a closed loop
    _locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if loop not in cls._locks:
            cls._locks[loop] = asyncio.Lock()
        return cls._locks[loop]

    @classmethod
    async def _create_pool(cls):
        pool = await aiomysql.create_pool(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASS,
            db=settings.DB_NAME,
            minsize=settings.DB_MIN,
            maxsize=settings.DB_MAX,
            autocommit=True,
            charset="utf8mb4",
        )
        _LOG.info("MySQL pool created")
        return pool

    @classmethod
    async def pool(cls) -> aiomysql.Pool:
        loop = asyncio.get_running_loop()
        if cls._pool is None or cls._pool_loop is not loop or getattr(cls._pool, "_closed", False):
            async with cls._get_lock():
                loop = asyncio.get_running_loop()
                need_new = (
                    cls._pool is None
                    or cls._pool_loop is not loop
                    or getattr(cls._pool, "_closed", False)
                )
                if need_new:
                    if cls._pool is not None:
                        try:
                            cls._pool.close()
                            await cls._pool.wait_closed()
                        except Exception:
                            pass
                    cls._pool = await cls._create_pool()
                    cls._pool_loop = loop
        return cls._pool

    @classmethod
    async def execute(cls, q: str, p: Sequence[Any] = ()) -> int | None:
        for attempt in (1, 2):
            pool = await cls.pool()
            try:
                async with pool.acquire() as conn, conn.cursor() as cur:
                    await cur.execute(q, p)
                    return cur.lastrowid or 0
            except RuntimeError as exc:
                if "Cannot acquire connection after closing pool" in str(exc) and attempt == 1:
                    _LOG.warning("MySQL pool was closed; recreating and retrying execute()")
                    await cls.close()
                    continue
                raise

    @classmethod
    async def execute_many_transaction(cls, q: str, params: list[Sequence[Any]]) -> None:
        """Execute multiple statements in a single atomic transaction.
        
        Uses executemany() within an explicit transaction - either all succeed
        or all are rolled back on failure.
        """
        for attempt in (1, 2):
            pool = await cls.pool()
            try:
                async with pool.acquire() as conn:
                    # Disable autocommit to enable manual transaction control
                    await conn.autocommit(False)
                    try:
                        async with conn.cursor() as cur:
                            await cur.executemany(q, params)
                        await conn.commit()
                    except Exception:
                        await conn.rollback()
                        raise
                    finally:
                        # Re-enable autocommit for connection reuse
                        await conn.autocommit(True)
                return
            except RuntimeError as exc:
                if "Cannot acquire connection after closing pool" in str(exc) and attempt == 1:
                    _LOG.warning("MySQL pool was closed; recreating and retrying execute_many_transaction()")
                    await cls.close()
                    continue
                raise


    @classmethod
    async def fetch_one(cls, q: str, p: Sequence[Any] = ()) -> dict | None:
        for attempt in (1, 2):
            pool = await cls.pool()
            try:
                async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(q, p)
                    return await cur.fetchone()
            except RuntimeError as exc:
                if "Cannot acquire connection after closing pool" in str(exc) and attempt == 1:
                    _LOG.warning("MySQL pool was closed; recreating and retrying fetch_one()")
                    await cls.close()
                    continue
                raise

    @classmethod
    async def fetch_all(cls, q: str, p: Sequence[Any] = ()) -> list[dict[str, Any]] | None:
        for attempt in (1, 2):
            pool = await cls.pool()
            try:
                async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(q, p)
                    return await cur.fetchall()
            except RuntimeError as exc:
                if "Cannot acquire connection after closing pool" in str(exc) and attempt == 1:
                    _LOG.warning("MySQL pool was closed; recreating and retrying fetch_all()")
                    await cls.close()
                    continue
                raise

    @classmethod
    async def close(cls):
        # Safely close if we can acquire the lock for the current loop
        try:
            lock = cls._get_lock()
            async with lock:
                if cls._pool:
                    cls._pool.close()
                    await cls._pool.wait_closed()
                    cls._pool = None
                    cls._pool_loop = None
                    _LOG.info("MySQL pool closed")
        except RuntimeError:
            # Loop might be closed or no loop running, just ignore
            pass
