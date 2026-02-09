"""
Persistence layer for LLM Client.

This module isolates direct database interactions, keeping SQL out of the
main application logic. It implements the Repository pattern.
"""

from __future__ import annotations

import asyncio
import json
import re
import zlib
from typing import Any

import asyncpg


class PostgresRepository:
    """
    Repository for storing LLM responses in PostgreSQL.

    Handles:
    - Safe table creation (DDL)
    - Data serialization/compression
    - CRUD operations
    """

    def __init__(self, pool: asyncpg.Pool, compress: bool = True, compression_level: int = 6):
        self.pool = pool
        self.compress = compress
        self._compression_level = compression_level
        self._ensured_tables: set[str] = set()
        self._lock = asyncio.Lock()

    @staticmethod
    def _sanitize_table_name(name: str) -> str:
        """Ensure the table name is safe for SQL interpolation."""
        if not name:
            raise ValueError("table_name cannot be empty")
        if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
            raise ValueError(f"Invalid table name: {name!r}")
        return name

    def _encode(self, obj: dict[str, Any]) -> bytes:
        """Encode dictionary to bytes (optionally compressed)."""
        raw = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if not self.compress:
            return raw
        return zlib.compress(raw, level=self._compression_level)

    def _decode(self, data: bytes) -> dict[str, Any]:
        """Decode bytes to dictionary (optionally decompressed)."""
        if self.compress:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                # Fallback in case data wasn't compressed (migration scenario)
                pass
        return json.loads(data.decode("utf-8"))

    async def ensure_table(self, table_name: str) -> None:
        """Ensure the cache table exists."""
        async with self._lock:
            if table_name in self._ensured_tables:
                return

            table = self._sanitize_table_name(table_name)

            if self.compress:
                ddl = f'''
                CREATE TABLE IF NOT EXISTS "{table}" (
                  cache_key     TEXT PRIMARY KEY,
                  client_type   TEXT NOT NULL,
                  model         TEXT NOT NULL,
                  status        INTEGER,
                  error         TEXT,
                  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                  response_blob BYTEA NOT NULL
                );
                CREATE INDEX IF NOT EXISTS "{table}_created_at_idx" ON "{table}" (created_at);
                '''
            else:
                ddl = f'''
                CREATE TABLE IF NOT EXISTS "{table}" (
                  cache_key     TEXT PRIMARY KEY,
                  client_type   TEXT NOT NULL,
                  model         TEXT NOT NULL,
                  status        INTEGER,
                  error         TEXT,
                  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                  response_json JSONB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS "{table}_created_at_idx" ON "{table}" (created_at);
                '''

            async with self.pool.acquire() as conn:
                # Execute statements individually
                for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
                    await conn.execute(stmt)

            self._ensured_tables.add(table_name)

    async def read(self, table_name: str, key: str, client_type: str) -> dict[str, Any] | None:
        """Read a response from the database."""
        table = self._sanitize_table_name(table_name)

        async with self.pool.acquire() as conn:
            if self.compress:
                q = f'''SELECT response_blob FROM "{table}" WHERE cache_key = $1 AND client_type = $2'''
                b = await conn.fetchval(q, key, client_type)
                if b is None:
                    return None
                return self._decode(bytes(b))
            else:
                q = f'''SELECT response_json FROM "{table}" WHERE cache_key = $1 AND client_type = $2'''
                j = await conn.fetchval(q, key, client_type)
                return dict(json.loads(j)) if j is not None else None

    async def upsert(
        self,
        table_name: str,
        key: str,
        client_type: str,
        model_name: str,
        response: dict[str, Any],
    ) -> None:
        """Insert or update a response."""
        table = self._sanitize_table_name(table_name)
        status = response.get("status")
        error = response.get("error")

        async with self.pool.acquire() as conn:
            if self.compress:
                blob = self._encode(response)
                q = f'''
                INSERT INTO "{table}" (cache_key, client_type, model, status, error, response_blob)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (cache_key)
                DO UPDATE SET
                  model = EXCLUDED.model,
                  status = EXCLUDED.status,
                  error = EXCLUDED.error,
                  response_blob = EXCLUDED.response_blob,
                  created_at = NOW()
                '''
                await conn.execute(q, key, client_type, model_name, status, error, blob)
            else:
                q = f'''
                INSERT INTO "{table}" (cache_key, client_type, model, status, error, response_json)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (cache_key)
                DO UPDATE SET
                  model = EXCLUDED.model,
                  status = EXCLUDED.status,
                  error = EXCLUDED.error,
                  response_json = EXCLUDED.response_json,
                  created_at = NOW()
                '''
                await conn.execute(q, key, client_type, model_name, status, error, json.dumps(response))

    async def delete_old(self, table_name: str, days: int) -> int:
        """Clean up old cache entries."""
        if days < 0:
            raise ValueError("days must be >= 0")

        table = self._sanitize_table_name(table_name)
        q = f'''DELETE FROM "{table}" WHERE created_at < NOW() - ($1::int * INTERVAL '1 day')'''

        async with self.pool.acquire() as conn:
            result = await conn.execute(q, days)
            # the result is usually "DELETE N"
            try:
                return int(result.split()[-1])
            except (IndexError, ValueError):
                return 0
