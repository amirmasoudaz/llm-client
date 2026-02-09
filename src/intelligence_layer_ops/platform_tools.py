from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any

from llm_client import tool

from .platform_context import load_funding_request_thread_context
from .platform_db import PlatformDB, PlatformDBConfig


_db: PlatformDB | None = None


def _get_db() -> PlatformDB:
    # Lazy-init global pool; Layer 2 owns process lifecycle.
    global _db
    if _db is not None:
        return _db

    import os

    cfg = PlatformDBConfig(
        host=os.getenv("PLATFORM_DB_HOST", os.getenv("DB_HOST", "127.0.0.1")),
        port=int(os.getenv("PLATFORM_DB_PORT", os.getenv("DB_PORT", "3306"))),
        user=os.getenv("PLATFORM_DB_USER", os.getenv("DB_USER", "funding")),
        password=os.getenv("PLATFORM_DB_PASS", os.getenv("DB_PASS", "secret")),
        db=os.getenv("PLATFORM_DB_NAME", os.getenv("DB_NAME", "emaildb")),
        minsize=int(os.getenv("PLATFORM_DB_MIN", os.getenv("DB_MIN", "1"))),
        maxsize=int(os.getenv("PLATFORM_DB_MAX", os.getenv("DB_MAX", "10"))),
    )
    _db = PlatformDB(cfg)
    return _db


def _json_serialize_value(val: Any) -> Any:
    """Convert non-JSON-serializable values to JSON-compatible types."""
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    if isinstance(val, datetime.time):
        return val.isoformat()
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, dict):
        return {k: _json_serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_serialize_value(v) for v in val]
    return val


@tool(
    name="platform_load_funding_thread_context",
    description="Load CanApply platform context for a funding_request_id (read-only).",
    strict=True,
)
async def platform_load_funding_thread_context(funding_request_id: int) -> dict[str, Any]:
    ctx = await load_funding_request_thread_context(_get_db(), funding_request_id=funding_request_id)
    # Ensure all values are JSON-serializable (dates, decimals, etc.)
    return _json_serialize_value(ctx.row)

