from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .platform_db import PlatformDB
from .platform_queries import SELECT_FUNDING_THREAD_CONTEXT


@dataclass(frozen=True)
class PlatformThreadContext:
    """Typed-ish projection of platform thread context.

    v0: kept as a dict-like payload to avoid premature schema lock-in.
    Layer 2 should progressively tighten this into versioned schemas.
    """

    funding_request_id: int
    row: dict[str, Any]


async def load_funding_request_thread_context(
    db: PlatformDB,
    *,
    funding_request_id: int,
) -> PlatformThreadContext:
    row = await db.fetch_one(SELECT_FUNDING_THREAD_CONTEXT, (funding_request_id,))
    if not row:
        raise ValueError(f"No platform funding_request found for id={funding_request_id}")
    return PlatformThreadContext(funding_request_id=funding_request_id, row=row)

