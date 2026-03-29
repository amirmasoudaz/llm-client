from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import uuid

from cookbook_support import (
    build_live_provider,
    close_provider,
    print_heading,
    print_json,
    require_database_dsn,
    require_optional_module,
    summarize_usage,
)

if not require_optional_module("asyncpg", "Install it with: pip install asyncpg"):
    raise SystemExit(0)

import asyncpg

from llm_client.persistence import PostgresRepository
from llm_client.providers.types import Message


TABLE_NAME = "llm_client_cookbook_repository"


def _run_prefix() -> str:
    return f"cookbook_repo_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"


async def _live_payloads(provider) -> tuple[dict, dict]:  # type: ignore[no-untyped-def]
    triage = await provider.complete(
        [
            Message.system(
                "Return a compact internal incident-routing note in markdown with 3 sections: "
                "severity, owner, immediate action."
            ),
            Message.user(
                "Finance export jobs started timing out after retention-policy changes. "
                "Finance cannot complete month-end reconciliation."
            ),
        ]
    )
    customer_update = await provider.complete(
        [
            Message.system(
                "Draft a concise customer-facing status update in markdown. "
                "Acknowledge impact, mention investigation, and avoid overpromising."
            ),
            Message.user(
                "Workspace export jobs are timing out after audit logging was enabled. "
                "Finance and compliance teams are blocked on month-end reconciliation."
            ),
        ]
    )
    return (
        {
            "kind": "incident_triage",
            "status": triage.status,
            "model": triage.model,
            "content": triage.content,
            "error": triage.error,
            "usage": summarize_usage(triage.usage),
            "tags": ["internal", "ops"],
        },
        {
            "kind": "customer_update",
            "status": customer_update.status,
            "model": customer_update.model,
            "content": customer_update.content,
            "error": customer_update.error,
            "usage": summarize_usage(customer_update.usage),
            "tags": ["external", "support"],
        },
    )


async def _table_rows(pool: asyncpg.Pool, keys: list[str], *, compress: bool) -> list[dict]:
    size_expr = "octet_length(response_blob)" if compress else "octet_length(response_json::text)"
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT cache_key, client_type, model, status, error, created_at,
                   {size_expr} AS stored_bytes
            FROM "{TABLE_NAME}"
            WHERE cache_key = ANY($1::text[])
            ORDER BY cache_key
            """,
            keys,
        )
    return [
        {
            "cache_key": row["cache_key"],
            "client_type": row["client_type"],
            "model": row["model"],
            "status": row["status"],
            "error": row["error"],
            "created_at": row["created_at"],
            "stored_bytes": row["stored_bytes"],
        }
        for row in rows
    ]


async def main() -> None:
    handle = build_live_provider()
    pool = None
    try:
        dsn = require_database_dsn()
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
        repository = PostgresRepository(pool, compress=True)
        await repository.ensure_table(TABLE_NAME)

        run_prefix = _run_prefix()
        triage_key = f"{run_prefix}:triage"
        customer_key = f"{run_prefix}:customer_update"

        triage_payload, customer_payload = await _live_payloads(handle.provider)

        await repository.upsert(TABLE_NAME, triage_key, "chat", handle.model, triage_payload)
        first_read = await repository.read(TABLE_NAME, triage_key, "chat")

        enriched_triage_payload = {
            **(first_read or triage_payload),
            "review_state": "approved_for_handoff",
            "handoff_owner": "platform-operations",
        }
        await repository.upsert(TABLE_NAME, triage_key, "chat", handle.model, enriched_triage_payload)
        await repository.upsert(TABLE_NAME, customer_key, "chat", handle.model, customer_payload)

        final_triage = await repository.read(TABLE_NAME, triage_key, "chat")
        final_customer = await repository.read(TABLE_NAME, customer_key, "chat")
        row_metadata = await _table_rows(pool, [triage_key, customer_key], compress=repository.compress)

        invalid_table_error = None
        try:
            await repository.ensure_table("llm_client_cookbook_repository;drop")
        except ValueError as exc:
            invalid_table_error = str(exc)

        print_heading("Persistence Repository")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "table": TABLE_NAME,
                "compression_enabled": repository.compress,
                "run_prefix": run_prefix,
                "artifacts": [
                    {
                        "key": triage_key,
                        "kind": "incident_triage",
                        "first_read_excerpt": (first_read or {}).get("content", "")[:180] if first_read else None,
                        "final_read_excerpt": (final_triage or {}).get("content", "")[:180] if final_triage else None,
                        "review_state": (final_triage or {}).get("review_state"),
                        "handoff_owner": (final_triage or {}).get("handoff_owner"),
                    },
                    {
                        "key": customer_key,
                        "kind": "customer_update",
                        "final_read_excerpt": (final_customer or {}).get("content", "")[:180] if final_customer else None,
                        "tags": (final_customer or {}).get("tags"),
                    },
                ],
                "stored_rows": row_metadata,
                "safety_checks": {
                    "invalid_table_name_error": invalid_table_error,
                },
            }
        )
    finally:
        if pool is not None:
            await pool.close()
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
