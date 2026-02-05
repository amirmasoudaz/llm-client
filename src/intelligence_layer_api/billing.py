from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from blake3 import blake3


@dataclass(frozen=True)
class CreditReservation:
    ok: bool
    reservation_id: uuid.UUID | None
    reserved_credits: int
    request_key: bytes
    expires_at: datetime | None
    reason: str | None = None


@dataclass(frozen=True)
class CreditSettlement:
    ok: bool
    ledger_id: uuid.UUID | None
    debited_credits: int
    balance_after: int | None
    reason: str | None = None


class CreditManager:
    def __init__(
        self,
        *,
        pool,
        tenant_id: int,
        bootstrap_enabled: bool,
        bootstrap_credits: int,
        reservation_ttl_sec: int,
        min_reserve_credits: int,
    ) -> None:
        self._pool = pool
        self._tenant_id = tenant_id
        self._bootstrap_enabled = bootstrap_enabled
        self._bootstrap_credits = bootstrap_credits
        self._reservation_ttl = int(reservation_ttl_sec)
        self._min_reserve = int(min_reserve_credits)
        self._pricing_version_id: uuid.UUID | None = None
        self._credit_rate_version: str | None = None

    async def estimate_reserve_credits(self, message: str) -> int:
        # Simple deterministic estimate based on message length.
        base = max(self._min_reserve, 1)
        bump = max(0, len(message) // 800)
        return base + bump

    async def reserve(
        self,
        *,
        principal_id: int,
        workflow_id: uuid.UUID,
        request_key: bytes,
        estimated_credits: int,
    ) -> CreditReservation:
        await self._ensure_defaults()
        if estimated_credits <= 0:
            estimated_credits = max(self._min_reserve, 1)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchrow(
                    """
                    SELECT reservation_id, reserved_credits, expires_at, status
                    FROM billing.credit_reservations
                    WHERE tenant_id=$1 AND request_key=$2
                    LIMIT 1;
                    """,
                    self._tenant_id,
                    request_key,
                )
                if existing:
                    return CreditReservation(
                        ok=True,
                        reservation_id=existing["reservation_id"],
                        reserved_credits=int(existing["reserved_credits"]),
                        request_key=request_key,
                        expires_at=existing["expires_at"],
                    )

                await self._ensure_balance(conn, principal_id)

                balance_row = await conn.fetchrow(
                    """
                    SELECT balance_credits, overdraft_limit
                    FROM billing.credit_balances
                    WHERE tenant_id=$1 AND principal_id=$2
                    FOR UPDATE;
                    """,
                    self._tenant_id,
                    principal_id,
                )
                if balance_row is None:
                    return CreditReservation(
                        ok=False,
                        reservation_id=None,
                        reserved_credits=0,
                        request_key=request_key,
                        expires_at=None,
                        reason="no_balance",
                    )
                balance = int(balance_row["balance_credits"])
                overdraft = int(balance_row["overdraft_limit"])

                reserved = await conn.fetchval(
                    """
                    SELECT COALESCE(SUM(reserved_credits), 0)
                    FROM billing.credit_reservations
                    WHERE tenant_id=$1 AND principal_id=$2 AND status='reserved' AND expires_at > now();
                    """,
                    self._tenant_id,
                    principal_id,
                )
                reserved = int(reserved or 0)

                available = balance + overdraft - reserved
                if available < estimated_credits:
                    return CreditReservation(
                        ok=False,
                        reservation_id=None,
                        reserved_credits=0,
                        request_key=request_key,
                        expires_at=None,
                        reason="insufficient_credits",
                    )

                reservation_id = uuid.uuid4()
                expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._reservation_ttl)
                await conn.execute(
                    """
                    INSERT INTO billing.credit_reservations (
                      tenant_id, reservation_id, principal_id, workflow_id,
                      request_key, reserved_credits, status, expires_at
                    ) VALUES ($1,$2,$3,$4,$5,$6,'reserved',$7);
                    """,
                    self._tenant_id,
                    reservation_id,
                    principal_id,
                    workflow_id,
                    request_key,
                    int(estimated_credits),
                    expires_at,
                )

                return CreditReservation(
                    ok=True,
                    reservation_id=reservation_id,
                    reserved_credits=int(estimated_credits),
                    request_key=request_key,
                    expires_at=expires_at,
                )

    async def settle(
        self,
        *,
        principal_id: int,
        workflow_id: uuid.UUID,
        request_key: bytes,
    ) -> CreditSettlement:
        await self._ensure_defaults()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                ledger = await conn.fetchrow(
                    """
                    SELECT credit_ledger_id, delta_credits, balance_after
                    FROM billing.credit_ledger
                    WHERE tenant_id=$1 AND request_key=$2
                    LIMIT 1;
                    """,
                    self._tenant_id,
                    request_key,
                )
                if ledger:
                    return CreditSettlement(
                        ok=True,
                        ledger_id=ledger["credit_ledger_id"],
                        debited_credits=abs(int(ledger["delta_credits"])),
                        balance_after=int(ledger["balance_after"]),
                    )

                reservation = await conn.fetchrow(
                    """
                    SELECT reserved_credits, status
                    FROM billing.credit_reservations
                    WHERE tenant_id=$1 AND request_key=$2
                    LIMIT 1;
                    """,
                    self._tenant_id,
                    request_key,
                )
                if reservation is None:
                    return CreditSettlement(
                        ok=False,
                        ledger_id=None,
                        debited_credits=0,
                        balance_after=None,
                        reason="reservation_missing",
                    )

                reserved_credits = int(reservation["reserved_credits"])
                actual_credits = await conn.fetchval(
                    """
                    SELECT COALESCE(SUM(credits_charged), 0)
                    FROM billing.usage_events
                    WHERE tenant_id=$1 AND workflow_id=$2;
                    """,
                    self._tenant_id,
                    workflow_id,
                )
                actual_credits = int(actual_credits or 0)

                if actual_credits <= 0:
                    actual_credits = reserved_credits
                    await self._insert_estimated_usage(
                        conn,
                        principal_id=principal_id,
                        workflow_id=workflow_id,
                        credits=actual_credits,
                    )

                if actual_credits > reserved_credits:
                    actual_credits = reserved_credits

                balance_row = await conn.fetchrow(
                    """
                    SELECT balance_credits
                    FROM billing.credit_balances
                    WHERE tenant_id=$1 AND principal_id=$2
                    FOR UPDATE;
                    """,
                    self._tenant_id,
                    principal_id,
                )
                if balance_row is None:
                    return CreditSettlement(
                        ok=False,
                        ledger_id=None,
                        debited_credits=0,
                        balance_after=None,
                        reason="no_balance",
                    )

                balance = int(balance_row["balance_credits"])
                new_balance = balance - actual_credits

                await conn.execute(
                    """
                    UPDATE billing.credit_balances
                    SET balance_credits=$3, updated_at=now()
                    WHERE tenant_id=$1 AND principal_id=$2;
                    """,
                    self._tenant_id,
                    principal_id,
                    new_balance,
                )

                ledger_id = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO billing.credit_ledger (
                      tenant_id, credit_ledger_id, principal_id, workflow_id,
                      request_key, delta_credits, balance_after, reason_code,
                      pricing_version_id, credit_rate_version, metadata
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb);
                    """,
                    self._tenant_id,
                    ledger_id,
                    principal_id,
                    workflow_id,
                    request_key,
                    -int(actual_credits),
                    new_balance,
                    "usage_settlement",
                    self._pricing_version_id,
                    self._credit_rate_version,
                    json.dumps({}),
                )

                await conn.execute(
                    """
                    UPDATE billing.credit_reservations
                    SET status='settled', updated_at=now()
                    WHERE tenant_id=$1 AND request_key=$2;
                    """,
                    self._tenant_id,
                    request_key,
                )

                return CreditSettlement(
                    ok=True,
                    ledger_id=ledger_id,
                    debited_credits=actual_credits,
                    balance_after=new_balance,
                )

    async def _ensure_defaults(self) -> None:
        if self._pricing_version_id and self._credit_rate_version:
            return
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT credit_rate_version
                    FROM billing.credit_rate_versions
                    WHERE effective_to IS NULL
                    ORDER BY effective_from DESC
                    LIMIT 1;
                    """,
                )
                if row is None:
                    credit_rate_version = "dev-v1"
                    await conn.execute(
                        """
                        INSERT INTO billing.credit_rate_versions (
                          credit_rate_version, credits_per_usd, effective_from
                        ) VALUES ($1,$2,now());
                        """,
                        credit_rate_version,
                        100.0,
                    )
                else:
                    credit_rate_version = str(row["credit_rate_version"])

                row = await conn.fetchrow(
                    """
                    SELECT pricing_version_id
                    FROM billing.pricing_versions
                    WHERE effective_to IS NULL
                    ORDER BY effective_from DESC
                    LIMIT 1;
                    """,
                )
                if row is None:
                    pricing_version_id = uuid.uuid4()
                    await conn.execute(
                        """
                        INSERT INTO billing.pricing_versions (
                          pricing_version_id, provider, version, effective_from, pricing_json
                        ) VALUES ($1,$2,$3,now(),$4::jsonb);
                        """,
                        pricing_version_id,
                        "dev",
                        "v1",
                        json.dumps({}),
                    )
                else:
                    pricing_version_id = row["pricing_version_id"]

        self._credit_rate_version = credit_rate_version
        self._pricing_version_id = pricing_version_id

    async def _ensure_balance(self, conn, principal_id: int) -> None:
        if not self._bootstrap_enabled:
            return
        await conn.execute(
            """
            INSERT INTO billing.credit_balances (
              tenant_id, principal_id, balance_credits, overdraft_limit, updated_at
            ) VALUES ($1,$2,$3,$4,now())
            ON CONFLICT (tenant_id, principal_id) DO NOTHING;
            """,
            self._tenant_id,
            principal_id,
            int(self._bootstrap_credits),
            0,
        )

    async def _insert_estimated_usage(
        self,
        conn,
        *,
        principal_id: int,
        workflow_id: uuid.UUID,
        credits: int,
    ) -> None:
        usage_event_id = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO billing.usage_events (
              tenant_id, usage_event_id, principal_id, workflow_id, job_id,
              operation_type, provider, model, usage, cost_usd, effective_cost_usd,
              credits_charged, pricing_version_id, credit_rate_version, estimated
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11,$12,$13,$14,$15);
            """,
            self._tenant_id,
            usage_event_id,
            principal_id,
            workflow_id,
            None,
            "workflow",
            "internal",
            "none",
            json.dumps({"estimated": True}),
            0.0,
            0.0,
            int(credits),
            self._pricing_version_id,
            self._credit_rate_version,
            True,
        )


def request_key_for_workflow(workflow_id: uuid.UUID) -> bytes:
    return blake3(str(workflow_id).encode("utf-8")).digest()
