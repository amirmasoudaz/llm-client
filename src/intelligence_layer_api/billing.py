from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from blake3 import blake3


_DEFAULT_PRICING_JSON: dict[str, Any] = {
    "models": {
        "openai:gpt-5-nano": {"input_per_1k_usd": 0.00005, "output_per_1k_usd": 0.00040},
        "openai:gpt-5-mini": {"input_per_1k_usd": 0.00025, "output_per_1k_usd": 0.00200},
        "openai:gpt-5": {"input_per_1k_usd": 0.00125, "output_per_1k_usd": 0.01000},
    }
}


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


@dataclass(frozen=True)
class UsageWriteResult:
    ok: bool
    usage_event_id: uuid.UUID | None
    credits_charged: int
    total_credits_charged: int
    effective_cost_usd: float
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
        settlement_outbox_max_retries: int = 8,
    ) -> None:
        self._pool = pool
        self._tenant_id = tenant_id
        self._bootstrap_enabled = bootstrap_enabled
        self._bootstrap_credits = bootstrap_credits
        self._reservation_ttl = int(reservation_ttl_sec)
        self._min_reserve = int(min_reserve_credits)
        self._settlement_outbox_max_retries = max(1, int(settlement_outbox_max_retries))
        self._pricing_version_id: uuid.UUID | None = None
        self._credit_rate_version: str | None = None
        self._credits_per_usd: float = 100.0
        self._pricing_json: dict[str, Any] = dict(_DEFAULT_PRICING_JSON)

    async def estimate_reserve_credits(
        self,
        message: str,
        *,
        provider: str = "openai",
        model: str = "gpt-5-nano",
        planned_steps: int = 1,
    ) -> int:
        prompt_tokens = _estimate_prompt_tokens(message)
        output_tokens = max(128, min(1200, prompt_tokens))
        quote = await self.quote_usage(
            provider=provider,
            model=model,
            tokens_in=prompt_tokens,
            tokens_out=output_tokens,
        )
        step_count = max(1, int(planned_steps))
        reserve_cost = float(quote["effective_cost_usd"]) * step_count * 1.20
        reserve_credits = int(math.ceil(reserve_cost * self._credits_per_usd))
        return max(self._min_reserve, reserve_credits)

    async def remaining_credits(self, *, principal_id: int) -> int:
        await self._ensure_defaults()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await self._ensure_balance(conn, principal_id)
                row = await conn.fetchrow(
                    """
                    SELECT balance_credits, overdraft_limit
                    FROM billing.credit_balances
                    WHERE tenant_id=$1 AND principal_id=$2;
                    """,
                    self._tenant_id,
                    principal_id,
                )
                if row is None:
                    return 0
                reserved = await conn.fetchval(
                    """
                    SELECT COALESCE(SUM(reserved_credits), 0)
                    FROM billing.credit_reservations
                    WHERE tenant_id=$1
                      AND principal_id=$2
                      AND status='reserved'
                      AND expires_at > now();
                    """,
                    self._tenant_id,
                    principal_id,
                )
                balance = int(row["balance_credits"])
                overdraft = int(row["overdraft_limit"])
                reserved_total = int(reserved or 0)
                return balance + overdraft - reserved_total

    async def reservation_snapshot(self, *, request_key: bytes) -> dict[str, Any] | None:
        await self._ensure_defaults()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT reservation_id, principal_id, reserved_credits, status, expires_at
                FROM billing.credit_reservations
                WHERE tenant_id=$1 AND request_key=$2
                LIMIT 1;
                """,
                self._tenant_id,
                request_key,
            )
        if row is None:
            return None
        return {
            "reservation_id": row["reservation_id"],
            "principal_id": row["principal_id"],
            "reserved_credits": int(row["reserved_credits"] or 0),
            "status": str(row["status"] or ""),
            "expires_at": row["expires_at"],
        }

    async def usage_credits_for_workflow(self, *, workflow_id: uuid.UUID) -> int:
        async with self._pool.acquire() as conn:
            used = await conn.fetchval(
                """
                SELECT COALESCE(SUM(credits_charged), 0)
                FROM billing.usage_events
                WHERE tenant_id=$1 AND workflow_id=$2;
                """,
                self._tenant_id,
                workflow_id,
            )
        return int(used or 0)

    async def credits_to_usd(self, *, credits: int) -> float:
        await self._ensure_defaults()
        if self._credits_per_usd <= 0:
            return 0.0
        return float(int(credits) / self._credits_per_usd)

    async def quote_usage(
        self,
        *,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
    ) -> dict[str, Any]:
        await self._ensure_defaults()
        rates = _resolve_model_rates(self._pricing_json, provider=provider, model=model)
        tokens_in_i = max(0, int(tokens_in))
        tokens_out_i = max(0, int(tokens_out))
        input_cost = (tokens_in_i / 1000.0) * rates["input_per_1k_usd"]
        output_cost = (tokens_out_i / 1000.0) * rates["output_per_1k_usd"]
        effective_cost = max(0.0, input_cost + output_cost)
        credits = int(math.ceil(effective_cost * self._credits_per_usd)) if effective_cost > 0 else 0
        return {
            "provider": provider,
            "model": model,
            "tokens_in": tokens_in_i,
            "tokens_out": tokens_out_i,
            "tokens_total": tokens_in_i + tokens_out_i,
            "cost_usd": effective_cost,
            "effective_cost_usd": effective_cost,
            "credits_charged": credits,
        }

    async def record_operator_usage(
        self,
        *,
        principal_id: int,
        workflow_id: uuid.UUID,
        request_key: bytes,
        step_id: str,
        operator_name: str,
        operator_version: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        template_id: str | None = None,
        template_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UsageWriteResult:
        await self._ensure_defaults()
        quote = await self.quote_usage(
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        credits_charged = int(quote["credits_charged"])
        usage_payload: dict[str, Any] = {
            "tokens_in": int(quote["tokens_in"]),
            "tokens_out": int(quote["tokens_out"]),
            "tokens_total": int(quote["tokens_total"]),
            "step_id": step_id,
            "operator_name": operator_name,
            "operator_version": operator_version,
        }
        if metadata:
            usage_payload["metadata"] = dict(metadata)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchrow(
                    """
                    SELECT usage_event_id, credits_charged
                    FROM billing.usage_events
                    WHERE tenant_id=$1 AND workflow_id=$2 AND step_id=$3
                    LIMIT 1;
                    """,
                    self._tenant_id,
                    workflow_id,
                    step_id,
                )
                if existing is not None:
                    total_credits = await conn.fetchval(
                        """
                        SELECT COALESCE(SUM(credits_charged), 0)
                        FROM billing.usage_events
                        WHERE tenant_id=$1 AND workflow_id=$2;
                        """,
                        self._tenant_id,
                        workflow_id,
                    )
                    return UsageWriteResult(
                        ok=True,
                        usage_event_id=existing["usage_event_id"],
                        credits_charged=int(existing["credits_charged"] or 0),
                        total_credits_charged=int(total_credits or 0),
                        effective_cost_usd=float(quote["effective_cost_usd"]),
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
                    return UsageWriteResult(
                        ok=False,
                        usage_event_id=None,
                        credits_charged=0,
                        total_credits_charged=0,
                        effective_cost_usd=float(quote["effective_cost_usd"]),
                        reason="reservation_missing",
                    )
                if str(reservation["status"]) != "reserved":
                    return UsageWriteResult(
                        ok=False,
                        usage_event_id=None,
                        credits_charged=0,
                        total_credits_charged=0,
                        effective_cost_usd=float(quote["effective_cost_usd"]),
                        reason="reservation_not_active",
                    )

                used_credits = await conn.fetchval(
                    """
                    SELECT COALESCE(SUM(credits_charged), 0)
                    FROM billing.usage_events
                    WHERE tenant_id=$1 AND workflow_id=$2;
                    """,
                    self._tenant_id,
                    workflow_id,
                )
                used_credits_i = int(used_credits or 0)
                reserved_credits_i = int(reservation["reserved_credits"] or 0)
                projected = used_credits_i + credits_charged
                if projected > reserved_credits_i:
                    return UsageWriteResult(
                        ok=False,
                        usage_event_id=None,
                        credits_charged=0,
                        total_credits_charged=used_credits_i,
                        effective_cost_usd=float(quote["effective_cost_usd"]),
                        reason="budget_exceeded",
                    )

                usage_event_id = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO billing.usage_events (
                      tenant_id, usage_event_id, principal_id, workflow_id, job_id,
                      operation_type, provider, model, usage, cost_usd, effective_cost_usd,
                      credits_charged, pricing_version_id, credit_rate_version, estimated,
                      template_id, template_hash, step_id, operator_name, operator_version
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20);
                    """,
                    self._tenant_id,
                    usage_event_id,
                    principal_id,
                    workflow_id,
                    None,
                    "operator",
                    provider,
                    model,
                    json.dumps(usage_payload),
                    float(quote["cost_usd"]),
                    float(quote["effective_cost_usd"]),
                    credits_charged,
                    self._pricing_version_id,
                    self._credit_rate_version,
                    False,
                    template_id,
                    template_hash,
                    step_id,
                    operator_name,
                    operator_version,
                )
                return UsageWriteResult(
                    ok=True,
                    usage_event_id=usage_event_id,
                    credits_charged=credits_charged,
                    total_credits_charged=projected,
                    effective_cost_usd=float(quote["effective_cost_usd"]),
                )

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

                reserved_credits = int(reservation["reserved_credits"] or 0)
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
                    json.dumps({"reserved_credits": reserved_credits, "actual_credits": actual_credits}),
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

    async def enqueue_settlement_retry(
        self,
        *,
        principal_id: int,
        workflow_id: uuid.UUID,
        request_key: bytes,
        reason: str | None,
    ) -> uuid.UUID:
        await self._ensure_defaults()
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT outbox_id
                    FROM billing.credit_settlement_outbox
                    WHERE tenant_id=$1 AND request_key=$2
                    LIMIT 1;
                    """,
                    self._tenant_id,
                    request_key,
                )
                if row is not None:
                    outbox_id = row["outbox_id"]
                    await conn.execute(
                        """
                        UPDATE billing.credit_settlement_outbox
                        SET status='pending',
                            next_attempt_at=now(),
                            last_error=$3,
                            updated_at=now()
                        WHERE tenant_id=$1 AND outbox_id=$2;
                        """,
                        self._tenant_id,
                        outbox_id,
                        reason,
                    )
                    return outbox_id

                outbox_id = uuid.uuid4()
                await conn.execute(
                    """
                    INSERT INTO billing.credit_settlement_outbox (
                      tenant_id, outbox_id, principal_id, workflow_id, request_key,
                      status, attempt_count, next_attempt_at, last_error
                    ) VALUES ($1,$2,$3,$4,$5,'pending',0,now(),$6);
                    """,
                    self._tenant_id,
                    outbox_id,
                    principal_id,
                    workflow_id,
                    request_key,
                    reason,
                )
                return outbox_id

    async def process_settlement_outbox(self, *, max_items: int = 20) -> int:
        await self._ensure_defaults()
        if max_items <= 0:
            return 0

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    SELECT outbox_id, principal_id, workflow_id, request_key, attempt_count
                    FROM billing.credit_settlement_outbox
                    WHERE tenant_id=$1
                      AND status='pending'
                      AND next_attempt_at <= now()
                    ORDER BY next_attempt_at ASC
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED;
                    """,
                    self._tenant_id,
                    int(max_items),
                )

        processed = 0
        for row in rows:
            processed += 1
            outbox_id = row["outbox_id"]
            principal_id = int(row["principal_id"])
            workflow_id = row["workflow_id"]
            request_key = bytes(row["request_key"])
            attempt_count = int(row["attempt_count"] or 0)

            settlement = await self.settle(
                principal_id=principal_id,
                workflow_id=workflow_id,
                request_key=request_key,
            )
            async with self._pool.acquire() as conn:
                if settlement.ok:
                    await conn.execute(
                        """
                        UPDATE billing.credit_settlement_outbox
                        SET status='settled',
                            updated_at=now(),
                            last_error=NULL
                        WHERE tenant_id=$1 AND outbox_id=$2;
                        """,
                        self._tenant_id,
                        outbox_id,
                    )
                    continue

                next_attempt = attempt_count + 1
                if next_attempt >= self._settlement_outbox_max_retries:
                    await conn.execute(
                        """
                        UPDATE billing.credit_settlement_outbox
                        SET status='failed',
                            attempt_count=$3,
                            updated_at=now(),
                            last_error=$4
                        WHERE tenant_id=$1 AND outbox_id=$2;
                        """,
                        self._tenant_id,
                        outbox_id,
                        next_attempt,
                        settlement.reason,
                    )
                    continue

                delay_seconds = _outbox_retry_delay_seconds(next_attempt)
                await conn.execute(
                    """
                    UPDATE billing.credit_settlement_outbox
                    SET status='pending',
                        attempt_count=$3,
                        next_attempt_at=now() + ($4 * INTERVAL '1 second'),
                        updated_at=now(),
                        last_error=$5
                    WHERE tenant_id=$1 AND outbox_id=$2;
                    """,
                    self._tenant_id,
                    outbox_id,
                    next_attempt,
                    delay_seconds,
                    settlement.reason,
                )

        return processed

    async def _ensure_defaults(self) -> None:
        if self._pricing_version_id and self._credit_rate_version:
            return
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    SELECT credit_rate_version, credits_per_usd
                    FROM billing.credit_rate_versions
                    WHERE effective_to IS NULL
                    ORDER BY effective_from DESC
                    LIMIT 1;
                    """,
                )
                if row is None:
                    credit_rate_version = "dev-v1"
                    credits_per_usd = 100.0
                    await conn.execute(
                        """
                        INSERT INTO billing.credit_rate_versions (
                          credit_rate_version, credits_per_usd, effective_from
                        ) VALUES ($1,$2,now());
                        """,
                        credit_rate_version,
                        credits_per_usd,
                    )
                else:
                    credit_rate_version = str(row["credit_rate_version"])
                    credits_per_usd = float(row["credits_per_usd"])

                pricing = await conn.fetchrow(
                    """
                    SELECT pricing_version_id, pricing_json
                    FROM billing.pricing_versions
                    WHERE effective_to IS NULL
                    ORDER BY effective_from DESC
                    LIMIT 1;
                    """,
                )
                if pricing is None:
                    pricing_version_id = uuid.uuid4()
                    pricing_json = dict(_DEFAULT_PRICING_JSON)
                    await conn.execute(
                        """
                        INSERT INTO billing.pricing_versions (
                          pricing_version_id, provider, version, effective_from, pricing_json
                        ) VALUES ($1,$2,$3,now(),$4::jsonb);
                        """,
                        pricing_version_id,
                        "openai",
                        "gpt-5-nano-2026-02",
                        json.dumps(pricing_json),
                    )
                else:
                    pricing_version_id = pricing["pricing_version_id"]
                    pricing_json_raw = pricing["pricing_json"]
                    if isinstance(pricing_json_raw, str):
                        try:
                            pricing_json = json.loads(pricing_json_raw)
                        except json.JSONDecodeError:
                            pricing_json = dict(_DEFAULT_PRICING_JSON)
                    elif isinstance(pricing_json_raw, dict):
                        pricing_json = dict(pricing_json_raw)
                    else:
                        pricing_json = dict(_DEFAULT_PRICING_JSON)

        self._credit_rate_version = credit_rate_version
        self._pricing_version_id = pricing_version_id
        self._credits_per_usd = float(credits_per_usd)
        self._pricing_json = pricing_json

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


def _estimate_prompt_tokens(message: str) -> int:
    text = (message or "").strip()
    if not text:
        return 64
    return max(64, int(len(text) / 4) + 32)


def _resolve_model_rates(pricing_json: dict[str, Any], *, provider: str, model: str) -> dict[str, float]:
    default_models = _DEFAULT_PRICING_JSON.get("models", {})
    target_key = f"{provider}:{model}"
    models = pricing_json.get("models") if isinstance(pricing_json.get("models"), dict) else {}
    raw = models.get(target_key)
    if not isinstance(raw, dict):
        raw = default_models.get(target_key)
    if not isinstance(raw, dict):
        raw = default_models.get("openai:gpt-5-nano", {})
    input_rate = float(raw.get("input_per_1k_usd", 0.00005))
    output_rate = float(raw.get("output_per_1k_usd", 0.00040))
    return {"input_per_1k_usd": input_rate, "output_per_1k_usd": output_rate}


def _outbox_retry_delay_seconds(attempt_count: int) -> int:
    base = min(300, 2 ** min(10, max(0, attempt_count)))
    jitter = random.randint(0, 5)
    return int(base + jitter)


def request_key_for_workflow(workflow_id: uuid.UUID) -> bytes:
    return blake3(str(workflow_id).encode("utf-8")).digest()
