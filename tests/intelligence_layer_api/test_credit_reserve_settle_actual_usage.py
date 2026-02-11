from __future__ import annotations

import uuid
from typing import Any

import pytest

from intelligence_layer_api.billing import CreditManager, request_key_for_workflow


class _Tx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return False


class _Acquire:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        return False


class _FakeConn:
    def __init__(self) -> None:
        self.balance_credits = 1000
        self.reservation: dict[str, Any] | None = {"reserved_credits": 100, "status": "reserved"}
        self.usage_events: list[dict[str, Any]] = []
        self.ledger: dict[str, Any] | None = None

    def transaction(self):
        return _Tx()

    async def fetchrow(self, sql: str, *args):
        if "FROM billing.credit_ledger" in sql:
            return self.ledger
        if "FROM billing.credit_reservations" in sql:
            return self.reservation
        if "FROM billing.credit_balances" in sql:
            return {"balance_credits": self.balance_credits}
        if "FROM billing.usage_events" in sql and "step_id=$3" in sql:
            step_id = args[2]
            for row in self.usage_events:
                if row["step_id"] == step_id:
                    return {"usage_event_id": row["usage_event_id"], "credits_charged": row["credits_charged"]}
            return None
        return None

    async def fetchval(self, sql: str, *args):
        _ = args
        if "SUM(credits_charged)" in sql:
            return sum(int(item["credits_charged"]) for item in self.usage_events)
        return None

    async def execute(self, sql: str, *args):
        if "INSERT INTO billing.usage_events" in sql:
            self.usage_events.append(
                {
                    "usage_event_id": args[1],
                    "credits_charged": int(args[11]),
                    "step_id": str(args[17]),
                }
            )
            return
        if "UPDATE billing.credit_balances" in sql:
            self.balance_credits = int(args[2])
            return
        if "INSERT INTO billing.credit_ledger" in sql:
            self.ledger = {
                "credit_ledger_id": args[1],
                "delta_credits": int(args[5]),
                "balance_after": int(args[6]),
            }
            return
        if "UPDATE billing.credit_reservations" in sql:
            if self.reservation is not None:
                self.reservation["status"] = "settled"
            return


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def acquire(self):
        return _Acquire(self._conn)


def _build_manager(conn: _FakeConn) -> CreditManager:
    manager = CreditManager(
        pool=_FakePool(conn),
        tenant_id=1,
        bootstrap_enabled=False,
        bootstrap_credits=1000,
        reservation_ttl_sec=900,
        min_reserve_credits=1,
    )
    manager._pricing_version_id = uuid.uuid4()
    manager._credit_rate_version = "dev-v1"
    manager._credits_per_usd = 100.0
    manager._pricing_json = {
        "models": {
            "openai:gpt-5-nano": {"input_per_1k_usd": 0.00005, "output_per_1k_usd": 0.00040},
        }
    }
    return manager


@pytest.mark.asyncio
async def test_usage_is_recorded_before_settlement_and_settled_from_actual_usage() -> None:
    conn = _FakeConn()
    manager = _build_manager(conn)
    workflow_id = uuid.uuid4()
    request_key = request_key_for_workflow(workflow_id)

    usage = await manager.record_operator_usage(
        principal_id=77,
        workflow_id=workflow_id,
        request_key=request_key,
        step_id="s1",
        operator_name="Email.ReviewDraft",
        operator_version="1.0.0",
        provider="openai",
        model="gpt-5-nano",
        tokens_in=1200,
        tokens_out=600,
        template_id="Email.ReviewDraft/1.0.0/review_draft",
        template_hash="hash-1",
    )

    assert usage.ok is True
    assert len(conn.usage_events) == 1

    settled = await manager.settle(
        principal_id=77,
        workflow_id=workflow_id,
        request_key=request_key,
    )

    assert settled.ok is True
    assert settled.debited_credits == usage.credits_charged
    assert conn.ledger is not None
    assert conn.ledger["delta_credits"] == -usage.credits_charged


@pytest.mark.asyncio
async def test_settle_without_usage_does_not_backfill_estimated_credits() -> None:
    conn = _FakeConn()
    manager = _build_manager(conn)
    workflow_id = uuid.uuid4()
    request_key = request_key_for_workflow(workflow_id)

    settled = await manager.settle(
        principal_id=77,
        workflow_id=workflow_id,
        request_key=request_key,
    )

    assert settled.ok is True
    assert settled.debited_credits == 0
    assert conn.ledger is not None
    assert conn.ledger["delta_credits"] == 0
