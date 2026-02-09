from __future__ import annotations

import json
from typing import Any

from blake3 import blake3

from .types import LedgerEvent


class EventWriter:
    def __init__(self, *, pool) -> None:
        self._pool = pool

    async def append(self, event: LedgerEvent) -> None:
        payload_hash = _hash_payload(event.payload)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ledger.events (
                  tenant_id,
                  event_id,
                  schema_version,
                  workflow_id,
                  thread_id,
                  intent_id,
                  plan_id,
                  step_id,
                  job_id,
                  outcome_id,
                  gate_id,
                  policy_decision_id,
                  event_type,
                  severity,
                  actor,
                  payload,
                  payload_hash,
                  correlation_id,
                  producer_kind,
                  producer_name,
                  producer_version
                ) VALUES (
                  $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21
                );
                """,
                event.tenant_id,
                event.event_id,
                event.schema_version,
                event.workflow_id,
                event.thread_id,
                event.intent_id,
                event.plan_id,
                event.step_id,
                event.job_id,
                event.outcome_id,
                event.gate_id,
                event.policy_decision_id,
                event.event_type,
                event.severity,
                json.dumps(event.actor),
                json.dumps(event.payload),
                payload_hash,
                event.correlation_id,
                event.producer_kind,
                event.producer_name,
                event.producer_version,
            )


def _hash_payload(payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake3(raw).digest()
