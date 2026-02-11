from __future__ import annotations

import json
import uuid

from .types import PolicyDecision
from ..events import EventWriter, LedgerEvent


class PolicyDecisionStore:
    def __init__(self, *, pool, tenant_id: int, event_writer: EventWriter) -> None:
        self._pool = pool
        self._tenant_id = tenant_id
        self._events = event_writer

    async def record(self, decision: PolicyDecision) -> uuid.UUID:
        policy_decision_id = uuid.uuid4()
        correlation_uuid = _coerce_uuid(decision.correlation_id) or _ZERO_UUID
        workflow_uuid = _coerce_uuid(decision.workflow_id) or correlation_uuid
        intent_uuid = _coerce_uuid(decision.intent_id)
        plan_uuid = _coerce_uuid(decision.plan_id)
        job_uuid = _coerce_uuid(decision.job_id)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ledger.policy_decisions (
                  tenant_id, policy_decision_id, stage, decision, reason_code, reason,
                  requirements, limits, redactions, transform, inputs_hash,
                  policy_engine_name, policy_engine_version,
                  workflow_id, intent_id, plan_id, step_id, job_id,
                  correlation_id
                ) VALUES (
                  $1,$2,$3,$4,$5,$6,
                  $7::jsonb,$8::jsonb,$9::jsonb,$10::jsonb,$11,
                  $12,$13,
                  $14::uuid,$15::uuid,$16::uuid,$17,$18::uuid,
                  $19::uuid
                );
                """,
                self._tenant_id,
                policy_decision_id,
                decision.stage,
                decision.decision,
                decision.reason_code,
                decision.reason,
                json.dumps(decision.requirements),
                json.dumps(decision.limits),
                json.dumps(decision.redactions),
                json.dumps(decision.transform) if decision.transform is not None else None,
                decision.inputs_hash,
                decision.policy_engine_name,
                decision.policy_engine_version,
                decision.workflow_id,
                decision.intent_id,
                decision.plan_id,
                decision.step_id,
                decision.job_id,
                str(correlation_uuid),
            )

        await self._events.append(
            LedgerEvent(
                tenant_id=self._tenant_id,
                event_id=uuid.uuid4(),
                workflow_id=workflow_uuid,
                thread_id=None,
                intent_id=intent_uuid,
                plan_id=plan_uuid,
                step_id=decision.step_id,
                job_id=job_uuid,
                policy_decision_id=policy_decision_id,
                event_type="policy.decision",
                actor={"type": "system", "id": "policy_engine", "role": "system"},
                payload={
                    "stage": decision.stage,
                    "decision": decision.decision,
                    "reason_code": decision.reason_code,
                    "reason": decision.reason,
                },
                correlation_id=correlation_uuid,
                producer_kind="kernel",
                producer_name=decision.policy_engine_name,
                producer_version=decision.policy_engine_version,
            )
        )
        return policy_decision_id


_ZERO_UUID = uuid.UUID(int=0)


def _coerce_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
