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
                decision.correlation_id,
            )

        await self._events.append(
            LedgerEvent(
                tenant_id=self._tenant_id,
                event_id=uuid.uuid4(),
                workflow_id=uuid.UUID(decision.workflow_id) if decision.workflow_id else uuid.uuid4(),
                thread_id=None,
                intent_id=uuid.UUID(decision.intent_id) if decision.intent_id else None,
                plan_id=uuid.UUID(decision.plan_id) if decision.plan_id else None,
                step_id=decision.step_id,
                job_id=uuid.UUID(decision.job_id) if decision.job_id else None,
                policy_decision_id=policy_decision_id,
                event_type="policy.decision",
                actor={"type": "system", "id": "policy_engine", "role": "system"},
                payload={
                    "stage": decision.stage,
                    "decision": decision.decision,
                    "reason_code": decision.reason_code,
                    "reason": decision.reason,
                },
                correlation_id=uuid.UUID(decision.correlation_id) if decision.correlation_id else uuid.uuid4(),
                producer_kind="kernel",
                producer_name=decision.policy_engine_name,
                producer_version=decision.policy_engine_version,
            )
        )
        return policy_decision_id
