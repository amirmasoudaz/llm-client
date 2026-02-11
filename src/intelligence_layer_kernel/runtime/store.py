from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass
from typing import Any

from blake3 import blake3

from .types import IntentRecord, PlanRecord


@dataclass(frozen=True)
class WorkflowRun:
    workflow_id: uuid.UUID
    correlation_id: uuid.UUID
    thread_id: int | None
    scope_type: str | None
    scope_id: str | None
    intent_id: uuid.UUID
    plan_id: uuid.UUID | None
    status: str
    execution_mode: str
    replay_mode: str
    parent_workflow_id: uuid.UUID | None = None


class IntentStore:
    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def insert(self, intent: IntentRecord) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ledger.intents (
                  tenant_id, intent_id, intent_type, schema_version, source,
                  thread_id, scope_type, scope_id,
                  actor, inputs, constraints, context_refs, data_classes,
                  correlation_id, producer_kind, producer_name, producer_version, created_at
                ) VALUES (
                  $1,$2,$3,$4,$5,
                  $6,$7,$8,
                  $9::jsonb,$10::jsonb,$11::jsonb,$12::jsonb,$13,
                  $14,'kernel','workflow_kernel','1.0',now()
                );
                """,
                self._tenant_id,
                intent.intent_id,
                intent.intent_type,
                intent.schema_version,
                intent.source,
                intent.thread_id,
                intent.scope_type,
                intent.scope_id,
                json.dumps(intent.actor),
                json.dumps(intent.inputs),
                json.dumps(intent.constraints),
                json.dumps(intent.context_refs),
                intent.data_classes,
                intent.correlation_id,
            )

    async def fetch(self, intent_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT intent_id, intent_type, schema_version, source, thread_id, scope_type, scope_id,
                       actor, inputs, constraints, context_refs, data_classes, correlation_id, created_at
                FROM ledger.intents
                WHERE tenant_id=$1 AND intent_id=$2;
                """,
                self._tenant_id,
                intent_id,
            )
            if not row:
                return None
            return {
                "intent_id": row["intent_id"],
                "intent_type": row["intent_type"],
                "schema_version": row["schema_version"],
                "source": row["source"],
                "thread_id": row["thread_id"],
                "scope_type": row["scope_type"],
                "scope_id": row["scope_id"],
                "actor": _coerce_json(row["actor"]),
                "inputs": _coerce_json(row["inputs"]),
                "constraints": _coerce_json(row["constraints"]),
                "context_refs": _coerce_json(row["context_refs"]),
                "data_classes": list(row["data_classes"] or []),
                "correlation_id": row["correlation_id"],
                "created_at": row["created_at"],
            }


class PlanStore:
    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def insert(self, plan: PlanRecord, plan_hash: bytes) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ledger.plans (
                  tenant_id, plan_id, intent_id, schema_version, planner_name, planner_version,
                  plan, plan_hash
                ) VALUES (
                  $1,$2,$3,$4,'planner','1.0',$5::jsonb,$6
                );
                """,
                self._tenant_id,
                plan.plan_id,
                plan.intent_id,
                plan.schema_version,
                json.dumps(plan.to_dict()),
                plan_hash,
            )

    async def fetch(self, plan_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT plan_id, intent_id, plan
                FROM ledger.plans
                WHERE tenant_id=$1 AND plan_id=$2;
                """,
                self._tenant_id,
                plan_id,
            )
            if not row:
                return None
            return {
                "plan_id": row["plan_id"],
                "intent_id": row["intent_id"],
                "plan": _coerce_json(row["plan"]),
            }


class WorkflowStore:
    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def create_run(
        self,
        *,
        workflow_id: uuid.UUID,
        correlation_id: uuid.UUID,
        thread_id: int | None,
        scope_type: str | None,
        scope_id: str | None,
        intent_id: uuid.UUID,
        plan_id: uuid.UUID,
        status: str,
        execution_mode: str,
        replay_mode: str,
        request_key: bytes | None = None,
        parent_workflow_id: uuid.UUID | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runtime.workflow_runs (
                  tenant_id, workflow_id, correlation_id, thread_id, scope_type, scope_id,
                  intent_id, plan_id, status, execution_mode, replay_mode, request_key,
                  parent_workflow_id, created_at, started_at, updated_at
                ) VALUES (
                  $1,$2,$3,$4,$5,$6,
                  $7,$8,$9,$10,$11,$12,
                  $13,now(),now(),now()
                );
                """,
                self._tenant_id,
                workflow_id,
                correlation_id,
                thread_id,
                scope_type,
                scope_id,
                intent_id,
                plan_id,
                status,
                execution_mode,
                replay_mode,
                request_key,
                parent_workflow_id,
            )

    async def update_run_status(self, *, workflow_id: uuid.UUID, status: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runtime.workflow_runs
                SET status=$3, updated_at=now()
                WHERE tenant_id=$1 AND workflow_id=$2;
                """,
                self._tenant_id,
                workflow_id,
                status,
            )

    async def finish_run(self, *, workflow_id: uuid.UUID, status: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runtime.workflow_runs
                SET status=$3, completed_at=now(), updated_at=now()
                WHERE tenant_id=$1 AND workflow_id=$2;
                """,
                self._tenant_id,
                workflow_id,
                status,
            )

    async def get_run(self, *, workflow_id: uuid.UUID) -> WorkflowRun | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT workflow_id, correlation_id, thread_id, scope_type, scope_id,
                       intent_id, plan_id, status, execution_mode, replay_mode, parent_workflow_id
                FROM runtime.workflow_runs
                WHERE tenant_id=$1 AND workflow_id=$2;
                """,
                self._tenant_id,
                workflow_id,
            )
            if not row:
                return None
            return WorkflowRun(
                workflow_id=row["workflow_id"],
                correlation_id=row["correlation_id"],
                thread_id=row["thread_id"],
                scope_type=row["scope_type"],
                scope_id=row["scope_id"],
                intent_id=row["intent_id"],
                plan_id=row["plan_id"],
                status=row["status"],
                execution_mode=row["execution_mode"],
                replay_mode=row["replay_mode"],
                parent_workflow_id=row["parent_workflow_id"],
            )

    async def create_steps(self, *, workflow_id: uuid.UUID, steps: list[dict[str, Any]]) -> None:
        async with self._pool.acquire() as conn:
            for step in steps:
                await conn.execute(
                    """
                    INSERT INTO runtime.workflow_steps (
                      tenant_id, workflow_id, step_id, kind, name, operator_name, operator_version,
                      effects, policy_tags, risk_level, cache_policy, status
                    ) VALUES (
                      $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'PENDING'
                    );
                    """,
                    self._tenant_id,
                    workflow_id,
                    step["step_id"],
                    step["kind"],
                    step["name"],
                    step.get("operator_name"),
                    step.get("operator_version"),
                    step.get("effects", []),
                    step.get("policy_tags", []),
                    step.get("risk_level", "low"),
                    step.get("cache_policy", "never"),
                )

    async def list_steps(self, *, workflow_id: uuid.UUID) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT step_id, kind, name, operator_name, operator_version, effects, policy_tags,
                       risk_level, cache_policy, idempotency_key, input_payload, input_hash,
                       status, attempt_count, next_retry_at, lease_owner, lease_expires_at,
                       last_job_id, gate_id, created_at, started_at, finished_at, updated_at
                FROM runtime.workflow_steps
                WHERE tenant_id=$1 AND workflow_id=$2
                ORDER BY step_id ASC;
                """,
                self._tenant_id,
                workflow_id,
            )
            return [dict(row) for row in rows]

    async def mark_step_ready(self, *, workflow_id: uuid.UUID, step_id: str) -> None:
        await self._update_step(workflow_id, step_id, status="READY", next_retry_at=None)

    async def mark_step_running(self, *, workflow_id: uuid.UUID, step_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runtime.workflow_steps
                SET status='RUNNING',
                    attempt_count=attempt_count + 1,
                    started_at=COALESCE(started_at, now()),
                    updated_at=now()
                WHERE tenant_id=$1 AND workflow_id=$2 AND step_id=$3;
                """,
                self._tenant_id,
                workflow_id,
                step_id,
            )

    async def mark_step_waiting(self, *, workflow_id: uuid.UUID, step_id: str, gate_id: uuid.UUID) -> None:
        await self._update_step(workflow_id, step_id, status="WAITING_APPROVAL", gate_id=gate_id)

    async def mark_step_succeeded(self, *, workflow_id: uuid.UUID, step_id: str) -> None:
        await self._update_step(workflow_id, step_id, status="SUCCEEDED", finished_at=True)

    async def mark_step_failed(self, *, workflow_id: uuid.UUID, step_id: str, status: str) -> None:
        await self._update_step(workflow_id, step_id, status=status, finished_at=True)

    async def mark_step_cancelled(self, *, workflow_id: uuid.UUID, step_id: str) -> None:
        await self._update_step(workflow_id, step_id, status="CANCELLED", finished_at=True)

    async def update_step_payload(
        self,
        *,
        workflow_id: uuid.UUID,
        step_id: str,
        idempotency_key: str | None,
        input_payload: dict[str, Any],
        last_job_id: uuid.UUID | None = None,
    ) -> None:
        payload_json = json.dumps(input_payload, sort_keys=True, separators=(",", ":"))
        input_hash = blake3(payload_json.encode("utf-8")).digest()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runtime.workflow_steps
                SET idempotency_key=$4,
                    input_payload=$5::jsonb,
                    input_hash=$6,
                    last_job_id=$7,
                    updated_at=now()
                WHERE tenant_id=$1 AND workflow_id=$2 AND step_id=$3;
                """,
                self._tenant_id,
                workflow_id,
                step_id,
                idempotency_key,
                json.dumps(input_payload),
                input_hash,
                last_job_id,
            )

    async def _update_step(
        self,
        workflow_id: uuid.UUID,
        step_id: str,
        *,
        status: str,
        gate_id: uuid.UUID | None = None,
        finished_at: bool = False,
        next_retry_at: Any | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE runtime.workflow_steps
                SET status=$4,
                    gate_id=COALESCE($5, gate_id),
                    finished_at=CASE WHEN $6 THEN now() ELSE finished_at END,
                    next_retry_at=$7,
                    updated_at=now()
                WHERE tenant_id=$1 AND workflow_id=$2 AND step_id=$3;
                """,
                self._tenant_id,
                workflow_id,
                step_id,
                status,
                gate_id,
                finished_at,
                next_retry_at,
            )


class OutcomeStore:
    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def record(
        self,
        *,
        workflow_id: uuid.UUID,
        thread_id: int | None,
        intent_id: uuid.UUID | None,
        plan_id: uuid.UUID | None,
        step_id: str | None,
        job_id: uuid.UUID | None,
        operator_name: str,
        operator_version: str,
        status: str,
        content: dict[str, Any] | None,
        template_id: str | None = None,
        template_hash: str | None = None,
        replay_mode: str = "reproduce",
        parent_workflow_id: uuid.UUID | None = None,
    ) -> uuid.UUID | None:
        if content is None:
            return None
        content_to_store = copy.deepcopy(content)
        outcome_id = uuid.uuid4()
        lineage_id = outcome_id
        parent_outcome_id: uuid.UUID | None = None
        version = 1
        if replay_mode == "regenerate" and parent_workflow_id is not None:
            parent = await self._find_parent_outcome(
                parent_workflow_id=parent_workflow_id,
                thread_id=thread_id,
                step_id=step_id,
                outcome_type=operator_name,
            )
            if parent is not None:
                lineage_id = parent["lineage_id"]
                parent_outcome_id = parent["outcome_id"]
                prior_version = parent["version"]
                if isinstance(prior_version, int) and prior_version > 0:
                    version = prior_version + 1
                else:
                    version = 2
        elif operator_name == "Email.OptimizeDraft":
            draft_parent = await self._resolve_email_draft_parent(
                thread_id=thread_id,
                content=content_to_store,
            )
            if draft_parent is not None:
                lineage_id = draft_parent["lineage_id"]
                parent_outcome_id = draft_parent["outcome_id"]
                prior_version = draft_parent["version"]
                if isinstance(prior_version, int) and prior_version > 0:
                    version = prior_version + 1
                else:
                    version = 2

        if operator_name == "Email.OptimizeDraft":
            _sync_email_draft_lineage_payload(
                content=content_to_store,
                version=version,
                parent_outcome_id=parent_outcome_id,
            )

        payload_json = json.dumps(content_to_store, sort_keys=True, separators=(",", ":"))
        content_hash = blake3(payload_json.encode("utf-8")).digest()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ledger.outcomes (
                  tenant_id, outcome_id, lineage_id, version, parent_outcome_id,
                  outcome_type, schema_version, status, visibility,
                  workflow_id, thread_id, intent_id, plan_id, step_id, job_id,
                  content, content_hash, template_id, template_hash,
                  confidence, data_classes,
                  producer_kind, producer_name, producer_version, created_at
                ) VALUES (
                  $1,$2,$3,$4,$5,
                  $6,'1.0',$7,'private',
                  $8,$9,$10,$11,$12,$13,
                  $14::jsonb,$15,$16,$17,
                  NULL,$18,
                  'operator',$19,$20,now()
                );
                """,
                self._tenant_id,
                outcome_id,
                lineage_id,
                version,
                parent_outcome_id,
                operator_name,
                status,
                workflow_id,
                thread_id,
                intent_id,
                plan_id,
                step_id,
                job_id,
                json.dumps(content_to_store),
                content_hash,
                template_id,
                template_hash,
                [],
                operator_name,
                operator_version,
            )
        return outcome_id

    async def _resolve_email_draft_parent(
        self,
        *,
        thread_id: int | None,
        content: dict[str, Any],
    ) -> dict[str, Any] | None:
        if thread_id is None:
            return None
        source_outcome_id = _extract_email_draft_source_outcome_id(content)
        async with self._pool.acquire() as conn:
            if source_outcome_id is not None:
                row = await conn.fetchrow(
                    """
                    SELECT outcome_id, lineage_id, version
                    FROM ledger.outcomes
                    WHERE tenant_id=$1
                      AND thread_id=$2
                      AND outcome_id=$3
                      AND status='succeeded'
                      AND content->'outcome'->>'outcome_type'='Email.Draft'
                    LIMIT 1;
                    """,
                    self._tenant_id,
                    thread_id,
                    source_outcome_id,
                )
                if row:
                    return {
                        "outcome_id": row["outcome_id"],
                        "lineage_id": row["lineage_id"],
                        "version": row["version"],
                    }

            row = await conn.fetchrow(
                """
                SELECT outcome_id, lineage_id, version
                FROM ledger.outcomes
                WHERE tenant_id=$1
                  AND thread_id=$2
                  AND status='succeeded'
                  AND content->'outcome'->>'outcome_type'='Email.Draft'
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                self._tenant_id,
                thread_id,
            )
            if not row:
                return None
            return {
                "outcome_id": row["outcome_id"],
                "lineage_id": row["lineage_id"],
                "version": row["version"],
            }

    async def list_by_workflow(self, *, workflow_id: uuid.UUID) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT outcome_id, lineage_id, version, parent_outcome_id,
                       outcome_type, status, workflow_id, step_id, content, template_id, template_hash, created_at
                FROM ledger.outcomes
                WHERE tenant_id=$1 AND workflow_id=$2
                ORDER BY created_at ASC;
                """,
                self._tenant_id,
                workflow_id,
            )
            return [
                {
                    "outcome_id": row["outcome_id"],
                    "lineage_id": row["lineage_id"],
                    "version": row["version"],
                    "parent_outcome_id": row["parent_outcome_id"],
                    "outcome_type": row["outcome_type"],
                    "status": row["status"],
                    "workflow_id": row["workflow_id"],
                    "step_id": row["step_id"],
                    "content": _coerce_json(row["content"]),
                    "template_id": row["template_id"],
                    "template_hash": row["template_hash"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    async def _find_parent_outcome(
        self,
        *,
        parent_workflow_id: uuid.UUID,
        thread_id: int | None,
        step_id: str | None,
        outcome_type: str,
    ) -> dict[str, Any] | None:
        if thread_id is None or step_id is None:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT outcome_id, lineage_id, version
                FROM ledger.outcomes
                WHERE tenant_id=$1
                  AND workflow_id=$2
                  AND thread_id=$3
                  AND step_id=$4
                  AND outcome_type=$5
                ORDER BY version DESC, created_at DESC
                LIMIT 1;
                """,
                self._tenant_id,
                parent_workflow_id,
                thread_id,
                step_id,
                outcome_type,
            )
            if not row:
                return None
            return {
                "outcome_id": row["outcome_id"],
                "lineage_id": row["lineage_id"],
                "version": row["version"],
            }


class GateStore:
    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def create_gate(
        self,
        *,
        workflow_id: uuid.UUID,
        step_id: str,
        gate_type: str,
        reason_code: str,
        title: str,
        preview: dict[str, Any],
        target_outcome_id: uuid.UUID | None,
        expires_at: Any | None,
    ) -> uuid.UUID:
        gate_id = uuid.uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ledger.gates (
                  tenant_id, gate_id, workflow_id, step_id, gate_type, reason_code,
                  summary, preview, target_outcome_id, status, expires_at
                ) VALUES (
                  $1,$2,$3,$4,$5,$6,
                  $7,$8::jsonb,$9,'waiting',$10
                );
                """,
                self._tenant_id,
                gate_id,
                workflow_id,
                step_id,
                gate_type,
                reason_code,
                title,
                json.dumps(preview),
                target_outcome_id,
                expires_at,
            )
        return gate_id

    async def get_gate(self, *, gate_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT gate_id, workflow_id, step_id, gate_type, reason_code, summary, preview,
                       target_outcome_id, status, expires_at
                FROM ledger.gates
                WHERE tenant_id=$1 AND gate_id=$2;
                """,
                self._tenant_id,
                gate_id,
            )
            if not row:
                return None
            return {
                "gate_id": row["gate_id"],
                "workflow_id": row["workflow_id"],
                "step_id": row["step_id"],
                "gate_type": row["gate_type"],
                "reason_code": row["reason_code"],
                "summary": row["summary"],
                "preview": _coerce_json(row["preview"]),
                "target_outcome_id": row["target_outcome_id"],
                "status": row["status"],
                "expires_at": row["expires_at"],
            }

    async def resolve_gate(
        self,
        *,
        gate_id: uuid.UUID,
        actor: dict[str, Any],
        decision: str,
        payload: dict[str, Any],
    ) -> uuid.UUID:
        gate_decision_id = uuid.uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ledger.gate_decisions (
                  tenant_id, gate_decision_id, gate_id, actor, decision, payload
                ) VALUES (
                  $1,$2,$3,$4::jsonb,$5,$6::jsonb
                );
                """,
                self._tenant_id,
                gate_decision_id,
                gate_id,
                json.dumps(actor),
                decision,
                json.dumps(payload),
            )
            await conn.execute(
                """
                UPDATE ledger.gates
                SET status=$3
                WHERE tenant_id=$1 AND gate_id=$2;
                """,
                self._tenant_id,
                gate_id,
                decision,
            )
        return gate_decision_id

    async def latest_decision(self, *, gate_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT gate_decision_id, decision, payload
                FROM ledger.gate_decisions
                WHERE tenant_id=$1 AND gate_id=$2
                ORDER BY created_at DESC
                LIMIT 1;
                """,
                self._tenant_id,
                gate_id,
            )
            if not row:
                return None
            return {
                "gate_decision_id": row["gate_decision_id"],
                "decision": row["decision"],
                "payload": _coerce_json(row["payload"]),
            }


def _coerce_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _extract_email_draft_source_outcome_id(content: dict[str, Any]) -> uuid.UUID | None:
    outcome = content.get("outcome")
    if not isinstance(outcome, dict):
        return None
    if str(outcome.get("outcome_type") or "") != "Email.Draft":
        return None
    payload = outcome.get("payload")
    if not isinstance(payload, dict):
        return None
    source_version_id = payload.get("source_version_id")
    if source_version_id is None:
        return None
    try:
        return uuid.UUID(str(source_version_id))
    except Exception:
        return None


def _sync_email_draft_lineage_payload(
    *,
    content: dict[str, Any],
    version: int,
    parent_outcome_id: uuid.UUID | None,
) -> None:
    outcome = content.get("outcome")
    if not isinstance(outcome, dict):
        return
    if str(outcome.get("outcome_type") or "") != "Email.Draft":
        return
    payload = outcome.get("payload")
    if not isinstance(payload, dict):
        return

    if version > 0:
        payload["version_number"] = version

    if parent_outcome_id is None:
        return
    source_version_id = payload.get("source_version_id")
    try:
        uuid.UUID(str(source_version_id))
    except Exception:
        payload["source_version_id"] = str(parent_outcome_id)
