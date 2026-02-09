from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from ..base import Operator
from ..types import OperatorCall, OperatorResult, OperatorError, OperatorMetrics
from ...runtime.store import GateStore, WorkflowStore


class WorkflowGateResolveOperator(Operator):
    name = "Workflow.Gate.Resolve"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id
        self._gate_store = GateStore(pool=pool, tenant_id=tenant_id)
        self._workflow_store = WorkflowStore(pool=pool, tenant_id=tenant_id)

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        action_id = payload.get("action_id")
        status = payload.get("status")
        decision_payload = payload.get("payload") or {}

        try:
            gate_id = uuid.UUID(str(action_id))
        except Exception:
            error = OperatorError(
                code="invalid_action_id",
                message="action_id must be a UUID",
                category="validation",
                retryable=False,
            )
            return OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=error,
            )

        gate = await self._gate_store.get_gate(gate_id=gate_id)
        if gate is None:
            error = OperatorError(
                code="gate_not_found",
                message="gate not found",
                category="validation",
                retryable=False,
            )
            return OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=error,
            )

        actor = {
            "tenant_id": call.auth_context.tenant_id,
            "principal": call.auth_context.principal,
            "role": "resolver",
            "scopes": call.auth_context.scopes,
        }
        await self._gate_store.resolve_gate(
            gate_id=gate_id,
            actor=actor,
            decision=status,
            payload=decision_payload,
        )

        if status == "accepted":
            await self._workflow_store.mark_step_ready(
                workflow_id=gate["workflow_id"],
                step_id=gate["step_id"],
            )
            await self._workflow_store.update_run_status(
                workflow_id=gate["workflow_id"],
                status="running",
            )
        else:
            await self._workflow_store.mark_step_cancelled(
                workflow_id=gate["workflow_id"],
                step_id=gate["step_id"],
            )
            await self._workflow_store.finish_run(
                workflow_id=gate["workflow_id"],
                status="cancelled",
            )

        result = {
            "action_id": str(gate_id),
            "status": status,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resumed_workflow_id": str(gate["workflow_id"]) if status == "accepted" else None,
        }

        return OperatorResult(
            status="succeeded",
            result=result,
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )
