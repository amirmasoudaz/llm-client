from __future__ import annotations

import json
from typing import Any, AsyncIterator
import uuid
from datetime import datetime
from decimal import Decimal


class SSEProjector:
    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def list_events(
        self,
        *,
        workflow_id: str,
        after_event_no: int,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_no, event_id, event_type, severity, actor, payload, correlation_id,
                       workflow_id, thread_id, intent_id, plan_id, step_id, job_id, outcome_id,
                       gate_id, policy_decision_id, created_at
                FROM ledger.events
                WHERE tenant_id = $1 AND workflow_id = $2::uuid AND event_no > $3
                ORDER BY event_no ASC
                LIMIT $4;
                """,
                self._tenant_id,
                workflow_id,
                after_event_no,
                limit,
            )

        out: list[dict[str, Any]] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {"_raw": payload}
            actor = row["actor"]
            if isinstance(actor, str):
                try:
                    actor = json.loads(actor)
                except Exception:
                    actor = {"_raw": actor}
            out.append({
                "event_no": row["event_no"],
                "event_id": str(row["event_id"]),
                "event_type": row["event_type"],
                "severity": row["severity"],
                "actor": actor,
                "payload": payload,
                "correlation_id": str(row["correlation_id"]),
                "workflow_id": str(row["workflow_id"]),
                "thread_id": row["thread_id"],
                "intent_id": str(row["intent_id"]) if row["intent_id"] else None,
                "plan_id": str(row["plan_id"]) if row["plan_id"] else None,
                "step_id": row["step_id"],
                "job_id": str(row["job_id"]) if row["job_id"] else None,
                "outcome_id": str(row["outcome_id"]) if row["outcome_id"] else None,
                "gate_id": str(row["gate_id"]) if row["gate_id"] else None,
                "policy_decision_id": str(row["policy_decision_id"]) if row["policy_decision_id"] else None,
                "created_at": row["created_at"].timestamp() if row["created_at"] else None,
            })
        return out

    async def stream(
        self,
        *,
        workflow_id: str,
        terminal_event_types: set[str] | None = None,
    ) -> AsyncIterator[str]:
        last_event_no = 0
        terminal = terminal_event_types or {"final_result", "final_error", "job_cancelled"}

        while True:
            rows = await self.list_events(
                workflow_id=workflow_id,
                after_event_no=last_event_no,
                limit=200,
            )
            for event in rows:
                last_event_no = max(last_event_no, int(event["event_no"]))
                event_type = str(event["event_type"])
                payload = {
                    "event_no": event["event_no"],
                    "event_id": event["event_id"],
                    "type": event_type,
                    "timestamp": event["created_at"],
                    "workflow_id": event["workflow_id"],
                    "thread_id": event["thread_id"],
                    "intent_id": event["intent_id"],
                    "plan_id": event["plan_id"],
                    "step_id": event["step_id"],
                    "job_id": event["job_id"],
                    "correlation_id": event["correlation_id"],
                    "severity": event["severity"],
                    "actor": event["actor"],
                    "data": event["payload"],
                }
                event_name = event_type.replace(".", "_")
                yield f"event: {event_name}\ndata: {json.dumps(payload, default=_json_default)}\n\n"
                if event_name in terminal or event_type in terminal:
                    return

            import asyncio

            await asyncio.sleep(0.25)


def _json_default(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
