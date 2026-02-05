from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorResult, OperatorMetrics, OperatorError


class FundingRequestFieldsUpdateProposeOperator(Operator):
    name = "FundingRequest.Fields.Update.Propose"
    version = "1.0.0"

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        funding_request_id = payload.get("funding_request_id")
        fields = payload.get("fields") or {}
        human_summary = payload.get("human_summary")

        if not funding_request_id or not isinstance(fields, dict) or not fields:
            error = OperatorError(
                code="invalid_payload",
                message="funding_request_id and fields are required",
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

        patch_id = uuid.uuid4()
        proposal = {
            "schema_version": "1.0",
            "patch_id": str(patch_id),
            "targets": [
                {
                    "table": "funding_requests",
                    "where": {"id": int(funding_request_id)},
                    "set": fields,
                }
            ],
            "human_summary": str(human_summary or "Update funding request fields"),
            "risk_level": "medium",
            "requires_approval": True,
            "idempotency_key": call.idempotency_key,
        }

        outcome_id = uuid.uuid4()
        payload_hash = _hash_obj(proposal)
        outcome = {
            "schema_version": "1.0",
            "outcome_id": str(outcome_id),
            "outcome_type": "PlatformPatch.Proposal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hash": payload_hash,
            "payload": proposal,
            "producer": {"name": self.name, "version": self.version, "plugin_type": "operator"},
        }

        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _hash_obj(value: Any) -> dict[str, Any]:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"alg": "blake3", "value": blake3(raw).hexdigest()}
