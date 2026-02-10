from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorResult, OperatorMetrics, OperatorError


class EmailReviewDraftOperator(Operator):
    name = "Email.ReviewDraft"
    version = "1.0.0"

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        body = payload.get("body") or payload.get("fallback_body") or ""
        subject = payload.get("subject") or payload.get("fallback_subject") or ""

        if not isinstance(body, str) or body.strip() == "":
            error = OperatorError(
                code="missing_email_body",
                message="email body is required for review",
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

        verdict = "pass"
        issues: list[dict[str, Any]] = []
        if len(body.strip()) < 40:
            verdict = "needs_edits"
            issues.append(
                {
                    "code": "too_short",
                    "severity": "warning",
                    "message": "Email draft is very short; consider adding more context.",
                }
            )

        review_payload = {
            "verdict": verdict,
            "overall_score": 0.75 if verdict == "pass" else 0.45,
            "issues": issues,
            "suggested_subject": subject or None,
            "suggested_body": body if verdict != "pass" else None,
            "notes": None,
        }

        review_payload = {k: v for k, v in review_payload.items() if v is not None}
        outcome = _build_outcome(review_payload)

        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _build_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Email.Review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {"name": "Email.ReviewDraft", "version": "1.0.0", "plugin_type": "operator"},
    }
