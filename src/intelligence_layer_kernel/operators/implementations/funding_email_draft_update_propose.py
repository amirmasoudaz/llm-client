from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .funding_request_fields_common import get_platform_db, to_iso_timestamp


class FundingEmailDraftUpdateProposeOperator(Operator):
    name = "FundingEmail.Draft.Update.Propose"
    version = "1.0.0"

    def __init__(self) -> None:
        self._db = get_platform_db()

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        funding_request_id = payload.get("funding_request_id")
        email_id = payload.get("email_id")
        draft_payload = _extract_draft_payload(payload.get("draft"))
        if isinstance(funding_request_id, bool) or not isinstance(funding_request_id, int) or funding_request_id <= 0:
            return _failed(start=start, code="invalid_payload", message="funding_request_id must be a positive integer")
        if email_id is not None and (isinstance(email_id, bool) or not isinstance(email_id, int) or email_id <= 0):
            return _failed(start=start, code="invalid_payload", message="email_id must be a positive integer when provided")
        if draft_payload is None:
            return _failed(start=start, code="invalid_payload", message="draft must be an Email.Draft outcome or payload")

        optimized_subject = _as_non_empty_text(draft_payload.get("subject"))
        optimized_body = _as_non_empty_text(draft_payload.get("body"))
        if optimized_subject is None or optimized_body is None:
            return _failed(start=start, code="invalid_draft", message="draft subject and body are required")

        snapshot = await self._load_snapshot(funding_request_id=funding_request_id, email_id=email_id)
        if snapshot is None:
            return _failed(start=start, code="funding_request_not_found", message="funding request not found")
        if snapshot.get("email_id") is None:
            return _failed(
                start=start,
                code="funding_email_not_found",
                message="no funding email draft exists; generate a draft first",
            )

        current_subject = _as_text(snapshot.get("request_email_subject"))
        current_body = _as_text(snapshot.get("request_email_body"))
        current_main_subject = _as_text(snapshot.get("main_email_subject"))
        current_main_body = _as_text(snapshot.get("main_email_body"))
        main_sent = _as_bool(snapshot.get("main_sent"))

        request_expected_updated_at = to_iso_timestamp(snapshot.get("request_updated_at"))
        proposal = {
            "schema_version": "1.0",
            "patch_id": str(uuid.uuid4()),
            "targets": [
                {
                    "table": "funding_requests",
                    "where": {"id": int(funding_request_id)},
                    "set": {
                        "email_subject": optimized_subject,
                        "email_content": optimized_body,
                    },
                    "expected": {
                        "updated_at": request_expected_updated_at,
                        "values": {
                            "email_subject": current_subject,
                            "email_content": current_body,
                        },
                    },
                },
                {
                    "table": "funding_emails",
                    "where": {"id": int(snapshot["email_id"])},
                    "set": {
                        "main_email_subject": optimized_subject,
                        "main_email_body": optimized_body,
                    },
                    "expected": {
                        "values": {
                            "main_sent": main_sent,
                            "main_email_subject": current_main_subject,
                            "main_email_body": current_main_body,
                        },
                    },
                },
            ],
            "human_summary": str(
                payload.get("human_summary") or "Apply optimized email subject/body to funding request"
            ),
            "risk_level": "medium",
            "requires_approval": True,
            "idempotency_key": call.idempotency_key,
        }

        outcome = _build_outcome(
            payload=proposal,
            outcome_type="PlatformPatch.Proposal",
            producer_name=self.name,
            producer_version=self.version,
        )
        preview = {
            "funding_request": {
                "before": {"email_subject": current_subject, "email_content": current_body},
                "after": {"email_subject": optimized_subject, "email_content": optimized_body},
            },
            "funding_email": {
                "email_id": int(snapshot["email_id"]),
                "before": {
                    "main_email_subject": current_main_subject,
                    "main_email_body": current_main_body,
                    "main_sent": main_sent,
                },
                "after": {"main_email_subject": optimized_subject, "main_email_body": optimized_body},
            },
            "can_apply": not bool(main_sent),
            "blocked_reason": "main_email_already_sent" if bool(main_sent) else None,
        }
        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome, "preview": preview},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )

    async def _load_snapshot(self, *, funding_request_id: int, email_id: int | None) -> dict[str, Any] | None:
        if email_id is not None:
            return await self._db.fetch_one(
                """
                SELECT fr.id AS request_id,
                       fr.email_subject AS request_email_subject,
                       fr.email_content AS request_email_body,
                       fr.updated_at AS request_updated_at,
                       fe.id AS email_id,
                       fe.main_email_subject,
                       fe.main_email_body,
                       fe.main_sent,
                       fe.main_sent_at
                FROM funding_requests fr
                LEFT JOIN funding_emails fe
                  ON fe.id=%s AND fe.funding_request_id=fr.id
                WHERE fr.id=%s
                LIMIT 1;
                """,
                (email_id, funding_request_id),
            )
        return await self._db.fetch_one(
            """
            SELECT fr.id AS request_id,
                   fr.email_subject AS request_email_subject,
                   fr.email_content AS request_email_body,
                   fr.updated_at AS request_updated_at,
                   fe.id AS email_id,
                   fe.main_email_subject,
                   fe.main_email_body,
                   fe.main_sent,
                   fe.main_sent_at
            FROM funding_requests fr
            LEFT JOIN funding_emails fe
              ON fe.funding_request_id=fr.id
            WHERE fr.id=%s
            ORDER BY fe.id DESC
            LIMIT 1;
            """,
            (funding_request_id,),
        )


def _extract_draft_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    payload = value.get("payload")
    if isinstance(payload, dict) and value.get("outcome_type") == "Email.Draft":
        return payload
    if "subject" in value and "body" in value:
        return value
    return None


def _as_non_empty_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        return bool(int(value))
    except Exception:
        return False


def _failed(*, start: float, code: str, message: str) -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(
            code=code,
            message=message,
            category="validation",
            retryable=False,
        ),
    )


def _build_outcome(
    *,
    payload: dict[str, Any],
    outcome_type: str,
    producer_name: str,
    producer_version: str,
) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": outcome_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {"name": producer_name, "version": producer_version, "plugin_type": "operator"},
    }
