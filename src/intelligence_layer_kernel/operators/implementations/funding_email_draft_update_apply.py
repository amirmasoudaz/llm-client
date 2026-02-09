from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .funding_request_fields_common import get_platform_db, to_iso_timestamp, to_utc_datetime


class FundingEmailDraftUpdateApplyOperator(Operator):
    name = "FundingEmail.Draft.Update.Apply"
    version = "1.0.0"

    def __init__(self) -> None:
        self._db = get_platform_db()

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        proposal = _extract_proposal_payload(payload.get("proposal"))
        if proposal is None:
            return _failed(
                start=start,
                code="invalid_payload",
                message="proposal is required and must be a PlatformPatch.Proposal outcome/payload",
            )

        patch_id = _as_text(proposal.get("patch_id"))
        targets = proposal.get("targets")
        if patch_id is None or not isinstance(targets, list):
            return _failed(start=start, code="invalid_proposal", message="proposal must include patch_id and targets")

        request_target = _find_target(targets, "funding_requests")
        email_target = _find_target(targets, "funding_emails")
        if request_target is None or email_target is None:
            return _failed(
                start=start,
                code="invalid_proposal",
                message="proposal must include funding_requests and funding_emails targets",
            )

        request_id = _extract_positive_int(request_target.get("where"), "id")
        email_id = _extract_positive_int(email_target.get("where"), "id")
        if request_id is None or email_id is None:
            return _failed(start=start, code="invalid_proposal", message="proposal target identifiers are required")

        explicit_request_id = payload.get("funding_request_id")
        explicit_email_id = payload.get("email_id")
        if (
            isinstance(explicit_request_id, int)
            and explicit_request_id > 0
            and explicit_request_id != request_id
        ):
            return _failed(
                start=start,
                code="invalid_payload",
                message="funding_request_id does not match proposal target",
            )
        if isinstance(explicit_email_id, int) and explicit_email_id > 0 and explicit_email_id != email_id:
            return _failed(
                start=start,
                code="invalid_payload",
                message="email_id does not match proposal target",
            )

        request_set = request_target.get("set") if isinstance(request_target.get("set"), dict) else {}
        email_set = email_target.get("set") if isinstance(email_target.get("set"), dict) else {}
        if not request_set or not email_set:
            return _failed(start=start, code="invalid_proposal", message="proposal targets must include set values")

        new_request_subject = _as_non_empty_text(request_set.get("email_subject"))
        new_request_body = _as_non_empty_text(request_set.get("email_content"))
        new_main_subject = _as_non_empty_text(email_set.get("main_email_subject"))
        new_main_body = _as_non_empty_text(email_set.get("main_email_body"))
        if not all([new_request_subject, new_request_body, new_main_subject, new_main_body]):
            return _failed(
                start=start,
                code="invalid_proposal",
                message="email subject/body values must be non-empty strings",
            )

        snapshot = await self._load_snapshot(request_id=request_id, email_id=email_id)
        if snapshot is None:
            return _failed(start=start, code="record_not_found", message="funding request/email records not found")

        main_sent = _as_bool(snapshot.get("main_sent"))
        if main_sent:
            return _failed(
                start=start,
                code="email_already_sent",
                message="main email already sent; pivot to follow-up drafting",
                details={"pivot_intent": "Funding.Outreach.Email.Generate", "goal": "follow_up"},
            )

        strict_optimistic_lock = bool(payload.get("strict_optimistic_lock", True))
        if strict_optimistic_lock:
            conflict = _check_expected_values(
                request_target=request_target,
                email_target=email_target,
                snapshot=snapshot,
            )
            if conflict is not None:
                return _failed(
                    start=start,
                    code="stale_update_conflict",
                    message="email draft has changed; refresh and retry",
                    details=conflict,
                )

        current_request_subject = _as_text(snapshot.get("request_email_subject"))
        current_request_body = _as_text(snapshot.get("request_email_body"))
        current_main_subject = _as_text(snapshot.get("main_email_subject"))
        current_main_body = _as_text(snapshot.get("main_email_body"))

        request_changed = (
            current_request_subject != new_request_subject or current_request_body != new_request_body
        )
        email_changed = current_main_subject != new_main_subject or current_main_body != new_main_body

        request_rows = 0
        email_rows = 0
        pool = await self._db.pool()
        async with pool.acquire() as conn:
            await conn.begin()
            try:
                async with conn.cursor() as cur:
                    if request_changed:
                        req_sql = (
                            "UPDATE funding_requests "
                            "SET email_subject=%s, email_content=%s, updated_at=NOW() "
                            "WHERE id=%s"
                        )
                        req_params: list[Any] = [new_request_subject, new_request_body, request_id]
                        if strict_optimistic_lock:
                            expected_updated_at = _extract_expected_updated_at(request_target)
                            if expected_updated_at:
                                req_sql += " AND updated_at=%s"
                                req_params.append(snapshot.get("request_updated_at"))
                        await cur.execute(req_sql, tuple(req_params))
                        request_rows = int(cur.rowcount or 0)
                        if strict_optimistic_lock and request_rows == 0:
                            await conn.rollback()
                            return _failed(
                                start=start,
                                code="stale_update_conflict",
                                message="funding request changed during apply; refresh and retry",
                            )

                    if email_changed:
                        email_sql = (
                            "UPDATE funding_emails "
                            "SET main_email_subject=%s, main_email_body=%s "
                            "WHERE id=%s"
                        )
                        email_params: list[Any] = [new_main_subject, new_main_body, email_id]
                        if strict_optimistic_lock:
                            email_sql += " AND main_sent=0"
                        await cur.execute(email_sql, tuple(email_params))
                        email_rows = int(cur.rowcount or 0)
                        if strict_optimistic_lock and email_rows == 0:
                            await conn.rollback()
                            return _failed(
                                start=start,
                                code="stale_update_conflict",
                                message="funding email changed during apply; refresh and retry",
                            )

                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        applied_at = datetime.now(timezone.utc).isoformat()
        receipt_payload = {
            "schema_version": "1.0",
            "patch_id": patch_id,
            "applied": True,
            "applied_at": applied_at,
            "applied_by": _normalize_principal(call.auth_context.principal),
            "results": [
                {"table": "funding_requests", "where": {"id": request_id}, "updated_rows": request_rows},
                {"table": "funding_emails", "where": {"id": email_id}, "updated_rows": email_rows},
            ],
        }
        outcome = _build_outcome(
            payload=receipt_payload,
            outcome_type="PlatformPatch.Receipt",
            producer_name=self.name,
            producer_version=self.version,
        )
        return OperatorResult(
            status="succeeded",
            result={
                "outcome": outcome,
                "rows_affected": request_rows + email_rows,
                "request_id": request_id,
                "email_id": email_id,
                "applied_at": applied_at,
            },
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )

    async def _load_snapshot(self, *, request_id: int, email_id: int) -> dict[str, Any] | None:
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
            JOIN funding_emails fe
              ON fe.id=%s AND fe.funding_request_id=fr.id
            WHERE fr.id=%s
            LIMIT 1;
            """,
            (email_id, request_id),
        )


def _extract_proposal_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    payload = value.get("payload")
    if isinstance(payload, dict) and value.get("outcome_type") == "PlatformPatch.Proposal":
        return payload
    if isinstance(value.get("targets"), list):
        return value
    return None


def _find_target(targets: list[Any], table: str) -> dict[str, Any] | None:
    for item in targets:
        if not isinstance(item, dict):
            continue
        if str(item.get("table") or "") == table:
            return item
    return None


def _extract_positive_int(where: Any, key: str) -> int | None:
    if not isinstance(where, dict):
        return None
    value = where.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def _extract_expected_updated_at(target: dict[str, Any]) -> str | None:
    expected = target.get("expected")
    if not isinstance(expected, dict):
        return None
    value = expected.get("updated_at")
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _check_expected_values(
    *,
    request_target: dict[str, Any],
    email_target: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any] | None:
    expected_request_updated_at = _extract_expected_updated_at(request_target)
    if expected_request_updated_at:
        expected_dt = to_utc_datetime(expected_request_updated_at)
        current_dt = to_utc_datetime(snapshot.get("request_updated_at"))
        if expected_dt is None or current_dt is None or expected_dt != current_dt:
            return {
                "target": "funding_requests",
                "expected_updated_at": expected_request_updated_at,
                "current_updated_at": to_iso_timestamp(snapshot.get("request_updated_at")),
            }

    expected = email_target.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("values"), dict):
        expected_values = expected.get("values")
        current_main_sent = _as_bool(snapshot.get("main_sent"))
        expected_main_sent = _as_bool(expected_values.get("main_sent"))
        if expected_main_sent != current_main_sent:
            return {"target": "funding_emails", "expected_main_sent": expected_main_sent, "current_main_sent": current_main_sent}
    return None


def _normalize_principal(principal: dict[str, Any]) -> dict[str, Any]:
    principal_type = str(principal.get("type") or "system")
    if principal_type not in {"student", "admin", "service", "system"}:
        principal_type = "system"
    principal_id = principal.get("id")
    if isinstance(principal_id, str):
        normalized_id: int | str = principal_id.strip() or "unknown"
    elif isinstance(principal_id, int) and principal_id > 0:
        normalized_id = principal_id
    else:
        normalized_id = "unknown"
    return {"type": principal_type, "id": normalized_id}


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


def _failed(
    *,
    start: float,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> OperatorResult:
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
            details=details,
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
