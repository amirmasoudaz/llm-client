from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .funding_request_fields_common import (
    get_platform_db,
    to_iso_timestamp,
    to_utc_datetime,
    validate_and_normalize_fields,
)


class FundingRequestFieldsUpdateApplyOperator(Operator):
    name = "FundingRequest.Fields.Update.Apply"
    version = "1.0.0"

    def __init__(self) -> None:
        self._db = get_platform_db()

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        proposal_wrapper = payload.get("proposal")
        proposal = _extract_proposal_payload(proposal_wrapper)
        if proposal is None:
            return _failed(
                start=start,
                code="invalid_payload",
                message="proposal is required and must be a PlatformPatch.Proposal outcome/payload",
            )

        patch_id = str(proposal.get("patch_id") or "")
        targets = proposal.get("targets")
        if not patch_id or not isinstance(targets, list) or len(targets) != 1:
            return _failed(
                start=start,
                code="invalid_proposal",
                message="proposal must include exactly one target and patch_id",
            )

        target = targets[0]
        if not isinstance(target, dict):
            return _failed(start=start, code="invalid_proposal", message="target must be an object")
        if str(target.get("table") or "") != "funding_requests":
            return _failed(
                start=start,
                code="invalid_proposal",
                message="proposal target table must be funding_requests",
            )

        where = target.get("where")
        if not isinstance(where, dict):
            return _failed(start=start, code="invalid_proposal", message="proposal target where must be an object")
        request_id = where.get("id")
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id <= 0:
            return _failed(start=start, code="invalid_proposal", message="proposal target where.id is required")

        explicit_request_id = payload.get("funding_request_id")
        if isinstance(explicit_request_id, int) and explicit_request_id > 0 and explicit_request_id != request_id:
            return _failed(
                start=start,
                code="invalid_payload",
                message="funding_request_id does not match proposal target",
            )

        raw_set = target.get("set")
        try:
            normalized_set = validate_and_normalize_fields(raw_set)
        except ValueError as exc:
            return _failed(start=start, code="invalid_proposal", message=str(exc))

        current_row = await self._db.fetch_one(
            """
            SELECT id, research_interest, paper_title, journal, `year`, research_connection, updated_at
            FROM funding_requests
            WHERE id=%s
            LIMIT 1;
            """,
            (request_id,),
        )
        if current_row is None:
            return _failed(
                start=start,
                code="funding_request_not_found",
                message="funding request not found",
            )

        expected_updated_at = _extract_expected_updated_at(proposal=proposal, payload=payload)
        current_updated_at = current_row.get("updated_at")
        current_updated_at_iso = to_iso_timestamp(current_updated_at)
        strict_optimistic_lock = bool(payload.get("strict_optimistic_lock", True))
        if strict_optimistic_lock and expected_updated_at:
            expected_dt = to_utc_datetime(expected_updated_at)
            current_dt = to_utc_datetime(current_updated_at)
            if expected_dt is None or current_dt is None or expected_dt != current_dt:
                return _failed(
                    start=start,
                    code="stale_update_conflict",
                    message="funding request has changed; refresh and retry",
                    details={
                        "expected_updated_at": expected_updated_at,
                        "current_updated_at": current_updated_at_iso,
                    },
                )

        effective_updates = {
            field_name: new_value
            for field_name, new_value in normalized_set.items()
            if current_row.get(field_name) != new_value
        }
        rows_affected = 0
        if effective_updates:
            set_clause = ", ".join([f"`{field_name}`=%s" for field_name in effective_updates.keys()])
            sql = f"UPDATE funding_requests SET {set_clause}, updated_at=NOW() WHERE id=%s"
            params: list[Any] = list(effective_updates.values())
            params.append(request_id)
            if strict_optimistic_lock and current_updated_at is not None:
                sql += " AND updated_at=%s"
                params.append(current_updated_at)

            pool = await self._db.pool()
            async with pool.acquire() as conn, conn.cursor() as cur:
                await cur.execute(sql, tuple(params))
                rows_affected = int(cur.rowcount or 0)
            if strict_optimistic_lock and rows_affected == 0:
                return _failed(
                    start=start,
                    code="stale_update_conflict",
                    message="funding request has changed during apply; refresh and retry",
                )

        applied_at = datetime.now(timezone.utc).isoformat()
        receipt_payload = {
            "schema_version": "1.0",
            "patch_id": patch_id,
            "applied": True,
            "applied_at": applied_at,
            "applied_by": _normalize_principal(call.auth_context.principal),
            "results": [
                {
                    "table": "funding_requests",
                    "where": {"id": request_id},
                    "updated_rows": rows_affected,
                }
            ],
        }
        outcome_payload = {
            "schema_version": "1.0",
            "outcome_id": str(uuid.uuid4()),
            "outcome_type": "PlatformPatch.Receipt",
            "created_at": applied_at,
            "hash": _hash_obj(receipt_payload),
            "payload": receipt_payload,
            "producer": {"name": self.name, "version": self.version, "plugin_type": "operator"},
        }
        return OperatorResult(
            status="succeeded",
            result={
                "outcome": outcome_payload,
                "rows_affected": rows_affected,
                "request_id": request_id,
                "applied_at": applied_at,
            },
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _extract_proposal_payload(proposal: Any) -> dict[str, Any] | None:
    if not isinstance(proposal, dict):
        return None
    payload = proposal.get("payload")
    if isinstance(payload, dict) and proposal.get("outcome_type") == "PlatformPatch.Proposal":
        return payload
    targets = proposal.get("targets")
    if isinstance(targets, list):
        return proposal
    return None


def _extract_expected_updated_at(*, proposal: dict[str, Any], payload: dict[str, Any]) -> str | None:
    override = payload.get("expected_updated_at")
    if isinstance(override, str) and override.strip():
        return override.strip()
    target = proposal.get("targets")[0]
    expected = target.get("expected")
    if not isinstance(expected, dict):
        return None
    value = expected.get("updated_at")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _normalize_principal(principal: dict[str, Any]) -> dict[str, Any]:
    principal_type = str(principal.get("type") or "system")
    if principal_type not in {"student", "admin", "service", "system"}:
        principal_type = "system"
    principal_id = principal.get("id")
    if isinstance(principal_id, str):
        principal_id = principal_id.strip() or "unknown"
    elif isinstance(principal_id, int):
        if principal_id <= 0:
            principal_id = "unknown"
    else:
        principal_id = "unknown"
    return {"type": principal_type, "id": principal_id}


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


def _hash_obj(value: Any) -> dict[str, Any]:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"alg": "blake3", "value": blake3(raw).hexdigest()}
