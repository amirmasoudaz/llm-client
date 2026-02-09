from __future__ import annotations

import json
import time
from typing import Any

from blake3 import blake3

from intelligence_layer_ops.platform_tools import platform_load_funding_thread_context

from ..base import Operator
from ..types import OperatorCall, OperatorResult, OperatorMetrics, OperatorError


class PlatformContextLoadOperator(Operator):
    name = "Platform.Context.Load"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        thread_id = payload.get("thread_id")
        if thread_id is None:
            error = OperatorError(
                code="missing_thread_id",
                message="thread_id is required",
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

        try:
            funding_request_id = await self._resolve_funding_request_id(int(thread_id))
        except Exception as exc:
            error = OperatorError(
                code="thread_lookup_failed",
                message=str(exc),
                category="dependency",
                retryable=False,
            )
            return OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=error,
            )

        try:
            result = await platform_load_funding_thread_context.execute(funding_request_id=funding_request_id)
        except Exception as exc:
            error = OperatorError(
                code="platform_context_failed",
                message=str(exc),
                category="dependency",
                retryable=False,
            )
            return OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=error,
            )

        if not result.success or not isinstance(result.content, dict):
            error = OperatorError(
                code="platform_context_failed",
                message=result.error or "platform context load failed",
                category="dependency",
                retryable=False,
            )
            return OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=error,
            )

        row = result.content
        platform = _shape_platform(row)
        intelligence = {
            "requirements": {},
            "preferences": {},
            "background": {},
            "composer_prereqs": {},
        }

        platform_hash = _hash_obj(platform)
        background_hash = _hash_obj(intelligence.get("background") or {})
        prereqs_hash = _hash_obj(intelligence.get("composer_prereqs") or {})

        output = {
            "platform": platform,
            "intelligence": intelligence,
            "platform_context_hash": platform_hash,
            "student_background_hash": background_hash,
            "composer_prereqs_hash": prereqs_hash,
        }

        return OperatorResult(
            status="succeeded",
            result=output,
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )

    async def _resolve_funding_request_id(self, thread_id: int) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT funding_request_id
                FROM runtime.threads
                WHERE tenant_id=$1 AND thread_id=$2;
                """,
                self._tenant_id,
                thread_id,
            )
            if not row:
                raise ValueError("thread not found")
            return int(row["funding_request_id"])


def _shape_platform(row: dict[str, Any]) -> dict[str, Any]:
    student = _drop_none(
        {
            "id": row.get("user_id"),
            "first_name": row.get("user_first_name"),
            "last_name": row.get("user_last_name"),
            "email": row.get("user_email_address"),
            "mobile_number": row.get("user_phone_number"),
            "date_of_birth": _coerce_str(row.get("user_date_of_birth")),
            "gender": row.get("user_gender"),
            "country_of_citizenship": row.get("user_country_of_citizenship"),
        },
    )
    funding_request = _drop_none(
        {
            "id": row.get("request_id"),
            "created_at": _coerce_str(row.get("request_creation_datetime")),
            "match_status": row.get("request_match_status"),
            "research_interest": row.get("request_research_interest"),
            "paper_title": row.get("request_paper_title"),
            "journal": row.get("request_journal"),
            "year": row.get("request_year"),
            "research_connection": row.get("request_research_connection"),
            "attachments": _maybe_json(row.get("request_attachments")),
            "student_template_ids": _maybe_json(row.get("request_student_template_ids")),
            "status": row.get("request_status"),
            "email_subject": row.get("request_email_subject"),
            "email_content": row.get("request_email_body"),
        },
    )
    professor = _drop_none(
        {
            "id": row.get("request_professor_id"),
            "first_name": row.get("professor_first_name"),
            "last_name": row.get("professor_last_name"),
            "full_name": row.get("professor_full_name"),
            "occupation": row.get("professor_occupation"),
            "research_areas": _coerce_str(row.get("professor_research_areas")),
            "credentials": row.get("professor_credentials"),
            "area_of_expertise": _coerce_str(row.get("professor_area_of_expertise")),
            "categories": _coerce_str(row.get("professor_categories")),
            "department": row.get("professor_department"),
            "email_address": row.get("professor_email_address"),
            "url": row.get("professor_url"),
            "others": _maybe_json(row.get("professor_others")),
            "canspider_digest_id": row.get("professor_digest_id") or row.get("professor_canspider_digest_id"),
        },
    )
    institute = _drop_none(
        {
            "institution_name": row.get("institute_name"),
            "department_name": row.get("department_name"),
            "country": row.get("institute_country"),
        }
    )
    metas = _drop_none({"funding_template_initial_data": _maybe_json(row.get("user_onboarding_data"))})
    return {
        "student": student,
        "funding_request": funding_request,
        "email": _shape_email(row),
        "reply": _shape_reply(row),
        "professor": professor,
        "institute": institute if institute else None,
        "metas": metas if metas else None,
    }


def _shape_email(row: dict[str, Any]) -> dict[str, Any] | None:
    if not row.get("email_id"):
        return None
    return _drop_none(
        {
            "id": row.get("email_id"),
            "main_sent": _coerce_bool(row.get("main_email_sent_status")),
            "main_sent_at": _coerce_str(row.get("main_email_sent_datetime")),
            "main_email_subject": row.get("main_email_subject"),
            "main_email_body": row.get("main_email_body"),
            "reminder_one_sent": _coerce_bool(row.get("reminder_one_sent_status")),
            "reminder_one_sent_at": _coerce_str(row.get("reminder_one_sent_datetime")),
            "reminder_one_body": row.get("reminder_one_body"),
            "reminder_two_sent": _coerce_bool(row.get("reminder_two_sent_status")),
            "reminder_two_sent_at": _coerce_str(row.get("reminder_two_sent_datetime")),
            "reminder_two_body": row.get("reminder_two_body"),
            "reminder_three_sent": _coerce_bool(row.get("reminder_three_sent_status")),
            "reminder_three_sent_at": _coerce_str(row.get("reminder_three_sent_datetime")),
            "reminder_three_body": row.get("reminder_three_body"),
            "professor_replied": _coerce_bool(row.get("professor_reply_status")),
            "professor_replied_at": _coerce_str(row.get("professor_reply_datetime")),
        }
    )


def _shape_reply(row: dict[str, Any]) -> dict[str, Any] | None:
    if not row.get("professor_reply_body") and not row.get("professor_reply_activity_status"):
        return None
    return _drop_none(
        {
            "reply_body_cleaned": row.get("professor_reply_body"),
            "engagement_label": row.get("professor_reply_engagement_label"),
            "activity_status": row.get("professor_reply_activity_status"),
            "short_rationale": row.get("professor_reply_short_rationale"),
            "key_phrases": row.get("professor_reply_key_phrases"),
            "auto_generated_type": row.get("professor_reply_auto_generated_type"),
        }
    )


def _hash_obj(value: Any) -> dict[str, Any]:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"alg": "blake3", "value": blake3(raw).hexdigest()}


def _maybe_json(value: Any) -> Any:
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


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except Exception:
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if v is not None}
