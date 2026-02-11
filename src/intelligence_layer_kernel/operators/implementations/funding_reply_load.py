from __future__ import annotations

import json
import re
import time
from typing import Any

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .documents_common import resolve_thread_scope
from .funding_request_fields_common import get_platform_db


class FundingReplyLoadOperator(Operator):
    name = "FundingReply.Load"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id
        self._db = get_platform_db()

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        thread_id = _coerce_positive_int(payload.get("thread_id"))
        funding_request_id = _coerce_positive_int(payload.get("funding_request_id"))
        reply_id = _coerce_positive_int(payload.get("reply_id"))

        if thread_id is None and funding_request_id is None:
            return _failed(start, "missing_scope", "thread_id or funding_request_id is required")

        if funding_request_id is None:
            try:
                scope = await resolve_thread_scope(pool=self._pool, tenant_id=self._tenant_id, thread_id=thread_id or 0)
            except Exception as exc:
                return _failed(
                    start,
                    "thread_lookup_failed",
                    str(exc),
                    category="dependency",
                    retryable=True,
                )
            funding_request_id = scope.funding_request_id

        max_items = _coerce_positive_int(payload.get("max_items")) or 5
        max_items = max(1, min(max_items, 20))

        try:
            rows = await self._fetch_replies(
                funding_request_id=funding_request_id,
                reply_id=reply_id,
                max_items=max_items,
            )
        except Exception as exc:
            return _failed(
                start,
                "platform_reply_query_failed",
                str(exc),
                category="dependency",
                retryable=True,
            )

        replies = [_shape_reply_row(row) for row in rows if isinstance(row, dict)]
        latest_reply = replies[0] if replies else None

        return OperatorResult(
            status="succeeded",
            result={
                "funding_request_id": funding_request_id,
                "reply_count": len(replies),
                "replies": replies,
                "latest_reply": latest_reply,
            },
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )

    async def _fetch_replies(
        self,
        *,
        funding_request_id: int,
        reply_id: int | None,
        max_items: int,
    ) -> list[dict[str, Any]]:
        # Some environments have legacy funding_replies schemas; select only available columns.
        columns = await self._db.fetch_all("SHOW COLUMNS FROM funding_replies")
        available = {str(item.get("Field") or "").strip() for item in columns if isinstance(item, dict)}
        if "funding_request_id" not in available:
            raise ValueError("funding_replies table is missing funding_request_id")

        select_candidates = [
            "id",
            "funding_request_id",
            "reply_body_raw",
            "reply_body_cleaned",
            "is_auto_generated",
            "auto_generated_type",
            "needs_human_review",
            "engagement_label",
            "engagement_bool",
            "activity_status",
            "activity_bool",
            "next_step_type",
            "short_rationale",
            "key_phrases",
            "confidence",
            "created_at",
            "updated_at",
        ]
        select_cols = [name for name in select_candidates if name in available]
        if not select_cols:
            raise ValueError("funding_replies table has no readable columns")

        where_sql = "WHERE funding_request_id=%s"
        params: list[Any] = [funding_request_id]
        if reply_id is not None and "id" in available:
            where_sql += " AND id=%s"
            params.append(reply_id)

        order_by = "id DESC" if "id" in available else "updated_at DESC" if "updated_at" in available else "funding_request_id DESC"
        sql = (
            "SELECT "
            + ", ".join(f"`{name}`" for name in select_cols)
            + f" FROM funding_replies {where_sql} ORDER BY {order_by} LIMIT %s;"
        )
        params.append(max_items)
        rows = await self._db.fetch_all(sql, tuple(params))
        return list(rows or [])


def _shape_reply_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = _as_non_empty_str(row.get("reply_body_raw"))
    cleaned = _as_non_empty_str(row.get("reply_body_cleaned"))
    text = cleaned or raw or ""

    key_phrases = row.get("key_phrases")
    if isinstance(key_phrases, str):
        try:
            parsed = json.loads(key_phrases)
            if isinstance(parsed, list):
                key_phrases = parsed
        except Exception:
            key_phrases = [part.strip() for part in key_phrases.split(",") if part.strip()]
    if not isinstance(key_phrases, list):
        key_phrases = []

    return {
        "id": _coerce_positive_int(row.get("id")),
        "funding_request_id": _coerce_positive_int(row.get("funding_request_id")),
        "reply_body_raw": raw,
        "reply_body_cleaned": cleaned or raw,
        "is_auto_generated": _coerce_bool(row.get("is_auto_generated")),
        "auto_generated_type": _normalize_auto_generated_type(row.get("auto_generated_type")),
        "needs_human_review": _coerce_bool(row.get("needs_human_review")),
        "engagement_label": _as_non_empty_str(row.get("engagement_label")),
        "engagement_bool": _coerce_bool_nullable(row.get("engagement_bool")),
        "activity_status": _as_non_empty_str(row.get("activity_status")),
        "activity_bool": _coerce_bool_nullable(row.get("activity_bool")),
        "next_step_type": _as_non_empty_str(row.get("next_step_type")),
        "short_rationale": _as_non_empty_str(row.get("short_rationale")),
        "key_phrases": [str(item).strip() for item in key_phrases if str(item).strip()],
        "confidence": _coerce_confidence(row.get("confidence")),
        "created_at": _coerce_str(row.get("created_at")),
        "updated_at": _coerce_str(row.get("updated_at")),
        "security_flags": _detect_untrusted_flags(text),
    }


def _detect_untrusted_flags(text: str) -> list[str]:
    lowered = text.lower()
    flags: list[str] = []
    if len(text) > 20000:
        flags.append("reply_too_long")

    injection_patterns = (
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"system\s+prompt",
        r"developer\s+message",
        r"do not follow",
        r"execute\s+this",
    )
    if any(re.search(pattern, lowered) for pattern in injection_patterns):
        flags.append("prompt_injection_pattern")

    link_count = len(re.findall(r"https?://", lowered))
    if link_count >= 5:
        flags.append("high_link_density")

    if "mail delivery" in lowered or "undeliver" in lowered or "out of office" in lowered:
        flags.append("auto_reply_signal")

    return flags


def _normalize_auto_generated_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"OUT_OF_OFFICE", "DELIVERY_FAILURE", "OTHER_AUTO", "NONE"}:
        return text
    return "NONE"


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except Exception:
        return False


def _coerce_bool_nullable(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _coerce_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.5
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


def _as_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _failed(
    start: float,
    code: str,
    message: str,
    *,
    category: str = "validation",
    retryable: bool = False,
) -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(
            code=code,
            message=message,
            category=category,
            retryable=retryable,
        ),
    )
