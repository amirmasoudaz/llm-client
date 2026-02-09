from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .funding_request_fields_common import get_platform_db


class ProfessorProfileRetrieveOperator(Operator):
    name = "Professor.Profile.Retrieve"
    version = "1.0.0"

    def __init__(self) -> None:
        self._db = get_platform_db()

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        professor_id = payload.get("professor_id")
        if isinstance(professor_id, bool) or not isinstance(professor_id, int) or professor_id <= 0:
            return _failed(
                start=start,
                code="invalid_professor_id",
                message="professor_id must be a positive integer",
            )

        manual_url = _normalize_url(payload.get("professor_url"))
        row = await self._db.fetch_one(
            """
            SELECT fp.id,
                   fp.first_name,
                   fp.last_name,
                   fp.full_name,
                   fp.occupation,
                   fp.research_areas,
                   fp.credentials,
                   fp.area_of_expertise,
                   fp.categories,
                   fp.department,
                   fp.email_address,
                   fp.url,
                   fp.others,
                   fp.canspider_digest_id,
                   fi.institution_name,
                   fi.department_name,
                   fi.country
            FROM funding_professors fp
            LEFT JOIN funding_institutes fi ON fi.id = fp.funding_institute_id
            WHERE fp.id = %s
            LIMIT 1;
            """,
            (professor_id,),
        )
        if not row:
            return _failed(
                start=start,
                code="professor_not_found",
                message="professor profile not found",
            )

        platform_url = _normalize_url(row.get("url"))
        canonical_url = manual_url or platform_url
        source_tags = ["platform"]
        if manual_url and manual_url != platform_url:
            source_tags.append("manual")
        if row.get("canspider_digest_id"):
            source_tags.append("canspider")

        research_areas = _merge_topics(
            row.get("research_areas"),
            row.get("area_of_expertise"),
            row.get("categories"),
        )
        others = _parse_json_object(row.get("others"))
        extracted = {
            "professor": _drop_none(
                {
                    "id": professor_id,
                    "first_name": _clean_text(row.get("first_name")),
                    "last_name": _clean_text(row.get("last_name")),
                    "full_name": _clean_text(row.get("full_name")),
                    "occupation": _clean_text(row.get("occupation")),
                    "department": _clean_text(row.get("department")),
                    "email_address": _clean_text(row.get("email_address")),
                    "credentials": _clean_text(row.get("credentials")),
                    "canspider_digest_id": _clean_text(row.get("canspider_digest_id")),
                }
            ),
            "institute": _drop_none(
                {
                    "institution_name": _clean_text(row.get("institution_name")),
                    "department_name": _clean_text(row.get("department_name")),
                    "country": _clean_text(row.get("country")),
                }
            ),
            "research_areas": research_areas,
            "area_of_expertise": _parse_string_collection(row.get("area_of_expertise")),
            "categories": _parse_string_collection(row.get("categories")),
            "citation_urls": [canonical_url] if canonical_url else [],
            "raw_others": others,
        }

        profile_hash = _hash_obj(extracted)
        profile = _drop_none(
            {
                "professor_id": professor_id,
                "profile_hash": profile_hash,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "canonical_url": canonical_url,
                "sources": source_tags,
                "extracted": extracted,
            }
        )
        return OperatorResult(
            status="succeeded",
            result={"profile": profile},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


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


def _hash_obj(value: Any) -> dict[str, str]:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"alg": "blake3", "value": blake3(raw).hexdigest()}


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            return {}
    return {}


def _parse_string_collection(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return _dedupe_strings(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return _dedupe_strings(decoded)
            if isinstance(decoded, str):
                return _split_topic_text(decoded)
        except Exception:
            pass
        return _split_topic_text(text)
    return []


def _split_topic_text(text: str) -> list[str]:
    parts = re.split(r"[|,;\n/]+", text)
    return _dedupe_strings(parts)


def _merge_topics(*values: Any) -> list[str]:
    merged: list[str] = []
    for value in values:
        for topic in _parse_string_collection(value):
            if topic not in merged:
                merged.append(topic)
    return merged


def _dedupe_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out


def _normalize_url(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    if text.startswith("http://") or text.startswith("https://"):
        return text
    return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: val for key, val in value.items() if val is not None}
