from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult


class ProfessorSummarizeOperator(Operator):
    name = "Professor.Summarize"
    version = "1.0.0"

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

        profile = payload.get("profile")
        if not isinstance(profile, dict):
            return _failed(
                start=start,
                code="invalid_profile",
                message="profile is required",
            )

        extracted = profile.get("extracted") if isinstance(profile.get("extracted"), dict) else {}
        professor = extracted.get("professor") if isinstance(extracted.get("professor"), dict) else {}
        institute = extracted.get("institute") if isinstance(extracted.get("institute"), dict) else {}

        name = _first_text(professor.get("full_name"), _join_name(professor), "Professor")
        occupation = _first_text(professor.get("occupation"), "faculty member")
        institution_name = _first_text(institute.get("institution_name"), institute.get("department_name"))
        research_areas = _normalize_list(
            _first_collection(extracted.get("research_areas"), extracted.get("area_of_expertise"), extracted.get("categories"))
        )

        highlights = _build_highlights(
            occupation=occupation,
            institution_name=institution_name,
            research_areas=research_areas,
            professor=professor,
            institute=institute,
        )
        summary = _build_summary(
            name=name,
            occupation=occupation,
            institution_name=institution_name,
            research_areas=research_areas,
        )

        custom_instruction = _clean_text(payload.get("custom_instruction"))
        if custom_instruction:
            summary = f"{summary} Focus requested: {custom_instruction}"

        citation_urls = _normalize_urls(
            _first_collection(
                extracted.get("citation_urls"),
                [profile.get("canonical_url")],
            )
        )
        profile_hash = profile.get("profile_hash")
        sources = _normalize_sources(profile.get("sources"))
        publications = _normalize_publications(extracted.get("notable_publications"))

        outcome_payload = _drop_none(
            {
                "professor_id": professor_id,
                "profile_hash": profile_hash if isinstance(profile_hash, dict) else None,
                "sources": sources,
                "canonical_url": _clean_url(profile.get("canonical_url")),
                "summary": summary,
                "highlights": highlights,
                "research_areas": research_areas,
                "notable_publications": publications,
                "citation_urls": citation_urls,
                "custom_instruction": custom_instruction,
            }
        )
        outcome = _build_outcome(outcome_payload, producer_name=self.name, producer_version=self.version)
        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _build_summary(*, name: str, occupation: str, institution_name: str | None, research_areas: list[str]) -> str:
    parts = [f"{name} is a {occupation}"]
    if institution_name:
        parts[-1] += f" at {institution_name}"
    if research_areas:
        parts.append("Research areas include " + ", ".join(research_areas[:5]) + ".")
    else:
        parts.append("Research topics are not explicitly listed in the retrieved profile.")
    return " ".join(parts).strip()


def _build_highlights(
    *,
    occupation: str,
    institution_name: str | None,
    research_areas: list[str],
    professor: dict[str, Any],
    institute: dict[str, Any],
) -> list[str]:
    highlights: list[str] = []
    if occupation:
        highlights.append(f"Role: {occupation}")
    department = _first_text(professor.get("department"), institute.get("department_name"))
    if department:
        highlights.append(f"Department: {department}")
    if institution_name:
        highlights.append(f"Institution: {institution_name}")
    country = _clean_text(institute.get("country"))
    if country:
        highlights.append(f"Country: {country}")
    if research_areas:
        highlights.append("Top areas: " + ", ".join(research_areas[:3]))
    return highlights[:5]


def _normalize_publications(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    publications: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title"))
        if not title:
            continue
        entry: dict[str, Any] = {"title": title}
        year = item.get("year")
        if isinstance(year, int) and 1900 <= year <= 2100:
            entry["year"] = year
        venue = _clean_text(item.get("venue"))
        if venue:
            entry["venue"] = venue
        url = _clean_url(item.get("url"))
        if url:
            entry["url"] = url
        publications.append(entry)
        if len(publications) >= 5:
            break
    return publications


def _normalize_sources(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed = {"platform", "canspider", "web", "manual"}
    out: list[str] = []
    for item in value:
        text = _clean_text(item)
        if not text:
            continue
        if text not in allowed:
            continue
        if text not in out:
            out.append(text)
    return out


def _normalize_urls(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        url = _clean_url(value)
        if not url:
            continue
        if url not in out:
            out.append(url)
    return out


def _join_name(value: dict[str, Any]) -> str | None:
    first = _clean_text(value.get("first_name"))
    last = _clean_text(value.get("last_name"))
    if first and last:
        return f"{first} {last}"
    return first or last


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return None


def _first_collection(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list) and value:
            return value
    return []


def _normalize_list(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out


def _clean_url(value: Any) -> str | None:
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


def _build_outcome(payload: dict[str, Any], *, producer_name: str, producer_version: str) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Professor.Summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {"name": producer_name, "version": producer_version, "plugin_type": "operator"},
    }


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: val for key, val in value.items() if val is not None}


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
