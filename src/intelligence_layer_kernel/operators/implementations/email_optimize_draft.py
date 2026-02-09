from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ...prompts import PromptTemplateLoader
from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult


class EmailOptimizeDraftOperator(Operator):
    name = "Email.OptimizeDraft"
    version = "1.0.0"

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        draft_subject = _first_text(
            payload.get("subject_override"),
            payload.get("current_subject"),
            payload.get("fallback_subject"),
        )
        draft_body = _first_text(payload.get("current_body"), payload.get("fallback_body"))
        requested_edits = _normalize_requested_edits(payload.get("requested_edits"))
        custom_instructions = _first_text(payload.get("custom_instructions"))

        if draft_body is None:
            return _failed(
                start=start,
                code="missing_email_body",
                message="email body is required to optimize a draft",
            )

        professor = payload.get("professor") if isinstance(payload.get("professor"), dict) else {}
        funding_request = (
            payload.get("funding_request") if isinstance(payload.get("funding_request"), dict) else {}
        )
        student_profile = (
            payload.get("student_profile") if isinstance(payload.get("student_profile"), dict) else {}
        )
        review_report = payload.get("review_report") if isinstance(payload.get("review_report"), dict) else None
        source_version_id = _maybe_text(payload.get("source_draft_outcome_id"))
        source_version_number = _maybe_int(payload.get("source_draft_version"))

        _render_optimize_prompt(
            payload=payload,
            subject=draft_subject or "",
            body=draft_body,
            requested_edits=requested_edits,
            review_report=review_report,
        )

        optimized_subject, optimized_body, rationale, diff_summary = _optimize_draft(
            subject=draft_subject or "Prospective Student Inquiry",
            body=draft_body,
            professor=professor,
            funding_request=funding_request,
            student_profile=student_profile,
            requested_edits=requested_edits,
            custom_instructions=custom_instructions,
        )
        version_id = str(uuid.uuid4())
        version_number = source_version_number + 1 if source_version_number is not None else 1

        draft_payload: dict[str, Any] = {
            "subject": optimized_subject,
            "body": optimized_body,
            "rationale": rationale,
            "diff_summary": diff_summary,
            "version_id": version_id,
            "version_number": version_number,
        }
        funding_request_id = _maybe_int(funding_request.get("id"))
        if funding_request_id is not None:
            draft_payload["funding_request_id"] = funding_request_id
        email_id = _maybe_int(payload.get("email_id"))
        if email_id is not None:
            draft_payload["email_id"] = email_id
        if custom_instructions is not None:
            draft_payload["custom_instructions"] = custom_instructions
        if source_version_id is not None:
            draft_payload["source_version_id"] = source_version_id
        draft_payload["notes"] = (
            f"Optimization edits: {', '.join(requested_edits) if requested_edits else 'none'}; "
            f"version={version_number}"
        )

        outcome = _build_outcome(draft_payload, producer_name=self.name, producer_version=self.version)
        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


_PROMPT_LOADER: PromptTemplateLoader | None = None


def _render_optimize_prompt(
    *,
    payload: dict[str, Any],
    subject: str,
    body: str,
    requested_edits: list[str],
    review_report: dict[str, Any] | None,
) -> None:
    global _PROMPT_LOADER
    if _PROMPT_LOADER is None:
        _PROMPT_LOADER = PromptTemplateLoader()
    _PROMPT_LOADER.render(
        "email_optimize_draft.v1",
        {
            "email": {"subject": subject, "body": body},
            "requested_edits": requested_edits,
            "review_report": review_report or {},
            "professor": payload.get("professor") or {},
            "funding_request": payload.get("funding_request") or {},
            "student_profile": payload.get("student_profile") or {},
            "custom_instructions": payload.get("custom_instructions"),
        },
    )


def _optimize_draft(
    *,
    subject: str,
    body: str,
    professor: dict[str, Any],
    funding_request: dict[str, Any],
    student_profile: dict[str, Any],
    requested_edits: list[str],
    custom_instructions: str | None,
) -> tuple[str, str, str, dict[str, Any]]:
    normalized_body = _normalize_whitespace(body)
    normalized_subject = _normalize_whitespace(subject)
    applied_edits = list(requested_edits)

    if custom_instructions:
        lowered = custom_instructions.lower()
        if "short" in lowered and "shorten" not in applied_edits:
            applied_edits.append("shorten")
        if "bullet" in lowered and "add_bullets" not in applied_edits:
            applied_edits.append("add_bullets")
        if "clarity" in lowered and "improve_clarity" not in applied_edits:
            applied_edits.append("improve_clarity")
        if "human" in lowered and "humanize" not in applied_edits:
            applied_edits.append("humanize")
        if "subject" in lowered and "change_subject" not in applied_edits:
            applied_edits.append("change_subject")

    if "change_subject" in applied_edits:
        normalized_subject = _compose_subject(professor=professor, fallback_subject=normalized_subject)
    normalized_subject = _shorten_subject(normalized_subject)

    optimized_body = normalized_body
    if "improve_clarity" in applied_edits:
        optimized_body = _improve_clarity(optimized_body)
    if "paraphrase" in applied_edits:
        optimized_body = _light_paraphrase(optimized_body)
    if "humanize" in applied_edits:
        optimized_body = _humanize_body(optimized_body)
    if "add_custom_hook" in applied_edits:
        optimized_body = _add_custom_hook(
            optimized_body,
            professor=professor,
            funding_request=funding_request,
            student_profile=student_profile,
        )
    if "add_bullets" in applied_edits:
        optimized_body = _append_key_points(optimized_body)
    if "shorten" in applied_edits:
        optimized_body = _shorten_body(optimized_body)

    rationale_parts = []
    if applied_edits:
        rationale_parts.append(f"Applied edits: {', '.join(applied_edits)}.")
    else:
        rationale_parts.append("No explicit edit flags provided; applied light normalization only.")
    rationale_parts.append("Preserved core intent and professor-specific context where possible.")
    rationale = " ".join(rationale_parts)

    diff_summary = {
        "subject_changed": normalized_subject != _normalize_whitespace(subject),
        "body_changed": optimized_body != _normalize_whitespace(body),
        "before_chars": len(_normalize_whitespace(body)),
        "after_chars": len(optimized_body),
        "delta_chars": len(optimized_body) - len(_normalize_whitespace(body)),
        "applied_edits": applied_edits,
    }
    return normalized_subject, optimized_body, rationale, diff_summary


def _compose_subject(*, professor: dict[str, Any], fallback_subject: str) -> str:
    full_name = _maybe_text(professor.get("full_name"))
    if full_name:
        return f"Prospective Student Inquiry - {full_name}"
    return fallback_subject or "Prospective Student Inquiry"


def _shorten_subject(subject: str) -> str:
    value = subject.strip()
    if len(value) <= 78:
        return value
    return value[:75].rstrip() + "..."


def _improve_clarity(body: str) -> str:
    text = _normalize_whitespace(body)
    text = text.replace("I am writing to inquire whether", "I am writing to ask whether")
    text = text.replace("I look forward to hearing from you.", "I would appreciate your response.")
    return text


def _light_paraphrase(body: str) -> str:
    replacements = {
        "I am writing to": "I am reaching out to",
        "I would like to": "I want to",
        "Thank you for your time and attention.": "Thank you for considering my request.",
    }
    text = body
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return _normalize_whitespace(text)


def _humanize_body(body: str) -> str:
    text = body.strip()
    if not text.lower().startswith("dear "):
        text = "Dear Professor,\n\n" + text
    if "Best regards" not in text and "Kind regards" not in text:
        text = text.rstrip() + "\n\nBest regards,"
    return text


def _add_custom_hook(
    body: str,
    *,
    professor: dict[str, Any],
    funding_request: dict[str, Any],
    student_profile: dict[str, Any],
) -> str:
    full_name = _maybe_text(professor.get("full_name")) or "Professor"
    interest = _first_text(
        funding_request.get("research_interest"),
        student_profile.get("research_interest"),
        student_profile.get("preferred_research_interest"),
    )
    if interest is None:
        interests = student_profile.get("interests")
        if isinstance(interests, list):
            joined = ", ".join(str(item).strip() for item in interests if str(item).strip())
            if joined:
                interest = joined
    if interest is None:
        interest = "your research area"
    hook = (
        f"I believe my background in {interest} can complement your ongoing work, "
        f"and I would value the chance to contribute to your group, {full_name}."
    )
    if hook in body:
        return body
    parts = body.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0] + "\n\n" + hook + "\n\n" + parts[1]
    return body.rstrip() + "\n\n" + hook


def _append_key_points(body: str) -> str:
    if "\n- " in body or "\n* " in body:
        return body
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    if len(sentences) < 2:
        return body
    points = sentences[:3]
    bullets = "\n".join(f"- {point}" for point in points)
    return body.rstrip() + "\n\nKey points:\n" + bullets


def _shorten_body(body: str, max_chars: int = 1600) -> str:
    text = body.strip()
    if len(text) <= max_chars:
        return text
    shortened = text[: max_chars - 3].rstrip()
    boundary = max(shortened.rfind(". "), shortened.rfind("\n"))
    if boundary > 400:
        shortened = shortened[: boundary + 1].rstrip()
    return shortened + "..."


def _normalize_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    collapsed = "\n".join(lines)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    collapsed = re.sub(r"[ \t]{2,}", " ", collapsed)
    return collapsed.strip()


def _normalize_requested_edits(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text:
            continue
        if text not in normalized:
            normalized.append(text)
    return normalized


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _maybe_text(value)
        if text is not None:
            return text
    return None


def _maybe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


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


def _build_outcome(payload: dict[str, Any], *, producer_name: str, producer_version: str) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Email.Draft",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {"name": producer_name, "version": producer_version, "plugin_type": "operator"},
    }
