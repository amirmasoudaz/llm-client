from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult


SUPPORTED_DOCUMENT_TYPES = {
    "cv",
    "resume",
    "sop",
    "letter",
    "transcript",
    "portfolio",
    "study_plan",
    "certificate",
    "language_test",
    "identification",
    "photograph",
    "other",
    "cover_letter",
}


class DocumentsReviewOperator(Operator):
    name = "Documents.Review"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        document_id_raw = payload.get("document_id")
        if not document_id_raw:
            return _failed(start, "missing_document_id", "document_id is required")
        try:
            document_id = uuid.UUID(str(document_id_raw))
        except Exception:
            return _failed(start, "invalid_document_id", "document_id must be a UUID")

        review_goal = str(payload.get("review_goal") or "quality").strip() or "quality"
        custom_instructions = str(payload.get("custom_instructions") or "").strip()
        document_processed = payload.get("document_processed")

        doc_row = await self._load_document(document_id=document_id)
        if doc_row is None:
            return _failed(start, "document_not_found", "document not found")

        text = str(doc_row.get("extracted_text") or "").strip()
        extracted_fields = _as_dict(doc_row.get("extracted_fields"))

        payload_document_type = payload.get("document_type")
        row_document_type = doc_row.get("document_type")
        document_type = _normalize_document_type(payload_document_type or row_document_type)
        if document_type not in SUPPORTED_DOCUMENT_TYPES:
            document_type = "other"

        if not text and isinstance(document_processed, dict):
            processed_payload = _as_dict(document_processed.get("payload"))
            candidate_fields = _as_dict(processed_payload.get("extracted_fields"))
            extracted_fields = {**candidate_fields, **extracted_fields}

        normalized_type, score, issues, rubric_scores, notes, structured_report = _review_document(
            document_type=document_type,
            text=text,
            extracted_fields=extracted_fields,
            review_goal=review_goal,
            custom_instructions=custom_instructions,
        )

        outcome_payload = {
            "document_id": str(document_id),
            "document_type": normalized_type,
            "review_goal": review_goal,
            "overall_score": score,
            "issues": issues,
            "rubric_scores": rubric_scores,
            "notes": notes,
            "structured_report": structured_report,
        }
        outcome = _build_outcome(payload=outcome_payload)

        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )

    async def _load_document(self, *, document_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT document_type, extracted_text, extracted_fields
                FROM ledger.documents
                WHERE tenant_id=$1 AND document_id=$2
                LIMIT 1;
                """,
                self._tenant_id,
                document_id,
            )
        if not row:
            return None
        return {
            "document_type": row["document_type"],
            "extracted_text": row["extracted_text"],
            "extracted_fields": _as_dict(row["extracted_fields"]),
        }


def _normalize_document_type(value: Any) -> str:
    normalized = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resume": "cv",
        "cover_letter": "letter",
        "coverletter": "letter",
        "motivation_letter": "letter",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved not in SUPPORTED_DOCUMENT_TYPES:
        return "other"
    if resolved == "resume":
        return "cv"
    if resolved == "cover_letter":
        return "letter"
    return resolved


def _review_document(
    *,
    document_type: str,
    text: str,
    extracted_fields: dict[str, Any],
    review_goal: str,
    custom_instructions: str,
) -> tuple[str, float, list[dict[str, str]], dict[str, float], str, dict[str, Any]]:
    normalized_type = _normalize_document_type(document_type)
    word_count = int(extracted_fields.get("word_count") or len(text.split()))
    has_education = bool(extracted_fields.get("has_education_section"))
    has_experience = bool(extracted_fields.get("has_experience_section"))
    has_skills = bool(extracted_fields.get("has_skills_section"))
    mentions_research = bool(extracted_fields.get("mentions_research"))
    mentions_program_fit = bool(extracted_fields.get("mentions_program_fit"))
    has_email = bool(extracted_fields.get("emails"))

    issues: list[dict[str, str]] = []
    score = 0.86

    if word_count < 120:
        score -= 0.25
        issues.append(
            {
                "severity": "error",
                "message": "Document appears too short for a reliable review.",
                "suggestion": "Provide a fuller draft with concrete details and outcomes.",
            }
        )
    elif word_count < 250:
        score -= 0.12
        issues.append(
            {
                "severity": "warning",
                "message": "Document is short and may miss important context.",
                "suggestion": "Expand key achievements, scope, and measurable impact.",
            }
        )
    elif word_count > 1600:
        score -= 0.10
        issues.append(
            {
                "severity": "warning",
                "message": "Document is long and may reduce readability.",
                "suggestion": "Condense repetitive paragraphs and keep only high-signal details.",
            }
        )

    if normalized_type == "cv":
        if not has_email:
            score -= 0.08
            issues.append(
                {
                    "severity": "warning",
                    "message": "Resume does not clearly expose contact email.",
                    "suggestion": "Add a visible contact block at the top.",
                }
            )
        if not has_education:
            score -= 0.12
            issues.append(
                {
                    "severity": "error",
                    "message": "Education section is missing or unclear.",
                    "suggestion": "Add degree, institution, date, and key academic highlights.",
                }
            )
        if not has_experience:
            score -= 0.12
            issues.append(
                {
                    "severity": "error",
                    "message": "Experience section is missing or unclear.",
                    "suggestion": "Add projects, internships, or research roles with measurable results.",
                }
            )
        if not has_skills:
            score -= 0.06
            issues.append(
                {
                    "severity": "warning",
                    "message": "Skills section is missing or unclear.",
                    "suggestion": "List technical tools, methods, and proficiency level.",
                }
            )
    elif normalized_type in {"sop", "letter", "study_plan"}:
        if not mentions_research:
            score -= 0.14
            issues.append(
                {
                    "severity": "error",
                    "message": "Document does not clearly describe research interests.",
                    "suggestion": "State your research direction and concrete topics you want to pursue.",
                }
            )
        if not mentions_program_fit:
            score -= 0.12
            issues.append(
                {
                    "severity": "warning",
                    "message": "Document does not explain program/professor fit clearly.",
                    "suggestion": "Add a concise paragraph on fit with the target lab or program.",
                }
            )

    if review_goal.lower() in {"clarity", "concise", "brevity"} and word_count > 900:
        score -= 0.06
        issues.append(
            {
                "severity": "warning",
                "message": "Draft is long for a clarity-first objective.",
                "suggestion": "Shorten long paragraphs and move details to appendices if needed.",
            }
        )

    if custom_instructions and "short" in custom_instructions.lower() and word_count > 1100:
        score -= 0.04
        issues.append(
            {
                "severity": "info",
                "message": "Custom instruction asks for brevity but current length is high.",
                "suggestion": "Trim non-essential context and keep only strongest evidence.",
            }
        )

    score = max(0.0, min(1.0, round(score, 3)))

    clarity = _bounded(score + (0.04 if word_count < 900 else -0.03))
    structure = _bounded(score + (0.04 if (has_education or has_experience or mentions_research) else -0.04))
    relevance = _bounded(score + (0.03 if (mentions_program_fit or has_skills) else -0.02))
    brevity = _bounded(score + (0.06 if word_count <= 900 else -0.06))
    rubric_scores = {
        "clarity": clarity,
        "structure": structure,
        "relevance": relevance,
        "brevity": brevity,
    }

    strengths = _derive_strengths(
        document_type=normalized_type,
        has_education=has_education,
        has_experience=has_experience,
        has_skills=has_skills,
        has_email=has_email,
        mentions_research=mentions_research,
        mentions_program_fit=mentions_program_fit,
        clarity=clarity,
        structure=structure,
    )
    improvements = _build_improvements(document_type=normalized_type, issues=issues)

    if issues:
        headline = "Review complete with actionable edits."
    else:
        headline = "Review complete; no major issues detected."
    notes = f"{headline} Type={normalized_type}, words={word_count}, score={score:.3f}."

    structured_report = _build_structured_report(
        document_type=normalized_type,
        headline=headline,
        strengths=strengths,
        improvements=improvements,
        rubric_scores=rubric_scores,
        has_email=has_email,
        has_education=has_education,
        has_experience=has_experience,
        has_skills=has_skills,
        mentions_research=mentions_research,
        mentions_program_fit=mentions_program_fit,
    )

    return normalized_type, score, issues, rubric_scores, notes, structured_report


def _derive_strengths(
    *,
    document_type: str,
    has_education: bool,
    has_experience: bool,
    has_skills: bool,
    has_email: bool,
    mentions_research: bool,
    mentions_program_fit: bool,
    clarity: float,
    structure: float,
) -> list[str]:
    strengths: list[str] = []

    if document_type == "cv":
        if has_email:
            strengths.append("Contact information is present and easy to locate.")
        if has_education:
            strengths.append("Education context is explicitly included.")
        if has_experience:
            strengths.append("Experience evidence is present for reviewer evaluation.")
        if has_skills:
            strengths.append("Skills coverage helps assess technical readiness.")
    elif document_type in {"sop", "letter", "study_plan"}:
        if mentions_research:
            strengths.append("Research direction is explicitly stated.")
        if mentions_program_fit:
            strengths.append("Program/professor fit is addressed.")
        if clarity >= 0.8:
            strengths.append("Narrative clarity is strong for this draft.")
    else:
        if clarity >= 0.8:
            strengths.append("Overall clarity is strong.")
        if structure >= 0.8:
            strengths.append("Document structure is easy to follow.")

    if not strengths:
        strengths.append("Draft contains a usable baseline for iterative improvement.")
    return strengths[:4]


def _build_improvements(*, document_type: str, issues: list[dict[str, str]]) -> list[dict[str, str]]:
    improvements: list[dict[str, str]] = []
    for issue in issues:
        message = str(issue.get("message") or "").strip()
        if not message:
            continue
        improvement: dict[str, str] = {
            "severity": str(issue.get("severity") or "warning"),
            "section": _issue_section(document_type=document_type, issue_message=message),
            "issue": message,
            "recommendation": str(issue.get("suggestion") or "Revise this area with specific, evidence-backed details."),
        }
        improvements.append(improvement)
    return improvements[:8]


def _issue_section(*, document_type: str, issue_message: str) -> str:
    lowered = issue_message.lower()
    if document_type == "cv":
        if "email" in lowered or "contact" in lowered:
            return "contact"
        if "education" in lowered:
            return "education"
        if "experience" in lowered:
            return "experience"
        if "skill" in lowered:
            return "skills"
    if document_type == "sop":
        if "research" in lowered:
            return "research_alignment"
        if "fit" in lowered or "program" in lowered or "professor" in lowered:
            return "program_fit"
        return "narrative"
    if document_type == "letter":
        if "fit" in lowered or "program" in lowered:
            return "program_fit"
        if "research" in lowered:
            return "evidence"
        if "tone" in lowered:
            return "tone"
        return "motivation"
    if "clarity" in lowered:
        return "clarity"
    if "structure" in lowered:
        return "structure"
    if "relevance" in lowered or "fit" in lowered:
        return "relevance"
    if "short" in lowered or "long" in lowered or "brevity" in lowered:
        return "brevity"
    return "content"


def _build_structured_report(
    *,
    document_type: str,
    headline: str,
    strengths: list[str],
    improvements: list[dict[str, str]],
    rubric_scores: dict[str, float],
    has_email: bool,
    has_education: bool,
    has_experience: bool,
    has_skills: bool,
    mentions_research: bool,
    mentions_program_fit: bool,
) -> dict[str, Any]:
    if document_type == "cv":
        section_scores = {
            "contact": _bounded(rubric_scores.get("clarity", 0.5) + (0.07 if has_email else -0.12)),
            "education": _bounded(rubric_scores.get("structure", 0.5) + (0.06 if has_education else -0.12)),
            "experience": _bounded(rubric_scores.get("relevance", 0.5) + (0.08 if has_experience else -0.12)),
            "skills": _bounded(rubric_scores.get("relevance", 0.5) + (0.06 if has_skills else -0.1)),
        }
        return {
            "type": "cv",
            "headline": headline,
            "strengths": strengths,
            "improvements": improvements,
            "section_scores": section_scores,
        }

    if document_type == "sop":
        section_scores = {
            "narrative": _bounded(rubric_scores.get("clarity", 0.5)),
            "research_alignment": _bounded(rubric_scores.get("relevance", 0.5) + (0.08 if mentions_research else -0.12)),
            "program_fit": _bounded(rubric_scores.get("relevance", 0.5) + (0.08 if mentions_program_fit else -0.12)),
            "writing_quality": _bounded(rubric_scores.get("structure", 0.5)),
        }
        return {
            "type": "sop",
            "headline": headline,
            "strengths": strengths,
            "improvements": improvements,
            "section_scores": section_scores,
        }

    if document_type == "letter":
        section_scores = {
            "motivation": _bounded(rubric_scores.get("clarity", 0.5)),
            "program_fit": _bounded(rubric_scores.get("relevance", 0.5) + (0.08 if mentions_program_fit else -0.12)),
            "evidence": _bounded(rubric_scores.get("structure", 0.5) + (0.06 if mentions_research else -0.08)),
            "tone": _bounded((rubric_scores.get("clarity", 0.5) + rubric_scores.get("brevity", 0.5)) / 2.0),
        }
        return {
            "type": "letter",
            "headline": headline,
            "strengths": strengths,
            "improvements": improvements,
            "section_scores": section_scores,
        }

    return {
        "type": "generic",
        "headline": headline,
        "strengths": strengths,
        "improvements": improvements,
        "section_scores": {
            "clarity": _bounded(rubric_scores.get("clarity", 0.5)),
            "structure": _bounded(rubric_scores.get("structure", 0.5)),
            "relevance": _bounded(rubric_scores.get("relevance", 0.5)),
            "brevity": _bounded(rubric_scores.get("brevity", 0.5)),
        },
    }


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, round(value, 3)))


def _build_outcome(*, payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Document.Review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {"name": "Documents.Review", "version": "1.0.0", "plugin_type": "operator"},
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _failed(start: float, code: str, message: str, *, category: str = "validation") -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(code=code, message=message, category=category, retryable=False),
    )
