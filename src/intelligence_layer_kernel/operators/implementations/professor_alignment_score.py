from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult


class ProfessorAlignmentScoreOperator(Operator):
    name = "Professor.Alignment.Score"
    version = "1.0.0"

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        student_id = payload.get("student_id")
        professor_id = payload.get("professor_id")
        if isinstance(student_id, bool) or not isinstance(student_id, int) or student_id <= 0:
            return _failed(
                start=start,
                code="invalid_student_id",
                message="student_id must be a positive integer",
            )
        if isinstance(professor_id, bool) or not isinstance(professor_id, int) or professor_id <= 0:
            return _failed(
                start=start,
                code="invalid_professor_id",
                message="professor_id must be a positive integer",
            )

        professor_summary = payload.get("professor_summary")
        if not isinstance(professor_summary, dict):
            return _failed(
                start=start,
                code="invalid_professor_summary",
                message="professor_summary is required",
            )
        summary_payload = professor_summary.get("payload")
        if not isinstance(summary_payload, dict):
            return _failed(
                start=start,
                code="invalid_professor_summary_payload",
                message="professor_summary.payload is required",
            )

        student_background = payload.get("student_background")
        if not isinstance(student_background, dict):
            student_background = {}

        focus_areas = _normalize_list(payload.get("focus_areas"))
        professor_terms, professor_topics = _extract_professor_signal(summary_payload)
        student_terms, student_topics = _extract_student_signal(student_background)

        matched_topics = _sorted_list(professor_topics.intersection(student_topics))
        if not matched_topics:
            matched_topics = _sorted_list(_token_overlap_topics(professor_terms, student_terms))

        focus_set = {item.lower() for item in focus_areas}
        focus_hits = _sorted_list(focus_set.intersection(student_topics))
        focus_misses = _sorted_list(focus_set.difference(student_topics))

        overlap_tokens = professor_terms.intersection(student_terms)
        professor_term_count = max(1, len(professor_terms))
        overlap_ratio = len(overlap_tokens) / professor_term_count
        focus_ratio = 0.0
        if focus_set:
            focus_ratio = len(focus_hits) / len(focus_set)
        elif professor_topics:
            topic_overlap = len(professor_topics.intersection(student_topics))
            focus_ratio = topic_overlap / max(1, len(professor_topics))

        evidence_count = min(12, len(student_terms))
        evidence_ratio = min(1.0, evidence_count / 12.0)
        score = _clamp01((0.15 + 0.6 * overlap_ratio) + (0.2 * focus_ratio) + (0.05 * evidence_ratio))

        label = "low"
        if score >= 0.7:
            label = "high"
        elif score >= 0.45:
            label = "medium"

        gaps = _build_gaps(
            professor_topics=professor_topics,
            student_topics=student_topics,
            focus_misses=focus_misses,
            overlap_ratio=overlap_ratio,
        )
        recommendations = _build_recommendations(
            label=label,
            focus_misses=focus_misses,
            matched_topics=matched_topics,
            gaps=gaps,
        )
        rationale_short = _build_rationale_short(
            label=label,
            overlap_ratio=overlap_ratio,
            matched_topics=matched_topics,
            gaps=gaps,
        )
        rationale_long = _build_rationale_long(
            professor_topics=professor_topics,
            student_topics=student_topics,
            matched_topics=matched_topics,
            focus_areas=focus_areas,
            focus_hits=focus_hits,
            focus_misses=focus_misses,
            overlap_tokens=overlap_tokens,
            score=score,
            label=label,
        )

        outcome_payload = {
            "student_id": student_id,
            "professor_id": professor_id,
            "score": round(score, 3),
            "label": label,
            "focus_areas": focus_areas,
            "matched_topics": matched_topics,
            "gaps": gaps,
            "rationale_short": rationale_short,
            "rationale_long": rationale_long,
            "recommendations": recommendations,
        }
        outcome = _build_outcome(outcome_payload, producer_name=self.name, producer_version=self.version)
        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "this",
    "your",
    "their",
    "into",
    "about",
    "than",
}

_CANONICAL_TOPICS: dict[str, tuple[str, ...]] = {
    "signal processing": (
        "signal",
        "signals",
        "audio",
        "speech",
        "acoustic",
        "dsp",
        "cochlear",
        "biomedical",
        "biocircuits",
    ),
    "number theory": ("number", "numbertheory", "l-functions", "lfunctions", "elliptic", "curves"),
    "machine learning": ("machine", "learning", "ml", "deep", "neural", "model"),
    "optimization": ("optimization", "optimisation", "convex", "gradient"),
    "control systems": ("control", "systems", "feedback", "stability"),
    "statistics": ("statistics", "statistical", "probability", "inference"),
}


def _extract_professor_signal(summary_payload: dict[str, Any]) -> tuple[set[str], set[str]]:
    terms = set()
    topics = set()

    for topic in _normalize_list(summary_payload.get("research_areas")):
        topics.add(topic.lower())
        terms.update(_tokenize(topic))
    for text in _normalize_list(summary_payload.get("highlights")):
        terms.update(_tokenize(text))
    terms.update(_tokenize(_as_text(summary_payload.get("summary"))))

    publications = summary_payload.get("notable_publications")
    if isinstance(publications, list):
        for item in publications:
            if isinstance(item, dict):
                terms.update(_tokenize(_as_text(item.get("title"))))
                terms.update(_tokenize(_as_text(item.get("venue"))))

    topics.update(_topics_from_terms(terms))
    return terms, topics


def _extract_student_signal(student_background: dict[str, Any]) -> tuple[set[str], set[str]]:
    terms = set()
    topics = set()

    for text in _collect_text_values(student_background):
        terms.update(_tokenize(text))

    if isinstance(student_background.get("funding_request"), dict):
        request = student_background["funding_request"]
        terms.update(_tokenize(_as_text(request.get("research_interest"))))
        terms.update(_tokenize(_as_text(request.get("paper_title"))))
        terms.update(_tokenize(_as_text(request.get("journal"))))
        terms.update(_tokenize(_as_text(request.get("research_connection"))))

    if isinstance(student_background.get("profile"), dict):
        profile = student_background["profile"]
        general = profile.get("general")
        if isinstance(general, dict):
            terms.update(_tokenize(_as_text(general.get("field"))))
        context = profile.get("context")
        if isinstance(context, dict):
            background = context.get("background")
            if isinstance(background, dict):
                terms.update(_tokenize(_as_text(background.get("research_interests"))))
                for key in ("degrees", "projects", "work_experience", "publications", "skills"):
                    terms.update(_tokenize(_as_text(background.get(key))))

    topics.update(_topics_from_terms(terms))
    return terms, topics


def _collect_text_values(value: Any) -> list[str]:
    collected: list[str] = []
    _walk_collect(value, collected, depth=0)
    return collected


def _walk_collect(value: Any, out: list[str], depth: int) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        text = value.strip()
        if text:
            out.append(text)
        return
    if isinstance(value, dict):
        for item in value.values():
            _walk_collect(item, out, depth + 1)
        return
    if isinstance(value, list):
        for item in value:
            _walk_collect(item, out, depth + 1)


def _topics_from_terms(terms: set[str]) -> set[str]:
    found: set[str] = set()
    for topic, synonyms in _CANONICAL_TOPICS.items():
        for synonym in synonyms:
            token = synonym.lower().replace("-", "")
            if token in terms:
                found.add(topic)
                break
    return found


def _token_overlap_topics(professor_terms: set[str], student_terms: set[str]) -> set[str]:
    overlap = professor_terms.intersection(student_terms)
    return {token for token in overlap if len(token) >= 5}


def _build_gaps(
    *,
    professor_topics: set[str],
    student_topics: set[str],
    focus_misses: list[str],
    overlap_ratio: float,
) -> list[str]:
    gaps: list[str] = []
    for item in focus_misses:
        gaps.append(f"Missing explicit evidence for focus area '{item}'")

    unmatched_professor_topics = _sorted_list(professor_topics.difference(student_topics))
    for topic in unmatched_professor_topics[:3]:
        gaps.append(f"Limited evidence of experience in '{topic}'")

    if overlap_ratio < 0.15 and "Low research-topic overlap detected" not in gaps:
        gaps.append("Low research-topic overlap detected")

    if not gaps:
        gaps.append("No major topical gap identified from available profile data")
    return gaps[:5]


def _build_recommendations(
    *,
    label: str,
    focus_misses: list[str],
    matched_topics: list[str],
    gaps: list[str],
) -> list[str]:
    recommendations: list[str] = []
    if label == "high":
        recommendations.append("Proceed with outreach and cite the strongest matched topics explicitly.")
        recommendations.append("Tailor one paragraph to shared methods and expected contribution.")
    elif label == "medium":
        recommendations.append("Refine the draft to foreground overlap before discussing broad background.")
        recommendations.append("Add one concrete project/publication that supports the matched topics.")
    else:
        recommendations.append("Adjust target professor or refine research-interest positioning before outreach.")
        recommendations.append("Update profile and request fields with stronger evidence in matching areas.")

    for area in focus_misses[:2]:
        recommendations.append(f"Add evidence for focus area '{area}' in your draft and profile.")

    if not matched_topics and gaps:
        recommendations.append("Ask for a lightweight exploratory call instead of immediate supervision request.")
    return recommendations[:5]


def _build_rationale_short(
    *,
    label: str,
    overlap_ratio: float,
    matched_topics: list[str],
    gaps: list[str],
) -> str:
    overlap_pct = int(round(overlap_ratio * 100))
    if matched_topics:
        return (
            f"{label.title()} fit based on ~{overlap_pct}% topical overlap; "
            f"strongest matches: {', '.join(matched_topics[:3])}."
        )
    return (
        f"{label.title()} fit with ~{overlap_pct}% topical overlap; "
        f"main gap: {gaps[0] if gaps else 'insufficient evidence'}."
    )


def _build_rationale_long(
    *,
    professor_topics: set[str],
    student_topics: set[str],
    matched_topics: list[str],
    focus_areas: list[str],
    focus_hits: list[str],
    focus_misses: list[str],
    overlap_tokens: set[str],
    score: float,
    label: str,
) -> str:
    lines: list[str] = []
    lines.append(f"Computed {label} alignment (score={score:.3f}) from deterministic topic overlap heuristics.")
    lines.append(
        "Professor signals: "
        + (", ".join(_sorted_list(professor_topics)[:6]) if professor_topics else "none extracted")
        + "."
    )
    lines.append(
        "Student signals: "
        + (", ".join(_sorted_list(student_topics)[:6]) if student_topics else "none extracted")
        + "."
    )
    if matched_topics:
        lines.append("Matched topics: " + ", ".join(matched_topics[:6]) + ".")
    if focus_areas:
        lines.append("Requested focus areas: " + ", ".join(focus_areas[:6]) + ".")
    if focus_hits:
        lines.append("Focus-area hits: " + ", ".join(focus_hits[:6]) + ".")
    if focus_misses:
        lines.append("Focus-area misses: " + ", ".join(focus_misses[:6]) + ".")
    if overlap_tokens:
        lines.append("Shared evidence tokens: " + ", ".join(_sorted_list(overlap_tokens)[:12]) + ".")
    return " ".join(lines)


def _build_outcome(payload: dict[str, Any], *, producer_name: str, producer_version: str) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Alignment.Score",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {"name": producer_name, "version": producer_version, "plugin_type": "operator"},
    }


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


def _tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9\-]+", text)}
    out: set[str] = set()
    for token in tokens:
        token = token.strip("-")
        if not token:
            continue
        normalized = token.replace("-", "")
        if len(normalized) < 3:
            continue
        if normalized in _STOPWORDS:
            continue
        out.add(normalized)
    return out


def _normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _as_text(item)
        if not text:
            continue
        if text not in out:
            out.append(text)
    return out


def _sorted_list(values: set[str]) -> list[str]:
    return sorted(values)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value).strip() or None


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value
