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


class ReplyInterpretOperator(Operator):
    name = "Reply.Interpret"
    version = "1.0.0"

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        reply = payload.get("reply") if isinstance(payload.get("reply"), dict) else None
        if not isinstance(reply, dict):
            return _failed(start, "missing_reply", "reply is required")

        reply_text = _extract_reply_text(reply)
        if reply_text == "":
            return _failed(start, "missing_reply_text", "reply body is required")

        engagement_label = _safe_text(reply.get("engagement_label"))
        activity_status = _safe_text(reply.get("activity_status"))
        next_step_type = _safe_text(reply.get("next_step_type"))

        is_auto_generated = _coerce_bool(reply.get("is_auto_generated"))
        auto_generated_type = _normalize_auto_generated_type(reply.get("auto_generated_type"))
        needs_human_review = _coerce_bool(reply.get("needs_human_review"))
        security_flags = _coerce_text_list(reply.get("security_flags"))

        heuristic = _heuristic_reply_signals(reply_text)
        if not engagement_label:
            engagement_label = heuristic.get("engagement_label")
        if not activity_status:
            activity_status = heuristic.get("activity_status")
        if not next_step_type:
            next_step_type = heuristic.get("next_step_type")
        if auto_generated_type == "NONE" and heuristic.get("auto_generated_type"):
            auto_generated_type = str(heuristic["auto_generated_type"])
        is_auto_generated = is_auto_generated or auto_generated_type != "NONE"

        classification = _derive_classification(
            engagement_label=engagement_label,
            is_auto_generated=is_auto_generated,
            auto_generated_type=auto_generated_type,
            text=reply_text,
        )
        recommended_action = _derive_recommended_action(
            classification=classification,
            next_step_type=next_step_type,
            needs_human_review=needs_human_review,
        )

        confidence = _coerce_confidence(reply.get("confidence"))
        if confidence < 0.6 and not needs_human_review:
            needs_human_review = True

        derived_flags = _detect_security_flags(reply_text)
        for flag in derived_flags:
            if flag not in security_flags:
                security_flags.append(flag)
        if "prompt_injection_pattern" in security_flags:
            needs_human_review = True

        key_phrases = _coerce_text_list(reply.get("key_phrases"))
        if not key_phrases:
            key_phrases = _extract_key_phrases(reply_text)

        short_rationale = _safe_text(reply.get("short_rationale")) or _build_short_rationale(
            classification=classification,
            engagement_label=engagement_label,
            next_step_type=next_step_type,
        )
        summary = _build_summary(
            classification=classification,
            recommended_action=recommended_action,
            engagement_label=engagement_label,
        )

        outcome_payload = {
            "classification": classification,
            "engagement_label": engagement_label,
            "activity_status": activity_status,
            "next_step_type": next_step_type,
            "needs_human_review": needs_human_review,
            "is_auto_generated": is_auto_generated,
            "auto_generated_type": auto_generated_type,
            "recommended_action": recommended_action,
            "summary": summary,
            "short_rationale": short_rationale,
            "key_phrases": key_phrases[:6],
            "confidence": confidence,
            "security_flags": security_flags,
        }
        outcome_payload = {k: v for k, v in outcome_payload.items() if v is not None}

        outcome = _build_outcome(outcome_payload, producer_name=self.name, producer_version=self.version)
        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _extract_reply_text(reply: dict[str, Any]) -> str:
    cleaned = _safe_text(reply.get("reply_body_cleaned"))
    raw = _safe_text(reply.get("reply_body_raw"))
    return cleaned or raw or ""


def _derive_classification(
    *,
    engagement_label: str | None,
    is_auto_generated: bool,
    auto_generated_type: str,
    text: str,
) -> str:
    if is_auto_generated or auto_generated_type != "NONE":
        return "auto_generated"

    mapping = {
        "INTERESTED_AND_WANTS_TO_PROCEED": "interview",
        "POTENTIALLY_INTERESTED_NEEDS_MORE_INFO": "needs_info",
        "NOT_INTERESTED": "rejection",
        "NO_AVAILABLE_POSITION": "no_position",
        "REFERRAL_TO_SOMEONE_ELSE": "out_of_scope",
        "OUT_OF_SCOPE_OR_WRONG_PERSON": "out_of_scope",
        "AUTO_REPLY_OR_OUT_OF_OFFICE": "auto_generated",
        "AMBIGUOUS_OR_UNCLEAR": "ambiguous",
    }
    if engagement_label in mapping:
        return mapping[engagement_label]

    lowered = text.lower()
    if any(token in lowered for token in ("interview", "meeting", "zoom", "call")):
        return "interview"
    if any(token in lowered for token in ("cv", "transcript", "proposal", "more information", "details")):
        return "needs_info"
    if any(token in lowered for token in ("not accepting", "no openings", "group full", "no funding")):
        return "no_position"
    if any(token in lowered for token in ("not a fit", "cannot supervise", "can't help", "decline")):
        return "rejection"
    if any(token in lowered for token in ("contact admissions", "apply through", "reach out to")):
        return "out_of_scope"
    return "ambiguous"


def _derive_recommended_action(*, classification: str, next_step_type: str | None, needs_human_review: bool) -> str:
    if needs_human_review:
        return "manual_review"
    if classification in {"interview", "needs_info"}:
        return "draft_follow_up"
    if classification == "auto_generated":
        return "wait"
    if classification in {"rejection", "no_position", "out_of_scope"}:
        if next_step_type and next_step_type in {
            "APPLY_VIA_PORTAL_OR_CONTACT_ADMISSIONS",
            "REFER_TO_OTHER_PROFESSOR_OR_PERSON",
        }:
            return "draft_follow_up"
        return "stop_outreach"
    return "manual_review"


def _build_summary(*, classification: str, recommended_action: str, engagement_label: str | None) -> str:
    label_text = engagement_label or "unknown"
    return f"Reply classified as {classification} (engagement={label_text}); recommended action: {recommended_action}."


def _build_short_rationale(*, classification: str, engagement_label: str | None, next_step_type: str | None) -> str:
    parts = [f"classification={classification}"]
    if engagement_label:
        parts.append(f"engagement={engagement_label}")
    if next_step_type:
        parts.append(f"next_step={next_step_type}")
    return ", ".join(parts)


def _heuristic_reply_signals(text: str) -> dict[str, str]:
    lowered = text.lower()
    signal: dict[str, str] = {}

    if any(token in lowered for token in ("out of office", "automatic reply", "auto-reply", "mail delivery")):
        signal["engagement_label"] = "AUTO_REPLY_OR_OUT_OF_OFFICE"
        signal["auto_generated_type"] = "OUT_OF_OFFICE"
        signal["activity_status"] = "UNKNOWN"
        signal["next_step_type"] = "NO_NEXT_STEP"
        return signal

    if any(token in lowered for token in ("interview", "meeting", "zoom", "call")):
        signal["engagement_label"] = "INTERESTED_AND_WANTS_TO_PROCEED"
        signal["activity_status"] = "ACTIVE_SUPERVISING"
        signal["next_step_type"] = "REQUEST_MEETING"
        return signal

    if any(token in lowered for token in ("cv", "transcript", "proposal", "share more details", "more information")):
        signal["engagement_label"] = "POTENTIALLY_INTERESTED_NEEDS_MORE_INFO"
        signal["activity_status"] = "ACTIVE_SUPERVISING"
        signal["next_step_type"] = "REQUEST_RESEARCH_PROPOSAL_OR_DETAILS"
        return signal

    if any(token in lowered for token in ("not accepting", "no openings", "group full", "no funding", "lab is full")):
        signal["engagement_label"] = "NO_AVAILABLE_POSITION"
        signal["activity_status"] = "ACTIVE_NOT_SUPERVISING"
        signal["next_step_type"] = "NO_NEXT_STEP"
        return signal

    if any(token in lowered for token in ("retired", "emeritus", "no longer", "cannot supervise", "can't help")):
        signal["engagement_label"] = "NOT_INTERESTED"
        signal["activity_status"] = "CLEARLY_INACTIVE_SUPERVISION"
        signal["next_step_type"] = "NO_NEXT_STEP"
        return signal

    if any(token in lowered for token in ("apply through", "contact admissions", "reach out to", "graduate coordinator")):
        signal["engagement_label"] = "REFERRAL_TO_SOMEONE_ELSE"
        signal["activity_status"] = "UNKNOWN"
        signal["next_step_type"] = "APPLY_VIA_PORTAL_OR_CONTACT_ADMISSIONS"
        return signal

    signal["engagement_label"] = "AMBIGUOUS_OR_UNCLEAR"
    signal["activity_status"] = "UNKNOWN"
    signal["next_step_type"] = "NO_NEXT_STEP"
    return signal


def _detect_security_flags(text: str) -> list[str]:
    lowered = text.lower()
    flags: list[str] = []
    if len(text) > 20000:
        flags.append("reply_too_long")
    if re.search(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", lowered):
        flags.append("prompt_injection_pattern")
    if len(re.findall(r"https?://", lowered)) >= 5:
        flags.append("high_link_density")
    return flags


def _extract_key_phrases(text: str) -> list[str]:
    snippets: list[str] = []
    for raw in re.split(r"[\n\r]+", text):
        cleaned = raw.strip(" -\t")
        if len(cleaned) < 6:
            continue
        snippets.append(cleaned)
        if len(snippets) >= 6:
            break
    if snippets:
        return snippets
    fallback = [part.strip() for part in re.split(r"[.!?]", text) if part.strip()]
    return fallback[:6]


def _build_outcome(payload: dict[str, Any], *, producer_name: str, producer_version: str) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Reply.Interpretation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {
            "name": producer_name,
            "version": producer_version,
            "plugin_type": "operator",
        },
    }


def _coerce_confidence(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.7
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return parsed


def _normalize_auto_generated_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"OUT_OF_OFFICE", "DELIVERY_FAILURE", "OTHER_AUTO", "NONE"}:
        return text
    return "NONE"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return bool(int(value))
    except Exception:
        return False


def _coerce_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _safe_text(item)
        if not text or text in out:
            continue
        out.append(text)
    return out


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _failed(start: float, code: str, message: str) -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(code=code, message=message, category="validation", retryable=False),
    )
