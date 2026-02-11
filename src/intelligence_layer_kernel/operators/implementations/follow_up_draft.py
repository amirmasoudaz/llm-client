from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult


class FollowUpDraftOperator(Operator):
    name = "FollowUp.Draft"
    version = "1.0.0"

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        interpretation_raw = payload.get("reply_interpretation")
        interpretation_payload = _extract_interpretation_payload(interpretation_raw)
        if not isinstance(interpretation_payload, dict):
            return _failed(start, "missing_reply_interpretation", "reply_interpretation is required")

        reply = payload.get("reply") if isinstance(payload.get("reply"), dict) else None
        if not isinstance(reply, dict):
            return _failed(start, "missing_reply", "reply is required")

        classification = str(interpretation_payload.get("classification") or "ambiguous")
        recommended_action = str(interpretation_payload.get("recommended_action") or "manual_review")
        summary = str(interpretation_payload.get("summary") or "")

        funding_request = payload.get("funding_request") if isinstance(payload.get("funding_request"), dict) else {}
        email_context = payload.get("email_context") if isinstance(payload.get("email_context"), dict) else {}
        student_profile = payload.get("student_profile") if isinstance(payload.get("student_profile"), dict) else {}

        subject_override = _safe_text(payload.get("subject_override"))
        subject = subject_override or _derive_subject(
            classification=classification,
            base_subject=_safe_text(email_context.get("main_email_subject"))
            or _safe_text(funding_request.get("email_subject")),
        )

        tone = _normalize_tone(payload.get("tone"))
        include_signature = _coerce_bool(payload.get("include_signature"), default=True)
        custom_instructions = _safe_text(payload.get("custom_instructions"))
        student_name = _resolve_student_name(student_profile=student_profile, funding_request=funding_request)

        body = _build_body(
            classification=classification,
            recommended_action=recommended_action,
            summary=summary,
            student_name=student_name,
            tone=tone,
            custom_instructions=custom_instructions,
            include_signature=include_signature,
        )

        outcome_payload: dict[str, Any] = {
            "funding_request_id": _coerce_positive_int(funding_request.get("id")),
            "email_id": _coerce_positive_int(email_context.get("id")),
            "goal": "follow_up",
            "language": "en",
            "subject": subject,
            "body": body,
            "rationale": f"Built from reply interpretation: {classification} / {recommended_action}",
            "notes": "Draft only. Review before sending.",
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


def _extract_interpretation_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("payload"), dict) and value.get("outcome_type") == "Reply.Interpretation":
        return value.get("payload")
    if isinstance(value.get("outcome"), dict):
        nested = value.get("outcome")
        if isinstance(nested.get("payload"), dict):
            return nested.get("payload")
    if isinstance(value.get("payload"), dict):
        return value.get("payload")
    return value if value else None


def _derive_subject(*, classification: str, base_subject: str | None) -> str:
    base = base_subject or "Follow-up on my outreach"
    if not base.lower().startswith("re:"):
        base = f"Re: {base}"
    if classification == "needs_info":
        return f"{base} - additional information"
    if classification == "interview":
        return f"{base} - meeting follow-up"
    if classification in {"rejection", "no_position", "out_of_scope"}:
        return f"{base} - thank you"
    return base


def _build_body(
    *,
    classification: str,
    recommended_action: str,
    summary: str,
    student_name: str | None,
    tone: str,
    custom_instructions: str | None,
    include_signature: bool,
) -> str:
    greeting = "Dear Professor,"
    signoff = "Best regards,"
    if tone == "warm":
        signoff = "Warm regards,"
    if tone == "concise":
        signoff = "Regards,"

    opener = "Thank you for your reply."
    core_lines: list[str] = []

    if classification == "interview":
        core_lines.append("I appreciate your openness to discussing next steps.")
        core_lines.append("I would be happy to share any materials you need and coordinate a meeting time.")
    elif classification == "needs_info":
        core_lines.append("Thank you for outlining what information would be helpful.")
        core_lines.append("I can provide the requested details and documents right away.")
    elif classification in {"rejection", "no_position"}:
        core_lines.append("Thank you for the clarification and for your time.")
        core_lines.append("I appreciate your response and wish you continued success in your research.")
    elif classification == "out_of_scope":
        core_lines.append("Thank you for pointing me to the correct process/contact.")
        core_lines.append("I will follow your suggestion for the next step.")
    elif classification == "auto_generated":
        core_lines.append("I understand this appears to be an automated reply.")
        core_lines.append("I will follow up again at a better time.")
    else:
        core_lines.append("Thank you for your response.")
        core_lines.append("Could you please clarify the best next step when you have a moment?")

    if recommended_action == "manual_review":
        core_lines.append("I am keeping this follow-up concise until we confirm the right direction.")

    if summary:
        core_lines.append(f"Summary noted: {summary}")

    if custom_instructions:
        core_lines.append(f"Context from my side: {custom_instructions}")

    lines = [greeting, "", opener, *core_lines, "", signoff]
    if include_signature and student_name:
        lines.append(student_name)

    return "\n".join(line for line in lines if line is not None)


def _resolve_student_name(*, student_profile: dict[str, Any], funding_request: dict[str, Any]) -> str | None:
    general = student_profile.get("general") if isinstance(student_profile.get("general"), dict) else {}
    first_name = _safe_text(general.get("first_name"))
    last_name = _safe_text(general.get("last_name"))
    if not first_name:
        first_name = _safe_text(funding_request.get("student_first_name"))
    if not last_name:
        last_name = _safe_text(funding_request.get("student_last_name"))
    full = " ".join(part for part in [first_name, last_name] if part)
    return full or None


def _normalize_tone(value: Any) -> str:
    tone = str(value or "").strip().lower()
    if tone in {"professional", "warm", "concise"}:
        return tone
    return "professional"


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
        "producer": {
            "name": producer_name,
            "version": producer_version,
            "plugin_type": "operator",
        },
    }


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _failed(start: float, code: str, message: str) -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(code=code, message=message, category="validation", retryable=False),
    )
