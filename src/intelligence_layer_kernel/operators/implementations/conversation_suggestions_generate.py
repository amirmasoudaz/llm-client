from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorResult, OperatorMetrics, OperatorError


class ConversationSuggestionsGenerateOperator(Operator):
    name = "Conversation.Suggestions.Generate"
    version = "1.0.0"

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        intent_type = payload.get("intent_type")
        if not isinstance(intent_type, str) or intent_type.strip() == "":
            error = OperatorError(
                code="missing_intent_type",
                message="intent_type is required",
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

        outcomes = payload.get("outcomes") if isinstance(payload.get("outcomes"), dict) else {}
        platform = payload.get("platform") if isinstance(payload.get("platform"), dict) else {}

        suggestions = _build_suggestions(intent_type.strip(), outcomes, platform)
        outcome = _build_outcome({"suggestions": suggestions})

        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _build_suggestions(intent_type: str, outcomes: dict[str, Any], platform: dict[str, Any]) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    email_review = outcomes.get("email_review") if isinstance(outcomes.get("email_review"), dict) else None
    alignment_outcome = outcomes.get("alignment_score") if isinstance(outcomes.get("alignment_score"), dict) else None
    verdict = None
    if email_review and isinstance(email_review.get("payload"), dict):
        verdict = email_review["payload"].get("verdict")
    alignment_payload = alignment_outcome.get("payload") if isinstance(alignment_outcome, dict) else None
    if not isinstance(alignment_payload, dict):
        alignment_payload = None

    if verdict == "needs_edits":
        suggestions.append({"type": "followup", "text": "Revise the email based on the review feedback."})
        suggestions.append({"type": "followup", "text": "Show me a concise summary of the review issues."})
    elif verdict == "pass":
        suggestions.append({"type": "followup", "text": "Generate a shorter, sharper version of this email."})

    if intent_type == "Funding.Outreach.Alignment.Score" and alignment_payload:
        label = str(alignment_payload.get("label") or "").strip().lower()
        focus_areas = alignment_payload.get("focus_areas")
        if not isinstance(focus_areas, list):
            focus_areas = []
        matched_topics = alignment_payload.get("matched_topics")
        if not isinstance(matched_topics, list):
            matched_topics = []
        if label == "high":
            suggestions.append(
                {"type": "followup", "text": "Generate an outreach draft that highlights this strong alignment."}
            )
            suggestions.append(
                {
                    "type": "followup",
                    "text": "Summarize the top alignment evidence I should mention in my first paragraph.",
                }
            )
        elif label == "medium":
            suggestions.append(
                {
                    "type": "followup",
                    "text": "Optimize my draft to emphasize the best-matched topics for this professor.",
                }
            )
            suggestions.append(
                {"type": "followup", "text": "Show concrete profile updates that can improve my match score."}
            )
        else:
            suggestions.append(
                {
                    "type": "followup",
                    "text": "Update my research interest and profile data to improve alignment before outreach.",
                }
            )
            suggestions.append(
                {"type": "followup", "text": "Find another professor with a closer topic match."}
            )
        if focus_areas:
            suggestions.append(
                {
                    "type": "followup",
                    "text": f"Re-score alignment with focus on: {', '.join(str(x) for x in focus_areas[:3])}.",
                }
            )
        elif matched_topics:
            suggestions.append(
                {
                    "type": "followup",
                    "text": f"Draft a targeted email around: {', '.join(str(x) for x in matched_topics[:3])}.",
                }
            )

    funding_request = platform.get("funding_request") if isinstance(platform.get("funding_request"), dict) else {}
    if not funding_request.get("research_interest"):
        suggestions.append({"type": "followup", "text": "Update my research interest for this request."})

    if intent_type.startswith("Funding.Outreach"):
        suggestions.append({"type": "followup", "text": "Draft a polite follow-up email for this professor."})

    if not suggestions:
        suggestions.append({"type": "followup", "text": "What should I do next for this request?"})

    return suggestions[:3]


def _build_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Conversation.Suggestions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {"name": "Conversation.Suggestions.Generate", "version": "1.0.0", "plugin_type": "operator"},
    }
