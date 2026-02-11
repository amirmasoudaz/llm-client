from __future__ import annotations

import json
import re
from typing import Any, Protocol


class IntentJsonFallback(Protocol):
    def __call__(
        self,
        *,
        message: str,
        allowed_intents: list[str],
        attachment_ids: list[int],
    ) -> Any:
        ...


class IntentSwitchboard:
    def __init__(
        self,
        *,
        llm_json_fallback: IntentJsonFallback | None = None,
    ) -> None:
        self._llm_json_fallback = llm_json_fallback
        self._rules: list[tuple[str, str]] = [
            ("complete my profile", "Student.Profile.Collect"),
            ("update my profile", "Student.Profile.Collect"),
            ("onboarding", "Student.Profile.Collect"),
            ("optimize email", "Funding.Outreach.Email.Optimize"),
            ("email optimize", "Funding.Outreach.Email.Optimize"),
            ("review email", "Funding.Outreach.Email.Review"),
            ("email review", "Funding.Outreach.Email.Review"),
            ("generate email", "Funding.Outreach.Email.Generate"),
            ("email draft", "Funding.Outreach.Email.Generate"),
            ("alignment", "Funding.Outreach.Alignment.Score"),
            ("match", "Funding.Outreach.Alignment.Score"),
            ("fit score", "Funding.Outreach.Alignment.Score"),
            ("good fit", "Funding.Outreach.Alignment.Score"),
            ("should i reach out", "Funding.Outreach.Alignment.Score"),
            ("professor match", "Funding.Outreach.Alignment.Score"),
            ("summarize professor", "Funding.Outreach.Professor.Summarize"),
            ("professor summary", "Funding.Outreach.Professor.Summarize"),
            ("update request", "Funding.Request.Fields.Update"),
            ("update fields", "Funding.Request.Fields.Update"),
            ("paper metadata", "Funding.Paper.Metadata.Extract"),
            ("extract metadata", "Funding.Paper.Metadata.Extract"),
            ("review document", "Documents.Review"),
            ("review cv", "Documents.Review"),
            ("review sop", "Documents.Review"),
        ]

    def classify(
        self,
        message: str,
        *,
        attachment_ids: list[int] | None = None,
        allowed_intents: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        text = message.strip()
        attachment_ids = [int(item) for item in (attachment_ids or []) if int(item) > 0]
        allowlist = self._normalize_allowed_intents(allowed_intents)

        deterministic = self._classify_deterministic(text, attachment_ids=attachment_ids)
        if deterministic is not None and self._is_allowed_intent(deterministic[0], allowlist):
            return deterministic

        llm_candidate = self._classify_with_llm_json(
            text=text,
            allowlist=allowlist,
            attachment_ids=attachment_ids,
        )
        if llm_candidate is not None and self._is_allowed_intent(llm_candidate[0], allowlist):
            return llm_candidate

        return self._fallback_intent(
            text=text,
            allowlist=allowlist,
            attachment_ids=attachment_ids,
        )

    def _classify_deterministic(
        self,
        text: str,
        *,
        attachment_ids: list[int],
    ) -> tuple[str, dict[str, Any]] | None:
        lowered = text.lower()
        field_values = self._extract_funding_request_fields(text)
        if field_values:
            field_list = ", ".join(sorted(field_values.keys()))
            return (
                "Funding.Request.Fields.Update",
                {
                    "fields": field_values,
                    "human_summary": f"Update funding request fields: {field_list}",
                },
            )
        optimize_inputs = self._extract_email_optimize_inputs(text)
        if optimize_inputs is not None:
            return "Funding.Outreach.Email.Optimize", optimize_inputs
        document_review_inputs = self._extract_document_review_inputs(text, attachment_ids=attachment_ids)
        if document_review_inputs is not None:
            return "Documents.Review", document_review_inputs
        if "profile" in lowered and ("update" in lowered or "complete" in lowered or "onboarding" in lowered):
            return "Student.Profile.Collect", {}
        if lowered.startswith("remember ") or "my preference" in lowered:
            return "Student.Profile.Collect", {}
        if "email" in lowered and ("review" in lowered or "feedback" in lowered or "critique" in lowered):
            return "Funding.Outreach.Email.Review", {}
        if "draft" in lowered and "review" in lowered:
            return "Funding.Outreach.Email.Review", {}
        for needle, intent_type in self._rules:
            if needle in lowered:
                if intent_type == "Documents.Review":
                    return intent_type, self._default_document_review_inputs(text, attachment_ids=attachment_ids)
                return intent_type, {}
        return None

    def _classify_with_llm_json(
        self,
        *,
        text: str,
        allowlist: list[str],
        attachment_ids: list[int],
    ) -> tuple[str, dict[str, Any]] | None:
        if self._llm_json_fallback is None:
            return None
        try:
            raw = self._llm_json_fallback(
                message=text,
                allowed_intents=list(allowlist),
                attachment_ids=list(attachment_ids),
            )
        except Exception:
            return None
        payload = self._coerce_fallback_payload(raw)
        if payload is None:
            return None
        intent_type = str(payload.get("intent_type") or payload.get("intent") or "").strip()
        if not intent_type:
            return None
        if allowlist and intent_type not in set(allowlist):
            return None
        inputs = payload.get("inputs")
        resolved_inputs = inputs if isinstance(inputs, dict) else {}
        if intent_type == "Documents.Review" and not resolved_inputs:
            resolved_inputs = self._default_document_review_inputs(text, attachment_ids=attachment_ids)
        return intent_type, resolved_inputs

    def _coerce_fallback_payload(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
        return None

    def _fallback_intent(
        self,
        *,
        text: str,
        allowlist: list[str],
        attachment_ids: list[int],
    ) -> tuple[str, dict[str, Any]]:
        default_intent = "Funding.Outreach.Email.Review"
        if allowlist and default_intent not in set(allowlist):
            default_intent = allowlist[0]
        if default_intent == "Documents.Review":
            return default_intent, self._default_document_review_inputs(text, attachment_ids=attachment_ids)
        return default_intent, {}

    def _normalize_allowed_intents(self, allowed_intents: list[str] | None) -> list[str]:
        if not isinstance(allowed_intents, list):
            return []
        out: list[str] = []
        for item in allowed_intents:
            value = str(item).strip()
            if not value or value in out:
                continue
            out.append(value)
        return out

    def _is_allowed_intent(self, intent_type: str, allowlist: list[str]) -> bool:
        if not allowlist:
            return True
        return intent_type in set(allowlist)

    def _extract_funding_request_fields(self, text: str) -> dict[str, Any]:
        field_labels = r"research interest|paper title|journal|year|research connection"
        patterns: dict[str, str] = {
            "research_interest": rf"research interest\s*(?:to|:|=)\s*(.+?)(?=(?:\s+and\s+(?:{field_labels})\s*(?:to|:|=))|$)",
            "paper_title": rf"paper title\s*(?:to|:|=)\s*(.+?)(?=(?:\s+and\s+(?:{field_labels})\s*(?:to|:|=))|$)",
            "journal": rf"journal\s*(?:to|:|=)\s*(.+?)(?=(?:\s+and\s+(?:{field_labels})\s*(?:to|:|=))|$)",
            "research_connection": rf"research connection\s*(?:to|:|=)\s*(.+?)(?=(?:\s+and\s+(?:{field_labels})\s*(?:to|:|=))|$)",
            "year": rf"year\s*(?:to|:|=)\s*(\d{{4}})(?=(?:\s+and\s+(?:{field_labels})\s*(?:to|:|=))|$)",
        }
        extracted: dict[str, Any] = {}
        for field_name, pattern in patterns.items():
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            raw_value = match.group(1).strip(" .,\t\n\r")
            if not raw_value:
                continue
            if field_name == "year":
                try:
                    year_value = int(raw_value)
                except ValueError:
                    continue
                extracted[field_name] = year_value
            else:
                extracted[field_name] = raw_value
        return extracted

    def _extract_email_optimize_inputs(self, text: str) -> dict[str, Any] | None:
        lowered = text.lower()
        optimize_cues = (
            "optimize",
            "improve",
            "shorter",
            "shorten",
            "concise",
            "humanize",
            "rewrite",
            "reword",
            "clarity",
            "polish",
            "refine",
        )
        if not any(cue in lowered for cue in optimize_cues):
            return None

        edits: list[str] = []
        if any(token in lowered for token in ("shorter", "shorten", "concise", "tighten")):
            edits.append("shorten")
        if any(token in lowered for token in ("humanize", "warmer", "friendlier", "tone")):
            edits.append("humanize")
        if any(token in lowered for token in ("rewrite", "reword", "paraphrase")):
            edits.append("paraphrase")
        if any(token in lowered for token in ("clarity", "clearer", "improve")):
            edits.append("improve_clarity")
        if "bullet" in lowered:
            edits.append("add_bullets")
        if "subject" in lowered:
            edits.append("change_subject")
        if "hook" in lowered:
            edits.append("add_custom_hook")

        normalized_edits: list[str] = []
        for edit in edits:
            if edit not in normalized_edits:
                normalized_edits.append(edit)
        if not normalized_edits:
            normalized_edits = ["improve_clarity"]

        inputs: dict[str, Any] = {
            "requested_edits": normalized_edits,
            "custom_instructions": text.strip(),
        }
        version_match = re.search(r"\bversion\s+(\d+)\b", lowered)
        if version_match:
            try:
                version_number = int(version_match.group(1))
            except ValueError:
                version_number = None
            if version_number is not None and version_number > 0:
                inputs["source_draft_version"] = version_number

        subject_override = self._extract_subject_override(text)
        if subject_override:
            inputs["subject_override"] = subject_override
        return inputs

    def _extract_subject_override(self, text: str) -> str | None:
        match = re.search(r"subject(?: line)?\s*(?:to|as|=)\s*(.+)", text, flags=re.IGNORECASE)
        if not match:
            return None
        value = match.group(1).strip(" '\"\t\n\r.")
        if not value:
            return None
        return value

    def _extract_document_review_inputs(
        self,
        text: str,
        *,
        attachment_ids: list[int] | None = None,
    ) -> dict[str, Any] | None:
        lowered = text.lower()
        review_cues = ("review", "assess", "evaluate", "critique", "feedback")
        if not any(cue in lowered for cue in review_cues):
            return None

        document_type = self._infer_document_type(lowered)
        if document_type is None and "document" not in lowered and "attachment" not in lowered:
            return None

        return self._default_document_review_inputs(
            text,
            document_type=document_type,
            attachment_ids=attachment_ids,
        )

    def _default_document_review_inputs(
        self,
        text: str,
        *,
        document_type: str | None = None,
        attachment_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        lowered = text.lower()
        doc_type = document_type or self._infer_document_type(lowered) or "cv"

        review_goal = "quality"
        if any(token in lowered for token in ("concise", "short", "brevity", "shorter")):
            review_goal = "brevity"
        elif any(token in lowered for token in ("grammar", "clarity", "clear", "readability")):
            review_goal = "clarity"

        inputs: dict[str, Any] = {
            "document_type": doc_type,
            "review_goal": review_goal,
            "custom_instructions": text.strip(),
        }
        if attachment_ids:
            inputs["attachment_ids"] = [int(item) for item in attachment_ids if int(item) > 0]
        return inputs

    def _infer_document_type(self, lowered_text: str) -> str | None:
        if any(token in lowered_text for token in ("cv", "resume")):
            return "cv"
        if any(
            token in lowered_text
            for token in ("sop", "statement of purpose", "personal statement", "cover letter", "motivation letter")
        ):
            if any(token in lowered_text for token in ("cover letter", "motivation letter")):
                return "letter"
            return "sop"
        if "transcript" in lowered_text:
            return "transcript"
        if "portfolio" in lowered_text:
            return "portfolio"
        if "study plan" in lowered_text:
            return "study_plan"
        return None
