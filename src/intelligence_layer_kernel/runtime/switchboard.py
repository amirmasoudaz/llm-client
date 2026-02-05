from __future__ import annotations

from typing import Any
import re


class IntentSwitchboard:
    def __init__(self) -> None:
        self._rules: list[tuple[str, str]] = [
            ("optimize email", "Funding.Outreach.Email.Optimize"),
            ("email optimize", "Funding.Outreach.Email.Optimize"),
            ("review email", "Funding.Outreach.Email.Review"),
            ("email review", "Funding.Outreach.Email.Review"),
            ("generate email", "Funding.Outreach.Email.Generate"),
            ("email draft", "Funding.Outreach.Email.Generate"),
            ("alignment", "Funding.Outreach.Alignment.Score"),
            ("match", "Funding.Outreach.Alignment.Score"),
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

    def classify(self, message: str) -> tuple[str, dict[str, Any]]:
        text = message.lower().strip()
        match = re.search(r"research interest(?:\s+to|\s*:)\s*(.+)", text)
        if match:
            value = match.group(1).strip()
            if value:
                return "Funding.Request.Fields.Update", {"fields": {"research_interest": value}}
        if "email" in text and ("review" in text or "feedback" in text or "critique" in text):
            return "Funding.Outreach.Email.Review", {}
        if "draft" in text and "review" in text:
            return "Funding.Outreach.Email.Review", {}
        for needle, intent_type in self._rules:
            if needle in text:
                return intent_type, {}
        # Default to email review when unsure.
        return "Funding.Outreach.Email.Review", {}
