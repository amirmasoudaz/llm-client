from __future__ import annotations

import asyncio
import difflib
import re

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.providers.types import Message
from llm_client.structured import StructuredOutputConfig, extract_structured


OLD_DRAFT = """
Security Review Policy

1. All production changes require peer review.
2. Rollback plans are recommended for high-risk changes.
3. Customer communications should be coordinated with support when incidents occur.
4. Analytics exports may run up to 12 hours behind during maintenance windows.
""".strip()

NEW_DRAFT = """
Security Review Policy

1. All production changes require peer review and release-manager acknowledgement.
2. Rollback plans are mandatory for high-risk changes and must be rehearsed before release.
3. Customer communications should be coordinated with support and legal when incidents may affect compliance reporting.
4. Analytics exports may run up to 4 hours behind during maintenance windows.
5. Sev-1 incidents require an executive update within 30 minutes.
""".strip()


def _extract_sections(text: str) -> dict[str, str]:
    matches = re.findall(r"^(\d+)\.\s+(.*)$", text, flags=re.MULTILINE)
    return {number: body.strip() for number, body in matches}


def _build_change_matrix(old_text: str, new_text: str) -> list[dict[str, str]]:
    old_sections = _extract_sections(old_text)
    new_sections = _extract_sections(new_text)
    clause_ids = sorted(set(old_sections) | set(new_sections), key=int)
    matrix: list[dict[str, str]] = []
    for clause_id in clause_ids:
        old_clause = old_sections.get(clause_id)
        new_clause = new_sections.get(clause_id)
        if old_clause == new_clause:
            continue
        matrix.append(
            {
                "clause": clause_id,
                "old": old_clause or "<missing>",
                "new": new_clause or "<missing>",
            }
        )
    return matrix


def _approval_summary(data: dict | None) -> str | None:
    if not data:
        return None
    changes = "\n".join(
        f"- Clause {item['clause']}: {item['risk_level']} risk, impacts {item['affected_team']}."
        for item in list(data.get("material_changes") or [])
    )
    signoffs = "\n".join(f"- {item}" for item in list(data.get("required_signoffs") or []))
    questions = "\n".join(f"- {item}" for item in list(data.get("follow_up_questions") or []))
    return (
        f"Review Verdict\n- {data.get('review_verdict')}\n\n"
        f"Approval Recommendation\n- {data.get('approval_recommendation')}\n\n"
        f"Material Changes\n{changes}\n\n"
        f"Required Signoffs\n{signoffs or '- None'}\n\n"
        f"Open Questions\n{questions or '- None'}"
    )


async def main() -> None:
    handle = build_live_provider()
    try:
        change_matrix = _build_change_matrix(OLD_DRAFT, NEW_DRAFT)
        unified_diff = "\n".join(
            difflib.unified_diff(
                OLD_DRAFT.splitlines(),
                NEW_DRAFT.splitlines(),
                fromfile="old_policy.md",
                tofile="new_policy.md",
                lineterm="",
            )
        )
        result = await extract_structured(
            handle.provider,
            [
                Message.system(
                    "You are a document review analyst. Compare policy drafts and extract only material changes with operational consequences."
                ),
                Message.user(
                    "Old draft:\n"
                    f"{OLD_DRAFT}\n\n"
                    "New draft:\n"
                    f"{NEW_DRAFT}\n\n"
                    "Clause change matrix:\n"
                    f"{change_matrix}\n\n"
                    "Unified diff:\n"
                    f"{unified_diff}"
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "material_changes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "clause": {"type": "string"},
                                    "change_summary": {"type": "string"},
                                    "operational_impact": {"type": "string"},
                                    "risk_level": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high"],
                                    },
                                    "affected_team": {"type": "string"},
                                },
                                "required": [
                                    "clause",
                                    "change_summary",
                                    "operational_impact",
                                    "risk_level",
                                    "affected_team",
                                ],
                                "additionalProperties": False,
                            },
                            "minItems": 3,
                        },
                        "review_verdict": {
                            "type": "string",
                            "enum": ["approve", "approve_with_conditions", "escalate"],
                        },
                        "approval_recommendation": {"type": "string"},
                        "required_signoffs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "follow_up_questions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "material_changes",
                        "review_verdict",
                        "approval_recommendation",
                        "required_signoffs",
                        "follow_up_questions",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
            reasoning_effort="low",
        )

        print_heading("Document Review Diff")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "old_draft": OLD_DRAFT,
                "new_draft": NEW_DRAFT,
                "change_matrix": change_matrix,
                "unified_diff": unified_diff,
                "structured_review": {
                    "valid": result.valid,
                    "repair_attempts": result.repair_attempts,
                    "usage": summarize_usage(getattr(result, "usage", None)),
                    "data": result.data,
                    "validation_errors": result.validation_errors,
                },
                "approval_summary": _approval_summary(result.data),
                "showcase_verdict": {
                    "has_clause_level_analysis": bool(result.data and result.data.get("material_changes")),
                    "has_signoff_routing": bool(result.data and result.data.get("required_signoffs")),
                    "ready_for_review_board": bool(result.valid and _approval_summary(result.data)),
                },
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
