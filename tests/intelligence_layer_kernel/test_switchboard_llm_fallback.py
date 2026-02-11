from __future__ import annotations

from typing import Any

from intelligence_layer_kernel.runtime.switchboard import IntentSwitchboard


def test_switchboard_uses_llm_json_fallback_when_deterministic_router_has_no_match() -> None:
    calls: list[dict[str, Any]] = []

    def fallback(*, message: str, allowed_intents: list[str], attachment_ids: list[int]) -> dict[str, Any]:
        calls.append(
            {
                "message": message,
                "allowed_intents": list(allowed_intents),
                "attachment_ids": list(attachment_ids),
            }
        )
        return {
            "intent_type": "Funding.Outreach.Professor.Summarize",
            "inputs": {"professor_id": 42},
        }

    switchboard = IntentSwitchboard(llm_json_fallback=fallback)
    intent_type, inputs = switchboard.classify(
        "Can you decide what I should do next?",
        attachment_ids=[5],
        allowed_intents=["Funding.Outreach.Professor.Summarize", "Funding.Outreach.Email.Review"],
    )

    assert intent_type == "Funding.Outreach.Professor.Summarize"
    assert inputs == {"professor_id": 42}
    assert len(calls) == 1
    assert calls[0]["attachment_ids"] == [5]


def test_switchboard_rejects_llm_fallback_intent_outside_allowlist() -> None:
    def fallback(*, message: str, allowed_intents: list[str], attachment_ids: list[int]) -> dict[str, Any]:
        _ = (message, allowed_intents, attachment_ids)
        return {"intent_type": "Workflow.Gate.Resolve", "inputs": {"status": "accepted"}}

    switchboard = IntentSwitchboard(llm_json_fallback=fallback)
    intent_type, inputs = switchboard.classify(
        "Not sure what this means",
        allowed_intents=["Funding.Outreach.Email.Review"],
    )

    assert intent_type == "Funding.Outreach.Email.Review"
    assert inputs == {}


def test_switchboard_accepts_json_string_payload_from_llm_fallback() -> None:
    def fallback(*, message: str, allowed_intents: list[str], attachment_ids: list[int]) -> str:
        _ = (message, allowed_intents, attachment_ids)
        return '{"intent_type":"Funding.Outreach.Email.Optimize","inputs":{"requested_edits":["shorten"]}}'

    switchboard = IntentSwitchboard(llm_json_fallback=fallback)
    intent_type, inputs = switchboard.classify(
        "Please decide",
        allowed_intents=["Funding.Outreach.Email.Optimize"],
    )

    assert intent_type == "Funding.Outreach.Email.Optimize"
    assert inputs == {"requested_edits": ["shorten"]}


def test_switchboard_enforces_documents_review_default_inputs_for_llm_fallback() -> None:
    def fallback(*, message: str, allowed_intents: list[str], attachment_ids: list[int]) -> dict[str, Any]:
        _ = (message, allowed_intents, attachment_ids)
        return {"intent_type": "Documents.Review"}

    switchboard = IntentSwitchboard(llm_json_fallback=fallback)
    intent_type, inputs = switchboard.classify(
        "Please figure it out",
        attachment_ids=[7],
        allowed_intents=["Documents.Review"],
    )

    assert intent_type == "Documents.Review"
    assert inputs["document_type"] == "cv"
    assert inputs["attachment_ids"] == [7]


def test_switchboard_routes_reply_followup_intent_with_reply_id() -> None:
    switchboard = IntentSwitchboard()
    intent_type, inputs = switchboard.classify(
        "Professor replied. Draft follow-up for reply id 42.",
        allowed_intents=["Funding.Outreach.FollowUp.Draft", "Funding.Outreach.Email.Review"],
    )

    assert intent_type == "Funding.Outreach.FollowUp.Draft"
    assert inputs["reply_id"] == 42
    assert "custom_instructions" in inputs
