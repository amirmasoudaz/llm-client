from __future__ import annotations

from llm_client.structured import normalize_structured_result_envelope


def test_normalize_structured_result_envelope_strips_empty_optional_metadata() -> None:
    envelope = normalize_structured_result_envelope(
        {
            "status": "succeeded",
            "result": {"outcome": {"assistant_message": {"text": "hello"}}},
            "nondeterminism": {
                "is_nondeterministic": True,
                "reasons": ["llm_structured"],
                "stability": "medium",
                "fallback_mode": "",
                "llm_failure_code": "",
            },
            "error": {"code": "", "message": "", "category": ""},
        },
        default_error_code="llm_operator_failed",
        default_error_message="llm_structured operator failed",
        default_error_category="operator_bug",
    )

    assert envelope.status == "succeeded"
    assert envelope.error is None
    assert "fallback_mode" not in envelope.nondeterminism
    assert "llm_failure_code" not in envelope.nondeterminism


def test_normalize_structured_result_envelope_promotes_in_progress_with_result() -> None:
    envelope = normalize_structured_result_envelope(
        {
            "status": "in_progress",
            "result": {
                "outcome": {"assistant_message": {"text": "Please share your first name."}},
                "workflow_control": {
                    "wait_for_action": {
                        "action_type": "collect_profile_fields",
                        "reason_code": "MISSING_REQUIRED_PROFILE_FIELDS",
                        "description": "Please share your first name.",
                        "requires_user_input": True,
                    }
                },
            },
        },
        promote_in_progress_with_result=True,
    )

    assert envelope.status == "succeeded"
    assert envelope.result is not None
    assert envelope.result["workflow_control"]["wait_for_action"]["action_type"] == "collect_profile_fields"
