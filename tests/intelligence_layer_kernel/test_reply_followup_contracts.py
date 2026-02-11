from __future__ import annotations

from typing import Any

import pytest
from jsonschema import Draft202012Validator

from intelligence_layer_kernel.contracts import ContractRegistry
from intelligence_layer_kernel.operators.implementations.follow_up_draft import (
    FollowUpDraftOperator,
)
from intelligence_layer_kernel.operators.implementations.funding_reply_load import (
    FundingReplyLoadOperator,
)
from intelligence_layer_kernel.operators.implementations.reply_interpret import (
    ReplyInterpretOperator,
)
from tests.intelligence_layer_kernel._phase_fg_testkit import build_operator_call


class _ReplyDB:
    async def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        normalized = " ".join(sql.split()).lower()
        if normalized.startswith("show columns from funding_replies"):
            return [
                {"Field": "id"},
                {"Field": "funding_request_id"},
                {"Field": "reply_body_raw"},
                {"Field": "reply_body_cleaned"},
                {"Field": "is_auto_generated"},
                {"Field": "auto_generated_type"},
                {"Field": "needs_human_review"},
                {"Field": "engagement_label"},
                {"Field": "engagement_bool"},
                {"Field": "activity_status"},
                {"Field": "activity_bool"},
                {"Field": "next_step_type"},
                {"Field": "short_rationale"},
                {"Field": "key_phrases"},
                {"Field": "confidence"},
            ]
        if "from funding_replies" in normalized:
            _ = params
            return [
                {
                    "id": 501,
                    "funding_request_id": 33,
                    "reply_body_raw": (
                        "Thanks for reaching out. Please send your CV and a short research summary, "
                        "then we can schedule a meeting next week."
                    ),
                    "reply_body_cleaned": (
                        "Please send your CV and a short research summary, then we can schedule a meeting next week."
                    ),
                    "is_auto_generated": 0,
                    "auto_generated_type": "NONE",
                    "needs_human_review": 0,
                    "engagement_label": "POTENTIALLY_INTERESTED_NEEDS_MORE_INFO",
                    "engagement_bool": 1,
                    "activity_status": "ACTIVE_SUPERVISING",
                    "activity_bool": 1,
                    "next_step_type": "REQUEST_CV",
                    "short_rationale": "Professor asked for additional material before deciding.",
                    "key_phrases": '["send your CV", "research summary", "schedule a meeting"]',
                    "confidence": 0.93,
                }
            ]
        raise AssertionError(f"Unexpected SQL in fake DB: {sql}")


def _validate_schema(registry: ContractRegistry, *, schema_ref: str, instance: dict[str, Any]) -> None:
    schema = registry.get_schema_by_ref(schema_ref)
    validator = Draft202012Validator(schema, resolver=registry.resolver_for(schema))
    errors = list(validator.iter_errors(instance))
    assert not errors, "; ".join(err.message for err in errors)


@pytest.mark.asyncio
async def test_reply_interpret_and_followup_operators_match_contract_schemas() -> None:
    registry = ContractRegistry()
    registry.load()

    load_operator = FundingReplyLoadOperator(pool=object(), tenant_id=1)
    load_operator._db = _ReplyDB()
    interpret_operator = ReplyInterpretOperator()
    followup_operator = FollowUpDraftOperator()

    load_result = await load_operator.run(build_operator_call({"funding_request_id": 33}))
    assert load_result.status == "succeeded"
    _validate_schema(
        registry,
        schema_ref="schemas/operators/funding_reply_load.output.v1.json",
        instance=load_result.to_dict(),
    )

    latest_reply = (load_result.result or {}).get("latest_reply")
    assert isinstance(latest_reply, dict)

    interpret_result = await interpret_operator.run(
        build_operator_call(
            {
                "reply": latest_reply,
                "funding_request": {"id": 33, "email_subject": "Prospective student inquiry"},
                "email_context": {"id": 11, "main_email_subject": "Prospective student inquiry"},
            }
        )
    )
    assert interpret_result.status == "succeeded"
    _validate_schema(
        registry,
        schema_ref="schemas/operators/reply_interpret.output.v1.json",
        instance=interpret_result.to_dict(),
    )

    interpretation_outcome = (interpret_result.result or {}).get("outcome")
    assert isinstance(interpretation_outcome, dict)
    _validate_schema(
        registry,
        schema_ref="schemas/outcomes/reply_interpretation.v1.json",
        instance=interpretation_outcome,
    )

    followup_result = await followup_operator.run(
        build_operator_call(
            {
                "reply_interpretation": interpretation_outcome,
                "reply": latest_reply,
                "funding_request": {"id": 33, "email_subject": "Prospective student inquiry"},
                "email_context": {"id": 11, "main_email_subject": "Prospective student inquiry"},
                "tone": "professional",
            }
        )
    )
    assert followup_result.status == "succeeded"
    _validate_schema(
        registry,
        schema_ref="schemas/operators/follow_up_draft.output.v1.json",
        instance=followup_result.to_dict(),
    )

    followup_outcome = (followup_result.result or {}).get("outcome")
    assert isinstance(followup_outcome, dict)
    _validate_schema(
        registry,
        schema_ref="schemas/outcomes/email_draft.v1.json",
        instance=followup_outcome,
    )

    draft_payload = followup_outcome.get("payload") if isinstance(followup_outcome, dict) else {}
    assert draft_payload.get("goal") == "follow_up"
