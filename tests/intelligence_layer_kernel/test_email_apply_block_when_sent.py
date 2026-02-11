from __future__ import annotations

import pytest

from intelligence_layer_kernel.operators.implementations.funding_email_draft_update_apply import (
    FundingEmailDraftUpdateApplyOperator,
)
from intelligence_layer_kernel.operators.implementations.funding_email_draft_update_propose import (
    FundingEmailDraftUpdateProposeOperator,
)
from tests.intelligence_layer_kernel._phase_fg_testkit import FakePlatformDB, build_operator_call


@pytest.mark.asyncio
async def test_email_apply_blocks_when_main_email_already_sent_and_returns_pivot_guidance() -> None:
    platform_db = FakePlatformDB(
        funding_requests={
            41: {
                "email_subject": "Old subject",
                "email_content": "Old body",
            }
        },
        funding_emails={
            101: {
                "funding_request_id": 41,
                "main_email_subject": "Old subject",
                "main_email_body": "Old body",
                "main_sent": 1,
                "main_sent_at": "2026-02-11T10:00:00+00:00",
            }
        },
    )
    propose = FundingEmailDraftUpdateProposeOperator()
    propose._db = platform_db
    apply = FundingEmailDraftUpdateApplyOperator()
    apply._db = platform_db

    proposal_result = await propose.run(
        build_operator_call(
            {
                "funding_request_id": 41,
                "email_id": 101,
                "draft": {
                    "outcome_type": "Email.Draft",
                    "payload": {"subject": "New subject", "body": "New body"},
                },
            }
        )
    )
    assert proposal_result.status == "succeeded"
    proposal = (proposal_result.result or {}).get("outcome")
    assert isinstance(proposal, dict)

    apply_result = await apply.run(
        build_operator_call(
            {
                "funding_request_id": 41,
                "email_id": 101,
                "proposal": proposal,
                "strict_optimistic_lock": True,
            }
        )
    )
    assert apply_result.status == "failed"
    assert apply_result.error is not None
    assert apply_result.error.code == "email_already_sent"
    details = apply_result.error.details or {}
    assert details == {
        "pivot_intent": "Funding.Outreach.Email.Generate",
        "goal": "follow_up",
    }
