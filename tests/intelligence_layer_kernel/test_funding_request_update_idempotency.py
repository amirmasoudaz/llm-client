from __future__ import annotations

import pytest

from intelligence_layer_kernel.operators.implementations.funding_request_fields_update_apply import (
    FundingRequestFieldsUpdateApplyOperator,
)
from intelligence_layer_kernel.operators.implementations.funding_request_fields_update_propose import (
    FundingRequestFieldsUpdateProposeOperator,
)
from tests.intelligence_layer_kernel._phase_fg_testkit import FakePlatformDB, build_operator_call


async def _build_proposal(*, operator: FundingRequestFieldsUpdateProposeOperator, request_id: int) -> dict:
    proposal_call = build_operator_call(
        {
            "funding_request_id": request_id,
            "fields": {"research_interest": "machine learning", "paper_title": "Kernel-first draft"},
            "human_summary": "Update request fields",
        },
        idempotency_key="proposal-key",
    )
    proposal_result = await operator.run(proposal_call)
    assert proposal_result.status == "succeeded"
    outcome = (proposal_result.result or {}).get("outcome")
    assert isinstance(outcome, dict)
    return outcome


@pytest.mark.asyncio
async def test_repeated_apply_with_same_idempotency_key_does_not_double_apply() -> None:
    platform_db = FakePlatformDB(
        funding_requests={
            41: {
                "research_interest": "robotics",
                "paper_title": "Original title",
                "journal": "Original Journal",
                "year": 2023,
                "research_connection": "initial",
            }
        }
    )
    propose = FundingRequestFieldsUpdateProposeOperator()
    propose._db = platform_db
    apply = FundingRequestFieldsUpdateApplyOperator()
    apply._db = platform_db

    proposal = await _build_proposal(operator=propose, request_id=41)
    apply_payload = {
        "funding_request_id": 41,
        "proposal": proposal,
        "strict_optimistic_lock": False,
    }
    first_apply = await apply.run(build_operator_call(apply_payload, idempotency_key="apply-key"))
    assert first_apply.status == "succeeded"
    assert (first_apply.result or {}).get("rows_affected") == 1

    second_apply = await apply.run(build_operator_call(apply_payload, idempotency_key="apply-key"))
    assert second_apply.status == "succeeded"
    assert (second_apply.result or {}).get("rows_affected") == 0

    row = platform_db.funding_requests[41]
    assert row["research_interest"] == "machine learning"
    assert row["paper_title"] == "Kernel-first draft"
