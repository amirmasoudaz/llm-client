from __future__ import annotations

from datetime import timedelta

import pytest

from intelligence_layer_kernel.operators.implementations.funding_request_fields_update_apply import (
    FundingRequestFieldsUpdateApplyOperator,
)
from intelligence_layer_kernel.operators.implementations.funding_request_fields_update_propose import (
    FundingRequestFieldsUpdateProposeOperator,
)
from tests.intelligence_layer_kernel._phase_fg_testkit import FakePlatformDB, build_operator_call


@pytest.mark.asyncio
async def test_apply_returns_stale_conflict_when_request_updated_at_changes() -> None:
    platform_db = FakePlatformDB(
        funding_requests={
            41: {
                "research_interest": "robotics",
                "paper_title": "Original title",
                "journal": "Journal A",
                "year": 2024,
                "research_connection": "baseline",
            }
        }
    )
    propose = FundingRequestFieldsUpdateProposeOperator()
    propose._db = platform_db
    apply = FundingRequestFieldsUpdateApplyOperator()
    apply._db = platform_db

    proposal_result = await propose.run(
        build_operator_call(
            {
                "funding_request_id": 41,
                "fields": {"research_interest": "machine learning"},
                "human_summary": "update interest",
            },
            idempotency_key="proposal-key",
        )
    )
    assert proposal_result.status == "succeeded"
    proposal = (proposal_result.result or {}).get("outcome")
    assert isinstance(proposal, dict)

    platform_db.funding_requests[41]["updated_at"] = (
        platform_db.funding_requests[41]["updated_at"] + timedelta(minutes=5)
    )

    apply_result = await apply.run(
        build_operator_call(
            {
                "funding_request_id": 41,
                "proposal": proposal,
                "strict_optimistic_lock": True,
            },
            idempotency_key="apply-key",
        )
    )
    assert apply_result.status == "failed"
    assert apply_result.error is not None
    assert apply_result.error.code == "stale_update_conflict"
    details = apply_result.error.details or {}
    assert isinstance(details.get("expected_updated_at"), str)
    assert isinstance(details.get("current_updated_at"), str)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "expected_message"),
    [
        ({"unsupported_field": "value"}, "unsupported fields"),
        ({"year": "2024"}, "year must be an integer"),
        ({"research_interest": "x" * 1001}, "research_interest exceeds max length 1000"),
    ],
)
async def test_propose_rejects_unsupported_or_invalid_field_values(
    fields: dict[str, object],
    expected_message: str,
) -> None:
    platform_db = FakePlatformDB(
        funding_requests={
            41: {
                "research_interest": "robotics",
                "paper_title": "Original title",
                "journal": "Journal A",
                "year": 2024,
                "research_connection": "baseline",
            }
        }
    )
    propose = FundingRequestFieldsUpdateProposeOperator()
    propose._db = platform_db

    result = await propose.run(
        build_operator_call(
            {
                "funding_request_id": 41,
                "fields": fields,
                "human_summary": "invalid update",
            }
        )
    )
    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "invalid_fields"
    assert expected_message in result.error.message
