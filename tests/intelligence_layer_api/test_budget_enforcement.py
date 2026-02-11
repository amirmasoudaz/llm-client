from __future__ import annotations

import uuid

import pytest

from tests.intelligence_layer_api._workflow_testkit import actor, build_kernel


class _UsageRecorderBudgetDeny:
    def __init__(self) -> None:
        self.record_called = False

    async def quote_usage(self, *, provider: str, model: str, tokens_in: int, tokens_out: int):
        _ = (provider, model, tokens_in, tokens_out)
        return {"credits_charged": 5}

    async def record_operator_usage(self, **kwargs):
        _ = kwargs
        self.record_called = True
        return {"ok": True, "total_credits_charged": 0}


@pytest.mark.asyncio
async def test_budget_enforced_before_llm_operator_execution() -> None:
    usage = _UsageRecorderBudgetDeny()
    kernel, operator_executor, event_writer = build_kernel(usage_recorder=usage)

    result = await kernel.start_intent(
        intent_type="Funding.Outreach.Email.Review",
        inputs={},
        thread_id=101,
        scope_type="funding_request",
        scope_id="88",
        actor=actor(reserved_credits=1),
        source="chat",
        workflow_id=uuid.uuid4(),
    )

    assert result.status == "failed"
    assert operator_executor.calls == []
    assert usage.record_called is False
    event_types = [event.event_type for event in event_writer.events]
    assert "final_error" in event_types
    assert "model_token" not in event_types
