from __future__ import annotations

import uuid

import pytest

from tests.intelligence_layer_api._workflow_testkit import actor, build_kernel


class _UsageResult:
    def __init__(self, *, total_credits_charged: int) -> None:
        self.ok = True
        self.reason = None
        self.total_credits_charged = total_credits_charged


class _UsageRecorderAllow:
    def __init__(self) -> None:
        self.record_calls: list[dict] = []

    async def quote_usage(self, *, provider: str, model: str, tokens_in: int, tokens_out: int):
        _ = (provider, model, tokens_in, tokens_out)
        return {"credits_charged": 1}

    async def record_operator_usage(self, **kwargs):
        self.record_calls.append(dict(kwargs))
        return _UsageResult(total_credits_charged=1)


@pytest.mark.asyncio
async def test_llm_backed_workflow_emits_model_token_events() -> None:
    usage = _UsageRecorderAllow()
    kernel, operator_executor, event_writer = build_kernel(usage_recorder=usage)

    result = await kernel.start_intent(
        intent_type="Funding.Outreach.Email.Review",
        inputs={},
        thread_id=101,
        scope_type="funding_request",
        scope_id="88",
        actor=actor(reserved_credits=100),
        source="chat",
        workflow_id=uuid.uuid4(),
    )

    assert result.status == "completed"
    assert len(operator_executor.calls) == 1
    assert len(usage.record_calls) == 1

    model_token_events = [event for event in event_writer.events if event.event_type == "model_token"]
    assert model_token_events
    first_payload = model_token_events[0].payload
    assert isinstance(first_payload.get("token"), str)
    assert first_payload.get("index") == 0
