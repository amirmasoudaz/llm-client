from __future__ import annotations

from llm_client.providers.types import CompletionResult
from llm_client.structured import (
    StructuredCompletionLoopResult,
    finalize_structured_completion_loop,
)
from llm_client.tools.runtime import StructuredToolLoopError


def test_finalize_structured_completion_loop_normalizes_successful_envelope() -> None:
    loop_result = StructuredCompletionLoopResult(
        completion=CompletionResult(content='{"status":"succeeded","result":{"value": 1}}', model="gpt-5-mini", status=200),
        completion_messages=[],
        content={"status": "succeeded", "result": {"value": 1}},
    )

    outcome = finalize_structured_completion_loop(
        loop_result,
        default_nondeterminism={"is_nondeterministic": True, "reasons": ["test"], "stability": "medium"},
    )

    assert outcome.ok is True
    assert outcome.envelope is not None
    assert outcome.envelope.result == {"value": 1}
    assert outcome.failure is None


def test_finalize_structured_completion_loop_maps_provider_failure() -> None:
    loop_result = StructuredCompletionLoopResult(
        completion=CompletionResult(content="", model="gpt-5-mini", status=503, error="upstream unavailable"),
        completion_messages=[],
    )

    outcome = finalize_structured_completion_loop(loop_result)

    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "provider_error"
    assert outcome.failure.retryable is True
    assert outcome.failure.details is not None
    assert outcome.failure.details["normalized_failure"]["category"] == "structured_output"


def test_finalize_structured_completion_loop_maps_tool_loop_failure() -> None:
    loop_result = StructuredCompletionLoopResult(
        completion=CompletionResult(content="", model="gpt-5-mini", status=200),
        completion_messages=[],
        tool_error=StructuredToolLoopError(code="tool_not_allowed", message="blocked", category="policy_denied"),
    )

    outcome = finalize_structured_completion_loop(loop_result)

    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "tool_not_allowed"
    assert outcome.failure.category == "policy_denied"
    assert outcome.failure.details is not None
    assert outcome.failure.details["normalized_failure"]["category"] == "tool_policy"


def test_finalize_structured_completion_loop_maps_schema_invalid_output() -> None:
    loop_result = StructuredCompletionLoopResult(
        completion=CompletionResult(content="not json", model="gpt-5-mini", status=200),
        completion_messages=[],
        raw_content="not json",
        validation_errors=["invalid json"],
        repair_attempts=2,
    )

    outcome = finalize_structured_completion_loop(
        loop_result,
        default_error_code="schema_invalid",
        default_error_message="schema invalid",
        default_error_category="operator_bug",
    )

    assert outcome.ok is False
    assert outcome.failure is not None
    assert outcome.failure.code == "schema_invalid"
    assert outcome.failure.details is not None
    assert outcome.failure.details["repair_attempts"] == 2
    assert outcome.failure.details["normalized_failure"]["category"] == "structured_output"
