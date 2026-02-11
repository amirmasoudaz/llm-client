from __future__ import annotations

from intelligence_layer_kernel.policy import PolicyContext, PolicyEngine


def test_policy_engine_denies_when_rule_matches() -> None:
    engine = PolicyEngine(
        rules=[
            {
                "decision": "DENY",
                "reason_code": "external_send_blocked",
                "stage": "action",
                "any_effects": ["external_send"],
                "reason": "external sends are blocked in this environment",
            }
        ]
    )
    ctx = PolicyContext(
        stage="action",
        operator_name="Gmail.SendEmail",
        operator_version="1.0.0",
        effects=["external_send"],
        policy_tags=["email_send"],
        data_classes=[],
        auth_context={"principal": {"type": "student", "id": 1}, "scopes": ["email:send"]},
        trace_context={"workflow_id": "00000000-0000-0000-0000-000000000001"},
        input_payload={},
    )

    decision = engine.evaluate(ctx)

    assert decision.decision == "DENY"
    assert decision.reason_code == "external_send_blocked"
    assert decision.reason == "external sends are blocked in this environment"


def test_policy_engine_returns_require_approval_with_requirements() -> None:
    engine = PolicyEngine(
        rules=[
            {
                "decision": "REQUIRE_APPROVAL",
                "reason_code": "platform_write_requires_gate",
                "stage": "action",
                "any_policy_tags": ["platform_write"],
                "requirements": {"approval": {"gate_type": "human_confirm"}},
            }
        ]
    )
    ctx = PolicyContext(
        stage="action",
        operator_name="FundingRequest.Fields.Update.Apply",
        operator_version="1.0.0",
        effects=["db_write_platform"],
        policy_tags=["platform_write"],
        data_classes=[],
        auth_context={"principal": {"type": "student", "id": 1}, "scopes": ["request:write"]},
        trace_context={},
        input_payload={},
    )

    decision = engine.evaluate(ctx)

    assert decision.decision == "REQUIRE_APPROVAL"
    assert decision.reason_code == "platform_write_requires_gate"
    assert decision.requirements == {"approval": {"gate_type": "human_confirm"}}


def test_policy_engine_precedence_prefers_deny_over_require_approval() -> None:
    engine = PolicyEngine(
        rules=[
            {
                "decision": "REQUIRE_APPROVAL",
                "reason_code": "needs_approval",
                "stage": "action",
                "any_effects": ["external_send"],
            },
            {
                "decision": "DENY",
                "reason_code": "hard_block",
                "stage": "action",
                "any_effects": ["external_send"],
            },
        ]
    )
    ctx = PolicyContext(
        stage="action",
        effects=["external_send"],
        policy_tags=[],
        data_classes=[],
        auth_context={},
        trace_context={},
        input_payload={},
    )

    decision = engine.evaluate(ctx)

    assert decision.decision == "DENY"
    assert decision.reason_code == "hard_block"


def test_policy_engine_default_allow_without_matching_rules() -> None:
    engine = PolicyEngine(rules=[{"decision": "DENY", "reason_code": "x", "stage": "action", "any_effects": ["external_send"]}])
    ctx = PolicyContext(
        stage="action",
        effects=["read_only"],
        policy_tags=[],
        data_classes=[],
        auth_context={},
        trace_context={},
        input_payload={},
    )

    decision = engine.evaluate(ctx)

    assert decision.decision == "ALLOW"
    assert decision.reason_code == "default_allow"
