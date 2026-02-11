from __future__ import annotations

from intelligence_layer_kernel.runtime.kernel import WorkflowKernel


def _kernel_stub(
    *,
    apply_steps_enabled: bool = True,
    apply_shadow_mode: bool = False,
    replies_enabled: bool = True,
    replies_canary_percent: int = 100,
    replies_canary_principals: tuple[int, ...] = (),
) -> WorkflowKernel:
    kernel = object.__new__(WorkflowKernel)
    object.__setattr__(kernel, "apply_steps_enabled", apply_steps_enabled)
    object.__setattr__(kernel, "apply_shadow_mode", apply_shadow_mode)
    object.__setattr__(kernel, "replies_enabled", replies_enabled)
    object.__setattr__(kernel, "replies_canary_percent", replies_canary_percent)
    object.__setattr__(kernel, "replies_canary_principals", replies_canary_principals)
    return kernel


def test_apply_step_mode_disabled_and_shadow_paths() -> None:
    apply_gate_step = {
        "kind": "human_gate",
        "effects": ["db_write_platform"],
        "gate": {"gate_type": "apply_platform_patch"},
    }
    apply_operator_step = {
        "kind": "operator",
        "effects": ["db_write_platform"],
        "operator_name": "FundingEmail.Draft.Update.Apply",
    }
    non_apply_step = {
        "kind": "operator",
        "effects": ["produce_outcome"],
        "operator_name": "Email.ReviewDraft",
    }

    disabled_kernel = _kernel_stub(apply_steps_enabled=False, apply_shadow_mode=False)
    assert disabled_kernel._apply_step_mode(apply_gate_step) == "disabled"
    assert disabled_kernel._apply_step_mode(apply_operator_step) == "disabled"
    assert disabled_kernel._apply_step_mode(non_apply_step) is None

    shadow_kernel = _kernel_stub(apply_steps_enabled=True, apply_shadow_mode=True)
    assert shadow_kernel._apply_step_mode(apply_gate_step) == "shadow"
    assert shadow_kernel._apply_step_mode(apply_operator_step) == "shadow"
    assert shadow_kernel._apply_step_mode(non_apply_step) is None


def test_reply_intent_canary_and_allowlist_controls() -> None:
    actor = {"principal": {"type": "student", "id": 55}}
    reply_intent = "Funding.Outreach.Reply.Interpret"

    disabled = _kernel_stub(replies_enabled=False)
    assert disabled._intent_enabled_for_actor(intent_type=reply_intent, actor=actor) is False

    canary_closed = _kernel_stub(replies_enabled=True, replies_canary_percent=0)
    assert canary_closed._intent_enabled_for_actor(intent_type=reply_intent, actor=actor) is False

    allowlisted = _kernel_stub(
        replies_enabled=True,
        replies_canary_percent=0,
        replies_canary_principals=(55,),
    )
    assert allowlisted._intent_enabled_for_actor(intent_type=reply_intent, actor=actor) is True
