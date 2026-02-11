from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from blake3 import blake3

from .types import PolicyContext, PolicyDecision


@dataclass(frozen=True)
class PolicyRule:
    decision: str
    reason_code: str
    stage: str | None = None
    operator_name: str | None = None
    operator_version: str | None = None
    any_effects: list[str] = field(default_factory=list)
    all_effects: list[str] = field(default_factory=list)
    any_policy_tags: list[str] = field(default_factory=list)
    all_policy_tags: list[str] = field(default_factory=list)
    any_data_classes: list[str] = field(default_factory=list)
    all_data_classes: list[str] = field(default_factory=list)
    required_scopes: list[str] = field(default_factory=list)
    principal_types: list[str] = field(default_factory=list)
    min_trust_level: int | None = None
    max_trust_level: int | None = None
    reason: str | None = None
    requirements: dict[str, Any] = field(default_factory=dict)
    limits: dict[str, Any] = field(default_factory=dict)
    redactions: list[dict[str, Any]] = field(default_factory=list)
    transform: dict[str, Any] | None = None
    priority: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PolicyRule":
        return cls(
            decision=str(value.get("decision") or "ALLOW"),
            reason_code=str(value.get("reason_code") or "policy_rule"),
            stage=_opt_text(value.get("stage")),
            operator_name=_opt_text(value.get("operator_name")),
            operator_version=_opt_text(value.get("operator_version")),
            any_effects=_list_of_text(value.get("any_effects")),
            all_effects=_list_of_text(value.get("all_effects")),
            any_policy_tags=_list_of_text(value.get("any_policy_tags")),
            all_policy_tags=_list_of_text(value.get("all_policy_tags")),
            any_data_classes=_list_of_text(value.get("any_data_classes")),
            all_data_classes=_list_of_text(value.get("all_data_classes")),
            required_scopes=_list_of_text(value.get("required_scopes")),
            principal_types=_list_of_text(value.get("principal_types")),
            min_trust_level=_opt_int(value.get("min_trust_level")),
            max_trust_level=_opt_int(value.get("max_trust_level")),
            reason=_opt_text(value.get("reason")),
            requirements=_as_object(value.get("requirements")),
            limits=_as_object(value.get("limits")),
            redactions=_as_redactions(value.get("redactions")),
            transform=value.get("transform") if isinstance(value.get("transform"), dict) else None,
            priority=int(value.get("priority") or 0),
        )


class PolicyEngine:
    def __init__(
        self,
        *,
        name: str = "il_policy",
        version: str = "0.2",
        rules: list[PolicyRule | dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        compiled: list[PolicyRule] = []
        for item in rules or []:
            if isinstance(item, PolicyRule):
                compiled.append(item)
            elif isinstance(item, dict):
                compiled.append(PolicyRule.from_dict(item))
            else:
                raise TypeError("policy rules must be PolicyRule or dict")
        self._rules = compiled

    def evaluate(self, context: PolicyContext) -> PolicyDecision:
        payload = json.dumps(context.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        inputs_hash = blake3(payload).digest()
        matched = [rule for rule in self._rules if _rule_matches(rule, context)]
        if matched:
            selected = _select_rule(matched)
            return PolicyDecision(
                stage=context.stage,
                decision=selected.decision.upper(),
                reason_code=selected.reason_code,
                reason=selected.reason,
                requirements=dict(selected.requirements),
                limits=dict(selected.limits),
                redactions=list(selected.redactions),
                transform=selected.transform,
                inputs_hash=inputs_hash,
                policy_engine_name=self.name,
                policy_engine_version=self.version,
            )
        return PolicyDecision(
            stage=context.stage,
            decision="ALLOW",
            reason_code="default_allow",
            reason=None,
            requirements={},
            limits={},
            redactions=[],
            transform=None,
            inputs_hash=inputs_hash,
            policy_engine_name=self.name,
            policy_engine_version=self.version,
        )


def _rule_matches(rule: PolicyRule, context: PolicyContext) -> bool:
    if rule.stage and rule.stage != context.stage:
        return False
    if rule.operator_name and rule.operator_name != context.operator_name:
        return False
    if rule.operator_version and rule.operator_version != context.operator_version:
        return False

    effects = set(context.effects or [])
    if rule.any_effects and effects.isdisjoint(set(rule.any_effects)):
        return False
    if rule.all_effects and not set(rule.all_effects).issubset(effects):
        return False

    tags = set(context.policy_tags or [])
    if rule.any_policy_tags and tags.isdisjoint(set(rule.any_policy_tags)):
        return False
    if rule.all_policy_tags and not set(rule.all_policy_tags).issubset(tags):
        return False

    data_classes = set(context.data_classes or [])
    if rule.any_data_classes and data_classes.isdisjoint(set(rule.any_data_classes)):
        return False
    if rule.all_data_classes and not set(rule.all_data_classes).issubset(data_classes):
        return False

    auth_context = context.auth_context or {}
    scopes = set(_list_of_text(auth_context.get("scopes")))
    if rule.required_scopes and not set(rule.required_scopes).issubset(scopes):
        return False

    principal = auth_context.get("principal") if isinstance(auth_context.get("principal"), dict) else {}
    principal_type = _opt_text(principal.get("type"))
    if rule.principal_types and principal_type not in set(rule.principal_types):
        return False

    trust_level = _extract_trust_level(auth_context)
    if rule.min_trust_level is not None and trust_level < rule.min_trust_level:
        return False
    if rule.max_trust_level is not None and trust_level > rule.max_trust_level:
        return False
    return True


def _select_rule(rules: list[PolicyRule]) -> PolicyRule:
    precedence = {
        "DENY": 500,
        "REQUIRE_APPROVAL": 400,
        "ALLOW_WITH_REDACTION": 300,
        "TRANSFORM": 200,
        "ALLOW": 100,
    }
    return max(
        rules,
        key=lambda rule: (
            precedence.get(rule.decision.upper(), 0),
            int(rule.priority),
        ),
    )


def _list_of_text(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _opt_text(item)
        if text is None:
            continue
        if text not in result:
            result.append(text)
    return result


def _opt_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text


def _opt_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_redactions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _extract_trust_level(auth_context: dict[str, Any]) -> int:
    value = auth_context.get("trust_level")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0
