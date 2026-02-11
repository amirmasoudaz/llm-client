from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import ContractRegistry
from .base import Operator


@dataclass(frozen=True)
class OperatorRef:
    name: str
    version: str


class OperatorAccessDenied(PermissionError):
    def __init__(
        self,
        *,
        reason_code: str,
        reason: str,
        requirements: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason_code = reason_code
        self.reason = reason
        self.requirements = requirements or {}


class OperatorRegistry:
    def __init__(
        self,
        contracts: ContractRegistry,
        allowlist: set[tuple[str, str]] | None = None,
        *,
        denylist: set[tuple[str, str]] | None = None,
        capability_allowlist: set[str] | None = None,
        capability_denylist: set[str] | None = None,
        policy_tag_allowlist: set[str] | None = None,
        policy_tag_denylist: set[str] | None = None,
        effect_allowlist: set[str] | None = None,
        effect_denylist: set[str] | None = None,
    ) -> None:
        self._contracts = contracts
        self._implementations: dict[tuple[str, str], Operator] = {}
        self._allowlist = allowlist
        self._denylist = denylist or set()
        self._capability_allowlist = capability_allowlist
        self._capability_denylist = capability_denylist or set()
        self._policy_tag_allowlist = policy_tag_allowlist
        self._policy_tag_denylist = policy_tag_denylist or set()
        self._effect_allowlist = effect_allowlist
        self._effect_denylist = effect_denylist or set()

    def register(self, operator: Operator) -> None:
        key = (operator.name, operator.version)
        if self._allowlist is not None and key not in self._allowlist:
            raise ValueError(f"operator {operator.name}@{operator.version} not in allowlist")
        if key in self._denylist:
            raise ValueError(f"operator {operator.name}@{operator.version} is denied by registry policy")
        manifest = self.get_manifest(operator.name, operator.version)
        self._enforce_manifest_constraints(operator.name, operator.version, manifest)
        self._implementations[key] = operator

    def get(self, name: str, version: str) -> Operator:
        key = (name, version)
        if self._allowlist is not None and key not in self._allowlist:
            raise KeyError(f"operator {name}@{version} not allowed")
        if key in self._denylist:
            raise KeyError(f"operator {name}@{version} denied")
        op = self._implementations.get(key)
        if op is None:
            raise KeyError(f"operator implementation not registered: {name}@{version}")
        return op

    def get_manifest(self, name: str, version: str) -> dict[str, Any]:
        key = (name, version)
        if self._allowlist is not None and key not in self._allowlist:
            raise KeyError(f"operator {name}@{version} not allowed")
        if key in self._denylist:
            raise KeyError(f"operator {name}@{version} denied")
        manifest = self._contracts.get_operator_manifest(name, version)
        self._enforce_manifest_constraints(name, version, manifest)
        return manifest

    def enforce_invocation_policy(
        self,
        *,
        name: str,
        version: str,
        auth_context: dict[str, Any],
        manifest: dict[str, Any] | None = None,
    ) -> None:
        resolved_manifest = manifest or self.get_manifest(name, version)
        required_scopes = _extract_required_scopes(resolved_manifest)
        auth_scopes = set(_coerce_text_list(auth_context.get("scopes")))
        missing_scopes = [scope for scope in required_scopes if scope not in auth_scopes]
        if missing_scopes:
            raise OperatorAccessDenied(
                reason_code="missing_required_scope",
                reason=f"Operator {name}@{version} requires scopes: {', '.join(missing_scopes)}",
                requirements={"required_scopes": missing_scopes},
            )

        principal = auth_context.get("principal") if isinstance(auth_context.get("principal"), dict) else {}
        principal_type = str(principal.get("type") or "").strip()
        denied_types = set(_coerce_text_list(resolved_manifest.get("deny_principal_types")))
        allowed_types = set(_coerce_text_list(resolved_manifest.get("allow_principal_types")))
        auth_block = resolved_manifest.get("auth")
        if isinstance(auth_block, dict):
            denied_types |= set(_coerce_text_list(auth_block.get("deny_principal_types")))
            auth_allowed = set(_coerce_text_list(auth_block.get("allow_principal_types")))
            if auth_allowed:
                allowed_types |= auth_allowed

        if denied_types and principal_type in denied_types:
            raise OperatorAccessDenied(
                reason_code="principal_type_denied",
                reason=f"Principal type '{principal_type}' is denied for operator {name}@{version}",
            )
        if allowed_types and principal_type not in allowed_types:
            raise OperatorAccessDenied(
                reason_code="principal_type_not_allowed",
                reason=f"Principal type '{principal_type}' is not allowed for operator {name}@{version}",
                requirements={"allowed_principal_types": sorted(allowed_types)},
            )

    def _enforce_manifest_constraints(self, name: str, version: str, manifest: dict[str, Any]) -> None:
        capabilities = set(_extract_capabilities(manifest))
        policy_tags = set(_coerce_text_list(manifest.get("policy_tags")))
        effects = set(_coerce_text_list(manifest.get("effects")))

        if self._capability_denylist and capabilities.intersection(self._capability_denylist):
            denied = sorted(capabilities.intersection(self._capability_denylist))
            raise KeyError(f"operator {name}@{version} denied by capability policy: {', '.join(denied)}")
        if self._capability_allowlist is not None and capabilities:
            not_allowed = sorted(capabilities - set(self._capability_allowlist))
            if not_allowed:
                raise KeyError(f"operator {name}@{version} uses non-allowlisted capabilities: {', '.join(not_allowed)}")

        if self._policy_tag_denylist and policy_tags.intersection(self._policy_tag_denylist):
            denied_tags = sorted(policy_tags.intersection(self._policy_tag_denylist))
            raise KeyError(f"operator {name}@{version} denied by policy tags: {', '.join(denied_tags)}")
        if self._policy_tag_allowlist is not None and policy_tags:
            disallowed_tags = sorted(policy_tags - set(self._policy_tag_allowlist))
            if disallowed_tags:
                raise KeyError(f"operator {name}@{version} uses non-allowlisted policy tags: {', '.join(disallowed_tags)}")

        if self._effect_denylist and effects.intersection(self._effect_denylist):
            denied_effects = sorted(effects.intersection(self._effect_denylist))
            raise KeyError(f"operator {name}@{version} denied by effects: {', '.join(denied_effects)}")
        if self._effect_allowlist is not None and effects:
            disallowed_effects = sorted(effects - set(self._effect_allowlist))
            if disallowed_effects:
                raise KeyError(f"operator {name}@{version} uses non-allowlisted effects: {', '.join(disallowed_effects)}")


def _coerce_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in out:
            continue
        out.append(text)
    return out


def _extract_capabilities(manifest: dict[str, Any]) -> list[str]:
    capabilities = _coerce_text_list(manifest.get("capabilities"))
    requires = manifest.get("requires")
    if isinstance(requires, dict):
        capabilities.extend(_coerce_text_list(requires.get("capabilities")))
    return list(dict.fromkeys(capabilities))


def _extract_required_scopes(manifest: dict[str, Any]) -> list[str]:
    scopes = _coerce_text_list(manifest.get("required_scopes"))
    requires = manifest.get("requires")
    if isinstance(requires, dict):
        scopes.extend(_coerce_text_list(requires.get("scopes")))
    auth = manifest.get("auth")
    if isinstance(auth, dict):
        scopes.extend(_coerce_text_list(auth.get("required_scopes")))
    return list(dict.fromkeys(scopes))
