from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import ContractRegistry
from .base import Operator


@dataclass(frozen=True)
class OperatorRef:
    name: str
    version: str


class OperatorRegistry:
    def __init__(self, contracts: ContractRegistry, allowlist: set[tuple[str, str]] | None = None) -> None:
        self._contracts = contracts
        self._implementations: dict[tuple[str, str], Operator] = {}
        self._allowlist = allowlist

    def register(self, operator: Operator) -> None:
        key = (operator.name, operator.version)
        if self._allowlist is not None and key not in self._allowlist:
            raise ValueError(f"operator {operator.name}@{operator.version} not in allowlist")
        self._implementations[key] = operator

    def get(self, name: str, version: str) -> Operator:
        key = (name, version)
        if self._allowlist is not None and key not in self._allowlist:
            raise KeyError(f"operator {name}@{version} not allowed")
        op = self._implementations.get(key)
        if op is None:
            raise KeyError(f"operator implementation not registered: {name}@{version}")
        return op

    def get_manifest(self, name: str, version: str) -> dict[str, Any]:
        return self._contracts.get_operator_manifest(name, version)
