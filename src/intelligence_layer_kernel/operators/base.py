from __future__ import annotations

from abc import ABC, abstractmethod

from .types import OperatorCall, OperatorResult


class Operator(ABC):
    name: str
    version: str

    @abstractmethod
    async def run(self, call: OperatorCall) -> OperatorResult:
        raise NotImplementedError
