"""
Provider routing utilities.
"""
from __future__ import annotations

from typing import Iterable, List, Protocol

from .providers.base import Provider
from .spec import RequestSpec


class ProviderRouter(Protocol):
    def select(self, spec: RequestSpec) -> Iterable[Provider]:
        ...


class StaticRouter:
    """
    Simple ordered fallback router.
    """

    def __init__(self, providers: List[Provider]) -> None:
        if not providers:
            raise ValueError("StaticRouter requires at least one provider.")
        self._providers = providers

    def select(self, spec: RequestSpec) -> Iterable[Provider]:
        return list(self._providers)


__all__ = ["ProviderRouter", "StaticRouter"]
