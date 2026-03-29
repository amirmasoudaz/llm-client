"""
Cache-backed metadata and summary stores.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..cache_keys import metadata_cache_key, summary_cache_key
from ..memory import SummaryRecord
from .core import CacheCore, CacheLookupResult, CacheWriteResult
from .policy import CachePolicy


@dataclass(frozen=True)
class MetadataCacheRecord:
    kind: str
    identifier: str
    value: dict[str, Any]
    scope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MetadataCacheStore:
    """Generic metadata cache built on top of CacheCore."""

    def __init__(
        self,
        cache: CacheCore,
        *,
        policy: CachePolicy | None = None,
        default_collection: str | None = None,
    ) -> None:
        self.cache = cache
        self.policy = policy or CachePolicy.metadata(collection=default_collection)
        self.default_collection = default_collection

    async def get(
        self,
        kind: str,
        identifier: str,
        *,
        scope: str | None = None,
        payload: dict[str, Any] | None = None,
        policy: CachePolicy | None = None,
    ) -> MetadataCacheRecord | None:
        resolved_policy = policy or self.policy
        key = metadata_cache_key(kind, identifier, scope=scope, payload=payload)
        result = await self.cache.lookup(
            key,
            rewrite_cache=resolved_policy.rewrite_cache,
            regen_cache=resolved_policy.regen_cache,
            only_ok=resolved_policy.only_ok,
            collection=resolved_policy.collection or self.default_collection,
        )
        return self._record_from_lookup(result, kind=kind, identifier=identifier)

    async def put(
        self,
        kind: str,
        identifier: str,
        value: dict[str, Any],
        *,
        scope: str | None = None,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        policy: CachePolicy | None = None,
    ) -> CacheWriteResult:
        resolved_policy = policy or self.policy
        key = metadata_cache_key(kind, identifier, scope=scope, payload=payload)
        return await self.cache.store(
            key,
            rewrite_cache=resolved_policy.rewrite_cache,
            regen_cache=resolved_policy.regen_cache,
            response={
                "error": "OK",
                "status": 200,
                "kind": kind,
                "identifier": identifier,
                "scope": scope,
                "value": dict(value),
                "metadata": dict(metadata or {}),
            },
            model_name="metadata",
            log_errors=resolved_policy.cache_errors,
            collection=resolved_policy.collection or self.default_collection,
        )

    @staticmethod
    def _record_from_lookup(
        lookup: CacheLookupResult,
        *,
        kind: str,
        identifier: str,
    ) -> MetadataCacheRecord | None:
        if not lookup.hit or not lookup.response:
            return None
        response = lookup.response
        return MetadataCacheRecord(
            kind=str(response.get("kind") or kind),
            identifier=str(response.get("identifier") or identifier),
            scope=response.get("scope"),
            value=dict(response.get("value") or {}),
            metadata=dict(response.get("metadata") or {}),
        )


class SummaryCacheStore:
    """Safe cache-backed summary store keyed by scope/model/strategy."""

    def __init__(
        self,
        cache: CacheCore,
        *,
        model: str | None = None,
        strategy: str | None = None,
        scope: str | None = None,
        policy: CachePolicy | None = None,
        default_collection: str | None = None,
    ) -> None:
        self.cache = cache
        self.model = model
        self.strategy = strategy
        self.scope = scope
        self.policy = policy or CachePolicy.summaries(collection=default_collection)
        self.default_collection = default_collection

    async def get(
        self,
        session_id: str,
        *,
        payload: dict[str, Any] | None = None,
        policy: CachePolicy | None = None,
    ) -> SummaryRecord | None:
        resolved_policy = policy or self.policy
        key = summary_cache_key(
            session_id=session_id,
            model=self.model,
            strategy=self.strategy,
            scope=self.scope,
            payload=payload,
        )
        result = await self.cache.lookup(
            key,
            rewrite_cache=resolved_policy.rewrite_cache,
            regen_cache=resolved_policy.regen_cache,
            only_ok=resolved_policy.only_ok,
            collection=resolved_policy.collection or self.default_collection,
        )
        if not result.hit or not result.response:
            return None
        response = result.response
        return SummaryRecord(
            scope=str(response.get("scope") or session_id),
            summary=str(response.get("summary") or ""),
            updated_at=float(response.get("updated_at") or time.time()),
            metadata=dict(response.get("metadata") or {}),
        )

    async def put(
        self,
        session_id: str,
        summary: str,
        *,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        policy: CachePolicy | None = None,
    ) -> CacheWriteResult:
        resolved_policy = policy or self.policy
        key = summary_cache_key(
            session_id=session_id,
            model=self.model,
            strategy=self.strategy,
            scope=self.scope,
            payload=payload,
        )
        return await self.cache.store(
            key,
            rewrite_cache=resolved_policy.rewrite_cache,
            regen_cache=resolved_policy.regen_cache,
            response={
                "error": "OK",
                "status": 200,
                "scope": session_id,
                "summary": summary,
                "updated_at": time.time(),
                "metadata": dict(metadata or {}),
                "model": self.model,
                "strategy": self.strategy,
            },
            model_name=self.model or "summary",
            log_errors=resolved_policy.cache_errors,
            collection=resolved_policy.collection or self.default_collection,
        )


__all__ = ["MetadataCacheRecord", "MetadataCacheStore", "SummaryCacheStore"]
