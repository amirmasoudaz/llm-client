"""
Canonical cache-key strategy for llm-client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from blake3 import blake3

from .serialization import stable_json_dumps


CACHE_KEY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CacheKeyDescriptor:
    namespace: str
    version: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_schema_version": self.version,
            "namespace": self.namespace,
            "payload": dict(self.payload),
        }

    def to_key(self) -> str:
        return blake3(stable_json_dumps(self.to_dict()).encode("utf-8")).hexdigest()


def build_cache_key(namespace: str, payload: dict[str, Any], *, version: int = CACHE_KEY_SCHEMA_VERSION) -> str:
    return CacheKeyDescriptor(namespace=namespace, version=version, payload=dict(payload)).to_key()


def request_cache_key(
    spec_or_payload: Any,
    *,
    provider: str | None = None,
    tenant_id: str | None = None,
    cache_scope: str | None = None,
    version: int = CACHE_KEY_SCHEMA_VERSION,
) -> str:
    if hasattr(spec_or_payload, "to_dict"):
        payload = spec_or_payload.to_dict()
    else:
        payload = dict(spec_or_payload)

    descriptor_payload = {
        "request": payload,
        "provider": provider,
        "tenant_id": tenant_id,
        "cache_scope": cache_scope,
    }
    return build_cache_key("request.completion", descriptor_payload, version=version)


def embedding_cache_key(
    *,
    model: str,
    inputs: list[str],
    provider: str | None = None,
    dimensions: int | None = None,
    tenant_id: str | None = None,
    cache_scope: str | None = None,
    extra: dict[str, Any] | None = None,
    version: int = CACHE_KEY_SCHEMA_VERSION,
) -> str:
    payload = {
        "model": model,
        "inputs": list(inputs),
        "provider": provider,
        "dimensions": dimensions,
        "tenant_id": tenant_id,
        "cache_scope": cache_scope,
        "extra": dict(extra or {}),
    }
    return build_cache_key("request.embedding", payload, version=version)


def metadata_cache_key(
    kind: str,
    identifier: str,
    *,
    scope: str | None = None,
    payload: dict[str, Any] | None = None,
    version: int = CACHE_KEY_SCHEMA_VERSION,
) -> str:
    return build_cache_key(
        f"metadata.{kind}",
        {
            "identifier": identifier,
            "scope": scope,
            "payload": dict(payload or {}),
        },
        version=version,
    )


def summary_cache_key(
    *,
    session_id: str,
    model: str | None = None,
    strategy: str | None = None,
    scope: str | None = None,
    payload: dict[str, Any] | None = None,
    version: int = CACHE_KEY_SCHEMA_VERSION,
) -> str:
    return build_cache_key(
        "summary.context",
        {
            "session_id": session_id,
            "model": model,
            "strategy": strategy,
            "scope": scope,
            "payload": dict(payload or {}),
        },
        version=version,
    )


__all__ = [
    "CACHE_KEY_SCHEMA_VERSION",
    "CacheKeyDescriptor",
    "build_cache_key",
    "request_cache_key",
    "embedding_cache_key",
    "metadata_cache_key",
    "summary_cache_key",
]
