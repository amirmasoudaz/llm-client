"""
Qdrant cache backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import aiohttp
from blake3 import blake3

from .base import CacheBackendName

logger = logging.getLogger(__name__)


def _u64_hash(s: str) -> int:
    return int.from_bytes(blake3(s.encode("utf-8")).digest()[:8], "big", signed=False)


class QdrantCache:
    name: CacheBackendName = "qdrant"

    def __init__(
        self,
        *,
        default_collection: str,
        client_type: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("QDRANT_URL") or "http://localhost:6333").rstrip("/")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY") or None
        self.default_collection = default_collection
        self.client_type = client_type
        self._ensured_collections: set[str] = set()
        self._ensure_lock = asyncio.Lock()

    def _get_collection(self, collection: str | None) -> str:
        return collection or self.default_collection

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["api-key"] = self.api_key
        return h

    async def ensure_ready(self) -> None:
        await self._ensure_collection(self.default_collection)

    async def close(self) -> None:
        return

    async def warm(self) -> None:
        return

    async def _ensure_collection(self, collection: str) -> None:
        async with self._ensure_lock:
            if collection in self._ensured_collections:
                return
            async with aiohttp.ClientSession() as s:
                url = f"{self.base_url}/collections/{collection}"
                async with s.get(url, headers=self._headers()) as r:
                    if r.status == 200:
                        self._ensured_collections.add(collection)
                        return
                body = {"vectors": {"size": 1, "distance": "Dot"}}
                async with s.put(url, headers=self._headers(), data=json.dumps(body)) as r:
                    if r.status in (200, 201, 409):
                        self._ensured_collections.add(collection)
                        return
                    txt = await r.text()
                    raise RuntimeError(f"Failed to create Qdrant collection: {r.status} {txt}")

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        coll = self._get_collection(collection)
        await self._ensure_collection(coll)
        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{coll}/points/scroll"
            body = {
                "filter": {
                    "must": [
                        {"key": "identifier", "match": {"value": effective_key}},
                        {"key": "client_type", "match": {"value": self.client_type}},
                    ]
                },
                "limit": 1,
                "with_payload": False,
            }
            async with s.post(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status != 200:
                    return False
                data = await r.json()
                return bool(data.get("result", {}).get("points"))

    async def resolve_key(
        self,
        identifier: str,
        rewrite_cache: bool,
        regen_cache: bool,
        collection: str | None = None,
    ) -> tuple[str, bool]:
        if self.client_type == "completions" and rewrite_cache and not regen_cache:
            for i in range(0, 1000):
                eff = f"{identifier}_{i}"
                if not await self.exists(eff, collection):
                    return eff, False
            return f"{identifier}_{int(time.time())}", False
        return identifier, (not regen_cache)

    async def read(self, effective_key: str, collection: str | None = None) -> dict | None:
        coll = self._get_collection(collection)
        await self._ensure_collection(coll)
        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{coll}/points/scroll"
            body = {
                "filter": {
                    "must": [
                        {"key": "identifier", "match": {"value": effective_key}},
                        {"key": "client_type", "match": {"value": self.client_type}},
                    ]
                },
                "limit": 1,
                "with_payload": True,
            }
            async with s.post(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                pts = data.get("result", {}).get("points", [])
                if not pts:
                    return None
                payload = pts[0].get("payload", {})
                return payload.get("cache")

    async def write(self, effective_key: str, response: dict, model_name: str, collection: str | None = None) -> None:
        coll = self._get_collection(collection)
        await self._ensure_collection(coll)
        payload = {
            "identifier": effective_key,
            "client_type": self.client_type,
            "model": model_name,
            "error": response.get("error"),
            "status": response.get("status"),
            "cache": response,
            "created_at": int(time.time()),
        }
        point = {
            "id": _u64_hash(effective_key),
            "vector": [0.0],
            "payload": payload,
        }
        async with aiohttp.ClientSession() as s:
            url = f"{self.base_url}/collections/{coll}/points?wait=true"
            body = {"points": [point]}
            async with s.put(url, headers=self._headers(), data=json.dumps(body)) as r:
                if r.status not in (200, 202):
                    txt = await r.text()
                    logger.warning("Qdrant upsert failed: %s %s", r.status, txt)
