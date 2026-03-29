from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import aiohttp

from cookbook_support import example_env, fail_or_skip


@dataclass(frozen=True)
class RetrieverDocument:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any]


def require_qdrant_url() -> str:
    url = example_env("QDRANT_URL", "http://127.0.0.1:6333")
    if not url:
        fail_or_skip("Set QDRANT_URL to run this example against Qdrant.")
    return url.rstrip("/")


def qdrant_api_key() -> str | None:
    value = example_env("QDRANT_API_KEY")
    return value or None


def qdrant_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = qdrant_api_key()
    if api_key:
        headers["api-key"] = api_key
    return headers


def chunk_text(text: str, *, max_chars: int = 400) -> list[str]:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        if end < len(cleaned):
            split = cleaned.rfind(" ", start, end)
            if split > start:
                end = split
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def excerpt(text: str | None, *, limit: int = 180) -> str | None:
    if text is None:
        return None
    compact = " ".join(str(text).split())
    return compact[:limit]


class QdrantRetriever:
    def __init__(self, collection: str, *, vector_size: int, base_url: str | None = None) -> None:
        self.collection = collection
        self.vector_size = vector_size
        self.base_url = (base_url or require_qdrant_url()).rstrip("/")

    async def recreate_collection(self) -> None:
        headers = qdrant_headers()
        try:
            async with aiohttp.ClientSession() as session:
                delete_url = f"{self.base_url}/collections/{self.collection}"
                async with session.delete(delete_url, headers=headers):
                    pass
                create_url = f"{self.base_url}/collections/{self.collection}"
                body = {
                    "vectors": {
                        "size": self.vector_size,
                        "distance": "Cosine",
                    }
                }
                async with session.put(create_url, headers=headers, data=json.dumps(body)) as response:
                    if response.status not in (200, 201):
                        raise RuntimeError(
                            f"Failed to create Qdrant collection {self.collection}: {response.status} {await response.text()}"
                        )
        except aiohttp.ClientError as exc:
            fail_or_skip(
                f"Could not reach Qdrant at {self.base_url}. Start Qdrant or set QDRANT_URL correctly. "
                f"Connection error: {type(exc).__name__}: {exc}"
            )

    async def upsert(self, points: list[dict[str, Any]]) -> None:
        headers = qdrant_headers()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/collections/{self.collection}/points?wait=true"
                async with session.put(url, headers=headers, data=json.dumps({"points": points})) as response:
                    if response.status not in (200, 202):
                        raise RuntimeError(
                            f"Failed to upsert Qdrant points into {self.collection}: {response.status} {await response.text()}"
                        )
        except aiohttp.ClientError as exc:
            fail_or_skip(
                f"Could not reach Qdrant at {self.base_url}. Start Qdrant or set QDRANT_URL correctly. "
                f"Connection error: {type(exc).__name__}: {exc}"
            )

    async def search(self, query_vector: list[float], *, limit: int = 4) -> list[dict[str, Any]]:
        headers = qdrant_headers()
        try:
            async with aiohttp.ClientSession() as session:
                query_points_url = f"{self.base_url}/collections/{self.collection}/points/query"
                body = {
                    "query": query_vector,
                    "limit": limit,
                    "with_payload": True,
                }
                async with session.post(query_points_url, headers=headers, data=json.dumps(body)) as response:
                    if response.status == 404:
                        search_url = f"{self.base_url}/collections/{self.collection}/points/search"
                        fallback = {
                            "vector": query_vector,
                            "limit": limit,
                            "with_payload": True,
                        }
                        async with session.post(search_url, headers=headers, data=json.dumps(fallback)) as fallback_response:
                            if fallback_response.status != 200:
                                raise RuntimeError(
                                    f"Failed to search Qdrant collection {self.collection}: "
                                    f"{fallback_response.status} {await fallback_response.text()}"
                                )
                            data = await fallback_response.json()
                            return list(data.get("result", []))
                    if response.status != 200:
                        raise RuntimeError(
                            f"Failed to query Qdrant collection {self.collection}: {response.status} {await response.text()}"
                        )
                    data = await response.json()
                    return list(data.get("result", {}).get("points", data.get("result", [])))
        except aiohttp.ClientError as exc:
            fail_or_skip(
                f"Could not reach Qdrant at {self.base_url}. Start Qdrant or set QDRANT_URL correctly. "
                f"Connection error: {type(exc).__name__}: {exc}"
            )


async def embed_text_or_fail(engine: Any, text: str, *, failure_message: str) -> list[float]:
    result = await engine.embed([text])
    vector = getattr(result, "embedding", None)
    if not getattr(result, "ok", False) or not vector:
        fail_or_skip(failure_message)
    return list(vector)


__all__ = [
    "QdrantRetriever",
    "RetrieverDocument",
    "chunk_text",
    "embed_text_or_fail",
    "excerpt",
    "qdrant_api_key",
    "qdrant_headers",
    "require_qdrant_url",
]
