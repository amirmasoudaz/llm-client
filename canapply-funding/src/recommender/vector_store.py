# src/recommender/vector_store.py

from __future__ import annotations

from src.config import settings
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from blake3 import blake3
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from tqdm import tqdm


def make_point_id(tag: str, domain: str, subfield: str) -> int:
    """
    Stable 64-bit identifier derived from tag/domain/subfield combo.
    """
    key = f"{tag}|{domain}|{subfield}".encode("utf-8")
    return int.from_bytes(blake3(key).digest()[:8], "big", signed=False)


def make_subfield_id(domain: str, subfield: str) -> int:
    key = f"subfield|{domain}|{subfield}".encode("utf-8")
    return int.from_bytes(blake3(key).digest()[:8], "big", signed=False)


@dataclass
class TagPoint:
    point_id: int
    vector: np.ndarray
    payload: dict


class TagVectorStore:
    def __init__(
        self,
        *,
        collection: str = "funding_tags",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        timeout: Optional[float] = None,
    ) -> None:
        self.collection = collection
        self.client = QdrantClient(
            url=url or settings.QDRANT_URL,
            api_key=api_key or settings.QDRANT_API_KEY,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )

    def ensure_collection(self, vector_size: int) -> None:
        try:
            info = self.client.get_collection(self.collection)
            existing = info.config.params.vectors
            if isinstance(existing, dict):
                size = list(existing.values())[0].size
            else:
                size = existing.size
            if size != vector_size:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' dimension mismatch "
                    f"(expected {vector_size}, found {size}). Recreate the collection manually."
                )
            return
        except Exception:
            pass

        vectors_config = qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
        try:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=vectors_config,
                optimizers_config=qmodels.OptimizersConfigDiff(
                    default_segment_number=2,
                    indexing_threshold=20000,
                ),
            )
        except Exception as exc:
            # If collection already exists (race), ignore; otherwise raise.
            if "already exists" not in str(exc).lower():
                raise

    def upsert_points(self, points: Sequence[TagPoint], *, batch_size: int = 512) -> None:
        if not points:
            return
        batch_size = max(1, batch_size)
        total = len(points)
        for i in tqdm(range(0, total, batch_size), desc="Upserting points", unit="batch"):
            chunk = points[i : i + batch_size]
            payload_points = [
                qmodels.PointStruct(
                    id=point.point_id,
                    vector=point.vector.astype(np.float32).tolist(),
                    payload=point.payload,
                )
                for point in chunk
            ]
            self.client.upsert(collection_name=self.collection, points=payload_points)

    def delete_points(self, point_ids: Iterable[int]) -> None:
        ids = list(point_ids)
        if not ids:
            return
        self.client.delete(collection_name=self.collection, points_selector=qmodels.PointIdsList(points=ids))

    def fetch_all(self, *, with_vectors: bool = True, batch: int = 256) -> List[qmodels.Record]:
        records: List[qmodels.Record] = []
        offset = None
        while True:
            response, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch,
                with_vectors=with_vectors,
                offset=offset,
            )
            records.extend(response)
            if offset is None:
                break
        return records

    def retrieve_by_ids(self, ids: Sequence[int], *, with_vectors: bool = False) -> List[qmodels.Record]:
        if not ids:
            return []
        chunks: List[qmodels.Record] = []
        for i in range(0, len(ids), 256):
            chunk = ids[i : i + 256]
            results = self.client.retrieve(
                collection_name=self.collection,
                ids=chunk,
                with_vectors=with_vectors,
            )
            chunks.extend(results)
        return chunks
