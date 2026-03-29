"""
Generic and backend-specific vector-store adaptors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .base import (
    AdaptorCapability,
    AdaptorError,
    AdaptorExecutionOptions,
    AdaptorMetadata,
    AdaptorOperation,
    AdaptorRuntime,
    await_adaptor_timeout,
    run_adaptor_operation,
)


@dataclass(frozen=True)
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorUpsertRequest:
    collection: str
    points: list[VectorPoint]
    create_if_missing: bool = False
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorUpsertResult:
    collection: str
    upserted_count: int
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(
            backend="vector",
            operation=AdaptorOperation.UPSERT,
            read_only=False,
        )
    )


@dataclass(frozen=True)
class VectorSearchRequest:
    collection: str
    query_vector: list[float]
    limit: int = 10
    filters: dict[str, Any] = field(default_factory=dict)
    with_payload: bool = True
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchMatch:
    point_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorSearchResult:
    collection: str
    matches: list[VectorSearchMatch] = field(default_factory=list)
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(backend="vector", operation=AdaptorOperation.SEARCH)
    )


@dataclass(frozen=True)
class VectorDeleteRequest:
    collection: str
    point_ids: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorDeleteResult:
    collection: str
    deleted_count: int
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(
            backend="vector",
            operation=AdaptorOperation.DELETE,
            read_only=False,
        )
    )


class VectorAdaptor(Protocol):
    backend_name: str

    async def upsert(self, request: VectorUpsertRequest) -> VectorUpsertResult:
        """Upsert points into one collection."""

    async def search(self, request: VectorSearchRequest) -> VectorSearchResult:
        """Search one collection."""

    async def delete(self, request: VectorDeleteRequest) -> VectorDeleteResult:
        """Delete points by id or filter."""


class VectorSafetyError(AdaptorError):
    """Raised when a vector adaptor operation violates safety policy."""


def _vector_metadata(
    *,
    operation: AdaptorOperation,
    read_only: bool,
    extra: dict[str, Any] | None = None,
) -> AdaptorMetadata:
    if read_only:
        capabilities = (AdaptorCapability.SEARCH,)
    elif operation is AdaptorOperation.DELETE:
        capabilities = (AdaptorCapability.SEARCH, AdaptorCapability.DELETE)
    else:
        capabilities = (AdaptorCapability.SEARCH, AdaptorCapability.UPSERT)
    return AdaptorMetadata(
        backend="vector",
        operation=operation,
        read_only=read_only,
        capabilities=capabilities,
        metadata=dict(extra or {}),
    )


def _qdrant_models() -> Any | None:
    try:
        from qdrant_client.http import models as qdrant_models

        return qdrant_models
    except Exception:
        return None


def _build_qdrant_vector_params(size: int, distance: str) -> Any:
    models = _qdrant_models()
    if models is None:
        return {"size": size, "distance": distance}
    distance_enum = getattr(models.Distance, distance.upper(), distance)
    return models.VectorParams(size=size, distance=distance_enum)


def _build_qdrant_points(points: list[VectorPoint]) -> list[Any]:
    models = _qdrant_models()
    if models is None:
        return [
            {"id": point.point_id, "vector": point.vector, "payload": point.payload}
            for point in points
        ]
    return [
        models.PointStruct(id=point.point_id, vector=point.vector, payload=point.payload)
        for point in points
    ]


def _build_qdrant_filter(filters: dict[str, Any]) -> Any:
    if not filters:
        return None
    models = _qdrant_models()
    if models is None:
        return filters
    must_conditions: list[Any] = []
    for key, value in filters.items():
        must_conditions.append(
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
        )
    return models.Filter(must=must_conditions)


def _build_qdrant_points_selector(point_ids: list[str], filters: dict[str, Any]) -> Any:
    models = _qdrant_models()
    selector_filter = _build_qdrant_filter(filters)
    if models is None:
        if point_ids:
            return {"points": point_ids}
        return {"filter": selector_filter}
    if point_ids:
        return models.PointIdsList(points=point_ids)
    return models.FilterSelector(filter=selector_filter)


def _coerce_qdrant_matches(raw: Any) -> list[VectorSearchMatch]:
    if isinstance(raw, list):
        items = raw
    elif hasattr(raw, "points"):
        items = list(raw.points)
    elif hasattr(raw, "result"):
        result = raw.result
        items = list(getattr(result, "points", result))
    else:
        items = []
    matches: list[VectorSearchMatch] = []
    for item in items:
        point_id = getattr(item, "id", None)
        score = getattr(item, "score", None)
        payload = getattr(item, "payload", None)
        if isinstance(item, dict):
            point_id = item.get("id")
            score = item.get("score")
            payload = item.get("payload")
        matches.append(
            VectorSearchMatch(
                point_id=str(point_id),
                score=float(score or 0.0),
                payload=dict(payload or {}),
            )
        )
    return matches


@dataclass
class QdrantVectorAdaptor:
    client: Any
    vector_size: int | None = None
    distance: str = "Cosine"
    max_search_limit: int = 25
    allow_delete: bool = False
    default_timeout_seconds: float | None = None
    backend_name: str = "qdrant"
    runtime: AdaptorRuntime = field(default_factory=AdaptorRuntime)

    def _effective_timeout(self, options: AdaptorExecutionOptions) -> float | None:
        if options.timeout_seconds is not None:
            return options.timeout_seconds
        return self.default_timeout_seconds

    async def _ensure_collection(self, collection: str, vector_size: int) -> None:
        exists = False
        collection_exists = getattr(self.client, "collection_exists", None)
        if callable(collection_exists):
            exists = bool(await collection_exists(collection))
        if exists:
            return
        create_collection = getattr(self.client, "create_collection", None)
        if not callable(create_collection):
            raise VectorSafetyError(
                "Qdrant client does not support collection creation",
                backend=self.backend_name,
                details={"collection": collection},
            )
        await await_adaptor_timeout(
            create_collection(
                collection_name=collection,
                vectors_config=_build_qdrant_vector_params(vector_size, self.distance),
                timeout=self.default_timeout_seconds,
            ),
            backend=self.backend_name,
            operation=AdaptorOperation.UPSERT,
            timeout_seconds=self.default_timeout_seconds,
            details={"collection": collection},
        )

    async def upsert(self, request: VectorUpsertRequest) -> VectorUpsertResult:
        if not request.points:
            return VectorUpsertResult(
                collection=request.collection,
                upserted_count=0,
                metadata=_vector_metadata(
                    operation=AdaptorOperation.UPSERT,
                    read_only=False,
                    extra={"collection": request.collection},
                ),
            )
        vector_size = self.vector_size or len(request.points[0].vector)
        if request.create_if_missing:
            await self._ensure_collection(request.collection, vector_size)

        async def _run() -> VectorUpsertResult:
            await await_adaptor_timeout(
                self.client.upsert(
                    collection_name=request.collection,
                    points=_build_qdrant_points(request.points),
                    wait=True,
                    timeout=self._effective_timeout(request.options),
                ),
                backend=self.backend_name,
                operation=AdaptorOperation.UPSERT,
                timeout_seconds=self._effective_timeout(request.options),
                details={"collection": request.collection},
            )
            return VectorUpsertResult(
                collection=request.collection,
                upserted_count=len(request.points),
                metadata=_vector_metadata(
                    operation=AdaptorOperation.UPSERT,
                    read_only=False,
                    extra={"collection": request.collection},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.UPSERT,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"collection": request.collection, "point_count": len(request.points)},
        )

    async def search(self, request: VectorSearchRequest) -> VectorSearchResult:
        if request.limit <= 0:
            raise VectorSafetyError("Vector search limit must be positive", backend=self.backend_name)
        if request.limit > self.max_search_limit:
            raise VectorSafetyError(
                f"Vector search limit exceeds max_search_limit={self.max_search_limit}",
                backend=self.backend_name,
                details={"limit": request.limit, "max_search_limit": self.max_search_limit},
            )
        query_filter = _build_qdrant_filter(request.filters)
        timeout = self._effective_timeout(request.options)

        async def _run() -> VectorSearchResult:
            query_points = getattr(self.client, "query_points", None)
            if callable(query_points):
                raw = await await_adaptor_timeout(
                    query_points(
                        collection_name=request.collection,
                        query=request.query_vector,
                        limit=request.limit,
                        with_payload=request.with_payload,
                        query_filter=query_filter,
                        timeout=timeout,
                    ),
                    backend=self.backend_name,
                    operation=AdaptorOperation.SEARCH,
                    timeout_seconds=timeout,
                    details={"collection": request.collection, "limit": request.limit},
                )
            else:
                raw = await await_adaptor_timeout(
                    self.client.search(
                        collection_name=request.collection,
                        query_vector=request.query_vector,
                        limit=request.limit,
                        with_payload=request.with_payload,
                        query_filter=query_filter,
                        timeout=timeout,
                    ),
                    backend=self.backend_name,
                    operation=AdaptorOperation.SEARCH,
                    timeout_seconds=timeout,
                    details={"collection": request.collection, "limit": request.limit},
                )
            return VectorSearchResult(
                collection=request.collection,
                matches=_coerce_qdrant_matches(raw),
                metadata=_vector_metadata(
                    operation=AdaptorOperation.SEARCH,
                    read_only=True,
                    extra={"collection": request.collection, "limit": request.limit},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.SEARCH,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"collection": request.collection, "limit": request.limit},
        )

    async def delete(self, request: VectorDeleteRequest) -> VectorDeleteResult:
        if not self.allow_delete:
            raise VectorSafetyError(
                "Vector delete operations require allow_delete=True on the adaptor",
                backend=self.backend_name,
            )
        if not request.point_ids and not request.filters:
            raise VectorSafetyError(
                "Vector delete requires point_ids or filters",
                backend=self.backend_name,
            )
        selector = _build_qdrant_points_selector(request.point_ids, request.filters)
        timeout = self._effective_timeout(request.options)

        async def _run() -> VectorDeleteResult:
            await await_adaptor_timeout(
                self.client.delete(
                    collection_name=request.collection,
                    points_selector=selector,
                    wait=True,
                    timeout=timeout,
                ),
                backend=self.backend_name,
                operation=AdaptorOperation.DELETE,
                timeout_seconds=timeout,
                details={"collection": request.collection},
            )
            deleted_count = len(request.point_ids) if request.point_ids else 0
            return VectorDeleteResult(
                collection=request.collection,
                deleted_count=deleted_count,
                metadata=_vector_metadata(
                    operation=AdaptorOperation.DELETE,
                    read_only=False,
                    extra={"collection": request.collection},
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.DELETE,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"collection": request.collection, "point_count": len(request.point_ids)},
        )


__all__ = [
    "QdrantVectorAdaptor",
    "VectorAdaptor",
    "VectorDeleteRequest",
    "VectorDeleteResult",
    "VectorPoint",
    "VectorSafetyError",
    "VectorSearchMatch",
    "VectorSearchRequest",
    "VectorSearchResult",
    "VectorUpsertRequest",
    "VectorUpsertResult",
]
