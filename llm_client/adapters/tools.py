"""
Generic tool-construction helpers for service adaptors.
"""

from __future__ import annotations

from typing import Any

from ..tools import Tool, ToolExecutionMetadata, tool_from_function
from .redis import (
    RedisAdaptor,
    RedisDeleteRequest,
    RedisGetRequest,
    RedisHashDeleteRequest,
    RedisHashGetRequest,
    RedisHashSetRequest,
    RedisSetRequest,
)
from .sql import SQLAdaptor, SQLMutationRequest, SQLQueryRequest
from .vector import VectorAdaptor, VectorDeleteRequest, VectorPoint, VectorSearchRequest, VectorUpsertRequest


def _tool_with_execution(
    func: Any,
    *,
    name: str,
    description: str,
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    tool = tool_from_function(func, name=name, description=description)
    if metadata is not None:
        tool.execution = metadata
    return tool


def build_sql_query_tool(
    adaptor: SQLAdaptor,
    *,
    name: str = "sql_query",
    description: str = "Run a read-only SQL query through a configured SQL adaptor.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _sql_query(statement: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        result = await adaptor.query(
            SQLQueryRequest(statement=statement, parameters=parameters or {})
        )
        return {
            "row_count": result.row_count,
            "rows": result.rows,
            "backend": result.metadata.backend,
        }

    return _tool_with_execution(_sql_query, name=name, description=description, metadata=metadata)


def build_sql_execute_tool(
    adaptor: SQLAdaptor,
    *,
    name: str = "sql_execute",
    description: str = "Run an explicitly write-enabled SQL statement through a configured SQL adaptor.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _sql_execute(
        statement: str,
        parameters: dict[str, Any] | None = None,
        allow_write: bool = False,
    ) -> dict[str, Any]:
        result = await adaptor.execute(
            SQLMutationRequest(
                statement=statement,
                parameters=parameters or {},
                allow_write=allow_write,
            )
        )
        return {
            "row_count": result.row_count,
            "affected_count": result.affected_count,
            "last_insert_id": result.last_insert_id,
            "backend": result.metadata.backend,
        }

    return _tool_with_execution(_sql_execute, name=name, description=description, metadata=metadata)


def build_redis_get_tool(
    adaptor: RedisAdaptor,
    *,
    name: str = "redis_get",
    description: str = "Fetch one value from Redis by key.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _redis_get(key: str) -> dict[str, Any]:
        result = await adaptor.get(RedisGetRequest(key=key))
        return {"key": result.key, "value": result.value, "found": result.found}

    return _tool_with_execution(_redis_get, name=name, description=description, metadata=metadata)


def build_redis_set_tool(
    adaptor: RedisAdaptor,
    *,
    name: str = "redis_set",
    description: str = "Write one Redis value by key.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _redis_set(key: str, value: str, ttl_seconds: int | None = None) -> dict[str, Any]:
        result = await adaptor.set(RedisSetRequest(key=key, value=value, ttl_seconds=ttl_seconds))
        return {"key": result.key, "written": result.written}

    return _tool_with_execution(_redis_set, name=name, description=description, metadata=metadata)


def build_redis_delete_tool(
    adaptor: RedisAdaptor,
    *,
    name: str = "redis_delete",
    description: str = "Delete one Redis key.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _redis_delete(key: str) -> dict[str, Any]:
        result = await adaptor.delete(RedisDeleteRequest(key=key))
        return {"key": result.key, "deleted_count": result.deleted_count}

    return _tool_with_execution(_redis_delete, name=name, description=description, metadata=metadata)


def build_redis_hash_get_tool(
    adaptor: RedisAdaptor,
    *,
    name: str = "redis_hash_get",
    description: str = "Fetch one Redis hash field.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _redis_hash_get(key: str, field_name: str) -> dict[str, Any]:
        result = await adaptor.hget(RedisHashGetRequest(key=key, field_name=field_name))
        return {
            "key": result.key,
            "field_name": result.field_name,
            "value": result.value,
            "found": result.found,
        }

    return _tool_with_execution(_redis_hash_get, name=name, description=description, metadata=metadata)


def build_redis_hash_set_tool(
    adaptor: RedisAdaptor,
    *,
    name: str = "redis_hash_set",
    description: str = "Write one Redis hash field.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _redis_hash_set(key: str, field_name: str, value: str) -> dict[str, Any]:
        result = await adaptor.hset(RedisHashSetRequest(key=key, field_name=field_name, value=value))
        return {
            "key": result.key,
            "field_name": result.field_name,
            "written": result.written,
        }

    return _tool_with_execution(_redis_hash_set, name=name, description=description, metadata=metadata)


def build_redis_hash_delete_tool(
    adaptor: RedisAdaptor,
    *,
    name: str = "redis_hash_delete",
    description: str = "Delete one Redis hash field.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _redis_hash_delete(key: str, field_name: str) -> dict[str, Any]:
        result = await adaptor.hdel(RedisHashDeleteRequest(key=key, field_name=field_name))
        return {
            "key": result.key,
            "field_name": result.field_name,
            "deleted_count": result.deleted_count,
        }

    return _tool_with_execution(_redis_hash_delete, name=name, description=description, metadata=metadata)


def build_vector_search_tool(
    adaptor: VectorAdaptor,
    *,
    name: str = "vector_search",
    description: str = "Search a vector collection through a configured vector adaptor.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _vector_search(
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = await adaptor.search(
            VectorSearchRequest(
                collection=collection,
                query_vector=query_vector,
                limit=limit,
                filters=filters or {},
            )
        )
        return {
            "collection": result.collection,
            "matches": [
                {"point_id": item.point_id, "score": item.score, "payload": item.payload}
                for item in result.matches
            ],
        }

    return _tool_with_execution(_vector_search, name=name, description=description, metadata=metadata)


def build_vector_upsert_tool(
    adaptor: VectorAdaptor,
    *,
    name: str = "vector_upsert",
    description: str = "Upsert vector points through a configured vector adaptor.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _vector_upsert(
        collection: str,
        points: list[dict[str, Any]],
        create_if_missing: bool = False,
    ) -> dict[str, Any]:
        result = await adaptor.upsert(
            VectorUpsertRequest(
                collection=collection,
                points=[
                    VectorPoint(
                        point_id=str(item["point_id"]),
                        vector=list(item["vector"]),
                        payload=dict(item.get("payload", {})),
                    )
                    for item in points
                ],
                create_if_missing=create_if_missing,
            )
        )
        return {"collection": result.collection, "upserted_count": result.upserted_count}

    return _tool_with_execution(_vector_upsert, name=name, description=description, metadata=metadata)


def build_vector_delete_tool(
    adaptor: VectorAdaptor,
    *,
    name: str = "vector_delete",
    description: str = "Delete vector points through a configured vector adaptor.",
    metadata: ToolExecutionMetadata | None = None,
) -> Tool:
    async def _vector_delete(
        collection: str,
        point_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = await adaptor.delete(
            VectorDeleteRequest(
                collection=collection,
                point_ids=list(point_ids or []),
                filters=dict(filters or {}),
            )
        )
        return {"collection": result.collection, "deleted_count": result.deleted_count}

    return _tool_with_execution(_vector_delete, name=name, description=description, metadata=metadata)


__all__ = [
    "build_redis_delete_tool",
    "build_redis_get_tool",
    "build_redis_hash_delete_tool",
    "build_redis_hash_get_tool",
    "build_redis_hash_set_tool",
    "build_redis_set_tool",
    "build_sql_execute_tool",
    "build_sql_query_tool",
    "build_vector_delete_tool",
    "build_vector_search_tool",
    "build_vector_upsert_tool",
]
