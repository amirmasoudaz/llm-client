from __future__ import annotations

from llm_client.adapters import (
    AdaptorRuntime,
    AdaptorExecutionOptions,
    AdaptorOperation,
    QdrantVectorAdaptor,
    RedisDeleteRequest,
    RedisGetRequest,
    RedisKVAdaptor,
    SQLDialect,
    SQLMutationRequest,
    SQLQueryRequest,
    VectorDeleteRequest,
    VectorPoint,
    VectorSearchRequest,
    VectorUpsertRequest,
    build_sql_query_tool,
    build_vector_search_tool,
)
from llm_client.adapters.base import AdaptorMetadata


def test_sql_request_contracts_capture_statement_parameters_and_options() -> None:
    request = SQLQueryRequest(
        statement="select * from users where id = :user_id",
        parameters={"user_id": "123"},
        options=AdaptorExecutionOptions(timeout_seconds=5.0, retry_attempts=1),
    )
    mutation = SQLMutationRequest(
        statement="update users set active = true where id = :user_id",
        parameters={"user_id": "123"},
        allow_write=True,
    )

    assert request.statement.startswith("select")
    assert request.parameters == {"user_id": "123"}
    assert request.options.timeout_seconds == 5.0
    assert mutation.allow_write is True
    assert SQLDialect.POSTGRES.value == "postgres"


def test_redis_and_vector_request_contracts_have_expected_defaults() -> None:
    redis_get = RedisGetRequest(key="profile:123")
    redis_delete = RedisDeleteRequest(key="profile:123")
    vector_upsert = VectorUpsertRequest(
        collection="docs",
        points=[VectorPoint(point_id="p1", vector=[0.1, 0.2], payload={"kind": "doc"})],
        create_if_missing=True,
    )
    vector_search = VectorSearchRequest(collection="docs", query_vector=[0.1, 0.2], limit=3)
    vector_delete = VectorDeleteRequest(collection="docs", point_ids=["p1"])

    assert redis_get.key == "profile:123"
    assert redis_delete.key == "profile:123"
    assert vector_upsert.create_if_missing is True
    assert vector_search.limit == 3
    assert vector_delete.point_ids == ["p1"]


def test_adaptor_metadata_tracks_backend_operation_and_readonly_state() -> None:
    metadata = AdaptorMetadata(
        backend="postgres",
        operation=AdaptorOperation.QUERY,
        read_only=True,
    )

    assert metadata.backend == "postgres"
    assert metadata.operation is AdaptorOperation.QUERY
    assert metadata.read_only is True
    assert AdaptorRuntime().retry_attempts == 0
    assert RedisKVAdaptor.backend_name == "redis"
    assert QdrantVectorAdaptor.backend_name == "qdrant"


def test_tool_builder_returns_named_tools_without_backend_implementation() -> None:
    class _StubSQLAdaptor:
        dialect = SQLDialect.POSTGRES
        backend_name = "postgres"
        read_only = True

        async def query(self, request: SQLQueryRequest):
            return type(
                "_Result",
                (),
                {
                    "row_count": 1,
                    "rows": [{"ok": True}],
                    "metadata": AdaptorMetadata(backend="postgres", operation=AdaptorOperation.QUERY),
                },
            )()

        async def execute(self, request: SQLMutationRequest):
            raise AssertionError("not used")

    class _StubVectorAdaptor:
        backend_name = "qdrant"

        async def upsert(self, request: VectorUpsertRequest):
            raise AssertionError("not used")

        async def search(self, request: VectorSearchRequest):
            return type(
                "_VectorResult",
                (),
                {
                    "collection": request.collection,
                    "matches": [],
                },
            )()

        async def delete(self, request: VectorDeleteRequest):
            raise AssertionError("not used")

    sql_tool = build_sql_query_tool(_StubSQLAdaptor(), name="read_user_sql")
    vector_tool = build_vector_search_tool(_StubVectorAdaptor(), name="search_docs")

    assert sql_tool.name == "read_user_sql"
    assert vector_tool.name == "search_docs"
