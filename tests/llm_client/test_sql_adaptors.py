from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from llm_client.adapters import (
    AdaptorExecutionOptions,
    AdaptorOperation,
    AdaptorTimeoutError,
    MySQLSQLAdaptor,
    PostgresSQLAdaptor,
    SQLMutationRequest,
    SQLQueryRequest,
    SQLSafetyError,
    is_read_only_statement,
)


class _AcquireContext:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = (exc_type, exc, tb)
        return False


class _FakePostgresPool:
    def __init__(self, conn: "_FakePostgresConn") -> None:
        self._conn = conn

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._conn)


class _FakePostgresRow(dict[str, Any]):
    pass


class _FakePostgresConn:
    def __init__(self) -> None:
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_delay_seconds = 0.0
        self.fetch_rows: list[dict[str, Any]] = []
        self.command_tag = "UPDATE 2"

    async def fetch(self, statement: str, *args: Any) -> list[_FakePostgresRow]:
        self.fetch_calls.append((statement, tuple(args)))
        if self.fetch_delay_seconds:
            await asyncio.sleep(self.fetch_delay_seconds)
        return [_FakePostgresRow(row) for row in self.fetch_rows]

    async def execute(self, statement: str, *args: Any) -> str:
        self.execute_calls.append((statement, tuple(args)))
        return self.command_tag


class _FakeMySQLCursor:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchall_rows: list[Any] = []
        self.description: Sequence[tuple[str, Any, Any, Any, Any, Any, Any]] = ()
        self.rowcount = 0
        self.lastrowid: int | None = None
        self.execute_delay_seconds = 0.0
        self.closed = False

    async def execute(self, statement: str, parameters: tuple[Any, ...]) -> None:
        self.execute_calls.append((statement, tuple(parameters)))
        if self.execute_delay_seconds:
            await asyncio.sleep(self.execute_delay_seconds)

    async def fetchall(self) -> list[Any]:
        return list(self.fetchall_rows)

    async def close(self) -> None:
        self.closed = True


class _FakeMySQLConn:
    def __init__(self, cursor: _FakeMySQLCursor) -> None:
        self._cursor = cursor
        self.committed = False

    async def cursor(self) -> _FakeMySQLCursor:
        return self._cursor

    async def commit(self) -> None:
        self.committed = True


class _FakeMySQLPool:
    def __init__(self, conn: _FakeMySQLConn) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireContext:
        return _AcquireContext(self._conn)


@pytest.mark.asyncio
async def test_postgres_sql_adaptor_translates_named_parameters_and_returns_rows() -> None:
    conn = _FakePostgresConn()
    conn.fetch_rows = [{"ticket_id": "inc-42", "summary": "database lag"}]
    adaptor = PostgresSQLAdaptor(_FakePostgresPool(conn))

    result = await adaptor.query(
        SQLQueryRequest(
            statement="select ticket_id, :topic::text as summary from incidents where tenant_id = :tenant_id",
            parameters={"topic": "database lag", "tenant_id": "tenant-1"},
        )
    )

    assert result.row_count == 1
    assert result.rows[0]["ticket_id"] == "inc-42"
    assert result.metadata.operation is AdaptorOperation.QUERY
    assert conn.fetch_calls == [
        (
            "select ticket_id, $1::text as summary from incidents where tenant_id = $2",
            ("database lag", "tenant-1"),
        )
    ]


@pytest.mark.asyncio
async def test_postgres_sql_adaptor_enforces_read_only_and_explicit_write_gate() -> None:
    read_only_adaptor = PostgresSQLAdaptor(_FakePostgresPool(_FakePostgresConn()))

    with pytest.raises(SQLSafetyError, match="read-only SQL statements"):
        await read_only_adaptor.query(SQLQueryRequest(statement="update incidents set state = 'closed'"))

    with pytest.raises(SQLSafetyError, match="configured read-only"):
        await read_only_adaptor.execute(
            SQLMutationRequest(
                statement="insert into incidents(id) values (:id)",
                parameters={"id": "inc-42"},
                allow_write=True,
            )
        )

    writable_conn = _FakePostgresConn()
    writable_adaptor = PostgresSQLAdaptor(_FakePostgresPool(writable_conn), read_only=False)

    with pytest.raises(SQLSafetyError, match="allow_write=True"):
        await writable_adaptor.execute(
            SQLMutationRequest(
                statement="insert into incidents(id) values (:id)",
                parameters={"id": "inc-42"},
            )
        )

    result = await writable_adaptor.execute(
        SQLMutationRequest(
            statement="insert into incidents(id) values (:id)",
            parameters={"id": "inc-42"},
            allow_write=True,
        )
    )

    assert result.affected_count == 2
    assert writable_conn.execute_calls == [("insert into incidents(id) values ($1)", ("inc-42",))]


@pytest.mark.asyncio
async def test_postgres_sql_adaptor_wraps_timeouts() -> None:
    conn = _FakePostgresConn()
    conn.fetch_delay_seconds = 0.05
    adaptor = PostgresSQLAdaptor(_FakePostgresPool(conn))

    with pytest.raises(AdaptorTimeoutError):
        await adaptor.query(
            SQLQueryRequest(
                statement="select 1",
                options=AdaptorExecutionOptions(timeout_seconds=0.001),
            )
        )


@pytest.mark.asyncio
async def test_mysql_sql_adaptor_translates_named_parameters_and_normalizes_rows() -> None:
    cursor = _FakeMySQLCursor()
    cursor.description = (("user_id", None, None, None, None, None, None), ("plan", None, None, None, None, None, None))
    cursor.fetchall_rows = [("u-1", "pro"), ("u-2", "team")]
    conn = _FakeMySQLConn(cursor)
    adaptor = MySQLSQLAdaptor(_FakeMySQLPool(conn))

    result = await adaptor.query(
        SQLQueryRequest(
            statement="select user_id, plan from users where tenant_id = :tenant_id and active = :active",
            parameters={"tenant_id": "tenant-1", "active": True},
        )
    )

    assert result.row_count == 2
    assert result.rows == [{"user_id": "u-1", "plan": "pro"}, {"user_id": "u-2", "plan": "team"}]
    assert cursor.execute_calls == [
        (
            "select user_id, plan from users where tenant_id = %s and active = %s",
            ("tenant-1", True),
        )
    ]
    assert cursor.closed is True


@pytest.mark.asyncio
async def test_mysql_sql_adaptor_requires_explicit_write_enablement_and_commits() -> None:
    cursor = _FakeMySQLCursor()
    cursor.rowcount = 3
    cursor.lastrowid = 41
    conn = _FakeMySQLConn(cursor)
    adaptor = MySQLSQLAdaptor(_FakeMySQLPool(conn), read_only=False)

    result = await adaptor.execute(
        SQLMutationRequest(
            statement="update users set plan = :plan where tenant_id = :tenant_id",
            parameters={"plan": "enterprise", "tenant_id": "tenant-1"},
            allow_write=True,
        )
    )

    assert result.affected_count == 3
    assert result.last_insert_id == 41
    assert conn.committed is True
    assert cursor.execute_calls == [
        (
            "update users set plan = %s where tenant_id = %s",
            ("enterprise", "tenant-1"),
        )
    ]


def test_is_read_only_statement_recognizes_common_safe_prefixes() -> None:
    assert is_read_only_statement("select 1") is True
    assert is_read_only_statement("  -- note\nwith ranked as (select 1) select * from ranked") is True
    assert is_read_only_statement("insert into events(id) values (1)") is False
