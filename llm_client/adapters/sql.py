"""
Generic and backend-specific SQL adaptors.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Protocol

from .base import (
    AdaptorCapability,
    AdaptorError,
    AdaptorExecutionOptions,
    AdaptorMetadata,
    AdaptorOperation,
    AdaptorRuntime,
    AdaptorTimeoutError,
    await_adaptor_timeout,
    run_adaptor_operation,
)


class SQLDialect(str, Enum):
    POSTGRES = "postgres"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    GENERIC = "generic"


@dataclass(frozen=True)
class SQLQueryRequest:
    statement: str
    parameters: dict[str, Any] | tuple[Any, ...] | list[Any] = field(default_factory=dict)
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLMutationRequest:
    statement: str
    parameters: dict[str, Any] | tuple[Any, ...] | list[Any] = field(default_factory=dict)
    allow_write: bool = False
    options: AdaptorExecutionOptions = field(default_factory=AdaptorExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SQLQueryResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(backend="sql", operation=AdaptorOperation.QUERY)
    )


@dataclass(frozen=True)
class SQLExecutionResult:
    row_count: int = 0
    affected_count: int = 0
    last_insert_id: str | int | None = None
    metadata: AdaptorMetadata = field(
        default_factory=lambda: AdaptorMetadata(
            backend="sql",
            operation=AdaptorOperation.EXECUTE,
            read_only=False,
        )
    )


class SQLAdaptor(Protocol):
    dialect: SQLDialect
    backend_name: str
    read_only: bool

    async def query(self, request: SQLQueryRequest) -> SQLQueryResult:
        """Run a read-oriented SQL query."""

    async def execute(self, request: SQLMutationRequest) -> SQLExecutionResult:
        """Run a mutation or write statement."""


_READ_ONLY_LEADING_KEYWORDS = {
    "select",
    "show",
    "describe",
    "desc",
    "explain",
    "with",
    "values",
    "pragma",
}
_NAMED_PARAMETER_PATTERN = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")


class SQLSafetyError(AdaptorError):
    """Raised when a SQL operation violates adaptor safety policy."""


def _statement_preview(statement: str, *, max_chars: int = 160) -> str:
    normalized = " ".join(statement.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[:max_chars].rstrip()}..."


def _strip_leading_sql_comments(statement: str) -> str:
    remaining = statement.lstrip()
    while remaining:
        if remaining.startswith("--"):
            newline_index = remaining.find("\n")
            if newline_index == -1:
                return ""
            remaining = remaining[newline_index + 1 :].lstrip()
            continue
        if remaining.startswith("/*"):
            close_index = remaining.find("*/")
            if close_index == -1:
                return ""
            remaining = remaining[close_index + 2 :].lstrip()
            continue
        break
    return remaining


def _leading_sql_keyword(statement: str) -> str:
    stripped = _strip_leading_sql_comments(statement)
    if not stripped:
        return ""
    return stripped.split(None, 1)[0].lower()


def is_read_only_statement(statement: str) -> bool:
    return _leading_sql_keyword(statement) in _READ_ONLY_LEADING_KEYWORDS


def _normalize_parameter_sequence(parameters: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    return tuple(parameters)


def _translate_named_parameters(
    statement: str,
    parameters: dict[str, Any],
    *,
    dialect: SQLDialect,
) -> tuple[str, tuple[Any, ...]]:
    ordered_values: list[Any] = []

    def _replace(match: re.Match[str]) -> str:
        parameter_name = match.group(1)
        if parameter_name not in parameters:
            raise SQLSafetyError(
                f"Missing SQL parameter '{parameter_name}'",
                backend="sql",
                details={"statement": _statement_preview(statement)},
            )
        ordered_values.append(parameters[parameter_name])
        if dialect is SQLDialect.POSTGRES:
            return f"${len(ordered_values)}"
        if dialect is SQLDialect.MYSQL:
            return "%s"
        return "?"

    translated = _NAMED_PARAMETER_PATTERN.sub(_replace, statement)
    return translated, tuple(ordered_values)


def _normalize_sql_request(
    statement: str,
    parameters: dict[str, Any] | tuple[Any, ...] | list[Any],
    *,
    dialect: SQLDialect,
) -> tuple[str, tuple[Any, ...]]:
    if isinstance(parameters, dict):
        return _translate_named_parameters(statement, parameters, dialect=dialect)
    if isinstance(parameters, (tuple, list)):
        return statement, _normalize_parameter_sequence(parameters)
    raise SQLSafetyError(
        "SQL parameters must be a dict, tuple, or list",
        backend="sql",
        details={"statement": _statement_preview(statement)},
    )


def _effective_timeout(
    options: AdaptorExecutionOptions,
    *,
    default_timeout_seconds: float | None,
) -> float | None:
    if options.timeout_seconds is not None:
        return options.timeout_seconds
    return default_timeout_seconds


async def _await_with_timeout(
    awaitable: Any,
    *,
    backend: str,
    operation: AdaptorOperation,
    timeout_seconds: float | None,
    details: dict[str, Any],
) -> Any:
    try:
        if timeout_seconds is None:
            return await awaitable
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise AdaptorTimeoutError(
            backend=backend,
            operation=operation,
            details={**details, "timeout_seconds": timeout_seconds},
        ) from exc


@asynccontextmanager
async def _acquire_connection(resource: Any):
    acquire = getattr(resource, "acquire", None)
    if callable(acquire):
        acquired = acquire()
        if hasattr(acquired, "__aenter__") and hasattr(acquired, "__aexit__"):
            async with acquired as conn:
                yield conn
            return
        if hasattr(acquired, "__await__"):
            conn = await acquired
            try:
                yield conn
            finally:
                release = getattr(resource, "release", None)
                if callable(release):
                    released = release(conn)
                    if hasattr(released, "__await__"):
                        await released
            return
        yield acquired
        return
    yield resource


@asynccontextmanager
async def _acquire_cursor(conn: Any):
    cursor_factory = getattr(conn, "cursor", None)
    if not callable(cursor_factory):
        raise SQLSafetyError(
            "MySQL adaptor requires a connection object with cursor() support",
            backend="mysql",
        )
    cursor = cursor_factory()
    if hasattr(cursor, "__aenter__") and hasattr(cursor, "__aexit__"):
        async with cursor as active_cursor:
            yield active_cursor
        return
    if hasattr(cursor, "__await__"):
        active_cursor = await cursor
        try:
            yield active_cursor
        finally:
            close = getattr(active_cursor, "close", None)
            if callable(close):
                closed = close()
                if hasattr(closed, "__await__"):
                    await closed
        return
    try:
        yield cursor
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            closed = close()
            if hasattr(closed, "__await__"):
                await closed


def _coerce_row_to_dict(row: Any, column_names: list[str] | None = None) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    keys = getattr(row, "keys", None)
    if callable(keys):
        try:
            return {str(key): row[key] for key in row.keys()}
        except Exception:
            pass
    if column_names is not None:
        return {column_names[index]: value for index, value in enumerate(row)}
    raise SQLSafetyError("Could not normalize SQL row output", backend="sql")


def _parse_postgres_command_tag(command_tag: str) -> int:
    parts = command_tag.split()
    if not parts:
        return 0
    tail = parts[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def _build_sql_metadata(
    *,
    backend: str,
    operation: AdaptorOperation,
    read_only: bool,
    dialect: SQLDialect,
    extra: dict[str, Any] | None = None,
) -> AdaptorMetadata:
    capabilities = (AdaptorCapability.READ,) if read_only else (AdaptorCapability.READ, AdaptorCapability.WRITE)
    return AdaptorMetadata(
        backend=backend,
        operation=operation,
        read_only=read_only,
        capabilities=capabilities,
        metadata={"dialect": dialect.value, **dict(extra or {})},
    )


@dataclass
class PostgresSQLAdaptor:
    pool: Any
    read_only: bool = True
    default_timeout_seconds: float | None = None
    backend_name: str = "postgres"
    dialect: SQLDialect = SQLDialect.POSTGRES
    runtime: AdaptorRuntime = field(default_factory=AdaptorRuntime)

    async def query(self, request: SQLQueryRequest) -> SQLQueryResult:
        if not is_read_only_statement(request.statement):
            raise SQLSafetyError(
                "Postgres query adaptor only permits read-only SQL statements",
                backend=self.backend_name,
                operation=AdaptorOperation.QUERY,
                details={"statement": _statement_preview(request.statement)},
            )
        statement, parameters = _normalize_sql_request(
            request.statement,
            request.parameters,
            dialect=self.dialect,
        )
        timeout_seconds = _effective_timeout(
            request.options,
            default_timeout_seconds=self.default_timeout_seconds,
        )
        details = {"statement": _statement_preview(statement)}
        async def _run() -> SQLQueryResult:
            try:
                async with _acquire_connection(self.pool) as conn:
                    rows = await await_adaptor_timeout(
                        conn.fetch(statement, *parameters),
                        backend=self.backend_name,
                        operation=AdaptorOperation.QUERY,
                        timeout_seconds=timeout_seconds,
                        details=details,
                    )
            except AdaptorError:
                raise
            except Exception as exc:
                raise AdaptorError(
                    f"Postgres query failed: {exc}",
                    backend=self.backend_name,
                    operation=AdaptorOperation.QUERY,
                    details=details,
                ) from exc
            normalized_rows = [_coerce_row_to_dict(row) for row in rows]
            return SQLQueryResult(
                rows=normalized_rows,
                row_count=len(normalized_rows),
                metadata=_build_sql_metadata(
                    backend=self.backend_name,
                    operation=AdaptorOperation.QUERY,
                    read_only=True,
                    dialect=self.dialect,
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.QUERY,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"statement": details["statement"]},
        )

    async def execute(self, request: SQLMutationRequest) -> SQLExecutionResult:
        if self.read_only:
            raise SQLSafetyError(
                "Postgres adaptor is configured read-only; create a writable adaptor explicitly to execute mutations",
                backend=self.backend_name,
                operation=AdaptorOperation.EXECUTE,
                details={"statement": _statement_preview(request.statement)},
            )
        if not request.allow_write:
            raise SQLSafetyError(
                "SQL mutations require allow_write=True",
                backend=self.backend_name,
                operation=AdaptorOperation.EXECUTE,
                details={"statement": _statement_preview(request.statement)},
            )
        if is_read_only_statement(request.statement):
            raise SQLSafetyError(
                "Use query() for read-only SQL statements",
                backend=self.backend_name,
                operation=AdaptorOperation.EXECUTE,
                details={"statement": _statement_preview(request.statement)},
            )
        statement, parameters = _normalize_sql_request(
            request.statement,
            request.parameters,
            dialect=self.dialect,
        )
        timeout_seconds = _effective_timeout(
            request.options,
            default_timeout_seconds=self.default_timeout_seconds,
        )
        details = {"statement": _statement_preview(statement)}
        async def _run() -> SQLExecutionResult:
            try:
                async with _acquire_connection(self.pool) as conn:
                    command_tag = await await_adaptor_timeout(
                        conn.execute(statement, *parameters),
                        backend=self.backend_name,
                        operation=AdaptorOperation.EXECUTE,
                        timeout_seconds=timeout_seconds,
                        details=details,
                    )
            except AdaptorError:
                raise
            except Exception as exc:
                raise AdaptorError(
                    f"Postgres execution failed: {exc}",
                    backend=self.backend_name,
                    operation=AdaptorOperation.EXECUTE,
                    details=details,
                ) from exc
            row_count = _parse_postgres_command_tag(str(command_tag))
            return SQLExecutionResult(
                row_count=row_count,
                affected_count=row_count,
                metadata=_build_sql_metadata(
                    backend=self.backend_name,
                    operation=AdaptorOperation.EXECUTE,
                    read_only=False,
                    dialect=self.dialect,
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.EXECUTE,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"statement": details["statement"]},
        )


@dataclass
class MySQLSQLAdaptor:
    pool: Any
    read_only: bool = True
    default_timeout_seconds: float | None = None
    backend_name: str = "mysql"
    dialect: SQLDialect = SQLDialect.MYSQL
    runtime: AdaptorRuntime = field(default_factory=AdaptorRuntime)

    async def query(self, request: SQLQueryRequest) -> SQLQueryResult:
        if not is_read_only_statement(request.statement):
            raise SQLSafetyError(
                "MySQL query adaptor only permits read-only SQL statements",
                backend=self.backend_name,
                operation=AdaptorOperation.QUERY,
                details={"statement": _statement_preview(request.statement)},
            )
        statement, parameters = _normalize_sql_request(
            request.statement,
            request.parameters,
            dialect=self.dialect,
        )
        timeout_seconds = _effective_timeout(
            request.options,
            default_timeout_seconds=self.default_timeout_seconds,
        )
        details = {"statement": _statement_preview(statement)}
        async def _run() -> SQLQueryResult:
            try:
                async with _acquire_connection(self.pool) as conn:
                    async with _acquire_cursor(conn) as cursor:
                        await await_adaptor_timeout(
                            cursor.execute(statement, parameters),
                            backend=self.backend_name,
                            operation=AdaptorOperation.QUERY,
                            timeout_seconds=timeout_seconds,
                            details=details,
                        )
                        rows = await await_adaptor_timeout(
                            cursor.fetchall(),
                            backend=self.backend_name,
                            operation=AdaptorOperation.QUERY,
                            timeout_seconds=timeout_seconds,
                            details=details,
                        )
                        description = list(getattr(cursor, "description", ()) or ())
            except AdaptorError:
                raise
            except Exception as exc:
                raise AdaptorError(
                    f"MySQL query failed: {exc}",
                    backend=self.backend_name,
                    operation=AdaptorOperation.QUERY,
                    details=details,
                ) from exc
            column_names = [str(item[0]) for item in description if item]
            normalized_rows = [_coerce_row_to_dict(row, column_names or None) for row in rows]
            return SQLQueryResult(
                rows=normalized_rows,
                row_count=len(normalized_rows),
                metadata=_build_sql_metadata(
                    backend=self.backend_name,
                    operation=AdaptorOperation.QUERY,
                    read_only=True,
                    dialect=self.dialect,
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.QUERY,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"statement": details["statement"]},
        )

    async def execute(self, request: SQLMutationRequest) -> SQLExecutionResult:
        if self.read_only:
            raise SQLSafetyError(
                "MySQL adaptor is configured read-only; create a writable adaptor explicitly to execute mutations",
                backend=self.backend_name,
                operation=AdaptorOperation.EXECUTE,
                details={"statement": _statement_preview(request.statement)},
            )
        if not request.allow_write:
            raise SQLSafetyError(
                "SQL mutations require allow_write=True",
                backend=self.backend_name,
                operation=AdaptorOperation.EXECUTE,
                details={"statement": _statement_preview(request.statement)},
            )
        if is_read_only_statement(request.statement):
            raise SQLSafetyError(
                "Use query() for read-only SQL statements",
                backend=self.backend_name,
                operation=AdaptorOperation.EXECUTE,
                details={"statement": _statement_preview(request.statement)},
            )
        statement, parameters = _normalize_sql_request(
            request.statement,
            request.parameters,
            dialect=self.dialect,
        )
        timeout_seconds = _effective_timeout(
            request.options,
            default_timeout_seconds=self.default_timeout_seconds,
        )
        details = {"statement": _statement_preview(statement)}
        async def _run() -> SQLExecutionResult:
            try:
                async with _acquire_connection(self.pool) as conn:
                    async with _acquire_cursor(conn) as cursor:
                        await await_adaptor_timeout(
                            cursor.execute(statement, parameters),
                            backend=self.backend_name,
                            operation=AdaptorOperation.EXECUTE,
                            timeout_seconds=timeout_seconds,
                            details=details,
                        )
                        commit = getattr(conn, "commit", None)
                        if callable(commit):
                            commit_result = commit()
                            if hasattr(commit_result, "__await__"):
                                await commit_result
                        row_count = max(int(getattr(cursor, "rowcount", 0) or 0), 0)
                        last_insert_id = getattr(cursor, "lastrowid", None)
            except AdaptorError:
                raise
            except Exception as exc:
                raise AdaptorError(
                    f"MySQL execution failed: {exc}",
                    backend=self.backend_name,
                    operation=AdaptorOperation.EXECUTE,
                    details=details,
                ) from exc
            return SQLExecutionResult(
                row_count=row_count,
                affected_count=row_count,
                last_insert_id=last_insert_id,
                metadata=_build_sql_metadata(
                    backend=self.backend_name,
                    operation=AdaptorOperation.EXECUTE,
                    read_only=False,
                    dialect=self.dialect,
                ),
            )

        return await run_adaptor_operation(
            self.runtime,
            backend=self.backend_name,
            operation=AdaptorOperation.EXECUTE,
            retry_attempts=request.options.retry_attempts,
            func=_run,
            metadata={"statement": details["statement"]},
        )


__all__ = [
    "MySQLSQLAdaptor",
    "PostgresSQLAdaptor",
    "SQLAdaptor",
    "SQLDialect",
    "SQLExecutionResult",
    "SQLMutationRequest",
    "SQLQueryRequest",
    "SQLQueryResult",
    "SQLSafetyError",
    "is_read_only_statement",
]
