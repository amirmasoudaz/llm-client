from __future__ import annotations

import asyncio

from cookbook_support import fail_or_skip, print_heading, print_json, require_database_dsn, require_optional_module

if not require_optional_module("asyncpg", "Install it with: pip install llm-client[postgres]"):
    raise SystemExit(0)

import asyncpg

from llm_client.adapters import PostgresSQLAdaptor, SQLMutationRequest, SQLQueryRequest, SQLSafetyError


async def main() -> None:
    dsn = require_database_dsn()
    try:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    except Exception as exc:
        fail_or_skip(
            "Could not connect to PostgreSQL for the SQL adaptor direct example. "
            f"Check LLM_CLIENT_EXAMPLE_PG_DSN and database availability. {type(exc).__name__}: {exc}"
        )
        return
    try:
        readonly = PostgresSQLAdaptor(pool, read_only=True, default_timeout_seconds=5.0)
        database_info = await readonly.query(
            SQLQueryRequest(
                statement=(
                    "select current_database() as database_name, "
                    "current_user as database_user, "
                    ":scenario::text as scenario_note"
                ),
                parameters={"scenario": "sql adaptor direct example"},
            )
        )

        safety_error = None
        try:
            await readonly.execute(
                SQLMutationRequest(
                    statement="create temporary table llm_client_sql_adaptor_demo (id integer primary key, label text)",
                    allow_write=True,
                )
            )
        except SQLSafetyError as exc:
            safety_error = str(exc)

        async with pool.acquire() as conn:
            writable = PostgresSQLAdaptor(conn, read_only=False, default_timeout_seconds=5.0)
            await writable.execute(
                SQLMutationRequest(
                    statement="create temporary table llm_client_sql_adaptor_demo (id integer primary key, label text)",
                    allow_write=True,
                )
            )
            insert_result = await writable.execute(
                SQLMutationRequest(
                    statement="insert into llm_client_sql_adaptor_demo(id, label) values (:id, :label)",
                    parameters={"id": 1, "label": "router-failover-drill"},
                    allow_write=True,
                )
            )
            temp_rows = await writable.query(
                SQLQueryRequest(
                    statement="select id, label from llm_client_sql_adaptor_demo where id = :id",
                    parameters={"id": 1},
                )
            )

        print_heading("SQL Adaptor Direct")
        print_json(
            {
                "backend": readonly.backend_name,
                "dialect": readonly.dialect.value,
                "read_only_query": {
                    "row_count": database_info.row_count,
                    "rows": database_info.rows,
                    "metadata": database_info.metadata.metadata,
                },
                "safety_checks": {
                    "read_only_execute_error": safety_error,
                },
                "explicit_write_flow": {
                    "insert_result": {
                        "affected_count": insert_result.affected_count,
                        "metadata": insert_result.metadata.metadata,
                    },
                    "selected_rows": temp_rows.rows,
                },
            }
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
