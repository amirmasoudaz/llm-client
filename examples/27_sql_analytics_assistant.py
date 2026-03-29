from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.agent import Agent, AgentDefinition, AgentExecutionPolicy, ToolExecutionMode
from llm_client.engine import ExecutionEngine
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import Message, StreamEventType, ToolCall, ToolCallDelta
from llm_client.spec import RequestContext
from llm_client.structured import StructuredOutputConfig, extract_structured
from llm_client.tools import Tool, ToolResult


ANALYTICS_SCOPE = "sql-analytics-assistant"
REFERENCE_NOW = datetime(2026, 3, 23, 12, 0, 0)
QUESTION = (
    "Prepare a finance-and-ops briefing on which region/channel combinations are seeing the worst "
    "paid-order confirmation drag from webhook latency over the last 7 days, quantify revenue at risk, "
    "and recommend the next actions."
)
BRIEFING_PACKET = {
    "company": "Northstar Commerce",
    "analysis_scope": "paid-order confirmation reliability",
    "objective": "identify where webhook latency is degrading payment-confirmation workflows",
    "audience": ["finance", "payments-ops", "support leadership"],
    "time_window_days": 7,
    "revenue_at_risk_rule": "count paid orders as at-risk when payment.captured webhook latency >= 1800 ms or retry_count >= 2",
    "constraints": [
        "read-only SQL only",
        "use only the provided SQLite schema",
        "ground conclusions in executed queries or deterministic tool outputs",
    ],
}

METRIC_DICTIONARY = {
    "revenue_at_risk_usd": (
        "Sum of order amount_usd for captured payments whose payment.captured webhook latency is at least 1800 ms "
        "or whose retry_count is at least 2."
    ),
    "delayed_orders": "Count of captured orders where payment.captured webhook latency >= 1800 ms or retry_count >= 2.",
    "support_case_count": "Count of linked support cases opened against an order in the analysis window.",
    "confirmation_gap_minutes": (
        "Minutes between payment capture and payment.captured webhook delivery, derived from delivery_latency_ms."
    ),
}

ALLOWED_TABLES = {"orders", "payments", "webhook_events", "support_cases"}
MAX_RESULT_ROWS = 12


@dataclass
class DatabaseSandbox:
    connection: sqlite3.Connection
    schema_context: dict[str, Any]
    table_counts: dict[str, int]


def _truncate(value: Any, max_chars: int = 220) -> str:
    text = str(value)
    return text if len(text) <= max_chars else f"{text[:max_chars].rstrip()}..."


def _serialize_tool_calls(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [{"tool_name": call.name, "arguments": call.parse_arguments()} for call in tool_calls]


def _serialize_tool_results(tool_results: list[ToolResult]) -> list[dict[str, Any]]:
    return [
        {
            "success": result.success,
            "error": result.error,
            "content_preview": _truncate(result.to_string(), 280),
        }
        for result in tool_results
    ]


def _serialize_turns(turns: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "turn_number": turn.turn_number + 1,
            "assistant_preview": _truncate(turn.content or "", 300),
            "tool_calls": _serialize_tool_calls(turn.tool_calls),
            "tool_results": _serialize_tool_results(turn.tool_results),
        }
        for turn in turns
    ]


def _seed_orders() -> list[dict[str, Any]]:
    seed_rows = [
        ("us-east", "web", 14, 620.0, 930, 0, 0.93),
        ("eu-west", "marketplace", 11, 910.0, 2140, 2, 0.88),
        ("ap-south", "api", 9, 1280.0, 2675, 3, 0.86),
        ("us-west", "partner", 10, 790.0, 760, 0, 0.95),
        ("latam", "web", 8, 540.0, 1710, 1, 0.90),
    ]
    rows: list[dict[str, Any]] = []
    order_num = 1001
    for region, channel, count, base_amount, latency_ms, retry_count, capture_ratio in seed_rows:
        for idx in range(count):
            created_at = REFERENCE_NOW - timedelta(hours=6 * idx + (order_num % 5))
            amount = round(base_amount + (idx % 4) * 45 + (order_num % 3) * 20, 2)
            captured = idx / count < capture_ratio
            payment_state = "captured" if captured else "failed"
            order_status = "paid" if captured else "payment_failed"
            payment_provider = "adyen" if channel in {"marketplace", "partner"} else "stripe"
            webhook_status = "delivered" if captured else "not_sent"
            delivery_latency_ms = latency_ms + (idx % 3) * 155 if captured else None
            retries = retry_count + (1 if captured and idx % 5 == 0 and retry_count else 0) if captured else 0
            support_case_reason = None
            support_case_severity = None
            if captured and delivery_latency_ms and (delivery_latency_ms >= 1800 or retries >= 2) and idx % 2 == 0:
                support_case_reason = "confirmation_delay"
                support_case_severity = "high" if delivery_latency_ms >= 2400 else "medium"
            rows.append(
                {
                    "order_id": f"ord_{order_num}",
                    "created_at": created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "customer_region": region,
                    "sales_channel": channel,
                    "amount_usd": amount,
                    "order_status": order_status,
                    "payment_provider": payment_provider,
                    "payment_state": payment_state,
                    "captured_at": (created_at + timedelta(minutes=4)).strftime("%Y-%m-%dT%H:%M:%SZ") if captured else None,
                    "webhook_latency_ms": delivery_latency_ms,
                    "retry_count": retries,
                    "delivery_status": webhook_status,
                    "support_case_reason": support_case_reason,
                    "support_case_severity": support_case_severity,
                }
            )
            order_num += 1
    return rows


def _build_database() -> DatabaseSandbox:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.executescript(
        """
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            customer_region TEXT NOT NULL,
            sales_channel TEXT NOT NULL,
            amount_usd REAL NOT NULL,
            order_status TEXT NOT NULL
        );

        CREATE TABLE payments (
            payment_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            payment_state TEXT NOT NULL,
            captured_at TEXT,
            amount_usd REAL NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        );

        CREATE TABLE webhook_events (
            event_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            delivered_at TEXT,
            delivery_latency_ms INTEGER,
            retry_count INTEGER NOT NULL,
            delivery_status TEXT NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        );

        CREATE TABLE support_cases (
            case_id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            opened_at TEXT NOT NULL,
            reason TEXT NOT NULL,
            severity TEXT NOT NULL,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        );
        """
    )

    seeded = _seed_orders()
    for idx, row in enumerate(seeded, start=1):
        cursor.execute(
            """
            INSERT INTO orders(order_id, created_at, customer_region, sales_channel, amount_usd, order_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["order_id"],
                row["created_at"],
                row["customer_region"],
                row["sales_channel"],
                row["amount_usd"],
                row["order_status"],
            ),
        )
        cursor.execute(
            """
            INSERT INTO payments(payment_id, order_id, provider, payment_state, captured_at, amount_usd)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"pay_{idx}",
                row["order_id"],
                row["payment_provider"],
                row["payment_state"],
                row["captured_at"],
                row["amount_usd"],
            ),
        )
        cursor.execute(
            """
            INSERT INTO webhook_events(event_id, order_id, event_type, delivered_at, delivery_latency_ms, retry_count, delivery_status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"evt_{idx}",
                row["order_id"],
                "payment.captured",
                (datetime.strptime(row["captured_at"], "%Y-%m-%dT%H:%M:%SZ") + timedelta(milliseconds=row["webhook_latency_ms"])).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                if row["captured_at"] and row["webhook_latency_ms"] is not None
                else None,
                row["webhook_latency_ms"],
                row["retry_count"],
                row["delivery_status"],
            ),
        )
        if row["support_case_reason"]:
            opened_at = datetime.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%SZ") + timedelta(hours=2)
            cursor.execute(
                """
                INSERT INTO support_cases(case_id, order_id, opened_at, reason, severity)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"case_{idx}",
                    row["order_id"],
                    opened_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    row["support_case_reason"],
                    row["support_case_severity"],
                ),
            )

    connection.commit()

    schema_context: dict[str, Any] = {"tables": {}}
    table_counts: dict[str, int] = {}
    for table_name in sorted(ALLOWED_TABLES):
        columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        schema_context["tables"][table_name] = [
            {
                "name": column["name"],
                "type": column["type"],
                "not_null": bool(column["notnull"]),
                "primary_key": bool(column["pk"]),
            }
            for column in columns
        ]
        table_counts[table_name] = int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])

    return DatabaseSandbox(connection=connection, schema_context=schema_context, table_counts=table_counts)


def _analysis_lower_bound(window_days: int) -> str:
    return (REFERENCE_NOW - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _recommended_sql(window_days: int = 7, limit: int = 6) -> str:
    lower_bound = _analysis_lower_bound(window_days)
    safe_limit = min(max(limit, 1), MAX_RESULT_ROWS)
    return (
        "SELECT\n"
        "    o.customer_region,\n"
        "    o.sales_channel,\n"
        "    COUNT(*) AS captured_orders,\n"
        "    SUM(CASE WHEN w.delivery_latency_ms >= 1800 OR w.retry_count >= 2 THEN 1 ELSE 0 END) AS delayed_orders,\n"
        "    ROUND(SUM(CASE WHEN w.delivery_latency_ms >= 1800 OR w.retry_count >= 2 THEN o.amount_usd ELSE 0 END), 2) AS revenue_at_risk_usd,\n"
        "    ROUND(AVG(w.delivery_latency_ms), 1) AS avg_webhook_latency_ms,\n"
        "    COUNT(DISTINCT s.case_id) AS support_case_count\n"
        "FROM orders o\n"
        "JOIN payments p ON p.order_id = o.order_id AND p.payment_state = 'captured'\n"
        "JOIN webhook_events w ON w.order_id = o.order_id AND w.event_type = 'payment.captured'\n"
        "LEFT JOIN support_cases s ON s.order_id = o.order_id\n"
        f"WHERE o.created_at >= '{lower_bound}'\n"
        "GROUP BY o.customer_region, o.sales_channel\n"
        "ORDER BY revenue_at_risk_usd DESC, avg_webhook_latency_ms DESC\n"
        f"LIMIT {safe_limit}"
    )


def _kpi_snapshot_data(connection: sqlite3.Connection, window_days: int = 7) -> dict[str, Any]:
    lower_bound = _analysis_lower_bound(window_days)
    summary = connection.execute(
        """
        SELECT
            COUNT(*) AS captured_orders,
            ROUND(SUM(o.amount_usd), 2) AS captured_revenue_usd,
            ROUND(AVG(w.delivery_latency_ms), 1) AS avg_webhook_latency_ms,
            SUM(CASE WHEN w.delivery_latency_ms >= 1800 OR w.retry_count >= 2 THEN 1 ELSE 0 END) AS delayed_orders,
            ROUND(SUM(CASE WHEN w.delivery_latency_ms >= 1800 OR w.retry_count >= 2 THEN o.amount_usd ELSE 0 END), 2) AS revenue_at_risk_usd,
            COUNT(DISTINCT s.case_id) AS support_case_count
        FROM orders o
        JOIN payments p ON p.order_id = o.order_id AND p.payment_state = 'captured'
        JOIN webhook_events w ON w.order_id = o.order_id AND w.event_type = 'payment.captured'
        LEFT JOIN support_cases s ON s.order_id = o.order_id
        WHERE o.created_at >= ?
        """,
        (lower_bound,),
    ).fetchone()
    hotspots = connection.execute(
        """
        SELECT
            o.customer_region,
            o.sales_channel,
            COUNT(*) AS captured_orders,
            ROUND(SUM(CASE WHEN w.delivery_latency_ms >= 1800 OR w.retry_count >= 2 THEN o.amount_usd ELSE 0 END), 2) AS revenue_at_risk_usd,
            ROUND(AVG(w.delivery_latency_ms), 1) AS avg_webhook_latency_ms
        FROM orders o
        JOIN payments p ON p.order_id = o.order_id AND p.payment_state = 'captured'
        JOIN webhook_events w ON w.order_id = o.order_id AND w.event_type = 'payment.captured'
        WHERE o.created_at >= ?
        GROUP BY o.customer_region, o.sales_channel
        ORDER BY revenue_at_risk_usd DESC, avg_webhook_latency_ms DESC
        LIMIT 3
        """,
        (lower_bound,),
    ).fetchall()
    return {
        "window_days": window_days,
        "summary": dict(summary) if summary else {},
        "top_hotspots": [dict(row) for row in hotspots],
        "thresholds": {
            "delayed_order_latency_ms": 1800,
            "retry_count_alert": 2,
        },
    }


def _validate_sql(candidate_sql: str) -> dict[str, Any]:
    sql = candidate_sql.strip()
    normalized = re.sub(r"\s+", " ", sql.lower())
    issues: list[str] = []

    if not sql:
        issues.append("SQL is empty.")
    if not (normalized.startswith("select") or normalized.startswith("with")):
        issues.append("Only SELECT or WITH queries are allowed.")
    if ";" in sql[:-1]:
        issues.append("Multiple statements are not allowed.")

    forbidden = (
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "truncate ",
        "attach ",
        "detach ",
        "create ",
        "replace ",
        "pragma ",
        "vacuum ",
    )
    matched_forbidden = [token.strip() for token in forbidden if token in normalized]
    if matched_forbidden:
        issues.append(f"Forbidden SQL detected: {matched_forbidden}.")

    table_refs = re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", normalized)
    unknown_tables = sorted({table for table in table_refs if table not in ALLOWED_TABLES})
    if unknown_tables:
        issues.append(f"References tables outside the approved schema: {unknown_tables}.")

    limit_match = re.search(r"\blimit\s+(\d+)\b", normalized)
    has_limit = limit_match is not None
    limit_value = int(limit_match.group(1)) if limit_match else None
    if has_limit and limit_value is not None and limit_value > MAX_RESULT_ROWS:
        issues.append(f"LIMIT must be <= {MAX_RESULT_ROWS} for this sandbox.")

    return {
        "sql": sql,
        "normalized_sql": normalized,
        "read_only": not matched_forbidden and (normalized.startswith("select") or normalized.startswith("with")),
        "allowed_tables_only": not unknown_tables,
        "has_limit": has_limit,
        "limit_value": limit_value,
        "referenced_tables": sorted(set(table_refs)),
        "issues": issues,
        "safe_to_run": not issues,
    }


def _execute_query(connection: sqlite3.Connection, sql: str) -> dict[str, Any]:
    plan_rows = connection.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    data_rows = connection.execute(sql).fetchmany(MAX_RESULT_ROWS)
    columns = [item[0] for item in connection.execute(sql).description or []]
    serialized_rows = [dict(zip(columns, row)) for row in data_rows]
    return {
        "columns": columns,
        "rows": serialized_rows,
        "row_count": len(serialized_rows),
        "query_plan": [dict(row) for row in plan_rows],
    }


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    seed_entries = [
        MemoryWrite(
            scope=ANALYTICS_SCOPE,
            content="Finance wants absolute revenue-at-risk figures first, then the operational cause behind them.",
            relevance=0.94,
            metadata={"kind": "finance_preference"},
        ),
        MemoryWrite(
            scope=ANALYTICS_SCOPE,
            content="Payments Ops prefers region/channel slices and wants recommended follow-up experiments, not only diagnosis.",
            relevance=0.92,
            metadata={"kind": "ops_preference"},
        ),
        MemoryWrite(
            scope=ANALYTICS_SCOPE,
            content="Support leadership wants confirmation-lag conclusions tied to support-case pressure wherever possible.",
            relevance=0.91,
            metadata={"kind": "support_signal"},
        ),
        MemoryWrite(
            scope=ANALYTICS_SCOPE,
            content="Leadership dislikes invented causal claims; caveats and open questions should be explicit when evidence is incomplete.",
            relevance=0.95,
            metadata={"kind": "communication_rule"},
        ),
    ]
    written: list[dict[str, Any]] = []
    for entry in seed_entries:
        record = await memory.write(entry)
        written.append({"kind": record.metadata.get("kind"), "content": record.content})
    return written


def _build_tools(
    sandbox: DatabaseSandbox,
    memory: ShortTermMemoryStore,
    query_audit: list[dict[str, Any]],
    guardrail_audit: list[dict[str, Any]],
) -> list[Tool]:
    connection = sandbox.connection

    async def schema_catalog(table_name: str | None = None) -> dict[str, Any]:
        if table_name:
            if table_name not in ALLOWED_TABLES:
                return {"error": f"Unknown table '{table_name}'."}
            return {
                "table_name": table_name,
                "columns": sandbox.schema_context["tables"][table_name],
                "row_count": sandbox.table_counts[table_name],
            }
        return {
            "tables": sandbox.schema_context["tables"],
            "table_counts": sandbox.table_counts,
            "join_hints": [
                "orders.order_id = payments.order_id",
                "orders.order_id = webhook_events.order_id",
                "orders.order_id = support_cases.order_id",
            ],
        }

    async def metric_dictionary(metric_name: str | None = None) -> dict[str, Any]:
        if metric_name:
            return {"metric_name": metric_name, "definition": METRIC_DICTIONARY.get(metric_name)}
        return {"metrics": METRIC_DICTIONARY}

    async def sample_rows(table_name: str, limit: int = 3) -> dict[str, Any]:
        if table_name not in ALLOWED_TABLES:
            return {"error": f"Unknown table '{table_name}'."}
        safe_limit = min(max(limit, 1), 5)
        rows = connection.execute(f"SELECT * FROM {table_name} LIMIT {safe_limit}").fetchall()
        return {"table_name": table_name, "rows": [dict(row) for row in rows]}

    async def kpi_snapshot(window_days: int = 7) -> dict[str, Any]:
        return _kpi_snapshot_data(connection, window_days=window_days)

    async def analyst_memory(topic: str) -> dict[str, Any]:
        records = await memory.retrieve(MemoryQuery(scope=ANALYTICS_SCOPE, limit=4))
        return {
            "topic": topic,
            "notes": [{"kind": record.metadata.get("kind"), "content": record.content} for record in records],
        }

    async def query_blueprint(question: str, window_days: int = 7) -> dict[str, Any]:
        sql = _recommended_sql(window_days=window_days, limit=6)
        return {
            "question": question,
            "window_days": window_days,
            "sql": sql,
            "expected_columns": [
                "customer_region",
                "sales_channel",
                "captured_orders",
                "delayed_orders",
                "revenue_at_risk_usd",
                "avg_webhook_latency_ms",
                "support_case_count",
            ],
            "rationale": [
                "Filter to captured payments and payment.captured webhook events in the requested time window.",
                "Compute delayed_orders and revenue_at_risk_usd using the explicit latency/retry rule from the briefing packet.",
                "Group by region and channel to surface finance-and-ops hotspots.",
                "Rank by revenue_at_risk_usd first and avg_webhook_latency_ms second.",
            ],
        }

    async def sql_guardrails(candidate_sql: str) -> dict[str, Any]:
        report = _validate_sql(candidate_sql)
        report["recommended_fix"] = (
            None
            if report["safe_to_run"]
            else "Revise to a single read-only SELECT/WITH query over approved tables and keep LIMIT <= 12."
        )
        guardrail_audit.append(report)
        return report

    async def execute_sql(candidate_sql: str) -> ToolResult:
        report = _validate_sql(candidate_sql)
        guardrail_audit.append(report)
        if not report["safe_to_run"]:
            query_audit.append({"sql": candidate_sql, "status": "rejected", "guardrails": report})
            return ToolResult.error_result(json.dumps(report, ensure_ascii=True, sort_keys=True))
        execution = _execute_query(connection, report["sql"])
        audit_entry = {
            "sql": report["sql"],
            "status": "executed",
            "guardrails": report,
            "query_plan": execution["query_plan"],
            "columns": execution["columns"],
            "row_count": execution["row_count"],
            "rows": execution["rows"],
        }
        query_audit.append(audit_entry)
        return ToolResult.success_result(
            {
                "sql": report["sql"],
                "guardrails": report,
                "query_plan": execution["query_plan"],
                "columns": execution["columns"],
                "row_count": execution["row_count"],
                "rows": execution["rows"],
            }
        )

    return [
        Tool(
            name="schema_catalog",
            description="Inspect the analytics schema, row counts, and join hints for approved tables.",
            parameters={
                "type": "object",
                "properties": {"table_name": {"type": "string"}},
                "additionalProperties": False,
            },
            handler=schema_catalog,
        ),
        Tool(
            name="metric_dictionary",
            description="Explain how analytics metrics like revenue_at_risk_usd and delayed_orders are defined.",
            parameters={
                "type": "object",
                "properties": {"metric_name": {"type": "string"}},
                "additionalProperties": False,
            },
            handler=metric_dictionary,
        ),
        Tool(
            name="sample_rows",
            description="Preview example rows from a table before writing SQL.",
            parameters={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["table_name"],
                "additionalProperties": False,
            },
            handler=sample_rows,
        ),
        Tool(
            name="kpi_snapshot",
            description="Return deterministic KPI summaries and hotspot slices for the last N days.",
            parameters={
                "type": "object",
                "properties": {"window_days": {"type": "integer", "minimum": 1, "maximum": 30}},
                "additionalProperties": False,
            },
            handler=kpi_snapshot,
        ),
        Tool(
            name="analyst_memory",
            description="Retrieve analyst notes about audience preferences, communication constraints, and prior guidance.",
            parameters={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
                "additionalProperties": False,
            },
            handler=analyst_memory,
        ),
        Tool(
            name="query_blueprint",
            description="Return a compliant SQL blueprint for the requested analytics question, including expected columns and rationale.",
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "window_days": {"type": "integer", "minimum": 1, "maximum": 30},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
            handler=query_blueprint,
        ),
        Tool(
            name="sql_guardrails",
            description="Validate whether a candidate SQL query is safe and compliant for the sandbox.",
            parameters={
                "type": "object",
                "properties": {"candidate_sql": {"type": "string"}},
                "required": ["candidate_sql"],
                "additionalProperties": False,
            },
            handler=sql_guardrails,
        ),
        Tool(
            name="execute_sql",
            description="Execute a safe read-only SQL query and return query plan plus bounded result rows.",
            parameters={
                "type": "object",
                "properties": {"candidate_sql": {"type": "string"}},
                "required": ["candidate_sql"],
                "additionalProperties": False,
            },
            handler=execute_sql,
        ),
    ]


async def _run_agent_stream(agent: Agent, prompt: str, context: RequestContext) -> tuple[Any, dict[str, Any]]:
    event_counts: Counter[str] = Counter()
    token_preview_parts: list[str] = []
    tool_call_events: list[dict[str, Any]] = []
    tool_result_events: list[dict[str, Any]] = []
    meta_events: list[dict[str, Any]] = []
    usage_events: list[dict[str, Any]] = []
    final_result: Any = None

    async for event in agent.stream(prompt, context=context):
        event_counts[event.type.value] += 1

        if event.type == StreamEventType.TOKEN:
            if sum(len(part) for part in token_preview_parts) < 300:
                token_preview_parts.append(str(event.data))
            continue

        if event.type in {StreamEventType.TOOL_CALL_START, StreamEventType.TOOL_CALL_DELTA, StreamEventType.TOOL_CALL_END}:
            payload = event.data
            if isinstance(payload, ToolCallDelta):
                tool_call_events.append(
                    {
                        "event": event.type.value,
                        "tool_name": payload.name,
                        "arguments_delta": _truncate(payload.arguments_delta, 180),
                    }
                )
            elif isinstance(payload, ToolCall):
                tool_call_events.append(
                    {
                        "event": event.type.value,
                        "tool_name": payload.name,
                        "arguments": payload.parse_arguments(),
                    }
                )
            continue

        if event.type == StreamEventType.META:
            payload = event.data if isinstance(event.data, dict) else {"value": str(event.data)}
            if payload.get("event") == "tool_result":
                tool_result_events.append(
                    {
                        "tool_name": payload.get("tool_name"),
                        "success": payload.get("success"),
                        "content_preview": _truncate(payload.get("content"), 240),
                    }
                )
            else:
                meta_events.append(payload)
            continue

        if event.type == StreamEventType.USAGE and hasattr(event.data, "to_dict"):
            usage_events.append(event.data.to_dict())
            continue

        if event.type == StreamEventType.DONE:
            final_result = event.data

    if final_result is None:
        raise RuntimeError("Agent stream completed without a final result.")

    return final_result, {
        "event_type_counts": dict(event_counts),
        "token_preview": "".join(token_preview_parts).strip(),
        "tool_call_events": tool_call_events,
        "tool_result_events": tool_result_events,
        "meta_events": meta_events,
        "usage_events": usage_events,
    }


def _assembled_summary(structured_data: dict[str, Any] | None) -> str:
    if not structured_data:
        return ""
    segments = "\n".join(
        [
            (
                f"- {item['customer_region']} / {item['sales_channel']}: "
                f"revenue_at_risk_usd={item['revenue_at_risk_usd']}, "
                f"avg_webhook_latency_ms={item['avg_webhook_latency_ms']}, "
                f"support_case_count={item['support_case_count']}"
            )
            for item in structured_data.get("top_segments", [])
        ]
    )
    actions = "\n".join(f"- {item}" for item in structured_data.get("recommended_actions", []))
    caveats = "\n".join(f"- {item}" for item in structured_data.get("caveats", []))
    return (
        f"Executive Takeaway\n- {structured_data.get('executive_takeaway', '')}\n\n"
        f"SQL Safety\n- {structured_data.get('sql_safety_summary', '')}\n\n"
        f"Top Segments\n{segments}\n\n"
        f"Recommended Actions\n{actions}\n\n"
        f"Caveats\n{caveats}"
    ).strip()


async def main() -> None:
    handle = build_live_provider()
    sandbox = _build_database()
    try:
        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)
        query_audit: list[dict[str, Any]] = []
        guardrail_audit: list[dict[str, Any]] = []
        recovery_applied = False
        deterministic_kpi_snapshot = _kpi_snapshot_data(sandbox.connection, window_days=BRIEFING_PACKET["time_window_days"])

        lifecycle = LifecycleRecorder()
        diagnostics = EngineDiagnosticsRecorder()
        hooks = HookManager([lifecycle, diagnostics])
        engine = ExecutionEngine(provider=handle.provider, hooks=hooks)

        tools = _build_tools(sandbox, memory, query_audit, guardrail_audit)
        agent = Agent(
            engine=engine,
            definition=AgentDefinition(
                name="sql-analytics-assistant",
                system_message=(
                    "You are a senior SQL analytics assistant for finance and operations. "
                    "Before finalizing, use at least five distinct tools, including query_blueprint, sql_guardrails, and execute_sql. "
                    "Ground every conclusion in executed SQL or deterministic tool outputs. "
                    "Do not invent columns, segments, or causal claims that are not supported by the schema and results. "
                    "Never send placeholder text to sql_guardrails or execute_sql. "
                    "Use query_blueprint to get a concrete query, validate that exact SQL with sql_guardrails, and if safe_to_run is true execute the same SQL. "
                    "If execute_sql has not succeeded, you are not done. "
                    "If details are missing, mark them as caveats or open questions. "
                    "Return sections: Executive Takeaway, Business Question, SQL Strategy, Findings, Query Evidence, Caveats, Recommended Actions."
                ),
                execution_policy=AgentExecutionPolicy(
                    max_turns=7,
                    max_tool_calls_per_turn=8,
                    tool_execution_mode=ToolExecutionMode.PARALLEL,
                    stop_on_tool_error=False,
                ),
            ),
            tools=tools,
        )

        prompt = (
            f"Analytics packet: {BRIEFING_PACKET}\n\n"
            f"Schema context: {sandbox.schema_context}\n\n"
            f"Question: {QUESTION}\n\n"
            "Build a finance-and-ops briefing. Use the tools to inspect schema and metrics, validate SQL, execute SQL, "
            "and then explain the findings in business terms. "
            "Use query_blueprint instead of inventing SQL from scratch if you need a compliant starting point."
        )
        request_context = RequestContext(
            session_id="cookbook-sql-analytics",
            job_id="sql-analytics-brief",
            tags={"analysis_scope": BRIEFING_PACKET["analysis_scope"]},
        )
        result, stream_summary = await _run_agent_stream(agent, prompt, request_context)

        if not any(item.get("status") == "executed" for item in query_audit):
            recovery_sql = _recommended_sql(window_days=BRIEFING_PACKET["time_window_days"], limit=6)
            recovery_guardrail = _validate_sql(recovery_sql)
            recovery_guardrail["recommended_fix"] = (
                None
                if recovery_guardrail["safe_to_run"]
                else "Revise to a single read-only SELECT/WITH query over approved tables and keep LIMIT <= 12."
            )
            recovery_guardrail["source"] = "auto_recovery"
            guardrail_audit.append(recovery_guardrail)
            if recovery_guardrail["safe_to_run"]:
                recovery_execution = _execute_query(sandbox.connection, recovery_sql)
                query_audit.append(
                    {
                        "sql": recovery_sql,
                        "status": "executed",
                        "source": "auto_recovery",
                        "guardrails": recovery_guardrail,
                        "query_plan": recovery_execution["query_plan"],
                        "columns": recovery_execution["columns"],
                        "row_count": recovery_execution["row_count"],
                        "rows": recovery_execution["rows"],
                    }
                )
                recovery_applied = True

        structured = await extract_structured(
            handle.provider,
            [
                Message.system(
                    (
                        "Convert the SQL analytics briefing into a structured analyst packet. "
                        "Keep findings grounded in the executed SQL audit and deterministic KPI outputs. "
                        "Do not invent segments or operational recommendations beyond the evidence."
                    )
                ),
                Message.user(
                    json.dumps(
                        {
                            "briefing_packet": BRIEFING_PACKET,
                            "agent_output": result.content,
                            "query_audit": query_audit,
                            "kpi_snapshot": deterministic_kpi_snapshot,
                            "guardrail_audit": guardrail_audit[-4:],
                            "recovery_applied": recovery_applied,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "executive_takeaway": {"type": "string"},
                        "sql_to_run": {"type": "string"},
                        "sql_safety_summary": {"type": "string"},
                        "top_segments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "customer_region": {"type": "string"},
                                    "sales_channel": {"type": "string"},
                                    "captured_orders": {"type": "integer"},
                                    "delayed_orders": {"type": "integer"},
                                    "revenue_at_risk_usd": {"type": "number"},
                                    "avg_webhook_latency_ms": {"type": "number"},
                                    "support_case_count": {"type": "integer"},
                                    "interpretation": {"type": "string"},
                                },
                                "required": [
                                    "customer_region",
                                    "sales_channel",
                                    "captured_orders",
                                    "delayed_orders",
                                    "revenue_at_risk_usd",
                                    "avg_webhook_latency_ms",
                                    "support_case_count",
                                    "interpretation",
                                ],
                                "additionalProperties": False,
                            },
                        },
                        "business_risks": {"type": "array", "items": {"type": "string"}},
                        "recommended_actions": {"type": "array", "items": {"type": "string"}},
                        "caveats": {"type": "array", "items": {"type": "string"}},
                        "evidence_used": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "executive_takeaway",
                        "sql_to_run",
                        "sql_safety_summary",
                        "top_segments",
                        "business_risks",
                        "recommended_actions",
                        "caveats",
                        "evidence_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
        )

        assembled_summary = _assembled_summary(structured.data)
        await memory.write(
            MemoryWrite(
                scope=ANALYTICS_SCOPE,
                content=json.dumps(structured.data, ensure_ascii=True, sort_keys=True),
                relevance=0.95,
                metadata={"kind": "analyst_packet"},
            )
        )
        memory_after = await memory.retrieve(MemoryQuery(scope=ANALYTICS_SCOPE, limit=6))

        latest_request_report = list(lifecycle.requests.values())[-1] if lifecycle.requests else None
        latest_session_report = lifecycle.sessions.get(request_context.session_id or "")
        distinct_tool_names = sorted({call.name for call in result.all_tool_calls})

        print_heading("SQL Analytics Assistant")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "question": QUESTION,
                "briefing_packet": BRIEFING_PACKET,
                "database_snapshot": {
                    "table_counts": sandbox.table_counts,
                    "schema_context": sandbox.schema_context,
                    "metric_dictionary": METRIC_DICTIONARY,
                },
                "memory_bootstrap": memory_bootstrap,
                "tool_catalog": [{"name": tool.name, "description": tool.description} for tool in tools],
                "stream_summary": stream_summary,
                "agent_result": {
                    "status": result.status,
                    "num_turns": result.num_turns,
                    "tool_names_used": distinct_tool_names,
                    "turns": _serialize_turns(result.turns),
                    "usage": summarize_usage(result.total_usage),
                    "final_content": result.content,
                },
                "query_audit": query_audit,
                "guardrail_audit": guardrail_audit,
                "recovery_applied": recovery_applied,
                "structured_packet": {
                    "valid": structured.valid,
                    "repair_attempts": structured.repair_attempts,
                    "usage": summarize_usage(getattr(structured, "usage", None)),
                    "data": structured.data,
                },
                "assembled_summary": assembled_summary,
                "observability": {
                    "hook_event_counts": dict(Counter(event for event, _, _ in diagnostics.events)),
                    "lifecycle_event_counts": dict(Counter(event.type.value for event in lifecycle.events)),
                    "latest_request_report": latest_request_report.to_dict() if latest_request_report else None,
                    "latest_session_report": latest_session_report.to_dict() if latest_session_report else None,
                },
                "memory_after_action": [
                    {"kind": record.metadata.get("kind"), "content": record.content}
                    for record in memory_after
                ],
                "showcase_verdict": {
                    "streamed_agent_run": bool(stream_summary["event_type_counts"]),
                    "used_five_plus_tools": len(distinct_tool_names) >= 5,
                    "executed_guarded_sql": any(item.get("status") == "executed" for item in query_audit),
                    "memory_backed": any(record.metadata.get("kind") == "analyst_packet" for record in memory_after),
                    "structured_packet_ready": structured.valid and bool(structured.data and structured.data.get("top_segments")),
                },
            }
        )
    finally:
        sandbox.connection.close()
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
