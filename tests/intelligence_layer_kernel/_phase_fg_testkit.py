from __future__ import annotations

import copy
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from intelligence_layer_kernel.operators.types import AuthContext, OperatorCall, TraceContext


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_operator_call(
    payload: dict[str, Any],
    *,
    idempotency_key: str | None = None,
    tenant_id: int = 1,
    principal_id: int = 7,
) -> OperatorCall:
    return OperatorCall(
        payload=payload,
        idempotency_key=idempotency_key or str(uuid.uuid4()),
        auth_context=AuthContext(
            tenant_id=tenant_id,
            principal={"type": "student", "id": principal_id},
            scopes=["test"],
        ),
        trace_context=TraceContext(
            correlation_id=str(uuid.uuid4()),
            workflow_id=str(uuid.uuid4()),
            step_id="s1",
        ),
    )


class FakePlatformDB:
    def __init__(
        self,
        *,
        funding_requests: dict[int, dict[str, Any]],
        funding_emails: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.funding_requests: dict[int, dict[str, Any]] = {
            int(request_id): _normalize_request_row(request_id=int(request_id), row=row)
            for request_id, row in funding_requests.items()
        }
        self.funding_emails: dict[int, dict[str, Any]] = {
            int(email_id): _normalize_email_row(email_id=int(email_id), row=row)
            for email_id, row in (funding_emails or {}).items()
        }

    async def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        normalized = " ".join(sql.split()).lower()

        if "from funding_requests fr" in normalized:
            return self._fetch_funding_request_email_snapshot(normalized=normalized, params=params)

        if "from funding_requests" in normalized:
            if not params:
                return None
            request_id = int(params[-1])
            row = self.funding_requests.get(request_id)
            if row is None:
                return None
            return copy.deepcopy(row)

        return None

    async def pool(self) -> _FakePool:
        return _FakePool(self)

    def _fetch_funding_request_email_snapshot(
        self,
        *,
        normalized: str,
        params: tuple[Any, ...],
    ) -> dict[str, Any] | None:
        if "where fr.id=%s" not in normalized:
            return None

        email_id: int | None = None
        request_id: int
        if "on fe.id=%s and fe.funding_request_id=fr.id" in normalized:
            email_id = int(params[0])
            request_id = int(params[1])
        elif "join funding_emails fe" in normalized:
            email_id = int(params[0])
            request_id = int(params[1])
        else:
            request_id = int(params[0])
            email_id = self._latest_email_id_for_request(request_id)

        request_row = self.funding_requests.get(request_id)
        if request_row is None:
            return None
        email_row = self.funding_emails.get(email_id) if email_id is not None else None

        snapshot = {
            "request_id": request_row["id"],
            "request_email_subject": request_row.get("email_subject"),
            "request_email_body": request_row.get("email_content"),
            "request_updated_at": request_row.get("updated_at"),
            "email_id": email_row["id"] if isinstance(email_row, dict) else None,
            "main_email_subject": email_row.get("main_email_subject") if isinstance(email_row, dict) else None,
            "main_email_body": email_row.get("main_email_body") if isinstance(email_row, dict) else None,
            "main_sent": email_row.get("main_sent") if isinstance(email_row, dict) else 0,
            "main_sent_at": email_row.get("main_sent_at") if isinstance(email_row, dict) else None,
        }
        return copy.deepcopy(snapshot)

    def _latest_email_id_for_request(self, funding_request_id: int) -> int | None:
        matches = [
            email_id
            for email_id, row in self.funding_emails.items()
            if int(row.get("funding_request_id") or 0) == funding_request_id
        ]
        if not matches:
            return None
        return max(matches)


class _FakeAcquire:
    def __init__(self, db: FakePlatformDB) -> None:
        self._conn = _FakeConn(db)

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


class _FakePool:
    def __init__(self, db: FakePlatformDB) -> None:
        self._db = db

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._db)


class _FakeConn:
    def __init__(self, db: FakePlatformDB) -> None:
        self._db = db
        self._in_tx = False

    def cursor(self, *_args, **_kwargs) -> _FakeCursor:
        return _FakeCursor(self._db)

    async def begin(self) -> None:
        self._in_tx = True

    async def commit(self) -> None:
        self._in_tx = False

    async def rollback(self) -> None:
        self._in_tx = False


class _FakeCursor:
    def __init__(self, db: FakePlatformDB) -> None:
        self._db = db
        self.rowcount = 0

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False

    async def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        normalized = " ".join(sql.split()).lower()
        if normalized.startswith("update funding_requests set"):
            self.rowcount = self._update_funding_requests(sql=sql, params=params)
            return
        if normalized.startswith("update funding_emails set"):
            self.rowcount = self._update_funding_emails(sql=sql, params=params)
            return
        raise AssertionError(f"unexpected query: {sql}")

    def _update_funding_requests(self, *, sql: str, params: tuple[Any, ...]) -> int:
        fields = re.findall(r"`([a-z_]+)`=%s", sql)
        if not fields:
            fields = re.findall(
                r"\b(research_interest|paper_title|journal|year|research_connection|email_subject|email_content)\s*=\s*%s",
                sql,
                flags=re.IGNORECASE,
            )
            fields = [str(item).lower() for item in fields]
        if not fields:
            raise AssertionError(f"unable to parse request fields in sql: {sql}")

        values = list(params)
        request_id_index = len(fields)
        if len(values) <= request_id_index:
            raise AssertionError(f"missing request identifier in params: {params}")
        request_id = int(values[request_id_index])
        expected_updated_at = values[request_id_index + 1] if len(values) > request_id_index + 1 else None

        row = self._db.funding_requests.get(request_id)
        if row is None:
            return 0
        if expected_updated_at is not None and row.get("updated_at") != expected_updated_at:
            return 0

        for field_name, value in zip(fields, values[: len(fields)], strict=False):
            row[field_name] = value
        row["updated_at"] = row.get("updated_at", utc_now()) + timedelta(seconds=1)
        return 1

    def _update_funding_emails(self, *, sql: str, params: tuple[Any, ...]) -> int:
        fields = re.findall(r"\b(main_email_subject|main_email_body)\s*=\s*%s", sql, flags=re.IGNORECASE)
        fields = [str(item).lower() for item in fields]
        if not fields:
            raise AssertionError(f"unable to parse email fields in sql: {sql}")

        values = list(params)
        email_id = int(values[len(fields)])
        row = self._db.funding_emails.get(email_id)
        if row is None:
            return 0
        if "and main_sent=0" in " ".join(sql.split()).lower() and bool(int(row.get("main_sent") or 0)):
            return 0

        for field_name, value in zip(fields, values[: len(fields)], strict=False):
            row[field_name] = value
        return 1


def _normalize_request_row(*, request_id: int, row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "id": request_id,
        "research_interest": row.get("research_interest"),
        "paper_title": row.get("paper_title"),
        "journal": row.get("journal"),
        "year": row.get("year"),
        "research_connection": row.get("research_connection"),
        "email_subject": row.get("email_subject"),
        "email_content": row.get("email_content"),
        "updated_at": row.get("updated_at") if isinstance(row.get("updated_at"), datetime) else utc_now(),
    }
    return normalized


def _normalize_email_row(*, email_id: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": email_id,
        "funding_request_id": int(row.get("funding_request_id") or 0),
        "main_email_subject": row.get("main_email_subject"),
        "main_email_body": row.get("main_email_body"),
        "main_sent": int(row.get("main_sent") or 0),
        "main_sent_at": row.get("main_sent_at"),
    }
