from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
Request = pytest.importorskip("starlette.requests").Request

from intelligence_layer_api.auth import PlatformSessionAuthAdapter


class _FakePlatformDB:
    def __init__(self) -> None:
        self.session_row: dict | None = None
        self.owner_row: dict | None = None

    async def fetch_one(self, sql: str, params=()):
        _ = params
        if "FROM auth_sessions" in sql or "FROM oauth_sessions" in sql:
            return self.session_row
        if "FROM funding_requests" in sql:
            return self.owner_row
        return None


def _build_request(*, cookies: dict[str, str] | None = None, headers: dict[str, str] | None = None) -> Request:
    raw_headers: list[tuple[bytes, bytes]] = []
    if headers:
        for key, value in headers.items():
            raw_headers.append((key.lower().encode("latin-1"), str(value).encode("latin-1")))
    if cookies:
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        raw_headers.append((b"cookie", cookie_header.encode("latin-1")))
    scope = {"type": "http", "method": "GET", "path": "/", "headers": raw_headers}
    return Request(scope)


@pytest.mark.asyncio
async def test_platform_session_auth_accepts_valid_cookie_session_and_owner() -> None:
    db = _FakePlatformDB()
    db.session_row = {"principal_id": 77, "scopes": "[\"chat\"]", "trust_level": 2}
    db.owner_row = {"owner_id": 77}
    adapter = PlatformSessionAuthAdapter(
        platform_db=db,
        session_cookie_names=("session_id",),
    )

    auth = await adapter.authenticate(
        request=_build_request(cookies={"session_id": "sess-abc"}),
        funding_request_id=88,
    )

    assert auth.ok is True
    assert auth.principal_id == 77
    assert auth.scopes == ["chat"]
    assert auth.trust_level == 2
    assert auth.bypass is False


@pytest.mark.asyncio
async def test_platform_session_auth_rejects_idor_owner_mismatch() -> None:
    db = _FakePlatformDB()
    db.session_row = {"principal_id": 77, "scopes": "chat", "trust_level": 1}
    db.owner_row = {"owner_id": 99}
    adapter = PlatformSessionAuthAdapter(
        platform_db=db,
        session_cookie_names=("session_id",),
    )

    auth = await adapter.authenticate(
        request=_build_request(cookies={"session_id": "sess-abc"}),
        funding_request_id=88,
    )

    assert auth.ok is False
    assert auth.status_code == 403
    assert auth.reason == "forbidden"


@pytest.mark.asyncio
async def test_platform_session_auth_rejects_missing_session_cookie() -> None:
    db = _FakePlatformDB()
    adapter = PlatformSessionAuthAdapter(
        platform_db=db,
        session_cookie_names=("session_id",),
    )

    auth = await adapter.authenticate(
        request=_build_request(),
        funding_request_id=88,
    )

    assert auth.ok is False
    assert auth.status_code == 401
    assert auth.reason == "missing_session"
