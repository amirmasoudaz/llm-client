from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import pytest

fastapi = pytest.importorskip("fastapi")
Request = pytest.importorskip("starlette.requests").Request

from intelligence_layer_api.auth import PlatformSessionAuthAdapter, SanctumBearerAuthAdapter


class _FakePlatformDB:
    def __init__(self) -> None:
        self.session_row: dict | None = None
        self.owner_row: dict | None = None
        self.token_row: dict | None = None
        self.token_lookup_params: tuple | None = None
        self.touched_token_ids: list[int] = []

    async def fetch_one(self, sql: str, params=()):
        _ = params
        if "FROM auth_sessions" in sql or "FROM oauth_sessions" in sql:
            return self.session_row
        if "FROM personal_access_tokens" in sql:
            self.token_lookup_params = tuple(params)
            return self.token_row
        if "FROM funding_requests" in sql:
            return self.owner_row
        return None

    async def execute(self, sql: str, params=()):
        if "UPDATE personal_access_tokens" in sql:
            try:
                self.touched_token_ids.append(int(params[0]))
            except (IndexError, TypeError, ValueError):
                pass
        return 0


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


@pytest.mark.asyncio
async def test_sanctum_bearer_auth_accepts_valid_token_and_owner() -> None:
    db = _FakePlatformDB()
    db.token_row = {
        "token_id": 19,
        "principal_id": 77,
        "abilities": "[\"chat\"]",
        "expires_at": None,
    }
    db.owner_row = {"owner_id": 77}
    adapter = SanctumBearerAuthAdapter(
        platform_db=db,
        update_last_used_at=True,
    )

    auth = await adapter.authenticate(
        request=_build_request(headers={"Authorization": "Bearer 19|plain-text-token"}),
        funding_request_id=88,
    )

    assert auth.ok is True
    assert auth.principal_id == 77
    assert auth.scopes == ["chat"]
    assert auth.bypass is False
    assert db.token_lookup_params is not None
    assert db.token_lookup_params[0] == 19
    assert db.token_lookup_params[1] == hashlib.sha256("plain-text-token".encode("utf-8")).hexdigest()
    assert db.token_lookup_params[2] == "App\\Models\\Student\\Student"
    assert db.touched_token_ids == [19]


@pytest.mark.asyncio
async def test_sanctum_bearer_auth_rejects_missing_bearer_header() -> None:
    db = _FakePlatformDB()
    adapter = SanctumBearerAuthAdapter(platform_db=db)

    auth = await adapter.authenticate(
        request=_build_request(),
        funding_request_id=88,
    )

    assert auth.ok is False
    assert auth.status_code == 401
    assert auth.reason == "missing_bearer_token"


@pytest.mark.asyncio
async def test_sanctum_bearer_auth_rejects_malformed_token() -> None:
    db = _FakePlatformDB()
    adapter = SanctumBearerAuthAdapter(platform_db=db)

    auth = await adapter.authenticate(
        request=_build_request(headers={"Authorization": "Bearer malformedtoken"}),
        funding_request_id=88,
    )

    assert auth.ok is False
    assert auth.status_code == 401
    assert auth.reason == "invalid_token_format"


@pytest.mark.asyncio
async def test_sanctum_bearer_auth_rejects_expired_token() -> None:
    db = _FakePlatformDB()
    db.token_row = {
        "token_id": 19,
        "principal_id": 77,
        "abilities": "[\"chat\"]",
        "expires_at": datetime.utcnow() - timedelta(minutes=1),
    }
    adapter = SanctumBearerAuthAdapter(platform_db=db)

    auth = await adapter.authenticate(
        request=_build_request(headers={"Authorization": "Bearer 19|plain-text-token"}),
        funding_request_id=88,
    )

    assert auth.ok is False
    assert auth.status_code == 401
    assert auth.reason == "token_expired"


@pytest.mark.asyncio
async def test_sanctum_bearer_auth_rejects_owner_mismatch() -> None:
    db = _FakePlatformDB()
    db.token_row = {
        "token_id": 19,
        "principal_id": 77,
        "abilities": "[\"chat\"]",
        "expires_at": None,
    }
    db.owner_row = {"owner_id": 99}
    adapter = SanctumBearerAuthAdapter(platform_db=db)

    auth = await adapter.authenticate(
        request=_build_request(headers={"Authorization": "Bearer 19|plain-text-token"}),
        funding_request_id=88,
    )

    assert auth.ok is False
    assert auth.status_code == 403
    assert auth.reason == "forbidden"
