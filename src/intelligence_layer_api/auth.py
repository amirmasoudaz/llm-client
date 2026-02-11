from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import Request

from intelligence_layer_ops.platform_db import PlatformDB, PlatformDBConfig
from intelligence_layer_ops.platform_tools import platform_load_funding_thread_context


_DEFAULT_SESSION_QUERIES: tuple[str, ...] = (
    """
    SELECT user_id AS principal_id,
           scopes AS scopes,
           trust_level AS trust_level
    FROM auth_sessions
    WHERE session_id=%s
      AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP())
      AND (revoked_at IS NULL)
    LIMIT 1;
    """,
    """
    SELECT student_id AS principal_id,
           scopes_json AS scopes,
           trust_level AS trust_level
    FROM auth_sessions
    WHERE session_id=%s
      AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP())
      AND (revoked_at IS NULL)
    LIMIT 1;
    """,
    """
    SELECT user_id AS principal_id
    FROM oauth_sessions
    WHERE state=%s
    LIMIT 1;
    """,
)

_DEFAULT_OWNER_QUERIES: tuple[str, ...] = (
    "SELECT student_id AS owner_id FROM funding_requests WHERE id=%s LIMIT 1;",
    "SELECT user_id AS owner_id FROM funding_requests WHERE id=%s LIMIT 1;",
)

_DEFAULT_SANCTUM_TOKEN_QUERY = """
SELECT pat.id AS token_id,
       pat.tokenable_id AS principal_id,
       pat.abilities AS abilities,
       pat.expires_at AS expires_at
FROM personal_access_tokens pat
INNER JOIN students s ON s.id = pat.tokenable_id
WHERE pat.id = %s
  AND pat.token = %s
  AND pat.tokenable_type = %s
LIMIT 1;
"""

_DEFAULT_SANCTUM_TOUCH_QUERY = """
UPDATE personal_access_tokens
SET last_used_at = UTC_TIMESTAMP()
WHERE id = %s;
"""

_DEFAULT_SANCTUM_TOKENABLE_TYPE = "App\\Models\\Student\\Student"


@dataclass(frozen=True)
class AuthResult:
    ok: bool
    principal_id: int | None = None
    scopes: list[str] = field(default_factory=list)
    trust_level: int = 0
    bypass: bool = False
    reason: str | None = None
    status_code: int = 401


class AuthAdapter:
    async def authenticate(
        self,
        *,
        request: Request,
        funding_request_id: int | None = None,
        student_id_override: int | None = None,
    ) -> AuthResult:
        raise NotImplementedError

    async def funding_request_owner_id(self, *, funding_request_id: int) -> int | None:
        _ = funding_request_id
        return None


class DevBypassAuthAdapter(AuthAdapter):
    def __init__(self, *, allow_bypass: bool) -> None:
        self._allow_bypass = allow_bypass

    async def authenticate(
        self,
        *,
        request: Request,
        funding_request_id: int | None = None,
        student_id_override: int | None = None,
    ) -> AuthResult:
        header_id = _parse_header_id(request)
        if header_id:
            return AuthResult(ok=True, principal_id=header_id, scopes=[], trust_level=0, bypass=False)

        if not self._allow_bypass:
            return AuthResult(ok=False, reason="auth_not_configured", status_code=401)

        principal_id = student_id_override
        if principal_id is None and funding_request_id is not None:
            result = await platform_load_funding_thread_context.execute(funding_request_id=funding_request_id)
            if not result.success or not isinstance(result.content, dict):
                return AuthResult(ok=False, reason="funding_request_not_found", status_code=404)
            principal_id = int(result.content.get("user_id") or result.content.get("student_id") or 0)

        if not principal_id:
            return AuthResult(ok=False, reason="principal_not_found", status_code=401)

        return AuthResult(
            ok=True,
            principal_id=int(principal_id),
            scopes=["debug"],
            trust_level=0,
            bypass=True,
        )

    async def funding_request_owner_id(self, *, funding_request_id: int) -> int | None:
        result = await platform_load_funding_thread_context.execute(funding_request_id=funding_request_id)
        if not result.success or not isinstance(result.content, dict):
            return None
        value = result.content.get("user_id") or result.content.get("student_id")
        try:
            owner_id = int(value or 0)
        except (TypeError, ValueError):
            return None
        return owner_id if owner_id > 0 else None


class PlatformSessionAuthAdapter(AuthAdapter):
    def __init__(
        self,
        *,
        platform_db: PlatformDB,
        session_cookie_names: tuple[str, ...],
        session_query: str | None = None,
        funding_owner_query: str | None = None,
        allow_header_override: bool = False,
    ) -> None:
        self._platform_db = platform_db
        self._session_cookie_names = tuple(name for name in session_cookie_names if name)
        self._allow_header_override = bool(allow_header_override)
        self._session_queries = (
            (session_query.strip(),) if isinstance(session_query, str) and session_query.strip() else _DEFAULT_SESSION_QUERIES
        )
        self._owner_queries = (
            (funding_owner_query.strip(),)
            if isinstance(funding_owner_query, str) and funding_owner_query.strip()
            else _DEFAULT_OWNER_QUERIES
        )

    @classmethod
    def from_settings(cls, settings: Any) -> PlatformSessionAuthAdapter:
        db = PlatformDB(
            PlatformDBConfig(
                host=str(settings.platform_db_host),
                port=int(settings.platform_db_port),
                user=str(settings.platform_db_user),
                password=str(settings.platform_db_pass),
                db=str(settings.platform_db_name),
                minsize=int(settings.platform_db_min),
                maxsize=int(settings.platform_db_max),
            )
        )
        cookie_names = tuple(settings.auth_session_cookie_names) if settings.auth_session_cookie_names else ("session_id",)
        return cls(
            platform_db=db,
            session_cookie_names=cookie_names,
            session_query=settings.auth_session_query,
            funding_owner_query=settings.auth_funding_owner_query,
            allow_header_override=bool(settings.auth_allow_header_override),
        )

    async def authenticate(
        self,
        *,
        request: Request,
        funding_request_id: int | None = None,
        student_id_override: int | None = None,
    ) -> AuthResult:
        if self._allow_header_override:
            header_id = _parse_header_id(request)
            if header_id:
                return AuthResult(
                    ok=True,
                    principal_id=header_id,
                    scopes=["debug_header"],
                    trust_level=0,
                    bypass=False,
                )

        session_token = _extract_session_token(request, cookie_names=self._session_cookie_names)
        if not session_token:
            return AuthResult(ok=False, reason="missing_session", status_code=401)

        session = await self._lookup_session(session_token=session_token)
        if session is None:
            return AuthResult(ok=False, reason="invalid_session", status_code=401)

        principal_id = _coerce_positive_int(session.get("principal_id"))
        if principal_id is None:
            return AuthResult(ok=False, reason="invalid_session_principal", status_code=401)

        if student_id_override is not None and int(student_id_override) != principal_id:
            return AuthResult(ok=False, reason="forbidden", status_code=403)

        if funding_request_id is not None:
            owner_id = await self.funding_request_owner_id(funding_request_id=funding_request_id)
            if owner_id is None:
                return AuthResult(ok=False, reason="funding_request_not_found", status_code=404)
            if owner_id != principal_id:
                return AuthResult(ok=False, reason="forbidden", status_code=403)

        return AuthResult(
            ok=True,
            principal_id=principal_id,
            scopes=_parse_scopes(session.get("scopes")),
            trust_level=_coerce_int(session.get("trust_level"), default=0),
            bypass=False,
        )

    async def funding_request_owner_id(self, *, funding_request_id: int) -> int | None:
        params = (int(funding_request_id),)
        for sql in self._owner_queries:
            row = await _safe_fetch_one(self._platform_db, sql=sql, params=params)
            if not row:
                continue
            owner = (
                row.get("owner_id")
                if isinstance(row, dict)
                else None
            )
            owner_id = _coerce_positive_int(owner)
            if owner_id is not None:
                return owner_id
        return None

    async def _lookup_session(self, *, session_token: str) -> dict[str, Any] | None:
        params = (session_token,)
        for sql in self._session_queries:
            row = await _safe_fetch_one(self._platform_db, sql=sql, params=params)
            if not row:
                continue
            principal = _coerce_positive_int(
                row.get("principal_id") if isinstance(row, dict) else None
            )
            if principal is None:
                continue
            return {
                "principal_id": principal,
                "scopes": row.get("scopes") if isinstance(row, dict) else None,
                "trust_level": row.get("trust_level") if isinstance(row, dict) else None,
            }
        return None


class SanctumBearerAuthAdapter(AuthAdapter):
    def __init__(
        self,
        *,
        platform_db: PlatformDB,
        token_query: str | None = None,
        tokenable_type: str = _DEFAULT_SANCTUM_TOKENABLE_TYPE,
        funding_owner_query: str | None = None,
        update_last_used_at: bool = False,
        touch_query: str | None = None,
    ) -> None:
        self._platform_db = platform_db
        self._token_query = (
            token_query.strip()
            if isinstance(token_query, str) and token_query.strip()
            else _DEFAULT_SANCTUM_TOKEN_QUERY
        )
        self._tokenable_type = tokenable_type or _DEFAULT_SANCTUM_TOKENABLE_TYPE
        self._owner_queries = (
            (funding_owner_query.strip(),)
            if isinstance(funding_owner_query, str) and funding_owner_query.strip()
            else _DEFAULT_OWNER_QUERIES
        )
        self._update_last_used_at = bool(update_last_used_at)
        self._touch_query = (
            touch_query.strip()
            if isinstance(touch_query, str) and touch_query.strip()
            else _DEFAULT_SANCTUM_TOUCH_QUERY
        )

    @classmethod
    def from_settings(cls, settings: Any) -> SanctumBearerAuthAdapter:
        db = PlatformDB(
            PlatformDBConfig(
                host=str(settings.platform_db_host),
                port=int(settings.platform_db_port),
                user=str(settings.platform_db_user),
                password=str(settings.platform_db_pass),
                db=str(settings.platform_db_name),
                minsize=int(settings.platform_db_min),
                maxsize=int(settings.platform_db_max),
            )
        )
        return cls(
            platform_db=db,
            token_query=settings.auth_sanctum_token_query,
            tokenable_type=str(settings.auth_sanctum_tokenable_type or _DEFAULT_SANCTUM_TOKENABLE_TYPE),
            funding_owner_query=settings.auth_funding_owner_query,
            update_last_used_at=bool(settings.auth_sanctum_update_last_used_at),
            touch_query=settings.auth_sanctum_touch_query,
        )

    async def authenticate(
        self,
        *,
        request: Request,
        funding_request_id: int | None = None,
        student_id_override: int | None = None,
    ) -> AuthResult:
        bearer_token = _extract_bearer_token(request)
        if not bearer_token:
            return AuthResult(ok=False, reason="missing_bearer_token", status_code=401)

        parsed = _parse_sanctum_token(bearer_token)
        if parsed is None:
            return AuthResult(ok=False, reason="invalid_token_format", status_code=401)
        token_id, plain_token = parsed

        hashed_token = hashlib.sha256(plain_token.encode("utf-8")).hexdigest()
        token_row = await self._lookup_token(token_id=token_id, hashed_token=hashed_token)
        if token_row is None:
            return AuthResult(ok=False, reason="invalid_token", status_code=401)

        if _is_expired(token_row.get("expires_at")):
            return AuthResult(ok=False, reason="token_expired", status_code=401)

        principal_id = _coerce_positive_int(token_row.get("principal_id"))
        if principal_id is None:
            return AuthResult(ok=False, reason="invalid_token_principal", status_code=401)

        if student_id_override is not None and int(student_id_override) != principal_id:
            return AuthResult(ok=False, reason="forbidden", status_code=403)

        if funding_request_id is not None:
            owner_id = await self.funding_request_owner_id(funding_request_id=funding_request_id)
            if owner_id is None:
                return AuthResult(ok=False, reason="funding_request_not_found", status_code=404)
            if owner_id != principal_id:
                return AuthResult(ok=False, reason="forbidden", status_code=403)

        if self._update_last_used_at:
            await _safe_execute(self._platform_db, sql=self._touch_query, params=(int(token_id),))

        return AuthResult(
            ok=True,
            principal_id=principal_id,
            scopes=_parse_scopes(token_row.get("abilities")),
            trust_level=0,
            bypass=False,
        )

    async def funding_request_owner_id(self, *, funding_request_id: int) -> int | None:
        params = (int(funding_request_id),)
        for sql in self._owner_queries:
            row = await _safe_fetch_one(self._platform_db, sql=sql, params=params)
            if not row:
                continue
            owner = row.get("owner_id") if isinstance(row, dict) else None
            owner_id = _coerce_positive_int(owner)
            if owner_id is not None:
                return owner_id
        return None

    async def _lookup_token(self, *, token_id: int, hashed_token: str) -> dict[str, Any] | None:
        params = (int(token_id), str(hashed_token), str(self._tokenable_type))
        row = await _safe_fetch_one(self._platform_db, sql=self._token_query, params=params)
        if not row:
            return None
        return row


def _parse_header_id(request: Request) -> int | None:
    header = request.headers.get("x-student-id") or request.headers.get("x-principal-id")
    if not header:
        return None
    try:
        value = int(header)
    except ValueError:
        return None
    return value if value > 0 else None


def _extract_session_token(request: Request, *, cookie_names: tuple[str, ...]) -> str | None:
    bearer = _extract_bearer_token(request)
    if bearer:
        return bearer
    for name in cookie_names:
        value = request.cookies.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_bearer_token(request: Request) -> str | None:
    authz = request.headers.get("authorization") or ""
    if not authz.lower().startswith("bearer "):
        return None
    token = authz.split(" ", 1)[1].strip()
    return token if token else None


def _parse_sanctum_token(raw: str) -> tuple[int, str] | None:
    parts = str(raw or "").split("|", 1)
    if len(parts) != 2:
        return None
    token_id = _coerce_positive_int(parts[0].strip())
    plain = parts[1].strip()
    if token_id is None or not plain:
        return None
    return token_id, plain


def _is_expired(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value < datetime.utcnow()
        return value < datetime.now(timezone.utc)
    return False


def _parse_scopes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    return []


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


async def _safe_fetch_one(db: PlatformDB, *, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    try:
        row = await db.fetch_one(sql, params)
    except Exception:
        return None
    if isinstance(row, dict):
        return row
    return None


async def _safe_execute(db: PlatformDB, *, sql: str, params: tuple[Any, ...]) -> bool:
    try:
        await db.execute(sql, params)
    except Exception:
        return False
    return True
