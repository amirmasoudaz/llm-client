from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from intelligence_layer_ops.platform_tools import platform_load_funding_thread_context


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


def _parse_header_id(request: Request) -> int | None:
    header = request.headers.get("x-student-id") or request.headers.get("x-principal-id")
    if not header:
        return None
    try:
        value = int(header)
    except ValueError:
        return None
    return value if value > 0 else None
