from __future__ import annotations

from dataclasses import dataclass
from os import getenv


def _env_bool(name: str) -> bool | None:
    raw = getenv(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _debug_default() -> bool:
    direct = _env_bool("IL_DEBUG")
    if direct is not None:
        return direct
    legacy = _env_bool("DEBUG_MODE")
    if legacy is not None:
        return legacy
    return False


def _workflow_kernel_default() -> bool:
    explicit = _env_bool("IL_USE_WORKFLOW_KERNEL")
    if explicit is not None:
        return explicit
    rollout = getenv("IL_WORKFLOW_KERNEL_ROLLOUT", "auto").strip().lower()
    if rollout in {"0", "false", "off"}:
        return False
    if rollout in {"1", "true", "on"}:
        return True
    # Staged rollout: default-on for production-like environments, default-off in debug.
    return not _debug_default()


def _csv_env(name: str, *, default: str) -> tuple[str, ...]:
    raw = getenv(name, default)
    values = [item.strip() for item in raw.split(",")]
    clean = tuple(item for item in values if item)
    return clean or tuple(item for item in default.split(",") if item)


def _auth_bypass_default() -> bool:
    value = _env_bool("IL_AUTH_BYPASS")
    if value is None:
        return False
    return value


def _credits_bootstrap_default() -> bool:
    value = _env_bool("IL_CREDITS_BOOTSTRAP")
    if value is None:
        return True
    return value


@dataclass(frozen=True)
class Settings:
    debug: bool = _debug_default()
    use_workflow_kernel: bool = _workflow_kernel_default()
    auth_bypass: bool = _auth_bypass_default()
    auth_mode: str = getenv("IL_AUTH_MODE", "dev_bypass").strip().lower()
    auth_allow_header_override: bool = (
        _env_bool("IL_AUTH_ALLOW_HEADER_OVERRIDE")
        if _env_bool("IL_AUTH_ALLOW_HEADER_OVERRIDE") is not None
        else False
    )
    auth_session_cookie_names: tuple[str, ...] = _csv_env(
        "IL_AUTH_SESSION_COOKIE_NAMES",
        default="session_id,session,auth_session,canapply_session",
    )
    auth_session_query: str | None = getenv("IL_AUTH_SESSION_QUERY")
    auth_funding_owner_query: str | None = getenv("IL_AUTH_FUNDING_OWNER_QUERY")
    credits_bootstrap: bool = _credits_bootstrap_default()
    credits_bootstrap_amount: int = int(getenv("IL_CREDITS_BOOTSTRAP_AMOUNT", "1000"))
    credits_reservation_ttl_sec: int = int(getenv("IL_CREDITS_RESERVATION_TTL_SEC", "900"))
    credits_min_reserve: int = int(getenv("IL_CREDITS_MIN_RESERVE", "1"))

    # Intelligence layer Postgres (ledgers/runtime)
    il_pg_dsn: str = getenv("IL_PG_DSN", getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/intelligence_layer"))

    # Platform (MariaDB/MySQL) connection
    platform_db_host: str = getenv("PLATFORM_DB_HOST", getenv("DB_HOST", "127.0.0.1"))
    platform_db_port: int = int(getenv("PLATFORM_DB_PORT", getenv("DB_PORT", "3306")))
    platform_db_user: str = getenv("PLATFORM_DB_USER", getenv("DB_USER", "funding"))
    platform_db_pass: str = getenv("PLATFORM_DB_PASS", getenv("DB_PASS", "secret"))
    platform_db_name: str = getenv("PLATFORM_DB_NAME", getenv("DB_NAME", "emaildb"))
    platform_db_min: int = int(getenv("PLATFORM_DB_MIN", getenv("DB_MIN", "1")))
    platform_db_max: int = int(getenv("PLATFORM_DB_MAX", getenv("DB_MAX", "10")))


def get_settings() -> Settings:
    return Settings()
