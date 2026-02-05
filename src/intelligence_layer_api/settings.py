from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class Settings:
    debug: bool = getenv("IL_DEBUG", getenv("DEBUG_MODE", "false")).lower().strip() in {"1", "true", "yes"}
    use_workflow_kernel: bool = getenv("IL_USE_WORKFLOW_KERNEL", "false").lower().strip() in {"1", "true", "yes"}
    auth_bypass: bool = getenv("IL_AUTH_BYPASS", "false").lower().strip() in {"1", "true", "yes"}
    credits_bootstrap: bool = getenv("IL_CREDITS_BOOTSTRAP", "true").lower().strip() in {"1", "true", "yes"}
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
