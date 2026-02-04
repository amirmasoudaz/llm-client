from __future__ import annotations

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class Settings:
    debug: bool = getenv("IL_DEBUG", getenv("DEBUG_MODE", "false")).lower().strip() in {"1", "true", "yes"}

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
