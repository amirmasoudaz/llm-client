from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from intelligence_layer_ops.platform_db import PlatformDB, PlatformDBConfig


ALLOWED_FIELDS: dict[str, dict[str, Any]] = {
    "research_interest": {"type": "string", "max_length": 1000},
    "paper_title": {"type": "string", "max_length": 500},
    "journal": {"type": "string", "max_length": 255},
    "year": {"type": "integer", "minimum": 1900, "maximum": 2100},
    "research_connection": {"type": "string", "max_length": 12000},
}

_PLATFORM_DB: PlatformDB | None = None


def get_platform_db() -> PlatformDB:
    global _PLATFORM_DB
    if _PLATFORM_DB is not None:
        return _PLATFORM_DB
    cfg = PlatformDBConfig(
        host=os.getenv("PLATFORM_DB_HOST", os.getenv("DB_HOST", "127.0.0.1")),
        port=int(os.getenv("PLATFORM_DB_PORT", os.getenv("DB_PORT", "3306"))),
        user=os.getenv("PLATFORM_DB_USER", os.getenv("DB_USER", "funding")),
        password=os.getenv("PLATFORM_DB_PASS", os.getenv("DB_PASS", "secret")),
        db=os.getenv("PLATFORM_DB_NAME", os.getenv("DB_NAME", "emaildb")),
        minsize=int(os.getenv("PLATFORM_DB_MIN", os.getenv("DB_MIN", "1"))),
        maxsize=int(os.getenv("PLATFORM_DB_MAX", os.getenv("DB_MAX", "10"))),
    )
    _PLATFORM_DB = PlatformDB(cfg)
    return _PLATFORM_DB


def validate_and_normalize_fields(raw_fields: Any) -> dict[str, Any]:
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ValueError("fields must be a non-empty object")

    normalized: dict[str, Any] = {}
    unknown = [str(key) for key in raw_fields.keys() if str(key) not in ALLOWED_FIELDS]
    if unknown:
        raise ValueError(
            "unsupported fields: "
            + ", ".join(sorted(unknown))
            + "; allowed: research_interest, paper_title, journal, year, research_connection"
        )

    for field_name, spec in ALLOWED_FIELDS.items():
        if field_name not in raw_fields:
            continue
        value = raw_fields[field_name]
        if spec["type"] == "string":
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
            cleaned = value.strip()
            if cleaned == "":
                raise ValueError(f"{field_name} cannot be empty")
            if len(cleaned) > int(spec["max_length"]):
                raise ValueError(f"{field_name} exceeds max length {spec['max_length']}")
            normalized[field_name] = cleaned
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an integer")
            if value < int(spec["minimum"]) or value > int(spec["maximum"]):
                raise ValueError(
                    f"{field_name} must be between {spec['minimum']} and {spec['maximum']}"
                )
            normalized[field_name] = int(value)

    if not normalized:
        raise ValueError("no valid fields provided")
    return normalized


def to_iso_timestamp(value: Any) -> str | None:
    dt = to_utc_datetime(value)
    if dt is None:
        return None
    return dt.isoformat()


def to_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text == "":
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
