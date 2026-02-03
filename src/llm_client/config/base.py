"""
Base types for configuration.
"""

from __future__ import annotations

from typing import Literal

CacheBackendType = Literal["none", "fs", "pg_redis", "qdrant"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["text", "json"]


__all__ = ["CacheBackendType", "LogLevel", "LogFormat"]
