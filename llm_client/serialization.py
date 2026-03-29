"""
Deterministic JSON serialization helpers for caching and hashing.

Uses orjson for performance when available, with fallback to standard json.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

# Try to use orjson for better performance
try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    import json

    _HAS_ORJSON = False


def _type_id(obj_type: type) -> str:
    return f"{obj_type.__module__}.{obj_type.__name__}"


def canonicalize(obj: Any) -> Any:
    """
    Convert complex objects into JSON-friendly, deterministic structures.
    """
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(canonicalize(v) for v in obj)
    if isinstance(obj, type):
        return {"__type__": _type_id(obj)}
    if hasattr(obj, "model_json_schema"):
        try:
            return {"__schema__": obj.model_json_schema()}  # Pydantic v2
        except Exception:
            return {"__type__": _type_id(obj)}
    if hasattr(obj, "schema") and callable(obj.schema):
        try:
            return {"__schema__": obj.schema()}  # Pydantic v1
        except Exception:
            return {"__type__": _type_id(obj)}
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return canonicalize(obj.to_dict())
        except Exception:
            return str(obj)
    if hasattr(obj, "to_openai_format") and callable(obj.to_openai_format):
        try:
            return canonicalize(obj.to_openai_format())
        except Exception:
            return str(obj)
    return obj


def stable_json_dumps(obj: Any) -> str:
    """
    Dump an object to JSON with stable ordering for hashing.

    Uses orjson when available for ~5x performance improvement.
    """
    canonical = canonicalize(obj)

    if _HAS_ORJSON:
        # orjson.OPT_SORT_KEYS ensures deterministic output
        return orjson.dumps(canonical, option=orjson.OPT_SORT_KEYS).decode("utf-8")
    else:
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )


def fast_json_dumps(obj: Any) -> bytes:
    """
    Fast JSON serialization to bytes (non-canonical, for API calls).

    Uses orjson when available, otherwise falls back to json.
    Returns bytes for efficiency.
    """
    if _HAS_ORJSON:
        return orjson.dumps(obj)
    else:
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def fast_json_loads(data: bytes | str) -> Any:
    """
    Fast JSON deserialization.

    Uses orjson when available.
    """
    if _HAS_ORJSON:
        return orjson.loads(data)
    else:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)


@lru_cache(maxsize=1024)
def cached_stable_json_dumps(obj_tuple: tuple) -> str:
    """
    Cached version of stable_json_dumps for hashable tuple inputs.

    Use this when you have repeated serialization of the same data.
    Convert your data to a tuple before calling.
    """
    # Convert tuple back to appropriate structure
    return stable_json_dumps(_tuple_to_obj(obj_tuple))


def _tuple_to_obj(t: tuple) -> Any:
    """Convert a nested tuple structure back to lists/dicts."""
    if isinstance(t, tuple):
        if len(t) == 2 and t[0] == "__dict__":
            return {k: _tuple_to_obj(v) for k, v in t[1]}
        return [_tuple_to_obj(v) for v in t]
    return t


def obj_to_hashable(obj: Any) -> tuple:
    """
    Convert an object to a hashable tuple for use with cached_stable_json_dumps.
    """
    if isinstance(obj, dict):
        return ("__dict__", tuple((k, obj_to_hashable(obj[k])) for k in sorted(obj)))
    if isinstance(obj, (list, tuple)):
        return tuple(obj_to_hashable(v) for v in obj)
    if isinstance(obj, set):
        return tuple(sorted(obj_to_hashable(v) for v in obj))
    return obj


__all__ = [
    "canonicalize",
    "stable_json_dumps",
    "fast_json_dumps",
    "fast_json_loads",
    "cached_stable_json_dumps",
    "obj_to_hashable",
]
