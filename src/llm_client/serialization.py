"""
Deterministic JSON serialization helpers for caching and hashing.
"""
from __future__ import annotations

import json
from typing import Any


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
    """
    return json.dumps(
        canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


__all__ = ["canonicalize", "stable_json_dumps"]
