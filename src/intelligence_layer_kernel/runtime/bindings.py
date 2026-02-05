from __future__ import annotations

import json
import re
from typing import Any, Callable

from blake3 import blake3


_TEMPLATE_RE = re.compile(r"\{([^{}]+)\}")


def get_path(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        else:
            return None
    return cur


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    cur: dict[str, Any] = data
    parts = path.split(".")
    for part in parts[:-1]:
        next_val = cur.get(part)
        if not isinstance(next_val, dict):
            next_val = {}
            cur[part] = next_val
        cur = next_val
    cur[parts[-1]] = value


def resolve_template_value(value: Any, ctx: "BindingContext") -> Any:
    if _is_binding(value):
        if "from" in value:
            return ctx.get(str(value["from"]))
        if "const" in value:
            return value.get("const")
        if "template" in value:
            return render_template(str(value["template"]), ctx)
    if isinstance(value, dict):
        return {k: resolve_template_value(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_template_value(v, ctx) for v in value]
    return value


def prune_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for k, v in value.items():
            pv = prune_nulls(v)
            if pv is None:
                continue
            cleaned[k] = pv
        return cleaned
    if isinstance(value, list):
        items = [prune_nulls(v) for v in value]
        return [v for v in items if v is not None]
    return value


def render_template(template: str, ctx: "BindingContext") -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        val = ctx.get(key)
        return _stringify(val)

    return _TEMPLATE_RE.sub(_replace, template)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _is_binding(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value.keys())
    return keys in ({"from"}, {"const"}, {"template"})


class BindingContext:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        if "computed" not in self.data:
            self.data["computed"] = {}

    def get(self, path: str) -> Any:
        if path.startswith("computed."):
            key = path.split(".", 1)[1]
            return self._compute(key)
        return get_path(self.data, path)

    def _compute(self, key: str) -> Any:
        computed = self.data.setdefault("computed", {})
        if key in computed:
            return computed[key]
        func = _COMPUTED.get(key)
        if func is None:
            computed[key] = None
            return None
        value = func(self.data)
        computed[key] = value
        return value


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake3(raw).hexdigest()


def _hash_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return blake3(text.encode("utf-8")).hexdigest()


def _compute_email_body_hash(ctx: dict[str, Any]) -> str:
    override = get_path(ctx, "intent.inputs.email_text_override")
    if override:
        return _hash_text(override)
    fallback = get_path(ctx, "context.platform.funding_request.email_content")
    if fallback:
        return _hash_text(fallback)
    fallback = get_path(ctx, "context.platform.email.main_email_body")
    if fallback:
        return _hash_text(fallback)
    return _hash_text("")


def _compute_requested_edits_hash(ctx: dict[str, Any]) -> str:
    edits = get_path(ctx, "intent.inputs.requested_edits") or []
    if not isinstance(edits, list):
        edits = []
    normalized = sorted({str(item) for item in edits})
    return _hash_json(normalized)


def _compute_fields_hash(ctx: dict[str, Any]) -> str:
    fields = get_path(ctx, "intent.inputs.fields") or {}
    if not isinstance(fields, dict):
        fields = {}
    return _hash_json(fields)


def _compute_source_hash(ctx: dict[str, Any]) -> str:
    source = get_path(ctx, "intent.inputs.source") or {}
    if not isinstance(source, dict):
        source = {"value": source}
    return _hash_json(source)


def _compute_target_fields_hash(ctx: dict[str, Any]) -> str:
    fields = get_path(ctx, "intent.inputs.target_fields") or []
    if not isinstance(fields, list):
        fields = []
    normalized = sorted({str(item) for item in fields})
    return _hash_json(normalized)


_COMPUTED: dict[str, Callable[[dict[str, Any]], Any]] = {
    "email_body_hash": _compute_email_body_hash,
    "requested_edits_hash": _compute_requested_edits_hash,
    "fields_hash": _compute_fields_hash,
    "source_hash": _compute_source_hash,
    "target_fields_hash": _compute_target_fields_hash,
}
