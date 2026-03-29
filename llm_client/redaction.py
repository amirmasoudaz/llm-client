"""
Shared redaction, provider-payload capture, and tool-output hardening policy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PayloadPreviewMode(str, Enum):
    OFF = "off"
    SUMMARY = "summary"
    TRUNCATED = "truncated"


class LogFieldClass(str, Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    FORBIDDEN = "forbidden"


class ProviderPayloadCaptureMode(str, Enum):
    OFF = "off"
    METADATA_ONLY = "metadata_only"
    REDACTED_PREVIEW = "redacted_preview"


DEFAULT_SENSITIVE_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "token",
    "secret",
    "password",
    "access_token",
    "refresh_token",
)

DEFAULT_FORBIDDEN_KEYS = (
    "raw_response",
    "raw_request",
    "request_payload",
    "response_payload",
    "provider_payload",
    "sdk_request",
    "sdk_response",
)

DEFAULT_SAFE_KEYS = (
    "request_id",
    "trace_id",
    "span_id",
    "session_id",
    "job_id",
    "provider",
    "model",
    "status",
    "latency_ms",
    "attempts",
    "fallbacks",
    "cache_hit",
    "cache_key_version",
)

DEFAULT_PREVIEW_KEYS = (
    "payload",
    "diagnostics",
    "spec",
    "params",
    "messages",
    "content",
    "result",
    "response",
    "output",
    "error_context",
)

DEFAULT_PROVIDER_PAYLOAD_KEYS = DEFAULT_FORBIDDEN_KEYS

DEFAULT_TOOL_REDACTION_PATTERNS = (
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:sk-|api_key[=:]\s*|token[=:]\s*|secret[=:]\s*)[A-Za-z0-9\-_]+",
)


@dataclass(frozen=True)
class ToolOutputPolicy:
    redact_sensitive: bool = True
    truncate_output: bool = True
    max_chars: int = 4000
    replacement: str = "[REDACTED]"
    truncation_suffix: str = "... [truncated]"
    patterns: tuple[str, ...] = DEFAULT_TOOL_REDACTION_PATTERNS


@dataclass(frozen=True)
class RedactionPolicy:
    sensitive_keys: tuple[str, ...] = DEFAULT_SENSITIVE_KEYS
    forbidden_keys: tuple[str, ...] = DEFAULT_FORBIDDEN_KEYS
    safe_keys: tuple[str, ...] = DEFAULT_SAFE_KEYS
    preview_keys: tuple[str, ...] = DEFAULT_PREVIEW_KEYS
    provider_payload_keys: tuple[str, ...] = DEFAULT_PROVIDER_PAYLOAD_KEYS
    provider_payload_capture: ProviderPayloadCaptureMode = ProviderPayloadCaptureMode.OFF
    preview_mode: PayloadPreviewMode = PayloadPreviewMode.TRUNCATED
    preview_max_chars: int = 160
    preview_max_items: int = 5
    placeholder: str = "[REDACTED]"
    summary_placeholder: str = "<omitted>"
    _normalized_sensitive: frozenset[str] = field(init=False, repr=False)
    _normalized_forbidden: frozenset[str] = field(init=False, repr=False)
    _normalized_safe: frozenset[str] = field(init=False, repr=False)
    _normalized_preview: frozenset[str] = field(init=False, repr=False)
    _normalized_provider_payload: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_normalized_sensitive",
            frozenset(_normalize_key(key) for key in self.sensitive_keys),
        )
        object.__setattr__(
            self,
            "_normalized_forbidden",
            frozenset(_normalize_key(key) for key in self.forbidden_keys),
        )
        object.__setattr__(
            self,
            "_normalized_safe",
            frozenset(_normalize_key(key) for key in self.safe_keys),
        )
        object.__setattr__(
            self,
            "_normalized_preview",
            frozenset(_normalize_key(key) for key in self.preview_keys),
        )
        object.__setattr__(
            self,
            "_normalized_provider_payload",
            frozenset(_normalize_key(key) for key in self.provider_payload_keys),
        )

    def is_sensitive_key(self, key: str) -> bool:
        return _normalize_key(key) in self._normalized_sensitive

    def is_forbidden_key(self, key: str) -> bool:
        return _normalize_key(key) in self._normalized_forbidden

    def is_safe_key(self, key: str) -> bool:
        return _normalize_key(key) in self._normalized_safe

    def should_preview_key(self, key: str) -> bool:
        return _normalize_key(key) in self._normalized_preview

    def is_provider_payload_key(self, key: str) -> bool:
        return _normalize_key(key) in self._normalized_provider_payload

    def classify_field(self, key: str) -> LogFieldClass:
        if self.is_forbidden_key(key):
            return LogFieldClass.FORBIDDEN
        if self.is_sensitive_key(key):
            return LogFieldClass.SENSITIVE
        return LogFieldClass.SAFE


def sanitize_payload(value: Any, policy: RedactionPolicy | None = None) -> Any:
    policy = policy or RedactionPolicy()
    return _sanitize_value(value, policy=policy, parent_key=None)


def preview_payload(value: Any, policy: RedactionPolicy | None = None) -> Any:
    policy = policy or RedactionPolicy()
    return _preview_value(value, policy=policy, parent_key=None)


def capture_provider_payload(value: Any, policy: RedactionPolicy | None = None) -> Any:
    policy = policy or RedactionPolicy()
    if policy.provider_payload_capture is ProviderPayloadCaptureMode.OFF:
        return None
    if policy.provider_payload_capture is ProviderPayloadCaptureMode.METADATA_ONLY:
        return _provider_payload_metadata(value, policy=policy)
    return _preview_value(value, policy=policy, parent_key=None)


def sanitize_tool_output(value: Any, policy: ToolOutputPolicy | None = None) -> Any:
    policy = policy or ToolOutputPolicy()
    return _sanitize_tool_output(value, policy=policy)


def sanitize_log_data(data: dict[str, Any], policy: RedactionPolicy | None = None) -> dict[str, Any]:
    policy = policy or RedactionPolicy()
    sanitized: dict[str, Any] = {}
    omitted_fields: list[str] = []
    for key, value in data.items():
        field_class = policy.classify_field(key)
        if field_class is LogFieldClass.FORBIDDEN:
            omitted_fields.append(key)
            captured = capture_provider_payload(value, policy=policy) if policy.is_provider_payload_key(key) else None
            if captured is not None:
                sanitized[f"{key}_capture"] = captured
            continue
        if policy.should_preview_key(key):
            preview = preview_payload(value, policy=policy)
            if preview is not None:
                sanitized[f"{key}_preview"] = preview
        else:
            sanitized[key] = _sanitize_value(value, policy=policy, parent_key=key)
    if omitted_fields:
        sanitized["_omitted_fields"] = omitted_fields
    return sanitized


def _sanitize_value(value: Any, *, policy: RedactionPolicy, parent_key: str | None) -> Any:
    if parent_key and policy.is_forbidden_key(parent_key):
        captured = capture_provider_payload(value, policy=policy) if policy.is_provider_payload_key(parent_key) else None
        return captured if captured is not None else policy.summary_placeholder
    if parent_key and policy.is_sensitive_key(parent_key):
        return policy.placeholder
    if isinstance(value, dict):
        return {
            str(key): _sanitize_value(item, policy=policy, parent_key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item, policy=policy, parent_key=parent_key) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item, policy=policy, parent_key=parent_key) for item in value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return value


def _preview_value(value: Any, *, policy: RedactionPolicy, parent_key: str | None) -> Any:
    if parent_key and policy.is_forbidden_key(parent_key):
        captured = capture_provider_payload(value, policy=policy) if policy.is_provider_payload_key(parent_key) else None
        return captured if captured is not None else policy.summary_placeholder
    if parent_key and policy.is_sensitive_key(parent_key):
        return policy.placeholder
    if policy.preview_mode is PayloadPreviewMode.OFF:
        return None
    if isinstance(value, str):
        if policy.preview_mode is PayloadPreviewMode.SUMMARY:
            return f"<str:{len(value)}>"
        return _truncate_string(value, policy.preview_max_chars)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        keys = list(value.keys())
        selected = keys[: policy.preview_max_items]
        preview = {
            str(key): _preview_value(value[key], policy=policy, parent_key=str(key))
            for key in selected
        }
        if len(keys) > len(selected):
            preview["_truncated_keys"] = len(keys) - len(selected)
        return preview
    if isinstance(value, (list, tuple)):
        selected = list(value[: policy.preview_max_items])
        preview_items = [_preview_value(item, policy=policy, parent_key=parent_key) for item in selected]
        if len(value) > len(selected):
            preview_items.append(f"<+{len(value) - len(selected)} more>")
        return preview_items
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = repr(value)
    if policy.preview_mode is PayloadPreviewMode.SUMMARY:
        return f"<{type(value).__name__}>"
    return _truncate_string(text, policy.preview_max_chars)


def _provider_payload_metadata(value: Any, *, policy: RedactionPolicy) -> dict[str, Any]:
    if isinstance(value, dict):
        keys = list(value.keys())
        selected = [str(key) for key in keys[: policy.preview_max_items]]
        metadata: dict[str, Any] = {
            "type": "dict",
            "key_count": len(keys),
            "keys": selected,
        }
        for key in ("id", "model", "status", "object", "type"):
            if key in value and not policy.is_sensitive_key(key):
                metadata[key] = _sanitize_value(value.get(key), policy=policy, parent_key=key)
        return metadata
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "item_count": len(value)}
    return {"type": type(value).__name__}


def _sanitize_tool_output(value: Any, *, policy: ToolOutputPolicy) -> Any:
    if isinstance(value, str):
        text = value
        if policy.redact_sensitive:
            for raw_pattern in policy.patterns:
                text = re.sub(raw_pattern, policy.replacement, text, flags=re.IGNORECASE)
        if policy.truncate_output and len(text) > policy.max_chars:
            limit = max(0, policy.max_chars - len(policy.truncation_suffix))
            text = text[:limit] + policy.truncation_suffix
        return text
    if isinstance(value, dict):
        return {str(key): _sanitize_tool_output(item, policy=policy) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_tool_output(item, policy=policy) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_tool_output(item, policy=policy) for item in value)
    return value


def _truncate_string(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... ({len(text)} chars total)"


def _normalize_key(key: str) -> str:
    return "".join(ch for ch in key.lower() if ch.isalnum())


__all__ = [
    "PayloadPreviewMode",
    "LogFieldClass",
    "ProviderPayloadCaptureMode",
    "RedactionPolicy",
    "ToolOutputPolicy",
    "capture_provider_payload",
    "sanitize_payload",
    "preview_payload",
    "sanitize_log_data",
    "sanitize_tool_output",
]
