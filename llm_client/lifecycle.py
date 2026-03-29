"""
Canonical lifecycle event taxonomy and report objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class LifecycleEventType(str, Enum):
    REQUEST_STARTED = "request.started"
    REQUEST_ATTEMPT = "request.attempt"
    REQUEST_DISPATCHED = "request.dispatched"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"
    STREAM_STARTED = "stream.started"
    STREAM_COMPLETED = "stream.completed"
    STREAM_FAILED = "stream.failed"
    EMBEDDING_STARTED = "embedding.started"
    EMBEDDING_COMPLETED = "embedding.completed"
    EMBEDDING_FAILED = "embedding.failed"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    ROUTER_FALLBACK = "router.fallback"
    PROVIDER_SUCCESS = "provider.success"
    PROVIDER_ERROR = "provider.error"
    IDEMPOTENCY_HIT = "idempotency.hit"
    IDEMPOTENCY_CONFLICT = "idempotency.conflict"
    IDEMPOTENCY_STARTED = "idempotency.started"
    IDEMPOTENCY_COMPLETED = "idempotency.completed"
    IDEMPOTENCY_FAILED = "idempotency.failed"
    TOOL_EXECUTED = "tool.executed"
    CONTEXT_PLANNED = "context.planned"
    DIAGNOSTICS_CAPTURED = "diagnostics.captured"


RAW_TO_LIFECYCLE_EVENT: dict[str, LifecycleEventType] = {
    "request.start": LifecycleEventType.REQUEST_STARTED,
    "request.attempt": LifecycleEventType.REQUEST_ATTEMPT,
    "request.pre_dispatch": LifecycleEventType.REQUEST_DISPATCHED,
    "request.end": LifecycleEventType.REQUEST_COMPLETED,
    "request.error": LifecycleEventType.REQUEST_FAILED,
    "request.diagnostics": LifecycleEventType.DIAGNOSTICS_CAPTURED,
    "stream.start": LifecycleEventType.STREAM_STARTED,
    "stream.end": LifecycleEventType.STREAM_COMPLETED,
    "stream.error": LifecycleEventType.STREAM_FAILED,
    "stream.diagnostics": LifecycleEventType.DIAGNOSTICS_CAPTURED,
    "embed.start": LifecycleEventType.EMBEDDING_STARTED,
    "embed.end": LifecycleEventType.EMBEDDING_COMPLETED,
    "embed.error": LifecycleEventType.EMBEDDING_FAILED,
    "cache.hit": LifecycleEventType.CACHE_HIT,
    "cache.miss": LifecycleEventType.CACHE_MISS,
    "router.fallback": LifecycleEventType.ROUTER_FALLBACK,
    "provider.success": LifecycleEventType.PROVIDER_SUCCESS,
    "provider.error": LifecycleEventType.PROVIDER_ERROR,
    "idempotency.hit": LifecycleEventType.IDEMPOTENCY_HIT,
    "idempotency.conflict": LifecycleEventType.IDEMPOTENCY_CONFLICT,
    "idempotency.start": LifecycleEventType.IDEMPOTENCY_STARTED,
    "idempotency.complete": LifecycleEventType.IDEMPOTENCY_COMPLETED,
    "idempotency.fail": LifecycleEventType.IDEMPOTENCY_FAILED,
    "tool.execute": LifecycleEventType.TOOL_EXECUTED,
    "context.plan": LifecycleEventType.CONTEXT_PLANNED,
}


@dataclass(frozen=True)
class LifecycleEvent:
    type: LifecycleEventType
    raw_event: str
    request_id: str | None = None
    session_id: str | None = None
    provider: str | None = None
    model: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UsageBreakdown:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    input_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    output_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))

    @classmethod
    def from_any(cls, value: Any) -> UsageBreakdown:
        if value is None:
            return cls()
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if not isinstance(value, dict):
            return cls()
        return cls(
            input_tokens=int(value.get("input_tokens", 0) or 0),
            output_tokens=int(value.get("output_tokens", 0) or 0),
            total_tokens=int(value.get("total_tokens", 0) or 0),
            cached_input_tokens=int(value.get("input_tokens_cached", 0) or 0),
            input_cost=Decimal(str(value.get("input_cost", 0) or 0)),
            output_cost=Decimal(str(value.get("output_cost", 0) or 0)),
            total_cost=Decimal(str(value.get("total_cost", 0) or 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_tokens_cached": self.cached_input_tokens,
            "input_cost": float(self.input_cost),
            "output_cost": float(self.output_cost),
            "total_cost": float(self.total_cost),
        }


@dataclass(frozen=True)
class RequestReport:
    request_id: str
    session_id: str | None = None
    lifecycle_type: LifecycleEventType = LifecycleEventType.REQUEST_COMPLETED
    provider: str | None = None
    model: str | None = None
    status: int | None = None
    success: bool = False
    latency_ms: float | None = None
    attempts: int = 0
    fallbacks: int = 0
    cache_hit: bool = False
    idempotency_hit: bool = False
    usage: UsageBreakdown = field(default_factory=UsageBreakdown)
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "lifecycle_type": self.lifecycle_type.value,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
            "fallbacks": self.fallbacks,
            "cache_hit": self.cache_hit,
            "idempotency_hit": self.idempotency_hit,
            "usage": self.usage.to_dict(),
            "error": self.error,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class SessionReport:
    session_id: str
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    cache_hits: int = 0
    idempotency_hits: int = 0
    total_latency_ms: float = 0.0
    usage: UsageBreakdown = field(default_factory=UsageBreakdown)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.request_count if self.request_count else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.request_count if self.request_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "cache_hits": self.cache_hits,
            "idempotency_hits": self.idempotency_hits,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "usage": self.usage.to_dict(),
        }


def normalize_lifecycle_event(event: str, payload: dict[str, Any], context: Any) -> LifecycleEvent | None:
    event_type = RAW_TO_LIFECYCLE_EVENT.get(event)
    if event_type is None:
        return None
    provider, model = _provider_model_from_payload(payload)
    return LifecycleEvent(
        type=event_type,
        raw_event=event,
        request_id=getattr(context, "request_id", None),
        session_id=getattr(context, "session_id", None),
        provider=provider,
        model=model,
        payload=dict(payload),
    )


def build_request_report(
    lifecycle_event: LifecycleEvent,
    *,
    prior_diagnostics: dict[str, Any] | None = None,
    cache_hit: bool = False,
    idempotency_hit: bool = False,
) -> RequestReport:
    diagnostics = dict(prior_diagnostics or {})
    diagnostics.update(_diagnostics_from_payload(lifecycle_event.payload))
    usage = UsageBreakdown.from_any(lifecycle_event.payload.get("usage"))
    status = _coerce_int(lifecycle_event.payload.get("status"))
    provider = str(diagnostics.get("final_provider") or lifecycle_event.provider or "") or None
    return RequestReport(
        request_id=lifecycle_event.request_id or "unknown",
        session_id=lifecycle_event.session_id,
        lifecycle_type=lifecycle_event.type,
        provider=provider,
        model=lifecycle_event.model,
        status=status,
        success=(status is not None and status < 400 and lifecycle_event.type not in {LifecycleEventType.REQUEST_FAILED, LifecycleEventType.STREAM_FAILED, LifecycleEventType.EMBEDDING_FAILED}),
        latency_ms=_coerce_float(lifecycle_event.payload.get("latency_ms")),
        attempts=int(diagnostics.get("attempts", 0) or 0),
        fallbacks=int(diagnostics.get("fallbacks", 0) or 0),
        cache_hit=cache_hit,
        idempotency_hit=idempotency_hit,
        usage=usage,
        error=str(lifecycle_event.payload.get("error")) if lifecycle_event.payload.get("error") else diagnostics.get("final_error"),
        diagnostics=diagnostics,
    )


def accumulate_session_report(
    session_id: str,
    reports: list[RequestReport],
) -> SessionReport:
    usage = UsageBreakdown()
    success_count = 0
    failure_count = 0
    cache_hits = 0
    idempotency_hits = 0
    total_latency_ms = 0.0

    for report in reports:
        if report.success:
            success_count += 1
        else:
            failure_count += 1
        if report.cache_hit:
            cache_hits += 1
        if report.idempotency_hit:
            idempotency_hits += 1
        total_latency_ms += float(report.latency_ms or 0.0)
        usage = UsageBreakdown(
            input_tokens=usage.input_tokens + report.usage.input_tokens,
            output_tokens=usage.output_tokens + report.usage.output_tokens,
            total_tokens=usage.total_tokens + report.usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens + report.usage.cached_input_tokens,
            input_cost=usage.input_cost + report.usage.input_cost,
            output_cost=usage.output_cost + report.usage.output_cost,
            total_cost=usage.total_cost + report.usage.total_cost,
        )

    return SessionReport(
        session_id=session_id,
        request_count=len(reports),
        success_count=success_count,
        failure_count=failure_count,
        cache_hits=cache_hits,
        idempotency_hits=idempotency_hits,
        total_latency_ms=total_latency_ms,
        usage=usage,
    )


def _provider_model_from_payload(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    spec = payload.get("spec") if isinstance(payload.get("spec"), dict) else {}
    provider = payload.get("provider") or payload.get("final_provider") or spec.get("provider")
    model = payload.get("model") or spec.get("model")
    return (str(provider) if provider else None, str(model) if model else None)


def _diagnostics_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keys = {"attempts", "fallbacks", "final_provider", "final_status", "final_error"}
    return {key: payload[key] for key in keys if key in payload}


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "LifecycleEventType",
    "LifecycleEvent",
    "RAW_TO_LIFECYCLE_EVENT",
    "UsageBreakdown",
    "RequestReport",
    "SessionReport",
    "normalize_lifecycle_event",
    "build_request_report",
    "accumulate_session_report",
]
