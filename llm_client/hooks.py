"""
Lightweight hooks for observability and integration.

This module provides hook implementations for:
- In-memory metrics (testing/debugging)
- OpenTelemetry tracing with GenAI semantic conventions
- Prometheus metrics export
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .lifecycle import (
    LifecycleEvent,
    RequestReport,
    SessionReport,
    accumulate_session_report,
    build_request_report,
    normalize_lifecycle_event,
)
from .logging import StructuredLogger
from .redaction import RedactionPolicy, sanitize_payload
from .telemetry import MetricRegistry, RequestUsage, UsageTracker, get_registry


class Hook(Protocol):
    """Protocol for observability hooks."""

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        """Emit an event with payload and request context."""
        ...


class HookManager:
    """Manages multiple hooks and broadcasts events to all of them."""

    def __init__(self, hooks: Iterable[Hook] | None = None) -> None:
        self._hooks = list(hooks or [])

    def add(self, hook: Hook) -> None:
        """Add a hook to the manager."""
        self._hooks.append(hook)

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        """Emit an event to all registered hooks."""
        for hook in self._hooks:
            result = hook.emit(event, payload, context)
            if asyncio.iscoroutine(result):
                await result


@dataclass(frozen=True)
class EngineDiagnosticsSnapshot:
    kind: str
    request_id: str | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class ContextPlanningSnapshot:
    request_id: str | None
    payload: dict[str, Any]


class EngineDiagnosticsRecorder:
    """
    Captures normalized engine diagnostics emitted by ExecutionEngine.

    This hook records the latest request/stream diagnostics per request ID and
    keeps a chronological event log for debugging and tests.
    """

    def __init__(self, *, redaction_policy: RedactionPolicy | None = None) -> None:
        self.requests: dict[str, EngineDiagnosticsSnapshot] = {}
        self.streams: dict[str, EngineDiagnosticsSnapshot] = {}
        self.events: list[tuple[str, dict[str, Any], str | None]] = []
        self.redaction_policy = redaction_policy or RedactionPolicy()

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        request_id = getattr(context, "request_id", None)
        payload_copy = sanitize_payload(payload, policy=self.redaction_policy)
        self.events.append((event, payload_copy, request_id))

        if event == "request.diagnostics":
            snapshot = EngineDiagnosticsSnapshot(
                kind="request",
                request_id=request_id,
                payload=payload_copy,
            )
            if request_id is not None:
                self.requests[request_id] = snapshot
        elif event == "stream.diagnostics":
            snapshot = EngineDiagnosticsSnapshot(
                kind="stream",
                request_id=request_id,
                payload=payload_copy,
            )
            if request_id is not None:
                self.streams[request_id] = snapshot

    def latest_request(self, request_id: str) -> EngineDiagnosticsSnapshot | None:
        return self.requests.get(request_id)

    def latest_stream(self, request_id: str) -> EngineDiagnosticsSnapshot | None:
        return self.streams.get(request_id)


class ContextPlanningRecorder:
    """Captures normalized context-planning decisions for tests and debugging."""

    def __init__(self, *, redaction_policy: RedactionPolicy | None = None) -> None:
        self.plans: list[ContextPlanningSnapshot] = []
        self.redaction_policy = redaction_policy or RedactionPolicy()

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        if event != "context.plan":
            return
        self.plans.append(
            ContextPlanningSnapshot(
                request_id=getattr(context, "request_id", None),
                payload=sanitize_payload(payload, policy=self.redaction_policy),
            )
        )

    def latest(self) -> ContextPlanningSnapshot | None:
        return self.plans[-1] if self.plans else None


class LifecycleRecorder:
    """Normalizes raw hook events into canonical lifecycle events and reports."""

    def __init__(self, *, redaction_policy: RedactionPolicy | None = None) -> None:
        self.events: list[LifecycleEvent] = []
        self.requests: dict[str, RequestReport] = {}
        self.sessions: dict[str, SessionReport] = {}
        self._diagnostics_by_request: dict[str, dict[str, Any]] = {}
        self._cache_hits: set[str] = set()
        self._idempotency_hits: set[str] = set()
        self._session_reports: dict[str, dict[str, RequestReport]] = {}
        self.redaction_policy = redaction_policy or RedactionPolicy()

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        lifecycle_event = normalize_lifecycle_event(event, payload, context)
        if lifecycle_event is None:
            return
        lifecycle_event = replace(
            lifecycle_event,
            payload=sanitize_payload(lifecycle_event.payload, policy=self.redaction_policy),
        )
        self.events.append(lifecycle_event)

        request_id = lifecycle_event.request_id
        session_id = lifecycle_event.session_id

        if request_id and lifecycle_event.type.name == "DIAGNOSTICS_CAPTURED":
            self._diagnostics_by_request[request_id] = dict(lifecycle_event.payload)
            return
        if request_id and lifecycle_event.type.name == "CACHE_HIT":
            self._cache_hits.add(request_id)
            return
        if request_id and lifecycle_event.type.name == "IDEMPOTENCY_HIT":
            self._idempotency_hits.add(request_id)
            return

        terminal = lifecycle_event.type.value in {
            "request.completed",
            "request.failed",
            "stream.completed",
            "stream.failed",
            "embedding.completed",
            "embedding.failed",
        }
        if not terminal or request_id is None:
            return

        report = build_request_report(
            lifecycle_event,
            prior_diagnostics=self._diagnostics_by_request.get(request_id),
            cache_hit=request_id in self._cache_hits,
            idempotency_hit=request_id in self._idempotency_hits,
        )
        self.requests[request_id] = report

        if session_id is not None:
            reports = self._session_reports.setdefault(session_id, {})
            reports[request_id] = report
            self.sessions[session_id] = accumulate_session_report(session_id, list(reports.values()))


class LifecycleLoggingHook:
    """Logs canonical lifecycle events and reports through StructuredLogger."""

    def __init__(
        self,
        logger: StructuredLogger | None = None,
        *,
        redaction_policy: RedactionPolicy | None = None,
        include_session_reports: bool = True,
    ) -> None:
        self.logger = logger or StructuredLogger(redaction_policy=redaction_policy)
        self.recorder = LifecycleRecorder(redaction_policy=redaction_policy)
        self.include_session_reports = include_session_reports
        self._reported_requests: set[str] = set()
        self._reported_sessions: dict[str, int] = {}

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        lifecycle_event = normalize_lifecycle_event(event, payload, context)
        if lifecycle_event is None:
            return
        await self.recorder.emit(event, payload, context)
        latest_event = self.recorder.events[-1]
        self.logger.log_lifecycle_event(latest_event)

        request_id = latest_event.request_id
        if request_id and request_id in self.recorder.requests and request_id not in self._reported_requests:
            report = self.recorder.requests[request_id]
            self.logger.log_request_report(report)
            self._reported_requests.add(request_id)

        session_id = latest_event.session_id
        if (
            self.include_session_reports
            and session_id
            and session_id in self.recorder.sessions
            and self._reported_sessions.get(session_id) != self.recorder.sessions[session_id].request_count
        ):
            report = self.recorder.sessions[session_id]
            self.logger.log_session_report(report)
            self._reported_sessions[session_id] = report.request_count


class LifecycleTelemetryHook:
    """Records lifecycle-normalized metrics and usage into telemetry primitives."""

    def __init__(
        self,
        registry: MetricRegistry | None = None,
        usage_tracker: UsageTracker | None = None,
        *,
        redaction_policy: RedactionPolicy | None = None,
    ) -> None:
        self.registry = registry or get_registry()
        self.usage_tracker = usage_tracker or UsageTracker(self.registry)
        self.recorder = LifecycleRecorder(redaction_policy=redaction_policy)
        self._reported_requests: set[str] = set()

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        lifecycle_event = normalize_lifecycle_event(event, payload, context)
        if lifecycle_event is None:
            return
        metric_name = f"llm.lifecycle.{lifecycle_event.type.value.replace('.', '_')}"
        self.registry.counter(metric_name).inc()

        await self.recorder.emit(event, payload, context)
        latest_event = self.recorder.events[-1]

        if latest_event.type.name == "CACHE_HIT":
            self.usage_tracker.record_cache_hit()
        elif latest_event.type.name == "CACHE_MISS":
            self.usage_tracker.record_cache_miss()
        elif latest_event.type.name == "TOOL_EXECUTED":
            session_id = latest_event.session_id or "unknown"
            tool_name = str(latest_event.payload.get("tool_name") or "unknown")
            duration_ms = float(latest_event.payload.get("duration_ms") or 0.0)
            self.usage_tracker.record_tool_call(session_id, tool_name, duration_ms)

        request_id = latest_event.request_id
        if request_id and request_id in self.recorder.requests and request_id not in self._reported_requests:
            report = self.recorder.requests[request_id]
            self.usage_tracker.record_request(
                report.session_id or "unknown",
                RequestUsage(
                    request_id=report.request_id,
                    provider=report.provider or "",
                    model=report.model or "",
                    input_tokens=report.usage.input_tokens,
                    output_tokens=report.usage.output_tokens,
                    total_tokens=report.usage.total_tokens,
                    input_cost=report.usage.input_cost,
                    output_cost=report.usage.output_cost,
                    total_cost=report.usage.total_cost,
                    latency_ms=float(report.latency_ms or 0.0),
                    cached=report.cache_hit,
                ),
            )
            self._reported_requests.add(request_id)


@dataclass(frozen=True)
class BenchmarkSnapshot:
    kind: str
    payload: dict[str, Any]


class BenchmarkRecorder:
    """Captures benchmark instrumentation events."""

    def __init__(self, *, redaction_policy: RedactionPolicy | None = None) -> None:
        self.redaction_policy = redaction_policy or RedactionPolicy()
        self.cases: list[BenchmarkSnapshot] = []
        self.reports: list[BenchmarkSnapshot] = []

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        _ = context
        if event == "benchmark.case":
            self.cases.append(BenchmarkSnapshot("case", sanitize_payload(payload, policy=self.redaction_policy)))
        elif event == "benchmark.report":
            self.reports.append(BenchmarkSnapshot("report", sanitize_payload(payload, policy=self.redaction_policy)))


class InMemoryMetricsHook:
    """
    Simple metrics accumulator for tests and local inspection.

    Stores all events as counters and captures latency values for requests.
    """

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}
        self.latencies_ms: list[float] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.tool_calls: dict[str, int] = {}
        self.tool_latencies_ms: list[float] = []
        self.errors: list[dict[str, Any]] = []

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        # Count all events
        self.counters[event] = self.counters.get(event, 0) + 1

        # Track request latency
        if event == "request.end" and "latency_ms" in payload:
            self.latencies_ms.append(float(payload["latency_ms"]))

        # Track cache operations
        elif event == "cache.hit":
            self.cache_hits += 1
        elif event == "cache.miss":
            self.cache_misses += 1

        # Track tool execution
        elif event == "tool.execute":
            tool_name = payload.get("tool_name", "unknown")
            self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
            if "duration_ms" in payload:
                self.tool_latencies_ms.append(float(payload["duration_ms"]))

        # Track errors
        elif event.endswith(".error"):
            self.errors.append({"event": event, "payload": payload})

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of all collected metrics."""
        return {
            "counters": dict(self.counters),
            "latencies_ms": list(self.latencies_ms),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0.0
            ),
            "tool_calls": dict(self.tool_calls),
            "tool_latencies_ms": list(self.tool_latencies_ms),
            "errors": list(self.errors),
        }

    def reset(self) -> dict[str, Any]:
        """Reset metrics and return the previous snapshot."""
        snapshot = self.snapshot()
        self.counters.clear()
        self.latencies_ms.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.tool_calls.clear()
        self.tool_latencies_ms.clear()
        self.errors.clear()
        return snapshot


class OpenTelemetryHook:
    """
    OpenTelemetry hook with GenAI semantic conventions.

    Creates spans for LLM requests with attributes following the
    OpenTelemetry GenAI semantic conventions.

    See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """

    def __init__(
        self,
        tracer_name: str = "llm_client",
        service_name: str = "llm-client",
    ) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("opentelemetry is required for OpenTelemetryHook") from exc

        self._trace = trace
        self._Status = Status
        self._StatusCode = StatusCode
        self._tracer = trace.get_tracer(tracer_name, schema_url="https://opentelemetry.io/schemas/1.21.0")
        self._spans: dict[str, Any] = {}
        self._request_metadata: dict[str, dict[str, Any]] = {}

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        request_id = getattr(context, "request_id", None)
        trace_id = getattr(context, "trace_id", None)
        span_id = getattr(context, "span_id", None)
        user_id = getattr(context, "user_id", None)
        tenant_id = getattr(context, "tenant_id", None)

        if event == "request.start" and request_id:
            # Start a new span for the request
            span = self._tracer.start_span(
                "gen_ai.request",
                attributes={
                    "gen_ai.request.id": request_id,
                    "gen_ai.trace.id": trace_id or "",
                    "gen_ai.span.id": span_id or "",
                    "gen_ai.user.id": user_id or "",
                    "gen_ai.tenant.id": tenant_id or "",
                },
            )

            # Store spec for later use
            if "spec" in payload:
                spec = payload["spec"]
                self._request_metadata[request_id] = {
                    "provider": spec.get("provider", ""),
                    "model": spec.get("model", ""),
                }

                # Set GenAI semantic convention attributes
                span.set_attribute("gen_ai.system", spec.get("provider", ""))
                span.set_attribute("gen_ai.request.model", spec.get("model", ""))

                if spec.get("temperature") is not None:
                    span.set_attribute("gen_ai.request.temperature", spec["temperature"])
                if spec.get("max_tokens") is not None:
                    span.set_attribute("gen_ai.request.max_tokens", spec["max_tokens"])

            self._spans[request_id] = span

        elif event == "request.end" and request_id:
            span = self._spans.pop(request_id, None)
            if span:
                # Add response attributes
                status = payload.get("status", 0)
                span.set_attribute("gen_ai.response.status", status)

                if "latency_ms" in payload:
                    span.set_attribute("gen_ai.response.latency_ms", payload["latency_ms"])

                # Token usage
                if "usage" in payload:
                    usage = payload["usage"]
                    if isinstance(usage, dict):
                        span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
                        span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
                        span.set_attribute("gen_ai.usage.total_tokens", usage.get("total_tokens", 0))
                    else:
                        span.set_attribute("gen_ai.usage.input_tokens", getattr(usage, "input_tokens", 0))
                        span.set_attribute("gen_ai.usage.output_tokens", getattr(usage, "output_tokens", 0))
                        span.set_attribute("gen_ai.usage.total_tokens", getattr(usage, "total_tokens", 0))

                # Set span status
                if status >= 400:
                    span.set_status(self._Status(self._StatusCode.ERROR))
                else:
                    span.set_status(self._Status(self._StatusCode.OK))

                span.end()

            # Cleanup metadata
            self._request_metadata.pop(request_id, None)

        elif event == "cache.hit" and request_id:
            span = self._spans.get(request_id)
            if span:
                span.add_event("cache.hit", {"cache.key": payload.get("key", "")})
                span.set_attribute("gen_ai.cache.hit", True)

        elif event == "cache.miss" and request_id:
            span = self._spans.get(request_id)
            if span:
                span.add_event("cache.miss", {"cache.key": payload.get("key", "")})
                span.set_attribute("gen_ai.cache.hit", False)

        elif event == "tool.execute" and request_id:
            span = self._spans.get(request_id)
            if span:
                span.add_event(
                    "tool.execute",
                    {
                        "tool.name": payload.get("tool_name", ""),
                        "tool.duration_ms": payload.get("duration_ms", 0),
                    },
                )

        elif event == "provider.error" and request_id:
            span = self._spans.get(request_id)
            if span:
                span.add_event(
                    "provider.error",
                    {
                        "error.type": payload.get("error_type", ""),
                        "error.message": payload.get("error", ""),
                    },
                )
                span.set_status(self._Status(self._StatusCode.ERROR))


class PrometheusHook:
    """
    Prometheus metrics hook.

    Exposes the following metrics:
    - llm_requests_total: Counter of LLM requests by provider, model, status
    - llm_request_latency_seconds: Histogram of request latency
    - llm_tokens_total: Counter of tokens by provider, model, type (input/output)
    - llm_cache_operations_total: Counter of cache hits/misses
    - llm_tool_calls_total: Counter of tool calls by name
    - llm_tool_latency_seconds: Histogram of tool execution latency
    - llm_active_requests: Gauge of currently active requests
    """

    def __init__(self, port: int = 8000, start_server: bool = True) -> None:
        try:
            from prometheus_client import Counter, Gauge, Histogram, start_http_server
        except ImportError as exc:
            raise ImportError("prometheus_client is required for PrometheusHook") from exc

        self._Counter = Counter
        self._Histogram = Histogram
        self._Gauge = Gauge

        # Requests counter
        self.requests = Counter("llm_requests_total", "Total LLM requests", ["provider", "model", "status"])

        # Request latency histogram
        self.latency = Histogram(
            "llm_request_latency_seconds",
            "Request latency in seconds",
            ["provider", "model"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )

        # Token counter
        self.tokens = Counter(
            "llm_tokens_total",
            "Total tokens consumed",
            ["provider", "model", "type"],  # type=input/output
        )

        # Cache operations counter
        self.cache_ops = Counter(
            "llm_cache_operations_total",
            "Total cache operations",
            ["operation"],  # operation=hit/miss/write
        )

        # Tool calls counter
        self.tool_calls = Counter(
            "llm_tool_calls_total",
            "Total tool calls",
            ["tool_name", "status"],  # status=success/error
        )

        # Tool latency histogram
        self.tool_latency = Histogram(
            "llm_tool_latency_seconds",
            "Tool execution latency in seconds",
            ["tool_name"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0),
        )

        # Active requests gauge
        self.active_requests = Gauge(
            "llm_active_requests", "Number of currently active requests", ["provider", "model"]
        )

        # Store request metadata to access on request.end
        self._request_metadata: dict[str, dict[str, str]] = {}

        # Start Prometheus HTTP server
        if start_server:
            try:
                start_http_server(port)
            except OSError:
                # Assumed already running or port busy
                pass

    async def emit(self, event: str, payload: dict, context: Any) -> None:
        request_id = getattr(context, "request_id", None)

        if event == "request.start" and request_id:
            # Store metadata for this request
            spec = payload.get("spec", {})
            provider = spec.get("provider", "unknown")
            model = spec.get("model", "unknown")

            self._request_metadata[request_id] = {
                "provider": provider,
                "model": model,
            }

            # Increment active requests
            self.active_requests.labels(provider=provider, model=model).inc()

        elif event == "request.end" and request_id:
            # Get stored metadata
            metadata = self._request_metadata.pop(request_id, {})
            provider = metadata.get("provider", "unknown")
            model = metadata.get("model", "unknown")
            status = str(payload.get("status", 0))

            # Decrement active requests
            self.active_requests.labels(provider=provider, model=model).dec()

            # Record request count
            self.requests.labels(provider=provider, model=model, status=status).inc()

            # Record latency (convert ms to seconds)
            if "latency_ms" in payload:
                self.latency.labels(provider=provider, model=model).observe(payload["latency_ms"] / 1000.0)

            # Record tokens
            if "usage" in payload:
                usage = payload["usage"]
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                else:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)

                if input_tokens > 0:
                    self.tokens.labels(provider=provider, model=model, type="input").inc(input_tokens)
                if output_tokens > 0:
                    self.tokens.labels(provider=provider, model=model, type="output").inc(output_tokens)

        elif event == "cache.hit":
            self.cache_ops.labels(operation="hit").inc()

        elif event == "cache.miss":
            self.cache_ops.labels(operation="miss").inc()

        elif event == "cache.write":
            self.cache_ops.labels(operation="write").inc()

        elif event == "tool.execute":
            tool_name = payload.get("tool_name", "unknown")
            status = "success" if payload.get("success", True) else "error"

            self.tool_calls.labels(tool_name=tool_name, status=status).inc()

            if "duration_ms" in payload:
                self.tool_latency.labels(tool_name=tool_name).observe(payload["duration_ms"] / 1000.0)


__all__ = [
    "Hook",
    "HookManager",
    "EngineDiagnosticsSnapshot",
    "EngineDiagnosticsRecorder",
    "ContextPlanningSnapshot",
    "ContextPlanningRecorder",
    "LifecycleRecorder",
    "InMemoryMetricsHook",
    "OpenTelemetryHook",
    "PrometheusHook",
]
