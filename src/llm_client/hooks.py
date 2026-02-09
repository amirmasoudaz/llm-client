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
from typing import Any, Protocol


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


__all__ = ["Hook", "HookManager", "InMemoryMetricsHook", "OpenTelemetryHook", "PrometheusHook"]
