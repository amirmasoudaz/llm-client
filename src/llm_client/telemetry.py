"""
Telemetry and Observability for LLM Client.

This module provides low-overhead metrics collection, including:
- MetricRegistry: Central registry for counters, gauges, and histograms
- UsageTracker: Per-request and aggregate token/cost tracking
- CacheMetrics: Hit/miss rates and timing
- LatencyRecorder: Request and operation timing with histograms
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection."""

    enabled: bool = True

    # Verbosity levels
    track_usage: bool = True
    track_latency: bool = True
    track_cache: bool = True
    track_tools: bool = True

    # Sampling (1.0 = 100%, 0.1 = 10%)
    sampling_rate: float = 1.0

    # Histogram bucket boundaries (in seconds for latency)
    latency_buckets: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __post_init__(self) -> None:
        if self.sampling_rate < 0.0 or self.sampling_rate > 1.0:
            raise ValueError("sampling_rate must be between 0.0 and 1.0")


class Counter:
    """Thread-safe counter metric."""

    __slots__ = ("_value", "_lock")

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        """Increment the counter."""
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value

    def reset(self) -> int:
        """Reset and return the previous value."""
        with self._lock:
            prev = self._value
            self._value = 0
            return prev


class Gauge:
    """Thread-safe gauge metric."""

    __slots__ = ("_value", "_lock")

    def __init__(self, initial: float = 0.0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set the gauge value."""
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment the gauge."""
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement the gauge."""
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class Histogram:
    """Thread-safe histogram metric with configurable buckets."""

    __slots__ = ("_buckets", "_counts", "_sum", "_count", "_lock")

    def __init__(self, buckets: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)) -> None:
        self._buckets = tuple(sorted(buckets)) + (float("inf"),)
        self._counts = [0] * len(self._buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation."""
        with self._lock:
            self._sum += value
            self._count += 1
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._counts[i] += 1
                    break

    @property
    def count(self) -> int:
        return self._count

    @property
    def sum(self) -> float:
        return self._sum

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the histogram."""
        with self._lock:
            return {
                "count": self._count,
                "sum": self._sum,
                "mean": self.mean,
                "buckets": {str(b): c for b, c in zip(self._buckets, self._counts, strict=False)},
            }


class MetricRegistry:
    """Central registry for all metrics."""

    def __init__(self, config: TelemetryConfig | None = None) -> None:
        self.config = config or TelemetryConfig()
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str) -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter()
            return self._counters[name]

    def gauge(self, name: str) -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge()
            return self._gauges[name]

    def histogram(self, name: str, buckets: tuple[float, ...] | None = None) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(buckets or self.config.latency_buckets)
            return self._histograms[name]

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of all metrics."""
        with self._lock:
            return {
                "counters": {k: v.value for k, v in self._counters.items()},
                "gauges": {k: v.value for k, v in self._gauges.items()},
                "histograms": {k: v.snapshot() for k, v in self._histograms.items()},
            }

    def reset(self) -> dict[str, Any]:
        """Reset all metrics and return the previous snapshot."""
        snapshot = self.snapshot()
        with self._lock:
            for c in self._counters.values():
                c.reset()
            for g in self._gauges.values():
                g.set(0.0)
            # Note: histograms are not reset to preserve distribution data
        return snapshot


@dataclass
class RequestUsage:
    """Usage data for a single request."""

    request_id: str
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    output_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    latency_ms: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": float(self.input_cost),
            "output_cost": float(self.output_cost),
            "total_cost": float(self.total_cost),
            "latency_ms": self.latency_ms,
            "cached": self.cached,
        }


@dataclass
class SessionUsage:
    """Aggregated usage for a session."""

    session_id: str
    total_requests: int = 0
    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    tool_calls: dict[str, int] = field(default_factory=dict)

    def add_request(self, usage: RequestUsage) -> None:
        """Accumulate usage from a request."""
        self.total_requests += 1
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost += usage.total_cost
        self.total_latency_ms += usage.latency_ms
        if usage.cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def add_tool_call(self, tool_name: str) -> None:
        """Track a tool call."""
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

    def increment_turns(self) -> None:
        """Increment turn count."""
        self.total_turns += 1

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency per request."""
        return self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_requests": self.total_requests,
            "total_turns": self.total_turns,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": float(self.total_cost),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "tool_calls": dict(self.tool_calls),
        }


class UsageTracker:
    """Tracks usage across requests and sessions."""

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self._registry = registry or MetricRegistry()
        self._sessions: dict[str, SessionUsage] = {}
        self._lock = threading.Lock()

    def get_session(self, session_id: str) -> SessionUsage:
        """Get or create a session usage tracker."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionUsage(session_id=session_id)
            return self._sessions[session_id]

    def record_request(self, session_id: str, usage: RequestUsage) -> None:
        """Record usage for a request."""
        session = self.get_session(session_id)
        session.add_request(usage)

        # Update registry counters
        self._registry.counter("llm.requests.total").inc()
        self._registry.counter("llm.tokens.input").inc(usage.input_tokens)
        self._registry.counter("llm.tokens.output").inc(usage.output_tokens)
        self._registry.histogram("llm.request.latency_ms").observe(usage.latency_ms)

    def record_tool_call(self, session_id: str, tool_name: str, duration_ms: float) -> None:
        """Record a tool call."""
        session = self.get_session(session_id)
        session.add_tool_call(tool_name)

        self._registry.counter("llm.tools.total").inc()
        self._registry.counter(f"llm.tools.{tool_name}").inc()
        self._registry.histogram("llm.tool.latency_ms").observe(duration_ms)

    def record_cache_hit(self, collection: str | None = None) -> None:
        """Record a cache hit."""
        self._registry.counter("llm.cache.hits").inc()
        if collection:
            self._registry.counter(f"llm.cache.{collection}.hits").inc()

    def record_cache_miss(self, collection: str | None = None) -> None:
        """Record a cache miss."""
        self._registry.counter("llm.cache.misses").inc()
        if collection:
            self._registry.counter(f"llm.cache.{collection}.misses").inc()

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get session summary."""
        with self._lock:
            session = self._sessions.get(session_id)
            return session.to_dict() if session else None

    def get_all_sessions(self) -> dict[str, dict[str, Any]]:
        """Get all session summaries."""
        with self._lock:
            return {sid: s.to_dict() for sid, s in self._sessions.items()}


@dataclass
class CacheStats:
    """Statistics for cache operations."""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0
    total_read_ms: float = 0.0
    total_write_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_read_ms(self) -> float:
        """Average read latency."""
        reads = self.hits + self.misses
        return self.total_read_ms / reads if reads > 0 else 0.0

    @property
    def avg_write_ms(self) -> float:
        """Average write latency."""
        return self.total_write_ms / self.writes if self.writes > 0 else 0.0

    def record_hit(self, latency_ms: float = 0.0) -> None:
        """Record a cache hit."""
        self.hits += 1
        self.total_read_ms += latency_ms

    def record_miss(self, latency_ms: float = 0.0) -> None:
        """Record a cache miss."""
        self.misses += 1
        self.total_read_ms += latency_ms

    def record_write(self, latency_ms: float = 0.0) -> None:
        """Record a cache write."""
        self.writes += 1
        self.total_write_ms += latency_ms

    def record_error(self) -> None:
        """Record a cache error."""
        self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "avg_read_ms": self.avg_read_ms,
            "avg_write_ms": self.avg_write_ms,
            "total_read_ms": self.total_read_ms,
            "total_write_ms": self.total_write_ms,
        }


class LatencyRecorder:
    """Context manager for recording operation latency."""

    __slots__ = ("_histogram", "_start", "_on_complete")

    def __init__(
        self,
        histogram: Histogram | None = None,
        on_complete: Callable[[float], None] | None = None,
    ) -> None:
        self._histogram = histogram
        self._on_complete = on_complete
        self._start: float = 0.0

    def __enter__(self) -> LatencyRecorder:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self._histogram:
            self._histogram.observe(elapsed_ms)
        if self._on_complete:
            self._on_complete(elapsed_ms)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self._start) * 1000


# =============================================================================
# Global Registry
# =============================================================================


# Global default registry
_default_registry: MetricRegistry | None = None


def get_registry() -> MetricRegistry:
    """Get the global metric registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricRegistry()
    return _default_registry


def set_registry(registry: MetricRegistry) -> None:
    """Set the global metric registry."""
    global _default_registry
    _default_registry = registry


def get_usage_tracker() -> UsageTracker:
    """Get a usage tracker with the global registry."""
    return UsageTracker(get_registry())


__all__ = [
    "TelemetryConfig",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricRegistry",
    "RequestUsage",
    "SessionUsage",
    "UsageTracker",
    "CacheStats",
    "LatencyRecorder",
    "get_registry",
    "set_registry",
    "get_usage_tracker",
]
