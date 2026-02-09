"""
Tests for the telemetry module.
"""

from decimal import Decimal

import pytest

from llm_client.cache import CacheStats
from llm_client.telemetry import (
    Counter,
    Gauge,
    Histogram,
    LatencyRecorder,
    MetricRegistry,
    RequestUsage,
    SessionUsage,
    TelemetryConfig,
    UsageTracker,
    get_registry,
    set_registry,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_initial_value(self):
        counter = Counter()
        assert counter.value == 0

    def test_increment(self):
        counter = Counter()
        counter.inc()
        assert counter.value == 1

    def test_increment_by_amount(self):
        counter = Counter()
        counter.inc(5)
        assert counter.value == 5

    def test_reset(self):
        counter = Counter()
        counter.inc(10)
        prev = counter.reset()
        assert prev == 10
        assert counter.value == 0


class TestGauge:
    """Tests for Gauge metric."""

    def test_initial_value(self):
        gauge = Gauge()
        assert gauge.value == 0.0

    def test_set(self):
        gauge = Gauge()
        gauge.set(42.5)
        assert gauge.value == 42.5

    def test_inc_dec(self):
        gauge = Gauge()
        gauge.inc(10)
        gauge.dec(3)
        assert gauge.value == 7.0


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        hist = Histogram(buckets=(1.0, 5.0, 10.0))
        hist.observe(0.5)
        hist.observe(3.0)
        hist.observe(7.0)

        assert hist.count == 3
        assert hist.sum == 10.5
        assert hist.mean == 3.5

    def test_snapshot(self):
        hist = Histogram(buckets=(1.0, 5.0))
        hist.observe(0.5)
        hist.observe(3.0)

        snap = hist.snapshot()
        assert snap["count"] == 2
        assert snap["sum"] == 3.5
        assert "buckets" in snap


class TestMetricRegistry:
    """Tests for MetricRegistry."""

    def test_counter_creation(self):
        registry = MetricRegistry()
        c1 = registry.counter("test.counter")
        c2 = registry.counter("test.counter")

        assert c1 is c2  # Same counter
        c1.inc()
        assert c2.value == 1

    def test_gauge_creation(self):
        registry = MetricRegistry()
        g = registry.gauge("test.gauge")
        g.set(42)
        assert registry.gauge("test.gauge").value == 42

    def test_histogram_creation(self):
        registry = MetricRegistry()
        h = registry.histogram("test.hist")
        h.observe(1.0)
        assert h.count == 1

    def test_snapshot(self):
        registry = MetricRegistry()
        registry.counter("c1").inc(5)
        registry.gauge("g1").set(10)
        registry.histogram("h1").observe(2.0)

        snap = registry.snapshot()
        assert snap["counters"]["c1"] == 5
        assert snap["gauges"]["g1"] == 10
        assert snap["histograms"]["h1"]["count"] == 1


class TestRequestUsage:
    """Tests for RequestUsage dataclass."""

    def test_to_dict(self):
        usage = RequestUsage(
            request_id="req-123",
            provider="openai",
            model="gpt-5",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            total_cost=Decimal("0.01"),
            latency_ms=500.0,
        )

        d = usage.to_dict()
        assert d["request_id"] == "req-123"
        assert d["input_tokens"] == 100
        assert d["total_cost"] == 0.01


class TestSessionUsage:
    """Tests for SessionUsage dataclass."""

    def test_add_request(self):
        session = SessionUsage(session_id="sess-1")

        request = RequestUsage(
            request_id="req-1",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            total_cost=Decimal("0.01"),
            latency_ms=500.0,
        )

        session.add_request(request)

        assert session.total_requests == 1
        assert session.total_input_tokens == 100
        assert session.total_cost == Decimal("0.01")

    def test_add_tool_call(self):
        session = SessionUsage(session_id="sess-1")
        session.add_tool_call("search")
        session.add_tool_call("search")
        session.add_tool_call("calculator")

        assert session.tool_calls["search"] == 2
        assert session.tool_calls["calculator"] == 1

    def test_cache_hit_rate(self):
        session = SessionUsage(session_id="sess-1")

        # Add cached request
        cached = RequestUsage(request_id="r1", cached=True)
        session.add_request(cached)

        # Add non-cached request
        uncached = RequestUsage(request_id="r2", cached=False)
        session.add_request(uncached)

        assert session.cache_hit_rate == 0.5


class TestUsageTracker:
    """Tests for UsageTracker."""

    def test_record_request(self):
        tracker = UsageTracker()

        usage = RequestUsage(
            request_id="req-1",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            latency_ms=500.0,
        )

        tracker.record_request("session-1", usage)

        summary = tracker.get_session_summary("session-1")
        assert summary is not None
        assert summary["total_requests"] == 1
        assert summary["total_input_tokens"] == 100

    def test_record_tool_call(self):
        tracker = UsageTracker()
        tracker.record_tool_call("session-1", "search", 100.0)
        tracker.record_tool_call("session-1", "search", 150.0)

        summary = tracker.get_session_summary("session-1")
        assert summary["tool_calls"]["search"] == 2

    def test_record_cache_operations(self):
        tracker = UsageTracker()
        tracker.record_cache_hit()
        tracker.record_cache_hit()
        tracker.record_cache_miss()

        registry = tracker._registry
        assert registry.counter("llm.cache.hits").value == 2
        assert registry.counter("llm.cache.misses").value == 1


class TestCacheStats:
    """Tests for CacheStats."""

    def test_record_hit(self):
        stats = CacheStats()
        stats.record_hit(10.0)
        stats.record_hit(20.0)

        assert stats.hits == 2
        assert stats.total_read_ms == 30.0

    def test_record_miss(self):
        stats = CacheStats()
        stats.record_miss(5.0)

        assert stats.misses == 1
        assert stats.hit_rate == 0.0

    def test_hit_rate(self):
        stats = CacheStats()
        stats.record_hit()
        stats.record_hit()
        stats.record_miss()

        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_reset(self):
        stats = CacheStats()
        stats.record_hit(10.0)
        stats.record_write(20.0)

        old = stats.reset()

        assert old.hits == 1
        assert old.writes == 1
        assert stats.hits == 0
        assert stats.writes == 0


class TestLatencyRecorder:
    """Tests for LatencyRecorder context manager."""

    def test_records_latency(self):
        hist = Histogram()

        with LatencyRecorder(histogram=hist) as recorder:
            # Simulate some work
            _ = sum(range(1000))

        assert hist.count == 1
        assert recorder.elapsed_ms > 0

    def test_callback_invoked(self):
        latencies = []

        with LatencyRecorder(on_complete=lambda ms: latencies.append(ms)):
            pass

        assert len(latencies) == 1
        assert latencies[0] >= 0


class TestTelemetryConfig:
    """Tests for TelemetryConfig."""

    def test_default_config(self):
        config = TelemetryConfig()
        assert config.enabled is True
        assert config.sampling_rate == 1.0

    def test_invalid_sampling_rate(self):
        with pytest.raises(ValueError):
            TelemetryConfig(sampling_rate=1.5)

        with pytest.raises(ValueError):
            TelemetryConfig(sampling_rate=-0.1)


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_and_set_registry(self):
        original = get_registry()

        new_registry = MetricRegistry()
        set_registry(new_registry)

        assert get_registry() is new_registry

        # Restore
        set_registry(original)
