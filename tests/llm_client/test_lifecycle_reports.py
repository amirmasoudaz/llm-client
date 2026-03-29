from __future__ import annotations

from dataclasses import dataclass

from llm_client.lifecycle import (
    LifecycleEventType,
    accumulate_session_report,
    build_request_report,
    normalize_lifecycle_event,
)
from llm_client.observability import LifecycleRecorder


@dataclass
class _Ctx:
    request_id: str | None = None
    session_id: str | None = None


def test_normalize_lifecycle_event_maps_engine_events() -> None:
    event = normalize_lifecycle_event(
        "request.end",
        {
            "status": 200,
            "latency_ms": 12,
            "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5, "total_cost": 0.01},
            "spec": {"provider": "openai", "model": "gpt-5-mini"},
            "attempts": 2,
        },
        _Ctx(request_id="req-1", session_id="sess-1"),
    )

    assert event is not None
    assert event.type is LifecycleEventType.REQUEST_COMPLETED
    assert event.provider == "openai"
    assert event.model == "gpt-5-mini"

    report = build_request_report(event)
    assert report.request_id == "req-1"
    assert report.success is True
    assert report.usage.total_tokens == 5
    assert float(report.usage.total_cost) == 0.01
    assert report.attempts == 2


def test_lifecycle_recorder_builds_request_and_session_reports() -> None:
    import asyncio

    recorder = LifecycleRecorder()
    ctx = _Ctx(request_id="req-1", session_id="sess-1")

    async def _emit() -> None:
        await recorder.emit("request.start", {"spec": {"provider": "openai", "model": "gpt-5-mini"}}, ctx)
        await recorder.emit("request.diagnostics", {"attempts": 2, "fallbacks": 1}, ctx)
        await recorder.emit("cache.hit", {"key": "abc"}, ctx)
        await recorder.emit(
            "request.end",
            {
                "status": 200,
                "latency_ms": 20,
                "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10, "total_cost": 0.05},
                "provider": "openai",
            },
            ctx,
        )

    asyncio.run(_emit())

    request_report = recorder.requests["req-1"]
    session_report = recorder.sessions["sess-1"]

    assert request_report.cache_hit is True
    assert request_report.fallbacks == 1
    assert request_report.success is True
    assert request_report.usage.total_tokens == 10
    assert session_report.request_count == 1
    assert session_report.success_count == 1
    assert session_report.cache_hits == 1
    assert session_report.avg_latency_ms == 20.0


def test_accumulate_session_report_sums_usage_and_outcomes() -> None:
    reports = [
        build_request_report(
            normalize_lifecycle_event(
                "request.end",
                {"status": 200, "latency_ms": 10, "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}},
                _Ctx(request_id="r1", session_id="s1"),
            )
        ),
        build_request_report(
            normalize_lifecycle_event(
                "request.error",
                {"status": 500, "error": "boom", "latency_ms": 30},
                _Ctx(request_id="r2", session_id="s1"),
            )
        ),
    ]

    session = accumulate_session_report("s1", reports)

    assert session.request_count == 2
    assert session.success_count == 1
    assert session.failure_count == 1
    assert session.total_latency_ms == 40.0
    assert session.usage.total_tokens == 3
