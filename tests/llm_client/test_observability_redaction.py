from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass

from llm_client.logging import StructuredLogger
from llm_client.observability import (
    LifecycleLoggingHook,
    LifecycleRecorder,
    LifecycleTelemetryHook,
    MetricRegistry,
    PayloadPreviewMode,
    RedactionPolicy,
    UsageTracker,
)


@dataclass
class _Ctx:
    request_id: str | None = None
    session_id: str | None = None


def test_lifecycle_recorder_applies_central_redaction_policy() -> None:
    recorder = LifecycleRecorder(redaction_policy=RedactionPolicy())
    ctx = _Ctx(request_id="req-1", session_id="sess-1")

    async def _emit() -> None:
        await recorder.emit(
            "request.diagnostics",
            {"attempts": 2, "api_key": "sk-secret-value", "authorization": "Bearer hidden"},
            ctx,
        )
        await recorder.emit(
            "request.end",
            {"status": 200, "usage": {"total_tokens": 5}},
            ctx,
        )

    asyncio.run(_emit())

    request_report = recorder.requests["req-1"]
    assert request_report.diagnostics["api_key"] == "[REDACTED]"
    assert request_report.diagnostics["authorization"] == "[REDACTED]"


def test_lifecycle_logging_hook_uses_safe_payload_previews(capsys) -> None:
    policy = RedactionPolicy(preview_mode=PayloadPreviewMode.TRUNCATED, preview_max_chars=24)
    logger = StructuredLogger(name=f"llm_client.test.{uuid.uuid4().hex}", redaction_policy=policy)
    hook = LifecycleLoggingHook(logger=logger, redaction_policy=policy, include_session_reports=False)
    ctx = _Ctx(request_id="req-2", session_id="sess-2")

    async def _emit() -> None:
        await hook.emit(
            "request.diagnostics",
            {
                "prompt": "x" * 80,
                "api_key": "sk-very-secret",
                "nested": {"authorization": "Bearer hidden", "detail": "y" * 60},
            },
            ctx,
        )

    asyncio.run(_emit())
    line = next(item for item in capsys.readouterr().out.splitlines() if "lifecycle" in item)
    payload = json.loads(line)

    assert "sk-very-secret" not in line
    assert payload["payload_preview"]["api_key"] == "[REDACTED]"
    assert payload["payload_preview"]["nested"]["authorization"] == "[REDACTED]"
    assert payload["payload_preview"]["prompt"].endswith("chars total)")


def test_lifecycle_telemetry_hook_records_lifecycle_metrics_and_usage() -> None:
    registry = MetricRegistry()
    usage_tracker = UsageTracker(registry)
    hook = LifecycleTelemetryHook(registry=registry, usage_tracker=usage_tracker)
    ctx = _Ctx(request_id="req-3", session_id="sess-3")

    async def _emit() -> None:
        await hook.emit("cache.hit", {"key": "abc"}, ctx)
        await hook.emit("request.diagnostics", {"attempts": 1}, ctx)
        await hook.emit(
            "request.end",
            {
                "status": 200,
                "latency_ms": 12,
                "provider": "openai",
                "model": "gpt-5-mini",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "total_tokens": 7,
                    "total_cost": 0.02,
                },
            },
            ctx,
        )

    asyncio.run(_emit())

    snapshot = registry.snapshot()
    assert snapshot["counters"]["llm.lifecycle.cache_hit"] == 1
    assert snapshot["counters"]["llm.lifecycle.request_completed"] == 1
    assert snapshot["counters"]["llm.requests.total"] == 1
    assert snapshot["counters"]["llm.cache.hits"] == 1

    session = usage_tracker.get_session_summary("sess-3")
    assert session is not None
    assert session["total_tokens"] == 7
    assert session["cache_hits"] == 1
    latency_histogram = snapshot["histograms"]["llm.request.latency_ms"]
    assert latency_histogram["buckets"]["25.0"] == 1
