"""
Reusable benchmark harness for deterministic local and labeled live runs.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import Any

from .cache import CachePolicy
from .context_planning import ContextPlanner, ContextPlanningRequest
from .hooks import HookManager
from .providers.types import Message, StreamEventType, ToolCall
from .spec import RequestContext, RequestSpec
from .structured import StructuredOutputConfig
from .tools.execution_engine import ToolExecutionEngine

BenchmarkRunner = Callable[[], Awaitable[dict[str, Any]]]


class BenchmarkRunMode(str, Enum):
    DETERMINISTIC_LOCAL = "deterministic_local"
    LIVE = "live"


class BenchmarkCategory(str, Enum):
    COMPLETION = "completion"
    STREAM = "stream"
    EMBEDDINGS = "embeddings"
    TOOLS = "tools"
    CACHE = "cache"
    FAILOVER = "failover"
    CONTEXT = "context"
    STRUCTURED = "structured"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: BenchmarkCategory
    runner: BenchmarkRunner = field(repr=False)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkRunMetadata:
    label: str
    mode: BenchmarkRunMode = BenchmarkRunMode.DETERMINISTIC_LOCAL
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        label = str(self.label or "").strip()
        if not label:
            raise ValueError("Benchmark label cannot be empty")
        if self.mode is BenchmarkRunMode.LIVE and not label.startswith("live:"):
            object.__setattr__(self, "label", f"live:{label}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mode": self.mode.value,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "tags": dict(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRunMetadata":
        return cls(
            label=str(data.get("label") or "benchmark"),
            mode=BenchmarkRunMode(str(data.get("mode") or BenchmarkRunMode.DETERMINISTIC_LOCAL.value)),
            run_id=str(data.get("run_id") or str(uuid.uuid4())),
            started_at=float(data.get("started_at") or time.time()),
            tags={str(key): str(value) for key, value in dict(data.get("tags") or {}).items()},
        )


@dataclass(frozen=True)
class BenchmarkRecord:
    name: str
    category: BenchmarkCategory
    status: str
    case_latency_ms: float
    metrics: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "status": self.status,
            "case_latency_ms": self.case_latency_ms,
            "metrics": _jsonable(self.metrics),
            "labels": dict(self.labels),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRecord":
        return cls(
            name=str(data.get("name") or "unknown"),
            category=BenchmarkCategory(str(data.get("category") or BenchmarkCategory.COMPLETION.value)),
            status=str(data.get("status") or "error"),
            case_latency_ms=float(data.get("case_latency_ms") or 0.0),
            metrics=dict(data.get("metrics") or {}),
            labels={str(key): str(value) for key, value in dict(data.get("labels") or {}).items()},
            error=data.get("error"),
        )


@dataclass(frozen=True)
class BenchmarkReport:
    metadata: BenchmarkRunMetadata
    records: list[BenchmarkRecord] = field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.records)

    @property
    def success_count(self) -> int:
        return sum(1 for record in self.records if record.ok)

    @property
    def failed_count(self) -> int:
        return self.total_cases - self.success_count

    @property
    def avg_case_latency_ms(self) -> float:
        if not self.records:
            return 0.0
        return mean(record.case_latency_ms for record in self.records)

    def find(self, name: str) -> BenchmarkRecord | None:
        for record in self.records:
            if record.name == name:
                return record
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "total_cases": self.total_cases,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "avg_case_latency_ms": self.avg_case_latency_ms,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkReport":
        return cls(
            metadata=BenchmarkRunMetadata.from_dict(dict(data.get("metadata") or {})),
            records=[BenchmarkRecord.from_dict(item) for item in list(data.get("records") or [])],
        )


@dataclass(frozen=True)
class BenchmarkComparisonRecord:
    name: str
    category: BenchmarkCategory
    metric_deltas: dict[str, float] = field(default_factory=dict)
    current_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkComparisonReport:
    current_label: str
    baseline_label: str
    records: list[BenchmarkComparisonRecord] = field(default_factory=list)


async def run_benchmarks(
    cases: list[BenchmarkCase],
    *,
    label: str,
    mode: BenchmarkRunMode = BenchmarkRunMode.DETERMINISTIC_LOCAL,
    hooks: HookManager | None = None,
    tags: dict[str, str] | None = None,
) -> BenchmarkReport:
    hook_manager = hooks or HookManager()
    metadata = BenchmarkRunMetadata(label=label, mode=mode, tags=dict(tags or {}))
    records: list[BenchmarkRecord] = []

    for case in cases:
        started = time.perf_counter()
        try:
            metrics = await case.runner()
            record = BenchmarkRecord(
                name=case.name,
                category=case.category,
                status="ok",
                case_latency_ms=(time.perf_counter() - started) * 1000,
                metrics=dict(metrics),
                labels=dict(case.labels),
            )
        except Exception as exc:
            record = BenchmarkRecord(
                name=case.name,
                category=case.category,
                status="error",
                case_latency_ms=(time.perf_counter() - started) * 1000,
                metrics={},
                labels=dict(case.labels),
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(record)
        await hook_manager.emit("benchmark.case", record.to_dict(), None)

    report = BenchmarkReport(metadata=metadata, records=records)
    await hook_manager.emit("benchmark.report", report.to_dict(), None)
    return report


def save_benchmark_report(report: BenchmarkReport, path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def load_benchmark_report(path: str | Path) -> BenchmarkReport:
    return BenchmarkReport.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def compare_benchmark_reports(
    current: BenchmarkReport,
    baseline: BenchmarkReport,
) -> BenchmarkComparisonReport:
    baseline_records = {record.name: record for record in baseline.records}
    comparison_records: list[BenchmarkComparisonRecord] = []

    for current_record in current.records:
        baseline_record = baseline_records.get(current_record.name)
        if baseline_record is None:
            continue
        deltas: dict[str, float] = {}
        for key, value in current_record.metrics.items():
            baseline_value = baseline_record.metrics.get(key)
            if _is_numeric(value) and _is_numeric(baseline_value):
                deltas[key] = float(value) - float(baseline_value)
        comparison_records.append(
            BenchmarkComparisonRecord(
                name=current_record.name,
                category=current_record.category,
                metric_deltas=deltas,
                current_metrics=dict(current_record.metrics),
                baseline_metrics=dict(baseline_record.metrics),
            )
        )

    return BenchmarkComparisonReport(
        current_label=current.metadata.label,
        baseline_label=baseline.metadata.label,
        records=comparison_records,
    )


def build_completion_benchmark_case(
    name: str,
    engine: Any,
    spec: RequestSpec,
    *,
    iterations: int = 1,
    context: RequestContext | None = None,
    timeout: float | None = None,
    labels: dict[str, str] | None = None,
) -> BenchmarkCase:
    async def _runner() -> dict[str, Any]:
        latencies: list[float] = []
        total_tokens: list[int] = []
        successes = 0
        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            result = await engine.complete(spec, context=context, timeout=timeout)
            latencies.append((time.perf_counter() - started) * 1000)
            if result.ok:
                successes += 1
            total_tokens.append(int((result.usage.total_tokens if result.usage else 0) or 0))
        return {
            "iterations": len(latencies),
            "success_rate": successes / len(latencies),
            "avg_completion_latency_ms": mean(latencies),
            "p50_completion_latency_ms": median(latencies),
            "min_completion_latency_ms": min(latencies),
            "max_completion_latency_ms": max(latencies),
            "avg_total_tokens": mean(total_tokens) if total_tokens else 0.0,
        }

    return BenchmarkCase(name=name, category=BenchmarkCategory.COMPLETION, runner=_runner, labels=dict(labels or {}))


def build_stream_benchmark_case(
    name: str,
    engine: Any,
    spec: RequestSpec,
    *,
    iterations: int = 1,
    context: RequestContext | None = None,
    timeout: float | None = None,
    labels: dict[str, str] | None = None,
) -> BenchmarkCase:
    async def _runner() -> dict[str, Any]:
        first_token_latencies: list[float] = []
        full_latencies: list[float] = []
        token_counts: list[int] = []
        successful_streams = 0

        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            first_token_ms: float | None = None
            token_count = 0
            final_ok = False
            async for event in engine.stream(spec, context=context, timeout=timeout):
                elapsed_ms = (time.perf_counter() - started) * 1000
                if event.type is StreamEventType.TOKEN:
                    token_count += 1
                    if first_token_ms is None:
                        first_token_ms = elapsed_ms
                elif event.type is StreamEventType.DONE:
                    final_ok = bool(getattr(event.data, "ok", False))
            full_latencies.append((time.perf_counter() - started) * 1000)
            first_token_latencies.append(first_token_ms or full_latencies[-1])
            token_counts.append(token_count)
            if final_ok:
                successful_streams += 1

        return {
            "iterations": len(full_latencies),
            "success_rate": successful_streams / len(full_latencies),
            "avg_first_token_latency_ms": mean(first_token_latencies),
            "p50_first_token_latency_ms": median(first_token_latencies),
            "avg_full_stream_latency_ms": mean(full_latencies),
            "p50_full_stream_latency_ms": median(full_latencies),
            "avg_stream_token_events": mean(token_counts) if token_counts else 0.0,
        }

    return BenchmarkCase(name=name, category=BenchmarkCategory.STREAM, runner=_runner, labels=dict(labels or {}))


def build_embeddings_benchmark_case(
    name: str,
    engine: Any,
    inputs: str | Iterable[str],
    *,
    iterations: int = 1,
    context: RequestContext | None = None,
    timeout: float | None = None,
    cache_policy: CachePolicy | None = None,
    labels: dict[str, str] | None = None,
) -> BenchmarkCase:
    async def _runner() -> dict[str, Any]:
        latencies: list[float] = []
        counts: list[int] = []
        inputs_list = [inputs] if isinstance(inputs, str) else list(inputs)
        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            result = await engine.embed(
                inputs_list,
                context=context,
                timeout=timeout,
                cache_policy=cache_policy,
            )
            latencies.append((time.perf_counter() - started) * 1000)
            counts.append(len(result.embeddings))
        avg_latency = mean(latencies)
        throughput = (mean(counts) / (avg_latency / 1000.0)) if avg_latency > 0 else math.inf
        return {
            "iterations": len(latencies),
            "avg_embedding_latency_ms": avg_latency,
            "p50_embedding_latency_ms": median(latencies),
            "throughput_items_per_second": throughput,
            "avg_embedding_count": mean(counts) if counts else 0.0,
        }

    return BenchmarkCase(name=name, category=BenchmarkCategory.EMBEDDINGS, runner=_runner, labels=dict(labels or {}))


def build_tool_execution_benchmark_case(
    name: str,
    execution_engine: ToolExecutionEngine,
    tool_calls: list[ToolCall],
    *,
    mode: Any,
    context: RequestContext | None = None,
    max_tool_calls: int | None = None,
    labels: dict[str, str] | None = None,
) -> BenchmarkCase:
    async def _runner() -> dict[str, Any]:
        batch = await execution_engine.execute_calls(
            tool_calls,
            mode=mode,
            request_context=context,
            max_tool_calls=max_tool_calls,
        )
        durations = [float(result.duration_ms or 0.0) for result in batch.results]
        return {
            "tool_call_count": len(batch.results),
            "success_count": sum(1 for result in batch.results if result.success),
            "error_count": sum(1 for result in batch.results if result.status == "error"),
            "avg_tool_duration_ms": mean(durations) if durations else 0.0,
            "max_tool_duration_ms": max(durations) if durations else 0.0,
            "execution_mode": str(mode.value if hasattr(mode, "value") else mode),
        }

    return BenchmarkCase(name=name, category=BenchmarkCategory.TOOLS, runner=_runner, labels=dict(labels or {}))


def build_cache_benchmark_case(
    name: str,
    engine: Any,
    spec: RequestSpec,
    *,
    context: RequestContext | None = None,
    cache_policy: CachePolicy | None = None,
    labels: dict[str, str] | None = None,
) -> BenchmarkCase:
    async def _runner() -> dict[str, Any]:
        if engine.cache is None:
            raise ValueError("Cache benchmark requires an engine with cache configured")
        if hasattr(engine.cache, "reset_stats"):
            engine.cache.reset_stats()
        resolved_policy = cache_policy or CachePolicy.default_response()

        cold_started = time.perf_counter()
        first = await engine.complete(spec, context=context, cache_policy=resolved_policy)
        cold_latency_ms = (time.perf_counter() - cold_started) * 1000

        warm_started = time.perf_counter()
        second = await engine.complete(spec, context=context, cache_policy=resolved_policy)
        warm_latency_ms = (time.perf_counter() - warm_started) * 1000
        stats = engine.cache.get_stats().to_dict()
        return {
            "cold_latency_ms": cold_latency_ms,
            "warm_latency_ms": warm_latency_ms,
            "cache_speedup_ratio": (cold_latency_ms / warm_latency_ms) if warm_latency_ms > 0 else math.inf,
            "hit_rate": stats["hit_rate"],
            "cache_hits": stats["hits"],
            "cache_misses": stats["misses"],
            "first_status": first.status,
            "second_status": second.status,
        }

    return BenchmarkCase(name=name, category=BenchmarkCategory.CACHE, runner=_runner, labels=dict(labels or {}))


def build_failover_benchmark_case(
    name: str,
    engine: Any,
    spec: RequestSpec,
    *,
    context: RequestContext | None = None,
    timeout: float | None = None,
    attempt_counters: dict[str, Callable[[], int]] | None = None,
    labels: dict[str, str] | None = None,
) -> BenchmarkCase:
    async def _runner() -> dict[str, Any]:
        baseline_counts = {key: counter() for key, counter in dict(attempt_counters or {}).items()}
        started = time.perf_counter()
        result = await engine.complete(spec, context=context, timeout=timeout)
        latency_ms = (time.perf_counter() - started) * 1000
        deltas = {
            key: counter() - baseline_counts.get(key, 0) for key, counter in dict(attempt_counters or {}).items()
        }
        providers_attempted = sum(max(0, value) for value in deltas.values()) if deltas else 1
        return {
            "failover_latency_ms": latency_ms,
            "status": result.status,
            "ok": bool(result.ok),
            "providers_attempted": providers_attempted,
            "fallback_count": max(0, providers_attempted - 1),
            "attempt_deltas": deltas,
        }

    return BenchmarkCase(name=name, category=BenchmarkCategory.FAILOVER, runner=_runner, labels=dict(labels or {}))


def build_context_planning_benchmark_case(
    name: str,
    planner: ContextPlanner,
    request: ContextPlanningRequest,
    *,
    labels: dict[str, str] | None = None,
) -> BenchmarkCase:
    async def _runner() -> dict[str, Any]:
        started = time.perf_counter()
        plan = await planner.plan(request)
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "context_latency_ms": latency_ms,
            "selected_entries": len(plan.entries),
            "memory_entries": len(plan.memory),
            "has_summary": bool(plan.summary),
            "has_persistent_summary": bool(plan.persistent_summary),
        }

    return BenchmarkCase(name=name, category=BenchmarkCategory.CONTEXT, runner=_runner, labels=dict(labels or {}))


def build_structured_quality_benchmark_case(
    name: str,
    cases: list[Any],
    *,
    hooks: HookManager | None = None,
    labels: dict[str, str] | None = None,
) -> BenchmarkCase:
    async def _runner() -> dict[str, Any]:
        from .structured_benchmarks import benchmark_structured_cases

        report = await benchmark_structured_cases(cases, hooks=hooks)
        return {
            "total_cases": report.total_cases,
            "success_rate": report.success_rate,
            "repaired_success_rate": report.repaired_success_rate,
            "repaired_share_of_successes": report.repaired_share_of_successes,
            "avg_latency_ms": report.avg_latency_ms,
            "avg_repair_attempts": report.avg_repair_attempts,
            "max_repair_attempts": report.max_repair_attempts,
            "repair_attempt_histogram": dict(report.repair_attempt_histogram),
        }

    return BenchmarkCase(name=name, category=BenchmarkCategory.STRUCTURED, runner=_runner, labels=dict(labels or {}))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "BenchmarkCase",
    "BenchmarkCategory",
    "BenchmarkComparisonRecord",
    "BenchmarkComparisonReport",
    "BenchmarkRecord",
    "BenchmarkReport",
    "BenchmarkRunMetadata",
    "BenchmarkRunMode",
    "build_cache_benchmark_case",
    "build_completion_benchmark_case",
    "build_context_planning_benchmark_case",
    "build_embeddings_benchmark_case",
    "build_failover_benchmark_case",
    "build_stream_benchmark_case",
    "build_structured_quality_benchmark_case",
    "build_tool_execution_benchmark_case",
    "compare_benchmark_reports",
    "load_benchmark_report",
    "run_benchmarks",
    "save_benchmark_report",
]
