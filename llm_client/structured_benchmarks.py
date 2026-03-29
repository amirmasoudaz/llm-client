"""
Deterministic benchmark harness for structured-output quality metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .hooks import HookManager
from .structured import StructuredOutputConfig, StructuredResult, extract_structured


@dataclass(frozen=True)
class StructuredBenchmarkCase:
    name: str
    provider: Any
    messages: list[Any]
    config: StructuredOutputConfig
    engine: Any | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructuredBenchmarkRecord:
    name: str
    valid: bool
    repair_attempts: int
    latency_ms: float
    response_kind: str
    validation_errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StructuredBenchmarkReport:
    total_cases: int
    success_count: int
    repaired_success_count: int
    avg_latency_ms: float
    avg_repair_attempts: float
    max_repair_attempts: int
    repair_attempt_histogram: dict[int, int]
    records: list[StructuredBenchmarkRecord] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_cases if self.total_cases else 0.0

    @property
    def repaired_success_rate(self) -> float:
        return self.repaired_success_count / self.total_cases if self.total_cases else 0.0

    @property
    def repaired_share_of_successes(self) -> float:
        return self.repaired_success_count / self.success_count if self.success_count else 0.0


async def benchmark_structured_cases(
    cases: list[StructuredBenchmarkCase],
    *,
    hooks: HookManager | None = None,
) -> StructuredBenchmarkReport:
    records: list[StructuredBenchmarkRecord] = []
    histogram: dict[int, int] = {}
    total_latency_ms = 0.0
    total_repairs = 0
    success_count = 0
    repaired_success_count = 0

    for case in cases:
        start = time.monotonic()
        result = await extract_structured(
            case.provider,
            case.messages,
            case.config,
            engine=case.engine,
            **dict(case.kwargs),
        )
        latency_ms = (time.monotonic() - start) * 1000
        record = _record_for_result(case.name, result, latency_ms)
        records.append(record)
        if hooks is not None:
            await hooks.emit(
                "benchmark.case",
                {
                    "name": record.name,
                    "valid": record.valid,
                    "repair_attempts": record.repair_attempts,
                    "latency_ms": record.latency_ms,
                    "response_kind": record.response_kind,
                    "validation_errors": list(record.validation_errors),
                },
                None,
            )

        total_latency_ms += latency_ms
        total_repairs += record.repair_attempts
        histogram[record.repair_attempts] = histogram.get(record.repair_attempts, 0) + 1
        if record.valid:
            success_count += 1
            if record.repair_attempts > 0:
                repaired_success_count += 1

    total_cases = len(records)
    report = StructuredBenchmarkReport(
        total_cases=total_cases,
        success_count=success_count,
        repaired_success_count=repaired_success_count,
        avg_latency_ms=(total_latency_ms / total_cases) if total_cases else 0.0,
        avg_repair_attempts=(total_repairs / total_cases) if total_cases else 0.0,
        max_repair_attempts=max(histogram) if histogram else 0,
        repair_attempt_histogram=histogram,
        records=records,
    )
    if hooks is not None:
        await hooks.emit(
            "benchmark.report",
            {
                "total_cases": report.total_cases,
                "success_rate": report.success_rate,
                "repaired_success_rate": report.repaired_success_rate,
                "repaired_share_of_successes": report.repaired_share_of_successes,
                "avg_latency_ms": report.avg_latency_ms,
                "avg_repair_attempts": report.avg_repair_attempts,
                "max_repair_attempts": report.max_repair_attempts,
                "repair_attempt_histogram": dict(report.repair_attempt_histogram),
            },
            None,
        )
    return report


def _record_for_result(name: str, result: StructuredResult, latency_ms: float) -> StructuredBenchmarkRecord:
    return StructuredBenchmarkRecord(
        name=name,
        valid=result.valid,
        repair_attempts=result.repair_attempts,
        latency_ms=latency_ms,
        response_kind=result.response_kind,
        validation_errors=list(result.validation_errors),
    )


__all__ = [
    "StructuredBenchmarkCase",
    "StructuredBenchmarkRecord",
    "StructuredBenchmarkReport",
    "benchmark_structured_cases",
]
