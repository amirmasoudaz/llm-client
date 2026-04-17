from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_client import (
    BenchmarkComparisonReport,
    BenchmarkRecorder,
    BenchmarkRecord,
    BenchmarkRunMode,
    ExecutionEngine,
    HookManager,
    Message,
    OpenAIProvider,
    RequestSpec,
    StructuredOutputConfig,
    build_cache_benchmark_case,
    build_completion_benchmark_case,
    build_context_planning_benchmark_case,
    build_embeddings_benchmark_case,
    build_stream_benchmark_case,
    build_structured_quality_benchmark_case,
    compare_benchmark_reports,
    load_benchmark_report,
    load_env,
    run_benchmarks,
    save_benchmark_report,
)
from llm_client.cache import CacheCore, CachePolicy
from llm_client.cache.base import BaseCacheBackend
from llm_client.context_planning import (
    ContextPlanningRequest,
    DefaultMemoryRetrievalStrategy,
    HeuristicContextPlanner,
    TieredTrimmingStrategy,
)
from llm_client.memory import InMemorySummaryStore, MemoryWrite, ShortTermMemoryStore
from llm_client.structured_benchmarks import StructuredBenchmarkCase
from llm_client.summarization import LLMSummarizer

load_env()

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "contracts" / "benchmarks" / "llm_client_deterministic_baseline.v1.json"
ARTIFACT_PATH = ROOT / "tmp" / "cookbook-live-benchmark-report.json"


class _InMemoryCacheBackend(BaseCacheBackend):
    name = "fs"
    default_collection = "cookbook-benchmarks"

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], dict[str, object]] = {}

    async def ensure_ready(self) -> None:
        return None

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        return (effective_key, collection or self.default_collection) in self._entries

    async def read(self, effective_key: str, collection: str | None = None) -> dict[str, object] | None:
        return self._entries.get((effective_key, collection or self.default_collection))

    async def write(
        self,
        effective_key: str,
        response: dict[str, object],
        model_name: str,
        collection: str | None = None,
    ) -> None:
        _ = model_name
        self._entries[(effective_key, collection or self.default_collection)] = dict(response)


@dataclass(frozen=True)
class _Entry:
    role: str
    content: str
    entry_type: str = "message"


def _build_live_structured_cases(live_provider: OpenAIProvider, engine: ExecutionEngine) -> list[StructuredBenchmarkCase]:
    return [
        StructuredBenchmarkCase(
            name="incident_triage_extract",
            provider=live_provider,
            engine=engine,
            messages=[
                Message.system("Return valid JSON only."),
                Message.user(
                    "Extract severity, owner, and primary_risk from this note.\n"
                    "Incident note: Workspace export jobs are timing out after audit logging was enabled. "
                    "Owner: platform operations. Impact: finance month-end reconciliation is blocked."
                ),
            ],
            config=StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["critical", "major", "minor", "unknown"]},
                        "owner": {"type": "string"},
                        "primary_risk": {"type": "string"},
                    },
                    "required": ["severity", "owner", "primary_risk"],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
            kwargs={"reasoning_effort": "low"},
        ),
        StructuredBenchmarkCase(
            name="release_gate_extract",
            provider=live_provider,
            engine=engine,
            messages=[
                Message.system("Return valid JSON only."),
                Message.user(
                    "Read this release note and return go_live_status, blocker_count, and next_action.\n"
                    "Release note: consumer migration sign-off is pending, dashboards are not validated, "
                    "and rollback rehearsal was skipped this week."
                ),
            ],
            config=StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "go_live_status": {"type": "string", "enum": ["go", "hold", "unknown"]},
                        "blocker_count": {"type": "integer"},
                        "next_action": {"type": "string"},
                    },
                    "required": ["go_live_status", "blocker_count", "next_action"],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
            kwargs={"reasoning_effort": "low"},
        ),
    ]


def _suite_summary(records: list[BenchmarkRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metric_summary: dict[str, Any] = {}
        if record.category.value == "completion":
            for key in ("avg_completion_latency_ms", "avg_total_tokens", "success_rate"):
                if key in record.metrics:
                    metric_summary[key] = record.metrics[key]
        elif record.category.value == "stream":
            for key in ("avg_first_token_latency_ms", "avg_full_stream_latency_ms", "avg_stream_token_events", "success_rate"):
                if key in record.metrics:
                    metric_summary[key] = record.metrics[key]
        elif record.category.value == "embeddings":
            for key in ("avg_embedding_latency_ms", "throughput_items_per_second", "avg_embedding_count"):
                if key in record.metrics:
                    metric_summary[key] = record.metrics[key]
        elif record.category.value == "cache":
            cold_latency = record.metrics.get("cold_latency_ms")
            warm_latency = record.metrics.get("warm_latency_ms")
            ratio = record.metrics.get("cache_speedup_ratio")
            if cold_latency is not None:
                metric_summary["cold_latency_ms"] = cold_latency
            if warm_latency is not None:
                metric_summary["warm_latency_ms"] = warm_latency
            if ratio is not None:
                metric_summary["cache_speedup_ratio"] = ratio
                metric_summary["cache_speedup_human"] = _format_speedup_ratio(ratio)
            for key in ("cache_hits", "cache_misses", "hit_rate"):
                if key in record.metrics:
                    metric_summary[key] = record.metrics[key]
        elif record.category.value == "context":
            for key in ("context_latency_ms", "selected_entries", "memory_entries", "has_summary", "has_persistent_summary"):
                if key in record.metrics:
                    metric_summary[key] = record.metrics[key]
        elif record.category.value == "structured":
            for key in ("success_rate", "repaired_success_rate", "avg_latency_ms", "avg_repair_attempts", "max_repair_attempts"):
                if key in record.metrics:
                    metric_summary[key] = record.metrics[key]
        else:
            metric_summary = dict(record.metrics)
        rows.append(
            {
                "name": record.name,
                "category": record.category.value,
                "status": record.status,
                "case_latency_ms": round(record.case_latency_ms, 2),
                "metrics": metric_summary,
            }
        )
    return rows


def _comparison_summary(comparison: BenchmarkComparisonReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in comparison.records:
        interesting = {
            key: value
            for key, value in record.metric_deltas.items()
            if key
            in {
                "avg_completion_latency_ms",
                "avg_first_token_latency_ms",
                "avg_full_stream_latency_ms",
                "avg_embedding_latency_ms",
                "cache_speedup_ratio",
                "context_latency_ms",
                "success_rate",
                "repaired_success_rate",
                "avg_latency_ms",
            }
        }
        rows.append(
            {
                "name": record.name,
                "category": record.category.value,
                "metric_deltas": interesting,
            }
        )
    return rows


def _format_speedup_ratio(value: Any) -> str:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if ratio == float("inf"):
        return "instant-warm-path"
    if ratio > 1000.0:
        return f">{1000:.0f}x"
    return f"{ratio:.2f}x"


def _benchmark_event_summary(recorder: BenchmarkRecorder, report: Any) -> dict[str, Any]:
    top_level_case_names = {record.name for record in report.records}
    nested_case_events = [
        snapshot.payload
        for snapshot in recorder.cases
        if str(snapshot.payload.get("name") or "") not in top_level_case_names
    ]
    nested_report_events = [
        snapshot.payload
        for snapshot in recorder.reports
        if int(snapshot.payload.get("total_cases", -1) or -1) != report.total_cases
    ]
    return {
        "top_level_case_events": report.total_cases,
        "top_level_report_events": 1,
        "nested_case_events": len(nested_case_events),
        "nested_report_events": len(nested_report_events),
        "nested_case_names": [str(item.get("name") or "") for item in nested_case_events],
    }


async def main() -> None:
    provider_name = "openai"
    chat_model = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    embedding_model = os.getenv("LLM_CLIENT_EXAMPLE_EMBEDDINGS_MODEL", "text-embedding-3-small")
    chat_provider = OpenAIProvider(model=chat_model)
    embeddings_provider = OpenAIProvider(model=embedding_model)

    try:
        recorder = BenchmarkRecorder()
        hooks = HookManager([recorder])

        primary_engine = ExecutionEngine(provider=chat_provider)
        cache_engine = ExecutionEngine(
            provider=chat_provider,
            cache=CacheCore(_InMemoryCacheBackend()),
        )
        embeddings_engine = ExecutionEngine(provider=embeddings_provider)

        memory_store = ShortTermMemoryStore()
        await memory_store.write(
            MemoryWrite(
                content="Customer-support reviewers care most about measurable impact, deployment realism, and explicit rollback plans.",
                scope="bench-context",
            )
        )
        await memory_store.write(
            MemoryWrite(
                content="Previous review cycles flagged weak deployment sequencing and unclear safety validation milestones.",
                scope="bench-context",
            )
        )
        planner = HeuristicContextPlanner(
            trimming_strategy=TieredTrimmingStrategy(tier1_tail=4),
            summarization_strategy=LLMSummarizer(engine=primary_engine),
            memory_reader=memory_store,
            retrieval_strategy=DefaultMemoryRetrievalStrategy(default_scope="bench-context", default_limit=2),
            summary_store=InMemorySummaryStore(),
        )

        completion_spec = RequestSpec(
            provider=provider_name,
            model=chat_model,
            messages=[
                Message.system("You are benchmarking completion quality for release operations."),
                Message.user("Explain, in 3 short bullets, why cache-aware retries help an LLM platform."),
            ],
        )
        stream_spec = RequestSpec(
            provider=provider_name,
            model=chat_model,
            messages=[
                Message.system("You are benchmarking streaming responsiveness."),
                Message.user("Stream a concise explanation of why observability matters during incident response."),
            ],
        )
        cache_spec = RequestSpec(
            provider=provider_name,
            model=chat_model,
            messages=[
                Message.system("You are benchmarking cacheable support triage generation."),
                Message.user("Write a short support triage summary for an export-timeout incident."),
            ],
        )
        context_request = ContextPlanningRequest(
            entries=[
                _Entry("user", "We are preparing a grant-style deployment brief for robotics automation."),
                _Entry("assistant", "What makes reviewers nervous?"),
                _Entry("user", "They dislike ambitious autonomy claims without phased validation."),
                _Entry("assistant", "What proof points do you have?"),
                _Entry("user", "A small pilot reduced operator intervention by 31%."),
                _Entry("assistant", "What help do you want now?"),
                _Entry("user", "I need a concise deployment-realism framing with risks and next move."),
            ],
            current_message="Build the strongest concise context packet for a deployment-realism benchmark.",
            max_entries=4,
            summarize_when_truncated=True,
            persist_summary=True,
            summary_scope="bench-context",
            max_memory_entries=2,
        )

        cases = [
            build_completion_benchmark_case(
                "completion_smoke",
                primary_engine,
                completion_spec,
                iterations=2,
                labels={"provider": provider_name, "model": chat_model},
            ),
            build_stream_benchmark_case(
                "stream_smoke",
                primary_engine,
                stream_spec,
                iterations=2,
                labels={"provider": provider_name, "model": chat_model},
            ),
            build_embeddings_benchmark_case(
                "embeddings_smoke",
                embeddings_engine,
                [
                    "cache-aware retries reduce duplicate work during transient failures",
                    "observability lets operators diagnose routing, latency, and quota problems",
                ],
                iterations=2,
                labels={"provider": provider_name, "model": embedding_model},
            ),
            build_cache_benchmark_case(
                "cache_smoke",
                cache_engine,
                cache_spec,
                cache_policy=CachePolicy.default_response(collection="cookbook-benchmark-cache"),
                labels={"provider": provider_name, "model": chat_model},
            ),
            build_context_planning_benchmark_case(
                "context_smoke",
                planner,
                context_request,
                labels={"provider": provider_name, "model": chat_model},
            ),
            build_structured_quality_benchmark_case(
                "structured_smoke",
                _build_live_structured_cases(chat_provider, primary_engine),
                hooks=hooks,
                labels={"provider": provider_name, "model": chat_model},
            ),
        ]

        report = await run_benchmarks(
            cases,
            label="cookbook-live-benchmarks",
            mode=BenchmarkRunMode.LIVE,
            hooks=hooks,
            tags={"provider": provider_name, "model": chat_model},
        )
        artifact_path = save_benchmark_report(report, ARTIFACT_PATH)
        baseline = load_benchmark_report(BASELINE_PATH)
        comparison = compare_benchmark_reports(report, baseline)

        print("\n=== Benchmark Suite ===\n")
        print(
            json.dumps(
                {
                    "run_label": report.metadata.label,
                    "mode": report.metadata.mode.value,
                    "chat_provider": {"provider": provider_name, "model": chat_model},
                    "embeddings_provider": {"provider": provider_name, "model": embedding_model},
                    "cases": [
                        {"name": case.name, "category": case.category.value, "labels": case.labels}
                        for case in cases
                    ],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Benchmark Summary ===\n")
        print(
            json.dumps(
                {
                    "total_cases": report.total_cases,
                    "success_count": report.success_count,
                    "failed_count": report.failed_count,
                    "avg_case_latency_ms": report.avg_case_latency_ms,
                    "records": _suite_summary(report.records),
                    "benchmark_events": _benchmark_event_summary(recorder, report),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Comparison To Deterministic Baseline ===\n")
        print(
            json.dumps(
                {
                    "baseline_label": baseline.metadata.label,
                    "current_label": report.metadata.label,
                    "records": _comparison_summary(comparison),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Benchmark Artifact ===\n")
        print(
            json.dumps(
                {
                    "report_path": str(artifact_path),
                    "baseline_path": str(BASELINE_PATH),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        await chat_provider.close()
        await embeddings_provider.close()


if __name__ == "__main__":
    asyncio.run(main())
