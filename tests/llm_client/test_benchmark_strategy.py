from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from llm_client.agent import ToolExecutionMode
from llm_client.benchmarks import (
    BenchmarkCategory,
    BenchmarkRunMode,
    build_cache_benchmark_case,
    build_completion_benchmark_case,
    build_context_planning_benchmark_case,
    build_embeddings_benchmark_case,
    build_failover_benchmark_case,
    build_stream_benchmark_case,
    build_structured_quality_benchmark_case,
    build_tool_execution_benchmark_case,
    compare_benchmark_reports,
    load_benchmark_report,
    run_benchmarks,
    save_benchmark_report,
)
from llm_client.cache import CacheCore, CachePolicy
from llm_client.cache.base import BaseCacheBackend
from llm_client.context_planning import (
    ContextPlanningRequest,
    DefaultMemoryRetrievalStrategy,
    HeuristicContextPlanner,
)
from llm_client.engine import ExecutionEngine, RetryConfig
from llm_client.hooks import BenchmarkRecorder, HookManager
from llm_client.memory import MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import (
    CompletionResult,
    EmbeddingResult,
    Message,
    StreamEvent,
    StreamEventType,
    ToolCall,
    Usage,
)
from llm_client.routing import StaticRouter
from llm_client.spec import RequestSpec
from llm_client.structured import StructuredOutputConfig
from llm_client.structured_benchmarks import StructuredBenchmarkCase
from llm_client.tools import Tool, ToolExecutionEngine, ToolRegistry
from tests.llm_client.fakes import ScriptedProvider, ok_result


class _InMemoryCacheBackend(BaseCacheBackend):
    name = "fs"
    default_collection = "bench"

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str | None], dict] = {}

    async def ensure_ready(self) -> None:
        return None

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        return (effective_key, collection or self.default_collection) in self._entries

    async def read(self, effective_key: str, collection: str | None = None) -> dict | None:
        return self._entries.get((effective_key, collection or self.default_collection))

    async def write(
        self,
        effective_key: str,
        response: dict,
        model_name: str,
        collection: str | None = None,
    ) -> None:
        _ = model_name
        self._entries[(effective_key, collection or self.default_collection)] = dict(response)


class _DelayedProvider(ScriptedProvider):
    def __init__(
        self,
        *,
        complete_delay: float = 0.002,
        stream_delay: float = 0.002,
        embed_delay: float = 0.002,
        model_name: str = "gpt-5-mini",
    ) -> None:
        super().__init__(model_name=model_name)
        self.complete_delay = complete_delay
        self.stream_delay = stream_delay
        self.embed_delay = embed_delay

    async def complete(self, messages, **kwargs):
        _ = (messages, kwargs)
        self.complete_calls.append({"messages": messages, "kwargs": kwargs})
        await asyncio.sleep(self.complete_delay)
        return ok_result("ok", model=self.model_name)

    async def stream(self, messages, **kwargs):
        _ = (messages, kwargs)
        self.stream_calls.append({"messages": messages, "kwargs": kwargs})
        await asyncio.sleep(self.stream_delay)
        yield StreamEvent(type=StreamEventType.TOKEN, data="hello")
        await asyncio.sleep(self.stream_delay)
        yield StreamEvent(type=StreamEventType.DONE, data=ok_result("hello", model=self.model_name))

    async def embed(self, inputs, **kwargs):
        _ = kwargs
        self.embed_calls.append({"inputs": inputs, "kwargs": kwargs})
        await asyncio.sleep(self.embed_delay)
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        return EmbeddingResult(
            embeddings=[[0.0, 1.0, 0.5] for _ in texts],
            usage=Usage(input_tokens=len(texts), total_tokens=len(texts)),
            model=self.model_name,
            status=200,
        )


@dataclass(frozen=True)
class _Entry:
    role: str
    content: str
    entry_type: str = "message"


def _spec(*, provider: str = "openai") -> RequestSpec:
    return RequestSpec(provider=provider, model="gpt-5-mini", messages=[Message.user("hello")])


def _tool_registry() -> ToolRegistry:
    async def echo_tool(text: str) -> str:
        await asyncio.sleep(0.001)
        return text.upper()

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="echo",
            description="Echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            handler=echo_tool,
        )
    )
    return registry


def _structured_cases() -> list[StructuredBenchmarkCase]:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    return [
        StructuredBenchmarkCase(
            name="structured_immediate",
            provider=ScriptedProvider(complete_script=[ok_result('{"value": 1}')]),
            messages=[Message.user("return json")],
            config=StructuredOutputConfig(schema=schema, max_repair_attempts=1),
        ),
        StructuredBenchmarkCase(
            name="structured_repaired",
            provider=ScriptedProvider(
                complete_script=[ok_result('{"value": "oops"}'), ok_result('{"value": 2}')]
            ),
            messages=[Message.user("return json")],
            config=StructuredOutputConfig(schema=schema, max_repair_attempts=1),
        ),
    ]


@pytest.mark.asyncio
async def test_benchmark_harness_covers_core_categories() -> None:
    provider = _DelayedProvider()
    cache_engine = ExecutionEngine(
        provider=_DelayedProvider(),
        cache=CacheCore(_InMemoryCacheBackend()),
    )
    first = ScriptedProvider(complete_script=[CompletionResult(status=503, error="unavailable", model="gpt-5-mini")])
    second = _DelayedProvider()
    failover_engine = ExecutionEngine(
        router=StaticRouter([first, second]),
        retry=RetryConfig(attempts=1, backoff=0.0, max_backoff=0.0),
    )

    memory_store = ShortTermMemoryStore()
    await memory_store.write(MemoryWrite(content="funding application status", scope="thread-1"))
    planner = HeuristicContextPlanner(
        memory_reader=memory_store,
        retrieval_strategy=DefaultMemoryRetrievalStrategy(default_scope="thread-1"),
    )
    recorder = BenchmarkRecorder()

    tool_engine = ToolExecutionEngine(_tool_registry())
    cases = [
        build_completion_benchmark_case("completion_smoke", ExecutionEngine(provider=provider), _spec(), iterations=2),
        build_stream_benchmark_case("stream_smoke", ExecutionEngine(provider=_DelayedProvider()), _spec(), iterations=2),
        build_embeddings_benchmark_case(
            "embeddings_smoke",
            ExecutionEngine(provider=_DelayedProvider()),
            ["alpha", "beta"],
            iterations=2,
        ),
        build_tool_execution_benchmark_case(
            "tool_smoke",
            tool_engine,
            [ToolCall(id="1", name="echo", arguments='{"text": "hi"}')],
            mode=ToolExecutionMode.SINGLE,
        ),
        build_cache_benchmark_case(
            "cache_smoke",
            cache_engine,
            _spec(),
            cache_policy=CachePolicy.default_response(collection="bench"),
        ),
        build_failover_benchmark_case(
            "failover_smoke",
            failover_engine,
            _spec(),
            attempt_counters={
                "first": lambda: len(first.complete_calls),
                "second": lambda: len(second.complete_calls),
            },
        ),
        build_context_planning_benchmark_case(
            "context_smoke",
            planner,
            ContextPlanningRequest(
                entries=[_Entry(role="user", content="Need help with my funding application")],
                current_message="funding application",
                max_entries=1,
                summary_scope="thread-1",
            ),
        ),
        build_structured_quality_benchmark_case("structured_smoke", _structured_cases()),
    ]

    report = await run_benchmarks(
        cases,
        label="phase14-smoke",
        hooks=HookManager([recorder]),
    )

    assert report.total_cases == 8
    assert report.success_count == 8
    assert report.metadata.label == "phase14-smoke"
    assert len(recorder.cases) == 8
    assert len(recorder.reports) == 1
    by_name = {record.name: record for record in report.records}
    assert by_name["completion_smoke"].category is BenchmarkCategory.COMPLETION
    assert by_name["completion_smoke"].metrics["avg_completion_latency_ms"] > 0
    assert by_name["stream_smoke"].metrics["avg_first_token_latency_ms"] > 0
    assert by_name["embeddings_smoke"].metrics["throughput_items_per_second"] > 0
    assert by_name["tool_smoke"].metrics["success_count"] == 1
    assert by_name["cache_smoke"].metrics["cache_hits"] >= 1
    assert by_name["failover_smoke"].metrics["fallback_count"] >= 1
    assert by_name["context_smoke"].metrics["memory_entries"] >= 1
    assert by_name["structured_smoke"].metrics["success_rate"] == 1.0


@pytest.mark.asyncio
async def test_benchmark_live_mode_is_explicitly_labeled() -> None:
    case = build_completion_benchmark_case(
        "completion_live",
        ExecutionEngine(provider=_DelayedProvider()),
        _spec(),
    )

    report = await run_benchmarks([case], label="provider-smoke", mode=BenchmarkRunMode.LIVE)

    assert report.metadata.mode is BenchmarkRunMode.LIVE
    assert report.metadata.label == "live:provider-smoke"


@pytest.mark.asyncio
async def test_benchmark_reports_can_be_stored_and_compared() -> None:
    report = await run_benchmarks(
        [
            build_completion_benchmark_case(
                "completion_smoke",
                ExecutionEngine(provider=_DelayedProvider()),
                _spec(),
            ),
            build_stream_benchmark_case(
                "stream_smoke",
                ExecutionEngine(provider=_DelayedProvider()),
                _spec(),
            ),
        ],
        label="phase14-compare",
    )

    saved_path = save_benchmark_report(report, Path("tmp/phase14-benchmark-report.json"))
    reloaded = load_benchmark_report(saved_path)
    baseline = load_benchmark_report(
        Path("contracts/benchmarks/llm_client_deterministic_baseline.v1.json")
    )
    comparison = compare_benchmark_reports(reloaded, baseline)

    assert reloaded.total_cases == report.total_cases
    assert reloaded.metadata.label == "phase14-compare"
    assert comparison.baseline_label == baseline.metadata.label
    assert {record.name for record in comparison.records} >= {"completion_smoke", "stream_smoke"}
