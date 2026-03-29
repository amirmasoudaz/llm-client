from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_client.context_planning import (
    ContextPlanningRequest,
    DefaultMemoryRetrievalStrategy,
    HeuristicContextPlanner,
    SemanticRelevanceSelector,
    SlidingWindowTrimmingStrategy,
    TieredTrimmingStrategy,
)
from llm_client.context_assembly import (
    ContextAssemblyRequest,
    ContextSourcePayload,
    MultiSourceContextAssembler,
)
from llm_client.memory import InMemorySummaryStore, MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.observability import ContextPlanningRecorder, HookManager
from llm_client.providers.types import Message
from llm_client.summarization import NoOpSummarizer, SummarizationRequest, SummarizationResult


@dataclass(frozen=True)
class _HistoryEntry:
    role: str
    content: str
    entry_type: str = "message"


class _StubSummarizer(NoOpSummarizer):
    async def summarize_request(self, request: SummarizationRequest):  # type: ignore[override]
        return SummarizationResult(
            summary=f"summary:{len(request.messages)}",
            metadata=dict(request.metadata),
        )


class _PlaceholderSummarizer(NoOpSummarizer):
    async def summarize_request(self, request: SummarizationRequest):  # type: ignore[override]
        return SummarizationResult(
            summary="[Unable to summarize]",
            metadata=dict(request.metadata),
        )


class _StubSourceLoader:
    def __init__(self, name: str, *, entries: list[_HistoryEntry] | None = None, summary: str | None = None) -> None:
        self.name = name
        self.entries = list(entries or [])
        self.summary = summary

    async def load(self, request):  # type: ignore[no-untyped-def]
        return ContextSourcePayload(
            source_name=self.name,
            entries=list(self.entries),
            summary=self.summary,
            metadata={"scope": getattr(request, "scope", None)},
        )


@pytest.mark.asyncio
async def test_short_term_memory_store_supports_scope_limit_and_eviction() -> None:
    store = ShortTermMemoryStore(max_entries=3)
    await store.write(MemoryWrite(content="global note"))
    await store.write(MemoryWrite(content="thread note 1", scope="thread:1", metadata={"topic": "funding"}))
    await store.write(MemoryWrite(content="thread note 2", scope="thread:1", metadata={"topic": "funding"}))
    await store.write(MemoryWrite(content="other scope", scope="thread:2"))

    thread_results = await store.retrieve(MemoryQuery(scope="thread:1", limit=5))
    assert [entry.content for entry in thread_results] == ["thread note 2", "thread note 1"]

    filtered = await store.retrieve(MemoryQuery(scope="thread:1", limit=5, metadata={"topic": "funding"}))
    assert len(filtered) == 2

    all_entries = await store.list_all()
    assert len(all_entries) == 3
    assert "global note" not in [entry.content for entry in all_entries]


@pytest.mark.asyncio
async def test_heuristic_context_planner_combines_trimming_memory_and_summary() -> None:
    store = ShortTermMemoryStore()
    await store.write(MemoryWrite(content="User prefers concise answers", scope="thread:1", relevance=0.8))

    planner = HeuristicContextPlanner(
        trimming_strategy=SlidingWindowTrimmingStrategy(),
        summarization_strategy=_StubSummarizer(),
        memory_reader=store,
    )

    entries = [_HistoryEntry(role="user", content=f"message {index}") for index in range(6)]
    plan = await planner.plan(
        ContextPlanningRequest(
            entries=entries,
            current_message="latest funding message",
            max_entries=3,
            memory_query=MemoryQuery(scope="thread:1", query="concise funding", limit=5),
            summarize_when_truncated=True,
            summary_max_tokens=64,
            metadata={"source": "test"},
        )
    )

    assert len(plan.entries) == 3
    assert plan.truncation is not None and plan.truncation.truncated is True
    assert plan.summary == "summary:3"
    assert len(plan.memory) == 1
    assert plan.memory[0].content == "User prefers concise answers"
    assert plan.metadata["memory_entries"] == 1


@pytest.mark.asyncio
async def test_context_planner_supports_persistent_summary_store() -> None:
    summary_store = InMemorySummaryStore()
    planner = HeuristicContextPlanner(
        trimming_strategy=SlidingWindowTrimmingStrategy(),
        summarization_strategy=_StubSummarizer(),
        summary_store=summary_store,
    )
    entries = [_HistoryEntry(role="user", content=f"message {index}") for index in range(6)]

    first_plan = await planner.plan(
        ContextPlanningRequest(
            entries=entries,
            max_entries=3,
            summarize_when_truncated=True,
            persist_summary=True,
            summary_scope="thread:1",
        )
    )
    second_plan = await planner.plan(
        ContextPlanningRequest(
            entries=[_HistoryEntry(role="user", content="short thread")],
            max_entries=5,
            summary_scope="thread:1",
        )
    )

    assert first_plan.summary == "summary:3"
    assert first_plan.persistent_summary == "summary:3"
    assert first_plan.metadata["persistent_summary_loaded"] is False
    assert first_plan.metadata["persistent_summary_updated"] is True
    assert second_plan.summary == "summary:3"
    assert second_plan.persistent_summary == "summary:3"
    assert second_plan.metadata["persistent_summary_loaded"] is True
    assert second_plan.metadata["persistent_summary_updated"] is False


@pytest.mark.asyncio
async def test_context_planner_does_not_persist_placeholder_summaries() -> None:
    summary_store = InMemorySummaryStore()
    planner = HeuristicContextPlanner(
        trimming_strategy=SlidingWindowTrimmingStrategy(),
        summarization_strategy=_PlaceholderSummarizer(),
        summary_store=summary_store,
    )
    entries = [_HistoryEntry(role="user", content=f"message {index}") for index in range(6)]

    plan = await planner.plan(
        ContextPlanningRequest(
            entries=entries,
            max_entries=3,
            summarize_when_truncated=True,
            persist_summary=True,
            summary_scope="thread:placeholder",
        )
    )

    assert plan.summary == "[Unable to summarize]"
    assert plan.persistent_summary is None
    assert plan.metadata["persistent_summary_loaded"] is False
    assert plan.metadata["persistent_summary_updated"] is False


@pytest.mark.asyncio
async def test_context_planner_supports_retrieval_strategy_hooks() -> None:
    store = ShortTermMemoryStore()
    await store.write(MemoryWrite(content="Research focus: AI", scope="thread:9"))

    planner = HeuristicContextPlanner(
        memory_reader=store,
        retrieval_strategy=DefaultMemoryRetrievalStrategy(default_scope="thread:9"),
    )
    plan = await planner.plan(
        ContextPlanningRequest(
            entries=[_HistoryEntry(role="user", content="hello")],
            current_message="Need AI outreach help",
            max_memory_entries=3,
        )
    )

    assert len(plan.memory) == 1
    assert plan.memory[0].content == "Research focus: AI"
    assert plan.metadata["retrieval_used"] is True


@pytest.mark.asyncio
async def test_default_memory_retrieval_strategy_does_not_treat_request_metadata_as_memory_filter() -> None:
    store = ShortTermMemoryStore()
    await store.write(MemoryWrite(content="Relevant robotics funding note", scope="thread:42", metadata={"kind": "note"}))

    planner = HeuristicContextPlanner(
        memory_reader=store,
        retrieval_strategy=DefaultMemoryRetrievalStrategy(default_scope="thread:42"),
    )
    plan = await planner.plan(
        ContextPlanningRequest(
            entries=[_HistoryEntry(role="user", content="hello")],
            current_message="Need robotics funding help",
            max_memory_entries=3,
            metadata={"scenario": "showcase"},
        )
    )

    assert len(plan.memory) == 1
    assert plan.memory[0].content == "Relevant robotics funding note"


@pytest.mark.asyncio
async def test_context_planner_emits_observability_events() -> None:
    recorder = ContextPlanningRecorder()
    planner = HeuristicContextPlanner(
        trimming_strategy=SlidingWindowTrimmingStrategy(),
        hook_manager=HookManager([recorder]),
    )
    plan = await planner.plan(
        ContextPlanningRequest(
            entries=[_HistoryEntry(role="user", content=f"message {index}") for index in range(5)],
            max_entries=2,
        )
    )

    snapshot = recorder.latest()
    assert plan.truncation is not None and plan.truncation.truncated is True
    assert snapshot is not None
    assert snapshot.payload["entries_before"] == 5
    assert snapshot.payload["entries_after"] == 2
    assert snapshot.payload["truncated"] is True


def test_trimming_strategy_interfaces_delegate_to_existing_heuristics() -> None:
    entries = [_HistoryEntry(role="user", content=f"message {index}") for index in range(12)]
    request = ContextPlanningRequest(entries=entries, current_message="message 11", max_entries=5)

    sliding = SlidingWindowTrimmingStrategy().trim(entries, request)
    tiered = TieredTrimmingStrategy(tier1_tail=2).trim(entries, request)

    assert sliding.total_after == 5
    assert tiered.total_after == 5
    assert sliding.truncated is True
    assert tiered.truncated is True


@pytest.mark.asyncio
async def test_semantic_relevance_selector_selects_best_matches() -> None:
    async def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "funding" in lowered else 0.0,
            1.0 if "travel" in lowered else 0.0,
        ]

    entries = [
        _HistoryEntry(role="user", content="travel logistics"),
        _HistoryEntry(role="assistant", content="funding request draft"),
        _HistoryEntry(role="user", content="meal planning"),
    ]
    selector = SemanticRelevanceSelector(embed_fn=_embed)

    selected = await selector.select(entries, current_message="need funding help", limit=1)

    assert len(selected) == 1
    assert selected[0].content == "funding request draft"


@pytest.mark.asyncio
async def test_multi_source_context_assembler_remains_generic() -> None:
    planner = HeuristicContextPlanner(trimming_strategy=SlidingWindowTrimmingStrategy())
    assembler = MultiSourceContextAssembler(
        planner=planner,
        source_loaders=[
            _StubSourceLoader("history", entries=[_HistoryEntry(role="user", content="history item")]),
            _StubSourceLoader("summary", summary="source summary"),
        ],
    )

    result = await assembler.assemble(
        ContextAssemblyRequest(
            current_message="current",
            base_entries=[_HistoryEntry(role="user", content="base item")],
            max_entries=5,
            metadata={"source": "test"},
        )
    )

    assert len(result.sources) == 2
    assert result.plan.metadata["source_count"] == 2
    assert result.plan.metadata["source_names"] == ["history", "summary"]
    assert any(entry.content == "history item" for entry in result.plan.entries)
    assert "source summary" in (result.plan.summary or "")


@pytest.mark.asyncio
async def test_summarization_strategy_request_interface_is_supported() -> None:
    summarizer = NoOpSummarizer()
    result = await summarizer.summarize_request(
        SummarizationRequest(
            messages=[Message.user("hello"), Message.assistant("hi")],
            max_tokens=32,
        )
    )

    assert result.summary == "[Summary of 2 earlier messages]"
    assert result.metadata["strategy"] == "noop"
