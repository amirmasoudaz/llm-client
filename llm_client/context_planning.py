"""
Generic context-history scoring and truncation primitives.

These utilities are intentionally policy-light so they can be reused by
different higher-level runtimes that need simple recency- and relevance-aware
context selection.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from .memory import MemoryQuery, MemoryReader, MemoryRecord, SummaryStore
from .summarization import SummarizationRequest, SummarizationStrategy

if TYPE_CHECKING:
    from .hooks import HookManager

TEntry = TypeVar("TEntry", bound="HistoryEntryLike")


class HistoryEntryLike(Protocol):
    role: str
    content: str
    entry_type: str


@dataclass(frozen=True)
class HistoryTruncationResult:
    entries: list[Any]
    truncated: bool
    total_before: int
    total_after: int
    omitted_count: int
    truncation_notice: str | None = None


@dataclass(frozen=True)
class ScoredHistoryEntry:
    entry: Any
    score: float
    components: dict[str, float]

    @property
    def role(self) -> str:
        return str(getattr(self.entry, "role", ""))

    @property
    def content(self) -> str:
        return str(getattr(self.entry, "content", ""))


EmbeddingFn = Callable[[str], Awaitable[list[float]]]


@dataclass(frozen=True)
class ContextPlanningRequest:
    entries: list[Any]
    current_message: str = ""
    max_entries: int = 50
    always_keep_first: bool = True
    memory_query: MemoryQuery | None = None
    max_memory_entries: int = 10
    summarize_when_truncated: bool = False
    summary_max_tokens: int = 256
    persist_summary: bool = False
    summary_scope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if self.max_memory_entries < 0:
            raise ValueError("max_memory_entries cannot be negative")
        if self.summary_max_tokens < 1:
            raise ValueError("summary_max_tokens must be at least 1")

@dataclass(frozen=True)
class ContextPlan:
    entries: list[Any]
    memory: list[MemoryRecord]
    summary: str | None = None
    persistent_summary: str | None = None
    truncation: HistoryTruncationResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TrimmingStrategy(Protocol):
    def trim(self, entries: list[Any], request: ContextPlanningRequest) -> HistoryTruncationResult:
        ...


class ContextPlanner(Protocol):
    async def plan(self, request: ContextPlanningRequest) -> ContextPlan:
        ...


class RelevanceSelectionStrategy(Protocol):
    async def select(
        self,
        entries: list[Any],
        *,
        current_message: str,
        limit: int,
    ) -> list[Any]:
        ...


class MemoryRetrievalStrategy(Protocol):
    async def build_query(self, request: ContextPlanningRequest) -> MemoryQuery | None:
        ...


@dataclass(frozen=True)
class SlidingWindowTrimmingStrategy:
    def trim(self, entries: list[Any], request: ContextPlanningRequest) -> HistoryTruncationResult:
        return truncate_history(
            entries,
            max_entries=request.max_entries,
            always_keep_first=request.always_keep_first,
        )


@dataclass(frozen=True)
class TieredTrimmingStrategy:
    tier1_tail: int = 10
    tier2_budget: int | None = None

    def trim(self, entries: list[Any], request: ContextPlanningRequest) -> HistoryTruncationResult:
        scored_entries = score_entries(
            entries,
            current_message=request.current_message,
        )
        return truncate_history_tiered(
            entries,
            max_entries=request.max_entries,
            tier1_tail=self.tier1_tail,
            tier2_budget=self.tier2_budget,
            scored_entries=scored_entries,
        )


@dataclass(frozen=True)
class DefaultMemoryRetrievalStrategy:
    default_scope: str | None = None
    default_limit: int = 10

    async def build_query(self, request: ContextPlanningRequest) -> MemoryQuery | None:
        query_text = request.current_message.strip()
        scope = request.summary_scope or self.default_scope
        if not query_text and scope is None:
            return None
        return MemoryQuery(
            query=query_text or None,
            scope=scope,
            limit=min(self.default_limit, request.max_memory_entries),
            metadata={},
        )


@dataclass(frozen=True)
class SemanticRelevanceSelector:
    embed_fn: EmbeddingFn
    preserve_order: bool = True

    async def select(
        self,
        entries: list[Any],
        *,
        current_message: str,
        limit: int,
    ) -> list[Any]:
        if limit <= 0 or not entries:
            return []
        scored = await score_with_embeddings(
            entries,
            current_message=current_message,
            embed_fn=self.embed_fn,
        )
        selected = select_top_k(scored, k=limit, preserve_order=self.preserve_order)
        return [item.entry for item in selected]


class HeuristicContextPlanner:
    """Minimal context planner composed from generic trimming, memory, and summary primitives."""

    def __init__(
        self,
        *,
        trimming_strategy: TrimmingStrategy | None = None,
        summarization_strategy: SummarizationStrategy | None = None,
        memory_reader: MemoryReader | None = None,
        retrieval_strategy: MemoryRetrievalStrategy | None = None,
        summary_store: SummaryStore | None = None,
        hook_manager: HookManager | None = None,
    ) -> None:
        self.trimming_strategy = trimming_strategy or SlidingWindowTrimmingStrategy()
        self.summarization_strategy = summarization_strategy
        self.memory_reader = memory_reader
        self.retrieval_strategy = retrieval_strategy
        self.summary_store = summary_store
        self.hook_manager = hook_manager

    async def plan(self, request: ContextPlanningRequest) -> ContextPlan:
        truncation = self.trimming_strategy.trim(request.entries, request)
        memory: list[MemoryRecord] = []
        summary: str | None = None
        persistent_summary: str | None = None
        persistent_summary_loaded = False
        persistent_summary_updated = False
        memory_query = request.memory_query
        if memory_query is None and self.retrieval_strategy is not None:
            memory_query = await self.retrieval_strategy.build_query(request)

        if self.summary_store is not None and request.summary_scope:
            persisted = await self.summary_store.get(request.summary_scope)
            if persisted is not None:
                persistent_summary = persisted.summary
                persistent_summary_loaded = True

        if self.memory_reader is not None and memory_query is not None and request.max_memory_entries > 0:
            query = MemoryQuery(
                query=memory_query.query,
                scope=memory_query.scope,
                limit=min(memory_query.limit, request.max_memory_entries),
                metadata=dict(memory_query.metadata),
            )
            memory = await self.memory_reader.retrieve(query)

        if (
            request.summarize_when_truncated
            and truncation.truncated
            and self.summarization_strategy is not None
        ):
            omitted_entries = _omitted_entries(request.entries, truncation.entries)
            if omitted_entries:
                summary_result = await self.summarization_strategy.summarize_request(
                    SummarizationRequest(
                        messages=[_entry_to_message(entry) for entry in omitted_entries],
                        max_tokens=request.summary_max_tokens,
                        metadata=dict(request.metadata),
                    )
                )
                candidate_summary = (summary_result.summary or "").strip()
                summary = candidate_summary or None
                if (
                    self.summary_store is not None
                    and request.persist_summary
                    and request.summary_scope
                    and _is_persistable_summary(summary)
                    and summary != persistent_summary
                ):
                    stored = await self.summary_store.put(
                        request.summary_scope,
                        summary,
                        metadata=dict(request.metadata),
                    )
                    persistent_summary = stored.summary
                    persistent_summary_updated = True

        if summary is None and persistent_summary is not None:
            summary = persistent_summary

        plan = ContextPlan(
            entries=list(truncation.entries),
            memory=memory,
            summary=summary,
            persistent_summary=persistent_summary,
            truncation=truncation,
            metadata={
                "memory_entries": len(memory),
                "truncated": truncation.truncated,
                "retrieval_used": memory_query is not None,
                "persistent_summary_loaded": persistent_summary_loaded,
                "persistent_summary_updated": persistent_summary_updated,
            },
        )

        if self.hook_manager is not None:
            await self.hook_manager.emit(
                "context.plan",
                {
                    "entries_before": len(request.entries),
                    "entries_after": len(plan.entries),
                    "truncated": truncation.truncated,
                    "omitted_count": truncation.omitted_count,
                    "memory_entries": len(memory),
                    "summary_present": bool(plan.summary),
                    "persistent_summary": bool(plan.persistent_summary),
                },
                request.metadata.get("request_context"),
            )

        return plan

ENTRY_TYPE_WEIGHTS: dict[str, float] = {
    "message": 1.0,
    "tool_call": 0.6,
    "action_summary": 0.4,
    "system": 0.3,
}

ROLE_WEIGHTS: dict[str, float] = {
    "user": 1.0,
    "assistant": 0.8,
    "system": 0.3,
    "tool": 0.5,
    "action_summary": 0.4,
}


def truncate_history(
    entries: list[TEntry],
    *,
    max_entries: int = 50,
    always_keep_first: bool = True,
) -> HistoryTruncationResult:
    total = len(entries)
    if total <= max_entries:
        return HistoryTruncationResult(
            entries=list(entries),
            truncated=False,
            total_before=total,
            total_after=total,
            omitted_count=0,
        )

    if not always_keep_first or max_entries < 2:
        kept = entries[-max_entries:] if max_entries > 0 else []
        omitted = total - len(kept)
        return HistoryTruncationResult(
            entries=kept,
            truncated=True,
            total_before=total,
            total_after=len(kept),
            omitted_count=omitted,
            truncation_notice=f"[{omitted} earlier messages omitted]",
        )

    first_entry = _find_first_user_entry(entries)
    if first_entry is None:
        kept = entries[-max_entries:]
        omitted = total - len(kept)
        return HistoryTruncationResult(
            entries=kept,
            truncated=True,
            total_before=total,
            total_after=len(kept),
            omitted_count=omitted,
            truncation_notice=f"[{omitted} earlier messages omitted]",
        )

    tail_budget = max_entries - 1
    tail = entries[-tail_budget:] if tail_budget > 0 else []

    if first_entry in tail:
        omitted = total - len(tail)
        return HistoryTruncationResult(
            entries=tail,
            truncated=True,
            total_before=total,
            total_after=len(tail),
            omitted_count=omitted,
            truncation_notice=f"[{omitted} earlier messages omitted]",
        )

    kept = [first_entry] + tail
    omitted = total - len(kept)
    return HistoryTruncationResult(
        entries=kept,
        truncated=True,
        total_before=total,
        total_after=len(kept),
        omitted_count=omitted,
        truncation_notice=f"[{omitted} messages from the middle of the conversation omitted]",
    )


def truncate_history_dicts(
    entries: list[dict[str, Any]],
    *,
    max_entries: int = 50,
    always_keep_first: bool = True,
) -> tuple[list[dict[str, Any]], str | None]:
    total = len(entries)
    if total <= max_entries:
        return list(entries), None

    if not always_keep_first or max_entries < 2:
        kept = entries[-max_entries:] if max_entries > 0 else []
        omitted = total - len(kept)
        return kept, f"[{omitted} earlier messages omitted]"

    first_entry = _find_first_user_entry_dict(entries)
    if first_entry is None:
        kept = entries[-max_entries:]
        omitted = total - len(kept)
        return kept, f"[{omitted} earlier messages omitted]"

    tail_budget = max_entries - 1
    tail = entries[-tail_budget:] if tail_budget > 0 else []

    if first_entry in tail:
        omitted = total - len(tail)
        return tail, f"[{omitted} earlier messages omitted]"

    kept = [first_entry] + tail
    omitted = total - len(kept)
    return kept, f"[{omitted} messages from the middle of the conversation omitted]"


def truncate_history_tiered(
    entries: list[TEntry],
    *,
    max_entries: int = 50,
    tier1_tail: int = 10,
    tier2_budget: int | None = None,
    scored_entries: list[Any] | None = None,
) -> HistoryTruncationResult:
    total = len(entries)
    if total <= max_entries:
        return HistoryTruncationResult(
            entries=list(entries),
            truncated=False,
            total_before=total,
            total_after=total,
            omitted_count=0,
        )

    first_entry = _find_first_user_entry(entries)
    first_idx = entries.index(first_entry) if first_entry and first_entry in entries else 0

    tail_start = max(0, total - tier1_tail)
    tier1_indices: set[int] = {first_idx}
    for i in range(tail_start, total):
        tier1_indices.add(i)

    remaining_budget = max_entries - len(tier1_indices)
    if remaining_budget <= 0:
        kept_indices = sorted(tier1_indices)[:max_entries]
        kept = [entries[i] for i in kept_indices]
        omitted = total - len(kept)
        return HistoryTruncationResult(
            entries=kept,
            truncated=True,
            total_before=total,
            total_after=len(kept),
            omitted_count=omitted,
            truncation_notice=f"[{omitted} messages omitted, tiered truncation]",
        )

    effective_tier2_budget = tier2_budget if tier2_budget is not None else remaining_budget
    effective_tier2_budget = min(effective_tier2_budget, remaining_budget)
    middle_indices = [i for i in range(total) if i not in tier1_indices]

    if scored_entries and len(scored_entries) == total:
        middle_scored = [(i, scored_entries[i].score) for i in middle_indices]
        middle_scored.sort(key=lambda item: item[1], reverse=True)
        tier2_indices = {pair[0] for pair in middle_scored[:effective_tier2_budget]}
    else:
        tier2_indices = set(middle_indices[-effective_tier2_budget:]) if middle_indices else set()

    all_kept_indices = sorted(tier1_indices | tier2_indices)
    kept = [entries[i] for i in all_kept_indices]
    omitted = total - len(kept)
    return HistoryTruncationResult(
        entries=kept,
        truncated=True,
        total_before=total,
        total_after=len(kept),
        omitted_count=omitted,
        truncation_notice=f"[{omitted} messages omitted, tiered truncation with {len(tier2_indices)} semantic picks]",
    )


def compute_recency_score(index: int, total: int, *, decay_factor: float = 0.95) -> float:
    if total <= 1:
        return 1.0
    distance_from_end = total - 1 - index
    return decay_factor**distance_from_end


def compute_keyword_overlap(entry_text: str, query_text: str) -> float:
    if not entry_text or not query_text:
        return 0.0
    entry_words = set(entry_text.lower().split())
    query_words = set(query_text.lower().split())
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "it",
        "its",
        "this",
        "that",
        "and",
        "or",
        "but",
        "not",
        "i",
        "you",
        "me",
        "my",
        "your",
        "we",
        "our",
        "they",
        "their",
        "can",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "have",
        "has",
        "had",
    }
    entry_words -= stop_words
    query_words -= stop_words
    if not entry_words or not query_words:
        return 0.0
    overlap = entry_words & query_words
    return len(overlap) / max(len(query_words), 1)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_entries(
    entries: list[TEntry],
    *,
    current_message: str,
    entry_embeddings: list[list[float]] | None = None,
    query_embedding: list[float] | None = None,
    recency_weight: float = 0.3,
    semantic_weight: float = 0.4,
    keyword_weight: float = 0.2,
    type_weight: float = 0.1,
) -> list[ScoredHistoryEntry]:
    total = len(entries)
    scored: list[ScoredHistoryEntry] = []

    for index, entry in enumerate(entries):
        recency = compute_recency_score(index, total)
        keyword = compute_keyword_overlap(str(entry.content), current_message)

        role_w = ROLE_WEIGHTS.get(str(entry.role), 0.5)
        type_w = ENTRY_TYPE_WEIGHTS.get(str(entry.entry_type), 0.5)
        type_score = (role_w + type_w) / 2.0

        semantic = 0.0
        if entry_embeddings is not None and query_embedding is not None and index < len(entry_embeddings):
            semantic = max(0.0, cosine_similarity(entry_embeddings[index], query_embedding))

        effective_semantic_weight = semantic_weight if entry_embeddings is not None else 0.0
        total_weight = recency_weight + effective_semantic_weight + keyword_weight + type_weight
        if total_weight <= 0:
            total_weight = 1.0

        final_score = (
            recency * recency_weight
            + semantic * effective_semantic_weight
            + keyword * keyword_weight
            + type_score * type_weight
        ) / total_weight

        scored.append(
            ScoredHistoryEntry(
                entry=entry,
                score=final_score,
                components={
                    "recency": recency,
                    "semantic": semantic,
                    "keyword": keyword,
                    "type": type_score,
                },
            )
        )

    return scored


def select_top_k(
    scored: list[ScoredHistoryEntry],
    *,
    k: int,
    preserve_order: bool = True,
) -> list[ScoredHistoryEntry]:
    if len(scored) <= k:
        return list(scored)

    ranked = sorted(scored, key=lambda item: item.score, reverse=True)
    selected = ranked[:k]
    if preserve_order:
        original_indices = {id(item.entry): i for i, item in enumerate(scored)}
        selected.sort(key=lambda item: original_indices.get(id(item.entry), 0))
    return selected


async def score_with_embeddings(
    entries: list[TEntry],
    *,
    current_message: str,
    embed_fn: EmbeddingFn,
    **kwargs: Any,
) -> list[ScoredHistoryEntry]:
    texts = [str(entry.content) for entry in entries]
    texts.append(current_message)

    embeddings: list[list[float]] = []
    for text in texts:
        embeddings.append(await embed_fn(text))

    return score_entries(
        entries,
        current_message=current_message,
        entry_embeddings=embeddings[:-1],
        query_embedding=embeddings[-1],
        **kwargs,
    )


def _find_first_user_entry(entries: list[TEntry]) -> TEntry | None:
    for entry in entries:
        if str(entry.role) == "user" and str(entry.content).strip():
            return entry
    return entries[0] if entries else None


def _find_first_user_entry_dict(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("role", "")) == "user":
            content = str(entry.get("content") or entry.get("message") or "")
            if content.strip():
                return entry
    return entries[0] if entries else None


def _context_entry_identity(entry: Any) -> tuple[str, str, str]:
    if isinstance(entry, dict):
        return (
            str(entry.get("role", "")),
            str(entry.get("content") or entry.get("message") or ""),
            str(entry.get("entry_type", "message")),
        )
    return (
        str(getattr(entry, "role", "")),
        str(getattr(entry, "content", "")),
        str(getattr(entry, "entry_type", "message")),
    )


def _entry_to_message_dict(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        role = str(entry.get("role", "user") or "user")
        content = str(entry.get("content") or entry.get("message") or "")
    else:
        role = str(getattr(entry, "role", "user") or "user")
        content = str(getattr(entry, "content", "") or "")
    return {"role": role, "content": content}


def _entry_to_message(entry: Any) -> Any:
    from .providers.types import Message, Role

    message_dict = _entry_to_message_dict(entry)
    role_name = str(message_dict["role"]).upper()
    role = getattr(Role, role_name, Role.USER)
    return Message(role=role, content=message_dict["content"])


def _omitted_entries(original: list[Any], kept: list[Any]) -> list[Any]:
    remaining = [_context_entry_identity(item) for item in kept]
    omitted: list[Any] = []
    for entry in original:
        ident = _context_entry_identity(entry)
        if ident in remaining:
            remaining.remove(ident)
        else:
            omitted.append(entry)
    return omitted


def _is_persistable_summary(summary: str | None) -> bool:
    if summary is None:
        return False
    compact = summary.strip()
    if not compact:
        return False
    if compact.startswith("[") and compact.endswith("]"):
        return False
    return True

__all__ = [
    "ContextPlan",
    "ContextPlanner",
    "ContextPlanningRequest",
    "DefaultMemoryRetrievalStrategy",
    "EmbeddingFn",
    "ENTRY_TYPE_WEIGHTS",
    "HeuristicContextPlanner",
    "HistoryEntryLike",
    "HistoryTruncationResult",
    "RelevanceSelectionStrategy",
    "ROLE_WEIGHTS",
    "SemanticRelevanceSelector",
    "ScoredHistoryEntry",
    "SlidingWindowTrimmingStrategy",
    "TieredTrimmingStrategy",
    "TrimmingStrategy",
    "MemoryRetrievalStrategy",
    "compute_keyword_overlap",
    "compute_recency_score",
    "cosine_similarity",
    "score_entries",
    "score_with_embeddings",
    "select_top_k",
    "truncate_history",
    "truncate_history_dicts",
    "truncate_history_tiered",
]
