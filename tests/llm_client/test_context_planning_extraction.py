from __future__ import annotations

from dataclasses import dataclass, field

from llm_client.context_planning import (
    score_entries,
    select_top_k,
    truncate_history,
    truncate_history_dicts,
    truncate_history_tiered,
)


@dataclass(frozen=True)
class _Entry:
    role: str
    content: str
    entry_type: str = "message"
    metadata: dict[str, str] = field(default_factory=dict)


def test_context_planning_truncate_history_preserves_first_user_entry() -> None:
    entries = [_Entry(role="user", content="first")] + [
        _Entry(role="assistant" if i % 2 else "user", content=f"msg {i}")
        for i in range(30)
    ]

    result = truncate_history(entries, max_entries=8)

    assert result.truncated is True
    assert len(result.entries) == 8
    assert result.entries[0].content == "first"
    assert result.omitted_count == 23


def test_context_planning_truncate_history_dicts_returns_notice() -> None:
    kept, notice = truncate_history_dicts(
        [{"role": "user", "content": f"msg {i}"} for i in range(12)],
        max_entries=5,
    )

    assert len(kept) == 5
    assert kept[0]["content"] == "msg 0"
    assert notice is not None
    assert "omitted" in notice


def test_context_planning_scoring_and_tiered_truncation_work_together() -> None:
    entries = [
        _Entry(role="user" if i % 2 == 0 else "assistant", content=f"funding message {i}")
        for i in range(20)
    ]

    scored = score_entries(entries, current_message="help with funding message")
    selected = select_top_k(scored, k=5, preserve_order=True)
    truncated = truncate_history_tiered(entries, max_entries=7, tier1_tail=3, scored_entries=scored)

    assert len(scored) == 20
    assert len(selected) == 5
    assert len(truncated.entries) <= 7
    assert truncated.truncated is True
