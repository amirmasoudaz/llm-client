from __future__ import annotations

from typing import Any

import pytest

from intelligence_layer_kernel.operators.implementations.professor_alignment_score import (
    ProfessorAlignmentScoreOperator,
)
from intelligence_layer_kernel.operators.implementations.professor_profile_retrieve import (
    ProfessorProfileRetrieveOperator,
)
from intelligence_layer_kernel.operators.implementations.professor_summarize import (
    ProfessorSummarizeOperator,
)
from tests.intelligence_layer_kernel._phase_fg_testkit import build_operator_call


class _ProfessorDB:
    def __init__(self, row: dict[str, Any]) -> None:
        self._row = row

    async def fetch_one(self, _sql: str, _params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        return dict(self._row)


def _base_professor_row() -> dict[str, Any]:
    return {
        "id": 9001,
        "first_name": "Ada",
        "last_name": "Byron",
        "full_name": "Ada Byron",
        "occupation": "Professor of Electrical Engineering",
        "research_areas": "Signal processing, machine learning, control systems",
        "credentials": "PhD",
        "area_of_expertise": '["Signal Processing", "Machine Learning"]',
        "categories": "Control Systems; Optimization",
        "department": "Electrical Engineering",
        "email_address": "ada@example.edu",
        "url": "https://example.edu/ada",
        "others": '{"notes":"Deterministic fixture"}',
        "canspider_digest_id": "digest-1",
        "institution_name": "Example University",
        "department_name": "EECS",
        "country": "USA",
    }


def _student_background() -> dict[str, Any]:
    return {
        "funding_request": {
            "research_interest": "signal processing and machine learning",
            "paper_title": "Adaptive filtering for speech denoising",
            "journal": "IEEE Transactions",
            "research_connection": "built models for speech enhancement and robust DSP pipelines",
        },
        "profile": {
            "general": {"field": "electrical engineering"},
            "context": {
                "background": {
                    "research_interests": "signal processing, control systems, optimization",
                    "projects": "designed neural signal enhancement for noisy audio streams",
                    "skills": "python, optimization, neural models",
                }
            },
        },
    }


@pytest.mark.asyncio
async def test_profile_summary_alignment_chain_is_deterministic_and_evidence_coherent() -> None:
    profile_operator = ProfessorProfileRetrieveOperator()
    profile_operator._db = _ProfessorDB(_base_professor_row())
    summarize_operator = ProfessorSummarizeOperator()
    alignment_operator = ProfessorAlignmentScoreOperator()

    profile_call = build_operator_call({"professor_id": 9001})
    profile_first = await profile_operator.run(profile_call)
    profile_second = await profile_operator.run(profile_call)
    assert profile_first.status == "succeeded"
    assert profile_second.status == "succeeded"

    profile_one = (profile_first.result or {}).get("profile")
    profile_two = (profile_second.result or {}).get("profile")
    assert isinstance(profile_one, dict)
    assert isinstance(profile_two, dict)
    assert profile_one["profile_hash"] == profile_two["profile_hash"]
    assert profile_one["extracted"] == profile_two["extracted"]

    summarize_call = build_operator_call({"professor_id": 9001, "profile": profile_one})
    summary_first = await summarize_operator.run(summarize_call)
    summary_second = await summarize_operator.run(summarize_call)
    assert summary_first.status == "succeeded"
    assert summary_second.status == "succeeded"

    summary_outcome_one = (summary_first.result or {}).get("outcome")
    summary_outcome_two = (summary_second.result or {}).get("outcome")
    assert isinstance(summary_outcome_one, dict)
    assert isinstance(summary_outcome_two, dict)
    assert summary_outcome_one["payload"] == summary_outcome_two["payload"]
    assert summary_outcome_one["hash"] == summary_outcome_two["hash"]

    alignment_call = build_operator_call(
        {
            "student_id": 7,
            "professor_id": 9001,
            "professor_summary": summary_outcome_one,
            "student_background": _student_background(),
            "focus_areas": ["signal processing", "machine learning"],
        }
    )
    alignment_first = await alignment_operator.run(alignment_call)
    alignment_second = await alignment_operator.run(alignment_call)
    assert alignment_first.status == "succeeded"
    assert alignment_second.status == "succeeded"

    alignment_payload_one = (alignment_first.result or {}).get("outcome", {}).get("payload")
    alignment_payload_two = (alignment_second.result or {}).get("outcome", {}).get("payload")
    assert isinstance(alignment_payload_one, dict)
    assert isinstance(alignment_payload_two, dict)
    assert alignment_payload_one == alignment_payload_two

    matched_topics = alignment_payload_one.get("matched_topics")
    assert isinstance(matched_topics, list)
    assert matched_topics
    rationale_long = str(alignment_payload_one.get("rationale_long") or "").lower()
    for topic in matched_topics[:3]:
        assert str(topic).lower() in rationale_long
    assert alignment_payload_one.get("label") in {"low", "medium", "high"}
    assert 0 <= float(alignment_payload_one.get("score", 0)) <= 1


@pytest.mark.asyncio
async def test_sparse_professor_data_still_produces_valid_summary_and_alignment() -> None:
    sparse_row = {
        "id": 77,
        "first_name": None,
        "last_name": None,
        "full_name": "",
        "occupation": "",
        "research_areas": None,
        "credentials": None,
        "area_of_expertise": None,
        "categories": None,
        "department": None,
        "email_address": None,
        "url": "not-a-url",
        "others": "",
        "canspider_digest_id": None,
        "institution_name": None,
        "department_name": None,
        "country": None,
    }
    profile_operator = ProfessorProfileRetrieveOperator()
    profile_operator._db = _ProfessorDB(sparse_row)
    summarize_operator = ProfessorSummarizeOperator()
    alignment_operator = ProfessorAlignmentScoreOperator()

    profile_result = await profile_operator.run(build_operator_call({"professor_id": 77}))
    assert profile_result.status == "succeeded"
    profile = (profile_result.result or {}).get("profile")
    assert isinstance(profile, dict)
    assert profile.get("canonical_url") is None
    extracted = profile.get("extracted")
    assert isinstance(extracted, dict)
    assert extracted.get("research_areas") == []

    summary_result = await summarize_operator.run(
        build_operator_call({"professor_id": 77, "profile": profile})
    )
    assert summary_result.status == "succeeded"
    summary_outcome = (summary_result.result or {}).get("outcome")
    assert isinstance(summary_outcome, dict)
    summary_payload = summary_outcome.get("payload")
    assert isinstance(summary_payload, dict)
    assert "Research topics are not explicitly listed" in str(summary_payload.get("summary"))

    alignment_result = await alignment_operator.run(
        build_operator_call(
            {
                "student_id": 7,
                "professor_id": 77,
                "professor_summary": summary_outcome,
                "student_background": {},
                "focus_areas": ["machine learning"],
            }
        )
    )
    assert alignment_result.status == "succeeded"
    alignment_payload = (alignment_result.result or {}).get("outcome", {}).get("payload")
    assert isinstance(alignment_payload, dict)
    assert isinstance(alignment_payload.get("gaps"), list)
    assert alignment_payload.get("gaps")
    assert isinstance(alignment_payload.get("recommendations"), list)
    assert alignment_payload.get("recommendations")
