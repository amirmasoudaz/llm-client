from __future__ import annotations

import uuid
from typing import Any

import pytest

from intelligence_layer_kernel.operators.implementations.documents_review import DocumentsReviewOperator
from tests.intelligence_layer_kernel._phase_fg_testkit import build_operator_call


class _Acquire:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn

    async def __aenter__(self) -> "_FakeConn":
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeConn:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row

    async def fetchrow(self, _query: str, *_args: Any) -> dict[str, Any] | None:
        return self.row


class _FakePool:
    def __init__(self, row: dict[str, Any] | None) -> None:
        self._conn = _FakeConn(row)

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


async def _run_review(*, row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    operator = DocumentsReviewOperator(pool=_FakePool(row), tenant_id=1)
    result = await operator.run(build_operator_call(payload, idempotency_key="doc-review"))
    assert result.status == "succeeded"
    assert isinstance(result.result, dict)
    outcome = result.result["outcome"]
    assert isinstance(outcome, dict)
    return outcome["payload"]


@pytest.fixture
def document_id() -> str:
    return str(uuid.uuid4())


@pytest.mark.asyncio
async def test_documents_review_emits_typed_cv_report(document_id: str) -> None:
    payload = await _run_review(
        row={
            "document_type": "cv",
            "extracted_text": "Education Experience Skills",
            "extracted_fields": {
                "word_count": 480,
                "has_education_section": True,
                "has_experience_section": True,
                "has_skills_section": True,
                "emails": ["student@example.com"],
            },
        },
        payload={
            "document_id": document_id,
            "document_type": "cv",
            "document_processed": {"payload": {"extracted_fields": {}}},
            "review_goal": "quality",
        },
    )

    report = payload["structured_report"]
    assert payload["document_type"] == "cv"
    assert report["type"] == "cv"
    assert set(report["section_scores"].keys()) == {"contact", "education", "experience", "skills"}


@pytest.mark.asyncio
async def test_documents_review_emits_typed_sop_report(document_id: str) -> None:
    payload = await _run_review(
        row={
            "document_type": "sop",
            "extracted_text": "Research interests and lab fit.",
            "extracted_fields": {
                "word_count": 620,
                "mentions_research": True,
                "mentions_program_fit": True,
            },
        },
        payload={
            "document_id": document_id,
            "document_type": "sop",
            "document_processed": {"payload": {"extracted_fields": {}}},
            "review_goal": "quality",
        },
    )

    report = payload["structured_report"]
    assert payload["document_type"] == "sop"
    assert report["type"] == "sop"
    assert set(report["section_scores"].keys()) == {
        "narrative",
        "research_alignment",
        "program_fit",
        "writing_quality",
    }


@pytest.mark.asyncio
async def test_documents_review_normalizes_cover_letter_to_letter_typed_report(document_id: str) -> None:
    payload = await _run_review(
        row={
            "document_type": "cover_letter",
            "extracted_text": "Motivation and fit details.",
            "extracted_fields": {
                "word_count": 410,
                "mentions_research": True,
                "mentions_program_fit": True,
            },
        },
        payload={
            "document_id": document_id,
            "document_type": "cover_letter",
            "document_processed": {"payload": {"extracted_fields": {}}},
            "review_goal": "quality",
        },
    )

    report = payload["structured_report"]
    assert payload["document_type"] == "letter"
    assert report["type"] == "letter"
    assert set(report["section_scores"].keys()) == {"motivation", "program_fit", "evidence", "tone"}


@pytest.mark.asyncio
async def test_documents_review_emits_generic_report_for_non_specialized_doc_types(document_id: str) -> None:
    payload = await _run_review(
        row={
            "document_type": "transcript",
            "extracted_text": "Course list and grades.",
            "extracted_fields": {
                "word_count": 310,
            },
        },
        payload={
            "document_id": document_id,
            "document_type": "transcript",
            "document_processed": {"payload": {"extracted_fields": {}}},
            "review_goal": "clarity",
        },
    )

    report = payload["structured_report"]
    assert payload["document_type"] == "transcript"
    assert report["type"] == "generic"
    assert set(report["section_scores"].keys()) == {"clarity", "structure", "relevance", "brevity"}
