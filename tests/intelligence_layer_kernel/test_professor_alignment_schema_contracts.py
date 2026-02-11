from __future__ import annotations

from typing import Any

import pytest
from jsonschema import Draft202012Validator

from intelligence_layer_kernel.contracts import ContractRegistry
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


def _fixture_professor_row() -> dict[str, Any]:
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
        "others": '{"notes":"contract-fixture"}',
        "canspider_digest_id": "digest-1",
        "institution_name": "Example University",
        "department_name": "EECS",
        "country": "USA",
    }


def _validate_schema(registry: ContractRegistry, *, schema_ref: str, instance: dict[str, Any]) -> None:
    schema = registry.get_schema_by_ref(schema_ref)
    validator = Draft202012Validator(schema, resolver=registry.resolver_for(schema))
    errors = list(validator.iter_errors(instance))
    assert not errors, "; ".join(err.message for err in errors)


@pytest.mark.asyncio
async def test_professor_alignment_operator_outputs_match_contract_schemas() -> None:
    registry = ContractRegistry()
    registry.load()

    profile_operator = ProfessorProfileRetrieveOperator()
    profile_operator._db = _ProfessorDB(_fixture_professor_row())
    summarize_operator = ProfessorSummarizeOperator()
    alignment_operator = ProfessorAlignmentScoreOperator()

    profile_result = await profile_operator.run(build_operator_call({"professor_id": 9001}))
    assert profile_result.status == "succeeded"
    _validate_schema(
        registry,
        schema_ref="schemas/operators/professor_profile_retrieve.output.v1.json",
        instance=profile_result.to_dict(),
    )

    profile_payload = (profile_result.result or {}).get("profile")
    assert isinstance(profile_payload, dict)
    summary_result = await summarize_operator.run(
        build_operator_call({"professor_id": 9001, "profile": profile_payload})
    )
    assert summary_result.status == "succeeded"
    _validate_schema(
        registry,
        schema_ref="schemas/operators/professor_summarize.output.v1.json",
        instance=summary_result.to_dict(),
    )

    summary_outcome = (summary_result.result or {}).get("outcome")
    assert isinstance(summary_outcome, dict)
    _validate_schema(
        registry,
        schema_ref="schemas/outcomes/professor_summary.v1.json",
        instance=summary_outcome,
    )

    alignment_result = await alignment_operator.run(
        build_operator_call(
            {
                "student_id": 7,
                "professor_id": 9001,
                "professor_summary": summary_outcome,
                "student_background": {
                    "funding_request": {
                        "research_interest": "signal processing and machine learning",
                        "research_connection": "signal enhancement and neural methods",
                    },
                    "profile": {
                        "general": {"field": "electrical engineering"},
                        "context": {
                            "background": {
                                "research_interests": "signal processing, optimization",
                                "projects": "speech enhancement and robust modeling",
                            }
                        },
                    },
                },
                "focus_areas": ["signal processing", "machine learning"],
            }
        )
    )
    assert alignment_result.status == "succeeded"
    _validate_schema(
        registry,
        schema_ref="schemas/operators/professor_alignment_score.output.v1.json",
        instance=alignment_result.to_dict(),
    )

    alignment_outcome = (alignment_result.result or {}).get("outcome")
    assert isinstance(alignment_outcome, dict)
    _validate_schema(
        registry,
        schema_ref="schemas/outcomes/alignment_score.v1.json",
        instance=alignment_outcome,
    )
