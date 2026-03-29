from __future__ import annotations

from llm_client.hooks import BenchmarkRecorder, HookManager
from llm_client.providers.types import Message
from llm_client.structured import StructuredOutputConfig
from llm_client.structured_benchmarks import StructuredBenchmarkCase, benchmark_structured_cases
from tests.llm_client.fakes import ScriptedProvider, ok_result


async def _run_benchmark_report():
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    cases = [
        StructuredBenchmarkCase(
            name="immediate_success",
            provider=ScriptedProvider(
                complete_script=[
                    ok_result('{"value": 1}'),
                ]
            ),
            messages=[Message.user("return json")],
            config=StructuredOutputConfig(schema=schema, max_repair_attempts=1),
        ),
        StructuredBenchmarkCase(
            name="repaired_success",
            provider=ScriptedProvider(
                complete_script=[
                    ok_result('{"value":"oops"}'),
                    ok_result('{"value": 2}'),
                ]
            ),
            messages=[Message.user("return json")],
            config=StructuredOutputConfig(schema=schema, max_repair_attempts=1),
        ),
        StructuredBenchmarkCase(
            name="failed_after_repairs",
            provider=ScriptedProvider(
                complete_script=[
                    ok_result('{"value":"oops"}'),
                    ok_result('{"value":"still wrong"}'),
                ]
            ),
            messages=[Message.user("return json")],
            config=StructuredOutputConfig(schema=schema, max_repair_attempts=1),
        ),
    ]
    return await benchmark_structured_cases(cases)


def test_structured_benchmark_report_tracks_success_and_repair_rates() -> None:
    import asyncio

    report = asyncio.run(_run_benchmark_report())

    assert report.total_cases == 3
    assert report.success_count == 2
    assert report.repaired_success_count == 1
    assert report.success_rate == 2 / 3
    assert report.repaired_success_rate == 1 / 3
    assert report.repaired_share_of_successes == 0.5
    assert report.repair_attempt_histogram[0] == 1
    assert report.repair_attempt_histogram[1] == 2
    assert report.max_repair_attempts == 1


def test_structured_benchmark_report_records_case_level_outcomes() -> None:
    import asyncio

    report = asyncio.run(_run_benchmark_report())
    records = {record.name: record for record in report.records}

    assert records["immediate_success"].valid is True
    assert records["immediate_success"].repair_attempts == 0
    assert records["repaired_success"].valid is True
    assert records["repaired_success"].repair_attempts == 1
    assert records["failed_after_repairs"].valid is False
    assert records["failed_after_repairs"].repair_attempts == 1
    assert "not of type 'integer'" in records["failed_after_repairs"].validation_errors[0]


def test_structured_benchmark_hooks_emit_case_and_report_events() -> None:
    import asyncio

    recorder = BenchmarkRecorder()

    async def _run():
        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        cases = [
            StructuredBenchmarkCase(
                name="instrumented_success",
                provider=ScriptedProvider(complete_script=[ok_result('{"value": 1}')]),
                messages=[Message.user("return json")],
                config=StructuredOutputConfig(schema=schema, max_repair_attempts=0),
            )
        ]
        return await benchmark_structured_cases(cases, hooks=HookManager([recorder]))

    report = asyncio.run(_run())

    assert report.total_cases == 1
    assert len(recorder.cases) == 1
    assert recorder.cases[0].payload["name"] == "instrumented_success"
    assert recorder.cases[0].payload["valid"] is True
    assert len(recorder.reports) == 1
    assert recorder.reports[0].payload["success_rate"] == 1.0
