from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COOKBOOK_DIR = ROOT / "examples"
README = ROOT / "examples" / "README.md"
RUNNER = ROOT / "scripts" / "ci" / "run_llm_client_examples.py"


EXPECTED_FILES = [
    "01_one_shot_completion.py",
    "02_streaming.py",
    "03_embeddings.py",
    "04_content_blocks.py",
    "05_structured_extraction.py",
    "06_provider_registry_and_routing.py",
    "07_engine_cache_retry_idempotency.py",
    "08_tool_execution_modes.py",
    "09_tool_calling_agent.py",
    "10_context_memory_planning.py",
    "11_observability_and_redaction.py",
    "12_benchmarks.py",
    "13_batch_processing.py",
    "14_sync_wrappers.py",
    "15_rate_limiting.py",
    "35_file_block_transport.py",
    "16_fastapi_sse.py",
    "17_persistence_repository.py",
    "18_memory_backed_assistant.py",
    "19_multi_provider_failover_gateway.py",
    "20_rag_with_citations.py",
    "21_document_review_diff.py",
    "22_human_in_the_loop_approvals.py",
    "23_async_job_queue_sse.py",
    "24_customer_support_copilot.py",
    "25_incident_war_room_assistant.py",
    "26_research_briefing_agent.py",
    "27_sql_analytics_assistant.py",
    "28_release_readiness_control_plane.py",
    "29_multimodal_intake_pipeline.py",
    "30_eval_and_regression_gate.py",
    "31_tool_calling_with_partial_failures.py",
    "32_cache_strategy_showdown.py",
    "33_compliance_redaction_pipeline.py",
    "34_end_to_end_mission_control.py",
    "36_sql_adaptor_direct.py",
    "37_sql_adaptor_tools.py",
]


def test_cookbook_examples_exist() -> None:
    missing = [name for name in EXPECTED_FILES if not (COOKBOOK_DIR / name).exists()]
    assert missing == []


def test_cookbook_readme_references_all_examples() -> None:
    readme = README.read_text(encoding="utf-8")
    for name in EXPECTED_FILES:
        assert name in readme


def test_cookbook_runner_references_all_examples() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    for name in EXPECTED_FILES:
        assert name in runner
    assert "--subset core" in README.read_text(encoding="utf-8")
    assert "--subset application" in README.read_text(encoding="utf-8")
