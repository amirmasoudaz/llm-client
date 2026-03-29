from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = ROOT / "examples" / "llm_client"

CORE_EXAMPLES = [
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
]

APPLICATION_EXAMPLES = [
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

ALL_EXAMPLES = [*CORE_EXAMPLES, *APPLICATION_EXAMPLES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the live llm_client cookbook examples.",
    )
    parser.add_argument(
        "--subset",
        choices=["core", "application", "all", "offline", "live"],
        default="all",
        help=(
            "Which example set to run. "
            "`offline` is a legacy alias for `core`; `live` is a legacy alias for `all`."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="Per-example timeout.",
    )
    return parser.parse_args()


def run_example(script_name: str, timeout_seconds: int) -> None:
    script_path = EXAMPLES_DIR / script_name
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(script_path)]
    print(f"[llm_client examples] running {script_name}")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(
            f"example failed: {script_name} (exit {completed.returncode})"
        )
    if completed.stdout.strip():
        print(completed.stdout.rstrip())


def main() -> int:
    args = parse_args()
    subset = args.subset
    if subset == "offline":
        subset = "core"
    elif subset == "live":
        subset = "all"

    if subset == "core":
        example_names = CORE_EXAMPLES
    elif subset == "application":
        example_names = APPLICATION_EXAMPLES
    else:
        example_names = ALL_EXAMPLES

    for script_name in example_names:
        run_example(script_name, timeout_seconds=args.timeout_seconds)
    print(f"[llm_client examples] completed {len(example_names)} example scripts for subset={subset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
