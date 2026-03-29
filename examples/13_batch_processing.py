from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json

from llm_client.batch_req import BatchManager
from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT / "tmp" / "cookbook-batch-processing-checkpoint.jsonl"


def build_batch_items() -> list[dict[str, Any]]:
    return [
        {
            "ticket_id": "SUP-3011",
            "customer_tier": "enterprise",
            "issue": "Audit export jobs started timing out after retention-policy changes.",
            "impact": "Finance cannot complete month-end reconciliation.",
            "signals": [
                "timeouts spike around five minutes",
                "queue depth grows after large exports",
                "support suspects retention fanout overhead",
            ],
        },
        {
            "ticket_id": "SUP-3012",
            "customer_tier": "growth",
            "issue": "Checkout webhooks are delayed after the payment routing update.",
            "impact": "Order-status emails are arriving 20 minutes late.",
            "signals": [
                "webhook retries increased 4x",
                "payment-edge queue lag is elevated",
                "no data loss has been observed",
            ],
        },
        {
            "ticket_id": "SUP-3013",
            "customer_tier": "enterprise",
            "issue": "SCIM sync is creating duplicate deactivation events for suspended users.",
            "impact": "IT administrators are pausing automated offboarding.",
            "signals": [
                "duplicate events correlate with nightly HR imports",
                "directory write latency is normal",
                "support needs clear rollback guidance",
            ],
        },
        {
            "ticket_id": "SUP-3014",
            "customer_tier": "midmarket",
            "issue": "Forecast dashboards are stale after the analytics materialization job was rescheduled.",
            "impact": "Revenue operations is using day-old pipeline numbers in executive reviews.",
            "signals": [
                "materialization job started 45 minutes later than normal",
                "warehouse slot saturation increased overnight",
                "no permanent refresh failures recorded",
            ],
        },
    ]


def build_batch_specs(provider_name: str, model_name: str, items: list[dict[str, Any]]) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    for item in items:
        specs.append(
            RequestSpec(
                provider=provider_name,
                model=model_name,
                messages=[
                    Message.system(
                        "You are an operations triage analyst. "
                        "Return a concise internal triage note with severity, likely owning team, and immediate next move."
                    ),
                    Message.user(
                        "Prepare a short triage note for this support escalation.\n\n"
                        f"Ticket: {item['ticket_id']}\n"
                        f"Customer tier: {item['customer_tier']}\n"
                        f"Issue: {item['issue']}\n"
                        f"Impact: {item['impact']}\n"
                        "Signals:\n- "
                        + "\n- ".join(item["signals"])
                    ),
                ],
            )
        )
    return specs


def summarize_engine_results(items: list[dict[str, Any]], results: list[Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for item, result in zip(items, results, strict=False):
        content = getattr(result, "content", None)
        excerpt = " ".join(str(content or "").split())
        if len(excerpt) > 220:
            excerpt = excerpt[:219].rstrip() + "…"
        summary.append(
            {
                "ticket_id": item["ticket_id"],
                "status": getattr(result, "status", None),
                "model": getattr(result, "model", None),
                "content_excerpt": excerpt,
                "total_tokens": getattr(getattr(result, "usage", None), "total_tokens", None),
            }
        )
    return summary


def summarize_manager_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for item in results:
        ordered.append(
            {
                "_batch_index": item.get("_batch_index"),
                "ticket_id": item.get("ticket_id"),
                "status": item.get("status"),
                "routing_owner": item.get("routing_owner"),
                "latency_ms": item.get("latency_ms"),
                "content_excerpt": item.get("content_excerpt"),
                "checkpointed": item.get("checkpointed"),
                "error": item.get("error"),
            }
        )
    return ordered


def infer_routing_owner(batch_item: dict[str, Any], model_excerpt: str) -> str:
    text = " ".join(
        [
            batch_item["issue"],
            batch_item["impact"],
            " ".join(batch_item["signals"]),
            model_excerpt,
        ]
    ).lower()

    if any(token in text for token in ("scim", "directory", "offboarding", "identity", "hr import")):
        return "identity-platform"
    if any(token in text for token in ("payment", "checkout", "webhook", "commerce", "order-status")):
        return "commerce-platform"
    if any(token in text for token in ("analytics", "warehouse", "forecast", "materialization", "revops", "data platform")):
        return "data-platform"
    if any(token in text for token in ("audit export", "retention", "reconciliation", "export pipeline")):
        return "data-platform"
    if any(token in text for token in ("infra", "platform operations", "sre", "backpressure")):
        return "platform-operations"
    return "triage-review"


async def main() -> None:
    handle = build_live_provider()
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    try:
        items = build_batch_items()
        specs = build_batch_specs(handle.name, handle.model, items)

        engine = ExecutionEngine(provider=handle.provider)
        batch_started = time.perf_counter()
        engine_results = await engine.batch_complete(specs, max_concurrency=3)
        engine_batch_duration_ms = (time.perf_counter() - batch_started) * 1000

        worker_engine = ExecutionEngine(provider=handle.provider)

        async def _processor(batch_item: dict[str, Any]) -> dict[str, Any]:
            started = time.perf_counter()
            spec = RequestSpec(
                provider=handle.name,
                model=handle.model,
                messages=[
                    Message.system(
                        "You are routing support escalations. "
                        "Return a compact owner recommendation and next action for the internal queue."
                    ),
                    Message.user(
                        "Route this escalation to the best owning team and give one immediate action.\n\n"
                        f"Ticket: {batch_item['ticket_id']}\n"
                        f"Customer tier: {batch_item['customer_tier']}\n"
                        f"Issue: {batch_item['issue']}\n"
                        f"Impact: {batch_item['impact']}\n"
                        "Signals:\n- "
                        + "\n- ".join(batch_item["signals"])
                    ),
                ],
            )
            result = await worker_engine.complete(spec)
            excerpt = " ".join(str(result.content or "").split())
            if len(excerpt) > 200:
                excerpt = excerpt[:199].rstrip() + "…"
            return {
                "ticket_id": batch_item["ticket_id"],
                "status": result.status,
                "routing_owner": infer_routing_owner(batch_item, excerpt),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "content_excerpt": excerpt,
                "checkpointed": True,
            }

        manager = BatchManager(max_workers=2, checkpoint_file=CHECKPOINT_PATH)
        manager_started = time.perf_counter()
        manager_results = await manager.process_batch(items, _processor, desc="cookbook-live-batch-routing")
        manager_duration_ms = (time.perf_counter() - manager_started) * 1000

        resume_manager = BatchManager(max_workers=2, checkpoint_file=CHECKPOINT_PATH)
        resume_started = time.perf_counter()
        resume_results = await resume_manager.process_batch(items, _processor, desc="cookbook-live-batch-resume")
        resume_duration_ms = (time.perf_counter() - resume_started) * 1000

        print_heading("Batch Workload")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "checkpoint_path": str(CHECKPOINT_PATH),
                "items": items,
            }
        )

        print_heading("Engine Batch Complete")
        print_json(
            {
                "max_concurrency": 3,
                "batch_duration_ms": round(engine_batch_duration_ms, 2),
                "ordered_results": summarize_engine_results(items, engine_results),
            }
        )

        print_heading("Checkpointed Batch Manager")
        print_json(
            {
                "max_workers": 2,
                "batch_duration_ms": round(manager_duration_ms, 2),
                "results": summarize_manager_results(manager_results),
            }
        )

        print_heading("Resume Pass")
        print_json(
            {
                "loaded_checkpoint_records": len(resume_manager.processed_indices),
                "resume_duration_ms": round(resume_duration_ms, 2),
                "results": summarize_manager_results(resume_results),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
