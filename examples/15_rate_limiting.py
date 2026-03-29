from __future__ import annotations

import asyncio
import time
from typing import Any

from cookbook_support import build_live_provider, close_provider, print_heading, print_json, summarize_usage

from llm_client.providers.types import Message
from llm_client.rate_limit import Limiter


DEMO_WINDOW_SECONDS = 6.0
DEMO_REQUESTS_PER_WINDOW = 2
DEMO_TOKENS_PER_WINDOW = 6000


def _build_rate_limited_workload() -> list[dict[str, Any]]:
    return [
        {
            "ticket_id": "OPS-4101",
            "prompt": (
                "Write a compact incident-routing note for a finance export timeout after audit retention changes. "
                "Include severity, likely owner, and one immediate action."
            ),
        },
        {
            "ticket_id": "OPS-4102",
            "prompt": (
                "Write a compact incident-routing note for delayed checkout webhooks after a payment routing update. "
                "Include severity, likely owner, and one immediate action."
            ),
        },
        {
            "ticket_id": "OPS-4103",
            "prompt": (
                "Write a compact incident-routing note for duplicate SCIM deactivation events after nightly HR imports. "
                "Include severity, likely owner, and one immediate action."
            ),
        },
        {
            "ticket_id": "OPS-4104",
            "prompt": (
                "Write a compact incident-routing note for stale revenue dashboards after analytics materialization was rescheduled. "
                "Include severity, likely owner, and one immediate action."
            ),
        },
    ]


async def _run_item(
    item: dict[str, Any],
    *,
    provider: Any,
    limiter: Limiter,
    suite_started_at: float,
) -> dict[str, Any]:
    estimated_prompt_tokens = max(1, provider.count_tokens(item["prompt"]))
    queued_at = time.monotonic()
    queued_offset_ms = round((queued_at - suite_started_at) * 1000, 2)

    async with limiter.limit(tokens=estimated_prompt_tokens, requests=1) as budget:
        admitted_at = time.monotonic()
        admitted_offset_ms = round((admitted_at - suite_started_at) * 1000, 2)
        result = await provider.complete(
            [
                Message.system(
                    "You are an operations routing assistant. Respond in 3 short bullets only: "
                    "severity, owner, immediate action."
                ),
                Message.user(item["prompt"]),
            ]
        )
        if result.usage is not None:
            budget.output_tokens = result.usage.output_tokens
        finished_at = time.monotonic()

    return {
        "ticket_id": item["ticket_id"],
        "status": result.status,
        "queued_offset_ms": queued_offset_ms,
        "admitted_offset_ms": admitted_offset_ms,
        "gate_wait_ms": round((admitted_at - queued_at) * 1000, 2),
        "provider_latency_ms": round((finished_at - admitted_at) * 1000, 2),
        "total_item_latency_ms": round((finished_at - queued_at) * 1000, 2),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "usage": summarize_usage(result.usage),
        "content_excerpt": (result.content or "")[:220],
    }


def _wave_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(results, key=lambda item: item["admitted_offset_ms"])
    bursts: list[dict[str, Any]] = []
    burst_gap_ms = 250.0
    current: dict[str, Any] | None = None
    last_offset_ms: float | None = None

    for item in ordered:
        offset_ms = float(item["admitted_offset_ms"])
        if (
            current is None
            or last_offset_ms is None
            or (offset_ms - last_offset_ms) > burst_gap_ms
        ):
            current = {
                "burst": len(bursts) + 1,
                "ticket_ids": [],
                "admission_offsets_ms": [],
            }
            bursts.append(current)
        current["ticket_ids"].append(item["ticket_id"])
        current["admission_offsets_ms"].append(item["admitted_offset_ms"])
        last_offset_ms = offset_ms

    return bursts


def _admission_timeline(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(results, key=lambda item: item["admitted_offset_ms"])
    return [
        {
            "ticket_id": item["ticket_id"],
            "admitted_offset_ms": item["admitted_offset_ms"],
            "gate_wait_ms": item["gate_wait_ms"],
        }
        for item in ordered
    ]


async def main() -> None:
    handle = build_live_provider()
    try:
        workload = _build_rate_limited_workload()
        limiter = Limiter(
            tokens_per_window=DEMO_TOKENS_PER_WINDOW,
            requests_per_window=DEMO_REQUESTS_PER_WINDOW,
            window_seconds=DEMO_WINDOW_SECONDS,
        )

        print_heading("Rate-Limited Workload")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "items": workload,
                "demo_limiter": {
                    "window_seconds": DEMO_WINDOW_SECONDS,
                    "requests_per_window": limiter.req_limiter.maximum_size,
                    "tokens_per_window": limiter.tkn_limiter.maximum_size,
                    "intent": (
                        "show admission control in a short demo window; the provider's native "
                        "rate limits remain unchanged"
                    ),
                },
            }
        )

        suite_started_at = time.monotonic()
        results = await asyncio.gather(
            *[
                _run_item(
                    item,
                    provider=handle.provider,
                    limiter=limiter,
                    suite_started_at=suite_started_at,
                )
                for item in workload
            ]
        )
        total_duration_ms = round((time.monotonic() - suite_started_at) * 1000, 2)

        print_heading("Concurrent Run With Limiter")
        print_json(
            {
                "batch_size": len(workload),
                "total_duration_ms": total_duration_ms,
                "max_parallel_submissions": len(workload),
                "rate_limit_behavior": "continuous refill token bucket, not fixed-window batching",
                "admission_timeline": _admission_timeline(results),
                "burst_groups": _wave_summary(results),
                "results": sorted(results, key=lambda item: item["ticket_id"]),
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
