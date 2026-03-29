from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from uuid import uuid4

from cookbook_support import (
    build_live_provider,
    close_provider,
    print_heading,
    print_json,
    summarize_usage,
)
from cookbook_expansion_support import qdrant_api_key, require_qdrant_url

from llm_client.cache import CachePolicy
from llm_client.cache.factory import CacheSettings, build_cache_core
from llm_client.engine import ExecutionEngine
from llm_client.hooks import EngineDiagnosticsRecorder, HookManager, LifecycleRecorder
from llm_client.idempotency import IdempotencyTracker
from llm_client.memory import MemoryQuery, MemoryWrite, ShortTermMemoryStore
from llm_client.providers.types import CompletionResult, Message, StreamEventType
from llm_client.spec import RequestContext, RequestSpec
from llm_client.structured import StructuredOutputConfig, extract_structured


CACHE_SCOPE = "cache-strategy-showdown"
WORKLOAD_PACKET = {
    "service": "incident-ops-control-plane",
    "objective": "choose the right repeat-request strategy for standalone package users and hosted multi-worker API consumers",
    "evaluation_axes": [
        "repeat latency improvement",
        "cache or replay hit behavior",
        "shared-fleet suitability",
        "duplicate-work protection",
    ],
    "release_context": "cache policy upgrades are part of the current rollout train",
}
SHARED_MESSAGES = [
    Message.system("Respond in 4 concise bullets. Keep wording stable across repeats."),
    Message.user(
        "Summarize why caching and idempotency matter for an LLM-backed incident operations control plane."
    ),
]


async def _timed_complete(
    engine: ExecutionEngine,
    spec: RequestSpec,
    *,
    context: RequestContext,
    cache_policy: CachePolicy | None = None,
    idempotency_key: str | None = None,
) -> tuple[float, CompletionResult]:
    started = time.perf_counter()
    result = await engine.complete(
        spec,
        context=context,
        cache_policy=cache_policy,
        idempotency_key=idempotency_key,
    )
    return (time.perf_counter() - started) * 1000.0, result


async def _bootstrap_memory(memory: ShortTermMemoryStore) -> list[dict[str, Any]]:
    entries = [
        MemoryWrite(
            scope=CACHE_SCOPE,
            content="For standalone package users, local filesystem cache is preferred when repeat latency drops cleanly and warm hits are reliable.",
            relevance=0.95,
            metadata={"kind": "standalone_rule"},
        ),
        MemoryWrite(
            scope=CACHE_SCOPE,
            content="For hosted multi-worker fleets, a centralized cache backend is preferred over per-node filesystem locality.",
            relevance=0.96,
            metadata={"kind": "shared_fleet_rule"},
        ),
        MemoryWrite(
            scope=CACHE_SCOPE,
            content="Idempotency prevents duplicate work and replay storms but is not a replacement for response caching.",
            relevance=0.97,
            metadata={"kind": "idempotency_rule"},
        ),
        MemoryWrite(
            scope=CACHE_SCOPE,
            content="Recommendations must call out repeat latency, hit behavior, and whether the strategy works across workers.",
            relevance=0.92,
            metadata={"kind": "decision_rule"},
        ),
    ]
    written: list[dict[str, Any]] = []
    for entry in entries:
        record = await memory.write(entry)
        written.append({"kind": record.metadata.get("kind"), "content": record.content})
    return written


def _safe_ratio(faster_baseline_ms: float, repeated_ms: float) -> float | None:
    if repeated_ms <= 0:
        return None
    return round(faster_baseline_ms / repeated_ms, 2)


def _diagnostics_payload(
    diagnostics: EngineDiagnosticsRecorder,
    context: RequestContext,
) -> dict[str, Any] | None:
    snapshot = diagnostics.latest_request(context.request_id)
    return snapshot.payload if snapshot else None


def _lifecycle_request_report(
    lifecycle: LifecycleRecorder,
    context: RequestContext,
) -> dict[str, Any] | None:
    report = lifecycle.requests.get(context.request_id)
    return report.to_dict() if report else None


def _strategy_record(
    *,
    name: str,
    cold_latency_ms: float | None = None,
    warm_latency_ms: float | None = None,
    replay_latency_ms: float | None = None,
    baseline_latency_ms: float,
    cold_result: CompletionResult | None = None,
    repeat_result: CompletionResult | None = None,
    request_payload: dict[str, Any] | None = None,
    cache_stats: dict[str, Any] | None = None,
    use_case: str,
) -> dict[str, Any]:
    repeat_ms = warm_latency_ms if warm_latency_ms is not None else replay_latency_ms
    return {
        "name": name,
        "use_case": use_case,
        "baseline_latency_ms": round(baseline_latency_ms, 2),
        "cold_latency_ms": round(cold_latency_ms, 2) if cold_latency_ms is not None else None,
        "warm_latency_ms": round(warm_latency_ms, 2) if warm_latency_ms is not None else None,
        "replay_latency_ms": round(replay_latency_ms, 2) if replay_latency_ms is not None else None,
        "repeat_speedup_ratio": _safe_ratio(
            cold_latency_ms if cold_latency_ms is not None else baseline_latency_ms,
            repeat_ms if repeat_ms is not None else 0.0,
        ),
        "repeat_faster_than_baseline": (repeat_ms < baseline_latency_ms) if repeat_ms is not None else None,
        "repeat_same_content": (
            bool(cold_result and repeat_result and (cold_result.content or "") == (repeat_result.content or ""))
        ),
        "request_report": request_payload,
        "cache_stats": cache_stats or {},
        "content_excerpt": ((repeat_result or cold_result).content or "")[:220] if (repeat_result or cold_result) else None,
    }


def _deterministic_scorecard(strategy_runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline_latency_ms = float(strategy_runs["no_cache"]["latency_ms"])
    fs = strategy_runs["fs_cache"]
    qdrant = strategy_runs["qdrant_cache"]
    idem = strategy_runs["idempotency_replay"]

    strongest_evidence: list[str] = []
    cautions: list[str] = []
    recommended_stack: list[str] = []

    if fs["cache_hit"]:
        strongest_evidence.append(
            f"fs_cache produced a warm cache hit with repeat latency {fs['warm_latency_ms']} ms and speedup ratio {fs['repeat_speedup_ratio']}."
        )
    else:
        cautions.append("fs_cache did not register a warm cache hit.")

    if qdrant["cache_hit"]:
        strongest_evidence.append(
            f"qdrant_cache produced a warm cache hit with repeat latency {qdrant['warm_latency_ms']} ms and speedup ratio {qdrant['repeat_speedup_ratio']}."
        )
    else:
        cautions.append("qdrant_cache did not register a warm cache hit.")

    if idem["idempotency_hit"]:
        strongest_evidence.append(
            f"idempotency_replay returned an idempotent replay with second latency {idem['second_latency_ms']} ms."
        )
    else:
        cautions.append("idempotency_replay did not register an idempotency hit.")

    if fs["repeat_speedup_ratio"] is not None and fs["repeat_speedup_ratio"] >= 1.1 and fs["cache_hit"]:
        recommended_stack.append("Use filesystem cache for standalone package and single-node local workflows.")
    else:
        cautions.append("filesystem cache did not show a decisive enough repeat advantage for standalone recommendation.")

    if qdrant["repeat_speedup_ratio"] is not None and qdrant["repeat_speedup_ratio"] >= 1.05 and qdrant["cache_hit"]:
        recommended_stack.append("Use Qdrant cache for hosted multi-worker API consumers that need shared repeat behavior.")
    else:
        cautions.append("Qdrant cache did not show a decisive enough repeat advantage for shared-fleet recommendation.")

    if idem["idempotency_hit"]:
        recommended_stack.append("Keep idempotency enabled for duplicate-safe retries and webhook or queue replay paths.")
    else:
        cautions.append("Idempotency replay path did not prove duplicate-work protection.")

    fastest_repeat_candidates = [
        ("fs_cache", fs["warm_latency_ms"]),
        ("qdrant_cache", qdrant["warm_latency_ms"]),
        ("idempotency_replay", idem["second_latency_ms"]),
    ]
    fastest_repeat_path, fastest_repeat_latency_ms = min(
        ((name, latency) for name, latency in fastest_repeat_candidates if latency is not None),
        key=lambda item: float(item[1]),
    )

    return {
        "overall_status": "cache_stack_ready" if len(recommended_stack) == 3 else "cache_stack_needs_attention",
        "baseline_latency_ms": round(baseline_latency_ms, 2),
        "fastest_repeat_path": fastest_repeat_path,
        "fastest_repeat_latency_ms": round(float(fastest_repeat_latency_ms), 2),
        "recommended_for_standalone": "fs_cache" if "filesystem cache" in " ".join(recommended_stack) else "undecided",
        "recommended_for_shared_fleet": "qdrant_cache" if "Qdrant cache" in " ".join(recommended_stack) else "undecided",
        "replay_safety_recommendation": "idempotency_replay" if idem["idempotency_hit"] else "undecided",
        "recommended_stack": recommended_stack,
        "strongest_evidence": strongest_evidence,
        "cautions": cautions,
        "evidence_used": list(strategy_runs.keys()),
    }


async def _stream_completion(
    engine: ExecutionEngine,
    spec: RequestSpec,
    *,
    context: RequestContext,
) -> tuple[CompletionResult, dict[str, Any]]:
    started = time.perf_counter()
    event_counts: Counter[str] = Counter()
    token_preview_parts: list[str] = []
    meta_events: list[dict[str, Any]] = []
    usage_events: list[dict[str, Any]] = []
    content_parts: list[str] = []
    final_result: CompletionResult | None = None
    error_payload: dict[str, Any] | None = None

    async for event in engine.stream(spec, context=context):
        event_counts[event.type.value] += 1
        if event.type == StreamEventType.TOKEN:
            token = str(event.data or "")
            content_parts.append(token)
            if sum(len(part) for part in token_preview_parts) < 320:
                token_preview_parts.append(token)
            continue
        if event.type == StreamEventType.META:
            payload = event.data if isinstance(event.data, dict) else {"value": str(event.data)}
            meta_events.append(payload)
            continue
        if event.type == StreamEventType.USAGE and hasattr(event.data, "to_dict"):
            usage_events.append(event.data.to_dict())
            continue
        if event.type == StreamEventType.ERROR:
            error_payload = event.data if isinstance(event.data, dict) else {"status": 500, "error": str(event.data)}
            break
        if event.type == StreamEventType.DONE:
            if isinstance(event.data, CompletionResult):
                final_result = event.data
            else:
                final_result = CompletionResult(
                    content="".join(content_parts).strip() or None,
                    status=200,
                    model=spec.model,
                    usage=None,
                )

    if final_result is None:
        final_result = CompletionResult(
            content="".join(content_parts).strip() or None,
            status=int((error_payload or {}).get("status", 500) or 500),
            error=str((error_payload or {}).get("error") or "Stream ended without a terminal completion result."),
            model=spec.model,
            usage=None,
        )

    return final_result, {
        "event_type_counts": dict(event_counts),
        "token_preview": "".join(token_preview_parts).strip(),
        "meta_events": meta_events,
        "usage_events": usage_events,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


def _normalize_structured_packet(
    structured_data: dict[str, Any] | None,
    *,
    deterministic_scorecard: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(structured_data or {})
    for key in (
        "recommended_stack",
        "strongest_evidence",
        "cautions",
        "evidence_used",
    ):
        normalized[key] = [
            str(item).strip()
            for item in list(normalized.get(key) or [])
            if str(item).strip()
        ]
    for key in (
        "overall_status",
        "baseline_latency_ms",
        "fastest_repeat_path",
        "fastest_repeat_latency_ms",
        "recommended_for_standalone",
        "recommended_for_shared_fleet",
        "replay_safety_recommendation",
    ):
        normalized[key] = deterministic_scorecard[key]
    if not normalized["recommended_stack"]:
        normalized["recommended_stack"] = list(deterministic_scorecard["recommended_stack"])
    else:
        normalized["recommended_stack"] = list(deterministic_scorecard["recommended_stack"])
    normalized["strongest_evidence"] = list(deterministic_scorecard["strongest_evidence"])
    normalized["cautions"] = list(deterministic_scorecard["cautions"])
    normalized["evidence_used"] = list(deterministic_scorecard["evidence_used"])
    return normalized


def _assembled_summary(structured_data: dict[str, Any] | None) -> str:
    if not structured_data:
        return ""
    stack = "\n".join(f"- {item}" for item in structured_data.get("recommended_stack", []))
    evidence = "\n".join(f"- {item}" for item in structured_data.get("strongest_evidence", []))
    cautions = "\n".join(f"- {item}" for item in structured_data.get("cautions", []))
    used = "\n".join(f"- {item}" for item in structured_data.get("evidence_used", []))
    return (
        f"Overall Status\n- {structured_data.get('overall_status')}\n\n"
        f"Decision\n- standalone={structured_data.get('recommended_for_standalone')} | "
        f"shared_fleet={structured_data.get('recommended_for_shared_fleet')} | "
        f"replay_safety={structured_data.get('replay_safety_recommendation')} | "
        f"fastest_repeat={structured_data.get('fastest_repeat_path')} ({structured_data.get('fastest_repeat_latency_ms')} ms)\n\n"
        f"Recommended Stack\n{stack}\n\n"
        f"Strongest Evidence\n{evidence}\n\n"
        f"Cautions\n{cautions}\n\n"
        f"Evidence Used\n{used}"
    ).strip()


async def main() -> None:
    handle = build_live_provider()
    try:
        qdrant_url = require_qdrant_url()
        run_suffix = uuid4().hex[:8]
        fs_collection = f"cookbook_cache_showdown_fs_{run_suffix}"
        qdrant_collection = f"cookbook_cache_showdown_qdrant_{run_suffix}"

        memory = ShortTermMemoryStore()
        memory_bootstrap = await _bootstrap_memory(memory)
        memory_notes = await memory.retrieve(MemoryQuery(scope=CACHE_SCOPE, limit=4))

        lifecycle = LifecycleRecorder()
        diagnostics = EngineDiagnosticsRecorder()
        hooks = HookManager([lifecycle, diagnostics])

        spec = RequestSpec(provider=handle.name, model=handle.model, messages=SHARED_MESSAGES)

        no_cache_engine = ExecutionEngine(provider=handle.provider, hooks=hooks)

        fs_cache_dir = Path(gettempdir()) / f"cookbook_cache_showdown_{run_suffix}"
        fs_engine = ExecutionEngine(
            provider=handle.provider,
            cache=build_cache_core(
                CacheSettings(
                    backend="fs",
                    client_type="completions",
                    default_collection=fs_collection,
                    cache_dir=fs_cache_dir,
                )
            ),
            hooks=hooks,
        )

        qdrant_engine = ExecutionEngine(
            provider=handle.provider,
            cache=build_cache_core(
                CacheSettings(
                    backend="qdrant",
                    client_type="completions",
                    default_collection=qdrant_collection,
                    qdrant_url=qdrant_url,
                    qdrant_api_key=qdrant_api_key(),
                )
            ),
            hooks=hooks,
        )

        idem_engine = ExecutionEngine(
            provider=handle.provider,
            idempotency_tracker=IdempotencyTracker(),
            hooks=hooks,
        )

        session_id = "cookbook-cache-strategy-showdown"

        no_cache_context = RequestContext(session_id=session_id, job_id="no-cache")
        no_cache_latency, no_cache_result = await _timed_complete(
            no_cache_engine,
            spec,
            context=no_cache_context,
        )

        fs_cold_context = RequestContext(session_id=session_id, job_id="fs-cold")
        fs_warm_context = RequestContext(session_id=session_id, job_id="fs-warm")
        fs_policy = CachePolicy.default_response(collection=fs_collection)
        fs_cold_latency, fs_cold_result = await _timed_complete(
            fs_engine,
            spec,
            context=fs_cold_context,
            cache_policy=fs_policy,
        )
        fs_warm_latency, fs_warm_result = await _timed_complete(
            fs_engine,
            spec,
            context=fs_warm_context,
            cache_policy=fs_policy,
        )

        qdrant_cold_context = RequestContext(session_id=session_id, job_id="qdrant-cold")
        qdrant_warm_context = RequestContext(session_id=session_id, job_id="qdrant-warm")
        qdrant_policy = CachePolicy.default_response(collection=qdrant_collection)
        qdrant_cold_latency, qdrant_cold_result = await _timed_complete(
            qdrant_engine,
            spec,
            context=qdrant_cold_context,
            cache_policy=qdrant_policy,
        )
        qdrant_warm_latency, qdrant_warm_result = await _timed_complete(
            qdrant_engine,
            spec,
            context=qdrant_warm_context,
            cache_policy=qdrant_policy,
        )

        idem_first_context = RequestContext(session_id=session_id, job_id="idempotency-first")
        idem_second_context = RequestContext(session_id=session_id, job_id="idempotency-second")
        idem_key = "cookbook-cache-showdown-idempotency"
        idem_first_latency, idem_first_result = await _timed_complete(
            idem_engine,
            spec,
            context=idem_first_context,
            idempotency_key=idem_key,
        )
        idem_second_latency, idem_second_result = await _timed_complete(
            idem_engine,
            spec,
            context=idem_second_context,
            idempotency_key=idem_key,
        )

        strategy_runs = {
            "no_cache": {
                "latency_ms": round(no_cache_latency, 2),
                "request_report": _lifecycle_request_report(lifecycle, no_cache_context),
                "content_excerpt": (no_cache_result.content or "")[:220],
            },
            "fs_cache": {
                **_strategy_record(
                    name="fs_cache",
                    cold_latency_ms=fs_cold_latency,
                    warm_latency_ms=fs_warm_latency,
                    baseline_latency_ms=no_cache_latency,
                    cold_result=fs_cold_result,
                    repeat_result=fs_warm_result,
                    request_payload=_lifecycle_request_report(lifecycle, fs_warm_context),
                    cache_stats=fs_engine.cache.get_stats().to_dict() if fs_engine.cache else {},
                    use_case="standalone_package",
                ),
                "cache_dir": str(fs_cache_dir),
                "collection": fs_collection,
                "cache_hit": bool((_lifecycle_request_report(lifecycle, fs_warm_context) or {}).get("cache_hit")),
            },
            "qdrant_cache": {
                **_strategy_record(
                    name="qdrant_cache",
                    cold_latency_ms=qdrant_cold_latency,
                    warm_latency_ms=qdrant_warm_latency,
                    baseline_latency_ms=no_cache_latency,
                    cold_result=qdrant_cold_result,
                    repeat_result=qdrant_warm_result,
                    request_payload=_lifecycle_request_report(lifecycle, qdrant_warm_context),
                    cache_stats=qdrant_engine.cache.get_stats().to_dict() if qdrant_engine.cache else {},
                    use_case="shared_hosted_fleet",
                ),
                "qdrant_url": qdrant_url,
                "collection": qdrant_collection,
                "cache_hit": bool((_lifecycle_request_report(lifecycle, qdrant_warm_context) or {}).get("cache_hit")),
            },
            "idempotency_replay": {
                "name": "idempotency_replay",
                "use_case": "duplicate_safe_retries",
                "first_latency_ms": round(idem_first_latency, 2),
                "second_latency_ms": round(idem_second_latency, 2),
                "repeat_speedup_ratio": _safe_ratio(idem_first_latency, idem_second_latency),
                "same_content": (idem_first_result.content or "") == (idem_second_result.content or ""),
                "idempotency_hit": bool(
                    (_lifecycle_request_report(lifecycle, idem_second_context) or {}).get("idempotency_hit")
                ),
                "request_report": _lifecycle_request_report(lifecycle, idem_second_context),
                "content_excerpt": (idem_second_result.content or "")[:220],
            },
        }

        deterministic_scorecard = _deterministic_scorecard(strategy_runs)

        memo_spec = RequestSpec(
            provider=handle.name,
            model=handle.model,
            messages=[
                Message.system(
                    "Write a cache strategy recommendation memo with sections: Baseline, Repeat Winners, Recommended Stack, Cautions. "
                    "Use strategy names exactly. Do not claim idempotency is a cache."
                ),
                Message.user(
                    json.dumps(
                        {
                            "workload_packet": WORKLOAD_PACKET,
                            "strategy_runs": strategy_runs,
                            "deterministic_scorecard": deterministic_scorecard,
                            "memory_notes": [record.content for record in memory_notes],
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                ),
            ],
        )
        memo_result, stream_summary = await _stream_completion(
            no_cache_engine,
            memo_spec,
            context=RequestContext(session_id=session_id, job_id="cache-strategy-memo"),
        )

        structured = await extract_structured(
            handle.provider,
            [
                Message.system(
                    "Convert the cache showdown result into a structured recommendation packet. "
                    "Use only executed strategy runs, deterministic scorecard, and memory notes. "
                    "Idempotency must remain a replay-safety recommendation, not a cache backend."
                ),
                Message.user(
                    json.dumps(
                        {
                            "workload_packet": WORKLOAD_PACKET,
                            "strategy_runs": strategy_runs,
                            "deterministic_scorecard": deterministic_scorecard,
                            "memo": memo_result.content,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                ),
            ],
            StructuredOutputConfig(
                schema={
                    "type": "object",
                    "properties": {
                        "overall_status": {"type": "string"},
                        "baseline_latency_ms": {"type": "number"},
                        "fastest_repeat_path": {"type": "string"},
                        "fastest_repeat_latency_ms": {"type": "number"},
                        "recommended_for_standalone": {"type": "string"},
                        "recommended_for_shared_fleet": {"type": "string"},
                        "replay_safety_recommendation": {"type": "string"},
                        "recommended_stack": {"type": "array", "items": {"type": "string"}},
                        "strongest_evidence": {"type": "array", "items": {"type": "string"}},
                        "cautions": {"type": "array", "items": {"type": "string"}},
                        "evidence_used": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "overall_status",
                        "baseline_latency_ms",
                        "fastest_repeat_path",
                        "fastest_repeat_latency_ms",
                        "recommended_for_standalone",
                        "recommended_for_shared_fleet",
                        "replay_safety_recommendation",
                        "recommended_stack",
                        "strongest_evidence",
                        "cautions",
                        "evidence_used",
                    ],
                    "additionalProperties": False,
                },
                max_repair_attempts=1,
            ),
            engine=no_cache_engine,
            context=RequestContext(session_id=session_id, job_id="structured-cache-showdown"),
            model=handle.model,
        )

        normalized_structured_data = _normalize_structured_packet(
            structured.data,
            deterministic_scorecard=deterministic_scorecard,
        )
        assembled_summary = _assembled_summary(normalized_structured_data)

        await memory.write(
            MemoryWrite(
                scope=CACHE_SCOPE,
                content=json.dumps(normalized_structured_data, ensure_ascii=True, sort_keys=True),
                relevance=0.96,
                metadata={"kind": "cache_strategy_packet"},
            )
        )
        memory_after = await memory.retrieve(MemoryQuery(scope=CACHE_SCOPE, limit=6))

        latest_request_report = list(lifecycle.requests.values())[-1] if lifecycle.requests else None
        latest_session_report = lifecycle.sessions.get(session_id)

        print_heading("Cache Strategy Showdown")
        print_json(
            {
                "provider": handle.name,
                "model": handle.model,
                "workload_packet": WORKLOAD_PACKET,
                "memory_bootstrap": memory_bootstrap,
                "strategy_runs": strategy_runs,
                "deterministic_scorecard": deterministic_scorecard,
                "stream_summary": stream_summary,
                "strategy_memo": {
                    "status": memo_result.status,
                    "usage": summarize_usage(memo_result.usage),
                    "content": memo_result.content,
                },
                "structured_packet": {
                    "valid": structured.valid,
                    "repair_attempts": structured.repair_attempts,
                    "usage": summarize_usage(getattr(structured, "usage", None)),
                    "data": normalized_structured_data,
                },
                "assembled_summary": assembled_summary,
                "observability": {
                    "hook_event_counts": dict(Counter(event for event, _, _ in diagnostics.events)),
                    "lifecycle_event_counts": dict(Counter(event.type.value for event in lifecycle.events)),
                    "latest_request_report": latest_request_report.to_dict() if latest_request_report else None,
                    "latest_session_report": latest_session_report.to_dict() if latest_session_report else None,
                },
                "memory_after_action": [
                    {"kind": record.metadata.get("kind"), "content": record.content}
                    for record in memory_after
                ],
                "showcase_verdict": {
                    "real_strategy_runs": bool(strategy_runs["fs_cache"]["cache_hit"]) and bool(strategy_runs["qdrant_cache"]["cache_hit"]),
                    "idempotency_replay_shown": bool(strategy_runs["idempotency_replay"]["idempotency_hit"]),
                    "deterministic_scorecard_present": bool(deterministic_scorecard["recommended_stack"]),
                    "structured_packet_ready": structured.valid and bool(normalized_structured_data.get("recommended_stack")),
                    "operator_ready": bool(assembled_summary),
                },
            }
        )
    finally:
        await close_provider(handle.provider)


if __name__ == "__main__":
    asyncio.run(main())
