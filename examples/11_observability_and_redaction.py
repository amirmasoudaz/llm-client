from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import uuid
from typing import Any

from llm_client import (
    EngineDiagnosticsRecorder,
    ExecutionEngine,
    HookManager,
    LifecycleLoggingHook,
    LifecycleRecorder,
    LifecycleTelemetryHook,
    Message,
    MetricRegistry,
    OpenAIProvider,
    PayloadPreviewMode,
    ProviderPayloadCaptureMode,
    RedactionPolicy,
    RequestContext,
    RequestSpec,
    ToolOutputPolicy,
    UsageTracker,
    capture_provider_payload,
    load_env,
    preview_payload,
    sanitize_payload,
    sanitize_tool_output,
)
from llm_client.cache import CacheCore, CachePolicy
from llm_client.cache.base import BaseCacheBackend
from llm_client.idempotency import IdempotencyTracker
from llm_client.logging import StructuredLogger

load_env()


class _InMemoryCacheBackend(BaseCacheBackend):
    name = "fs"
    default_collection = "cookbook-observability"

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], dict[str, object]] = {}

    async def ensure_ready(self) -> None:
        return None

    async def exists(self, effective_key: str, collection: str | None = None) -> bool:
        return (effective_key, collection or self.default_collection) in self._entries

    async def read(self, effective_key: str, collection: str | None = None) -> dict[str, object] | None:
        return self._entries.get((effective_key, collection or self.default_collection))

    async def write(
        self,
        effective_key: str,
        response: dict[str, object],
        model_name: str,
        collection: str | None = None,
    ) -> None:
        _ = model_name
        self._entries[(effective_key, collection or self.default_collection)] = dict(response)


def _make_capturing_logger(policy: RedactionPolicy) -> tuple[StructuredLogger, io.StringIO]:
    stream = io.StringIO()
    logger = StructuredLogger(
        name=f"llm_client.cookbook.observability.{uuid.uuid4().hex}",
        redaction_policy=policy,
    )
    logger._logger.handlers.clear()
    logger._logger.propagate = False
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger._logger.addHandler(handler)
    return logger, stream


def _result_excerpt(text: str | None, *, max_chars: int = 220) -> str | None:
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "…"


def _extract_log_samples(lines: list[str]) -> list[dict[str, Any]]:
    parsed = [json.loads(line) for line in lines if line.strip()]
    samples: list[dict[str, Any]] = []

    for entry in parsed:
        event_type = entry.get("event_type")
        if event_type == "lifecycle" and entry.get("lifecycle_type") == "request.started":
            samples.append(
                {
                    "event_type": event_type,
                    "message": entry.get("message"),
                    "payload_preview": entry.get("payload_preview"),
                }
            )
            break

    for target in ("request_report", "session_report"):
        for entry in parsed:
            if entry.get("event_type") == target:
                samples.append(
                    {
                        "event_type": target,
                        "message": entry.get("message"),
                        "request_id": entry.get("request_id"),
                        "session_id": entry.get("session_id"),
                        "status": entry.get("status"),
                        "cache_hit": entry.get("cache_hit"),
                        "idempotency_hit": entry.get("idempotency_hit"),
                        "usage": entry.get("usage_preview") or entry.get("usage"),
                    }
                )
                break

    return samples


def _select_counters(snapshot: dict[str, Any]) -> dict[str, int]:
    counters = snapshot.get("counters", {})
    keys = [
        "llm.lifecycle.request_started",
        "llm.lifecycle.request_completed",
        "llm.lifecycle.cache_hit",
        "llm.lifecycle.idempotency_hit",
        "llm.requests.total",
        "llm.tokens.input",
        "llm.tokens.output",
        "llm.cache.hits",
    ]
    return {key: counters.get(key, 0) for key in keys}


async def main() -> None:
    model_name = os.getenv("LLM_CLIENT_EXAMPLE_MODEL", "gpt-5-nano")
    provider_name = "openai"
    provider = OpenAIProvider(model=model_name)
    try:
        escalation_packet = {
            "case_id": "SUP-2081",
            "customer_tier": "enterprise",
            "customer_email": "morgan@acme.example",
            "auth_token": "tok_live_demo_secret",
            "issue_summary": "Workspace export jobs started timing out after audit logging was enabled.",
            "business_impact": "Finance and compliance teams cannot complete month-end reconciliation.",
            "internal_notes": "Renewal is in 14 days. Customer has threatened escalation to procurement.",
            "observed_symptoms": [
                "timeouts begin around the 5-minute mark",
                "the queue drains slowly after large CSV exports",
                "support suspects audit-log fanout overhead",
            ],
        }
        model_packet = {
            "case_id": escalation_packet["case_id"],
            "customer_tier": escalation_packet["customer_tier"],
            "issue_summary": escalation_packet["issue_summary"],
            "business_impact": escalation_packet["business_impact"],
            "observed_symptoms": escalation_packet["observed_symptoms"],
        }

        redaction_policy = RedactionPolicy(
            sensitive_keys=("customer_email", "auth_token", "internal_notes", "secret"),
            preview_mode=PayloadPreviewMode.SUMMARY,
            provider_payload_capture=ProviderPayloadCaptureMode.METADATA_ONLY,
            preview_max_chars=96,
        )
        logger, log_stream = _make_capturing_logger(redaction_policy)

        diagnostics = EngineDiagnosticsRecorder(redaction_policy=redaction_policy)
        lifecycle = LifecycleRecorder(redaction_policy=redaction_policy)
        registry = MetricRegistry()
        usage_tracker = UsageTracker(registry)
        telemetry = LifecycleTelemetryHook(
            registry=registry,
            usage_tracker=usage_tracker,
            redaction_policy=redaction_policy,
        )
        lifecycle_logging = LifecycleLoggingHook(
            logger=logger,
            redaction_policy=redaction_policy,
            include_session_reports=True,
        )
        engine = ExecutionEngine(
            provider=provider,
            cache=CacheCore(_InMemoryCacheBackend()),
            idempotency_tracker=IdempotencyTracker(),
            hooks=HookManager([diagnostics, lifecycle, telemetry, lifecycle_logging]),
        )

        session_id = "cookbook-observability-session"

        triage_spec = RequestSpec(
            provider=provider_name,
            model=model_name,
            messages=[
                Message.system(
                    "You are an LLM platform operations assistant. "
                    "Produce concise, operator-facing triage notes grounded in the input."
                ),
                Message.user(
                    "Prepare an internal support triage note with:\n"
                    "1. severity\n"
                    "2. likely root cause\n"
                    "3. immediate next actions\n\n"
                    f"Escalation packet:\n{json.dumps(model_packet, ensure_ascii=False)}"
                ),
            ],
        )

        cold_context = RequestContext(session_id=session_id)
        cold_result = await engine.complete(
            triage_spec,
            context=cold_context,
            cache_policy=CachePolicy.default_response(collection="support-observability"),
        )

        warm_context = RequestContext(session_id=session_id)
        warm_result = await engine.complete(
            triage_spec,
            context=warm_context,
            cache_policy=CachePolicy.default_response(collection="support-observability"),
        )

        reply_spec = RequestSpec(
            provider=provider_name,
            model=model_name,
            messages=[
                Message.system(
                    "You are a customer support incident communicator. "
                    "Write a calm, customer-safe update without exposing internal hypotheses."
                ),
                Message.user(
                    "Write a 3-bullet customer update for this escalation. "
                    "Mention impact, current mitigation work, and what happens next.\n\n"
                    f"Escalation packet:\n{json.dumps(model_packet, ensure_ascii=False)}"
                ),
            ],
        )

        reply_first_context = RequestContext(session_id=session_id)
        reply_second_context = RequestContext(session_id=session_id)
        reply_first = await engine.complete(
            reply_spec,
            context=reply_first_context,
            idempotency_key="cookbook-observability-customer-update",
        )
        reply_second = await engine.complete(
            reply_spec,
            context=reply_second_context,
            idempotency_key="cookbook-observability-customer-update",
        )

        session_report = lifecycle.sessions[session_id].to_dict()
        telemetry_snapshot = registry.snapshot()
        usage_summary = usage_tracker.get_session_summary(session_id)
        parsed_logs = _extract_log_samples(log_stream.getvalue().splitlines())

        print("\n=== Escalation Packet ===\n")
        print(
            json.dumps(
                {
                    "raw_packet": escalation_packet,
                    "model_packet": model_packet,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Live Request Reports ===\n")
        print(
            json.dumps(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "triage_cold": {
                        "content_excerpt": _result_excerpt(cold_result.content),
                        "request_report": lifecycle.requests[cold_context.request_id].to_dict(),
                        "diagnostics": diagnostics.latest_request(cold_context.request_id).payload
                        if diagnostics.latest_request(cold_context.request_id)
                        else {},
                    },
                    "triage_warm_cache_hit": {
                        "content_excerpt": _result_excerpt(warm_result.content),
                        "request_report": lifecycle.requests[warm_context.request_id].to_dict(),
                        "diagnostics": diagnostics.latest_request(warm_context.request_id).payload
                        if diagnostics.latest_request(warm_context.request_id)
                        else {},
                    },
                    "customer_update_idempotent_replay": {
                        "same_content": reply_first.content == reply_second.content,
                        "first_excerpt": _result_excerpt(reply_first.content),
                        "second_request_report": lifecycle.requests[reply_second_context.request_id].to_dict(),
                        "second_diagnostics": diagnostics.latest_request(reply_second_context.request_id).payload
                        if diagnostics.latest_request(reply_second_context.request_id)
                        else {},
                    },
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Session Telemetry + Logs ===\n")
        print(
            json.dumps(
                {
                    "session_report": session_report,
                    "usage_tracker_summary": usage_summary,
                    "selected_metric_counters": _select_counters(telemetry_snapshot),
                    "latency_histogram": telemetry_snapshot.get("histograms", {}).get("llm.request.latency_ms", {}),
                    "structured_log_samples": parsed_logs,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        print("\n=== Redaction Utilities ===\n")
        print(
            json.dumps(
                {
                    "sanitize_payload": sanitize_payload(escalation_packet, policy=redaction_policy),
                    "preview_payload": preview_payload(escalation_packet, policy=redaction_policy),
                    "provider_payload_capture": capture_provider_payload(
                        {
                            "id": "resp_demo_123",
                            "model": model_name,
                            "status": 200,
                            "raw_response": {"secret": "should-not-leak"},
                            "usage": {"total_tokens": 123},
                        },
                        policy=redaction_policy,
                    ),
                    "sanitize_tool_output": sanitize_tool_output(
                        "Customer email morgan@acme.example token=tok_live_demo_secret "
                        "requested priority follow-up before renewal.",
                        policy=ToolOutputPolicy(max_chars=96),
                    ),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
