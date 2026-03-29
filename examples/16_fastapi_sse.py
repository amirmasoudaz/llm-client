from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
import time
import uuid
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from cookbook_support import (
    build_live_provider,
    close_provider,
    require_optional_module,
    resolve_model_name,
    resolve_provider_name,
)

if not require_optional_module("fastapi", "Install it with: pip install fastapi uvicorn"):
    raise SystemExit(0)

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import CompletionResult, Message, StreamEvent, StreamEventType
from llm_client.streaming import format_sse_event
from llm_client.spec import RequestSpec


class IncidentBriefRequest(BaseModel):
    service: str = Field(..., description="Impacted service name.")
    business_impact: str = Field(..., description="Customer or business impact.")
    symptoms: list[str] = Field(default_factory=list, description="Observed symptoms or alerts.")
    severity_hint: str | None = Field(default=None, description="Optional severity guess.")
    owner_hint: str | None = Field(default=None, description="Optional suspected owner.")
    deployment_context: str | None = Field(default=None, description="Recent rollout or config context.")


def _prompt_from_request(payload: IncidentBriefRequest) -> list[Message]:
    symptoms = "\n".join(f"- {item}" for item in payload.symptoms) if payload.symptoms else "- none supplied"
    return [
        Message.system(
            "You are an incident commander assistant. Stream a concise operational briefing in markdown. "
            "Use only supplied facts. Structure the response with these headings: Severity, Likely Cause, "
            "Immediate Actions, Owners, Evidence."
        ),
        Message.user(
            "\n".join(
                [
                    f"Service: {payload.service}",
                    f"Business impact: {payload.business_impact}",
                    f"Severity hint: {payload.severity_hint or 'unspecified'}",
                    f"Owner hint: {payload.owner_hint or 'unspecified'}",
                    f"Deployment context: {payload.deployment_context or 'not provided'}",
                    "Observed symptoms:",
                    symptoms,
                ]
            )
        ),
    ]


def _json_sse(event_name: str, payload: dict[str, Any]) -> str:
    return format_sse_event(event_name, json.dumps(payload, ensure_ascii=False, default=str))


def _serialize_completion(result: CompletionResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "model": result.model,
        "finish_reason": result.finish_reason,
        "usage": result.usage.to_dict() if result.usage else None,
        "content_excerpt": (result.content or "")[:240] if isinstance(result.content, str) else None,
        "error": result.error,
    }


def _serialize_stream_event(
    event: StreamEvent,
    *,
    request_id: str,
    started_at: float,
) -> tuple[str, dict[str, Any]]:
    elapsed_ms = round((time.monotonic() - started_at) * 1000, 2)
    base: dict[str, Any] = {
        "request_id": request_id,
        "elapsed_ms": elapsed_ms,
    }
    if event.type == StreamEventType.TOKEN:
        return "token", {**base, "delta": event.data}
    if event.type == StreamEventType.REASONING:
        return "reasoning", {**base, "delta": event.data}
    if event.type == StreamEventType.META:
        return "provider.meta", {**base, "data": event.data}
    if event.type == StreamEventType.USAGE:
        usage = event.data.to_dict() if hasattr(event.data, "to_dict") else event.data
        return "usage", {**base, "usage": usage}
    if event.type == StreamEventType.TOOL_CALL_START:
        return "tool_call_start", {
            **base,
            "id": event.data.id,
            "index": event.data.index,
            "name": event.data.name,
        }
    if event.type == StreamEventType.TOOL_CALL_DELTA:
        return "tool_call_delta", {
            **base,
            "id": event.data.id,
            "index": event.data.index,
            "arguments_delta": event.data.arguments_delta,
        }
    if event.type == StreamEventType.TOOL_CALL_END:
        return "tool_call_end", {
            **base,
            "id": event.data.id,
            "name": event.data.name,
            "arguments": event.data.arguments,
        }
    if event.type == StreamEventType.ERROR:
        return "error", {**base, "error": event.data}
    if event.type == StreamEventType.DONE:
        return "done", {**base, "result": _serialize_completion(event.data)}
    return event.type.value, {**base, "data": str(event.data)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    handle = build_live_provider()
    app.state.handle = handle
    app.state.engine = ExecutionEngine(provider=handle.provider)
    try:
        yield
    finally:
        await close_provider(handle.provider)


app = FastAPI(
    title="llm_client FastAPI SSE cookbook",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    handle = app.state.handle
    return JSONResponse(
        {
            "status": "ok",
            "provider": handle.name,
            "model": handle.model,
            "service": "incident-brief-stream",
        }
    )


@app.post("/v1/incident-brief/stream")
async def incident_brief_stream(payload: IncidentBriefRequest) -> StreamingResponse:
    handle = app.state.handle
    engine = app.state.engine
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    started_at = time.monotonic()

    async def _events():
        yield _json_sse(
            "server.meta",
            {
                "request_id": request_id,
                "provider": handle.name,
                "model": handle.model,
                "service": payload.service,
                "endpoint": "/v1/incident-brief/stream",
            },
        )
        async for event in engine.stream(
            RequestSpec(
                provider=handle.name,
                model=handle.model,
                messages=_prompt_from_request(payload),
            )
        ):
            event_name, body = _serialize_stream_event(
                event,
                request_id=request_id,
                started_at=started_at,
            )
            yield _json_sse(event_name, body)
        yield _json_sse(
            "server.end",
            {
                "request_id": request_id,
                "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
            },
        )

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _preview() -> None:
    provider_name = resolve_provider_name()
    model_name = resolve_model_name(provider_name)
    sample_payload = {
        "service": "checkout-api",
        "business_impact": "Checkout failures are blocking payment completion for a subset of users.",
        "symptoms": [
            "5xx errors spiked after a payment routing rollout",
            "checkout queue lag is rising",
            "support reports intermittent webhook delays",
        ],
        "severity_hint": "sev-1",
        "owner_hint": "payments-platform",
        "deployment_context": "payment routing config changed 12 minutes ago",
    }
    print("FastAPI SSE app loaded for a live incident-brief streaming endpoint.")
    print(f"provider={provider_name} model={model_name}")
    print("Run with:")
    print("  uvicorn examples.llm_client.16_fastapi_sse:app --reload")
    print("Health check:")
    print("  curl -sS http://127.0.0.1:8000/healthz")
    print("Stream request:")
    print(
        "  curl -N -X POST http://127.0.0.1:8000/v1/incident-brief/stream "
        "-H 'Content-Type: application/json' "
        f"-d '{json.dumps(sample_payload, ensure_ascii=False)}'"
    )


if __name__ == "__main__":
    asyncio.run(_preview())
