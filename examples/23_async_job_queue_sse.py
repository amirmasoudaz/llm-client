from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
import time
import uuid
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from cookbook_support import build_live_provider, close_provider, require_optional_module

if not require_optional_module("fastapi", "Install it with: pip install fastapi uvicorn"):
    raise SystemExit(0)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llm_client.engine import ExecutionEngine
from llm_client.providers.types import Message
from llm_client.spec import RequestSpec
from llm_client.streaming import format_sse_event


class SupportJobRequest(BaseModel):
    case_id: str
    issue_summary: str
    business_impact: str
    symptoms: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None


@dataclass
class _JobState:
    job_id: str
    created_at: float
    request: dict[str, Any]
    status: str = "queued"
    result: dict[str, Any] | None = None
    artifact_id: str | None = None
    error: str | None = None
    event_log: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class _JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, _JobState] = {}
        self.idempotency_index: dict[str, str] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}

    def create_or_reuse(self, request: dict[str, Any]) -> tuple[_JobState, bool]:
        idempotency_key = request.get("idempotency_key")
        if idempotency_key and idempotency_key in self.idempotency_index:
            return self.jobs[self.idempotency_index[idempotency_key]], True
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        state = _JobState(job_id=job_id, created_at=time.time(), request=request)
        self.jobs[job_id] = state
        if idempotency_key:
            self.idempotency_index[idempotency_key] = job_id
        return state, False

    def get(self, job_id: str) -> _JobState | None:
        return self.jobs.get(job_id)

    def save_artifact(self, job: _JobState, content: dict[str, Any]) -> str:
        artifact_id = f"artifact_{job.job_id}"
        self.artifacts[artifact_id] = content
        job.artifact_id = artifact_id
        return artifact_id

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return self.artifacts.get(artifact_id)


def _record_history(job: _JobState, status: str, detail: str) -> None:
    job.history.append({"at": round(time.time(), 3), "status": status, "detail": detail})


async def _emit(job: _JobState, event: str, data: dict[str, Any]) -> None:
    item = {
        "id": len(job.event_log) + 1,
        "event": event,
        "data": data,
    }
    async with job.condition:
        job.event_log.append(item)
        job.condition.notify_all()


async def _run_job(
    job: _JobState,
    store: _JobStore,
    engine: ExecutionEngine,
    provider_name: str,
    model_name: str,
) -> None:
    job.status = "running"
    _record_history(job, job.status, "Job dequeued for execution")
    await _emit(job, "job.started", {"job_id": job.job_id, "status": job.status})
    await asyncio.sleep(0.05)
    _record_history(job, job.status, "Building escalation context")
    await _emit(job, "job.progress", {"stage": "triage", "message": "Building escalation context"})
    await asyncio.sleep(0.05)
    _record_history(job, job.status, "Generating executive-ready brief")
    await _emit(job, "job.progress", {"stage": "analysis", "message": "Generating executive-ready brief"})

    payload = job.request
    try:
        result = await engine.complete(
            RequestSpec(
                provider=provider_name,
                model=model_name,
                messages=[
                    Message.system(
                        "Draft a concise support escalation brief with sections: Situation, Likely Owner, Immediate Action, Customer Update Guidance, Handoff Status."
                    ),
                    Message.user(
                        "Support escalation packet:\n"
                        f"case_id={payload['case_id']}\n"
                        f"issue_summary={payload['issue_summary']}\n"
                        f"business_impact={payload['business_impact']}\n"
                        f"symptoms={payload['symptoms']}"
                    ),
                ],
            )
        )
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        _record_history(job, job.status, "Job failed during model execution")
        await _emit(
            job,
            "job.failed",
            {"job_id": job.job_id, "status": job.status, "error": job.error},
        )
        await _emit(job, "job.end", {"job_id": job.job_id, "status": job.status})
        return

    artifact = {
        "job_id": job.job_id,
        "case_id": payload["case_id"],
        "artifact_type": "support_escalation_brief",
        "generated_at": round(time.time(), 3),
        "provider": provider_name,
        "model": result.model,
        "content": result.content,
        "usage": result.usage.to_dict() if result.usage else None,
    }
    artifact_id = store.save_artifact(job, artifact)
    job.result = {
        "status": result.status,
        "model": result.model,
        "artifact_id": artifact_id,
        "artifact_url": f"/v1/jobs/{job.job_id}/artifact",
        "content_preview": (result.content or "")[:240],
        "usage": result.usage.to_dict() if result.usage else None,
    }
    job.status = "completed" if result.status == 200 else "failed"
    if job.status == "failed":
        job.error = f"provider returned status {result.status}"
    _record_history(job, job.status, "Artifact stored and job finalized")
    await _emit(
        job,
        "job.completed",
        {"job_id": job.job_id, "status": job.status, "result": job.result},
    )
    await _emit(job, "job.end", {"job_id": job.job_id, "status": job.status})


@asynccontextmanager
async def lifespan(app: FastAPI):
    handle = build_live_provider()
    app.state.handle = handle
    app.state.engine = ExecutionEngine(provider=handle.provider)
    app.state.jobs = _JobStore()
    try:
        yield
    finally:
        await close_provider(handle.provider)


app = FastAPI(title="llm_client async job queue cookbook", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    handle = app.state.handle
    return JSONResponse({"status": "ok", "provider": handle.name, "model": handle.model, "service": "job-queue-sse"})


@app.post("/v1/jobs/support-brief")
async def submit_job(payload: SupportJobRequest) -> JSONResponse:
    job, idempotency_hit = app.state.jobs.create_or_reuse(payload.model_dump())
    if not idempotency_hit:
        _record_history(job, job.status, "Job accepted into queue")
        await _emit(
            job,
            "job.accepted",
            {
                "job_id": job.job_id,
                "status": job.status,
                "idempotency_key": payload.idempotency_key,
            },
        )
        asyncio.create_task(_run_job(job, app.state.jobs, app.state.engine, app.state.handle.name, app.state.handle.model))
    return JSONResponse(
        {
            "job_id": job.job_id,
            "status": job.status,
            "idempotency_key": payload.idempotency_key,
            "idempotency_hit": idempotency_hit,
            "poll_after_ms": 500,
            "status_url": f"/v1/jobs/{job.job_id}",
            "sse_url": f"/v1/jobs/{job.job_id}/events",
            "artifact_url": f"/v1/jobs/{job.job_id}/artifact",
        }
    )


@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    job = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse(
        {
            "job_id": job.job_id,
            "status": job.status,
            "created_at": job.created_at,
            "idempotency_key": job.request.get("idempotency_key"),
            "history": job.history,
            "error": job.error,
            "result": job.result,
        }
    )


@app.get("/v1/jobs/{job_id}/artifact")
async def get_job_artifact(job_id: str) -> JSONResponse:
    job = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.artifact_id is None:
        raise HTTPException(status_code=409, detail="artifact not ready")
    artifact = app.state.jobs.get_artifact(job.artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return JSONResponse(artifact)


@app.get("/v1/jobs/{job_id}/events")
async def stream_job(job_id: str) -> StreamingResponse:
    job = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def _events():
        cursor = 0
        yield format_sse_event(
            "job.meta",
            json.dumps(
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "replayable": True,
                    "existing_events": len(job.event_log),
                }
            ),
        )
        while True:
            async with job.condition:
                while cursor >= len(job.event_log):
                    if job.status in {"completed", "failed"}:
                        return
                    await job.condition.wait()
                item = job.event_log[cursor]
                cursor += 1
            yield format_sse_event(item["event"], json.dumps(item["data"], default=str))
            if item["event"] == "job.end":
                return

    return StreamingResponse(_events(), media_type="text/event-stream")


async def _preview() -> None:
    print("Async job queue SSE app loaded.")
    print("Run with:")
    print("  uvicorn examples.llm_client.23_async_job_queue_sse:app --reload")
    print("Submit a job:")
    print(
        "  curl -sS -X POST http://127.0.0.1:8000/v1/jobs/support-brief "
        "-H 'Content-Type: application/json' "
        "-d '{\"case_id\":\"SUP-5001\",\"issue_summary\":\"Exports timing out after audit logging change\",\"business_impact\":\"Finance reconciliation blocked\",\"symptoms\":[\"queue lag rising\",\"timeouts around 5 minutes\"],\"idempotency_key\":\"support-brief-sup-5001\"}'"
    )
    print("Watch the SSE stream:")
    print("  curl -N http://127.0.0.1:8000/v1/jobs/<job_id>/events")
    print("Fetch the final artifact:")
    print("  curl -sS http://127.0.0.1:8000/v1/jobs/<job_id>/artifact")


if __name__ == "__main__":
    asyncio.run(_preview())
