from __future__ import annotations

import uuid
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi import Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_runtime import ExecutionRequest, BudgetSpec, PolicyRef

from llm_client import load_env

from .il_db import ILDB, get_pool
from .runtime_kernel import KernelContainer, build_kernel
from .settings import get_settings
from intelligence_layer_ops.platform_tools import platform_load_funding_thread_context


load_env()  # keep prototyping simple: auto-load repo `.env` for the Layer 2 API process

app = FastAPI(title="CanApply Intelligence Layer (Layer 2)", version="0.0.1")


class ThreadInitRequest(BaseModel):
    student_id: int = Field(..., ge=1)
    funding_request_id: int = Field(..., ge=1)
    client_context: dict[str, Any] | None = None


class ThreadInitResponse(BaseModel):
    thread_id: str
    thread_status: str
    is_new: bool = False
    message: str = "existing_thread"
    onboarding_gate: str = "ready"
    missing_requirements: list[str] = Field(default_factory=list)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"ok": "true"}


@app.get("/v1/debug/platform/funding_request/{funding_request_id}/context")
async def debug_platform_context(funding_request_id: int) -> dict[str, Any]:
    settings = get_settings()
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not found")
    # `platform_load_funding_thread_context` is a llm-client Tool object (not directly callable).
    # For debug we execute the tool and unwrap its result.
    result = await platform_load_funding_thread_context.execute(funding_request_id=funding_request_id)
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "platform context load failed")
    if isinstance(result.content, dict):
        return result.content
    raise HTTPException(status_code=500, detail="platform context load returned non-object")


@app.on_event("startup")
async def _startup() -> None:
    settings = get_settings()
    pool = await get_pool(settings.il_pg_dsn)
    ildb = ILDB(pool=pool)
    await ildb.ensure_schema()

    kernel_container = await build_kernel(pg_pool=pool)

    app.state.ildb = ildb
    app.state.kernel_container = kernel_container


@app.on_event("shutdown")
async def _shutdown() -> None:
    kc: KernelContainer | None = getattr(app.state, "kernel_container", None)
    if kc is not None:
        try:
            await kc.engine.provider.close()
        except Exception:
            pass


@app.post("/v1/threads/init", response_model=ThreadInitResponse)
async def init_thread(req: ThreadInitRequest, response: Response) -> ThreadInitResponse:
    ildb: ILDB = app.state.ildb
    thread_id, status, is_new = await ildb.get_or_create_thread(student_id=req.student_id, funding_request_id=req.funding_request_id)
    response.status_code = 201 if is_new else 200
    return ThreadInitResponse(
        thread_id=str(thread_id),
        thread_status=status,
        is_new=is_new,
        message="created_thread" if is_new else "existing_thread",
    )


class SubmitQueryRequest(BaseModel):
    message: str = Field(..., min_length=1)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    # Optional client-supplied idempotency/correlation ID (UUID)
    query_id: str | None = None

    # Debug-only execution overrides (require IL_DEBUG=true)
    operator_id: str | None = None
    metadata: dict[str, Any] | None = None
    budgets: dict[str, Any] | None = None
    policy_ref: dict[str, Any] | None = None
    max_turns: int | None = Field(default=None, ge=1, le=200)


class SubmitQueryResponse(BaseModel):
    query_id: str
    sse_url: str


@app.post("/v1/threads/{thread_id}/queries", response_model=SubmitQueryResponse)
async def submit_query(thread_id: str, req: SubmitQueryRequest) -> SubmitQueryResponse:
    ildb: ILDB = app.state.ildb
    kc: KernelContainer = app.state.kernel_container
    settings = get_settings()

    try:
        thread_id_int = int(thread_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="thread_id must be an integer")

    thread = await ildb.get_thread(thread_id=thread_id_int)
    if not thread:
        raise HTTPException(status_code=404, detail="thread not found")

    # Query idempotency: allow client to supply query_id (UUID). If it already exists for
    # this thread, return the existing mapping; if it exists for a different thread, 409.
    if req.query_id is not None:
        try:
            query_uuid = uuid.UUID(req.query_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="query_id must be a UUID")

        existing = await ildb.get_query(query_id=str(query_uuid))
        if existing is not None:
            if int(existing["thread_id"]) != thread_id_int:
                raise HTTPException(status_code=409, detail="query_id already exists for a different thread")
            return SubmitQueryResponse(query_id=str(query_uuid), sse_url=f"/v1/queries/{query_uuid}/events")
        query_id = str(query_uuid)
    else:
        query_id = str(uuid.uuid4())

    # Debug-only overrides
    operator_id = None
    extra_metadata: dict[str, Any] = {}
    budgets_obj: BudgetSpec | None = None
    policy_obj: PolicyRef | None = None
    max_turns = 10

    if (
        req.operator_id is not None
        or req.metadata is not None
        or req.budgets is not None
        or req.policy_ref is not None
        or req.max_turns is not None
    ):
        if not settings.debug:
            raise HTTPException(status_code=400, detail="debug-only fields require IL_DEBUG=true")
        operator_id = req.operator_id
        extra_metadata = dict(req.metadata or {})
        budgets_obj = BudgetSpec.from_dict(req.budgets) if req.budgets else None
        policy_obj = PolicyRef.from_dict(req.policy_ref) if req.policy_ref else None
        max_turns = int(req.max_turns or max_turns)

    handle = await kc.kernel.execute(
        ExecutionRequest(
            prompt=req.message,
            scope_id=str(ildb.tenant_id),
            principal_id=str(thread["student_id"]),
            session_id=str(thread_id_int),
            run_id=query_id,
            operator_id=operator_id,
            budgets=budgets_obj,
            policy_ref=policy_obj,
            max_turns=max_turns,
            metadata={
                "funding_request_id": int(thread["funding_request_id"]),
                "thread_id": thread_id_int,
                **{
                    k: v
                    for (k, v) in extra_metadata.items()
                    if k not in {"funding_request_id", "thread_id"}
                },
            },
        )
    )
    await ildb.insert_query(query_id=query_id, thread_id=thread_id_int, job_id=handle.job_id)

    return SubmitQueryResponse(query_id=query_id, sse_url=f"/v1/queries/{query_id}/events")


@app.get("/v1/queries/{query_id}/events")
async def query_events(query_id: str) -> StreamingResponse:
    ildb: ILDB = app.state.ildb
    _kc: KernelContainer = app.state.kernel_container

    job_id = await ildb.get_job_id_for_query(query_id=query_id)
    if not job_id:
        raise HTTPException(status_code=404, detail="query not found")

    async def gen() -> AsyncIterator[str]:
        last_ts = 0.0
        terminal = {"final.result", "final.error", "job.cancelled"}
        # Poll DB for new events so SSE works even if events were produced by another process.
        while True:
            rows = await ildb.list_runtime_events(job_id=job_id, after_ts=last_ts, limit=200)
            for r in rows:
                last_ts = max(last_ts, float(r["ts"]))
                event_type = str(r["type"])
                payload = {
                    "event_id": r["event_id"],
                    "type": event_type,
                    "timestamp": float(r["ts"]),
                    "job_id": r["job_id"],
                    "run_id": r["run_id"],
                    "trace_id": r["trace_id"],
                    "span_id": r["span_id"],
                    "scope_id": r["scope_id"],
                    "principal_id": r["principal_id"],
                    "session_id": r["session_id"],
                    "data": r["data"] or {},
                    "schema_version": int(r.get("schema_version") or 1),
                }
                import json

                yield f"event: {event_type.replace('.', '_')}\ndata: {json.dumps(payload)}\n\n"
                if event_type in terminal:
                    return

            # Backoff a bit if nothing new.
            import asyncio

            await asyncio.sleep(0.25)

    return StreamingResponse(gen(), media_type="text/event-stream")


class ResolveActionRequest(BaseModel):
    status: str = Field(..., pattern="^(accepted|declined)$")
    payload: dict[str, Any] = Field(default_factory=dict)


@app.post("/v1/actions/{action_id}/resolve")
async def resolve_action(action_id: str, req: ResolveActionRequest) -> dict[str, Any]:
    kc: KernelContainer = app.state.kernel_container
    # v0: treat accepted/declined as a generic resolution payload
    resolution = {"status": req.status, **req.payload}
    await kc.kernel.resolve_action(action_id=action_id, resolution=resolution)
    return {"ok": True, "action_id": action_id, "status": req.status}


class CancelQueryRequest(BaseModel):
    reason: str | None = None


@app.post("/v1/queries/{query_id}/cancel")
async def cancel_query(query_id: str, req: CancelQueryRequest | None = None) -> dict[str, Any]:
    ildb: ILDB = app.state.ildb
    kc: KernelContainer = app.state.kernel_container

    job_id = await ildb.get_job_id_for_query(query_id=query_id)
    if not job_id:
        raise HTTPException(status_code=404, detail="query not found")

    await kc.kernel.cancel(job_id, reason=req.reason if req else None)
    return {"ok": True, "query_id": query_id, "job_id": job_id}


@app.post("/v1/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str, req: CancelQueryRequest | None = None) -> dict[str, Any]:
    # Alias: workflow_id == query_id (constitution alignment)
    return await cancel_query(workflow_id, req)
