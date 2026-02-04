"""
FastAPI wrapper for the demo kernel.

This file intentionally depends on FastAPI, which is not installed in this repo by default.
See demo/README.md for run instructions.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "FastAPI is required to run this demo app.\n"
        "Install: pip install fastapi uvicorn\n"
        f"Import error: {e}"
    )

from demo.kernel_email_review import DemoKernel


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
kernel = DemoKernel(repo_root=repo_root)

app = FastAPI(title="CanApply IL Demo: Email Review")


@app.post("/v1/demo/funding/email-review")
async def submit_email_review(body: Dict[str, Any], background: BackgroundTasks):
    """
    Submit a Funding.Outreach.Email.Review workflow.

    Body example:
    {
      "thread_id": 4512,
      "student_id": 88,
      "funding_request_id": 556,
      "platform_email_subject": "Hello",
      "platform_email_body": "Dear Prof ...",
      "email_subject_override": null,
      "email_text_override": null,
      "custom_instructions": "focus on clarity"
    }
    """

    def req_int(name: str) -> int:
        if name not in body:
            raise HTTPException(status_code=400, detail=f"Missing field: {name}")
        return int(body[name])

    thread_id = req_int("thread_id")
    student_id = req_int("student_id")
    funding_request_id = req_int("funding_request_id")

    run = kernel.submit_email_review(
        thread_id=thread_id,
        student_id=student_id,
        funding_request_id=funding_request_id,
        platform_email_subject=body.get("platform_email_subject"),
        platform_email_body=body.get("platform_email_body"),
        email_subject_override=body.get("email_subject_override"),
        email_text_override=body.get("email_text_override"),
        custom_instructions=body.get("custom_instructions"),
    )

    # Execute in background; SSE endpoint reads run.events as they are appended.
    background.add_task(kernel.run_email_review_to_completion, run)

    return {"query_id": run.query_id, "sse_url": f"/v1/queries/{run.query_id}/events"}


@app.get("/v1/queries/{query_id}/events")
async def query_events(query_id: str):
    run = kernel.workflow_by_query_id.get(query_id)
    if not run:
        raise HTTPException(status_code=404, detail="Unknown query_id")

    async def stream():
        idx = 0
        # Keep streaming until the workflow completes or fails.
        while True:
            while idx < len(run.events):
                event = run.events[idx]
                idx += 1
                yield _sse_format(event_type=event["event_type"], data=event)

            if run.status in ("completed", "failed"):
                # Ensure we flush any last events.
                if idx >= len(run.events):
                    break

            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/v1/actions/{action_id}/resolve")
async def resolve_action(action_id: str, body: Dict[str, Any], background: BackgroundTasks):
    """
    Resolve an action_required emitted by a workflow.

    Body example:
    {
      "status": "accepted",
      "payload": {
        "email_subject_override": "…",
        "email_text_override": "…"
      }
    }
    """
    status = body.get("status")
    if status not in ("accepted", "declined"):
        raise HTTPException(status_code=400, detail="status must be 'accepted' or 'declined'")
    payload = body.get("payload") or {}

    def do_resume():
        kernel.resolve_action_and_resume(action_id=action_id, accepted=(status == "accepted"), payload=payload)

    background.add_task(do_resume)
    return {"ok": True}


def _sse_format(*, event_type: str, data: Dict[str, Any]) -> str:
    # Minimal SSE: include `event:` and `data:` lines.
    # `data:` must be one line; json.dumps produces a single line by default.
    return f"event: {event_type}\n" f"data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"

