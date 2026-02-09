#!/usr/bin/env python3
"""
Contract test runner for the Intelligence Layer API.

This script uses only the Python standard library so it can run anywhere.

Run:
    python test_intelligence_layer.py

Env:
    API_BASE_URL (default http://127.0.0.1:8080)
"""

import json
import os
import time
import sys
import threading
import uuid as uuidlib
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")


class ContractTestFailure(RuntimeError):
    pass


def _fail(msg: str) -> "None":
    raise ContractTestFailure(msg)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        _fail(msg)


def _request(method: str, path: str, data: dict | None = None, headers: dict | None = None) -> tuple[int, str]:
    url = f"{BASE_URL}{path}"
    h = {"Content-Type": "application/json", **(headers or {})}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = Request(url, data=body, headers=h, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8")
    except HTTPError as e:
        return e.code, e.read().decode("utf-8") if e.fp else str(e.reason)
    except URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        raise


def _get(path: str) -> tuple[int, str]:
    return _request("GET", path)


def _post(path: str, data: dict) -> tuple[int, str]:
    return _request("POST", path, data=data)


def _parse_json(body: str):
    """Parse JSON response; if the API returns a JSON-encoded string, parse again."""
    out = json.loads(body)
    if isinstance(out, str):
        out = json.loads(out)
    return out


# Event types that are collapsed when consecutive: print first, "... N more <type> events ...", last.
_COLLAPSE_EVENT_TYPES = frozenset({"model_token"})


def _flush_sse_run(
    run: list[tuple[str, list[str]]],
    *,
    quiet: bool,
) -> None:
    """Print a run of same-type events, collapsing long runs for _COLLAPSE_EVENT_TYPES."""
    if quiet or not run:
        return
    event_type = run[0][0]
    if event_type in _COLLAPSE_EVENT_TYPES and len(run) > 2:
        # First event full
        for ln in run[0][1]:
            print(ln)
        print(f"... {len(run) - 2} more {event_type} events ...")
        # Last event full
        for ln in run[-1][1]:
            print(ln)
    else:
        for _t, lines in run:
            for ln in lines:
                print(ln)


def stream_sse_events(
    path: str,
    *,
    timeout_sec: float = 180.0,
    max_events: int | None = None,
    quiet: bool = False,
) -> list[tuple[str, dict]]:
    """Stream SSE events from path and return parsed events.

    Stops on terminal event or max_events. Raises if no terminal event observed.
    When not quiet, long runs of the same event type (e.g. model_token) are
    printed as: first event, "... N more <type> events ...", last event.
    """
    url = f"{BASE_URL}{path}"
    req = Request(url)
    req.add_header("Accept", "text/event-stream")
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            count = 0
            last_event: str | None = None
            last_event_line: str = ""
            events: list[tuple[str, dict]] = []
            run: list[tuple[str, list[str]]] = []  # (event_type, [line1, line2])
            start = time.time()
            for line in resp:
                if (time.time() - start) > timeout_sec:
                    break
                line = line.decode("utf-8").rstrip()
                if line.startswith("event:"):
                    last_event = line.split(":", 1)[1].strip()
                    last_event_line = line
                if line.startswith("data:"):
                    count += 1
                    raw = line.split(":", 1)[1].strip()
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        payload = {"_raw": raw}
                    event_type = last_event or "message"
                    event_lines = [last_event_line, line]

                    # Flush run if type changed
                    if run and run[0][0] != event_type:
                        _flush_sse_run(run, quiet=quiet)
                        run = []
                    run.append((event_type, event_lines))

                    events.append((event_type, payload))
                    if event_type in {"final_result", "final_error", "job_cancelled"}:
                        _flush_sse_run(run, quiet=quiet)
                        return events
                    if max_events is not None and count >= max_events:
                        _flush_sse_run(run, quiet=quiet)
                        run = []
                        break
            _flush_sse_run(run, quiet=quiet)
    except HTTPError as e:
        print(f"SSE error: {e.code} {e.read().decode('utf-8')}", file=sys.stderr)
        return []
    except URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        return []
    _fail(f"SSE stream did not reach terminal event within {timeout_sec}s: {path}")


def _start_cancel_thread(query_id: str, delay_sec: float, reason: str = "contract_test") -> threading.Thread:
    def _run() -> None:
        time.sleep(delay_sec)
        _post(f"/v1/queries/{query_id}/cancel", {"reason": reason})

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def main() -> None:
    student_id = 1583
    funding_request_id = 88
    funding_request_id_2 = int(os.getenv("FUNDING_REQUEST_ID_2", "5995"))
    timeout_sec = 180

    print("=== CONTRACT SUITE: Intelligence Layer API ===")
    print(f"BASE_URL={BASE_URL}")
    print(
        f"student_id={student_id} funding_request_id={funding_request_id} funding_request_id_2={funding_request_id_2} timeout_sec={timeout_sec}"
    )

    # ------------------------------------------------------------------
    # Contract 1: init is idempotent (same ids -> same thread)
    # ------------------------------------------------------------------
    print(f"\n=== 1) POST /v1/threads/init (funding_request_id={funding_request_id}) ===")
    status, body = _post("/v1/threads/init", {"funding_request_id": funding_request_id})
    # body is raw string; may be double JSON-encoded by the API
    try:
        init_resp = _parse_json(body)
    except json.JSONDecodeError:
        init_resp = {}
    print(f"Status: {status}\n{json.dumps(init_resp, indent=2)}")
    thread_id = init_resp.get("thread_id")
    _assert(status in {200, 201}, f"init expected 200/201, got {status}: {body[:200]}")
    _assert(thread_id is not None, f"init missing thread_id: {init_resp}")
    _assert(isinstance(init_resp.get("is_new"), bool), f"init missing is_new bool: {init_resp}")

    status2, body2 = _post("/v1/threads/init", {"funding_request_id": funding_request_id})
    init_resp2 = _parse_json(body2) if body2 else {}
    _assert(status2 in {200, 201}, f"second init expected 200/201, got {status2}: {body2[:200]}")
    _assert(init_resp2.get("thread_id") == thread_id, f"init not idempotent: {thread_id} vs {init_resp2.get('thread_id')}")
    print("PASS: init idempotency")

    print(f"\nThread ID: {thread_id}")

    # Create a second thread for cross-thread idempotency tests.
    print(f"\n=== 1b) POST /v1/threads/init (funding_request_id={funding_request_id_2}) ===")
    status_b, body_b = _post("/v1/threads/init", {"funding_request_id": funding_request_id_2})
    init_b = _parse_json(body_b)
    thread_id_2 = init_b.get("thread_id")
    _assert(status_b in {200, 201}, f"init2 expected 200/201, got {status_b}: {body_b[:200]}")
    _assert(thread_id_2 is not None, f"init2 missing thread_id: {init_b}")
    _assert(thread_id_2 != thread_id, "second thread init unexpectedly returned the same thread_id")
    print(f"PASS: second thread created (thread_id_2={thread_id_2})")

    # ------------------------------------------------------------------
    # Contract 2: platform context read works (debug endpoint)
    # ------------------------------------------------------------------
    print(f"\n=== 2) GET /v1/debug/platform/funding_request/{funding_request_id}/context ===")
    status, body = _get(f"/v1/debug/platform/funding_request/{funding_request_id}/context")
    _assert(status == 200, f"platform context expected 200, got {status}: {body[:200]}")
    platform_ctx = _parse_json(body)
    _assert(isinstance(platform_ctx, dict), "platform context must be a JSON object")
    _assert(int(platform_ctx.get("request_id", -1)) == funding_request_id, "platform context request_id mismatch")
    _assert(int(platform_ctx.get("user_id", -1)) == student_id, "platform context user_id mismatch")
    print("PASS: platform context read")

    # ------------------------------------------------------------------
    # Contract 2b: operator execution (Thread.CreateOrLoad) + idempotency
    # ------------------------------------------------------------------
    print("\n=== 2b) POST /v1/debug/operators/thread_create_or_load ===")
    op_id_key = f"contract-thread-init-{uuidlib.uuid4()}"
    status, body = _post(
        "/v1/debug/operators/thread_create_or_load",
        {
            "student_id": student_id,
            "funding_request_id": funding_request_id,
            "idempotency_key": op_id_key,
        },
    )
    _assert(status == 200, f"operator debug expected 200, got {status}: {body[:200]}")
    op_resp = _parse_json(body)
    _assert(op_resp.get("status") == "succeeded", f"operator status not succeeded: {op_resp}")
    result = op_resp.get("result") or {}
    _assert(result.get("thread_id") is not None, f"operator result missing thread_id: {op_resp}")

    status2, body2 = _post(
        "/v1/debug/operators/thread_create_or_load",
        {
            "student_id": student_id,
            "funding_request_id": funding_request_id,
            "idempotency_key": op_id_key,
        },
    )
    _assert(status2 == 200, f"operator debug repeat expected 200, got {status2}: {body2[:200]}")
    op_resp2 = _parse_json(body2)
    _assert(op_resp2.get("status") == "succeeded", f"operator repeat status not succeeded: {op_resp2}")
    result2 = op_resp2.get("result") or {}
    _assert(result2.get("thread_id") == result.get("thread_id"), "operator idempotency thread_id mismatch")
    print("PASS: operator execution + idempotency")

    # ------------------------------------------------------------------
    # Contract 3: query submit returns query_id and SSE reaches terminal
    # ------------------------------------------------------------------
    query_message = (
        "Summarize the professor + my drafted email + status from platform context "
        "(use platform_load_funding_thread_context if needed). Then suggest next actions."
    )
    print(f"\n=== 3) POST /v1/threads/{thread_id}/queries ===")
    status, body = _post(f"/v1/threads/{thread_id}/queries", {"message": query_message, "attachments": []})
    _assert(status == 200, f"submit_query expected 200, got {status}: {body[:200]}")
    payload = _parse_json(body)
    query_id = payload.get("query_id")
    _assert(isinstance(query_id, str) and query_id, f"submit_query missing query_id: {payload}")
    events_path = payload.get("sse_url") or f"/v1/queries/{query_id}/events"
    if not events_path.startswith("/"):
        events_path = "/" + events_path
    print(f"PASS: submit_query (query_id={query_id})")

    print(f"\n=== 4) SSE stream ({events_path}) ===")
    events = stream_sse_events(events_path, timeout_sec=timeout_sec, max_events=None, quiet=False)
    types = [t for (t, _d) in events]
    _assert(types.count("final_result") + types.count("final_error") + types.count("job_cancelled") == 1, "expected exactly one terminal SSE event")
    _assert("job_started" in types, "missing job_started event")
    _assert(any(t == "model_token" for t in types), "expected at least one model_token event")
    terminal = [e for e in events if e[0] in {"final_result", "final_error", "job_cancelled"}][0]
    if terminal[0] != "final_result":
        _fail(f"query did not succeed; terminal={terminal[0]} payload={terminal[1]}")
    print("PASS: SSE reaches terminal and returns final_result")

    # ------------------------------------------------------------------
    # Contract 3b: ledger-backed workflow SSE stream
    # ------------------------------------------------------------------
    print(f"\n=== 4b) Ledger SSE stream (/v1/workflows/{query_id}/events) ===")
    ledger_events = stream_sse_events(f"/v1/workflows/{query_id}/events", timeout_sec=timeout_sec, max_events=None, quiet=False)
    ledger_types = [t for (t, _d) in ledger_events]
    _assert(
        ledger_types.count("final_result") + ledger_types.count("final_error") + ledger_types.count("job_cancelled") == 1,
        "ledger SSE expected exactly one terminal event",
    )
    _assert("job_started" in ledger_types, "ledger SSE missing job_started event")
    _assert(any(t == "model_token" for t in ledger_types), "ledger SSE expected at least one model_token event")
    terminal_ledger = [e for e in ledger_events if e[0] in {"final_result", "final_error", "job_cancelled"}][0]
    if terminal_ledger[0] != "final_result":
        _fail(f"ledger SSE did not succeed; terminal={terminal_ledger[0]} payload={terminal_ledger[1]}")
    print("PASS: ledger SSE reaches terminal and returns final_result")


    # ------------------------------------------------------------------
    # Contract 4: query_id idempotency and cross-thread conflict
    # ------------------------------------------------------------------
    print("\n=== 5) Query idempotency with client-supplied query_id ===")
    fixed_query_id = str(uuidlib.uuid4())
    status, body = _post(
        f"/v1/threads/{thread_id}/queries",
        {
            "query_id": fixed_query_id,
            "message": "contract test: idempotency (fast)",
            "attachments": [],
            "operator_id": "contract.stream_slow",
            "metadata": {"text": "ok", "delay_ms": 0},
        },
    )
    _assert(status == 200, f"submit_query (fixed query_id) expected 200, got {status}: {body[:200]}")
    r1 = _parse_json(body)
    _assert(r1.get("query_id") == fixed_query_id, f"server did not echo query_id: expected {fixed_query_id}, got {r1}")

    status, body = _post(
        f"/v1/threads/{thread_id}/queries",
        {
            "query_id": fixed_query_id,
            "message": "contract test: idempotency repeat",
            "attachments": [],
        },
    )
    _assert(status == 200, f"repeat submit_query (fixed query_id) expected 200, got {status}: {body[:200]}")
    r2 = _parse_json(body)
    _assert(r2.get("query_id") == fixed_query_id, f"repeat submit_query changed query_id: {r2}")

    # Cross-thread: same query_id should 409 conflict.
    status, body = _post(
        f"/v1/threads/{thread_id_2}/queries",
        {
            "query_id": fixed_query_id,
            "message": "contract test: cross-thread conflict",
            "attachments": [],
        },
    )
    _assert(status == 409, f"cross-thread query_id reuse expected 409, got {status}: {body[:200]}")
    print("PASS: query_id idempotency + cross-thread conflict")

    # ------------------------------------------------------------------
    # Contract 5: cancellation works and produces job_cancelled terminal
    # ------------------------------------------------------------------
    print("\n=== 6) Cancel during streaming (contract.stream_slow) ===")
    status, body = _post(
        f"/v1/threads/{thread_id}/queries",
        {
            "message": "contract test: cancel during streaming",
            "attachments": [],
            "operator_id": "contract.stream_slow",
            "metadata": {"text": "x" * 500, "delay_ms": 10},
        },
    )
    _assert(status == 200, f"submit_query (cancel streaming) expected 200, got {status}: {body[:200]}")
    payload = _parse_json(body)
    qid = payload.get("query_id")
    _assert(isinstance(qid, str) and qid, f"missing query_id: {payload}")
    _start_cancel_thread(qid, delay_sec=0.25)
    events = stream_sse_events(payload.get("sse_url") or f"/v1/queries/{qid}/events", timeout_sec=timeout_sec, quiet=True)
    types = [t for (t, _d) in events]
    _assert("job_cancelled" in types, f"expected job_cancelled terminal event, got types={types}")
    _assert("final_result" not in types, f"cancelled job unexpectedly produced final_result: types={types}")
    print("PASS: cancel during streaming")

    # ------------------------------------------------------------------
    # Contract 6: cancellation during tool-like operation (contract.tool_sleep)
    # ------------------------------------------------------------------
    print("\n=== 7) Cancel during tool (contract.tool_sleep) ===")
    status, body = _post(
        f"/v1/threads/{thread_id}/queries",
        {
            "message": "contract test: cancel during tool",
            "attachments": [],
            "operator_id": "contract.tool_sleep",
            "metadata": {"seconds": 10, "tool_name": "debug_sleep"},
        },
    )
    _assert(status == 200, f"submit_query (cancel tool) expected 200, got {status}: {body[:200]}")
    payload = _parse_json(body)
    qid = payload.get("query_id")
    _assert(isinstance(qid, str) and qid, f"missing query_id: {payload}")
    _start_cancel_thread(qid, delay_sec=0.25)
    events = stream_sse_events(payload.get("sse_url") or f"/v1/queries/{qid}/events", timeout_sec=timeout_sec, quiet=True)
    types = [t for (t, _d) in events]
    _assert("tool_start" in types or "tool_end" in types, f"expected tool_start/tool_end events, got types={types}")
    _assert("job_cancelled" in types, f"expected job_cancelled terminal event, got types={types}")
    print("PASS: cancel during tool")

    # ------------------------------------------------------------------
    # Contract 7: cancellation during retry/backoff (contract.retry_backoff)
    # ------------------------------------------------------------------
    print("\n=== 8) Cancel during retry/backoff (contract.retry_backoff) ===")
    status, body = _post(
        f"/v1/threads/{thread_id}/queries",
        {
            "message": "contract test: cancel during retry",
            "attachments": [],
            "operator_id": "contract.retry_backoff",
            "metadata": {"attempts": 6, "base_backoff_ms": 800},
        },
    )
    _assert(status == 200, f"submit_query (cancel retry) expected 200, got {status}: {body[:200]}")
    payload = _parse_json(body)
    qid = payload.get("query_id")
    _assert(isinstance(qid, str) and qid, f"missing query_id: {payload}")
    _start_cancel_thread(qid, delay_sec=0.25)
    events = stream_sse_events(payload.get("sse_url") or f"/v1/queries/{qid}/events", timeout_sec=timeout_sec, quiet=True)
    types = [t for (t, _d) in events]
    _assert("progress" in types, f"expected progress events, got types={types}")
    _assert("job_cancelled" in types, f"expected job_cancelled terminal event, got types={types}")
    print("PASS: cancel during retry/backoff")

    # ------------------------------------------------------------------
    # Contract 8: policy denial produces final_error without streaming tokens
    # ------------------------------------------------------------------
    print("\n=== 9) Policy deny produces final_error (policy_ref=deny_all) ===")
    status, body = _post(
        f"/v1/threads/{thread_id}/queries",
        {
            "message": "contract test: policy deny",
            "attachments": [],
            "policy_ref": {"name": "deny_all"},
        },
    )
    _assert(status == 200, f"submit_query (policy deny) expected 200, got {status}: {body[:200]}")
    payload = _parse_json(body)
    qid = payload.get("query_id")
    events = stream_sse_events(payload.get("sse_url") or f"/v1/queries/{qid}/events", timeout_sec=timeout_sec, quiet=True)
    types = [t for (t, _d) in events]
    _assert("final_error" in types, f"expected final_error, got types={types}")
    _assert("model_token" not in types, f"policy-denied job unexpectedly streamed tokens: types={types}")
    print("PASS: policy deny -> final_error (no tokens)")

    # ------------------------------------------------------------------
    # Contract 9: budget denial produces final_error without streaming tokens
    # ------------------------------------------------------------------
    print("\n=== 10) Budget deny produces final_error (budgets.max_turns=0) ===")
    status, body = _post(
        f"/v1/threads/{thread_id}/queries",
        {
            "message": "contract test: budget deny",
            "attachments": [],
            "budgets": {"max_turns": 0},
            "max_turns": 1,
        },
    )
    _assert(status == 200, f"submit_query (budget deny) expected 200, got {status}: {body[:200]}")
    payload = _parse_json(body)
    qid = payload.get("query_id")
    events = stream_sse_events(payload.get("sse_url") or f"/v1/queries/{qid}/events", timeout_sec=timeout_sec, quiet=True)
    types = [t for (t, _d) in events]
    _assert("final_error" in types, f"expected final_error, got types={types}")
    _assert("model_token" not in types, f"budget-denied job unexpectedly streamed tokens: types={types}")
    print("PASS: budget deny -> final_error (no tokens)")

    print("\nALL CONTRACTS PASSED.")


if __name__ == "__main__":
    main()
