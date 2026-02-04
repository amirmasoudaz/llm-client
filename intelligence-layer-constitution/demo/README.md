# Demo: Funding outreach email review (simple reference implementation)

This folder contains a **minimal, readable** implementation of one flow:

- `Funding.Outreach.Email.Review`

It is intentionally *not production-ready* — it exists to make the schema/template system tangible.

## What this demo shows

1) API request → canonical **Intent** (validated by JSON schema)
2) Intent → **Plan template** selection (via manifests)
3) Plan execution:
   - run `Platform.Context.Load`
   - run `EnsureEmailPresent` policy check (may emit `action_required`)
   - run `Email.ReviewDraft` and produce an `Email.Review` outcome
4) Emit typed **SSE events** (`progress`, `action_required`, `final`, `error`)
5) (Optional) Resolve an `action_required` via `Workflow.Gate.Resolve` and resume

## Files

- `demo/kernel_email_review.py`: core “kernel + executor + stub operators” (no FastAPI dependency)
- `demo/run_email_review_demo.py`: CLI runner that prints events to stdout
- `demo/fastapi_email_review_app.py`: FastAPI wrapper with SSE endpoints (requires installing FastAPI)

## Quick run (no FastAPI)

```bash
python demo/run_email_review_demo.py
```

## AI Switchboard + Chat demo (no HTTP)

This demo simulates the v1 “user sends a query, switchboard chooses intent, kernel runs plan, tokens stream, action_required pauses”.

Default mode is **offline** (mock LLM):

```bash
python demo/run_ai_outreach_chat.py
```

To use the real OpenAI provider via `llm-client` (requires deps + network + API key), run:

```bash
llm-client/.venv/bin/python demo/run_ai_outreach_chat.py --real --model gpt-5-nano
```

## Run with FastAPI (optional)

Install dependencies in your real build environment:

```bash
pip install fastapi uvicorn
```

Run:

```bash
uvicorn demo.fastapi_email_review_app:app --reload
```

### Try it with curl

1) Submit a workflow:

```bash
curl -s -X POST http://localhost:8000/v1/demo/funding/email-review \
  -H 'content-type: application/json' \
  -d '{"thread_id":4512,"student_id":88,"funding_request_id":556,"platform_email_subject":null,"platform_email_body":null}'
```

2) Stream events (SSE):

```bash
curl -N http://localhost:8000/v1/queries/<QUERY_ID>/events
```

3) If you receive an `action_required`, resolve it:

```bash
curl -s -X POST http://localhost:8000/v1/actions/<ACTION_ID>/resolve \
  -H 'content-type: application/json' \
  -d '{"status":"accepted","payload":{"email_subject_override":"Quick question","email_text_override":"Dear Prof...\\n\\n...\\n\\nBest,\\nStudent"}}'
```
