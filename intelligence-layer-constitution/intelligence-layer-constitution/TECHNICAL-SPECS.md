# CanApply Intelligence Layer (v1)

## Tech Stack, Services, Practices, and Engines

### Goals

The Intelligence Layer is a production-grade runtime for agentic workflows (chat, tools, retrieval, document processing) with:

* low latency for interactive requests
* reliable async execution for long-running tasks
* strict observability and reproducibility
* strong data lineage and idempotent side effects

Non-goals for v1:

* complex microservice RPC mesh
* Kafka-based internal messaging
* “eventual consistency everywhere” without a ledger

---

## 1) Tech Stack Overview

### Language + Runtime

* **Python 3.12**
* Async-first code style for IO-heavy operators
* Typing: `pyright` + Pydantic models

### API Layer

* **FastAPI** (HTTP + SSE)
* Server:

  * Dev: `uvicorn`
  * Prod: **gunicorn + uvicorn workers** (tuned for SSE)
* Reverse proxy: **Caddy**

### Caching

* **Redis (Hot Cache)**

  * ultra-fast response reuse
  * rate-limit counters and dedupe keys
  * transient state for workflows (optional)
* **PostgreSQL (Cold Cache)**

  * durable cached responses and metadata
  * cache analytics (hit rates, costs)
  * prompt/model versioned cache entries

### Vector Database

* **Qdrant**

  * semantic search, similarity lookup
  * embeddings store + retrieval filters
  * collection versioning for reproducibility

### Object Storage

* Local/dev: **MinIO (S3-compatible)**
* Prod: **AWS S3**
* Used for:

  * uploaded documents (PDF/DOCX/etc.)
  * derived artifacts (chunks, embeddings payloads, OCR outputs)
  * workflow snapshots and outputs (when large)

### Relational Databases

* **Primary transactional DB**: (you likely have MariaDB/MySQL already for CanApply core)
* **Intelligence layer DBs (v1)**:

  * PostgreSQL mandatory for cold cache + potentially workflow ledger
  * If you keep MySQL/MariaDB for the main app, that’s fine. Just avoid “two sources of truth” for the same entity.

### Background Execution

* **DB-leased Worker Executor** (your chosen model)

  * a worker polls/leasing steps from DB
  * executes operators
  * streams events back (SSE or event store)
* Optional queue (v1): Redis Streams or simple Redis queue for “kick the worker now”

  * not required if polling is adequate

### LLM Providers (Handled by LLM Client Module)

* OpenAI (primary), fallback to Anthropic
* LLM access through your **LLM Client module** (centralized gateway: retries, caching, telemetry, normalization)

### Observability (Handled by Observability Module)

* Logs: **OpenTelemetry** + **Grafana Loki**
* Metrics: **OpenTelemetry** + **Prometheus** + **Grafana**
* Tracing: **OpenTelemetry** + **Grafana Tempo**
* Error reporting: **Sentry** (high ROI)

### Infra + Deployment

* Containers: **Docker** + **Docker Compose** (v1)
* Orchestration: later **Kubernetes** if needed (v1.1)
* CI/CD: **GitHub Actions** or **GitLab CI**
* Secrets: **SSM/Secrets Manager** (prod), `.env` only for dev

---

## 2) Service Map (v1)

### 2.1 AI Gateway API (FastAPI)

**Responsibilities**

* Accepts chat/tool/workflow requests
* AuthN/AuthZ (token, service keys)
* Starts a workflow run
* Returns immediate response or SSE stream
* Exposes admin/debug endpoints (careful)

**Endpoints**

* `POST /chat` (interactive, likely SSE)
* `POST /workflows/run` (async job trigger)
* `GET /workflows/{id}/events` (SSE)
* `POST /tools/{name}` (direct tool call, optional)

**Key properties**

* No heavy work inside request thread
* Always writes intent to the workflow ledger first

---

### 2.2 Orchestrator Runtime

This is not a separate service necessarily; it can be a library used by both API and workers.

**Responsibilities**

* Builds a plan: steps, operators, tool calls, retrieval calls
* Manages state machine: step dependencies, retries, timeouts
* Writes all state transitions to the ledger

**Implementation notes**

* Pydantic schemas for all internal messages
* Strict versioning of operator contracts

---

### 2.3 Worker Pool(s)

At minimum:

* **worker-default**: general operators (LLM calls, light retrieval, formatting)
* optional later:

  * **worker-doc**: PDF/DOCX extraction, chunking
  * **worker-embed**: embeddings generation + Qdrant indexing
  * **worker-io**: external integrations (email, CRM, etc.)

**Responsibilities**

* Lease runnable steps (DB row lock + lease expiry)
* Execute operator
* Persist outputs (DB or S3)
* Emit events for SSE updates

---

### 2.4 Document Service

Can be a module or separate worker responsibilities.

**Responsibilities**

* Document ingestion:

  * upload -> S3/MinIO
  * parsing -> text
  * chunking -> chunks
  * optional OCR
* Stores derived artifacts in object storage
* Produces chunk manifests + embeddings jobs

---

### 2.5 Retrieval Service (Qdrant Integration)

Again, can be an internal module used by workers.

**Responsibilities**

* Embed queries
* Query Qdrant with filters (tenant, source, doc type, time)
* Return topK chunks + metadata
* Record retrieval snapshot for reproducibility

---

### 2.6 Cache Service (Your LLM Client)

This is a cornerstone module.

**Responsibilities**

* Request normalization
* Key derivation and versioning
* Redis hot cache lookups/writes
* Postgres cold cache lookups/writes
* Cost tracking: token usage, provider latency
* Policy-based caching: TTLs, bypass, safety constraints

---

## 3) Engines and Core Components

### 3.1 Workflow Engine (DB-leased)

**Engine structure**

* `workflow_run` (high-level run)
* `workflow_step` (atomic step with input/output refs)
* `workflow_event` (append-only events for streaming)
* Leasing mechanism:

  * `leased_by`, `lease_expires_at`
  * worker renews lease periodically
  * stale leases get reclaimed

**Why it’s good**

* deterministic recovery after crashes
* easy to debug (ledger-based)
* no “where did the job go” mystery

---

### 3.2 Operator Engine

Operators are “pure-ish” functions executed by workers.

**Operator types**

* LLM call operator
* Retrieval operator
* Doc parse operator
* Embed/index operator
* Tool operator (external side effects)

**Contracts**

* Inputs: JSON schema / Pydantic
* Outputs: JSON schema / Pydantic
* Side effects must be idempotent (keyed by step_id)

---

### 3.3 Event Engine (SSE)

**Event categories**

* `workflow_started`
* `step_started`, `step_progress`, `step_completed`, `step_failed`
* `tokens_stream` (optional)
* `final_result`

**Storage**

* Append-only table or log
* SSE endpoint reads from event store (poll or tail)

---

## 4) Engineering Practices

### 4.1 Reproducibility and Lineage (non-negotiable)

Everything important gets versioned and recorded:

* prompt version, tool schema version
* model + parameters
* retrieval snapshot id (query embed hash + collection version + filters + topK)
* document version/chunking version

This prevents “it worked yesterday” ghost hunts.

---

### 4.2 Idempotency Everywhere

* Every step has a deterministic idempotency key:

  * typically `(workflow_step_id)`
* External calls and writes must be safe to retry
* Side-effect operators must “check first” before creating duplicates

---

### 4.3 Caching Rules

* cache key must include:

  * model + temperature + system prompt version + tool schema version
  * normalized user input
  * retrieval snapshot id (if retrieval used)
* bypass mode for debugging
* TTL policy:

  * short TTL for unstable answers (news, web)
  * longer TTL for stable transformations (summaries, formatting)

---

### 4.4 Multi-tenancy and Security

* tenant_id always included in:

  * DB rows
  * S3 object keys prefixes
  * Qdrant payload filters
* encryption:

  * TLS in transit
  * S3 server-side encryption in prod
* secrets never in code or logs

---

### 4.5 Observability Standards

Every request gets:

* `request_id`, `workflow_id`, `step_id`, `tenant_id`
* structured logs
* metrics: latency, cache hit rate, token usage
* tracing: spans across API -> worker -> provider

---

### 4.6 Testing Strategy

* unit tests for operators and cache key normalization
* integration tests for:

  * DB leasing correctness
  * retry/idempotency behavior
  * Qdrant retrieval correctness
* load test SSE endpoint and worker concurrency

---

## 5) Recommended Repo Structure (practical)

* `apps/api` FastAPI app
* `apps/worker` worker entrypoints + pool configs
* `packages/llm_client` (your module)
* `packages/orchestrator` planning + workflow engine
* `packages/retrieval` qdrant + embeddings
* `packages/documents` parsing + chunking
* `packages/common` schemas, logging, tracing

Single repo, multi-package. Microservices later only if forced.

---

## 6) v1 Deployment Topology

Minimal, ship-fast:

* `api` (FastAPI + gunicorn/uvicorn)
* `worker-default` (same image, different command)
* `redis`
* `postgres`
* `qdrant`
* `minio` (dev/stage), `aws s3` (prod external)
* `prometheus + grafana` (stage/prod)
* `otel-collector` (optional but recommended)

---

## 7) What’s explicitly deferred (v2+)

* Kafka/event bus for large-scale distributed workloads
* Temporal (or similar) as a workflow engine replacement
* K8s multi-region, autoscaling pools
* advanced policy engine for cost routing across providers
* automated evaluation harness and prompt regression suite (worth doing later)
