# llm-client Secure Deployment Defaults

This note describes the default production posture recommended for projects
using `llm_client`.

## Recommended Defaults

- keep raw provider payload capture disabled by default
- enable structured logging redaction
- treat tool output as untrusted and apply truncation plus secret/PII redaction
- prefer engine-backed execution over direct provider calls
- enable request timeouts, retries, and failover intentionally rather than
  leaving vendor SDK defaults implicit
- isolate cache keys by tenant and request shape
- do not persist ledger/billing/business state inside `llm_client`

## Logging and Observability

Recommended:

- use `RedactionPolicy` defaults or stricter
- log metadata, diagnostics, and safe previews rather than raw prompts or raw
  SDK payloads
- keep provider payload capture at `off` or `metadata_only`
- avoid logging tool arguments or outputs verbatim unless there is a clear
  incident/debugging need

Avoid:

- printing raw `raw_response` or `raw_request`
- enabling unrestricted prompt logging in sensitive workloads
- treating logs as a safe place for secrets just because they are internal

## Tool Execution

Recommended:

- validate tool arguments
- use explicit tool execution modes
- set timeouts and concurrency limits
- apply `ToolOutputPolicy` before logging or reinserting tool output into model
  context
- keep product-specific allowlists and authorization checks above the package

## Cache and Persistence

Recommended:

- use versioned cache keys
- keep cache invalidation policy explicit
- cache summaries and metadata only where the scope and isolation boundary are
  clear
- keep storage-backed ledgers, billing, and tenant lifecycle persistence
  outside the package

Avoid:

- caching raw provider payloads by default
- sharing one cache namespace across unrelated tenants or scopes
- assuming `rewrite` and `regenerate` have the same semantics

## Runtime Safety

Recommended:

- run all higher-level workflows through `ExecutionEngine`
- keep direct provider use for low-level or controlled fallback paths only
- propagate request context and cancellation tokens
- treat memory, retrieval, and external documents as untrusted input

## Deployment Checklist

- redaction enabled
- provider payload capture reviewed
- tool-output hardening enabled where tools can surface user/private data
- timeouts configured
- retry policy configured
- failover policy configured
- cache isolation reviewed
- secret injection handled outside code and logs
