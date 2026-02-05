from __future__ import annotations

from typing import Iterable


DDL_STATEMENTS: list[str] = [
    "CREATE EXTENSION IF NOT EXISTS pgcrypto;",
    "CREATE SCHEMA IF NOT EXISTS runtime;",
    "CREATE SCHEMA IF NOT EXISTS ledger;",
    "CREATE SCHEMA IF NOT EXISTS cache;",
    "CREATE SCHEMA IF NOT EXISTS registry;",
    # runtime.threads (scope binding)
    """
    CREATE TABLE IF NOT EXISTS runtime.threads (
      tenant_id BIGINT NOT NULL,
      thread_id BIGINT GENERATED ALWAYS AS IDENTITY,
      student_id BIGINT NOT NULL,
      funding_request_id BIGINT NOT NULL,
      status TEXT NOT NULL DEFAULT 'active',
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      PRIMARY KEY (tenant_id, thread_id),
      CONSTRAINT threads_scope_uq UNIQUE (tenant_id, student_id, funding_request_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS threads_student_created
      ON runtime.threads (tenant_id, student_id, created_at DESC);
    """,
    # runtime.workflow_runs
    """
    CREATE TABLE IF NOT EXISTS runtime.workflow_runs (
      tenant_id              BIGINT       NOT NULL,
      workflow_id            UUID         NOT NULL,
      correlation_id         UUID         NOT NULL,

      thread_id              BIGINT       NULL,
      scope_type             TEXT         NULL,
      scope_id               TEXT         NULL,

      intent_id              UUID         NOT NULL,
      plan_id                UUID         NULL,

      capability_name        TEXT         NULL,
      capability_version     TEXT         NULL,

      status                 TEXT         NOT NULL,
      execution_mode         TEXT         NOT NULL,
      replay_mode            TEXT         NOT NULL DEFAULT 'reproduce',

      request_key            BYTEA        NULL,
      parent_workflow_id     UUID         NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
      started_at             TIMESTAMPTZ  NULL,
      completed_at           TIMESTAMPTZ  NULL,
      updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, workflow_id),
      CONSTRAINT workflow_runs_thread_fk
        FOREIGN KEY (tenant_id, thread_id) REFERENCES runtime.threads(tenant_id, thread_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS workflow_runs_tenant_status_created
      ON runtime.workflow_runs (tenant_id, status, created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS workflow_runs_thread_created
      ON runtime.workflow_runs (tenant_id, thread_id, created_at DESC);
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS workflow_runs_request_key_uq
      ON runtime.workflow_runs (tenant_id, request_key)
      WHERE request_key IS NOT NULL;
    """,
    # runtime.workflow_steps
    """
    CREATE TABLE IF NOT EXISTS runtime.workflow_steps (
      tenant_id              BIGINT       NOT NULL,
      workflow_id            UUID         NOT NULL,
      step_id                TEXT         NOT NULL,

      kind                   TEXT         NOT NULL,
      name                   TEXT         NOT NULL,
      operator_name          TEXT         NULL,
      operator_version       TEXT         NULL,

      effects                TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
      policy_tags            TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
      risk_level             TEXT         NOT NULL,
      cache_policy           TEXT         NOT NULL,

      idempotency_key        TEXT         NULL,
      input_payload          JSONB        NOT NULL DEFAULT '{}'::jsonb,
      input_hash             BYTEA        NULL,

      status                 TEXT         NOT NULL,
      attempt_count          INT          NOT NULL DEFAULT 0,
      next_retry_at          TIMESTAMPTZ  NULL,

      lease_owner            TEXT         NULL,
      lease_expires_at       TIMESTAMPTZ  NULL,

      last_job_id            UUID         NULL,
      gate_id                UUID         NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
      started_at             TIMESTAMPTZ  NULL,
      finished_at            TIMESTAMPTZ  NULL,
      updated_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, workflow_id, step_id),
      CONSTRAINT workflow_steps_run_fk
        FOREIGN KEY (tenant_id, workflow_id) REFERENCES runtime.workflow_runs(tenant_id, workflow_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS workflow_steps_runnable
      ON runtime.workflow_steps (tenant_id, status, next_retry_at, lease_expires_at)
      WHERE status IN ('READY', 'FAILED_RETRYABLE');
    """,
    """
    CREATE INDEX IF NOT EXISTS workflow_steps_workflow
      ON runtime.workflow_steps (tenant_id, workflow_id);
    """,
    # ledger.intents
    """
    CREATE TABLE IF NOT EXISTS ledger.intents (
      tenant_id              BIGINT       NOT NULL,
      intent_id              UUID         NOT NULL,
      intent_type            TEXT         NOT NULL,
      schema_version         TEXT         NOT NULL,
      source                 TEXT         NOT NULL,

      thread_id              BIGINT       NULL,
      scope_type             TEXT         NULL,
      scope_id               TEXT         NULL,

      actor                  JSONB        NOT NULL,
      inputs                 JSONB        NOT NULL,
      constraints            JSONB        NOT NULL DEFAULT '{}'::jsonb,
      context_refs           JSONB        NOT NULL DEFAULT '{}'::jsonb,

      data_classes           TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],

      redacted_inputs        JSONB        NULL,

      correlation_id         UUID         NOT NULL,
      producer_kind          TEXT         NOT NULL,
      producer_name          TEXT         NOT NULL,
      producer_version       TEXT         NOT NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, intent_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS intents_thread_created
      ON ledger.intents (tenant_id, thread_id, created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS intents_type_created
      ON ledger.intents (tenant_id, intent_type, created_at DESC);
    """,
    # ledger.plans
    """
    CREATE TABLE IF NOT EXISTS ledger.plans (
      tenant_id              BIGINT       NOT NULL,
      plan_id                UUID         NOT NULL,
      intent_id              UUID         NOT NULL,

      schema_version         TEXT         NOT NULL,
      planner_name           TEXT         NOT NULL,
      planner_version        TEXT         NOT NULL,

      plan                   JSONB        NOT NULL,
      plan_hash              BYTEA        NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, plan_id),
      CONSTRAINT plans_intent_fk
        FOREIGN KEY (tenant_id, intent_id) REFERENCES ledger.intents(tenant_id, intent_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS plans_intent_created
      ON ledger.plans (tenant_id, intent_id, created_at DESC);
    """,
    # ledger.events
    """
    CREATE TABLE IF NOT EXISTS ledger.events (
      event_no               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

      tenant_id              BIGINT       NOT NULL,
      event_id               UUID         NOT NULL,
      schema_version         TEXT         NOT NULL DEFAULT '1.0',

      workflow_id            UUID         NOT NULL,
      thread_id              BIGINT       NULL,
      intent_id              UUID         NULL,
      plan_id                UUID         NULL,
      step_id                TEXT         NULL,
      job_id                 UUID         NULL,
      outcome_id             UUID         NULL,
      gate_id                UUID         NULL,
      policy_decision_id     UUID         NULL,

      event_type             TEXT         NOT NULL,
      severity               TEXT         NOT NULL DEFAULT 'info',
      actor                  JSONB        NOT NULL,

      payload                JSONB        NOT NULL DEFAULT '{}'::jsonb,
      payload_hash           BYTEA        NULL,

      correlation_id         UUID         NOT NULL,
      producer_kind          TEXT         NOT NULL,
      producer_name          TEXT         NOT NULL,
      producer_version       TEXT         NOT NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now()
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS events_event_id_uq
      ON ledger.events (tenant_id, event_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS events_workflow_ordered
      ON ledger.events (tenant_id, workflow_id, event_no);
    """,
    """
    CREATE INDEX IF NOT EXISTS events_thread_created
      ON ledger.events (tenant_id, thread_id, created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS events_correlation
      ON ledger.events (tenant_id, correlation_id);
    """,
    # ledger.outcomes
    """
    CREATE TABLE IF NOT EXISTS ledger.outcomes (
      tenant_id              BIGINT       NOT NULL,
      outcome_id             UUID         NOT NULL,
      lineage_id             UUID         NOT NULL,
      version                INT          NOT NULL,
      parent_outcome_id      UUID         NULL,

      outcome_type           TEXT         NOT NULL,
      schema_version         TEXT         NOT NULL,
      status                 TEXT         NOT NULL,
      visibility             TEXT         NOT NULL DEFAULT 'private',

      workflow_id            UUID         NULL,
      thread_id              BIGINT       NULL,
      intent_id              UUID         NULL,
      plan_id                UUID         NULL,
      step_id                TEXT         NULL,
      job_id                 UUID         NULL,

      content                JSONB        NULL,
      content_object_uri     TEXT         NULL,
      content_hash           BYTEA        NULL,

      confidence             DOUBLE PRECISION NULL,
      data_classes           TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],

      producer_kind          TEXT         NOT NULL,
      producer_name          TEXT         NOT NULL,
      producer_version       TEXT         NOT NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, outcome_id),
      CONSTRAINT outcomes_lineage_version_uq UNIQUE (tenant_id, lineage_id, version)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS outcomes_workflow_created
      ON ledger.outcomes (tenant_id, workflow_id, created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS outcomes_thread_type_created
      ON ledger.outcomes (tenant_id, thread_id, outcome_type, created_at DESC);
    """,
    # ledger.policy_decisions
    """
    CREATE TABLE IF NOT EXISTS ledger.policy_decisions (
      tenant_id              BIGINT       NOT NULL,
      policy_decision_id     UUID         NOT NULL,

      stage                  TEXT         NOT NULL,
      decision               TEXT         NOT NULL,
      reason_code            TEXT         NOT NULL,
      reason                 TEXT         NULL,

      requirements           JSONB        NOT NULL DEFAULT '{}'::jsonb,
      limits                 JSONB        NOT NULL DEFAULT '{}'::jsonb,
      redactions             JSONB        NOT NULL DEFAULT '[]'::jsonb,
      transform              JSONB        NULL,

      inputs_hash            BYTEA        NOT NULL,
      policy_engine_name     TEXT         NOT NULL,
      policy_engine_version  TEXT         NOT NULL,

      workflow_id            UUID         NULL,
      intent_id              UUID         NULL,
      plan_id                UUID         NULL,
      step_id                TEXT         NULL,
      job_id                 UUID         NULL,

      correlation_id         UUID         NOT NULL,
      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, policy_decision_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS policy_decisions_workflow_created
      ON ledger.policy_decisions (tenant_id, workflow_id, created_at DESC);
    """,
    # ledger.jobs
    """
    CREATE TABLE IF NOT EXISTS ledger.jobs (
      tenant_id              BIGINT       NOT NULL,
      job_id                 UUID         NOT NULL,
      schema_version         TEXT         NOT NULL DEFAULT '1.0',

      workflow_id            UUID         NULL,
      thread_id              BIGINT       NULL,
      intent_id              UUID         NULL,
      plan_id                UUID         NULL,
      step_id                TEXT         NULL,

      operator_name          TEXT         NOT NULL,
      operator_version       TEXT         NOT NULL,
      idempotency_key        TEXT         NOT NULL,

      effects                TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
      policy_tags            TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],
      data_classes           TEXT[]       NOT NULL DEFAULT ARRAY[]::TEXT[],

      status                 TEXT         NOT NULL,
      attempt_count          INT          NOT NULL DEFAULT 0,

      input_payload          JSONB        NOT NULL,
      input_hash             BYTEA        NULL,

      result_payload         JSONB        NULL,
      result_hash            BYTEA        NULL,

      error                  JSONB        NULL,
      nondeterminism         JSONB        NULL,

      trace_id               TEXT         NULL,
      trace_type             TEXT         NULL,

      metrics                JSONB        NULL,

      correlation_id         UUID         NOT NULL,
      producer_kind          TEXT         NOT NULL,
      producer_name          TEXT         NOT NULL,
      producer_version       TEXT         NOT NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
      started_at             TIMESTAMPTZ  NULL,
      finished_at            TIMESTAMPTZ  NULL,

      PRIMARY KEY (tenant_id, job_id)
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS jobs_idempotency_uq
      ON ledger.jobs (tenant_id, operator_name, idempotency_key);
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_workflow_created
      ON ledger.jobs (tenant_id, workflow_id, created_at DESC);
    """,
    """
    CREATE INDEX IF NOT EXISTS jobs_status_created
      ON ledger.jobs (tenant_id, status, created_at DESC);
    """,
    # ledger.job_attempts
    """
    CREATE TABLE IF NOT EXISTS ledger.job_attempts (
      tenant_id              BIGINT       NOT NULL,
      job_id                 UUID         NOT NULL,
      attempt_no             INT          NOT NULL,

      status                 TEXT         NOT NULL,
      started_at             TIMESTAMPTZ  NULL,
      finished_at            TIMESTAMPTZ  NULL,

      error                  JSONB        NULL,
      metrics                JSONB        NULL,

      trace_id               TEXT         NULL,
      trace_type             TEXT         NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, job_id, attempt_no),
      CONSTRAINT job_attempts_job_fk
        FOREIGN KEY (tenant_id, job_id) REFERENCES ledger.jobs(tenant_id, job_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS job_attempts_status_created
      ON ledger.job_attempts (tenant_id, status, created_at DESC);
    """,
    # ledger.gates
    """
    CREATE TABLE IF NOT EXISTS ledger.gates (
      tenant_id              BIGINT       NOT NULL,
      gate_id                UUID         NOT NULL,

      workflow_id            UUID         NOT NULL,
      step_id                TEXT         NOT NULL,

      gate_type              TEXT         NOT NULL,
      reason_code            TEXT         NOT NULL,
      summary                TEXT         NOT NULL,
      preview                JSONB        NOT NULL DEFAULT '{}'::jsonb,

      target_outcome_id      UUID         NULL,
      status                 TEXT         NOT NULL,
      expires_at             TIMESTAMPTZ  NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, gate_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS gates_workflow_step
      ON ledger.gates (tenant_id, workflow_id, step_id);
    """,
    # ledger.gate_decisions
    """
    CREATE TABLE IF NOT EXISTS ledger.gate_decisions (
      tenant_id              BIGINT       NOT NULL,
      gate_decision_id       UUID         NOT NULL,
      gate_id                UUID         NOT NULL,

      actor                  JSONB        NOT NULL,
      decision               TEXT         NOT NULL,
      payload                JSONB        NULL,

      created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),

      PRIMARY KEY (tenant_id, gate_decision_id),
      CONSTRAINT gate_decisions_gate_fk
        FOREIGN KEY (tenant_id, gate_id) REFERENCES ledger.gates(tenant_id, gate_id)
    );
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS gate_decisions_one_per_actor
      ON ledger.gate_decisions (tenant_id, gate_id, (actor->>'id'));
    """,
]


async def ensure_kernel_schema(conn) -> None:
    for stmt in _compact_statements(DDL_STATEMENTS):
        await conn.execute(stmt)


def _compact_statements(statements: Iterable[str]) -> Iterable[str]:
    for stmt in statements:
        cleaned = stmt.strip()
        if not cleaned:
            continue
        yield cleaned
