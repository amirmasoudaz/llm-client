from __future__ import annotations

import json
import time
import uuid
from typing import Any

from jsonschema import Draft202012Validator

from ..contracts import ContractRegistry
from ..events import EventWriter, LedgerEvent
from ..prompts import PromptTemplateLoader
from ..policy import PolicyEngine, PolicyDecisionStore, PolicyContext
from .base import Operator
from .registry import OperatorRegistry
from .store import OperatorJobStore
from .types import OperatorCall, OperatorResult, OperatorError, OperatorMetrics


class OperatorExecutor:
    def __init__(
        self,
        *,
        contracts: ContractRegistry,
        registry: OperatorRegistry,
        job_store: OperatorJobStore,
        policy_engine: PolicyEngine,
        policy_store: PolicyDecisionStore,
        event_writer: EventWriter,
        prompt_loader: PromptTemplateLoader | None = None,
    ) -> None:
        self._contracts = contracts
        self._registry = registry
        self._job_store = job_store
        self._policy_engine = policy_engine
        self._policy_store = policy_store
        self._event_writer = event_writer
        self._prompt_loader = prompt_loader

    async def execute(self, *, operator_name: str, operator_version: str, call: OperatorCall) -> OperatorResult:
        manifest = self._registry.get_manifest(operator_name, operator_version)
        input_ref = manifest.get("schemas", {}).get("input")
        output_ref = manifest.get("schemas", {}).get("output")
        if not input_ref or not output_ref:
            raise ValueError(f"operator manifest missing schemas: {operator_name}@{operator_version}")

        self._validate_schema(input_ref, call.to_dict())

        effects = list(manifest.get("effects", []) or [])
        policy_tags = list(manifest.get("policy_tags", []) or [])

        trace = call.trace_context.to_dict()
        workflow_id = trace.get("workflow_id")
        correlation_id = trace.get("correlation_id")
        step_id = trace.get("step_id")

        claim = await self._job_store.claim_job(
            operator_name=operator_name,
            operator_version=operator_version,
            idempotency_key=call.idempotency_key,
            workflow_id=workflow_id,
            thread_id=None,
            intent_id=None,
            plan_id=trace.get("plan_id"),
            step_id=step_id,
            correlation_id=correlation_id,
            input_payload=call.payload,
            effects=effects,
            policy_tags=policy_tags,
        )

        if claim.status == "existing_success" and claim.result_payload is not None:
            result = OperatorResult(
                status="succeeded",
                result=claim.result_payload.get("result"),
                artifacts=claim.result_payload.get("artifacts", []),
                metrics=OperatorMetrics(latency_ms=0),
                error=None,
            )
            self._validate_schema(output_ref, result.to_dict())
            return result

        if claim.status == "in_progress":
            result = OperatorResult(
                status="in_progress",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=0),
                error=None,
            )
            self._validate_schema(output_ref, result.to_dict())
            return result

        # Policy: action stage
        policy_ctx = PolicyContext(
            stage="action",
            operator_name=operator_name,
            operator_version=operator_version,
            effects=effects,
            policy_tags=policy_tags,
            data_classes=[],
            auth_context=call.auth_context.to_dict(),
            trace_context=trace,
            input_payload=call.payload,
        )
        decision = self._policy_engine.evaluate(policy_ctx)
        decision.workflow_id = workflow_id
        decision.step_id = step_id
        decision.job_id = str(claim.job_id) if claim.job_id else None
        decision.correlation_id = correlation_id
        await self._policy_store.record(decision)

        if decision.decision != "ALLOW":
            error = OperatorError(
                code="policy_denied",
                message="Operator invocation denied by policy",
                category="policy_denied",
                retryable=False,
            )
            result = OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=0),
                error=error,
            )
            if claim.job_id and claim.attempt_no:
                await self._job_store.complete_job(
                    job_id=claim.job_id,
                    attempt_no=claim.attempt_no,
                    status="failed",
                    result_payload=result.to_dict(),
                    error=error.to_dict(),
                    metrics=result.metrics.to_dict(),
                )
            self._validate_schema(output_ref, result.to_dict())
            return result

        await self._event_writer.append(
            LedgerEvent(
                tenant_id=call.auth_context.tenant_id,
                event_id=uuid.uuid4(),
                workflow_id=uuid.UUID(workflow_id),
                step_id=step_id,
                job_id=claim.job_id,
                event_type="job.started",
                actor={"type": "system", "id": call.auth_context.principal.get("id"), "role": "operator"},
                payload={"operator": operator_name, "version": operator_version},
                correlation_id=uuid.UUID(correlation_id),
                producer_kind="kernel",
                producer_name="operator_executor",
                producer_version="1.0",
            )
        )

        start = time.monotonic()
        try:
            operator = self._registry.get(operator_name, operator_version)
        except KeyError as exc:
            error = OperatorError(
                code="operator_not_registered",
                message=str(exc),
                category="operator_bug",
                retryable=False,
            )
            result = OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=0),
                error=error,
            )
            if claim.job_id and claim.attempt_no:
                await self._job_store.complete_job(
                    job_id=claim.job_id,
                    attempt_no=claim.attempt_no,
                    status="failed",
                    result_payload=result.to_dict(),
                    error=error.to_dict(),
                    metrics=result.metrics.to_dict(),
                )
            await self._event_writer.append(
                LedgerEvent(
                    tenant_id=call.auth_context.tenant_id,
                    event_id=uuid.uuid4(),
                    workflow_id=uuid.UUID(workflow_id),
                    step_id=step_id,
                    job_id=claim.job_id,
                    event_type="job.failed",
                    actor={"type": "system", "id": call.auth_context.principal.get("id"), "role": "operator"},
                    payload={"operator": operator_name, "version": operator_version, "status": "failed", "error": str(exc)},
                    correlation_id=uuid.UUID(correlation_id),
                    producer_kind="kernel",
                    producer_name="operator_executor",
                    producer_version="1.0",
                )
            )
            self._validate_schema(output_ref, result.to_dict())
            return result
        try:
            result = await operator.run(call)
        except Exception as exc:  # pragma: no cover - defensive
            error = OperatorError(
                code="operator_exception",
                message=str(exc),
                category="operator_bug",
                retryable=False,
            )
            result = OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=error,
            )

        # Ensure latency
        result.metrics.latency_ms = int((time.monotonic() - start) * 1000)

        self._validate_schema(output_ref, result.to_dict())

        # Outcome policy stage (record only for now)
        outcome_ctx = PolicyContext(
            stage="outcome",
            operator_name=operator_name,
            operator_version=operator_version,
            effects=effects,
            policy_tags=policy_tags,
            data_classes=[],
            auth_context=call.auth_context.to_dict(),
            trace_context=trace,
            input_payload=result.to_dict(),
        )
        outcome_decision = self._policy_engine.evaluate(outcome_ctx)
        outcome_decision.workflow_id = workflow_id
        outcome_decision.step_id = step_id
        outcome_decision.job_id = str(claim.job_id) if claim.job_id else None
        outcome_decision.correlation_id = correlation_id
        await self._policy_store.record(outcome_decision)

        if claim.job_id and claim.attempt_no:
            await self._job_store.complete_job(
                job_id=claim.job_id,
                attempt_no=claim.attempt_no,
                status="succeeded" if result.status == "succeeded" else "failed",
                result_payload=result.to_dict(),
                error=result.error.to_dict() if result.error else None,
                metrics=result.metrics.to_dict(),
            )

        await self._event_writer.append(
            LedgerEvent(
                tenant_id=call.auth_context.tenant_id,
                event_id=uuid.uuid4(),
                workflow_id=uuid.UUID(workflow_id),
                step_id=step_id,
                job_id=claim.job_id,
                event_type="job.completed" if result.status == "succeeded" else "job.failed",
                actor={"type": "system", "id": call.auth_context.principal.get("id"), "role": "operator"},
                payload={"operator": operator_name, "version": operator_version, "status": result.status},
                correlation_id=uuid.UUID(correlation_id),
                producer_kind="kernel",
                producer_name="operator_executor",
                producer_version="1.0",
            )
        )

        return result

    def _validate_schema(self, schema_ref: str, instance: dict[str, Any]) -> None:
        schema = self._contracts.get_schema_by_ref(schema_ref)
        validator = Draft202012Validator(schema, resolver=self._contracts.resolver_for(schema))
        errors = list(validator.iter_errors(instance))
        if errors:
            messages = "; ".join([f"{err.message}" for err in errors])
            raise ValueError(f"schema validation failed for {schema_ref}: {messages}")
