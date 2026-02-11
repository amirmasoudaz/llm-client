from __future__ import annotations

import json
import time
import uuid
from typing import Any

from jsonschema import Draft202012Validator

from ..contracts import ContractRegistry
from ..events import EventWriter, LedgerEvent
from ..prompts import PromptTemplateLoader, PromptRenderResult
from ..policy import PolicyEngine, PolicyDecisionStore, PolicyContext
from .base import Operator
from .registry import OperatorRegistry, OperatorAccessDenied
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
        prompt_render = self._render_manifest_prompt(manifest=manifest, payload=call.payload)

        effects = list(manifest.get("effects", []) or [])
        policy_tags = list(manifest.get("policy_tags", []) or [])

        trace = call.trace_context.to_dict()
        workflow_id = trace.get("workflow_id")
        correlation_id = trace.get("correlation_id")
        step_id = trace.get("step_id")
        plan_id = trace.get("plan_id")
        thread_id = trace.get("thread_id")
        intent_id = trace.get("intent_id")

        claim = await self._job_store.claim_job(
            operator_name=operator_name,
            operator_version=operator_version,
            idempotency_key=call.idempotency_key,
            workflow_id=workflow_id,
            thread_id=thread_id if isinstance(thread_id, int) else None,
            intent_id=intent_id if isinstance(intent_id, str) else None,
            plan_id=plan_id if isinstance(plan_id, str) else None,
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
            self._attach_prompt_metadata(result, prompt_render)
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
            self._attach_prompt_metadata(result, prompt_render)
            self._validate_schema(output_ref, result.to_dict())
            return result

        pre_invoke_denial = await self._pre_invoke_access_check(
            manifest=manifest,
            operator_name=operator_name,
            operator_version=operator_version,
            call=call,
            workflow_id=workflow_id,
            intent_id=intent_id,
            plan_id=plan_id,
            step_id=step_id,
            correlation_id=correlation_id,
            claim=claim,
        )
        if pre_invoke_denial is not None:
            self._attach_prompt_metadata(pre_invoke_denial, prompt_render)
            self._validate_schema(output_ref, pre_invoke_denial.to_dict())
            return pre_invoke_denial

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
        decision.intent_id = intent_id
        decision.plan_id = plan_id
        decision.step_id = step_id
        decision.job_id = str(claim.job_id) if claim.job_id else None
        decision.correlation_id = correlation_id
        await self._policy_store.record(decision)

        if decision.decision != "ALLOW":
            reason_message = "Operator invocation denied by policy"
            error_code = "policy_denied"
            if decision.decision == "REQUIRE_APPROVAL":
                reason_message = "Operator invocation requires approval"
                error_code = "policy_requires_approval"
            error = OperatorError(
                code=error_code,
                message=reason_message,
                category="policy_denied",
                retryable=False,
                details={"reason_code": decision.reason_code, "requirements": decision.requirements},
            )
            result = OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=0),
                error=error,
            )
            self._attach_prompt_metadata(result, prompt_render)
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
                thread_id=thread_id if isinstance(thread_id, int) else None,
                intent_id=_to_uuid_or_none(intent_id),
                plan_id=_to_uuid_or_none(plan_id),
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
            self._attach_prompt_metadata(result, prompt_render)
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
                    thread_id=thread_id if isinstance(thread_id, int) else None,
                    intent_id=_to_uuid_or_none(intent_id),
                    plan_id=_to_uuid_or_none(plan_id),
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
        self._attach_prompt_metadata(result, prompt_render)

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
        outcome_decision.intent_id = intent_id
        outcome_decision.plan_id = plan_id
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
                thread_id=thread_id if isinstance(thread_id, int) else None,
                intent_id=_to_uuid_or_none(intent_id),
                plan_id=_to_uuid_or_none(plan_id),
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

    def _render_manifest_prompt(
        self,
        *,
        manifest: dict[str, Any],
        payload: dict[str, Any],
    ) -> PromptRenderResult | None:
        template_id = self._resolve_manifest_prompt_template(manifest=manifest, payload=payload)
        if not template_id:
            return None
        if self._prompt_loader is None:
            raise ValueError("prompt loader is required for manifest-defined prompt templates")
        return self._prompt_loader.render(template_id, self._build_prompt_context(payload))

    def _resolve_manifest_prompt_template(
        self,
        *,
        manifest: dict[str, Any],
        payload: dict[str, Any],
    ) -> str | None:
        direct = manifest.get("prompt_template_id") or manifest.get("prompt_template")
        direct_template = _coerce_template_id(direct)
        if direct_template:
            return direct_template

        prompt_templates = manifest.get("prompt_templates")
        if isinstance(prompt_templates, str) and prompt_templates.strip():
            return prompt_templates.strip()
        if isinstance(prompt_templates, dict):
            variant = str(payload.get("prompt_variant") or "").strip().lower()
            document_type = _normalize_document_type_variant(payload.get("document_type"))
            for key in (variant, document_type):
                if not key:
                    continue
                candidate = _coerce_template_id(prompt_templates.get(key))
                if candidate:
                    return candidate
                nested = prompt_templates.get(key)
                if isinstance(nested, dict):
                    candidate = _coerce_template_id(nested.get("primary") or nested.get("default"))
                    if candidate:
                        return candidate

            by_document_type = prompt_templates.get("by_document_type")
            if isinstance(by_document_type, dict):
                if document_type:
                    candidate = _coerce_template_id(by_document_type.get(document_type))
                    if candidate:
                        return candidate
                candidate = _coerce_template_id(by_document_type.get("default") or by_document_type.get("primary"))
                if candidate:
                    return candidate

            primary = prompt_templates.get("primary")
            primary_template = _coerce_template_id(primary)
            if primary_template:
                return primary_template
            fallback = _coerce_template_id(prompt_templates.get("default"))
            if fallback:
                return fallback
            for value in prompt_templates.values():
                template_id = _coerce_template_id(value)
                if template_id:
                    return template_id
                if isinstance(value, dict):
                    nested_primary = _coerce_template_id(value.get("primary") or value.get("default"))
                    if nested_primary:
                        return nested_primary
                    for nested_value in value.values():
                        template_id = _coerce_template_id(nested_value)
                        if template_id:
                            return template_id
        if isinstance(prompt_templates, list):
            for item in prompt_templates:
                template_id = _coerce_template_id(item)
                if template_id:
                    return template_id
        return None

    def _build_prompt_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = dict(payload)
        if not isinstance(context.get("email"), dict):
            context["email"] = {
                "subject": (
                    payload.get("subject")
                    or payload.get("current_subject")
                    or payload.get("fallback_subject")
                    or payload.get("subject_override")
                    or ""
                ),
                "body": payload.get("body") or payload.get("current_body") or payload.get("fallback_body") or "",
            }
        context["professor"] = payload.get("professor") if isinstance(payload.get("professor"), dict) else {}
        context["funding_request"] = (
            payload.get("funding_request") if isinstance(payload.get("funding_request"), dict) else {}
        )
        context["student_profile"] = (
            payload.get("student_profile") if isinstance(payload.get("student_profile"), dict) else {}
        )
        context["requested_edits"] = payload.get("requested_edits") if isinstance(payload.get("requested_edits"), list) else []
        context["review_report"] = payload.get("review_report") if isinstance(payload.get("review_report"), dict) else {}
        context["custom_instructions"] = payload.get("custom_instructions")
        return context

    def _attach_prompt_metadata(self, result: OperatorResult, prompt_render: PromptRenderResult | None) -> None:
        if prompt_render is None:
            return
        result.prompt_template_id = prompt_render.template_id
        result.prompt_template_hash = prompt_render.template_hash

    async def _pre_invoke_access_check(
        self,
        *,
        manifest: dict[str, Any],
        operator_name: str,
        operator_version: str,
        call: OperatorCall,
        workflow_id: str | None,
        intent_id: str | None,
        plan_id: str | None,
        step_id: str | None,
        correlation_id: str | None,
        claim,
    ) -> OperatorResult | None:
        try:
            self._registry.enforce_invocation_policy(
                name=operator_name,
                version=operator_version,
                auth_context=call.auth_context.to_dict(),
                manifest=manifest,
            )
            return None
        except OperatorAccessDenied as exc:
            denial = PolicyContext(
                stage="action",
                operator_name=operator_name,
                operator_version=operator_version,
                effects=list(manifest.get("effects") or []),
                policy_tags=list(manifest.get("policy_tags") or []),
                data_classes=[],
                auth_context=call.auth_context.to_dict(),
                trace_context=call.trace_context.to_dict(),
                input_payload=call.payload,
            )
            decision = self._policy_engine.evaluate(denial)
            decision.decision = "DENY"
            decision.reason_code = exc.reason_code
            decision.reason = exc.reason
            decision.requirements = dict(exc.requirements)
            decision.workflow_id = workflow_id
            decision.intent_id = intent_id
            decision.plan_id = plan_id
            decision.step_id = step_id
            decision.job_id = str(claim.job_id) if claim.job_id else None
            decision.correlation_id = correlation_id
            await self._policy_store.record(decision)

            if claim.job_id and claim.attempt_no:
                await self._job_store.complete_job(
                    job_id=claim.job_id,
                    attempt_no=claim.attempt_no,
                    status="failed",
                    result_payload={
                        "schema_version": "1.0",
                        "status": "failed",
                        "result": None,
                        "artifacts": [],
                        "metrics": {"latency_ms": 0},
                        "error": {
                            "code": "policy_denied",
                            "message": exc.reason,
                            "category": "policy_denied",
                            "retryable": False,
                            "details": {"reason_code": exc.reason_code, "requirements": exc.requirements},
                        },
                    },
                    error={
                        "code": "policy_denied",
                        "message": exc.reason,
                        "category": "policy_denied",
                        "retryable": False,
                        "details": {"reason_code": exc.reason_code, "requirements": exc.requirements},
                    },
                    metrics={"latency_ms": 0},
                )

            return OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=0),
                error=OperatorError(
                    code="policy_denied",
                    message=exc.reason,
                    category="policy_denied",
                    retryable=False,
                    details={"reason_code": exc.reason_code, "requirements": exc.requirements},
                ),
            )


def _to_uuid_or_none(value: Any) -> uuid.UUID | None:
    if not isinstance(value, str):
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def _coerce_template_id(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        direct = value.get("template_id") or value.get("id")
        if isinstance(direct, str):
            stripped = direct.strip()
            return stripped or None
    return None


def _normalize_document_type_variant(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"resume": "cv", "cover_letter": "letter", "coverletter": "letter", "motivation_letter": "letter"}
    return aliases.get(normalized, normalized)
