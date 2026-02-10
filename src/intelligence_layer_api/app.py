from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_runtime import ExecutionRequest, BudgetSpec, PolicyRef

from llm_client import load_env

from .il_db import ILDB, get_pool
from .documents import (
    build_presigned_download_url,
    fetch_document_metadata,
    fetch_document_revisions,
    ingest_thread_document,
    stage_upload_artifact,
    upload_storage_health,
)
from .runtime_kernel import KernelContainer, build_kernel
from .settings import get_settings
from .auth import DevBypassAuthAdapter
from .billing import CreditManager, request_key_for_workflow
from intelligence_layer_ops.platform_tools import platform_load_funding_thread_context
from intelligence_layer_kernel.contracts import ContractRegistry
from intelligence_layer_kernel.events import EventWriter, SSEProjector, LedgerEvent
from intelligence_layer_kernel.operators import (
    OperatorRegistry,
    OperatorJobStore,
    OperatorExecutor,
    AuthContext,
    TraceContext as OperatorTraceContext,
    OperatorCall,
)
from intelligence_layer_kernel.policy import PolicyEngine as ILPolicyEngine, PolicyDecisionStore
from intelligence_layer_kernel.operators.implementations.thread_create_or_load import ThreadCreateOrLoadOperator
from intelligence_layer_kernel.operators.implementations.workflow_gate_resolve import WorkflowGateResolveOperator
from intelligence_layer_kernel.operators.implementations.platform_context_load import PlatformContextLoadOperator
from intelligence_layer_kernel.operators.implementations.funding_request_fields_update_propose import (
    FundingRequestFieldsUpdateProposeOperator,
)
from intelligence_layer_kernel.operators.implementations.funding_request_fields_update_apply import (
    FundingRequestFieldsUpdateApplyOperator,
)
from intelligence_layer_kernel.operators.implementations.platform_attachments_list import (
    PlatformAttachmentsListOperator,
)
from intelligence_layer_kernel.operators.implementations.documents_import_from_platform_attachment import (
    DocumentsImportFromPlatformAttachmentOperator,
)
from intelligence_layer_kernel.operators.implementations.documents_process import (
    DocumentsProcessOperator,
)
from intelligence_layer_kernel.operators.implementations.documents_review import (
    DocumentsReviewOperator,
)
from intelligence_layer_kernel.operators.implementations.documents_upload import (
    DocumentsUploadOperator,
)
from intelligence_layer_kernel.operators.implementations.email_optimize_draft import (
    EmailOptimizeDraftOperator,
)
from intelligence_layer_kernel.operators.implementations.funding_email_draft_update_propose import (
    FundingEmailDraftUpdateProposeOperator,
)
from intelligence_layer_kernel.operators.implementations.funding_email_draft_update_apply import (
    FundingEmailDraftUpdateApplyOperator,
)
from intelligence_layer_kernel.operators.implementations.email_apply_to_platform_propose import (
    EmailApplyToPlatformProposeOperator,
)
from intelligence_layer_kernel.operators.implementations.email_review_draft import EmailReviewDraftOperator
from intelligence_layer_kernel.operators.implementations.conversation_suggestions_generate import (
    ConversationSuggestionsGenerateOperator,
)
from intelligence_layer_kernel.operators.implementations.professor_profile_retrieve import (
    ProfessorProfileRetrieveOperator,
)
from intelligence_layer_kernel.operators.implementations.professor_summarize import (
    ProfessorSummarizeOperator,
)
from intelligence_layer_kernel.operators.implementations.professor_alignment_score import (
    ProfessorAlignmentScoreOperator,
)
from intelligence_layer_kernel.operators.implementations.student_profile_load_or_create import (
    StudentProfileLoadOrCreateOperator,
)
from intelligence_layer_kernel.operators.implementations.student_profile_update import (
    StudentProfileUpdateOperator,
)
from intelligence_layer_kernel.operators.implementations.student_profile_requirements_evaluate import (
    StudentProfileRequirementsEvaluateOperator,
)
from intelligence_layer_kernel.operators.implementations.memory_upsert import MemoryUpsertOperator
from intelligence_layer_kernel.operators.implementations.memory_retrieve import MemoryRetrieveOperator
from intelligence_layer_kernel.operators.implementations.documents_common import extract_attachment_ids
from intelligence_layer_kernel.runtime import WorkflowKernel
from intelligence_layer_kernel.runtime.store import OutcomeStore


load_env()  # keep prototyping simple: auto-load repo `.env` for the Layer 2 API process

logger = logging.getLogger("intelligence_layer_api")

app = FastAPI(
    title="CanApply Intelligence Layer (Layer 2)",
    version="0.0.1",
    description="API for workflow execution, thread management, and operator actions.",
    docs_url="/docs",       # Swagger UI (set to None to disable)
    redoc_url="/redoc",     # ReDoc (set to None to disable)
    openapi_url="/openapi.json",
)


async def _emit_progress_event(
    *,
    event_writer: EventWriter,
    tenant_id: int,
    workflow_id: uuid.UUID,
    actor: dict[str, Any],
    stage: str,
    correlation_id: uuid.UUID,
    thread_id: int | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    payload = {"stage": stage}
    if detail:
        payload.update(detail)
    await event_writer.append(
        LedgerEvent(
            tenant_id=tenant_id,
            event_id=uuid.uuid4(),
            workflow_id=workflow_id,
            event_type="progress",
            actor=actor,
            payload=payload,
            correlation_id=correlation_id,
            producer_kind="api",
            producer_name="intelligence_layer_api",
            producer_version="0.0.1",
            thread_id=thread_id,
        )
    )


async def _emit_final_error_event(
    *,
    event_writer: EventWriter,
    tenant_id: int,
    workflow_id: uuid.UUID,
    actor: dict[str, Any],
    message: str,
    correlation_id: uuid.UUID,
    thread_id: int | None = None,
) -> None:
    await event_writer.append(
        LedgerEvent(
            tenant_id=tenant_id,
            event_id=uuid.uuid4(),
            workflow_id=workflow_id,
            event_type="final_error",
            actor=actor,
            payload={"error": message},
            correlation_id=correlation_id,
            producer_kind="api",
            producer_name="intelligence_layer_api",
            producer_version="0.0.1",
            thread_id=thread_id,
        )
    )


class ThreadInitRequest(BaseModel):
    funding_request_id: int = Field(..., ge=1)
    student_id: int | None = Field(default=None, ge=1)
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


@app.get("/v1/debug/storage/upload-health")
async def debug_upload_storage_health() -> dict[str, Any]:
    settings = get_settings()
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not found")
    return upload_storage_health()


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
    app.state.sse_projector = SSEProjector(pool=pool, tenant_id=ildb.tenant_id)

    contracts = ContractRegistry()
    contracts.load()
    operator_registry = OperatorRegistry(contracts)
    operator_registry.register(ThreadCreateOrLoadOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(WorkflowGateResolveOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(PlatformContextLoadOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(PlatformAttachmentsListOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(DocumentsUploadOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(
        DocumentsImportFromPlatformAttachmentOperator(pool=pool, tenant_id=ildb.tenant_id)
    )
    operator_registry.register(DocumentsProcessOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(DocumentsReviewOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(FundingRequestFieldsUpdateProposeOperator())
    operator_registry.register(FundingRequestFieldsUpdateApplyOperator())
    operator_registry.register(EmailOptimizeDraftOperator())
    operator_registry.register(FundingEmailDraftUpdateProposeOperator())
    operator_registry.register(FundingEmailDraftUpdateApplyOperator())
    operator_registry.register(EmailApplyToPlatformProposeOperator())
    operator_registry.register(EmailReviewDraftOperator())
    operator_registry.register(ConversationSuggestionsGenerateOperator())
    operator_registry.register(ProfessorProfileRetrieveOperator())
    operator_registry.register(ProfessorSummarizeOperator())
    operator_registry.register(ProfessorAlignmentScoreOperator())
    operator_registry.register(StudentProfileLoadOrCreateOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(StudentProfileUpdateOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(
        StudentProfileRequirementsEvaluateOperator(pool=pool, tenant_id=ildb.tenant_id)
    )
    operator_registry.register(MemoryUpsertOperator(pool=pool, tenant_id=ildb.tenant_id))
    operator_registry.register(MemoryRetrieveOperator(pool=pool, tenant_id=ildb.tenant_id))
    job_store = OperatorJobStore(pool=pool, tenant_id=ildb.tenant_id)
    event_writer = EventWriter(pool=pool)
    policy_engine = ILPolicyEngine()
    policy_store = PolicyDecisionStore(pool=pool, tenant_id=ildb.tenant_id, event_writer=event_writer)
    operator_executor = OperatorExecutor(
        contracts=contracts,
        registry=operator_registry,
        job_store=job_store,
        policy_engine=policy_engine,
        policy_store=policy_store,
        event_writer=event_writer,
    )

    app.state.operator_registry = operator_registry
    app.state.outcome_store = OutcomeStore(pool=pool, tenant_id=ildb.tenant_id)
    app.state.operator_executor = operator_executor
    app.state.event_writer = event_writer
    app.state.workflow_kernel = WorkflowKernel(
        contracts=contracts,
        operator_executor=operator_executor,
        policy_engine=policy_engine,
        policy_store=policy_store,
        event_writer=event_writer,
        pool=pool,
        tenant_id=ildb.tenant_id,
    )
    app.state.auth_adapter = DevBypassAuthAdapter(allow_bypass=settings.auth_bypass)
    app.state.credit_manager = CreditManager(
        pool=pool,
        tenant_id=ildb.tenant_id,
        bootstrap_enabled=settings.credits_bootstrap,
        bootstrap_credits=settings.credits_bootstrap_amount,
        reservation_ttl_sec=settings.credits_reservation_ttl_sec,
        min_reserve_credits=settings.credits_min_reserve,
    )
    storage_health = upload_storage_health()
    app.state.upload_storage_health = storage_health
    if storage_health.get("ready"):
        logger.info(
            "upload storage ready mode=%s bucket=%s prefix=%s",
            storage_health.get("mode"),
            storage_health.get("bucket"),
            storage_health.get("prefix"),
        )
    else:
        logger.warning(
            "upload storage not ready mode=%s reasons=%s",
            storage_health.get("mode"),
            ",".join(storage_health.get("reasons") or []),
        )


@app.on_event("shutdown")
async def _shutdown() -> None:
    kc: KernelContainer | None = getattr(app.state, "kernel_container", None)
    if kc is not None:
        try:
            await kc.event_bridge.stop()
            await kc.engine.provider.close()
        except Exception:
            pass


@app.post("/v1/threads/init", response_model=ThreadInitResponse)
async def init_thread(req: ThreadInitRequest, response: Response, request: Request) -> ThreadInitResponse:
    ildb: ILDB = app.state.ildb
    auth_adapter: DevBypassAuthAdapter = app.state.auth_adapter
    operator_registry: OperatorRegistry = app.state.operator_registry
    auth = await auth_adapter.authenticate(
        request=request,
        funding_request_id=req.funding_request_id,
        student_id_override=req.student_id,
    )
    if not auth.ok:
        raise HTTPException(status_code=auth.status_code, detail=auth.reason or "unauthorized")
    student_id = int(auth.principal_id or 0)
    if student_id <= 0:
        raise HTTPException(status_code=401, detail="unauthorized")

    thread_id, status, is_new = await ildb.get_or_create_thread(
        student_id=student_id,
        funding_request_id=req.funding_request_id,
    )
    response.status_code = 201 if is_new else 200
    onboarding_gate = "ready"
    missing_requirements: list[str] = []
    try:
        workflow_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        auth_context = AuthContext(
            tenant_id=ildb.tenant_id,
            principal={"type": "student", "id": student_id},
            scopes=list(auth.scopes),
        )
        load_call = OperatorCall(
            payload={"thread_id": thread_id, "source": "system"},
            idempotency_key=f"profile_load_or_create:{ildb.tenant_id}:{thread_id}",
            auth_context=auth_context,
            trace_context=OperatorTraceContext(
                correlation_id=correlation_id,
                workflow_id=workflow_id,
                step_id="init_profile_load",
            ),
        )
        load_operator = operator_registry.get("StudentProfile.LoadOrCreate", "1.0.0")
        load_result = await load_operator.run(load_call)
        if load_result.status == "succeeded":
            evaluate_call = OperatorCall(
                payload={
                    "thread_id": thread_id,
                    "intent_type": "Funding.Outreach.Email.Generate",
                    "required_requirements": ["base_profile_complete", "background_data_complete"],
                    "strict": False,
                },
                idempotency_key=f"profile_requirements_eval:{ildb.tenant_id}:{thread_id}:init",
                auth_context=auth_context,
                trace_context=OperatorTraceContext(
                    correlation_id=correlation_id,
                    workflow_id=workflow_id,
                    step_id="init_profile_requirements",
                ),
            )
            evaluate_operator = operator_registry.get("StudentProfile.Requirements.Evaluate", "1.0.0")
            evaluate_result = await evaluate_operator.run(evaluate_call)
            if evaluate_result.status == "succeeded":
                requirements = (evaluate_result.result or {}).get("requirements")
                if isinstance(requirements, dict):
                    missing_raw = requirements.get("missing_requirements")
                    if isinstance(missing_raw, list):
                        missing_requirements = [str(item) for item in missing_raw if str(item).strip()]
                    onboarding_gate = "needs_onboarding" if missing_requirements else "ready"
    except Exception:
        onboarding_gate = "ready"
        missing_requirements = []

    resp = ThreadInitResponse(
        thread_id=str(thread_id),
        thread_status=status,
        is_new=is_new,
        message="created_thread" if is_new else "existing_thread",
        onboarding_gate=onboarding_gate,
        missing_requirements=missing_requirements,
    )
    if auth.bypass:
        resp.message = "auth_bypass"
    return resp


class SubmitQueryRequest(BaseModel):
    message: str = Field(..., min_length=1)
    attachments: list[Any] = Field(default_factory=list)
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


class ThreadAttachmentUploadResponse(BaseModel):
    attachment_id: int
    document_id: str
    document_type: str
    content_hash: str
    mime: str
    size_bytes: int
    dedupe_reused: bool
    parsed: bool


class DocumentRevisionResponse(BaseModel):
    revision_id: str
    revision_no: int
    revision_kind: str
    content_hash: str
    processor_version: str | None = None
    created_at: str | None = None
    download_url: str | None = None


class DocumentMetadataResponse(BaseModel):
    document_id: str
    thread_id: int
    funding_request_id: int
    student_id: int
    source_attachment_id: int | None = None
    document_type: str
    lifecycle: str
    mime: str
    content_hash: str
    content_size_bytes: int
    source_object_uri: str
    current_revision_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    download_url: str | None = None
    revisions: list[DocumentRevisionResponse] = Field(default_factory=list)


_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")


def _extract_profile_gate_payload(message: str, gate_preview: dict[str, Any] | None) -> dict[str, Any]:
    text = message.strip()
    if not text:
        return {}

    missing_fields: set[str] = set()
    if isinstance(gate_preview, dict):
        data = gate_preview.get("data")
        if isinstance(data, dict):
            raw_missing = data.get("missing_fields")
            if isinstance(raw_missing, list):
                for item in raw_missing:
                    item_text = str(item).strip()
                    if item_text:
                        missing_fields.add(item_text)

    updates: dict[str, Any] = {}

    email_match = _EMAIL_RE.search(text)
    if email_match:
        updates["email"] = email_match.group(1)

    first_name_match = re.search(r"\bfirst\s+name\s*(?:is|=|:)\s*([A-Za-z][A-Za-z'\-]{0,63})", text, re.IGNORECASE)
    if first_name_match:
        updates["first_name"] = first_name_match.group(1)
    last_name_match = re.search(r"\blast\s+name\s*(?:is|=|:)\s*([A-Za-z][A-Za-z'\-]{0,63})", text, re.IGNORECASE)
    if last_name_match:
        updates["last_name"] = last_name_match.group(1)

    name_match = re.search(
        r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z'\-]{0,63})(?:\s+([A-Za-z][A-Za-z'\-]{0,63}))?",
        text,
        re.IGNORECASE,
    )
    if name_match:
        updates.setdefault("first_name", name_match.group(1))
        if name_match.group(2):
            updates.setdefault("last_name", name_match.group(2))

    iam_match = re.search(
        r"\bi\s+am\s+([A-Za-z][A-Za-z'\-]{0,63})(?:\s+([A-Za-z][A-Za-z'\-]{0,63}))?",
        text,
        re.IGNORECASE,
    )
    if iam_match:
        updates.setdefault("first_name", iam_match.group(1))
        if iam_match.group(2):
            updates.setdefault("last_name", iam_match.group(2))

    research_match = re.search(
        r"\bresearch\s+interest\s*(?:is|to|=|:)?\s*([^,\n;]+)",
        text,
        re.IGNORECASE,
    )
    if research_match:
        research_interest = research_match.group(1).strip(" .")
        if research_interest:
            updates["research_interest"] = research_interest
    elif "context.background.research_interests" in missing_fields:
        fallback_interest = re.search(r"\binterested\s+in\s+([^,\n;]+)", text, re.IGNORECASE)
        if fallback_interest:
            research_interest = fallback_interest.group(1).strip(" .")
            if research_interest:
                updates["research_interest"] = research_interest

    return updates


@app.post("/v1/threads/{thread_id}/attachments", response_model=ThreadAttachmentUploadResponse)
async def upload_thread_attachment(
    thread_id: str,
    request: Request,
    document_type: str | None = Query(default=None),
    title: str | None = Query(default=None),
    file_name: str | None = Query(default=None),
) -> ThreadAttachmentUploadResponse:
    ildb: ILDB = app.state.ildb
    auth_adapter: DevBypassAuthAdapter = app.state.auth_adapter
    settings = get_settings()

    try:
        thread_id_int = int(thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="thread_id must be an integer") from exc

    thread = await ildb.get_thread(thread_id=thread_id_int)
    if not thread:
        raise HTTPException(status_code=404, detail="thread not found")

    auth = await auth_adapter.authenticate(
        request=request,
        funding_request_id=int(thread["funding_request_id"]),
        student_id_override=int(thread["student_id"]),
    )
    if not auth.ok:
        raise HTTPException(status_code=auth.status_code, detail=auth.reason or "unauthorized")
    principal_id = int(auth.principal_id or 0)
    if principal_id != int(thread["student_id"]):
        raise HTTPException(status_code=403, detail="forbidden")

    try:
        data = await request.body()
        incoming_name = file_name or request.headers.get("x-file-name") or "document.bin"
        incoming_content_type = request.headers.get("content-type")
        lifecycle = "sandbox"
        if settings.use_workflow_kernel:
            kernel: WorkflowKernel = app.state.workflow_kernel
            outcome_store: OutcomeStore = app.state.outcome_store
            staged_artifact = stage_upload_artifact(
                tenant_id=ildb.tenant_id,
                thread_id=thread_id_int,
                student_id=int(thread["student_id"]),
                file_bytes=data,
                file_name=incoming_name,
                content_type=incoming_content_type,
            )
            actor = {
                "tenant_id": ildb.tenant_id,
                "principal": {"type": "student", "id": principal_id},
                "role": "student",
                "trust_level": auth.trust_level,
                "scopes": list(auth.scopes),
            }
            workflow_id = uuid.uuid4()
            result = await kernel.start_intent(
                intent_type="Documents.Upload",
                inputs={
                    "document_type": document_type or "cv",
                    "title": title,
                    "lifecycle": lifecycle,
                    "artifact": {
                        "object_uri": staged_artifact["object_uri"],
                        "hash": staged_artifact["hash"],
                        "mime": staged_artifact["mime"],
                        "name": staged_artifact["name"],
                        "size_bytes": staged_artifact["size_bytes"],
                    },
                },
                thread_id=thread_id_int,
                scope_type="funding_request",
                scope_id=str(thread["funding_request_id"]),
                actor=actor,
                source="api.upload",
                workflow_id=workflow_id,
            )
            if result.status != "completed":
                raise HTTPException(status_code=500, detail="document upload workflow did not complete")
            outcomes = await outcome_store.list_by_workflow(workflow_id=result.workflow_id)
            uploaded_document_id = _extract_uploaded_document_id(outcomes)
            if not uploaded_document_id:
                raise HTTPException(status_code=500, detail="document upload outcome missing")
            uploaded_row = await fetch_document_metadata(
                pool=ildb.pool,
                tenant_id=ildb.tenant_id,
                document_id=uploaded_document_id,
            )
            if uploaded_row is None:
                raise HTTPException(status_code=500, detail="uploaded document not found")
            source_metadata = uploaded_row.get("source_metadata")
            dedupe_reused = False
            if isinstance(source_metadata, dict):
                dedupe_reused = bool(source_metadata.get("dedupe_reused"))
            uploaded = ThreadAttachmentUploadResponse(
                attachment_id=int(uploaded_row.get("source_attachment_id") or 0),
                document_id=uploaded_document_id,
                document_type=str(uploaded_row.get("document_type") or document_type or "cv"),
                content_hash=str(uploaded_row.get("content_hash") or ""),
                mime=str(uploaded_row.get("mime") or incoming_content_type or "application/octet-stream"),
                size_bytes=int(uploaded_row.get("content_size_bytes") or len(data)),
                dedupe_reused=dedupe_reused,
                parsed=bool(
                    (isinstance(source_metadata, dict) and source_metadata.get("parsed"))
                    or dedupe_reused
                ),
            )
            return uploaded
        uploaded = await ingest_thread_document(
            pool=ildb.pool,
            tenant_id=ildb.tenant_id,
            thread_id=thread_id_int,
            student_id=int(thread["student_id"]),
            funding_request_id=int(thread["funding_request_id"]),
            file_bytes=data,
            file_name=incoming_name,
            content_type=incoming_content_type,
            document_type_hint=document_type,
            title=title,
            lifecycle=lifecycle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"document upload failed: {exc}") from exc

    return ThreadAttachmentUploadResponse(
        attachment_id=uploaded.attachment_id,
        document_id=uploaded.document_id,
        document_type=uploaded.document_type,
        content_hash=uploaded.content_hash,
        mime=uploaded.mime,
        size_bytes=uploaded.size_bytes,
        dedupe_reused=uploaded.dedupe_reused,
        parsed=uploaded.parsed,
    )


def _extract_uploaded_document_id(outcomes: list[dict[str, Any]]) -> str | None:
    for item in outcomes:
        if str(item.get("outcome_type") or "") != "Documents.Upload":
            continue
        content = item.get("content")
        if not isinstance(content, dict):
            continue
        outcome = content.get("outcome")
        if not isinstance(outcome, dict):
            continue
        payload = outcome.get("payload")
        if not isinstance(payload, dict):
            continue
        document_id = payload.get("document_id")
        if isinstance(document_id, str) and document_id.strip():
            return document_id.strip()
    return None


@app.get("/v1/documents/{document_id}", response_model=DocumentMetadataResponse)
async def get_document_metadata(document_id: str, request: Request) -> DocumentMetadataResponse:
    ildb: ILDB = app.state.ildb
    auth_adapter: DevBypassAuthAdapter = app.state.auth_adapter

    metadata = await fetch_document_metadata(pool=ildb.pool, tenant_id=ildb.tenant_id, document_id=document_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="document not found")

    auth = await auth_adapter.authenticate(
        request=request,
        funding_request_id=int(metadata["funding_request_id"]),
        student_id_override=int(metadata["student_id"]),
    )
    if not auth.ok:
        raise HTTPException(status_code=auth.status_code, detail=auth.reason or "unauthorized")
    if int(auth.principal_id or 0) != int(metadata["student_id"]):
        raise HTTPException(status_code=403, detail="forbidden")

    revisions = await fetch_document_revisions(
        pool=ildb.pool,
        tenant_id=ildb.tenant_id,
        document_id=document_id,
    )
    revision_models: list[DocumentRevisionResponse] = []
    for revision in revisions:
        revision_models.append(
            DocumentRevisionResponse(
                revision_id=str(revision["revision_id"]),
                revision_no=int(revision["revision_no"]),
                revision_kind=str(revision["revision_kind"]),
                content_hash=str(revision["content_hash"]),
                processor_version=str(revision.get("processor_version") or "") or None,
                created_at=revision.get("created_at"),
                download_url=build_presigned_download_url(object_uri=str(revision.get("object_uri") or "")),
            )
        )

    return DocumentMetadataResponse(
        document_id=str(metadata["document_id"]),
        thread_id=int(metadata["thread_id"]),
        funding_request_id=int(metadata["funding_request_id"]),
        student_id=int(metadata["student_id"]),
        source_attachment_id=metadata.get("source_attachment_id"),
        document_type=str(metadata["document_type"]),
        lifecycle=str(metadata["lifecycle"]),
        mime=str(metadata["mime"]),
        content_hash=str(metadata["content_hash"]),
        content_size_bytes=int(metadata["content_size_bytes"]),
        source_object_uri=str(metadata["source_object_uri"]),
        current_revision_id=str(metadata["current_revision_id"]) if metadata.get("current_revision_id") else None,
        created_at=metadata.get("created_at"),
        updated_at=metadata.get("updated_at"),
        download_url=build_presigned_download_url(object_uri=str(metadata["source_object_uri"])),
        revisions=revision_models,
    )


@app.get("/v1/documents/{document_id}/download")
async def get_document_download_url(
    document_id: str,
    request: Request,
    revision_id: str | None = Query(default=None),
    expires_sec: int = Query(default=900, ge=60, le=86400),
) -> dict[str, Any]:
    ildb: ILDB = app.state.ildb
    auth_adapter: DevBypassAuthAdapter = app.state.auth_adapter

    metadata = await fetch_document_metadata(pool=ildb.pool, tenant_id=ildb.tenant_id, document_id=document_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="document not found")

    auth = await auth_adapter.authenticate(
        request=request,
        funding_request_id=int(metadata["funding_request_id"]),
        student_id_override=int(metadata["student_id"]),
    )
    if not auth.ok:
        raise HTTPException(status_code=auth.status_code, detail=auth.reason or "unauthorized")
    if int(auth.principal_id or 0) != int(metadata["student_id"]):
        raise HTTPException(status_code=403, detail="forbidden")

    object_uri = str(metadata["source_object_uri"] or "")
    resolved_revision_id: str | None = None
    if revision_id:
        revisions = await fetch_document_revisions(
            pool=ildb.pool,
            tenant_id=ildb.tenant_id,
            document_id=document_id,
        )
        matched = next((item for item in revisions if str(item.get("revision_id")) == revision_id), None)
        if matched is None:
            raise HTTPException(status_code=404, detail="revision not found")
        object_uri = str(matched.get("object_uri") or object_uri)
        resolved_revision_id = str(matched.get("revision_id"))

    download_url = build_presigned_download_url(object_uri=object_uri, expires_sec=expires_sec)
    if not download_url:
        raise HTTPException(status_code=422, detail="download URL is unavailable for this document")

    return {
        "document_id": document_id,
        "revision_id": resolved_revision_id,
        "expires_sec": expires_sec,
        "download_url": download_url,
    }


@app.post("/v1/threads/{thread_id}/queries", response_model=SubmitQueryResponse)
async def submit_query(thread_id: str, req: SubmitQueryRequest, request: Request) -> SubmitQueryResponse:
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

    auth_adapter: DevBypassAuthAdapter = app.state.auth_adapter

    if settings.use_workflow_kernel:
        kernel: WorkflowKernel = app.state.workflow_kernel
        event_writer: EventWriter = app.state.event_writer
        workflow_id_override: uuid.UUID | None = None

        if req.query_id is not None:
            try:
                query_uuid = uuid.UUID(req.query_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="query_id must be a UUID")

            existing = await ildb.get_workflow_run(workflow_id=str(query_uuid))
            if existing is not None:
                return SubmitQueryResponse(query_id=str(query_uuid), sse_url=f"/v1/workflows/{query_uuid}/events")
            workflow_id_override = query_uuid

        workflow_id = workflow_id_override or uuid.uuid4()
        preflight_correlation_id = uuid.uuid4()

        auth = await auth_adapter.authenticate(
            request=request,
            funding_request_id=int(thread["funding_request_id"]),
            student_id_override=int(thread["student_id"]),
        )
        if not auth.ok:
            await _emit_progress_event(
                event_writer=event_writer,
                tenant_id=ildb.tenant_id,
                workflow_id=workflow_id,
                actor={"role": "system", "type": "system", "id": "request"},
                stage="query_received",
                correlation_id=preflight_correlation_id,
                thread_id=thread_id_int,
            )
            await _emit_progress_event(
                event_writer=event_writer,
                tenant_id=ildb.tenant_id,
                workflow_id=workflow_id,
                actor={"role": "system", "type": "system", "id": "auth"},
                stage="checking_auth",
                correlation_id=preflight_correlation_id,
                thread_id=thread_id_int,
            )
            await _emit_final_error_event(
                event_writer=event_writer,
                tenant_id=ildb.tenant_id,
                workflow_id=workflow_id,
                actor={"role": "system", "type": "system", "id": "auth"},
                message=auth.reason or "unauthorized",
                correlation_id=preflight_correlation_id,
                thread_id=thread_id_int,
            )
            return SubmitQueryResponse(
                query_id=str(workflow_id),
                sse_url=f"/v1/workflows/{workflow_id}/events",
            )

        principal_id = int(auth.principal_id or 0)
        if principal_id != int(thread["student_id"]):
            await _emit_final_error_event(
                event_writer=event_writer,
                tenant_id=ildb.tenant_id,
                workflow_id=workflow_id,
                actor={"role": "system", "type": "system", "id": "auth"},
                message="forbidden",
                correlation_id=preflight_correlation_id,
                thread_id=thread_id_int,
            )
            return SubmitQueryResponse(
                query_id=str(workflow_id),
                sse_url=f"/v1/workflows/{workflow_id}/events",
            )

        actor = {
            "tenant_id": ildb.tenant_id,
            "principal": {"type": "student", "id": principal_id},
            "role": "student",
            "trust_level": auth.trust_level,
            "scopes": list(auth.scopes),
        }

        pending_gate = await ildb.get_latest_waiting_profile_gate(thread_id=thread_id_int)
        if pending_gate is not None:
            gate_id = str(pending_gate["gate_id"])
            gate_workflow_id = str(pending_gate["workflow_id"])
            gate_preview = pending_gate.get("preview")
            profile_payload = _extract_profile_gate_payload(req.message, gate_preview if isinstance(gate_preview, dict) else None)
            await kernel.resolve_action(
                action_id=gate_id,
                status="accepted",
                payload=profile_payload,
                actor=actor,
                source="chat",
            )
            return SubmitQueryResponse(
                query_id=gate_workflow_id,
                sse_url=f"/v1/workflows/{gate_workflow_id}/events",
            )

        credit_manager: CreditManager = app.state.credit_manager
        request_key = request_key_for_workflow(workflow_id)
        reserve_amount = await credit_manager.estimate_reserve_credits(req.message)
        await _emit_progress_event(
            event_writer=event_writer,
            tenant_id=ildb.tenant_id,
            workflow_id=workflow_id,
            actor={"role": "system", "type": "system", "id": "request"},
            stage="query_received",
            correlation_id=preflight_correlation_id,
            thread_id=thread_id_int,
        )
        await _emit_progress_event(
            event_writer=event_writer,
            tenant_id=ildb.tenant_id,
            workflow_id=workflow_id,
            actor=actor,
            stage="checking_auth",
            correlation_id=preflight_correlation_id,
            thread_id=thread_id_int,
        )
        await _emit_progress_event(
            event_writer=event_writer,
            tenant_id=ildb.tenant_id,
            workflow_id=workflow_id,
            actor=actor,
            stage="reserving_credits",
            correlation_id=preflight_correlation_id,
            thread_id=thread_id_int,
            detail={"reserve_credits": reserve_amount},
        )
        reservation = await credit_manager.reserve(
            principal_id=principal_id,
            workflow_id=workflow_id,
            request_key=request_key,
            estimated_credits=reserve_amount,
        )
        if not reservation.ok:
            await _emit_final_error_event(
                event_writer=event_writer,
                tenant_id=ildb.tenant_id,
                workflow_id=workflow_id,
                actor=actor,
                message=reservation.reason or "insufficient_credits",
                correlation_id=preflight_correlation_id,
                thread_id=thread_id_int,
            )
            return SubmitQueryResponse(
                query_id=str(workflow_id),
                sse_url=f"/v1/workflows/{workflow_id}/events",
            )
        try:
            result = await kernel.handle_message(
                thread_id=thread_id_int,
                scope_type="funding_request",
                scope_id=str(thread["funding_request_id"]),
                actor=actor,
                message=req.message,
                attachments=extract_attachment_ids(req.attachments),
                source="chat",
                workflow_id=workflow_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if result.status != "waiting":
            await credit_manager.settle(
                principal_id=principal_id,
                workflow_id=workflow_id,
                request_key=request_key,
            )

        return SubmitQueryResponse(
            query_id=str(result.workflow_id),
            sse_url=f"/v1/workflows/{result.workflow_id}/events",
        )

    auth = await auth_adapter.authenticate(
        request=request,
        funding_request_id=int(thread["funding_request_id"]),
        student_id_override=int(thread["student_id"]),
    )
    if not auth.ok:
        raise HTTPException(status_code=auth.status_code, detail=auth.reason or "unauthorized")
    principal_id = int(auth.principal_id or 0)
    if principal_id != int(thread["student_id"]):
        raise HTTPException(status_code=403, detail="forbidden")

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


@app.get("/v1/workflows/{workflow_id}/events")
async def workflow_events(workflow_id: str) -> StreamingResponse:
    projector: SSEProjector = app.state.sse_projector

    async def gen() -> AsyncIterator[str]:
        async for chunk in projector.stream(workflow_id=workflow_id):
            yield chunk

    return StreamingResponse(gen(), media_type="text/event-stream")


class HelloWorkflowRequest(BaseModel):
    student_id: int = Field(..., ge=1)
    funding_request_id: int = Field(..., ge=1)


class HelloWorkflowResponse(BaseModel):
    workflow_id: str
    thread_id: str
    intent_id: str
    plan_id: str


@app.post("/v1/debug/workflows/hello", response_model=HelloWorkflowResponse)
async def debug_workflow_hello(req: HelloWorkflowRequest) -> HelloWorkflowResponse:
    settings = get_settings()
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not found")

    ildb: ILDB = app.state.ildb
    pool = ildb.pool
    thread_id, _status, _is_new = await ildb.get_or_create_thread(
        student_id=req.student_id,
        funding_request_id=req.funding_request_id,
    )

    registry = ContractRegistry()
    plan_template = registry.get_plan_template("Thread.Init")

    workflow_id = uuid.uuid4()
    correlation_id = uuid.uuid4()
    intent_id = uuid.uuid4()
    plan_id = uuid.uuid4()

    actor = {"type": "student", "id": str(req.student_id), "role": "student"}
    intent_inputs = {
        "student_id": req.student_id,
        "funding_request_id": req.funding_request_id,
        "client_context": {},
    }

    import json
    from blake3 import blake3

    plan_hash = blake3(json.dumps(plan_template, sort_keys=True, separators=(",", ":")).encode("utf-8")).digest()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO runtime.workflow_runs (
              tenant_id, workflow_id, correlation_id, thread_id, intent_id, plan_id,
              status, execution_mode, replay_mode, created_at, started_at, completed_at, updated_at
            ) VALUES (
              $1,$2,$3,$4,$5,$6,$7,$8,$9,now(),now(),now(),now()
            );
            """,
            ildb.tenant_id,
            workflow_id,
            correlation_id,
            thread_id,
            intent_id,
            plan_id,
            "completed",
            "draft_only",
            "reproduce",
        )
        await conn.execute(
            """
            INSERT INTO ledger.intents (
              tenant_id, intent_id, intent_type, schema_version, source,
              thread_id, actor, inputs, constraints, context_refs, data_classes,
              correlation_id, producer_kind, producer_name, producer_version
            ) VALUES (
              $1,$2,$3,$4,$5,
              $6,$7,$8,$9,$10,$11,
              $12,$13,$14,$15
            );
            """,
            ildb.tenant_id,
            intent_id,
            "Thread.Init",
            "1.0",
            "api",
            thread_id,
            json.dumps(actor),
            json.dumps(intent_inputs),
            json.dumps({}),
            json.dumps({}),
            [],
            correlation_id,
            "adapter",
            "debug_hello_workflow",
            "1.0",
        )
        await conn.execute(
            """
            INSERT INTO ledger.plans (
              tenant_id, plan_id, intent_id, schema_version, planner_name, planner_version,
              plan, plan_hash
            ) VALUES (
              $1,$2,$3,$4,$5,$6,$7,$8
            );
            """,
            ildb.tenant_id,
            plan_id,
            intent_id,
            "1.0",
            "debug",
            "0.1",
            json.dumps(plan_template),
            plan_hash,
        )

    writer = EventWriter(pool=pool)
    await writer.append(
        LedgerEvent(
            tenant_id=ildb.tenant_id,
            event_id=uuid.uuid4(),
            workflow_id=workflow_id,
            thread_id=thread_id,
            intent_id=intent_id,
            plan_id=plan_id,
            event_type="progress",
            actor=actor,
            payload={"stage": "hello_started", "detail": "debug"},
            correlation_id=correlation_id,
            producer_kind="adapter",
            producer_name="debug_hello_workflow",
            producer_version="1.0",
        )
    )
    await writer.append(
        LedgerEvent(
            tenant_id=ildb.tenant_id,
            event_id=uuid.uuid4(),
            workflow_id=workflow_id,
            thread_id=thread_id,
            intent_id=intent_id,
            plan_id=plan_id,
            event_type="final_result",
            actor=actor,
            payload={"message": "hello from ledger", "timestamp": datetime.now(timezone.utc).isoformat()},
            correlation_id=correlation_id,
            producer_kind="adapter",
            producer_name="debug_hello_workflow",
            producer_version="1.0",
        )
    )

    return HelloWorkflowResponse(
        workflow_id=str(workflow_id),
        thread_id=str(thread_id),
        intent_id=str(intent_id),
        plan_id=str(plan_id),
    )


class DebugExecuteIntentRequest(BaseModel):
    thread_id: int = Field(..., ge=1)
    intent_type: str | None = None
    message: str | None = None
    inputs: dict[str, Any] | None = None


class DebugExecuteIntentResponse(BaseModel):
    workflow_id: str
    intent_id: str
    plan_id: str
    status: str
    gate_id: str | None = None
    sse_url: str


@app.post("/v1/debug/workflows/execute", response_model=DebugExecuteIntentResponse)
async def debug_execute_intent(req: DebugExecuteIntentRequest) -> DebugExecuteIntentResponse:
    settings = get_settings()
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not found")

    ildb: ILDB = app.state.ildb
    kernel: WorkflowKernel = app.state.workflow_kernel

    thread = await ildb.get_thread(thread_id=req.thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="thread not found")

    actor = {
        "tenant_id": ildb.tenant_id,
        "principal": {"type": "student", "id": int(thread["student_id"])},
        "role": "student",
        "trust_level": 0,
        "scopes": ["debug"],
    }
    scope_type = "funding_request"
    scope_id = str(thread["funding_request_id"])

    try:
        if req.intent_type:
            result = await kernel.start_intent(
                intent_type=req.intent_type,
                inputs=req.inputs or {},
                thread_id=req.thread_id,
                scope_type=scope_type,
                scope_id=scope_id,
                actor=actor,
                source="api",
            )
        else:
            if not req.message:
                raise HTTPException(status_code=400, detail="message is required when intent_type is not provided")
            result = await kernel.handle_message(
                thread_id=req.thread_id,
                scope_type=scope_type,
                scope_id=scope_id,
                actor=actor,
                message=req.message,
                source="chat",
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DebugExecuteIntentResponse(
        workflow_id=str(result.workflow_id),
        intent_id=str(result.intent_id),
        plan_id=str(result.plan_id),
        status=result.status,
        gate_id=str(result.gate_id) if result.gate_id else None,
        sse_url=f"/v1/workflows/{result.workflow_id}/events",
    )


class DebugOperatorThreadInitRequest(BaseModel):
    student_id: int = Field(..., ge=1)
    funding_request_id: int = Field(..., ge=1)
    idempotency_key: str | None = None


@app.post("/v1/debug/operators/thread_create_or_load")
async def debug_operator_thread_create(req: DebugOperatorThreadInitRequest) -> dict[str, Any]:
    settings = get_settings()
    if not settings.debug:
        raise HTTPException(status_code=404, detail="not found")

    ildb: ILDB = app.state.ildb
    executor: OperatorExecutor = app.state.operator_executor

    thread_id, _status, _is_new = await ildb.get_or_create_thread(
        student_id=req.student_id,
        funding_request_id=req.funding_request_id,
    )

    id_key = req.idempotency_key or f"thread_init:{ildb.tenant_id}:{req.student_id}:{req.funding_request_id}"
    workflow_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())
    call = OperatorCall(
        payload={
            "student_id": req.student_id,
            "funding_request_id": req.funding_request_id,
            "client_context": {},
        },
        idempotency_key=id_key,
        auth_context=AuthContext(
            tenant_id=ildb.tenant_id,
            principal={"type": "student", "id": req.student_id},
            scopes=["debug"],
        ),
        trace_context=OperatorTraceContext(
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            step_id="s1",
        ),
    )

    result = await executor.execute(
        operator_name="Thread.CreateOrLoad",
        operator_version="1.0.0",
        call=call,
    )
    payload = result.to_dict()
    payload["thread_id"] = thread_id
    payload["idempotency_key"] = id_key
    return payload


class ResolveActionRequest(BaseModel):
    status: str = Field(..., pattern="^(accepted|declined)$")
    payload: dict[str, Any] = Field(default_factory=dict)


@app.post("/v1/actions/{action_id}/resolve")
async def resolve_action(action_id: str, req: ResolveActionRequest) -> dict[str, Any]:
    kernel: WorkflowKernel | None = getattr(app.state, "workflow_kernel", None)
    if kernel is not None:
        ildb: ILDB = app.state.ildb
        credit_manager: CreditManager = app.state.credit_manager
        actor = {
            "tenant_id": getattr(app.state, "ildb").tenant_id,
            "principal": {"type": "system", "id": "action_resolver"},
            "role": "system",
            "trust_level": 0,
            "scopes": ["resolve_action"],
        }
        await kernel.resolve_action(
            action_id=action_id,
            status=req.status,
            payload=req.payload,
            actor=actor,
            source="api",
        )
        gate = await ildb.get_gate(gate_id=action_id)
        if gate and gate.get("workflow_id"):
            run = await ildb.get_workflow_run(workflow_id=str(gate["workflow_id"]))
            if run and str(run.get("status")) in {"completed", "failed", "cancelled"}:
                thread_id = run.get("thread_id")
                if thread_id is not None:
                    thread = await ildb.get_thread(thread_id=int(thread_id))
                    if thread:
                        principal_id = int(thread["student_id"])
                        await credit_manager.settle(
                            principal_id=principal_id,
                            workflow_id=uuid.UUID(str(gate["workflow_id"])),
                            request_key=request_key_for_workflow(uuid.UUID(str(gate["workflow_id"]))),
                        )
        return {"ok": True, "action_id": action_id, "status": req.status}

    kc: KernelContainer = app.state.kernel_container
    # v0 fallback: treat accepted/declined as a generic resolution payload
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
