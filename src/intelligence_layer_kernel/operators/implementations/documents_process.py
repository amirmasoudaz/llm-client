from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .documents_common import extract_text_from_bytes, fetch_attachment_bytes, read_cached_bytes


class DocumentsProcessOperator(Operator):
    name = "Documents.Process"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        document_id_raw = payload.get("document_id")
        if not document_id_raw:
            return _failed(start, "missing_document_id", "document_id is required")
        try:
            document_id = uuid.UUID(str(document_id_raw))
        except Exception:
            return _failed(start, "invalid_document_id", "document_id must be a UUID")

        processing_profile = str(payload.get("processing_profile") or "default").strip() or "default"

        doc_row = await self._load_document(document_id=document_id)
        if doc_row is None:
            return _failed(start, "document_not_found", "document not found")

        existing_text = str(doc_row.get("extracted_text") or "").strip()
        existing_fields = _as_dict(doc_row.get("extracted_fields"))
        parser_strategy = str(existing_fields.get("parser_strategy") or "").strip()

        if existing_text and parser_strategy:
            text = existing_text
            page_count = existing_fields.get("page_count")
            strategy = parser_strategy
            extracted_fields = {
                key: value
                for key, value in existing_fields.items()
                if key not in {"parser_strategy", "page_count"}
            }
        else:
            cached_bytes = read_cached_bytes(doc_row.get("storage_path"))
            if not isinstance(cached_bytes, (bytes, bytearray)):
                fetched = fetch_attachment_bytes(
                    {
                        "object_uri": doc_row.get("source_object_uri"),
                        "file_path": doc_row.get("source_path") or doc_row.get("storage_path"),
                        "disk": doc_row.get("source_disk") or "s3",
                        "mime": doc_row.get("mime"),
                    }
                )
                if isinstance(fetched.get("bytes"), bytes):
                    cached_bytes = fetched["bytes"]
            text, page_count, strategy = extract_text_from_bytes(
                cached_bytes,
                mime=doc_row.get("mime"),
                file_name=doc_row.get("source_path"),
            )
            if not text:
                text = _fallback_text_from_metadata(doc_row.get("source_metadata"))
            extracted_fields = _extract_fields(
                text=text,
                document_type=str(doc_row.get("document_type") or "other"),
            )

        text_bytes = text.encode("utf-8")
        text_hash_hex = blake3(text_bytes).hexdigest()
        text_artifact = {
            "object_uri": _artifact_uri(doc_row.get("source_object_uri"), str(document_id)),
            "hash": {"alg": "blake3", "value": text_hash_hex},
            "mime": "text/plain",
            "name": f"{document_id}.txt",
            "size_bytes": len(text_bytes),
        }

        await self._update_document_processed(
            document_id=document_id,
            extracted_text=text,
            extracted_fields={**extracted_fields, "parser_strategy": strategy},
            page_count=page_count,
        )

        outcome_payload: dict[str, Any] = {
            "document_id": str(document_id),
            "processing_profile": processing_profile,
            "text_hash": {"alg": "blake3", "value": text_hash_hex},
            "text_artifact": text_artifact,
            "extracted_fields": extracted_fields,
        }
        if page_count is not None:
            outcome_payload["page_count"] = page_count
        outcome = _build_processed_outcome(outcome_payload)

        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )

    async def _load_document(self, *, document_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT document_id,
                       document_type,
                       source_disk,
                       source_path,
                       source_object_uri,
                       source_metadata,
                       mime,
                       storage_path,
                       extracted_text,
                       extracted_fields
                FROM ledger.documents
                WHERE tenant_id=$1 AND document_id=$2
                LIMIT 1;
                """,
                self._tenant_id,
                document_id,
            )
        if not row:
            return None
        return {
            "document_id": row["document_id"],
            "document_type": row["document_type"],
            "source_disk": row["source_disk"],
            "source_path": row["source_path"],
            "source_object_uri": row["source_object_uri"],
            "source_metadata": _coerce_json(row["source_metadata"]),
            "mime": row["mime"],
            "storage_path": row["storage_path"],
            "extracted_text": row["extracted_text"],
            "extracted_fields": _coerce_json(row["extracted_fields"]),
        }

    async def _update_document_processed(
        self,
        *,
        document_id: uuid.UUID,
        extracted_text: str,
        extracted_fields: dict[str, Any],
        page_count: int | None,
    ) -> None:
        fields_payload = dict(extracted_fields)
        if page_count is not None:
            fields_payload["page_count"] = page_count
        revision_id = uuid.uuid4()
        content_json = {
            "extracted_fields": fields_payload,
            "text_hash": blake3(extracted_text.encode("utf-8")).hexdigest(),
        }
        content_hash = blake3(
            json.dumps(content_json, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        async with self._pool.acquire() as conn:
            current_max = await conn.fetchval(
                """
                SELECT COALESCE(MAX(revision_no), 0)
                FROM ledger.document_revisions
                WHERE tenant_id=$1 AND document_id=$2;
                """,
                self._tenant_id,
                document_id,
            )
            next_revision = int(current_max or 0) + 1
            await conn.execute(
                """
                INSERT INTO ledger.document_revisions (
                  tenant_id,
                  revision_id,
                  document_id,
                  revision_no,
                  revision_kind,
                  object_uri,
                  content_json,
                  content_hash,
                  processor_version
                ) VALUES (
                  $1,$2,$3,$4,$5,$6,$7::jsonb,$8,$9
                );
                """,
                self._tenant_id,
                revision_id,
                document_id,
                next_revision,
                "processed",
                None,
                json.dumps(content_json),
                content_hash,
                "documents_process@1.0.0",
            )
            await conn.execute(
                """
                UPDATE ledger.documents
                SET extracted_text=$3,
                    extracted_fields=$4::jsonb,
                    current_revision_id=$5,
                    updated_at=now()
                WHERE tenant_id=$1 AND document_id=$2;
                """,
                self._tenant_id,
                document_id,
                extracted_text,
                json.dumps(fields_payload),
                revision_id,
            )


def _artifact_uri(source_object_uri: Any, document_id: str) -> str:
    uri = str(source_object_uri or "").strip()
    if uri.startswith("s3://"):
        return uri
    return f"s3://local-artifacts/{document_id}.txt"


def _extract_fields(*, text: str, document_type: str) -> dict[str, Any]:
    fields: dict[str, Any] = {"char_count": len(text), "word_count": len(text.split())}
    email_matches = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    if email_matches:
        fields["emails"] = sorted(set(email_matches))
    if document_type in {"resume", "cv"}:
        fields["has_education_section"] = "education" in text.lower()
        fields["has_experience_section"] = "experience" in text.lower()
        fields["has_skills_section"] = "skill" in text.lower()
    if document_type in {"sop", "letter", "study_plan"}:
        fields["mentions_research"] = "research" in text.lower()
        fields["mentions_program_fit"] = any(
            phrase in text.lower()
            for phrase in ("professor", "lab", "program", "supervisor", "fit")
        )
    return fields


def _fallback_text_from_metadata(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return ""
    for key in ("text", "summary", "description", "notes"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_processed_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Document.Processed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {"name": "Documents.Process", "version": "1.0.0", "plugin_type": "operator"},
    }


def _coerce_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _failed(start: float, code: str, message: str, *, category: str = "validation") -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(code=code, message=message, category=category, retryable=False),
    )
