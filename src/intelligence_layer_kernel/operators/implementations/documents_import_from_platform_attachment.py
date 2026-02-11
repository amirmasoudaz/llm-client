from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .documents_common import (
    compatible_mime,
    extract_attachment_ids,
    fetch_attachment_bytes,
    infer_document_type,
    list_platform_attachments,
    normalize_requested_document_type,
    persist_streamed_attachment,
    remove_cached_file,
    requested_attachment_kinds,
    resolve_thread_scope,
)


class DocumentsImportFromPlatformAttachmentOperator(Operator):
    name = "Documents.ImportFromPlatformAttachment"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload
        thread_id_raw = payload.get("thread_id")
        if thread_id_raw is None:
            return _failed(start, "missing_thread_id", "thread_id is required")
        try:
            thread_id = int(thread_id_raw)
        except Exception:
            return _failed(start, "invalid_thread_id", "thread_id must be an integer")

        requested_document_type = normalize_requested_document_type(payload.get("document_type"))
        if payload.get("document_type") is None:
            requested_document_type = "cv"
        preferred_attachment_ids = extract_attachment_ids(payload.get("attachment_ids"))

        try:
            scope = await resolve_thread_scope(pool=self._pool, tenant_id=self._tenant_id, thread_id=thread_id)
            selected = _coerce_attachment(payload.get("selected_attachment"))
            attachments = _coerce_attachments(payload.get("attachments"))
            if selected is None:
                if attachments:
                    selected = attachments[0]
                else:
                    _, selected = await list_platform_attachments(
                        thread_id=thread_id,
                        funding_request_id=scope.funding_request_id,
                        student_id=scope.student_id,
                        requested_document_type=requested_document_type,
                        attachment_ids=preferred_attachment_ids,
                    )
        except Exception as exc:
            return _failed(start, "platform_attachment_lookup_failed", str(exc), category="dependency")

        if selected is None:
            return _failed(
                start,
                "missing_attachment",
                f"No {requested_document_type} document found. Please upload your document first.",
            )

        existing = await self._load_existing_document(
            thread_id=thread_id,
            source_attachment_id=_safe_int(selected.get("attachment_id")),
        )
        if existing is not None:
            outcome = _build_document_uploaded_outcome(
                document_id=str(existing["document_id"]),
                thread_id=thread_id,
                student_id=scope.student_id,
                document_type=str(existing["document_type"]),
                title=selected.get("name") or str(existing["document_type"]),
                object_uri=str(existing["source_object_uri"]),
                hash_hex=existing["content_hash_hex"],
                size_bytes=int(existing["content_size_bytes"]),
                mime=selected.get("mime"),
            )
            return OperatorResult(
                status="succeeded",
                result={
                    "outcome": outcome,
                    "selected_attachment": selected,
                    "fetch_status": "cached",
                    "requested_attachment_kinds": requested_attachment_kinds(requested_document_type),
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=None,
            )

        fetched = fetch_attachment_bytes(selected)
        fetch_status = str(fetched.get("status") or "unavailable")
        if fetch_status != "downloaded":
            if fetch_status == "blocked_mime":
                return _failed(
                    start,
                    "attachment_blocked_mime",
                    "attachment MIME type is not allowed for document review",
                    category="validation",
                )
            if fetch_status == "too_large":
                return _failed(
                    start,
                    "attachment_too_large",
                    "attachment exceeds maximum allowed size for document review",
                    category="validation",
                )
            return _failed(
                start,
                "attachment_unavailable",
                "unable to fetch attachment bytes; upload a readable file or configure attachment storage",
                category="dependency",
            )

        stream_path = str(fetched.get("stream_path") or "")
        if not stream_path:
            return _failed(
                start,
                "attachment_unavailable",
                "unable to fetch attachment bytes; upload a readable file or configure attachment storage",
                category="dependency",
            )

        content_hash_hex = str(fetched["hash_hex"])
        content_hash = bytes.fromhex(content_hash_hex)
        size_bytes = int(fetched["size_bytes"])
        mime = str(fetched.get("mime") or selected.get("mime") or "application/octet-stream")
        storage_path: str | None = None
        normalized_document_type = infer_document_type(
            document_type_hint=requested_document_type,
            mime=mime,
            file_name=str(selected.get("file_path") or selected.get("name") or ""),
        )

        dedupe = await self._load_existing_by_student_hash(
            student_id=scope.student_id,
            content_hash=content_hash,
            mime=mime,
        )
        if dedupe is not None:
            document_id = uuid.uuid4()
            source_object_uri = str(selected.get("object_uri") or dedupe["source_object_uri"])
            if not source_object_uri:
                source_object_uri = f"s3://local/{document_id}.bin"
            try:
                await self._insert_document(
                    document_id=document_id,
                    thread_id=thread_id,
                    funding_request_id=scope.funding_request_id,
                    student_id=scope.student_id,
                    source_attachment_id=_safe_int(selected.get("attachment_id")),
                    document_type=normalized_document_type,
                    lifecycle="sandbox",
                    source_disk=str(selected.get("disk") or "s3"),
                    source_path=str(selected.get("file_path") or ""),
                    source_object_uri=source_object_uri,
                    source_metadata=selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {},
                    content_hash=content_hash,
                    content_size_bytes=size_bytes,
                    mime=mime,
                    storage_path=dedupe.get("storage_path"),
                    extracted_text=str(dedupe.get("extracted_text") or ""),
                    extracted_fields=dedupe.get("extracted_fields") if isinstance(dedupe.get("extracted_fields"), dict) else {},
                )
            except Exception as exc:
                remove_cached_file(stream_path)
                return _failed(start, "document_import_failed", str(exc), category="dependency")

            outcome = _build_document_uploaded_outcome(
                document_id=str(document_id),
                thread_id=thread_id,
                student_id=scope.student_id,
                document_type=normalized_document_type,
                title=selected.get("name") or normalized_document_type,
                object_uri=source_object_uri,
                hash_hex=content_hash_hex,
                size_bytes=size_bytes,
                mime=mime,
            )
            remove_cached_file(stream_path)
            return OperatorResult(
                status="succeeded",
                result={
                    "outcome": outcome,
                    "selected_attachment": selected,
                    "fetch_status": "dedupe_reused",
                    "requested_attachment_kinds": requested_attachment_kinds(requested_document_type),
                },
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=None,
            )

        document_id = uuid.uuid4()
        storage_path = persist_streamed_attachment(str(document_id), stream_path)
        if storage_path is None:
            storage_path = stream_path

        source_object_uri = str(selected.get("object_uri") or "")
        if not source_object_uri:
            source_object_uri = f"s3://local/{document_id}.bin"

        try:
            await self._insert_document(
                document_id=document_id,
                thread_id=thread_id,
                funding_request_id=scope.funding_request_id,
                student_id=scope.student_id,
                source_attachment_id=_safe_int(selected.get("attachment_id")),
                document_type=normalized_document_type,
                lifecycle="sandbox",
                source_disk=str(selected.get("disk") or "s3"),
                source_path=str(selected.get("file_path") or ""),
                source_object_uri=source_object_uri,
                source_metadata=selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {},
                content_hash=content_hash,
                content_size_bytes=size_bytes,
                mime=mime,
                storage_path=storage_path,
                extracted_text=None,
                extracted_fields={},
            )
        except Exception as exc:
            remove_cached_file(storage_path)
            return _failed(start, "document_import_failed", str(exc), category="dependency")

        outcome = _build_document_uploaded_outcome(
            document_id=str(document_id),
            thread_id=thread_id,
            student_id=scope.student_id,
            document_type=normalized_document_type,
            title=selected.get("name") or normalized_document_type,
            object_uri=source_object_uri,
            hash_hex=content_hash_hex,
            size_bytes=size_bytes,
            mime=mime,
        )

        return OperatorResult(
            status="succeeded",
            result={
                "outcome": outcome,
                "selected_attachment": selected,
                "fetch_status": fetched.get("status"),
                "requested_attachment_kinds": requested_attachment_kinds(requested_document_type),
            },
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )

    async def _load_existing_document(
        self,
        *,
        thread_id: int,
        source_attachment_id: int | None,
    ) -> dict[str, Any] | None:
        if source_attachment_id is None:
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT document_id,
                       document_type,
                       source_object_uri,
                       content_hash,
                       content_size_bytes,
                       storage_path
                FROM ledger.documents
                WHERE tenant_id=$1
                  AND thread_id=$2
                  AND source_attachment_id=$3
                LIMIT 1;
                """,
                self._tenant_id,
                thread_id,
                source_attachment_id,
            )
        if not row:
            return None
        content_size_bytes = int(row["content_size_bytes"] or 0)
        storage_path = row["storage_path"]
        if content_size_bytes <= 0 or not storage_path:
            return None
        content_hash: bytes = row["content_hash"]
        return {
            "document_id": row["document_id"],
            "document_type": row["document_type"],
            "source_object_uri": row["source_object_uri"],
            "content_hash_hex": content_hash.hex(),
            "content_size_bytes": content_size_bytes,
        }

    async def _load_existing_by_student_hash(
        self,
        *,
        student_id: int,
        content_hash: bytes,
        mime: str,
    ) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT document_id,
                       source_object_uri,
                       storage_path,
                       mime,
                       extracted_text,
                       extracted_fields
                FROM ledger.documents
                WHERE tenant_id=$1
                  AND student_id=$2
                  AND content_hash=$3
                  AND extracted_text IS NOT NULL
                  AND length(extracted_text) > 0
                ORDER BY updated_at DESC
                LIMIT 20;
                """,
                self._tenant_id,
                student_id,
                content_hash,
            )
        for row in rows:
            if compatible_mime(str(row["mime"] or ""), mime):
                return {
                    "document_id": row["document_id"],
                    "source_object_uri": row["source_object_uri"],
                    "storage_path": row["storage_path"],
                    "extracted_text": row["extracted_text"],
                    "extracted_fields": _coerce_json(row["extracted_fields"]),
                }
        return None

    async def _insert_document(
        self,
        *,
        document_id: uuid.UUID,
        thread_id: int,
        funding_request_id: int,
        student_id: int,
        source_attachment_id: int | None,
        document_type: str,
        lifecycle: str,
        source_disk: str,
        source_path: str,
        source_object_uri: str,
        source_metadata: dict[str, Any],
        content_hash: bytes,
        content_size_bytes: int,
        mime: str,
        storage_path: str | None,
        extracted_text: str | None,
        extracted_fields: dict[str, Any],
    ) -> None:
        async with self._pool.acquire() as conn:
            revision_id = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO ledger.documents (
                  tenant_id,
                  document_id,
                  thread_id,
                  funding_request_id,
                  student_id,
                  source_attachment_id,
                  document_type,
                  lifecycle,
                  source_disk,
                  source_path,
                  source_object_uri,
                  source_metadata,
                  content_hash,
                  content_size_bytes,
                  mime,
                  storage_path,
                  extracted_text,
                  extracted_fields,
                  current_revision_id
                ) VALUES (
                  $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$14,$15,$16,$17,$18::jsonb,$19
                );
                """,
                self._tenant_id,
                document_id,
                thread_id,
                funding_request_id,
                student_id,
                source_attachment_id,
                document_type,
                lifecycle,
                source_disk,
                source_path,
                source_object_uri,
                json.dumps(source_metadata),
                content_hash,
                content_size_bytes,
                mime,
                storage_path,
                extracted_text,
                json.dumps(extracted_fields or {}),
                revision_id,
            )
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
                  $1,$2,$3,1,$4,$5,$6::jsonb,$7,$8
                );
                """,
                self._tenant_id,
                revision_id,
                document_id,
                "imported_attachment",
                source_object_uri,
                json.dumps(
                    {
                        "document_type": document_type,
                        "mime": mime,
                        "content_size_bytes": content_size_bytes,
                        "source_metadata": source_metadata or {},
                    }
                ),
                content_hash,
                "documents_import_from_platform_attachment@1.0.0",
            )


def _build_document_uploaded_outcome(
    *,
    document_id: str,
    thread_id: int,
    student_id: int,
    document_type: str,
    title: Any,
    object_uri: str,
    hash_hex: str,
    size_bytes: int,
    mime: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "document_id": document_id,
        "student_id": student_id,
        "thread_id": thread_id,
        "document_type": document_type,
        "lifecycle": "sandbox",
        "artifact": {
            "object_uri": object_uri,
            "hash": {"alg": "blake3", "value": hash_hex},
            "mime": mime,
            "size_bytes": size_bytes,
        },
    }
    title_text = str(title or "").strip()
    if title_text:
        payload["title"] = title_text

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = blake3(raw).hexdigest()
    return {
        "schema_version": "1.0",
        "outcome_id": str(uuid.uuid4()),
        "outcome_type": "Document.Uploaded",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hash": {"alg": "blake3", "value": digest},
        "payload": payload,
        "producer": {
            "name": "Documents.ImportFromPlatformAttachment",
            "version": "1.0.0",
            "plugin_type": "operator",
        },
    }


def _coerce_attachments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        parsed = _coerce_attachment(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _coerce_attachment(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    file_path = str(value.get("file_path") or "").strip()
    if not file_path:
        return None
    attachment = dict(value)
    attachment["file_path"] = file_path
    return attachment


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def _failed(
    start: float,
    code: str,
    message: str,
    *,
    category: str = "validation",
) -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(code=code, message=message, category=category, retryable=False),
    )
