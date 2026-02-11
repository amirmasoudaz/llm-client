from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from blake3 import blake3

from intelligence_layer_api.documents import ingest_thread_document

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .documents_common import (
    fetch_attachment_bytes,
    normalize_requested_document_type,
    read_cached_bytes,
    remove_cached_file,
    resolve_thread_scope,
)


class DocumentsUploadOperator(Operator):
    name = "Documents.Upload"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        payload = call.payload

        thread_id = _safe_int(payload.get("thread_id"))
        if thread_id is None:
            return _failed(start, "invalid_thread_id", "thread_id is required")
        student_id = _safe_int(payload.get("student_id"))
        if student_id is None:
            return _failed(start, "invalid_student_id", "student_id is required")
        document_type = normalize_requested_document_type(payload.get("document_type"))
        lifecycle = str(payload.get("lifecycle") or "sandbox").strip().lower()
        if lifecycle not in {"temp", "sandbox", "final"}:
            return _failed(start, "invalid_lifecycle", "lifecycle must be temp, sandbox, or final")

        artifact = payload.get("artifact")
        if not isinstance(artifact, dict):
            return _failed(start, "invalid_artifact", "artifact is required")

        try:
            scope = await resolve_thread_scope(pool=self._pool, tenant_id=self._tenant_id, thread_id=thread_id)
        except Exception as exc:
            return _failed(start, "thread_lookup_failed", str(exc), category="dependency")
        if scope.student_id != student_id:
            return _failed(start, "forbidden", "student does not own this thread", category="auth")

        fetched = fetch_attachment_bytes(_artifact_to_fetch_input(artifact))
        stream_path = str(fetched.get("stream_path") or "")
        file_bytes = read_cached_bytes(stream_path)
        if not isinstance(file_bytes, bytes):
            remove_cached_file(stream_path)
            return _failed(start, "artifact_unavailable", "unable to fetch uploaded artifact", category="dependency")

        expected_hash = _extract_artifact_hash(artifact)
        actual_hash = str(fetched.get("hash_hex") or "")
        if expected_hash and actual_hash and expected_hash != actual_hash:
            remove_cached_file(stream_path)
            return _failed(start, "artifact_hash_mismatch", "artifact hash does not match uploaded content")

        file_name = str(artifact.get("name") or "document.bin")
        mime = str(artifact.get("mime") or fetched.get("mime") or "application/octet-stream")
        title = str(payload.get("title") or "").strip() or None

        try:
            uploaded = await ingest_thread_document(
                pool=self._pool,
                tenant_id=self._tenant_id,
                thread_id=thread_id,
                student_id=student_id,
                funding_request_id=scope.funding_request_id,
                file_bytes=file_bytes,
                file_name=file_name,
                content_type=mime,
                document_type_hint=document_type,
                title=title,
                lifecycle=lifecycle,
            )
        except ValueError as exc:
            remove_cached_file(stream_path)
            return _failed(start, "document_upload_validation_failed", str(exc))
        except Exception as exc:
            remove_cached_file(stream_path)
            return _failed(start, "document_upload_failed", str(exc), category="dependency")
        remove_cached_file(stream_path)

        outcome = _build_document_uploaded_outcome(
            document_id=uploaded.document_id,
            thread_id=thread_id,
            student_id=student_id,
            document_type=uploaded.document_type,
            title=title or file_name,
            lifecycle=lifecycle,
            object_uri=uploaded.source_object_uri,
            hash_hex=uploaded.content_hash,
            size_bytes=uploaded.size_bytes,
            mime=uploaded.mime,
        )
        return OperatorResult(
            status="succeeded",
            result={"outcome": outcome},
            artifacts=[
                {
                    "object_uri": uploaded.source_object_uri,
                    "hash": {"alg": "blake3", "value": uploaded.content_hash},
                    "mime": uploaded.mime,
                    "name": file_name,
                    "size_bytes": uploaded.size_bytes,
                }
            ],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _artifact_to_fetch_input(artifact: dict[str, Any]) -> dict[str, Any]:
    object_uri = str(artifact.get("object_uri") or "")
    disk = "s3" if object_uri.startswith("s3://") else "local"
    file_path = object_uri
    if object_uri.startswith("s3://"):
        stripped = object_uri[5:]
        if "/" in stripped:
            _, key = stripped.split("/", 1)
            file_path = key
    return {
        "object_uri": object_uri,
        "disk": disk,
        "file_path": file_path,
        "mime": artifact.get("mime"),
    }


def _extract_artifact_hash(artifact: dict[str, Any]) -> str:
    hash_obj = artifact.get("hash")
    if not isinstance(hash_obj, dict):
        return ""
    return str(hash_obj.get("value") or "").strip().lower()


def _build_document_uploaded_outcome(
    *,
    document_id: str,
    thread_id: int,
    student_id: int,
    document_type: str,
    title: str,
    lifecycle: str,
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
        "lifecycle": lifecycle,
        "artifact": {
            "object_uri": object_uri,
            "hash": {"alg": "blake3", "value": hash_hex},
            "mime": mime,
            "size_bytes": size_bytes,
        },
    }
    if title.strip():
        payload["title"] = title.strip()

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
            "name": "Documents.Upload",
            "version": "1.0.0",
            "plugin_type": "operator",
        },
    }


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
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
