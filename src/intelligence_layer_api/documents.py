from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from blake3 import blake3

from intelligence_layer_kernel.operators.implementations.documents_common import (
    ATTACHABLE_TYPE_THREAD,
    allowed_mime,
    build_source_object_uri,
    compatible_mime,
    extract_text_from_bytes,
    infer_document_type,
    max_attachment_bytes,
    normalize_upload_filename,
)
from intelligence_layer_kernel.operators.implementations.funding_request_fields_common import get_platform_db


@dataclass(frozen=True)
class IngestedThreadDocument:
    attachment_id: int
    document_id: str
    document_type: str
    content_hash: str
    mime: str
    size_bytes: int
    dedupe_reused: bool
    parsed: bool
    source_object_uri: str


def upload_storage_health() -> dict[str, Any]:
    mode = os.getenv("IL_UPLOAD_STORAGE_DISK", "s3").strip().lower() or "s3"
    details: dict[str, Any] = {"mode": mode}
    reasons: list[str] = []

    if mode == "s3":
        bucket = _resolve_upload_bucket()
        prefix = _resolve_upload_prefix()
        details["bucket"] = bucket
        details["prefix"] = prefix
        details["endpoint"] = os.getenv("PLATFORM_ATTACHMENTS_S3_ENDPOINT", "").strip() or None
        details["region"] = os.getenv("PLATFORM_ATTACHMENTS_S3_REGION", "").strip() or None
        details["access_key_configured"] = bool(
            os.getenv("PLATFORM_ATTACHMENTS_S3_ACCESS_KEY", "").strip()
        )
        details["secret_key_configured"] = bool(
            os.getenv("PLATFORM_ATTACHMENTS_S3_SECRET_KEY", "").strip()
        )
        try:
            import boto3  # type: ignore
            details["boto3_available"] = True
            try:
                session = boto3.Session()
                creds = session.get_credentials()
                if creds is None:
                    details["credentials_resolved"] = False
                    details["credential_provider"] = None
                    reasons.append("missing_credentials")
                else:
                    frozen = creds.get_frozen_credentials()
                    resolved = bool(frozen.access_key and frozen.secret_key)
                    details["credentials_resolved"] = resolved
                    details["credential_provider"] = getattr(creds, "method", None)
                    if not resolved:
                        reasons.append("missing_credentials")
            except Exception:
                details["credentials_resolved"] = False
                details["credential_provider"] = None
                reasons.append("credentials_check_failed")
        except Exception:
            details["boto3_available"] = False
            reasons.append("boto3_not_installed")
        if not bucket:
            reasons.append("missing_bucket")
    else:
        local_dir = Path(os.getenv("IL_UPLOAD_STORAGE_DIR", "/tmp/intelligence_layer_uploads")).expanduser()
        details["local_dir"] = str(local_dir)
        details["writable"] = os.access(local_dir.parent if local_dir.parent else local_dir, os.W_OK)

    details["ready"] = len(reasons) == 0
    details["reasons"] = reasons
    return details


async def ingest_thread_document(
    *,
    pool,
    tenant_id: int,
    thread_id: int,
    student_id: int,
    funding_request_id: int,
    file_bytes: bytes,
    file_name: str,
    content_type: str | None,
    document_type_hint: str | None,
    title: str | None,
    lifecycle: str | None = None,
) -> IngestedThreadDocument:
    if not file_bytes:
        raise ValueError("uploaded file is empty")

    if len(file_bytes) > max_attachment_bytes():
        raise ValueError("uploaded file exceeds maximum allowed size")

    normalized_name = normalize_upload_filename(file_name)
    mime = str(content_type or "").strip().lower() or "application/octet-stream"
    if not allowed_mime(mime):
        raise ValueError("uploaded file MIME type is not allowed")

    preview_text, _, _ = extract_text_from_bytes(file_bytes, mime=mime, file_name=normalized_name)
    document_type = infer_document_type(
        document_type_hint=document_type_hint,
        mime=mime,
        file_name=normalized_name,
        text=preview_text,
    )

    content_hash_bytes = blake3(file_bytes).digest()
    content_hash_hex = content_hash_bytes.hex()
    lifecycle = _resolve_upload_lifecycle(lifecycle)

    existing_attachment = await _find_existing_thread_attachment_by_hash(
        student_id=student_id,
        thread_id=thread_id,
        content_hash=content_hash_hex,
        mime=mime,
        document_type=document_type,
    )
    if existing_attachment is not None:
        attachment_id = int(existing_attachment["attachment_id"])
        storage_abs = str(existing_attachment["file_path"] or "")
        storage_disk = str(existing_attachment["disk"] or "s3")
        storage_uri = build_source_object_uri(disk=storage_disk, file_path=storage_abs)
        await _backfill_platform_attachment_columns(
            attachment_id=attachment_id,
            file_name=normalized_name,
            content_hash=content_hash_hex,
            size_bytes=len(file_bytes),
        )
    else:
        storage_abs, storage_uri, storage_disk = _store_uploaded_bytes(
            tenant_id=tenant_id,
            thread_id=thread_id,
            student_id=student_id,
            lifecycle=lifecycle,
            file_name=normalized_name,
            data=file_bytes,
        )

        attachment_id = await _insert_platform_attachment(
            student_id=student_id,
            thread_id=thread_id,
            funding_request_id=funding_request_id,
            file_name=normalized_name,
            file_path=storage_abs,
            disk=storage_disk,
            mime=mime,
            document_type=document_type,
            lifecycle=lifecycle,
            content_hash=content_hash_hex,
            title=title,
            size_bytes=len(file_bytes),
        )

    existing_document = await _find_existing_document_for_attachment(
        pool=pool,
        tenant_id=tenant_id,
        thread_id=thread_id,
        source_attachment_id=attachment_id,
    )
    if existing_document is not None:
        return IngestedThreadDocument(
            attachment_id=attachment_id,
            document_id=str(existing_document["document_id"]),
            document_type=str(existing_document.get("document_type") or document_type),
            content_hash=content_hash_hex,
            mime=str(existing_document.get("mime") or mime),
            size_bytes=int(existing_document.get("content_size_bytes") or len(file_bytes)),
            dedupe_reused=True,
            parsed=bool(str(existing_document.get("extracted_text") or "").strip()),
            source_object_uri=str(existing_document.get("source_object_uri") or storage_uri),
        )

    dedupe = await _find_student_hash_match(
        pool=pool,
        tenant_id=tenant_id,
        student_id=student_id,
        content_hash=content_hash_bytes,
        mime=mime,
    )
    document_id = uuid.uuid4()

    if dedupe is not None:
        await _insert_ledger_document(
            pool=pool,
            tenant_id=tenant_id,
            document_id=document_id,
            thread_id=thread_id,
            funding_request_id=funding_request_id,
            student_id=student_id,
            source_attachment_id=attachment_id,
            document_type=document_type,
            lifecycle=lifecycle,
            source_disk=storage_disk,
            source_path=storage_abs,
            source_object_uri=storage_uri,
            source_metadata={
                "source": "thread_upload",
                "lifecycle": lifecycle,
                "dedupe_reused": True,
                "parsed": True,
            },
            content_hash=content_hash_bytes,
            content_size_bytes=len(file_bytes),
            mime=mime,
            storage_path=dedupe.get("storage_path") or storage_abs,
            extracted_text=str(dedupe.get("extracted_text") or ""),
            extracted_fields=dedupe.get("extracted_fields") if isinstance(dedupe.get("extracted_fields"), dict) else {},
        )
        return IngestedThreadDocument(
            attachment_id=attachment_id,
            document_id=str(document_id),
            document_type=document_type,
            content_hash=content_hash_hex,
            mime=mime,
            size_bytes=len(file_bytes),
            dedupe_reused=True,
            parsed=True,
            source_object_uri=storage_uri,
        )

    text, page_count, parser_strategy = extract_text_from_bytes(file_bytes, mime=mime, file_name=normalized_name)
    extracted_fields = _extract_fields(text=text, document_type=document_type)
    extracted_fields["parser_strategy"] = parser_strategy
    if page_count is not None:
        extracted_fields["page_count"] = page_count

    await _insert_ledger_document(
        pool=pool,
        tenant_id=tenant_id,
        document_id=document_id,
        thread_id=thread_id,
        funding_request_id=funding_request_id,
        student_id=student_id,
        source_attachment_id=attachment_id,
        document_type=document_type,
        lifecycle=lifecycle,
        source_disk=storage_disk,
        source_path=storage_abs,
        source_object_uri=storage_uri,
        source_metadata={
            "source": "thread_upload",
            "lifecycle": lifecycle,
            "dedupe_reused": False,
            "parsed": bool(text.strip()),
        },
        content_hash=content_hash_bytes,
        content_size_bytes=len(file_bytes),
        mime=mime,
        storage_path=storage_abs,
        extracted_text=text,
        extracted_fields=extracted_fields,
    )
    return IngestedThreadDocument(
        attachment_id=attachment_id,
        document_id=str(document_id),
        document_type=document_type,
        content_hash=content_hash_hex,
        mime=mime,
        size_bytes=len(file_bytes),
        dedupe_reused=False,
        parsed=bool(text.strip()),
        source_object_uri=storage_uri,
    )


def stage_upload_artifact(
    *,
    tenant_id: int,
    thread_id: int,
    student_id: int,
    file_bytes: bytes,
    file_name: str,
    content_type: str | None,
) -> dict[str, Any]:
    if not file_bytes:
        raise ValueError("uploaded file is empty")
    if len(file_bytes) > max_attachment_bytes():
        raise ValueError("uploaded file exceeds maximum allowed size")

    normalized_name = normalize_upload_filename(file_name)
    mime = str(content_type or "").strip().lower() or "application/octet-stream"
    if not allowed_mime(mime):
        raise ValueError("uploaded file MIME type is not allowed")

    storage_abs, source_uri, _ = _store_uploaded_bytes(
        tenant_id=tenant_id,
        thread_id=thread_id,
        student_id=student_id,
        lifecycle="temp",
        file_name=normalized_name,
        data=file_bytes,
    )
    content_hash_hex = blake3(file_bytes).hexdigest()
    return {
        "object_uri": source_uri,
        "hash": {"alg": "blake3", "value": content_hash_hex},
        "mime": mime,
        "name": normalized_name,
        "size_bytes": len(file_bytes),
        "storage_path": storage_abs,
    }


def _store_uploaded_bytes(
    *,
    tenant_id: int,
    thread_id: int,
    student_id: int,
    lifecycle: str,
    file_name: str,
    data: bytes,
) -> tuple[str, str, str]:
    source_disk = os.getenv("IL_UPLOAD_STORAGE_DISK", "s3").strip() or "s3"
    doc_id = uuid.uuid4()
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    rel_path = (
        Path("layer")
        / str(tenant_id)
        / str(student_id)
        / lifecycle
        / str(thread_id)
        / day
        / f"{doc_id}_{file_name}"
    )

    if source_disk.lower() == "s3":
        bucket = _resolve_upload_bucket()
        prefix = _resolve_upload_prefix()
        key = f"{prefix}/{rel_path.as_posix()}".lstrip("/")
        _put_s3_object(bucket=bucket, key=key, data=data)
        source_uri = f"s3://{bucket}/{key}"
        return key, source_uri, "s3"

    base_dir = Path(os.getenv("IL_UPLOAD_STORAGE_DIR", "/tmp/intelligence_layer_uploads")).expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    abs_path = (base_dir / rel_path).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(data)
    source_uri = build_source_object_uri(disk=source_disk, file_path=str(rel_path))
    return str(abs_path), source_uri, source_disk


def _resolve_upload_lifecycle(value: str | None = None) -> str:
    raw = str(value or os.getenv("IL_UPLOAD_LIFECYCLE", "sandbox")).strip().lower()
    if raw in {"temp", "sandbox", "final"}:
        return raw
    return "sandbox"


def _resolve_upload_bucket() -> str:
    bucket = os.getenv("IL_UPLOAD_S3_BUCKET", "").strip()
    if bucket:
        return bucket
    platform_bucket = os.getenv("PLATFORM_ATTACHMENTS_S3_BUCKET", "").strip()
    if platform_bucket:
        return platform_bucket
    env_name = (
        os.getenv("IL_ENV", "")
        or os.getenv("APP_ENV", "")
        or os.getenv("ENV", "")
    ).strip().lower()
    if env_name in {"prod", "production"}:
        return "canapply-platform-prod"
    return "canapply-platform-stage"


def _resolve_upload_prefix() -> str:
    prefix = os.getenv("IL_UPLOAD_S3_PREFIX", "platform/intelligence_layer").strip().strip("/")
    return prefix or "platform/intelligence_layer"


def _put_s3_object(*, bucket: str, key: str, data: bytes) -> None:
    if not bucket:
        raise ValueError("S3 bucket is not configured for intelligence-layer uploads")
    try:
        import boto3  # type: ignore
    except Exception as exc:
        raise ValueError("boto3 is required for S3 uploads") from exc

    endpoint = os.getenv("PLATFORM_ATTACHMENTS_S3_ENDPOINT", "").strip() or None
    access_key = os.getenv("PLATFORM_ATTACHMENTS_S3_ACCESS_KEY", "").strip() or None
    secret_key = os.getenv("PLATFORM_ATTACHMENTS_S3_SECRET_KEY", "").strip() or None
    region = os.getenv("PLATFORM_ATTACHMENTS_S3_REGION", "").strip() or None

    client_kwargs: dict[str, Any] = {"service_name": "s3"}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if access_key:
        client_kwargs["aws_access_key_id"] = access_key
    if secret_key:
        client_kwargs["aws_secret_access_key"] = secret_key
    if region:
        client_kwargs["region_name"] = region

    try:
        client = boto3.client(**client_kwargs)
        client.put_object(Bucket=bucket, Key=key, Body=data)
    except Exception as exc:
        raise ValueError(f"failed to upload file to s3://{bucket}/{key}: {exc}") from exc


async def _insert_platform_attachment(
    *,
    student_id: int,
    thread_id: int,
    funding_request_id: int,
    file_name: str,
    file_path: str,
    disk: str,
    mime: str,
    document_type: str,
    lifecycle: str,
    content_hash: str,
    title: str | None,
    size_bytes: int,
) -> int:
    platform_db = get_platform_db()
    columns = await platform_db.fetch_all("SHOW COLUMNS FROM attachments")
    available = {str(col.get("Field") or "").strip() for col in columns if isinstance(col, dict)}
    column_types = {
        str(col.get("Field") or "").strip(): str(col.get("Type") or "").strip().lower()
        for col in columns
        if isinstance(col, dict)
    }
    if not available:
        raise ValueError("platform attachments table is unavailable")

    attachable_type = os.getenv("IL_THREAD_ATTACHABLE_TYPE", ATTACHABLE_TYPE_THREAD).strip() or ATTACHABLE_TYPE_THREAD
    metadata = {
        "source": "intelligence_layer",
        "thread_id": thread_id,
        "funding_request_id": funding_request_id,
        "document_type": document_type,
        "lifecycle": lifecycle,
        "hash": {"alg": "blake3", "value": content_hash},
        "mime": mime,
    }
    if title:
        metadata["title"] = title.strip()

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    candidate_values: dict[str, Any] = {
        "student_id": student_id,
        "attachable_type": attachable_type,
        "attachable_id": thread_id,
        "disk": disk,
        "file_path": file_path,
        "type": document_type,
        "collection": "intelligence_layer",
        "mime_type": mime,
        "file_name": file_name,
        "metadata": json.dumps(metadata),
        "created_at": now,
        "updated_at": now,
    }
    if "content_hash" in available and _supports_full_hash(column_types.get("content_hash")):
        candidate_values["content_hash"] = content_hash
    if "size" in available:
        candidate_values["size"] = size_bytes
    if "file_original_name" in available:
        candidate_values["file_original_name"] = file_name
    if "title" in available and title:
        candidate_values["title"] = title.strip()

    insert_columns = [col for col in candidate_values.keys() if col in available]
    if "file_path" not in insert_columns or "attachable_type" not in insert_columns or "attachable_id" not in insert_columns:
        raise ValueError("platform attachments table missing required columns")

    values = [candidate_values[col] for col in insert_columns]
    sql = (
        "INSERT INTO attachments ("
        + ", ".join(f"`{col}`" for col in insert_columns)
        + ") VALUES ("
        + ", ".join("%s" for _ in insert_columns)
        + ");"
    )
    attachment_id = await platform_db.execute(sql, tuple(values))
    if attachment_id > 0:
        return attachment_id

    row = await platform_db.fetch_one(
        """
        SELECT id
        FROM attachments
        WHERE attachable_type=%s AND attachable_id=%s AND file_path=%s
        ORDER BY id DESC
        LIMIT 1;
        """,
        (attachable_type, thread_id, file_path),
    )
    if not row or row.get("id") is None:
        raise ValueError("failed to create platform attachment record")
    return int(row["id"])


async def _find_existing_thread_attachment_by_hash(
    *,
    student_id: int,
    thread_id: int,
    content_hash: str,
    mime: str,
    document_type: str,
) -> dict[str, Any] | None:
    platform_db = get_platform_db()
    columns = await platform_db.fetch_all("SHOW COLUMNS FROM attachments")
    available = {str(col.get("Field") or "").strip() for col in columns if isinstance(col, dict)}
    if not available:
        return None

    attachable_type = os.getenv("IL_THREAD_ATTACHABLE_TYPE", ATTACHABLE_TYPE_THREAD).strip() or ATTACHABLE_TYPE_THREAD
    where_sql = " WHERE attachable_type=%s AND attachable_id=%s"
    params: list[Any] = [attachable_type, thread_id]
    if "student_id" in available:
        where_sql += " AND student_id=%s"
        params.append(student_id)

    hash_predicates: list[str] = []
    if "content_hash" in available:
        hash_predicates.append("content_hash=%s")
        params.append(content_hash)
    if "metadata" in available:
        hash_predicates.append("JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.hash.value'))=%s")
        params.append(content_hash)
    if not hash_predicates:
        return None
    where_sql += " AND (" + " OR ".join(hash_predicates) + ")"

    select_candidates = ["id", "file_path", "disk", "mime_type", "type", "metadata"]
    select_cols = [name for name in select_candidates if name in available]
    if "id" not in select_cols or "file_path" not in select_cols:
        return None

    sql = (
        "SELECT "
        + ", ".join(f"`{name}`" for name in select_cols)
        + " FROM attachments"
        + where_sql
        + " ORDER BY id DESC LIMIT 20;"
    )
    rows = await platform_db.fetch_all(sql, tuple(params))
    for row in rows:
        row_mime = str(row.get("mime_type") or "")
        if row_mime and not compatible_mime(row_mime, mime):
            continue
        row_kind = str(row.get("type") or "").strip().lower()
        metadata = _coerce_json(row.get("metadata"))
        if not row_kind and isinstance(metadata, dict):
            row_kind = str(metadata.get("document_type") or "").strip().lower()
        if row_kind and row_kind != document_type:
            continue
        file_path = str(row.get("file_path") or "").strip()
        if not file_path:
            continue
        return {
            "attachment_id": int(row["id"]),
            "file_path": file_path,
            "disk": str(row.get("disk") or "s3"),
        }
    return None


async def _backfill_platform_attachment_columns(
    *,
    attachment_id: int,
    file_name: str,
    content_hash: str,
    size_bytes: int,
) -> None:
    platform_db = get_platform_db()
    columns = await platform_db.fetch_all("SHOW COLUMNS FROM attachments")
    available = {str(col.get("Field") or "").strip() for col in columns if isinstance(col, dict)}
    column_types = {
        str(col.get("Field") or "").strip(): str(col.get("Type") or "").strip().lower()
        for col in columns
        if isinstance(col, dict)
    }
    if not available:
        return

    updates: list[str] = []
    params: list[Any] = []

    if "file_original_name" in available:
        updates.append("`file_original_name` = COALESCE(NULLIF(`file_original_name`, ''), %s)")
        params.append(file_name)
    if "content_hash" in available and _supports_full_hash(column_types.get("content_hash")):
        updates.append("`content_hash` = COALESCE(NULLIF(`content_hash`, ''), %s)")
        params.append(content_hash)
    if "size" in available:
        updates.append("`size` = COALESCE(`size`, %s)")
        params.append(size_bytes)
    if "updated_at" in available:
        updates.append("`updated_at` = %s")
        params.append(datetime.now(timezone.utc).replace(tzinfo=None))

    if not updates:
        return

    params.append(attachment_id)
    sql = "UPDATE attachments SET " + ", ".join(updates) + " WHERE id=%s;"
    await platform_db.execute(sql, tuple(params))


def _supports_full_hash(column_type: str | None) -> bool:
    if not column_type:
        return False
    type_text = column_type.lower().strip()
    if any(kind in type_text for kind in ("text", "blob", "json")):
        return True
    match = re.search(r"(?:char|varchar|binary|varbinary)\((\d+)\)", type_text)
    if not match:
        return False
    return int(match.group(1)) >= 64


async def _find_student_hash_match(
    *,
    pool,
    tenant_id: int,
    student_id: int,
    content_hash: bytes,
    mime: str,
) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT document_id, mime, storage_path, extracted_text, extracted_fields
            FROM ledger.documents
            WHERE tenant_id=$1
              AND student_id=$2
              AND content_hash=$3
              AND extracted_text IS NOT NULL
              AND length(extracted_text) > 0
            ORDER BY updated_at DESC
            LIMIT 20;
            """,
            tenant_id,
            student_id,
            content_hash,
        )
    for row in rows:
        if compatible_mime(str(row["mime"] or ""), mime):
            return {
                "document_id": row["document_id"],
                "storage_path": row["storage_path"],
                "extracted_text": row["extracted_text"],
                "extracted_fields": _coerce_json(row["extracted_fields"]),
            }
    return None


async def _find_existing_document_for_attachment(
    *,
    pool,
    tenant_id: int,
    thread_id: int,
    source_attachment_id: int,
) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
                """
            SELECT document_id, document_type, mime, content_size_bytes, extracted_text, source_object_uri
            FROM ledger.documents
            WHERE tenant_id=$1
              AND thread_id=$2
              AND source_attachment_id=$3
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            tenant_id,
            thread_id,
            source_attachment_id,
        )
    if not row:
        return None
    return {
        "document_id": row["document_id"],
        "document_type": row["document_type"],
        "mime": row["mime"],
        "content_size_bytes": row["content_size_bytes"],
        "extracted_text": row["extracted_text"],
        "source_object_uri": row["source_object_uri"],
    }


async def _insert_ledger_document(
    *,
    pool,
    tenant_id: int,
    document_id: uuid.UUID,
    thread_id: int,
    funding_request_id: int,
    student_id: int,
    source_attachment_id: int,
    document_type: str,
    lifecycle: str,
    source_disk: str,
    source_path: str,
    source_object_uri: str,
    source_metadata: dict[str, Any],
    content_hash: bytes,
    content_size_bytes: int,
    mime: str,
    storage_path: str,
    extracted_text: str | None,
    extracted_fields: dict[str, Any],
) -> None:
    async with pool.acquire() as conn:
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
            json.dumps(source_metadata or {}),
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
            tenant_id,
            revision_id,
            document_id,
            "ingested",
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
            "documents_upload@1.0.0",
        )


def _extract_fields(*, text: str, document_type: str) -> dict[str, Any]:
    lowered = text.lower()
    fields: dict[str, Any] = {
        "char_count": len(text),
        "word_count": len(text.split()),
    }
    email_matches = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    if email_matches:
        fields["emails"] = sorted(set(email_matches))

    if document_type == "cv":
        fields["has_education_section"] = "education" in lowered
        fields["has_experience_section"] = "experience" in lowered
        fields["has_skills_section"] = "skill" in lowered
    if document_type in {"sop", "letter", "study_plan"}:
        fields["mentions_research"] = "research" in lowered
        fields["mentions_program_fit"] = any(
            phrase in lowered for phrase in ("professor", "lab", "program", "supervisor", "fit")
        )
    return fields


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


def build_presigned_download_url(*, object_uri: str, expires_sec: int = 900) -> str | None:
    bucket, key = _parse_s3_uri(object_uri)
    if not bucket or not key:
        return None
    try:
        import boto3  # type: ignore
    except Exception:
        return None

    endpoint = os.getenv("PLATFORM_ATTACHMENTS_S3_ENDPOINT", "").strip() or None
    access_key = os.getenv("PLATFORM_ATTACHMENTS_S3_ACCESS_KEY", "").strip() or None
    secret_key = os.getenv("PLATFORM_ATTACHMENTS_S3_SECRET_KEY", "").strip() or None
    region = os.getenv("PLATFORM_ATTACHMENTS_S3_REGION", "").strip() or None

    client_kwargs: dict[str, Any] = {"service_name": "s3"}
    if endpoint:
        client_kwargs["endpoint_url"] = endpoint
    if access_key:
        client_kwargs["aws_access_key_id"] = access_key
    if secret_key:
        client_kwargs["aws_secret_access_key"] = secret_key
    if region:
        client_kwargs["region_name"] = region

    try:
        client = boto3.client(**client_kwargs)
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=max(60, int(expires_sec)),
        )
    except Exception:
        return None


async def fetch_document_metadata(
    *,
    pool,
    tenant_id: int,
    document_id: str,
) -> dict[str, Any] | None:
    try:
        document_uuid = uuid.UUID(str(document_id))
    except Exception:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT document_id,
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
                   created_at,
                   updated_at,
                   current_revision_id
            FROM ledger.documents
            WHERE tenant_id=$1 AND document_id=$2
            LIMIT 1;
            """,
            tenant_id,
            document_uuid,
        )
    if not row:
        return None
    return {
        "document_id": str(row["document_id"]),
        "thread_id": int(row["thread_id"]),
        "funding_request_id": int(row["funding_request_id"]),
        "student_id": int(row["student_id"]),
        "source_attachment_id": int(row["source_attachment_id"]) if row["source_attachment_id"] is not None else None,
        "document_type": str(row["document_type"]),
        "lifecycle": str(row["lifecycle"] or "sandbox"),
        "source_disk": str(row["source_disk"] or ""),
        "source_path": str(row["source_path"] or ""),
        "source_object_uri": str(row["source_object_uri"] or ""),
        "source_metadata": _coerce_json(row["source_metadata"]) if row["source_metadata"] is not None else {},
        "content_hash": (row["content_hash"] or b"").hex(),
        "content_size_bytes": int(row["content_size_bytes"] or 0),
        "mime": str(row["mime"] or "application/octet-stream"),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "current_revision_id": str(row["current_revision_id"]) if row["current_revision_id"] else None,
    }


async def fetch_document_revisions(
    *,
    pool,
    tenant_id: int,
    document_id: str,
) -> list[dict[str, Any]]:
    try:
        document_uuid = uuid.UUID(str(document_id))
    except Exception:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT revision_id,
                   document_id,
                   revision_no,
                   revision_kind,
                   object_uri,
                   content_json,
                   content_hash,
                   processor_version,
                   created_at
            FROM ledger.document_revisions
            WHERE tenant_id=$1 AND document_id=$2
            ORDER BY revision_no DESC, created_at DESC;
            """,
            tenant_id,
            document_uuid,
        )
    out: list[dict[str, Any]] = []
    for row in rows:
        object_uri = str(row["object_uri"] or "")
        out.append(
            {
                "revision_id": str(row["revision_id"]),
                "document_id": str(row["document_id"]),
                "revision_no": int(row["revision_no"]),
                "revision_kind": str(row["revision_kind"]),
                "object_uri": object_uri,
                "content_hash": (row["content_hash"] or b"").hex(),
                "processor_version": str(row["processor_version"] or ""),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "content_json": _coerce_json(row["content_json"]) if row["content_json"] is not None else {},
            }
        )
    return out


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    value = str(uri or "").strip()
    if not value.startswith("s3://"):
        return "", ""
    stripped = value[5:]
    if "/" not in stripped:
        return stripped, ""
    bucket, key = stripped.split("/", 1)
    return bucket.strip(), key.strip()
