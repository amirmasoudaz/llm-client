from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blake3 import blake3

from .funding_request_fields_common import get_platform_db


DOCUMENT_TYPES = {
    "certificate",
    "cv",
    "language_test",
    "identification",
    "photograph",
    "portfolio",
    "sop",
    "letter",
    "transcript",
    "study_plan",
    "other",
}
DEFAULT_DOCUMENT_TYPE = "cv"
ATTACHABLE_TYPE_THREAD = "intelligence_thread"

DOCUMENT_TYPE_ALIASES: dict[str, str] = {
    "resume": "cv",
    "coverletter": "letter",
    "cover_letter": "letter",
    "motivation_letter": "letter",
    "statement_of_purpose": "sop",
    "personal_statement": "sop",
    "lang_test": "language_test",
    "language": "language_test",
    "id": "identification",
    "photo": "photograph",
    "plan_of_study": "study_plan",
}

REQUESTED_TO_ATTACHMENT_KINDS: dict[str, list[str]] = {
    "cv": ["cv"],
    "sop": ["sop"],
    "letter": ["letter", "sop"],
    "transcript": ["transcript"],
    "portfolio": ["portfolio"],
    "study_plan": ["study_plan", "sop"],
    "certificate": ["certificate"],
    "language_test": ["language_test"],
    "identification": ["identification"],
    "photograph": ["photograph"],
    "other": [],
}

SOURCE_PRIORITY: dict[str, int] = {
    "thread_upload": 0,
    "funding_request": 1,
    "profile": 2,
}


@dataclass(frozen=True)
class ThreadScope:
    student_id: int
    funding_request_id: int


def normalize_requested_document_type(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    resolved = DOCUMENT_TYPE_ALIASES.get(raw, raw)
    if resolved not in DOCUMENT_TYPES:
        return DEFAULT_DOCUMENT_TYPE
    return resolved


def requested_attachment_kinds(requested_document_type: str) -> list[str]:
    doc_type = normalize_requested_document_type(requested_document_type)
    kinds = REQUESTED_TO_ATTACHMENT_KINDS.get(doc_type, [])
    return [kind for kind in kinds if kind]


def extract_attachment_ids(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    result: list[int] = []
    for item in raw:
        value = None
        if isinstance(item, int):
            value = item
        elif isinstance(item, str) and item.strip().isdigit():
            value = int(item.strip())
        elif isinstance(item, dict):
            candidate = item.get("attachment_id", item.get("id"))
            if isinstance(candidate, int):
                value = candidate
            elif isinstance(candidate, str) and candidate.strip().isdigit():
                value = int(candidate.strip())
        if value is None or value <= 0:
            continue
        if value not in result:
            result.append(value)
    return result


def normalize_upload_filename(file_name: str | None) -> str:
    candidate = str(file_name or "").strip()
    if not candidate:
        return "document.bin"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate)
    return sanitized.strip("._") or "document.bin"


def canonical_mime_family(mime: str | None) -> str:
    value = str(mime or "").lower()
    if "pdf" in value:
        return "pdf"
    if "wordprocessingml.document" in value or value.endswith("docx"):
        return "docx"
    if value == "application/msword" or value.endswith("/doc"):
        return "doc"
    if value.startswith("text/"):
        return "text"
    if value.startswith("image/"):
        return "image"
    return "binary"


def compatible_mime(mime_a: str | None, mime_b: str | None) -> bool:
    family_a = canonical_mime_family(mime_a)
    family_b = canonical_mime_family(mime_b)
    if family_a == family_b:
        return True
    # DOC and DOCX contain equivalent textual content for our parse+review path.
    if {family_a, family_b} == {"doc", "docx"}:
        return True
    return False


def infer_document_type(
    *,
    document_type_hint: Any = None,
    mime: str | None = None,
    file_name: str | None = None,
    text: str | None = None,
) -> str:
    hinted = normalize_requested_document_type(document_type_hint)
    if document_type_hint is not None and hinted in DOCUMENT_TYPES:
        return hinted

    name = str(file_name or "").lower()
    if any(token in name for token in ("cv", "resume")):
        return "cv"
    if "transcript" in name:
        return "transcript"
    if "portfolio" in name:
        return "portfolio"
    if "study" in name and "plan" in name:
        return "study_plan"
    if "certificate" in name:
        return "certificate"
    if "sop" in name or "statement" in name:
        return "sop"
    if "cover" in name and "letter" in name:
        return "letter"
    if "photo" in name or "passport" in name:
        return "photograph"
    if "language" in name or "ielts" in name or "toefl" in name:
        return "language_test"

    family = canonical_mime_family(mime)
    if family in {"pdf", "doc", "docx", "text"}:
        lowered = str(text or "").lower()
        if "curriculum vitae" in lowered or "experience" in lowered:
            return "cv"
        if "statement of purpose" in lowered or "research interests" in lowered:
            return "sop"
        return "cv"
    if family == "image":
        return "photograph"
    return "other"


async def resolve_thread_scope(*, pool, tenant_id: int, thread_id: int) -> ThreadScope:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT student_id, funding_request_id
            FROM runtime.threads
            WHERE tenant_id=$1 AND thread_id=$2;
            """,
            tenant_id,
            thread_id,
        )
    if not row:
        raise ValueError("thread not found")
    return ThreadScope(student_id=int(row["student_id"]), funding_request_id=int(row["funding_request_id"]))


async def list_platform_attachments(
    *,
    funding_request_id: int,
    student_id: int,
    thread_id: int,
    requested_document_type: str,
    attachment_ids: list[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    platform_db = get_platform_db()
    columns = await platform_db.fetch_all("SHOW COLUMNS FROM attachments")
    available = {str(col.get("Field") or "").strip() for col in columns if isinstance(col, dict)}
    if not available:
        return [], None

    select_candidates = [
        "id",
        "student_id",
        "attachable_type",
        "attachable_id",
        "disk",
        "file_path",
        "type",
        "collection",
        "metadata",
        "mime_type",
        "file_name",
        "created_at",
        "updated_at",
    ]
    select_cols = [name for name in select_candidates if name in available]
    if "id" not in select_cols or "file_path" not in select_cols:
        return [], None

    attachment_id_filter = [item for item in (attachment_ids or []) if item > 0]
    scopes: list[tuple[str, int, str]] = [
        (ATTACHABLE_TYPE_THREAD, thread_id, "thread_upload"),
        ("funding_request", funding_request_id, "funding_request"),
        ("student", student_id, "profile"),
    ]

    gathered: list[dict[str, Any]] = []
    for attachable_type, attachable_id, source_scope in scopes:
        where_sql = " WHERE attachable_type=%s AND attachable_id=%s"
        params: list[Any] = [attachable_type, attachable_id]
        if "student_id" in available:
            where_sql += " AND student_id=%s"
            params.append(student_id)
        if attachment_id_filter:
            placeholders = ",".join("%s" for _ in attachment_id_filter)
            where_sql += f" AND id IN ({placeholders})"
            params.extend(attachment_id_filter)

        sql = (
            "SELECT "
            + ", ".join(f"`{name}`" for name in select_cols)
            + " FROM attachments"
            + where_sql
            + " ORDER BY id DESC LIMIT 200;"
        )
        rows = await platform_db.fetch_all(sql, tuple(params))
        for row in rows:
            shaped = _shape_attachment_row(row, source_scope=source_scope)
            if shaped is not None:
                gathered.append(shaped)

    dedup: dict[int, dict[str, Any]] = {}
    for attachment in gathered:
        attachment_id = int(attachment["attachment_id"])
        existing = dedup.get(attachment_id)
        if existing is None:
            dedup[attachment_id] = attachment
            continue
        old_priority = SOURCE_PRIORITY.get(str(existing.get("source_scope")), 99)
        new_priority = SOURCE_PRIORITY.get(str(attachment.get("source_scope")), 99)
        if new_priority < old_priority:
            dedup[attachment_id] = attachment

    attachments = list(dedup.values())
    attachments.sort(
        key=lambda item: (
            SOURCE_PRIORITY.get(str(item.get("source_scope")), 99),
            -int(item.get("attachment_id") or 0),
        )
    )

    requested_kinds = requested_attachment_kinds(requested_document_type)
    if requested_kinds:
        filtered = [item for item in attachments if item.get("kind") in requested_kinds]
    else:
        filtered = attachments

    selected = _pick_attachment(filtered, requested_kinds=requested_kinds, preferred_ids=attachment_id_filter)
    return filtered, selected


def _shape_attachment_row(row: dict[str, Any], *, source_scope: str) -> dict[str, Any] | None:
    attachment_id = row.get("id")
    file_path = _as_non_empty_str(row.get("file_path"))
    if attachment_id is None or file_path is None:
        return None

    metadata = _coerce_json(row.get("metadata"))
    if not isinstance(metadata, dict):
        metadata = {}
    kind = _infer_attachment_kind(row=row, metadata=metadata, file_path=file_path)
    disk = _as_non_empty_str(row.get("disk")) or "s3"
    mime = _as_non_empty_str(row.get("mime_type")) or _as_non_empty_str(metadata.get("mime"))
    name = _as_non_empty_str(row.get("file_name")) or Path(file_path).name
    object_uri = build_source_object_uri(disk=disk, file_path=file_path)

    return {
        "attachment_id": int(attachment_id),
        "student_id": _coerce_int(row.get("student_id")),
        "attachable_type": _as_non_empty_str(row.get("attachable_type")),
        "attachable_id": _coerce_int(row.get("attachable_id")),
        "source_scope": source_scope,
        "disk": disk,
        "file_path": file_path,
        "kind": kind,
        "mime": mime or guess_mime(file_path),
        "name": name,
        "metadata": metadata,
        "object_uri": object_uri,
        "created_at": _coerce_iso(row.get("created_at")),
        "updated_at": _coerce_iso(row.get("updated_at")),
    }


def _pick_attachment(
    attachments: list[dict[str, Any]],
    *,
    requested_kinds: list[str],
    preferred_ids: list[int] | None = None,
) -> dict[str, Any] | None:
    if not attachments:
        return None
    preferred = set(preferred_ids or [])
    if preferred:
        for attachment in attachments:
            if int(attachment.get("attachment_id") or 0) in preferred:
                return attachment
    if not requested_kinds:
        return attachments[0]
    for kind in requested_kinds:
        for attachment in attachments:
            if attachment.get("kind") == kind:
                return attachment
    return attachments[0]


def _infer_attachment_kind(*, row: dict[str, Any], metadata: dict[str, Any], file_path: str) -> str:
    candidates = [
        _as_non_empty_str(row.get("type")),
        _as_non_empty_str(metadata.get("type")),
        _as_non_empty_str(row.get("collection")),
        _as_non_empty_str(metadata.get("collection")),
    ]
    for candidate in candidates:
        if candidate:
            normalized = normalize_requested_document_type(candidate)
            if normalized in DOCUMENT_TYPES:
                return normalized
    return infer_document_type(file_name=file_path, mime=guess_mime(file_path))


def build_source_object_uri(*, disk: str, file_path: str) -> str:
    path = file_path.strip().lstrip("/")
    if file_path.startswith("s3://"):
        return file_path
    if disk.lower() == "s3":
        bucket = _resolve_platform_bucket()
        if bucket:
            return f"s3://{bucket}/{path}"
    safe = re.sub(r"[^A-Za-z0-9/_\\.-]+", "_", path)
    return f"s3://local/{safe}"


def guess_mime(file_path: str) -> str:
    guess, _ = mimetypes.guess_type(file_path)
    return guess or "application/octet-stream"


def allowed_mime(mime: str) -> bool:
    allowed_raw = os.getenv(
        "IL_ATTACHMENT_ALLOWED_MIME",
        "application/pdf,text/plain,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,image/jpeg,image/png",
    )
    allowed = {item.strip().lower() for item in allowed_raw.split(",") if item.strip()}
    if not allowed:
        return True
    return mime.lower() in allowed


def max_attachment_bytes() -> int:
    try:
        return max(1024, int(os.getenv("IL_ATTACHMENT_MAX_BYTES", str(12 * 1024 * 1024))))
    except Exception:
        return 12 * 1024 * 1024


def cache_directory() -> Path:
    root = os.getenv("IL_DOCUMENT_CACHE_DIR", "/tmp/intelligence_layer_documents")
    path = Path(root).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path_for_document(document_id: str) -> Path:
    return cache_directory() / f"{document_id}.bin"


def write_cached_bytes(document_id: str, data: bytes) -> str:
    path = cache_path_for_document(document_id)
    path.write_bytes(data)
    return str(path)


def persist_streamed_attachment(document_id: str, stream_path: str | None) -> str | None:
    if not stream_path:
        return None
    source = Path(stream_path)
    if not source.exists() or not source.is_file():
        return None
    destination = cache_path_for_document(document_id)
    try:
        source.replace(destination)
        return str(destination)
    except Exception:
        return None


def remove_cached_file(path: str | None) -> None:
    if not path:
        return
    try:
        file_path = Path(path)
        if file_path.exists() and file_path.is_file():
            file_path.unlink(missing_ok=True)
    except Exception:
        return


def read_cached_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        return file_path.read_bytes()
    except Exception:
        return None


def fetch_attachment_bytes(attachment: dict[str, Any]) -> dict[str, Any]:
    object_uri = str(attachment.get("object_uri") or "")
    mime = str(attachment.get("mime") or guess_mime(str(attachment.get("file_path") or "")))
    if not allowed_mime(mime):
        digest = blake3(object_uri.encode("utf-8")).hexdigest()
        return {
            "bytes": None,
            "stream_path": None,
            "hash_hex": digest,
            "size_bytes": 0,
            "mime": mime,
            "status": "blocked_mime",
        }

    max_bytes = max_attachment_bytes()
    hasher = blake3()
    size = 0
    stream_path: str | None = None
    wrote_any = False

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".bin",
            prefix="attachment_",
            dir=str(cache_directory()),
            delete=False,
        ) as handle:
            stream_path = handle.name
            for chunk in _iter_attachment_chunks(attachment):
                if not chunk:
                    continue
                wrote_any = True
                size += len(chunk)
                if size > max_bytes:
                    handle.flush()
                    remove_cached_file(stream_path)
                    digest = blake3(object_uri.encode("utf-8")).hexdigest()
                    return {
                        "bytes": None,
                        "stream_path": None,
                        "hash_hex": digest,
                        "size_bytes": size,
                        "mime": mime,
                        "status": "too_large",
                    }
                hasher.update(chunk)
                handle.write(chunk)
            handle.flush()
    except Exception:
        remove_cached_file(stream_path)
        digest = blake3(object_uri.encode("utf-8")).hexdigest()
        return {
            "bytes": None,
            "stream_path": None,
            "hash_hex": digest,
            "size_bytes": 0,
            "mime": mime,
            "status": "unavailable",
        }

    if not wrote_any:
        remove_cached_file(stream_path)
        digest = blake3(object_uri.encode("utf-8")).hexdigest()
        return {
            "bytes": None,
            "stream_path": None,
            "hash_hex": digest,
            "size_bytes": 0,
            "mime": mime,
            "status": "unavailable",
        }

    return {
        "bytes": None,
        "stream_path": stream_path,
        "hash_hex": hasher.hexdigest(),
        "size_bytes": size,
        "mime": mime,
        "status": "downloaded",
    }


def _iter_attachment_chunks(attachment: dict[str, Any]):
    file_path = str(attachment.get("file_path") or "")
    disk = str(attachment.get("disk") or "").lower()
    local_root = os.getenv("PLATFORM_ATTACHMENTS_LOCAL_ROOT", "").strip()

    if local_root:
        candidate = Path(local_root).expanduser() / file_path.lstrip("/")
        if candidate.exists() and candidate.is_file():
            with candidate.open("rb") as handle:
                while True:
                    chunk = handle.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
            return

    direct = Path(file_path)
    if direct.exists() and direct.is_file():
        with direct.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        return

    object_uri = str(attachment.get("object_uri") or "")
    if object_uri.startswith("s3://") or disk == "s3":
        for chunk in _iter_s3_chunks(attachment):
            yield chunk


def _iter_s3_chunks(attachment: dict[str, Any]):
    object_uri = str(attachment.get("object_uri") or "")
    bucket = ""
    key = ""
    if object_uri.startswith("s3://"):
        stripped = object_uri[len("s3://") :]
        if "/" in stripped:
            bucket, key = stripped.split("/", 1)
    if not bucket:
        bucket = _resolve_platform_bucket()
        key = str(attachment.get("file_path") or "").lstrip("/")
    if not bucket or not key:
        return

    try:
        import boto3  # type: ignore
    except Exception:
        return

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

    client = boto3.client(**client_kwargs)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response.get("Body")
        if body is None:
            return
        while True:
            chunk = body.read(64 * 1024)
            if not chunk:
                break
            yield chunk
    except Exception:
        return


def _resolve_platform_bucket() -> str:
    explicit = os.getenv("PLATFORM_ATTACHMENTS_S3_BUCKET", "").strip()
    if explicit:
        return explicit
    upload_bucket = os.getenv("IL_UPLOAD_S3_BUCKET", "").strip()
    if upload_bucket:
        return upload_bucket
    env_name = (
        os.getenv("IL_ENV", "")
        or os.getenv("APP_ENV", "")
        or os.getenv("ENV", "")
    ).strip().lower()
    if env_name in {"prod", "production"}:
        return "canapply-platform-prod"
    return "canapply-platform-stage"


def extract_text_from_bytes(data: bytes | None, *, mime: str | None, file_name: str | None) -> tuple[str, int | None, str]:
    if not data:
        return "", None, "none"

    if data.startswith(b"%PDF"):
        text, page_count = _extract_pdf_text(data)
        if text:
            return text, page_count, "pdf"

    if file_name and str(file_name).lower().endswith(".docx"):
        text = _extract_docx_text(data)
        if text:
            return text, None, "docx"

    text = _decode_text_bytes(data, mime=mime)
    return text, None, "text"


def _extract_pdf_text(data: bytes) -> tuple[str, int | None]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return "", None
    try:
        reader = PdfReader(io.BytesIO(data))
        chunks: list[str] = []
        for page in reader.pages:
            content = page.extract_text() or ""
            if content:
                chunks.append(content)
        return _normalize_text("\n".join(chunks)), len(reader.pages)
    except Exception:
        return "", None


def _extract_docx_text(data: bytes) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception:
        return ""
    try:
        document = Document(io.BytesIO(data))
        parts = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
        return _normalize_text("\n".join(parts))
    except Exception:
        return ""


def _decode_text_bytes(data: bytes, *, mime: str | None) -> str:
    encodings = ["utf-8", "utf-16", "latin-1"]
    if mime and "utf" in mime.lower():
        encodings = ["utf-8", "utf-16", "latin-1"]
    for encoding in encodings:
        try:
            return _normalize_text(data.decode(encoding))
        except Exception:
            continue
    return _normalize_text(data.decode("utf-8", errors="ignore"))


def _normalize_text(value: str) -> str:
    text = value.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _coerce_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _coerce_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _as_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
