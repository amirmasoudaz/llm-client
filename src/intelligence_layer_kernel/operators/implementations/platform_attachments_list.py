from __future__ import annotations

import time
from typing import Any

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .documents_common import (
    list_platform_attachments,
    normalize_requested_document_type,
    requested_attachment_kinds,
    resolve_thread_scope,
)


class PlatformAttachmentsListOperator(Operator):
    name = "Platform.Attachments.List"
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
        attachment_ids = payload.get("attachment_ids")
        selected_attachment_ids = []
        if isinstance(attachment_ids, list):
            for item in attachment_ids:
                try:
                    parsed = int(item)
                except Exception:
                    continue
                if parsed > 0 and parsed not in selected_attachment_ids:
                    selected_attachment_ids.append(parsed)

        try:
            scope = await resolve_thread_scope(pool=self._pool, tenant_id=self._tenant_id, thread_id=thread_id)
            attachments, selected = await list_platform_attachments(
                thread_id=thread_id,
                funding_request_id=scope.funding_request_id,
                student_id=scope.student_id,
                requested_document_type=requested_document_type,
                attachment_ids=selected_attachment_ids,
            )
        except Exception as exc:
            return _failed(start, "platform_attachments_list_failed", str(exc), category="dependency")

        result: dict[str, Any] = {
            "thread_id": thread_id,
            "funding_request_id": scope.funding_request_id,
            "student_id": scope.student_id,
            "requested_document_type": requested_document_type,
            "requested_attachment_kinds": requested_attachment_kinds(requested_document_type),
            "attachments": attachments,
            "selected_attachment": selected,
        }

        return OperatorResult(
            status="succeeded",
            result=result,
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


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
