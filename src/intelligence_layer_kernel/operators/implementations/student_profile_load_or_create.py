from __future__ import annotations

import json
import time
from typing import Any

from intelligence_layer_ops.platform_tools import platform_load_funding_thread_context

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .profile_memory_utils import (
    evaluate_requirements,
    group_memory_by_type,
    normalize_profile,
    prefill_profile_from_platform,
    validate_profile,
)


class StudentProfileLoadOrCreateOperator(Operator):
    name = "StudentProfile.LoadOrCreate"
    version = "1.0.0"

    def __init__(self, *, pool, tenant_id: int) -> None:
        self._pool = pool
        self._tenant_id = tenant_id

    async def run(self, call: OperatorCall) -> OperatorResult:
        start = time.monotonic()
        thread_id = call.payload.get("thread_id")
        if not isinstance(thread_id, int) or thread_id <= 0:
            return _failed(
                start=start,
                code="missing_thread_id",
                message="thread_id is required",
            )

        async with self._pool.acquire() as conn:
            thread = await conn.fetchrow(
                """
                SELECT student_id, funding_request_id
                FROM runtime.threads
                WHERE tenant_id=$1 AND thread_id=$2;
                """,
                self._tenant_id,
                thread_id,
            )
        if not thread:
            return _failed(
                start=start,
                code="thread_not_found",
                message="thread not found",
            )

        student_id = int(thread["student_id"])
        funding_request_id = int(thread["funding_request_id"])

        created = False
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT profile_json
                FROM profile.student_profiles
                WHERE tenant_id=$1 AND student_id=$2;
                """,
                self._tenant_id,
                student_id,
            )

        if row and isinstance(row["profile_json"], dict):
            profile = dict(row["profile_json"])
        else:
            platform_row: dict[str, Any] = {}
            try:
                platform = await platform_load_funding_thread_context.execute(
                    funding_request_id=funding_request_id
                )
                if platform.success and isinstance(platform.content, dict):
                    platform_row = platform.content
            except Exception:
                platform_row = {}

            profile = prefill_profile_from_platform(platform_row, student_id=student_id)
            profile = normalize_profile(profile, student_id=student_id)
            validation_errors = validate_profile(profile)
            if validation_errors:
                return _failed(
                    start=start,
                    code="profile_prefill_invalid",
                    message="prefilled profile is invalid",
                    details={"errors": validation_errors},
                )

            completeness_state = _completeness_state(profile)
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO profile.student_profiles (
                      tenant_id, student_id, profile_json, schema_version, completeness_state
                    ) VALUES ($1,$2,$3::jsonb,'2.0.0',$4::jsonb)
                    ON CONFLICT (tenant_id, student_id) DO NOTHING;
                    """,
                    self._tenant_id,
                    student_id,
                    json.dumps(profile),
                    json.dumps(completeness_state),
                )
                row = await conn.fetchrow(
                    """
                    SELECT profile_json
                    FROM profile.student_profiles
                    WHERE tenant_id=$1 AND student_id=$2;
                    """,
                    self._tenant_id,
                    student_id,
                )
            created = True
            if row and isinstance(row["profile_json"], dict):
                profile = dict(row["profile_json"])

        profile = normalize_profile(profile, student_id=student_id)
        validation_errors = validate_profile(profile)
        if validation_errors:
            return _failed(
                start=start,
                code="profile_invalid",
                message="stored profile is invalid",
                details={"errors": validation_errors},
            )

        async with self._pool.acquire() as conn:
            memory_rows = await conn.fetch(
                """
                SELECT memory_id, memory_type, memory_content, source, updated_at
                FROM profile.student_memories
                WHERE tenant_id=$1 AND student_id=$2 AND is_active=true
                ORDER BY updated_at DESC
                LIMIT 50;
                """,
                self._tenant_id,
                student_id,
            )
        memory_entries = [
            {
                "memory_id": str(row["memory_id"]),
                "type": str(row["memory_type"]),
                "content": str(row["memory_content"]),
                "source": str(row["source"] or "user"),
                "updated_at": row["updated_at"].isoformat(),
            }
            for row in memory_rows
        ]
        memory = {"entries": memory_entries, "by_type": group_memory_by_type(memory_entries)}

        return OperatorResult(
            status="succeeded",
            result={
                "student_id": student_id,
                "funding_request_id": funding_request_id,
                "created": created,
                "profile": profile,
                "memory": memory,
            },
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _completeness_state(profile: dict[str, Any]) -> dict[str, Any]:
    requirements = evaluate_requirements(
        profile,
        intent_type="Student.Profile.Collect",
        required_requirements=[
            "base_profile_complete",
            "background_data_complete",
            "composer_prereqs_complete",
        ],
    )
    status = requirements["status_by_requirement"]
    return {
        "general_complete": bool(status.get("base_profile_complete")),
        "background_complete": bool(status.get("background_data_complete")),
        "sop_intelligence_complete": bool(status.get("composer_prereqs_complete")),
    }


def _failed(
    *,
    start: float,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> OperatorResult:
    return OperatorResult(
        status="failed",
        result=None,
        artifacts=[],
        metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
        error=OperatorError(
            code=code,
            message=message,
            category="validation",
            retryable=False,
            details=details,
        ),
    )
