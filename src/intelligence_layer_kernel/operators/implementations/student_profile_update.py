from __future__ import annotations

import json
import time
from typing import Any

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .profile_memory_utils import evaluate_requirements, merge_profile_updates, normalize_profile, validate_profile


class StudentProfileUpdateOperator(Operator):
    name = "StudentProfile.Update"
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

        profile_updates = call.payload.get("profile_updates")
        if not isinstance(profile_updates, dict):
            profile_updates = {
                key: value
                for key, value in call.payload.items()
                if key not in {"thread_id", "source", "entries", "memory_updates"}
            }

        async with self._pool.acquire() as conn:
            thread = await conn.fetchrow(
                """
                SELECT student_id
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

            row = await conn.fetchrow(
                """
                SELECT profile_json
                FROM profile.student_profiles
                WHERE tenant_id=$1 AND student_id=$2;
                """,
                self._tenant_id,
                student_id,
            )
            base_profile = dict(row["profile_json"]) if row and isinstance(row["profile_json"], dict) else {}

        base_profile = normalize_profile(base_profile, student_id=student_id)
        merged_profile, updated_fields = merge_profile_updates(base_profile, profile_updates)
        merged_profile = normalize_profile(merged_profile, student_id=student_id)

        errors = validate_profile(merged_profile)
        if errors:
            return OperatorResult(
                status="failed",
                result=None,
                artifacts=[],
                metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
                error=OperatorError(
                    code="invalid_profile_update",
                    message="profile update failed schema validation",
                    category="validation",
                    retryable=False,
                    details={"errors": errors},
                ),
            )

        completeness_state = _completeness_state(merged_profile)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO profile.student_profiles (
                  tenant_id, student_id, profile_json, schema_version, completeness_state, updated_at
                ) VALUES ($1,$2,$3::jsonb,'2.0.0',$4::jsonb,now())
                ON CONFLICT (tenant_id, student_id) DO UPDATE
                SET profile_json=EXCLUDED.profile_json,
                    schema_version='2.0.0',
                    completeness_state=EXCLUDED.completeness_state,
                    updated_at=now();
                """,
                self._tenant_id,
                student_id,
                json.dumps(merged_profile),
                json.dumps(completeness_state),
            )

        return OperatorResult(
            status="succeeded",
            result={
                "student_id": student_id,
                "profile": merged_profile,
                "updated_fields": updated_fields,
                "validation": {"is_valid": True, "errors": []},
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


def _failed(*, start: float, code: str, message: str) -> OperatorResult:
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
        ),
    )
