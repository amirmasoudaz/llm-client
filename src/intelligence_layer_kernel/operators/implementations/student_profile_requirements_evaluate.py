from __future__ import annotations

import time

from ..base import Operator
from ..types import OperatorCall, OperatorError, OperatorMetrics, OperatorResult
from .profile_memory_utils import evaluate_requirements, normalize_profile, validate_profile


class StudentProfileRequirementsEvaluateOperator(Operator):
    name = "StudentProfile.Requirements.Evaluate"
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

        intent_type = call.payload.get("intent_type")
        if not isinstance(intent_type, str) or not intent_type.strip():
            intent_type = "Funding.Outreach.Email.Generate"
        required_requirements = call.payload.get("required_requirements")
        if not isinstance(required_requirements, list):
            required_requirements = []
        strict = bool(call.payload.get("strict", False))

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
        profile = dict(row["profile_json"]) if row and isinstance(row["profile_json"], dict) else {}
        profile = normalize_profile(profile, student_id=student_id)

        validation_errors = validate_profile(profile)
        if validation_errors:
            return _failed(
                start=start,
                code="profile_invalid",
                message="student profile is invalid",
                details={"errors": validation_errors},
            )

        requirements = evaluate_requirements(
            profile,
            intent_type=intent_type,
            required_requirements=[str(item) for item in required_requirements if str(item).strip()],
        )
        if strict and not requirements["is_satisfied"]:
            return _failed(
                start=start,
                code="missing_required_profile_fields",
                message="profile requirements are not satisfied",
                details={
                    "missing_requirements": requirements["missing_requirements"],
                    "missing_fields": requirements["missing_fields"],
                    "targeted_questions": requirements["targeted_questions"],
                },
            )

        return OperatorResult(
            status="succeeded",
            result={"requirements": requirements},
            artifacts=[],
            metrics=OperatorMetrics(latency_ms=int((time.monotonic() - start) * 1000)),
            error=None,
        )


def _failed(
    *,
    start: float,
    code: str,
    message: str,
    details: dict | None = None,
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
