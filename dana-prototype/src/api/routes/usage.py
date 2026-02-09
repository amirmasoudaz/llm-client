# src/api/routes/usage.py
"""Usage and billing API routes."""

from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from src.api.dependencies import DBDep, StudentIDDep


router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class TokenUsage(BaseModel):
    """Token usage breakdown."""
    input: int = 0
    output: int = 0
    total: int = 0


class CostBreakdown(BaseModel):
    """Cost breakdown in credits."""
    input: float = 0.0
    output: float = 0.0
    total: float = 0.0


class FileUsage(BaseModel):
    """File usage breakdown."""
    input: int = 0  # Files uploaded
    output: int = 0  # Files generated
    total: int = 0


class SessionStats(BaseModel):
    """Session statistics."""
    threads: int = 0
    messages_sent: int = 0
    messages_received: int = 0


class CreditStatus(BaseModel):
    """Credit status."""
    used: float = 0.0
    remaining: float = 0.0
    total: float = 0.0
    active: bool = True


class UsageResponse(BaseModel):
    """Complete usage response."""
    student_id: int
    period_start: datetime
    period_end: datetime
    tokens: TokenUsage
    cost: CostBreakdown
    files: FileUsage
    sessions: SessionStats
    credits: CreditStatus


class UsageSummary(BaseModel):
    """Simplified usage summary."""
    tokens_used: int
    credits_used: float
    credits_remaining: float
    files_generated: int
    threads_active: int


class JobUsageItem(BaseModel):
    """Individual job usage item."""
    job_id: int
    job_type: str
    model: str
    token_input: int
    token_output: int
    cost_total: float
    created_at: datetime


class DetailedUsageResponse(BaseModel):
    """Detailed usage with job breakdown."""
    summary: UsageSummary
    jobs: list[JobUsageItem]
    total_jobs: int


# ============================================================================
# Usage Endpoints
# ============================================================================

@router.get("", response_model=UsageResponse)
async def get_usage(
    student_id: StudentIDDep,
    db: DBDep,
    from_date: Optional[datetime] = Query(
        None, 
        alias="from",
        description="Start of period (default: 30 days ago)"
    ),
    to_date: Optional[datetime] = Query(
        None,
        alias="to", 
        description="End of period (default: now)"
    ),
) -> UsageResponse:
    """
    Get usage statistics for the authenticated student.
    
    Returns token usage, costs, file stats, and credit status.
    """
    # Default to last 30 days
    if not to_date:
        to_date = datetime.utcnow()
    if not from_date:
        from_date = to_date - timedelta(days=30)
    
    # Get usage data from database
    usage_data = await db.get_student_usage(
        student_id=student_id,
        from_date=from_date,
        to_date=to_date,
    )
    
    # Get credit status from platform (or local cache)
    credit_status = await db.get_student_credits(student_id)
    
    return UsageResponse(
        student_id=student_id,
        period_start=from_date,
        period_end=to_date,
        tokens=TokenUsage(
            input=usage_data.get("token_input", 0),
            output=usage_data.get("token_output", 0),
            total=usage_data.get("token_total", 0),
        ),
        cost=CostBreakdown(
            input=usage_data.get("cost_input", 0.0),
            output=usage_data.get("cost_output", 0.0),
            total=usage_data.get("cost_total", 0.0),
        ),
        files=FileUsage(
            input=usage_data.get("files_uploaded", 0),
            output=usage_data.get("files_generated", 0),
            total=usage_data.get("files_total", 0),
        ),
        sessions=SessionStats(
            threads=usage_data.get("threads", 0),
            messages_sent=usage_data.get("messages_sent", 0),
            messages_received=usage_data.get("messages_received", 0),
        ),
        credits=CreditStatus(
            used=credit_status.get("used", 0.0),
            remaining=credit_status.get("remaining", 0.0),
            total=credit_status.get("total", 0.0),
            active=credit_status.get("active", True),
        ),
    )


@router.get("/summary", response_model=UsageSummary)
async def get_usage_summary(
    student_id: StudentIDDep,
    db: DBDep,
) -> UsageSummary:
    """Get a simplified usage summary for dashboard display."""
    # Get current month usage
    now = datetime.utcnow()
    from_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    usage_data = await db.get_student_usage(
        student_id=student_id,
        from_date=from_date,
        to_date=now,
    )
    
    credit_status = await db.get_student_credits(student_id)
    
    return UsageSummary(
        tokens_used=usage_data.get("token_total", 0),
        credits_used=usage_data.get("cost_total", 0.0),
        credits_remaining=credit_status.get("remaining", 0.0),
        files_generated=usage_data.get("files_generated", 0),
        threads_active=usage_data.get("threads_active", 0),
    )


@router.get("/detailed", response_model=DetailedUsageResponse)
async def get_detailed_usage(
    student_id: StudentIDDep,
    db: DBDep,
    from_date: Optional[datetime] = Query(None, alias="from"),
    to_date: Optional[datetime] = Query(None, alias="to"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> DetailedUsageResponse:
    """
    Get detailed usage with job-by-job breakdown.
    
    Useful for auditing and debugging.
    """
    if not to_date:
        to_date = datetime.utcnow()
    if not from_date:
        from_date = to_date - timedelta(days=30)
    
    # Get job usage records
    jobs, total_jobs = await db.get_job_usage(
        student_id=student_id,
        from_date=from_date,
        to_date=to_date,
        limit=limit,
        offset=offset,
    )
    
    # Calculate summary
    summary_data = await db.get_student_usage(
        student_id=student_id,
        from_date=from_date,
        to_date=to_date,
    )
    credit_status = await db.get_student_credits(student_id)
    
    return DetailedUsageResponse(
        summary=UsageSummary(
            tokens_used=summary_data.get("token_total", 0),
            credits_used=summary_data.get("cost_total", 0.0),
            credits_remaining=credit_status.get("remaining", 0.0),
            files_generated=summary_data.get("files_generated", 0),
            threads_active=summary_data.get("threads_active", 0),
        ),
        jobs=[
            JobUsageItem(
                job_id=j.id,
                job_type=j.job_type,
                model=j.model or "unknown",
                token_input=j.token_input or 0,
                token_output=j.token_output or 0,
                cost_total=float(j.cost_total or 0),
                created_at=j.created_at,
            )
            for j in jobs
        ],
        total_jobs=total_jobs,
    )





