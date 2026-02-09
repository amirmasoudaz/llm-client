# src/api/routes/retry.py
"""Job retry API routes."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.api.dependencies import DBDep, JobServiceDep, StudentIDDep


router = APIRouter()


class RetryResponse(BaseModel):
    """Response for job retry."""
    success: bool
    original_job_id: int
    new_job_id: int
    message: str


@router.post("/retry/{job_id}", response_model=RetryResponse)
async def retry_job(
    job_id: int,
    student_id: StudentIDDep,
    db: DBDep,
    job_service: JobServiceDep,
) -> RetryResponse:
    """
    Retry a failed job.
    
    Creates a new job with the same parameters as the original.
    """
    # Get original job
    job = await db.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this job"
        )
    
    if job.status not in ("failed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can only retry failed or cancelled jobs (current status: {job.status})"
        )
    
    try:
        new_job_id = await job_service.retry_job(job_id)
        
        return RetryResponse(
            success=True,
            original_job_id=job_id,
            new_job_id=new_job_id,
            message="Job queued for retry",
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


class JobStatusResponse(BaseModel):
    """Response for job status."""
    job_id: int
    job_type: str
    status: str
    progress: int
    model: str | None
    error_message: str | None
    error_code: str | None
    started_at: str | None
    finished_at: str | None


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: int,
    student_id: StudentIDDep,
    db: DBDep,
) -> JobStatusResponse:
    """Get the status of a job."""
    job = await db.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this job"
        )
    
    return JobStatusResponse(
        job_id=int(job.id),
        job_type=job.job_type,
        status=job.status,
        progress=job.progress or 0,
        model=job.model,
        error_message=job.error_message,
        error_code=job.error_code,
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
    )





