# src/services/jobs.py
"""Job service for managing AI jobs and tracking."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional
from dataclasses import dataclass, field

from src.services.db import DatabaseService


@dataclass
class JobProgress:
    """Job progress tracking."""
    job_id: int
    progress: int
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class JobResult:
    """Job result container."""
    job_id: int
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class JobService:
    """
    Service for managing AI jobs.
    
    Handles job creation, progress tracking, completion, and retries.
    """
    
    # Job type to model mapping
    DEFAULT_MODELS = {
        "email_generate": "gpt-4o-mini",
        "email_review": "gpt-4o-mini",
        "email_optimize": "gpt-4o-mini",
        "resume_generate": "gpt-4o-mini",
        "resume_review": "gpt-4o-mini",
        "resume_optimize": "gpt-4o-mini",
        "letter_generate": "gpt-4o-mini",
        "letter_review": "gpt-4o-mini",
        "letter_optimize": "gpt-4o-mini",
        "alignment_professor": "gpt-4o-mini",
        "alignment_program": "gpt-4o-mini",
        "chat_thread": "gpt-4o",
        "chat_tool_call": "gpt-4o-mini",
        "chat_switchboard": "gpt-4o",
        "chat_completion": "gpt-4o-mini",
        "doc_to_json": "gpt-4o-mini",
        "thread_summarization": "gpt-4o-mini",
        "thread_moderation": "gpt-4o-mini",
        "embeddings": "text-embedding-3-small",
    }
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self._progress_callbacks: Dict[int, List[Callable]] = {}
    
    async def create_job(
        self,
        student_id: int,
        job_type: str,
        thread_id: Optional[int] = None,
        target_type: str = "chat_thread",
        target_id: int = 0,
        input_payload: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> int:
        """
        Create a new AI job.
        
        Returns the job ID.
        """
        job = await self.db.create_job(
            student_id=student_id,
            job_type=job_type,
            thread_id=thread_id,
            target_type=target_type,
            target_id=target_id,
            input_payload=input_payload,
            model=model or self.DEFAULT_MODELS.get(job_type, "gpt-4o-mini"),
        )
        
        return job.id
    
    async def start_job(self, job_id: int) -> None:
        """Mark a job as running."""
        await self.db.update_job_status(job_id, "running", progress=0)
    
    async def update_progress(
        self,
        job_id: int,
        progress: int,
        message: str = "",
    ) -> None:
        """
        Update job progress.
        
        Also notifies any registered progress callbacks.
        """
        await self.db.update_job_status(job_id, "running", progress=progress)
        
        # Notify callbacks
        if job_id in self._progress_callbacks:
            progress_update = JobProgress(
                job_id=job_id,
                progress=progress,
                message=message,
            )
            for callback in self._progress_callbacks[job_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(progress_update)
                    else:
                        callback(progress_update)
                except Exception:
                    pass  # Don't let callback errors affect job
    
    async def complete_job(
        self,
        job_id: int,
        result: Dict[str, Any],
        usage: Dict[str, Any],
        trace_id: str = "",
    ) -> JobResult:
        """Mark a job as completed successfully."""
        await self.db.complete_job(job_id, result, usage, trace_id)
        
        # Clean up callbacks
        self._progress_callbacks.pop(job_id, None)
        
        return JobResult(
            job_id=job_id,
            success=True,
            result=result,
            usage=usage,
        )
    
    async def fail_job(
        self,
        job_id: int,
        error: str,
        error_code: Optional[str] = None,
    ) -> JobResult:
        """Mark a job as failed."""
        await self.db.fail_job(job_id, error, error_code)
        
        # Clean up callbacks
        self._progress_callbacks.pop(job_id, None)
        
        return JobResult(
            job_id=job_id,
            success=False,
            error=error,
        )
    
    async def cancel_job(self, job_id: int) -> None:
        """Cancel a job."""
        await self.db.update_job_status(job_id, "cancelled")
        self._progress_callbacks.pop(job_id, None)
    
    def register_progress_callback(
        self,
        job_id: int,
        callback: Callable[[JobProgress], Any],
    ) -> None:
        """Register a callback for job progress updates."""
        if job_id not in self._progress_callbacks:
            self._progress_callbacks[job_id] = []
        self._progress_callbacks[job_id].append(callback)
    
    def unregister_progress_callback(
        self,
        job_id: int,
        callback: Callable[[JobProgress], Any],
    ) -> None:
        """Unregister a progress callback."""
        if job_id in self._progress_callbacks:
            try:
                self._progress_callbacks[job_id].remove(callback)
            except ValueError:
                pass
    
    async def get_job(self, job_id: int) -> Optional[Any]:
        """Get job details."""
        return await self.db.get_job(job_id)
    
    async def retry_job(self, job_id: int) -> int:
        """
        Retry a failed job.
        
        Creates a new job with the same parameters and returns the new job ID.
        """
        original = await self.db.get_job(job_id)
        if not original:
            raise ValueError(f"Job {job_id} not found")
        
        if original.status not in ("failed", "cancelled"):
            raise ValueError(f"Can only retry failed/cancelled jobs, got {original.status}")
        
        import json
        input_payload = json.loads(original.input_payload) if original.input_payload else None
        
        new_job = await self.db.create_job(
            student_id=original.student_id,
            job_type=original.job_type,
            thread_id=original.thread_id,
            target_type=original.target_type,
            target_id=original.target_id,
            input_payload=input_payload,
            model=original.model,
        )
        
        # Update attempt count
        await self.db.client.aijob.update(
            where={"id": new_job.id},
            data={"attempt": (original.attempt or 0) + 1},
        )
        
        return new_job.id


class JobContext:
    """
    Context manager for job execution.
    
    Handles progress tracking and automatic success/failure marking.
    """
    
    def __init__(
        self,
        job_service: JobService,
        job_id: int,
        auto_complete: bool = True,
    ):
        self.job_service = job_service
        self.job_id = job_id
        self.auto_complete = auto_complete
        self._result: Optional[Dict[str, Any]] = None
        self._usage: Optional[Dict[str, Any]] = None
        self._trace_id: str = ""
    
    async def __aenter__(self) -> "JobContext":
        await self.job_service.start_job(self.job_id)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Job failed
            await self.job_service.fail_job(
                self.job_id,
                str(exc_val),
                error_code=exc_type.__name__,
            )
            return False  # Re-raise exception
        
        if self.auto_complete and self._result is not None:
            await self.job_service.complete_job(
                self.job_id,
                self._result,
                self._usage or {},
                self._trace_id,
            )
        
        return False
    
    async def progress(self, percent: int, message: str = "") -> None:
        """Report progress."""
        await self.job_service.update_progress(self.job_id, percent, message)
    
    def set_result(
        self,
        result: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None,
        trace_id: str = "",
    ) -> None:
        """Set the job result for auto-completion."""
        self._result = result
        self._usage = usage
        self._trace_id = trace_id





