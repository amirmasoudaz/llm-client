# src/api/routes/threads.py
"""Thread management API routes."""

from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import (
    DBDep, StudentIDDep, JobServiceDep, EventServiceDep
)


router = APIRouter()


# ============================================================================
# Pydantic Models
# ============================================================================

class ThreadCreate(BaseModel):
    """Request body for creating a new thread."""
    funding_request_id: int
    title: Optional[str] = None


class ThreadResponse(BaseModel):
    """Response model for a thread."""
    id: int
    funding_request_id: int
    student_id: int
    title: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ThreadListResponse(BaseModel):
    """Response model for thread list."""
    threads: List[ThreadResponse]
    total: int


class MessageCreate(BaseModel):
    """Request body for sending a message."""
    content: str
    document_ids: Optional[List[int]] = Field(default_factory=list)
    contexts: Optional[dict] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """Response model for a message."""
    id: int
    thread_id: int
    message_idx: int
    role: str
    message_type: str
    content: dict
    tool_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    """Response model for conversation history."""
    messages: List[MessageResponse]
    total: int
    has_more: bool


class SuggestionResponse(BaseModel):
    """Response model for follow-up suggestions."""
    suggestions: List[str]


# ============================================================================
# Thread Endpoints
# ============================================================================

@router.get("", response_model=ThreadListResponse)
async def list_threads(
    student_id: StudentIDDep,
    db: DBDep,
    funding_request_id: Optional[int] = Query(None, description="Filter by funding request"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ThreadListResponse:
    """
    List all threads for the authenticated student.
    
    Optionally filter by funding_request_id or status.
    """
    threads, total = await db.list_threads(
        student_id=student_id,
        funding_request_id=funding_request_id,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    
    return ThreadListResponse(
        threads=[ThreadResponse.model_validate(t) for t in threads],
        total=total,
    )


@router.post("", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(
    body: ThreadCreate,
    student_id: StudentIDDep,
    db: DBDep,
) -> ThreadResponse:
    """
    Create a new chat thread for a funding request.
    
    Each funding request can have multiple threads.
    """
    # Verify student has access to the funding request
    request = await db.get_funding_request(body.funding_request_id)
    if not request:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Funding request not found"
        )
    if request.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this funding request"
        )
    
    thread = await db.create_thread(
        funding_request_id=body.funding_request_id,
        student_id=student_id,
        title=body.title,
    )
    
    return ThreadResponse.model_validate(thread)


@router.get("/{thread_id}", response_model=ThreadResponse)
async def get_thread(
    thread_id: int,
    student_id: StudentIDDep,
    db: DBDep,
) -> ThreadResponse:
    """Get a specific thread by ID."""
    thread = await db.get_thread(thread_id)
    
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )
    
    if thread.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    
    return ThreadResponse.model_validate(thread)


# ============================================================================
# Message Endpoints
# ============================================================================

@router.post("/{thread_id}/messages")
async def send_message(
    thread_id: int,
    body: MessageCreate,
    student_id: StudentIDDep,
    db: DBDep,
    job_service: JobServiceDep,
    event_service: EventServiceDep,
) -> StreamingResponse:
    """
    Send a message to a thread and get a streaming response.
    
    Returns a Server-Sent Events (SSE) stream with:
    - `response` events: Token-by-token chat response
    - `meta` events: Actions, button prompts, redirects
    - `progress` events: Job lifecycle updates
    """
    # Verify access
    thread = await db.get_thread(thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )
    if thread.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    
    # Import here to avoid circular imports
    from src.agents.orchestrator.engine import DanaOrchestrator
    
    orchestrator = DanaOrchestrator(db, job_service, event_service)
    
    return StreamingResponse(
        orchestrator.process_stream(
            thread_id=thread_id,
            message=body.content,
            document_ids=body.document_ids,
            contexts=body.contexts,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/{thread_id}/history", response_model=HistoryResponse)
async def get_history(
    thread_id: int,
    student_id: StudentIDDep,
    db: DBDep,
    limit: int = Query(50, ge=1, le=200),
    before_idx: Optional[int] = Query(None, description="Get messages before this index"),
) -> HistoryResponse:
    """
    Get conversation history for a thread.
    
    Supports pagination with `before_idx` for loading older messages.
    """
    # Verify access
    thread = await db.get_thread(thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )
    if thread.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    
    messages, total, has_more = await db.get_thread_messages(
        thread_id=thread_id,
        limit=limit,
        before_idx=before_idx,
    )
    
    return HistoryResponse(
        messages=[MessageResponse.model_validate(m) for m in messages],
        total=total,
        has_more=has_more,
    )


@router.get("/{thread_id}/suggestions", response_model=SuggestionResponse)
async def get_suggestions(
    thread_id: int,
    student_id: StudentIDDep,
    db: DBDep,
    n: int = Query(3, ge=1, le=5, description="Number of suggestions"),
) -> SuggestionResponse:
    """
    Get follow-up prompt suggestions for a thread.
    
    Returns AI-generated suggestions based on conversation context.
    """
    # Verify access
    thread = await db.get_thread(thread_id)
    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found"
        )
    if thread.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this thread"
        )
    
    # Check for cached suggestions first
    cached = await db.get_thread_suggestions(thread_id)
    if cached:
        return SuggestionResponse(suggestions=cached[:n])
    
    # Generate new suggestions
    from src.agents.orchestrator.helpers import FollowUpAgent
    
    follow_up_agent = FollowUpAgent(db)
    suggestions = await follow_up_agent.generate_suggestions(thread_id, n=n)
    
    # Cache suggestions
    await db.save_thread_suggestions(thread_id, suggestions)
    
    return SuggestionResponse(suggestions=suggestions)





