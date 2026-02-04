# src/schemas/thread.py
"""Thread and message schemas."""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ThreadStatus(str, Enum):
    """Thread status enumeration."""
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MessageType(str, Enum):
    """Message type enumeration."""
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


# ============================================================================
# Thread Schemas
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
    status: ThreadStatus
    summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ThreadListResponse(BaseModel):
    """Response model for thread list."""
    threads: List[ThreadResponse]
    total: int


class ThreadUpdate(BaseModel):
    """Request body for updating a thread."""
    title: Optional[str] = None
    status: Optional[ThreadStatus] = None


# ============================================================================
# Message Schemas
# ============================================================================

class MessageContent(BaseModel):
    """Message content structure."""
    text: Optional[str] = None
    attachments: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


class MessageCreate(BaseModel):
    """Request body for sending a message."""
    content: str
    document_ids: Optional[List[int]] = Field(default_factory=list)
    contexts: Optional[Dict[str, Any]] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """Response model for a message."""
    id: int
    thread_id: int
    message_idx: int
    role: MessageRole
    message_type: MessageType
    content: Dict[str, Any]
    tool_name: Optional[str] = None
    tool_payload: Optional[Dict[str, Any]] = None
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
# Streaming Schemas
# ============================================================================

class StreamEvent(BaseModel):
    """SSE stream event."""
    event: str
    data: Any
    id: Optional[str] = None


class ResponseStartEvent(BaseModel):
    """Response start event data."""
    thread_id: int
    message_idx: int
    job_id: Optional[int] = None


class ResponseTokenEvent(BaseModel):
    """Response token event data."""
    token: str


class ResponseEndEvent(BaseModel):
    """Response end event data."""
    thread_id: int
    message_idx: int
    total_tokens: Optional[int] = None


class ProgressEvent(BaseModel):
    """Progress update event data."""
    percent: int
    message: str
    stage: Optional[str] = None


class MetaActionEvent(BaseModel):
    """Meta action event data."""
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    """Error event data."""
    error: str
    code: Optional[str] = None
    recoverable: bool = False





