# src/schemas/__init__.py
"""Pydantic schemas for Dana AI Copilot."""

from src.schemas.common import (
    BaseResponse,
    ErrorResponse,
    PaginatedResponse,
)
from src.schemas.thread import (
    ThreadCreate,
    ThreadResponse,
    ThreadListResponse,
    MessageCreate,
    MessageResponse,
)
from src.schemas.document import (
    DocumentType,
    DocumentStatus,
    DocumentResponse,
    DocumentUploadResponse,
)
from src.schemas.context import (
    UserContext,
    ProfessorContext,
    RequestContext,
    OrchestrationContext,
)

__all__ = [
    # Common
    "BaseResponse",
    "ErrorResponse",
    "PaginatedResponse",
    # Thread
    "ThreadCreate",
    "ThreadResponse",
    "ThreadListResponse",
    "MessageCreate",
    "MessageResponse",
    # Document
    "DocumentType",
    "DocumentStatus",
    "DocumentResponse",
    "DocumentUploadResponse",
    # Context
    "UserContext",
    "ProfessorContext",
    "RequestContext",
    "OrchestrationContext",
]





