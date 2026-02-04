# src/schemas/document.py
"""Document schemas."""

from typing import List, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class DocumentType(str, Enum):
    """Document type enumeration."""
    RESUME = "resume"
    SOP = "sop"
    TRANSCRIPT = "transcript"
    PORTFOLIO = "portfolio"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    QUEUED_PROCESSING = "queued_processing"
    PROCESSING = "processing"
    PROCESSED = "processed"
    QUEUED_EXPORT = "queued_export"
    EXPORTING = "exporting"
    EXPORTED = "exported"
    FAILED = "failed"


class AttachmentPurpose(str, Enum):
    """Attachment purpose for funding requests."""
    CV = "cv"
    SOP = "sop"
    TRANSCRIPT = "transcript"
    PORTFOLIO = "portfolio"
    OTHER = "other"


# ============================================================================
# Document Schemas
# ============================================================================

class DocumentResponse(BaseModel):
    """Response model for a document."""
    id: int
    student_id: int
    title: str
    document_type: DocumentType
    upload_status: DocumentStatus
    source_file_hash: str
    has_pdf: bool = False
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Response model for document list."""
    documents: List[DocumentResponse]
    total: int


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: int
    title: str
    status: DocumentStatus
    message: str


class DocumentProcessedContent(BaseModel):
    """Processed document content."""
    raw_text: Optional[str] = None
    structured_data: Optional[dict] = None
    missing_fields: Optional[List[str]] = None
    processor_version: Optional[str] = None


# ============================================================================
# Attachment Schemas
# ============================================================================

class AttachmentResponse(BaseModel):
    """Response model for an attachment."""
    id: int
    funding_request_id: int
    document_id: int
    purpose: AttachmentPurpose
    document: Optional[DocumentResponse] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ApplyDocumentRequest(BaseModel):
    """Request body for applying a document to a request."""
    document_id: int
    funding_request_id: int
    purpose: AttachmentPurpose = AttachmentPurpose.CV


class ApplyDocumentResponse(BaseModel):
    """Response model for apply document."""
    funding_request_id: int
    attachment_id: int
    document_id: int
    purpose: str
    success: bool


# ============================================================================
# Thread Files Schemas
# ============================================================================

class ThreadFilesResponse(BaseModel):
    """Response model for thread files."""
    thread_id: int
    files: List[DocumentResponse]





