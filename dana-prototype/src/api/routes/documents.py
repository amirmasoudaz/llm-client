# src/api/routes/documents.py
"""Document management API routes."""

from typing import List, Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import DBDep, StorageDep, StudentIDDep


router = APIRouter()


# ============================================================================
# Enums
# ============================================================================

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
# Pydantic Models
# ============================================================================

class DocumentResponse(BaseModel):
    """Response model for a document."""
    id: int
    student_id: int
    title: str
    document_type: DocumentType
    upload_status: DocumentStatus
    source_file_hash: str
    has_pdf: bool
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


class ThreadFilesResponse(BaseModel):
    """Response model for thread files."""
    thread_id: int
    files: List[DocumentResponse]


# ============================================================================
# Document Endpoints
# ============================================================================

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    student_id: StudentIDDep,
    db: DBDep,
    document_type: Optional[DocumentType] = Query(None, description="Filter by type"),
    status_filter: Optional[DocumentStatus] = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> DocumentListResponse:
    """List all documents for the authenticated student."""
    documents, total = await db.list_documents(
        student_id=student_id,
        document_type=document_type,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in documents],
        total=total,
    )


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    student_id: StudentIDDep,
    db: DBDep,
    storage: StorageDep,
    file: UploadFile = File(...),
    title: str = Form(...),
    document_type: DocumentType = Form(DocumentType.OTHER),
) -> DocumentUploadResponse:
    """
    Upload a new document.
    
    Supported formats: PDF, DOCX, TXT, MD, PNG, JPG, JPEG, WEBP, TEX
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )
    
    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    allowed_extensions = {"pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp", "tex"}
    
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    content = await file.read()
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )
    
    # Upload to storage and create database record
    document = await storage.upload_document(
        student_id=student_id,
        content=content,
        filename=file.filename,
        title=title,
        document_type=document_type,
        db=db,
    )
    
    return DocumentUploadResponse(
        document_id=document.id,
        title=document.title,
        status=DocumentStatus(document.upload_status),
        message="Document uploaded successfully. Processing will begin shortly.",
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    student_id: StudentIDDep,
    db: DBDep,
) -> DocumentResponse:
    """Get a specific document by ID."""
    document = await db.get_document(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if document.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this document"
        )
    
    return DocumentResponse.model_validate(document)


@router.get("/{document_id}/download")
async def download_document(
    document_id: int,
    student_id: StudentIDDep,
    db: DBDep,
    storage: StorageDep,
    format: str = Query("pdf", description="Download format: pdf or source"),
) -> StreamingResponse:
    """
    Download a document.
    
    - `format=pdf`: Download the exported PDF (if available)
    - `format=source`: Download the original source file
    """
    document = await db.get_document(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if document.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this document"
        )
    
    if format == "pdf":
        if not document.exported_pdf_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF export not available for this document"
            )
        file_path = document.exported_pdf_path
        media_type = "application/pdf"
        filename = f"{document.title}.pdf"
    else:
        file_path = document.source_file_path
        # Determine media type from path
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else "bin"
        media_types = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "txt": "text/plain",
            "md": "text/markdown",
            "tex": "application/x-tex",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
        }
        media_type = media_types.get(ext, "application/octet-stream")
        filename = f"{document.title}.{ext}"
    
    # Stream the file from storage
    stream = storage.stream_download(file_path)
    
    return StreamingResponse(
        stream,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    student_id: StudentIDDep,
    db: DBDep,
    storage: StorageDep,
):
    """Delete a document."""
    document = await db.get_document(document_id)
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    if document.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this document"
        )
    
    # Delete from storage
    await storage.delete_document(document)
    
    # Delete from database
    await db.delete_document(document_id)


# ============================================================================
# Apply Document Endpoints
# ============================================================================

@router.post("/apply", response_model=ApplyDocumentResponse)
async def apply_document(
    body: ApplyDocumentRequest,
    student_id: StudentIDDep,
    db: DBDep,
) -> ApplyDocumentResponse:
    """
    Apply a document to a funding request as an attachment.
    
    This links a generated or uploaded document to a specific funding request.
    """
    # Verify document access
    document = await db.get_document(body.document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    if document.student_id != student_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this document"
        )
    
    # Verify request access
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
    
    # Create attachment
    attachment = await db.create_request_attachment(
        funding_request_id=body.funding_request_id,
        document_id=body.document_id,
        purpose=body.purpose,
    )
    
    return ApplyDocumentResponse(
        funding_request_id=body.funding_request_id,
        attachment_id=attachment.id,
        document_id=body.document_id,
        purpose=body.purpose,
        success=True,
    )


# ============================================================================
# Thread Files Endpoints
# ============================================================================

@router.get("/thread/{thread_id}", response_model=ThreadFilesResponse)
async def get_thread_files(
    thread_id: int,
    student_id: StudentIDDep,
    db: DBDep,
) -> ThreadFilesResponse:
    """Get all documents generated in a specific thread."""
    # Verify thread access
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
    
    # Get documents associated with this thread
    documents = await db.get_thread_documents(thread_id)
    
    return ThreadFilesResponse(
        thread_id=thread_id,
        files=[DocumentResponse.model_validate(d) for d in documents],
    )





