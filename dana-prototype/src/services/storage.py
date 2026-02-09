# src/services/storage.py
"""Storage service for S3 and file management."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

from blake3 import blake3

from src.config import get_settings
from src.tools.async_s3 import AsyncS3, S3ClientSettings


class StorageService:
    """
    Storage service for managing files in S3.
    
    Handles three file lifecycles:
    - temps: Temporary files for processing (TTL-based cleanup)
    - sandbox: Work-in-progress files pending approval
    - finals: Approved/finalized documents
    """
    
    def __init__(self):
        settings = get_settings()
        
        self._s3_settings = S3ClientSettings(
            region_name=settings.s3_region,
            bucket=settings.s3_bucket,
            max_pool_connections=64,
            max_concurrency=64,
            reqs_per_sec=50.0,
            req_burst=100.0,
        )
        
        self._s3: Optional[AsyncS3] = None
        self._base_prefix = "platform/dana"
    
    async def _get_s3(self) -> AsyncS3:
        """Get or create S3 client."""
        if self._s3 is None:
            self._s3 = AsyncS3(self._s3_settings)
            await self._s3.__aenter__()
        return self._s3
    
    async def close(self) -> None:
        """Close S3 connection."""
        if self._s3:
            await self._s3.__aexit__(None, None, None)
            self._s3 = None
    
    # =========================================================================
    # Key Helpers
    # =========================================================================
    
    def _temp_key(self, student_id: int, thread_id: int, filename: str) -> str:
        """Generate key for temporary file."""
        return f"{self._base_prefix}/{student_id}/temporary/{thread_id}/{filename}"
    
    def _sandbox_key(self, student_id: int, thread_id: int, filename: str) -> str:
        """Generate key for sandbox file."""
        return f"{self._base_prefix}/{student_id}/sandbox/{thread_id}/{filename}"
    
    def _final_key(self, student_id: int, content_hash: str, extension: str) -> str:
        """Generate key for finalized file."""
        return f"{self._base_prefix}/{student_id}/documents/{content_hash}.{extension}"
    
    def _source_key(self, student_id: int, file_hash: str, extension: str) -> str:
        """Generate key for source file upload."""
        return f"{self._base_prefix}/{student_id}/sources/{file_hash}.{extension}"
    
    @staticmethod
    def compute_hash(content: bytes) -> str:
        """Compute blake3 hash of content."""
        return blake3(content).hexdigest()
    
    @staticmethod
    def compute_sha256(content: bytes) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()
    
    # =========================================================================
    # Temp File Operations
    # =========================================================================
    
    async def create_temp(
        self,
        student_id: int,
        thread_id: int,
        content: bytes,
        extension: str = "bin",
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Create a temporary file.
        
        Temp files are scoped to a thread and have a short TTL.
        """
        s3 = await self._get_s3()
        
        filename = f"{uuid4().hex}.{extension}"
        key = self._temp_key(student_id, thread_id, filename)
        
        await s3.put_bytes(
            key,
            content,
            content_type=content_type,
            metadata={"created": datetime.utcnow().isoformat()},
        )
        
        return key
    
    async def get_temp(self, key: str) -> bytes:
        """Get content of a temporary file."""
        s3 = await self._get_s3()
        return await s3.get_bytes(key)
    
    async def delete_temp(self, key: str) -> None:
        """Delete a temporary file."""
        s3 = await self._get_s3()
        await s3.delete(key)
    
    async def cleanup_thread_temps(self, student_id: int, thread_id: int) -> int:
        """Clean up all temporary files for a thread."""
        s3 = await self._get_s3()
        prefix = self._temp_key(student_id, thread_id, "")
        return await s3.delete_prefix(prefix)
    
    # =========================================================================
    # Sandbox File Operations
    # =========================================================================
    
    async def create_sandbox(
        self,
        student_id: int,
        thread_id: int,
        content: bytes,
        extension: str = "pdf",
        content_type: str = "application/pdf",
    ) -> str:
        """
        Create a sandbox file.
        
        Sandbox files are work-in-progress, pending user approval.
        """
        s3 = await self._get_s3()
        
        content_hash = self.compute_hash(content)[:16]
        filename = f"{content_hash}.{extension}"
        key = self._sandbox_key(student_id, thread_id, filename)
        
        await s3.put_bytes(
            key,
            content,
            content_type=content_type,
            metadata={
                "created": datetime.utcnow().isoformat(),
                "content_hash": content_hash,
            },
        )
        
        return key
    
    async def get_sandbox(self, key: str) -> bytes:
        """Get content of a sandbox file."""
        s3 = await self._get_s3()
        return await s3.get_bytes(key)
    
    async def promote_to_sandbox(
        self,
        temp_key: str,
        student_id: int,
        thread_id: int,
        extension: str = "pdf",
    ) -> str:
        """Move a temp file to sandbox."""
        s3 = await self._get_s3()
        
        # Read temp file
        content = await s3.get_bytes(temp_key)
        
        # Create sandbox file
        sandbox_key = await self.create_sandbox(
            student_id, thread_id, content, extension
        )
        
        # Delete temp file
        await s3.delete(temp_key)
        
        return sandbox_key
    
    # =========================================================================
    # Final File Operations
    # =========================================================================
    
    async def finalize(
        self,
        sandbox_key: str,
        student_id: int,
        extension: str = "pdf",
        content_type: str = "application/pdf",
    ) -> str:
        """
        Finalize a sandbox file.
        
        Moves from sandbox to permanent storage with content-addressed naming.
        """
        s3 = await self._get_s3()
        
        # Read sandbox file
        content = await s3.get_bytes(sandbox_key)
        content_hash = self.compute_hash(content)
        
        # Create final key
        final_key = self._final_key(student_id, content_hash, extension)
        
        # Check if already exists (dedup)
        if await s3.exists(final_key):
            # Delete sandbox since final already exists
            await s3.delete(sandbox_key)
            return final_key
        
        # Upload to final location
        await s3.put_bytes(
            final_key,
            content,
            content_type=content_type,
            metadata={
                "created": datetime.utcnow().isoformat(),
                "content_hash": content_hash,
            },
        )
        
        # Delete sandbox file
        await s3.delete(sandbox_key)
        
        return final_key
    
    async def get_final(self, key: str) -> bytes:
        """Get content of a finalized file."""
        s3 = await self._get_s3()
        return await s3.get_bytes(key)
    
    # =========================================================================
    # Document Upload Operations
    # =========================================================================
    
    async def upload_document(
        self,
        student_id: int,
        content: bytes,
        filename: str,
        title: str,
        document_type: str,
        db: Any,  # DatabaseService
    ) -> Any:  # StudentDocument
        """
        Upload a document and create database record.
        
        Uses content-addressed storage for deduplication.
        """
        s3 = await self._get_s3()
        
        # Compute hashes
        file_hash = self.compute_hash(content)
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
        
        # Determine content type
        content_types = {
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
        content_type = content_types.get(extension, "application/octet-stream")
        
        # Generate storage key
        key = self._source_key(student_id, file_hash, extension)
        
        # Check if file already exists (dedup)
        if not await s3.exists(key):
            await s3.put_bytes(
                key,
                content,
                content_type=content_type,
                metadata={
                    "original_filename": filename,
                    "uploaded": datetime.utcnow().isoformat(),
                },
            )
        
        # Create database record
        document = await db.create_document(
            student_id=student_id,
            title=title,
            document_type=document_type,
            source_file_path=key,
            source_file_hash=file_hash,
        )
        
        return document
    
    async def delete_document(self, document: Any) -> None:
        """Delete a document and its files."""
        s3 = await self._get_s3()
        
        # Delete source file
        if document.source_file_path:
            try:
                await s3.delete(document.source_file_path)
            except Exception:
                pass  # Ignore if file doesn't exist
        
        # Delete PDF if exists
        if document.exported_pdf_path:
            try:
                await s3.delete(document.exported_pdf_path)
            except Exception:
                pass
    
    # =========================================================================
    # Streaming Operations
    # =========================================================================
    
    async def stream_download(
        self,
        key: str,
        chunk_size: int = 1024 * 1024,
    ) -> AsyncIterator[bytes]:
        """Stream download a file in chunks."""
        s3 = await self._get_s3()
        async for chunk in s3.iter_download(key, chunk_size=chunk_size):
            yield chunk
    
    # =========================================================================
    # Utility Operations
    # =========================================================================
    
    async def exists(self, key: str) -> bool:
        """Check if a file exists."""
        s3 = await self._get_s3()
        return await s3.exists(key)
    
    async def get_metadata(self, key: str) -> dict:
        """Get file metadata."""
        s3 = await self._get_s3()
        response = await s3.head(key)
        return response.get("Metadata", {})
    
    async def list_keys(self, prefix: str, limit: Optional[int] = None) -> list:
        """List keys under a prefix."""
        s3 = await self._get_s3()
        return await s3.list_keys(prefix, limit=limit)





