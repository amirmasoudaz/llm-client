# src/services/db.py
"""Database service for Dana AI Copilot using Prisma."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Literal
from decimal import Decimal

from prisma import Prisma
from prisma.models import (
    ChatThread, ChatThreadMessage, AIJob, AIMemory,
    StudentDocument, FundingRequestAttachment
)

from src.config import get_settings


class DatabaseService:
    """
    Database service providing async database operations.
    
    Uses Prisma ORM for type-safe database access.
    """
    
    def __init__(self):
        self._client: Optional[Prisma] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Connect to the database."""
        if self._connected:
            return
        
        self._client = Prisma()
        await self._client.connect()
        self._connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._client and self._connected:
            await self._client.disconnect()
            self._connected = False
    
    async def is_connected(self) -> bool:
        """Check if database is connected."""
        if not self._client or not self._connected:
            return False
        try:
            # Simple query to check connection
            await self._client.execute_raw("SELECT 1")
            return True
        except Exception:
            return False
    
    @property
    def client(self) -> Prisma:
        """Get the Prisma client."""
        if not self._client:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._client
    
    # =========================================================================
    # Thread Operations
    # =========================================================================
    
    async def create_thread(
        self,
        funding_request_id: int,
        student_id: int,
        title: Optional[str] = None,
    ) -> ChatThread:
        """Create a new chat thread."""
        return await self.client.chatthread.create(
            data={
                "funding_request_id": funding_request_id,
                "student_id": student_id,
                "title": title,
                "status": "active",
            }
        )
    
    async def get_thread(self, thread_id: int) -> Optional[ChatThread]:
        """Get a thread by ID."""
        return await self.client.chatthread.find_unique(
            where={"id": thread_id}
        )
    
    async def list_threads(
        self,
        student_id: int,
        funding_request_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[ChatThread], int]:
        """List threads with filtering and pagination."""
        where: Dict[str, Any] = {"student_id": student_id}
        
        if funding_request_id:
            where["funding_request_id"] = funding_request_id
        if status:
            where["status"] = status
        
        threads = await self.client.chatthread.find_many(
            where=where,
            order={"created_at": "desc"},
            take=limit,
            skip=offset,
        )
        
        total = await self.client.chatthread.count(where=where)
        
        return threads, total
    
    async def update_thread(
        self,
        thread_id: int,
        **updates: Any,
    ) -> ChatThread:
        """Update a thread."""
        return await self.client.chatthread.update(
            where={"id": thread_id},
            data=updates,
        )
    
    async def update_thread_status(
        self,
        thread_id: int,
        status: Literal["active", "running", "completed", "archived", "failed"],
    ) -> ChatThread:
        """Update thread status."""
        return await self.update_thread(thread_id, status=status)
    
    # =========================================================================
    # Message Operations
    # =========================================================================
    
    async def create_message(
        self,
        thread_id: int,
        role: Literal["user", "assistant", "system", "tool"],
        content: Dict[str, Any],
        message_type: Literal["message", "tool_call", "tool_result"] = "message",
        tool_name: Optional[str] = None,
        tool_payload: Optional[Dict[str, Any]] = None,
    ) -> ChatThreadMessage:
        """Create a new message in a thread."""
        # Get next message index
        last_msg = await self.client.chatthreadmessage.find_first(
            where={"thread_id": thread_id},
            order={"message_idx": "desc"},
        )
        next_idx = (last_msg.message_idx + 1) if last_msg else 0
        
        return await self.client.chatthreadmessage.create(
            data={
                "thread_id": thread_id,
                "message_idx": next_idx,
                "role": role,
                "message_type": message_type,
                "content": json.dumps(content),
                "tool_name": tool_name,
                "tool_payload": json.dumps(tool_payload) if tool_payload else None,
            }
        )
    
    async def get_thread_messages(
        self,
        thread_id: int,
        limit: int = 50,
        before_idx: Optional[int] = None,
    ) -> Tuple[List[ChatThreadMessage], int, bool]:
        """Get messages for a thread with pagination."""
        where: Dict[str, Any] = {"thread_id": thread_id}
        
        if before_idx is not None:
            where["message_idx"] = {"lt": before_idx}
        
        messages = await self.client.chatthreadmessage.find_many(
            where=where,
            order={"message_idx": "desc"},
            take=limit + 1,  # Fetch one extra to check if there's more
        )
        
        has_more = len(messages) > limit
        if has_more:
            messages = messages[:limit]
        
        # Reverse to get chronological order
        messages.reverse()
        
        total = await self.client.chatthreadmessage.count(
            where={"thread_id": thread_id}
        )
        
        return messages, total, has_more
    
    async def get_recent_messages(
        self,
        thread_id: int,
        limit: int = 10,
    ) -> List[ChatThreadMessage]:
        """Get the most recent messages for context."""
        messages = await self.client.chatthreadmessage.find_many(
            where={"thread_id": thread_id},
            order={"message_idx": "desc"},
            take=limit,
        )
        messages.reverse()
        return messages
    
    # =========================================================================
    # Suggestions Operations
    # =========================================================================
    
    async def get_thread_suggestions(self, thread_id: int) -> Optional[List[str]]:
        """Get cached suggestions for a thread."""
        thread = await self.get_thread(thread_id)
        if thread and thread.suggestions:
            return json.loads(thread.suggestions)
        return None
    
    async def save_thread_suggestions(
        self,
        thread_id: int,
        suggestions: List[str],
    ) -> None:
        """Save suggestions for a thread."""
        await self.client.chatthread.update(
            where={"id": thread_id},
            data={"suggestions": json.dumps(suggestions)},
        )
    
    # =========================================================================
    # Job Operations
    # =========================================================================
    
    async def create_job(
        self,
        student_id: int,
        job_type: str,
        thread_id: Optional[int] = None,
        target_type: str = "chat_thread",
        target_id: int = 0,
        input_payload: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4o-mini",
    ) -> AIJob:
        """Create a new AI job."""
        return await self.client.aijob.create(
            data={
                "student_id": student_id,
                "job_type": job_type,
                "status": "queued",
                "progress": 0,
                "target_type": target_type,
                "target_id": target_id,
                "thread_id": thread_id,
                "input_payload": json.dumps(input_payload) if input_payload else None,
                "model": model,
                "trace_id": "",
                "trace_type": "openai",
                "token_input": 0,
                "token_total": 0,
                "cost_input": Decimal("0"),
                "cost_total": Decimal("0"),
            }
        )
    
    async def get_job(self, job_id: int) -> Optional[AIJob]:
        """Get a job by ID."""
        return await self.client.aijob.find_unique(where={"id": job_id})
    
    async def update_job_status(
        self,
        job_id: int,
        status: Literal["queued", "running", "succeeded", "failed", "cancelled"],
        progress: int = 0,
    ) -> AIJob:
        """Update job status."""
        updates: Dict[str, Any] = {"status": status, "progress": progress}
        
        if status == "running":
            updates["started_at"] = datetime.utcnow()
        elif status in ("succeeded", "failed", "cancelled"):
            updates["finished_at"] = datetime.utcnow()
            updates["progress"] = 100 if status == "succeeded" else progress
        
        return await self.client.aijob.update(
            where={"id": job_id},
            data=updates,
        )
    
    async def complete_job(
        self,
        job_id: int,
        result_payload: Dict[str, Any],
        usage: Dict[str, Any],
        trace_id: str = "",
    ) -> AIJob:
        """Mark a job as completed with results."""
        return await self.client.aijob.update(
            where={"id": job_id},
            data={
                "status": "succeeded",
                "progress": 100,
                "finished_at": datetime.utcnow(),
                "result_payload": json.dumps(result_payload),
                "trace_id": trace_id,
                "token_input": usage.get("input_tokens", 0),
                "token_output": usage.get("output_tokens", 0),
                "token_total": usage.get("total_tokens", 0),
                "cost_input": Decimal(str(usage.get("input_cost", 0))),
                "cost_output": Decimal(str(usage.get("output_cost", 0))),
                "cost_total": Decimal(str(usage.get("total_cost", 0))),
            }
        )
    
    async def fail_job(
        self,
        job_id: int,
        error_message: str,
        error_code: Optional[str] = None,
    ) -> AIJob:
        """Mark a job as failed."""
        return await self.client.aijob.update(
            where={"id": job_id},
            data={
                "status": "failed",
                "finished_at": datetime.utcnow(),
                "error_message": error_message,
                "error_code": error_code,
            }
        )
    
    # =========================================================================
    # Memory Operations
    # =========================================================================
    
    async def create_memory(
        self,
        student_id: int,
        memory_type: str,
        content: str,
        content_hash: Optional[str] = None,
        source: Literal["user", "system", "inferred"] = "inferred",
        confidence: float = 0.7,
        expires_at: Optional[datetime] = None,
    ) -> AIMemory:
        """Create a new memory entry."""
        return await self.client.aimemory.create(
            data={
                "student_id": student_id,
                "memory_type": memory_type,
                "content": content,
                "content_hash": content_hash,
                "source": source,
                "confidence": Decimal(str(confidence)),
                "is_active": True,
                "expires_at": expires_at,
            }
        )
    
    async def get_student_memories(
        self,
        student_id: int,
        memory_type: Optional[str] = None,
        active_only: bool = True,
    ) -> List[AIMemory]:
        """Get memories for a student."""
        where: Dict[str, Any] = {"student_id": student_id}
        
        if memory_type:
            where["memory_type"] = memory_type
        if active_only:
            where["is_active"] = True
        
        return await self.client.aimemory.find_many(
            where=where,
            order={"created_at": "desc"},
        )
    
    async def deactivate_memory(self, memory_id: int) -> AIMemory:
        """Deactivate a memory entry."""
        return await self.client.aimemory.update(
            where={"id": memory_id},
            data={"is_active": False},
        )
    
    # =========================================================================
    # Document Operations
    # =========================================================================
    
    async def create_document(
        self,
        student_id: int,
        title: str,
        document_type: str,
        source_file_path: str,
        source_file_hash: str,
    ) -> StudentDocument:
        """Create a new document record."""
        return await self.client.studentdocument.create(
            data={
                "student_id": student_id,
                "title": title,
                "document_type": document_type,
                "source_file_path": source_file_path,
                "source_file_hash": source_file_hash,
                "upload_status": "uploaded",
            }
        )
    
    async def get_document(self, document_id: int) -> Optional[StudentDocument]:
        """Get a document by ID."""
        return await self.client.studentdocument.find_unique(
            where={"id": document_id}
        )
    
    async def list_documents(
        self,
        student_id: int,
        document_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Tuple[List[StudentDocument], int]:
        """List documents with filtering."""
        where: Dict[str, Any] = {"student_id": student_id}
        
        if document_type:
            where["document_type"] = document_type
        if status:
            where["upload_status"] = status
        
        documents = await self.client.studentdocument.find_many(
            where=where,
            order={"created_at": "desc"},
            take=limit,
            skip=offset,
        )
        
        total = await self.client.studentdocument.count(where=where)
        
        return documents, total
    
    async def update_document(
        self,
        document_id: int,
        **updates: Any,
    ) -> StudentDocument:
        """Update a document."""
        return await self.client.studentdocument.update(
            where={"id": document_id},
            data=updates,
        )
    
    async def delete_document(self, document_id: int) -> None:
        """Delete a document."""
        await self.client.studentdocument.delete(where={"id": document_id})
    
    async def get_thread_documents(self, thread_id: int) -> List[StudentDocument]:
        """Get documents generated in a thread (via jobs)."""
        # Find jobs for this thread that generated documents
        jobs = await self.client.aijob.find_many(
            where={
                "thread_id": thread_id,
                "status": "succeeded",
                "target_type": "content_item",
            }
        )
        
        if not jobs:
            return []
        
        # Get documents from job targets
        doc_ids = [j.target_id for j in jobs]
        return await self.client.studentdocument.find_many(
            where={"id": {"in": doc_ids}}
        )
    
    # =========================================================================
    # Attachment Operations
    # =========================================================================
    
    async def create_request_attachment(
        self,
        funding_request_id: int,
        document_id: int,
        purpose: str,
    ) -> FundingRequestAttachment:
        """Create a request-document attachment."""
        return await self.client.fundingrequestattachment.create(
            data={
                "funding_request_id": funding_request_id,
                "student_document_id": document_id,
                "purpose": purpose,
            }
        )
    
    # =========================================================================
    # Funding Request Operations
    # =========================================================================
    
    async def get_funding_request(self, request_id: int) -> Optional[Any]:
        """Get a funding request by ID."""
        return await self.client.fundingrequest.find_unique(
            where={"id": request_id}
        )
    
    async def get_request_context(self, thread_id: int) -> Dict[str, Any]:
        """Get full context for a thread's funding request."""
        thread = await self.get_thread(thread_id)
        if not thread:
            return {}
        
        request = await self.client.fundingrequest.find_unique(
            where={"id": thread.funding_request_id},
            include={
                "professor": {
                    "include": {"institute": True}
                },
            }
        )
        
        if not request:
            return {}
        
        return {
            "request": request,
            "professor": request.professor if hasattr(request, "professor") else None,
            "institute": request.professor.institute if hasattr(request, "professor") and request.professor else None,
        }
    
    # =========================================================================
    # Usage Operations
    # =========================================================================
    
    async def get_student_usage(
        self,
        student_id: int,
        from_date: datetime,
        to_date: datetime,
    ) -> Dict[str, Any]:
        """Get usage statistics for a student."""
        # Aggregate job stats
        jobs = await self.client.aijob.find_many(
            where={
                "student_id": student_id,
                "created_at": {
                    "gte": from_date,
                    "lte": to_date,
                },
            }
        )
        
        token_input = sum(j.token_input or 0 for j in jobs)
        token_output = sum(j.token_output or 0 for j in jobs)
        cost_input = sum(float(j.cost_input or 0) for j in jobs)
        cost_output = sum(float(j.cost_output or 0) for j in jobs)
        
        # Count threads
        threads = await self.client.chatthread.count(
            where={
                "student_id": student_id,
                "created_at": {
                    "gte": from_date,
                    "lte": to_date,
                },
            }
        )
        
        # Count messages
        messages_sent = await self.client.chatthreadmessage.count(
            where={
                "thread": {"student_id": student_id},
                "role": "user",
                "created_at": {
                    "gte": from_date,
                    "lte": to_date,
                },
            }
        )
        
        messages_received = await self.client.chatthreadmessage.count(
            where={
                "thread": {"student_id": student_id},
                "role": "assistant",
                "created_at": {
                    "gte": from_date,
                    "lte": to_date,
                },
            }
        )
        
        # Count documents
        docs = await self.client.studentdocument.count(
            where={
                "student_id": student_id,
                "created_at": {
                    "gte": from_date,
                    "lte": to_date,
                },
            }
        )
        
        return {
            "token_input": token_input,
            "token_output": token_output,
            "token_total": token_input + token_output,
            "cost_input": cost_input,
            "cost_output": cost_output,
            "cost_total": cost_input + cost_output,
            "threads": threads,
            "threads_active": await self.client.chatthread.count(
                where={"student_id": student_id, "status": "active"}
            ),
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "files_uploaded": 0,  # TODO: Track uploads separately
            "files_generated": docs,
            "files_total": docs,
        }
    
    async def get_student_credits(self, student_id: int) -> Dict[str, Any]:
        """Get credit status for a student."""
        # TODO: Integrate with platform backend
        # For now, return mock data
        return {
            "used": 0.0,
            "remaining": 1000.0,
            "total": 1000.0,
            "active": True,
        }
    
    async def get_job_usage(
        self,
        student_id: int,
        from_date: datetime,
        to_date: datetime,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[AIJob], int]:
        """Get job usage records."""
        where = {
            "student_id": student_id,
            "created_at": {
                "gte": from_date,
                "lte": to_date,
            },
        }
        
        jobs = await self.client.aijob.find_many(
            where=where,
            order={"created_at": "desc"},
            take=limit,
            skip=offset,
        )
        
        total = await self.client.aijob.count(where=where)
        
        return jobs, total





