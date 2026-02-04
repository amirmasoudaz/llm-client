# src/agents/orchestrator/helpers.py
"""Helper agents for Dana orchestrator."""

from __future__ import annotations

import json
from typing import List, Optional

from llm_client import OpenAIClient, GPT5Mini

from src.services.db import DatabaseService
from src.agents.orchestrator.prompts import (
    FOLLOW_UP_SUGGESTIONS_PROMPT,
    CHAT_TITLE_PROMPT,
    THREAD_SUMMARIZATION_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
)


class FollowUpAgent:
    """Agent for generating follow-up suggestions."""
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="follow_up_suggestions",
        )
    
    async def generate_suggestions(
        self,
        thread_id: int,
        n: int = 3,
    ) -> List[str]:
        """Generate follow-up suggestions for a thread."""
        # Get thread and recent messages
        thread = await self.db.get_thread(thread_id)
        if not thread:
            return self._default_suggestions()
        
        messages = await self.db.get_recent_messages(thread_id, limit=5)
        
        # Get request status
        request = await self.db.get_funding_request(thread.funding_request_id)
        email = await self.db.client.fundingemail.find_unique(
            where={"funding_request_id": thread.funding_request_id}
        )
        
        # Check for documents
        docs = await self.db.list_documents(
            student_id=int(thread.student_id),
            limit=10,
        )
        has_cv = any(d.document_type == "resume" for d, _ in [docs])
        has_sop = any(d.document_type == "sop" for d, _ in [docs])
        
        # Format conversation
        conversation = "\n".join([
            f"{m.role.upper()}: {json.loads(m.content).get('text', '') if isinstance(m.content, str) else m.content.get('text', '')}"
            for m in messages
        ])
        
        prompt = FOLLOW_UP_SUGGESTIONS_PROMPT.format(
            n=n,
            conversation=conversation,
            request_status=request.status if request else "unknown",
            email_status="sent" if email and email.main_sent else "not sent",
            has_cv=has_cv,
            has_sop=has_sop,
        )
        
        try:
            response = await self.llm.get_response(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                temperature=0.7,
            )
            
            output = response.get("output", [])
            if isinstance(output, list):
                return output[:n]
            elif isinstance(output, dict) and "suggestions" in output:
                return output["suggestions"][:n]
            else:
                return self._default_suggestions()[:n]
                
        except Exception:
            return self._default_suggestions()[:n]
    
    @staticmethod
    def _default_suggestions() -> List[str]:
        """Default suggestions when generation fails."""
        return [
            "Can you review my email draft?",
            "How aligned am I with this professor?",
            "Can you help me improve my CV?",
        ]


class TitleAgent:
    """Agent for generating chat titles."""
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="chat_titles",
        )
    
    async def generate_title(self, thread_id: int) -> str:
        """Generate a title for a thread."""
        messages = await self.db.get_recent_messages(thread_id, limit=3)
        
        if not messages:
            return "New Conversation"
        
        # Format messages
        messages_text = "\n".join([
            f"{m.role.upper()}: {json.loads(m.content).get('text', '')[:200] if isinstance(m.content, str) else str(m.content)[:200]}"
            for m in messages
        ])
        
        prompt = CHAT_TITLE_PROMPT.format(messages=messages_text)
        
        try:
            response = await self.llm.get_response(
                messages=[{"role": "user", "content": prompt}],
                response_format="text",
                temperature=0.3,
            )
            
            title = response.get("output", "").strip()
            # Truncate if too long
            if len(title) > 50:
                title = title[:47] + "..."
            return title or "New Conversation"
            
        except Exception:
            return "New Conversation"
    
    async def update_thread_title(self, thread_id: int) -> str:
        """Generate and update thread title."""
        title = await self.generate_title(thread_id)
        await self.db.update_thread(thread_id, title=title)
        return title


class SummarizationAgent:
    """Agent for summarizing threads for context compression."""
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="thread_summaries",
        )
    
    async def summarize_thread(self, thread_id: int) -> str:
        """Summarize a thread's conversation."""
        messages, total, _ = await self.db.get_thread_messages(thread_id, limit=100)
        
        if not messages:
            return ""
        
        # Format conversation
        conversation = "\n".join([
            f"{m.role.upper()}: {json.loads(m.content).get('text', '') if isinstance(m.content, str) else str(m.content)}"
            for m in messages
        ])
        
        prompt = THREAD_SUMMARIZATION_PROMPT.format(conversation=conversation)
        
        try:
            response = await self.llm.get_response(
                messages=[{"role": "user", "content": prompt}],
                response_format="text",
                temperature=0,
            )
            
            return response.get("output", "").strip()
            
        except Exception:
            return ""
    
    async def update_thread_summary(self, thread_id: int) -> str:
        """Generate and update thread summary."""
        summary = await self.summarize_thread(thread_id)
        if summary:
            await self.db.update_thread(thread_id, summary=summary)
        return summary
    
    async def should_summarize(self, thread_id: int) -> bool:
        """Check if a thread needs summarization."""
        thread = await self.db.get_thread(thread_id)
        if not thread:
            return False
        
        # Count messages
        _, total, _ = await self.db.get_thread_messages(thread_id, limit=1)
        
        # Summarize if more than 20 messages and no recent summary
        return total > 20 and not thread.summary


class MemoryExtractionAgent:
    """Agent for extracting memories from conversations."""
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="memory_extraction",
        )
    
    async def extract_memories(
        self,
        thread_id: int,
        student_id: int,
    ) -> List[dict]:
        """Extract and store memories from a thread."""
        messages = await self.db.get_recent_messages(thread_id, limit=20)
        
        if not messages:
            return []
        
        # Format conversation
        conversation = "\n".join([
            f"{m.role.upper()}: {json.loads(m.content).get('text', '') if isinstance(m.content, str) else str(m.content)}"
            for m in messages
        ])
        
        prompt = MEMORY_EXTRACTION_PROMPT.format(conversation=conversation)
        
        try:
            response = await self.llm.get_response(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                temperature=0,
            )
            
            output = response.get("output", {})
            memories = output.get("memories", []) if isinstance(output, dict) else []
            
            stored = []
            for mem in memories:
                if not isinstance(mem, dict):
                    continue
                    
                memory_type = mem.get("type", "other")
                content = mem.get("content", "")
                confidence = float(mem.get("confidence", 0.7))
                
                if not content or confidence < 0.5:
                    continue
                
                # Store memory
                await self.db.create_memory(
                    student_id=student_id,
                    memory_type=memory_type,
                    content=content,
                    source="inferred",
                    confidence=confidence,
                )
                
                stored.append({
                    "type": memory_type,
                    "content": content,
                    "confidence": confidence,
                })
            
            return stored
            
        except Exception:
            return []





