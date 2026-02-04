# src/agents/orchestrator/context.py
"""Context builder for Dana orchestrator."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.services.db import DatabaseService
from src.schemas.context import (
    OrchestrationContext,
    UserContext,
    ProfessorContext,
    RequestContext,
    EmailContext,
    MemoryContext,
    MemoryItem,
    ConversationContext,
    ConversationMessage,
    UserDegree,
    UserExperience,
    UserPublication,
)


class ContextBuilder:
    """
    Builds orchestration context from various data sources.
    
    Assembles user, professor, request, memory, and conversation context
    for the switchboard agent.
    """
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    async def build(
        self,
        thread_id: int,
        current_message: str = "",
        document_ids: Optional[List[int]] = None,
        additional_contexts: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationContext:
        """Build complete orchestration context for a thread."""
        # Get thread info
        thread = await self.db.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found")
        
        # Load all context components in parallel where possible
        user_ctx = await self.load_user_context(thread.student_id)
        request_ctx = await self.load_request_context(thread.funding_request_id)
        professor_ctx = await self.load_professor_context(thread.funding_request_id)
        memory_ctx = await self.load_memory_context(thread.student_id)
        conversation_ctx = await self.load_conversation_context(thread_id)
        email_ctx = await self.load_email_context(thread.funding_request_id)
        
        return OrchestrationContext(
            thread_id=thread_id,
            funding_request_id=thread.funding_request_id,
            user=user_ctx,
            professor=professor_ctx,
            request=request_ctx,
            email=email_ctx,
            memory=memory_ctx,
            conversation=conversation_ctx,
            current_message=current_message,
            document_ids=document_ids or [],
            additional_contexts=additional_contexts or {},
        )
    
    async def load_user_context(self, student_id: int) -> UserContext:
        """Load user context from database."""
        # Get student data
        student = await self.db.client.student.find_unique(
            where={"id": student_id}
        )
        
        if not student:
            raise ValueError(f"Student {student_id} not found")
        
        # Get onboarding data if exists
        onboarding_data = {}
        # Try to get from metas table (existing platform structure)
        # This would need to be implemented based on actual data structure
        
        # Parse degrees, experiences, publications from processed documents
        degrees = []
        experiences = []
        publications = []
        research_interests = []
        skills = []
        
        # Try to get from processed resume document
        resume = await self.db.client.studentdocument.find_first(
            where={
                "student_id": student_id,
                "document_type": "resume",
                "upload_status": "processed",
            },
            order={"created_at": "desc"},
        )
        
        if resume and resume.processed_content:
            content = json.loads(resume.processed_content) if isinstance(resume.processed_content, str) else resume.processed_content
            
            # Extract degrees
            for edu in content.get("education", []):
                degrees.append(UserDegree(
                    level=edu.get("degree_level", ""),
                    field=edu.get("field_of_study", ""),
                    institution=edu.get("institution", ""),
                    gpa=edu.get("gpa"),
                    graduation_year=edu.get("graduation_year"),
                    thesis_title=edu.get("thesis_title"),
                    achievements=edu.get("achievements", []),
                ))
            
            # Extract experiences
            for exp in content.get("experience", []):
                experiences.append(UserExperience(
                    title=exp.get("title", ""),
                    organization=exp.get("organization", ""),
                    location=exp.get("location"),
                    start_date=exp.get("start_date"),
                    end_date=exp.get("end_date"),
                    description=exp.get("description"),
                    achievements=exp.get("achievements", []),
                ))
            
            # Extract publications
            for pub in content.get("publications", []):
                publications.append(UserPublication(
                    title=pub.get("title", ""),
                    authors=pub.get("authors", []),
                    venue=pub.get("venue", ""),
                    year=pub.get("year", 0),
                    type=pub.get("type", "paper"),
                    url=pub.get("url"),
                ))
            
            # Extract skills and interests
            skills = content.get("skills", [])
            research_interests = content.get("research_interests", [])
        
        return UserContext(
            student_id=student_id,
            first_name=student.first_name or "",
            last_name=student.last_name or "",
            email=student.email,
            phone=student.mobile_number,
            degrees=degrees,
            experiences=experiences,
            publications=publications,
            research_interests=research_interests,
            skills=skills,
            onboarding_data=onboarding_data,
        )
    
    async def load_professor_context(self, funding_request_id: int) -> ProfessorContext:
        """Load professor context from database."""
        request = await self.db.client.fundingrequest.find_unique(
            where={"id": funding_request_id},
            include={
                "professor": {
                    "include": {"institute": True}
                }
            }
        )
        
        if not request or not request.professor:
            raise ValueError(f"Professor not found for request {funding_request_id}")
        
        prof = request.professor
        inst = prof.institute
        
        # Parse JSON fields
        research_areas = json.loads(prof.research_areas) if isinstance(prof.research_areas, str) else (prof.research_areas or [])
        area_of_expertise = json.loads(prof.area_of_expertise) if isinstance(prof.area_of_expertise, str) else (prof.area_of_expertise or [])
        categories = json.loads(prof.categories) if isinstance(prof.categories, str) else (prof.categories or [])
        others = json.loads(prof.others) if isinstance(prof.others, str) else (prof.others or {})
        
        return ProfessorContext(
            professor_id=prof.id,
            first_name=prof.first_name,
            last_name=prof.last_name,
            full_name=prof.full_name,
            occupation=prof.occupation,
            department=prof.department,
            email_address=prof.email_address,
            url=prof.url,
            research_areas=research_areas if isinstance(research_areas, list) else [],
            area_of_expertise=area_of_expertise if isinstance(area_of_expertise, list) else [],
            categories=categories if isinstance(categories, list) else [],
            credentials=prof.credentials,
            institution_name=inst.institution_name if inst else "",
            institution_department=inst.department_name if inst else None,
            institution_city=inst.city if inst else None,
            institution_country=inst.country if inst else None,
            others=others if isinstance(others, dict) else {},
        )
    
    async def load_request_context(self, funding_request_id: int) -> RequestContext:
        """Load funding request context."""
        request = await self.db.client.fundingrequest.find_unique(
            where={"id": funding_request_id}
        )
        
        if not request:
            raise ValueError(f"Funding request {funding_request_id} not found")
        
        # Parse JSON fields
        attachments = json.loads(request.attachments) if isinstance(request.attachments, str) else request.attachments
        template_ids = json.loads(request.student_template_ids) if isinstance(request.student_template_ids, str) else request.student_template_ids
        
        return RequestContext(
            request_id=int(request.id),
            student_id=int(request.student_id),
            professor_id=int(request.professor_id),
            match_status=request.match_status,
            status=request.status or "0",
            research_interest=request.research_interest,
            paper_title=request.paper_title,
            journal=request.journal,
            year=request.year,
            research_connection=request.research_connection,
            email_subject=request.email_subject,
            email_content=request.email_content,
            attachments=attachments,
            template_ids=template_ids,
            created_at=request.created_at,
        )
    
    async def load_email_context(self, funding_request_id: int) -> Optional[EmailContext]:
        """Load email context if exists."""
        email = await self.db.client.fundingemail.find_unique(
            where={"funding_request_id": funding_request_id}
        )
        
        if not email:
            return None
        
        return EmailContext(
            email_id=email.id,
            subject=email.main_email_subject,
            body=email.main_email_body,
            sent=email.main_sent,
            sent_at=email.main_sent_at,
            professor_replied=email.professor_replied,
            professor_replied_at=email.professor_replied_at,
            professor_reply_body=email.professor_reply_body,
            reminder_one_sent=email.reminder_one_sent,
            reminder_two_sent=email.reminder_two_sent,
            reminder_three_sent=email.reminder_three_sent,
        )
    
    async def load_memory_context(self, student_id: int) -> MemoryContext:
        """Load user memory context."""
        memories = await self.db.get_student_memories(student_id, active_only=True)
        
        memory_items = []
        tone_preferences = []
        dos_and_donts = []
        instructions = []
        goals = []
        bio_facts = []
        
        for mem in memories:
            item = MemoryItem(
                memory_id=int(mem.id),
                memory_type=mem.memory_type,
                content=mem.content,
                source=mem.source,
                confidence=float(mem.confidence),
                created_at=mem.created_at,
            )
            memory_items.append(item)
            
            # Categorize
            if mem.memory_type == "tone":
                tone_preferences.append(item)
            elif mem.memory_type == "do_dont":
                dos_and_donts.append(item)
            elif mem.memory_type == "instruction":
                instructions.append(item)
            elif mem.memory_type == "goal":
                goals.append(item)
            elif mem.memory_type == "bio":
                bio_facts.append(item)
        
        return MemoryContext(
            student_id=student_id,
            memories=memory_items,
            tone_preferences=tone_preferences,
            dos_and_donts=dos_and_donts,
            instructions=instructions,
            goals=goals,
            bio_facts=bio_facts,
        )
    
    async def load_conversation_context(
        self,
        thread_id: int,
        max_messages: int = 20,
    ) -> ConversationContext:
        """Load conversation history context."""
        thread = await self.db.get_thread(thread_id)
        if not thread:
            return ConversationContext(thread_id=thread_id)
        
        messages = await self.db.get_recent_messages(thread_id, limit=max_messages)
        
        conversation_messages = []
        for msg in messages:
            content = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            text = content.get("text", "") if isinstance(content, dict) else str(content)
            
            conversation_messages.append(ConversationMessage(
                role=msg.role,
                content=text,
                message_idx=msg.message_idx,
            ))
        
        return ConversationContext(
            thread_id=thread_id,
            messages=conversation_messages,
            summary=thread.summary,
            total_messages=len(messages),
        )
    
    async def refresh_context(
        self,
        existing_context: OrchestrationContext,
        refresh_components: Optional[List[str]] = None,
    ) -> OrchestrationContext:
        """Refresh specific components of an existing context."""
        refresh = refresh_components or ["conversation"]
        
        if "user" in refresh:
            existing_context.user = await self.load_user_context(existing_context.user.student_id)
        
        if "professor" in refresh:
            existing_context.professor = await self.load_professor_context(existing_context.funding_request_id)
        
        if "request" in refresh:
            existing_context.request = await self.load_request_context(existing_context.funding_request_id)
        
        if "email" in refresh:
            existing_context.email = await self.load_email_context(existing_context.funding_request_id)
        
        if "memory" in refresh:
            existing_context.memory = await self.load_memory_context(existing_context.user.student_id)
        
        if "conversation" in refresh:
            existing_context.conversation = await self.load_conversation_context(existing_context.thread_id)
        
        return existing_context





