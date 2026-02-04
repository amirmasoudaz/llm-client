# src/agents/onboarding/data.py
"""Data Onboarding Agent - Collects and validates user background information."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from llm_client import OpenAIClient, GPT5Mini

from src.services.db import DatabaseService
from src.services.storage import StorageService
from src.agents.converter.engine import Converter


class DataField(str, Enum):
    """Required data fields for onboarding."""
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    EMAIL = "email"
    PHONE = "phone"
    DEGREES = "degrees"
    RESEARCH_INTERESTS = "research_interests"
    SKILLS = "skills"
    EXPERIENCES = "experiences"
    PUBLICATIONS = "publications"


@dataclass
class FieldStatus:
    """Status of a data field."""
    field: DataField
    is_complete: bool
    value: Any = None
    missing_details: Optional[List[str]] = None


@dataclass
class OnboardingProgress:
    """Overall onboarding progress."""
    total_fields: int
    completed_fields: int
    missing_fields: List[DataField]
    fields_status: List[FieldStatus]
    percent_complete: float
    is_complete: bool
    next_recommended: Optional[DataField] = None


class DataOnboardingAgent:
    """
    Agent for collecting and validating user background data.
    
    Handles:
    - Checking data completeness
    - Extracting data from uploaded resumes
    - Guiding users through manual data entry
    - Validating and enriching data
    """
    
    # Required fields for basic onboarding
    REQUIRED_FIELDS = [
        DataField.FIRST_NAME,
        DataField.LAST_NAME,
        DataField.EMAIL,
        DataField.DEGREES,
        DataField.RESEARCH_INTERESTS,
    ]
    
    # Optional but recommended fields
    RECOMMENDED_FIELDS = [
        DataField.PHONE,
        DataField.SKILLS,
        DataField.EXPERIENCES,
        DataField.PUBLICATIONS,
    ]
    
    def __init__(
        self,
        db: DatabaseService,
        storage: Optional[StorageService] = None,
    ):
        self.db = db
        self.storage = storage
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="data_onboarding",
        )
    
    async def get_progress(self, student_id: int) -> OnboardingProgress:
        """Get current data onboarding progress."""
        # Get student data
        student = await self.db.client.student.find_unique(
            where={"id": student_id}
        )
        
        # Get onboarding status
        status = await self.db.client.onboardingstatus.find_unique(
            where={"student_id": student_id}
        )
        
        # Get processed resume if exists
        resume = await self.db.client.studentdocument.find_first(
            where={
                "student_id": student_id,
                "document_type": "resume",
                "upload_status": "processed",
            },
            order={"created_at": "desc"},
        )
        
        resume_data = {}
        if resume and resume.processed_content:
            import json
            resume_data = json.loads(resume.processed_content) if isinstance(resume.processed_content, str) else resume.processed_content
        
        # Check each field
        fields_status = []
        missing = []
        
        # Basic info from student record
        basic_fields = {
            DataField.FIRST_NAME: student.first_name if student else None,
            DataField.LAST_NAME: student.last_name if student else None,
            DataField.EMAIL: student.email if student else None,
            DataField.PHONE: student.mobile_number if student else None,
        }
        
        # Enriched fields from resume
        enriched_fields = {
            DataField.DEGREES: resume_data.get("education", []),
            DataField.RESEARCH_INTERESTS: resume_data.get("research_interests", []),
            DataField.SKILLS: resume_data.get("skills", []),
            DataField.EXPERIENCES: resume_data.get("experience", []),
            DataField.PUBLICATIONS: resume_data.get("publications", []),
        }
        
        all_fields = {**basic_fields, **enriched_fields}
        
        for f in self.REQUIRED_FIELDS + self.RECOMMENDED_FIELDS:
            value = all_fields.get(f)
            is_complete = bool(value) and (not isinstance(value, list) or len(value) > 0)
            
            fs = FieldStatus(
                field=f,
                is_complete=is_complete,
                value=value,
            )
            
            if not is_complete and f in self.REQUIRED_FIELDS:
                missing.append(f)
            
            fields_status.append(fs)
        
        completed = len(self.REQUIRED_FIELDS) - len(missing)
        total = len(self.REQUIRED_FIELDS)
        percent = (completed / total * 100) if total > 0 else 0
        
        return OnboardingProgress(
            total_fields=total,
            completed_fields=completed,
            missing_fields=missing,
            fields_status=fields_status,
            percent_complete=percent,
            is_complete=len(missing) == 0,
            next_recommended=missing[0] if missing else None,
        )
    
    async def extract_from_resume(
        self,
        student_id: int,
        document_id: int,
    ) -> Dict[str, Any]:
        """
        Extract user data from an uploaded resume.
        
        Uses the Converter to parse the resume and extract structured data.
        """
        # Get document
        doc = await self.db.get_document(document_id)
        if not doc:
            return {"error": "Document not found"}
        
        if doc.student_id != student_id:
            return {"error": "Access denied"}
        
        # Check if already processed
        if doc.processed_content:
            import json
            return json.loads(doc.processed_content) if isinstance(doc.processed_content, str) else doc.processed_content
        
        # Get file content
        if not self.storage:
            return {"error": "Storage service not available"}
        
        content = await self.storage.get_final(doc.source_file_path)
        
        # Convert to structured data
        converter = Converter()
        result = await converter.convert(
            content=content,
            filename=doc.source_file_path.split("/")[-1],
        )
        
        # Store processed content
        import json
        await self.db.update_document(
            document_id=document_id,
            processed_content=json.dumps(result),
            upload_status="processed",
        )
        
        # Identify missing fields
        missing = []
        if not result.get("education"):
            missing.append("education history")
        if not result.get("research_interests"):
            missing.append("research interests")
        if not result.get("skills"):
            missing.append("skills")
        
        return {
            "extracted": result,
            "missing": missing,
            "suggestions": self._generate_suggestions(result, missing),
        }
    
    def _generate_suggestions(
        self,
        data: Dict[str, Any],
        missing: List[str],
    ) -> List[str]:
        """Generate suggestions for improving profile data."""
        suggestions = []
        
        if "education history" in missing:
            suggestions.append("Please add your educational background, including degrees, institutions, and graduation years.")
        
        if "research interests" in missing:
            suggestions.append("What are your main research interests? List 3-5 specific topics or areas.")
        
        if "skills" in missing:
            suggestions.append("List your technical skills, programming languages, and methodologies.")
        
        # Check for quality issues
        education = data.get("education", [])
        if education:
            for edu in education:
                if not edu.get("gpa"):
                    suggestions.append("Consider adding your GPA if it's competitive (3.5+).")
                    break
        
        publications = data.get("publications", [])
        if not publications:
            suggestions.append("If you have any publications, presentations, or posters, adding them can strengthen your profile.")
        
        return suggestions
    
    async def update_field(
        self,
        student_id: int,
        field: DataField,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Update a specific field for the user.
        
        Validates the value and stores it appropriately.
        """
        # Validate value
        is_valid, error = self._validate_field(field, value)
        if not is_valid:
            return {"success": False, "error": error}
        
        # Update based on field type
        if field in [DataField.FIRST_NAME, DataField.LAST_NAME, DataField.EMAIL, DataField.PHONE]:
            # Update student record
            update_data = {field.value: value}
            await self.db.client.student.update(
                where={"id": student_id},
                data=update_data,
            )
        else:
            # Update onboarding status with pending data
            import json
            status = await self.db.client.onboardingstatus.find_unique(
                where={"student_id": student_id}
            )
            
            pending = {}
            if status and status.pending_data:
                pending = json.loads(status.pending_data) if isinstance(status.pending_data, str) else status.pending_data
            
            pending[field.value] = value
            
            await self.db.client.onboardingstatus.upsert(
                where={"student_id": student_id},
                create={
                    "student_id": student_id,
                    "pending_data": json.dumps(pending),
                },
                update={
                    "pending_data": json.dumps(pending),
                },
            )
        
        return {"success": True, "field": field.value, "value": value}
    
    def _validate_field(
        self,
        field: DataField,
        value: Any,
    ) -> tuple[bool, Optional[str]]:
        """Validate a field value."""
        if field == DataField.EMAIL:
            import re
            if not re.match(r"[^@]+@[^@]+\.[^@]+", str(value)):
                return False, "Invalid email format"
        
        if field in [DataField.FIRST_NAME, DataField.LAST_NAME]:
            if not value or len(str(value).strip()) < 1:
                return False, "Name cannot be empty"
        
        if field == DataField.DEGREES:
            if not isinstance(value, list):
                return False, "Degrees must be a list"
        
        if field == DataField.RESEARCH_INTERESTS:
            if isinstance(value, str):
                # Convert comma-separated string to list
                value = [v.strip() for v in value.split(",") if v.strip()]
            if not isinstance(value, list) or len(value) == 0:
                return False, "Please provide at least one research interest"
        
        return True, None
    
    async def get_conversation_response(
        self,
        student_id: int,
        user_message: str,
        current_field: Optional[DataField] = None,
    ) -> Dict[str, Any]:
        """
        Handle data onboarding conversation.
        
        Returns appropriate response based on progress and user input.
        """
        progress = await self.get_progress(student_id)
        
        message_lower = user_message.lower()
        
        if progress.is_complete:
            return {
                "message": "Your profile is complete! You're ready to start reaching out to professors.",
                "action": "complete",
                "progress": progress.percent_complete,
            }
        
        # Check if user is providing data for current field
        if current_field:
            result = await self.update_field(student_id, current_field, user_message)
            if result["success"]:
                # Move to next field
                next_field = progress.next_recommended
                if next_field:
                    return {
                        "message": f"Got it! Now, what's your {next_field.value.replace('_', ' ')}?",
                        "action": "prompt_field",
                        "current_field": next_field.value,
                        "progress": progress.percent_complete,
                    }
                else:
                    return {
                        "message": "Your basic profile is complete!",
                        "action": "complete",
                        "progress": 100,
                    }
            else:
                return {
                    "message": f"That doesn't look right: {result['error']}. Please try again.",
                    "action": "retry_field",
                    "current_field": current_field.value,
                    "error": result["error"],
                }
        
        # Prompt for next missing field
        if progress.next_recommended:
            field_prompts = {
                DataField.FIRST_NAME: "What's your first name?",
                DataField.LAST_NAME: "What's your last name?",
                DataField.EMAIL: "What's your email address?",
                DataField.PHONE: "What's your phone number? (optional)",
                DataField.DEGREES: "Tell me about your education. What degrees do you have or are pursuing?",
                DataField.RESEARCH_INTERESTS: "What are your main research interests? (list 3-5 topics)",
                DataField.SKILLS: "What technical skills do you have? (programming, tools, methods)",
                DataField.EXPERIENCES: "Do you have any research or work experience to add?",
                DataField.PUBLICATIONS: "Do you have any publications, presentations, or posters?",
            }
            
            prompt = field_prompts.get(
                progress.next_recommended,
                f"Please provide your {progress.next_recommended.value.replace('_', ' ')}"
            )
            
            return {
                "message": prompt,
                "action": "prompt_field",
                "current_field": progress.next_recommended.value,
                "progress": progress.percent_complete,
            }
        
        return {
            "message": "Let's complete your profile. This helps us create better emails and CVs for you.",
            "action": "start_onboarding",
            "progress": progress.percent_complete,
        }





