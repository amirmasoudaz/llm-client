# src/schemas/context.py
"""Context schemas for orchestration."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field


# ============================================================================
# User Context
# ============================================================================

class UserDegree(BaseModel):
    """User degree information."""
    level: str  # Bachelor, Master, PhD
    field: str
    institution: str
    gpa: Optional[float] = None
    graduation_year: Optional[int] = None
    thesis_title: Optional[str] = None
    achievements: Optional[List[str]] = None


class UserExperience(BaseModel):
    """User work/research experience."""
    title: str
    organization: str
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    achievements: Optional[List[str]] = None


class UserPublication(BaseModel):
    """User publication."""
    title: str
    authors: List[str]
    venue: str  # Journal/conference name
    year: int
    type: str  # journal, conference, thesis, etc.
    url: Optional[str] = None


class UserContext(BaseModel):
    """Complete user context for orchestration."""
    # Basic info
    student_id: int
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    
    # Background
    degrees: List[UserDegree] = Field(default_factory=list)
    experiences: List[UserExperience] = Field(default_factory=list)
    publications: List[UserPublication] = Field(default_factory=list)
    
    # Research
    research_interests: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    
    # Social/contact
    linkedin_url: Optional[str] = None
    website_url: Optional[str] = None
    orcid_url: Optional[str] = None
    github_url: Optional[str] = None
    
    # Onboarding data
    onboarding_data: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


# ============================================================================
# Professor Context
# ============================================================================

class ProfessorContext(BaseModel):
    """Professor context for orchestration."""
    professor_id: int
    first_name: str
    last_name: str
    full_name: str
    
    # Position
    occupation: Optional[str] = None
    department: str
    
    # Contact
    email_address: str
    url: Optional[str] = None
    
    # Research
    research_areas: List[str] = Field(default_factory=list)
    area_of_expertise: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    credentials: Optional[str] = None
    
    # Institution
    institution_name: str
    institution_department: Optional[str] = None
    institution_city: Optional[str] = None
    institution_country: Optional[str] = None
    
    # Additional
    others: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


# ============================================================================
# Request Context
# ============================================================================

class RequestContext(BaseModel):
    """Funding request context for orchestration."""
    request_id: int
    student_id: int
    professor_id: int
    
    # Request details
    match_status: int
    status: str
    
    # Research connection
    research_interest: Optional[str] = None
    paper_title: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    research_connection: Optional[str] = None
    
    # Email
    email_subject: Optional[str] = None
    email_content: Optional[str] = None
    
    # Attachments
    attachments: Optional[Dict[str, Any]] = None
    template_ids: Optional[Dict[str, Any]] = None
    
    # Timestamps
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# Email Context
# ============================================================================

class EmailContext(BaseModel):
    """Email context for email operations."""
    email_id: Optional[int] = None
    
    # Main email
    subject: Optional[str] = None
    body: Optional[str] = None
    sent: bool = False
    sent_at: Optional[datetime] = None
    
    # Reply tracking
    professor_replied: bool = False
    professor_replied_at: Optional[datetime] = None
    professor_reply_body: Optional[str] = None
    
    # Reminders
    reminder_one_sent: bool = False
    reminder_two_sent: bool = False
    reminder_three_sent: bool = False


# ============================================================================
# Memory Context
# ============================================================================

class MemoryItem(BaseModel):
    """Single memory item."""
    memory_id: int
    memory_type: str
    content: str
    source: str
    confidence: float
    created_at: datetime


class MemoryContext(BaseModel):
    """User memory context."""
    student_id: int
    memories: List[MemoryItem] = Field(default_factory=list)
    
    # Categorized memories
    tone_preferences: List[MemoryItem] = Field(default_factory=list)
    dos_and_donts: List[MemoryItem] = Field(default_factory=list)
    instructions: List[MemoryItem] = Field(default_factory=list)
    goals: List[MemoryItem] = Field(default_factory=list)
    bio_facts: List[MemoryItem] = Field(default_factory=list)


# ============================================================================
# Conversation Context
# ============================================================================

class ConversationMessage(BaseModel):
    """Simplified message for context."""
    role: str
    content: str
    message_idx: int


class ConversationContext(BaseModel):
    """Conversation context."""
    thread_id: int
    messages: List[ConversationMessage] = Field(default_factory=list)
    summary: Optional[str] = None
    total_messages: int = 0


# ============================================================================
# Full Orchestration Context
# ============================================================================

class OrchestrationContext(BaseModel):
    """
    Complete orchestration context.
    
    Assembled by ContextBuilder for the switchboard agent.
    """
    # Thread info
    thread_id: int
    funding_request_id: int
    
    # Core contexts
    user: UserContext
    professor: ProfessorContext
    request: RequestContext
    
    # Optional contexts
    email: Optional[EmailContext] = None
    memory: Optional[MemoryContext] = None
    conversation: Optional[ConversationContext] = None
    
    # Current message
    current_message: str = ""
    document_ids: List[int] = Field(default_factory=list)
    additional_contexts: Dict[str, Any] = Field(default_factory=dict)
    
    def to_prompt_context(self) -> str:
        """Format context for LLM prompt."""
        parts = []
        
        # User context
        parts.append(f"## User: {self.user.first_name} {self.user.last_name}")
        if self.user.research_interests:
            parts.append(f"Research interests: {', '.join(self.user.research_interests)}")
        if self.user.degrees:
            deg = self.user.degrees[0]
            parts.append(f"Current degree: {deg.level} in {deg.field} at {deg.institution}")
        
        # Professor context
        parts.append(f"\n## Professor: {self.professor.full_name}")
        parts.append(f"Department: {self.professor.department} at {self.professor.institution_name}")
        if self.professor.research_areas:
            parts.append(f"Research areas: {', '.join(self.professor.research_areas)}")
        
        # Request context
        parts.append(f"\n## Request Status: {self.request.status}")
        if self.request.research_interest:
            parts.append(f"Research interest: {self.request.research_interest}")
        if self.request.email_subject:
            parts.append(f"Email subject: {self.request.email_subject}")
        
        # Memory context
        if self.memory:
            if self.memory.instructions:
                parts.append("\n## User Instructions:")
                for m in self.memory.instructions[:3]:
                    parts.append(f"- {m.content}")
            if self.memory.tone_preferences:
                parts.append("\n## Tone Preferences:")
                for m in self.memory.tone_preferences[:2]:
                    parts.append(f"- {m.content}")
        
        return "\n".join(parts)





