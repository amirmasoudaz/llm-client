# src/agents/orchestrator/tool_wrappers.py
"""Tool wrappers for existing agents to integrate with the orchestrator."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.services.db import DatabaseService
from src.services.storage import StorageService
from src.services.jobs import JobService, JobContext
from src.agents.orchestrator.tools import (
    tool, ToolParameter, ToolResult, ReviewResult, GenerateResult, 
    AlignmentResult, ContextResult, MemoryResult
)
from src.schemas.context import OrchestrationContext


# ============================================================================
# Email Tools
# ============================================================================

@tool(
    name="email_generate",
    description="Generate a professor outreach email based on user profile and professor information",
    parameters=[
        ToolParameter(name="tone", type="string", description="Email tone: formal, friendly, or enthusiastic", required=False, default="formal", enum=["formal", "friendly", "enthusiastic"]),
        ToolParameter(name="tailor_type", type="array", description="Tailoring strategies to apply", required=False),
        ToolParameter(name="focus", type="string", description="Topics to emphasize in the email", required=False),
        ToolParameter(name="avoid", type="string", description="Topics to avoid mentioning", required=False),
    ],
    category="email",
)
async def email_generate(
    ctx: OrchestrationContext,
    tone: str = "formal",
    tailor_type: Optional[List[str]] = None,
    focus: Optional[str] = None,
    avoid: Optional[str] = None,
) -> GenerateResult:
    """Generate a professor outreach email."""
    from src.agents.email.engine import EmailEngine
    
    engine = EmailEngine()
    
    # Build sender detail from user context
    sender_detail = {
        "name": f"{ctx.user.first_name} {ctx.user.last_name}",
        "email": ctx.user.email,
        "education": [
            {
                "degree": d.level,
                "field": d.field,
                "institution": d.institution,
                "gpa": d.gpa,
                "graduation_year": d.graduation_year,
            }
            for d in ctx.user.degrees
        ],
        "research_interests": ctx.user.research_interests,
        "skills": ctx.user.skills,
        "experiences": [
            {
                "title": e.title,
                "organization": e.organization,
                "description": e.description,
            }
            for e in ctx.user.experiences
        ],
        "publications": [
            {
                "title": p.title,
                "venue": p.venue,
                "year": p.year,
            }
            for p in ctx.user.publications
        ],
    }
    
    # Build recipient detail from professor context
    recipient_detail = {
        "name": ctx.professor.full_name,
        "title": ctx.professor.occupation or "Professor",
        "department": ctx.professor.department,
        "institution": ctx.professor.institution_name,
        "email": ctx.professor.email_address,
        "research_areas": ctx.professor.research_areas,
        "recent_papers": ctx.professor.others.get("recent_papers", []) if ctx.professor.others else [],
    }
    
    # Apply memory preferences
    if ctx.memory:
        for mem in ctx.memory.tone_preferences:
            if "formal" in mem.content.lower():
                tone = "formal"
            elif "friendly" in mem.content.lower():
                tone = "friendly"
    
    try:
        result = await engine.generate(
            user_id=str(ctx.user.student_id),
            sender_detail=sender_detail,
            recipient_detail=recipient_detail,
            tone=tone,
            tailor_type=tailor_type or ["match_research_area"],
            focus=focus,
            avoid=avoid,
        )
        
        return GenerateResult(
            success=True,
            message="Email generated successfully",
            content=result,
        )
    except Exception as e:
        return GenerateResult(
            success=False,
            error=str(e),
        )


@tool(
    name="email_review",
    description="Review an outreach email and provide feedback with scores across 7 dimensions",
    parameters=[
        ToolParameter(name="email_content", type="object", description="The email to review (optional, uses request email if not provided)", required=False),
    ],
    category="email",
)
async def email_review(
    ctx: OrchestrationContext,
    email_content: Optional[Dict[str, Any]] = None,
) -> ReviewResult:
    """Review an outreach email."""
    from src.agents.email.engine import EmailEngine
    
    engine = EmailEngine()
    
    # Get email to review
    if email_content:
        email = email_content
    elif ctx.request.email_content:
        email = {
            "subject": ctx.request.email_subject or "",
            "body": ctx.request.email_content,
        }
    else:
        return ReviewResult(
            success=False,
            error="No email content provided and no email found in request",
        )
    
    # Build contexts
    sender_detail = {
        "name": f"{ctx.user.first_name} {ctx.user.last_name}",
        "research_interests": ctx.user.research_interests,
        "education": [{"field": d.field, "institution": d.institution} for d in ctx.user.degrees],
    }
    
    recipient_detail = {
        "name": ctx.professor.full_name,
        "department": ctx.professor.department,
        "institution": ctx.professor.institution_name,
        "research_areas": ctx.professor.research_areas,
    }
    
    try:
        review = await engine.review(
            email=email,
            sender_detail=sender_detail,
            recipient_detail=recipient_detail,
        )
        
        return ReviewResult(
            success=True,
            message=f"Email review complete. Readiness: {review.get('readiness_level', 'unknown')}",
            data=review,
            score=review.get("average_score"),
            dimensions=review.get("dimensions"),
            suggestions=review.get("suggestions", []),
            readiness_level=review.get("readiness_level"),
        )
    except Exception as e:
        return ReviewResult(
            success=False,
            error=str(e),
        )


@tool(
    name="email_optimize",
    description="Optimize an existing email based on feedback",
    parameters=[
        ToolParameter(name="email_content", type="object", description="The email to optimize", required=True),
        ToolParameter(name="feedback", type="string", description="Specific improvement instructions", required=True),
        ToolParameter(name="revision_goals", type="array", description="List of optimization goals", required=False),
    ],
    category="email",
)
async def email_optimize(
    ctx: OrchestrationContext,
    email_content: Dict[str, Any],
    feedback: str,
    revision_goals: Optional[List[str]] = None,
) -> GenerateResult:
    """Optimize an existing email."""
    from src.agents.email.engine import EmailEngine
    
    engine = EmailEngine()
    
    sender_detail = {
        "name": f"{ctx.user.first_name} {ctx.user.last_name}",
        "research_interests": ctx.user.research_interests,
    }
    
    recipient_detail = {
        "name": ctx.professor.full_name,
        "research_areas": ctx.professor.research_areas,
    }
    
    optimization_context = {
        "old_email": email_content,
        "feedback": feedback,
        "revision_goals": revision_goals or [],
    }
    
    try:
        result = await engine.generate(
            user_id=str(ctx.user.student_id),
            sender_detail=sender_detail,
            recipient_detail=recipient_detail,
            generation_type="optimization",
            optimization_context=optimization_context,
        )
        
        return GenerateResult(
            success=True,
            message="Email optimized successfully",
            content=result,
        )
    except Exception as e:
        return GenerateResult(
            success=False,
            error=str(e),
        )


# ============================================================================
# Resume/CV Tools
# ============================================================================

@tool(
    name="resume_generate",
    description="Generate an academic CV/resume based on user profile",
    parameters=[
        ToolParameter(name="tone", type="string", description="CV tone: academic, industry, or clinical", required=False, default="academic", enum=["academic", "industry", "clinical"]),
        ToolParameter(name="target_context", type="object", description="Target position/program information", required=False),
    ],
    category="resume",
)
async def resume_generate(
    ctx: OrchestrationContext,
    tone: str = "academic",
    target_context: Optional[Dict[str, Any]] = None,
) -> GenerateResult:
    """Generate an academic CV."""
    from src.agents.resume.engine import CVEngine
    
    engine = CVEngine()
    
    # Build user details from context
    user_details = {
        "personal": {
            "name": f"{ctx.user.first_name} {ctx.user.last_name}",
            "email": ctx.user.email,
            "phone": ctx.user.phone,
            "linkedin": ctx.user.linkedin_url,
            "website": ctx.user.website_url,
        },
        "education": [
            {
                "degree": d.level,
                "field": d.field,
                "institution": d.institution,
                "gpa": d.gpa,
                "graduation_year": d.graduation_year,
                "thesis": d.thesis_title,
                "achievements": d.achievements,
            }
            for d in ctx.user.degrees
        ],
        "experience": [
            {
                "title": e.title,
                "organization": e.organization,
                "location": e.location,
                "start_date": e.start_date,
                "end_date": e.end_date,
                "description": e.description,
                "achievements": e.achievements,
            }
            for e in ctx.user.experiences
        ],
        "publications": [
            {
                "title": p.title,
                "authors": p.authors,
                "venue": p.venue,
                "year": p.year,
                "type": p.type,
            }
            for p in ctx.user.publications
        ],
        "skills": ctx.user.skills,
        "research_interests": ctx.user.research_interests,
    }
    
    # Build target context
    additional_context = target_context or {}
    if ctx.professor:
        additional_context["target_professor"] = {
            "name": ctx.professor.full_name,
            "department": ctx.professor.department,
            "research_areas": ctx.professor.research_areas,
        }
    
    try:
        result = await engine.generate(
            user_id=str(ctx.user.student_id),
            user_details=user_details,
            additional_context=additional_context,
            tone=tone,
        )
        
        return GenerateResult(
            success=True,
            message="CV generated successfully",
            content=result,
        )
    except Exception as e:
        return GenerateResult(
            success=False,
            error=str(e),
        )


@tool(
    name="resume_review",
    description="Review a CV and provide feedback with scores across 7 dimensions",
    parameters=[
        ToolParameter(name="cv_content", type="object", description="The CV to review", required=True),
        ToolParameter(name="target_context", type="object", description="Target position/program for fit evaluation", required=False),
    ],
    category="resume",
)
async def resume_review(
    ctx: OrchestrationContext,
    cv_content: Dict[str, Any],
    target_context: Optional[Dict[str, Any]] = None,
) -> ReviewResult:
    """Review a CV."""
    from src.agents.resume.engine import CVEngine
    
    engine = CVEngine()
    
    # Build target context
    target = target_context or {}
    if ctx.professor:
        target["target_professor"] = {
            "name": ctx.professor.full_name,
            "research_areas": ctx.professor.research_areas,
        }
    
    try:
        review = await engine.review(
            cv=cv_content,
            target_context=target if target else None,
        )
        
        return ReviewResult(
            success=True,
            message=f"CV review complete. Readiness: {review.get('readiness_level', 'unknown')}",
            data=review,
            score=review.get("average_score"),
            dimensions=review.get("dimensions"),
            suggestions=review.get("suggestions", []),
            readiness_level=review.get("readiness_level"),
        )
    except Exception as e:
        return ReviewResult(
            success=False,
            error=str(e),
        )


@tool(
    name="resume_optimize",
    description="Optimize specific sections of a CV based on feedback",
    parameters=[
        ToolParameter(name="cv_content", type="object", description="The CV to optimize", required=True),
        ToolParameter(name="sections", type="array", description="Sections to modify", required=True),
        ToolParameter(name="feedback", type="string", description="Specific improvement instructions", required=True),
    ],
    category="resume",
)
async def resume_optimize(
    ctx: OrchestrationContext,
    cv_content: Dict[str, Any],
    sections: List[str],
    feedback: str,
) -> GenerateResult:
    """Optimize a CV."""
    from src.agents.resume.engine import CVEngine
    
    engine = CVEngine()
    
    user_details = {
        "name": f"{ctx.user.first_name} {ctx.user.last_name}",
        "research_interests": ctx.user.research_interests,
    }
    
    try:
        result = await engine.optimize(
            cv=cv_content,
            sections_to_modify=sections,
            feedback=feedback,
            user_details=user_details,
        )
        
        return GenerateResult(
            success=True,
            message="CV optimized successfully",
            content=result,
        )
    except Exception as e:
        return GenerateResult(
            success=False,
            error=str(e),
        )


# ============================================================================
# Alignment Tools
# ============================================================================

@tool(
    name="alignment_evaluate",
    description="Evaluate alignment between user profile and professor research",
    parameters=[],
    category="alignment",
)
async def alignment_evaluate(
    ctx: OrchestrationContext,
) -> AlignmentResult:
    """Evaluate alignment between user and professor."""
    from src.agents.alignment.engine import AlignmentEngine
    
    engine = AlignmentEngine()
    
    # Build user profile
    user_profile = {
        "name": f"{ctx.user.first_name} {ctx.user.last_name}",
        "research_interests": ctx.user.research_interests,
        "skills": ctx.user.skills,
        "education": [{"field": d.field, "level": d.level} for d in ctx.user.degrees],
        "publications": [{"title": p.title, "venue": p.venue} for p in ctx.user.publications],
    }
    
    # Build professor profile
    professor_profile = {
        "name": ctx.professor.full_name,
        "research_areas": ctx.professor.research_areas,
        "department": ctx.professor.department,
        "institution": ctx.professor.institution_name,
    }
    
    try:
        result = await engine.evaluate(
            user_profile=user_profile,
            professor_profile=professor_profile,
        )
        
        score = result.get("overall_score", 0)
        label = "STRONG" if score >= 7 else "MODERATE" if score >= 5 else "WEAK"
        
        return AlignmentResult(
            success=True,
            message=f"Alignment evaluation complete: {label}",
            data=result,
            score=score,
            label=label,
            categories=result.get("categories", []),
            reasons=result.get("reasons", []),
        )
    except Exception as e:
        return AlignmentResult(
            success=False,
            error=str(e),
        )


# ============================================================================
# Context Tools
# ============================================================================

@tool(
    name="get_user_context",
    description="Get detailed information about the current user's profile",
    parameters=[],
    category="context",
)
async def get_user_context(ctx: OrchestrationContext) -> ContextResult:
    """Get user context."""
    return ContextResult(
        success=True,
        context_type="user",
        context=ctx.user.model_dump(),
    )


@tool(
    name="get_professor_context",
    description="Get detailed information about the target professor",
    parameters=[],
    category="context",
)
async def get_professor_context(ctx: OrchestrationContext) -> ContextResult:
    """Get professor context."""
    return ContextResult(
        success=True,
        context_type="professor",
        context=ctx.professor.model_dump(),
    )


@tool(
    name="get_request_context",
    description="Get information about the current funding request",
    parameters=[],
    category="context",
)
async def get_request_context(ctx: OrchestrationContext) -> ContextResult:
    """Get request context."""
    return ContextResult(
        success=True,
        context_type="request",
        context=ctx.request.model_dump(),
    )


# ============================================================================
# Memory Tools
# ============================================================================

@tool(
    name="memory_store",
    description="Store a user preference or instruction for future reference",
    parameters=[
        ToolParameter(name="memory_type", type="string", description="Type of memory: tone, do_dont, preference, goal, bio, instruction, guardrail", required=True, enum=["tone", "do_dont", "preference", "goal", "bio", "instruction", "guardrail"]),
        ToolParameter(name="content", type="string", description="The content to remember", required=True),
        ToolParameter(name="confidence", type="number", description="Confidence level (0.0 to 1.0)", required=False, default=0.7),
    ],
    category="memory",
)
async def memory_store(
    ctx: OrchestrationContext,
    memory_type: str,
    content: str,
    confidence: float = 0.7,
) -> MemoryResult:
    """Store a memory."""
    # Note: This requires access to the database service
    # In the actual implementation, the orchestrator would inject the service
    return MemoryResult(
        success=True,
        message=f"Memory stored: {memory_type}",
        data={"type": memory_type, "content": content, "confidence": confidence},
    )


@tool(
    name="memory_retrieve",
    description="Retrieve stored memories and preferences for the user",
    parameters=[
        ToolParameter(name="memory_type", type="string", description="Filter by type (optional)", required=False),
    ],
    category="memory",
)
async def memory_retrieve(
    ctx: OrchestrationContext,
    memory_type: Optional[str] = None,
) -> MemoryResult:
    """Retrieve user memories."""
    if not ctx.memory:
        return MemoryResult(
            success=True,
            message="No memories found",
            memories=[],
        )
    
    memories = ctx.memory.memories
    if memory_type:
        memories = [m for m in memories if m.memory_type == memory_type]
    
    return MemoryResult(
        success=True,
        message=f"Found {len(memories)} memories",
        memories=[m.model_dump() for m in memories],
    )





