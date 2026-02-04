# src/api/routes/enhance.py
"""AI enhancement API routes."""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from llm_client import OpenAIClient, GPT5Mini

from src.api.dependencies import StudentIDDep


router = APIRouter()


class EnhanceRequest(BaseModel):
    """Request for AI enhancement."""
    field_type: str  # e.g., "research_interest", "skill", "experience_bullet"
    content: str
    context: Optional[str] = None


class EnhanceResponse(BaseModel):
    """Response for AI enhancement."""
    original: str
    enhanced: str
    alternatives: list[str]
    suggestions: list[str]


class ExpandRequest(BaseModel):
    """Request for content expansion."""
    content: str
    target_length: int = 200  # words
    style: str = "academic"


class ExpandResponse(BaseModel):
    """Response for content expansion."""
    original: str
    expanded: str
    word_count: int


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance_field(
    body: EnhanceRequest,
    student_id: StudentIDDep,
) -> EnhanceResponse:
    """
    AI-enhance a field value.
    
    Takes rough content and improves it for professional use.
    """
    llm = OpenAIClient(
        GPT5Mini,
        cache_backend="pg_redis",
        cache_collection="field_enhancement",
    )
    
    prompts = {
        "research_interest": f"""Enhance this research interest description for an academic application.

Original: {body.content}

Requirements:
- Be specific and technical
- Keep it concise (1-2 sentences)
- Use academic language

Provide the enhanced version and 2 alternatives.
Format as JSON with "enhanced" (string) and "alternatives" (array of strings).""",

        "skill": f"""Enhance this skill description for an academic CV.

Original: {body.content}

Requirements:
- Be specific about proficiency level
- Include relevant tools/technologies
- Keep it professional

Provide the enhanced version and 2 alternatives.
Format as JSON with "enhanced" (string) and "alternatives" (array of strings).""",

        "experience_bullet": f"""Enhance this experience bullet point for an academic CV.

Original: {body.content}

Requirements:
- Start with a strong action verb
- Include quantifiable results if possible
- Be specific about impact

Provide the enhanced version and 2 alternatives.
Format as JSON with "enhanced" (string) and "alternatives" (array of strings).""",

        "bio": f"""Enhance this bio/summary for an academic application.

Original: {body.content}

Requirements:
- Professional and engaging tone
- Highlight key strengths
- Keep under 150 words

Provide the enhanced version and 2 alternatives.
Format as JSON with "enhanced" (string) and "alternatives" (array of strings).""",
    }
    
    prompt = prompts.get(body.field_type)
    if not prompt:
        prompt = f"""Enhance this content for professional use.

Original: {body.content}
Context: {body.context or 'None'}

Provide the enhanced version and 2 alternatives.
Format as JSON with "enhanced" (string) and "alternatives" (array of strings)."""
    
    try:
        response = await llm.get_response(
            messages=[{"role": "user", "content": prompt}],
            response_format="json_object",
            temperature=0.7,
        )
        
        output = response.get("output", {})
        if isinstance(output, str):
            import json
            output = json.loads(output)
        
        return EnhanceResponse(
            original=body.content,
            enhanced=output.get("enhanced", body.content),
            alternatives=output.get("alternatives", []),
            suggestions=output.get("suggestions", []),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhancement failed: {str(e)}"
        )


@router.post("/expand", response_model=ExpandResponse)
async def expand_content(
    body: ExpandRequest,
    student_id: StudentIDDep,
) -> ExpandResponse:
    """
    Expand short content into a longer form.
    
    Useful for turning brief notes into full paragraphs.
    """
    llm = OpenAIClient(
        GPT5Mini,
        cache_backend="pg_redis",
        cache_collection="content_expansion",
    )
    
    style_instructions = {
        "academic": "Use formal academic language with proper terminology.",
        "professional": "Use professional business language.",
        "conversational": "Use a friendly, conversational tone.",
    }
    
    style_inst = style_instructions.get(body.style, style_instructions["academic"])
    
    prompt = f"""Expand this content to approximately {body.target_length} words.

Original: {body.content}

Style: {style_inst}

Requirements:
- Maintain the original meaning
- Add relevant details and context
- Use complete sentences
- Be coherent and well-structured

Return only the expanded content, no other text."""
    
    try:
        response = await llm.get_response(
            messages=[{"role": "user", "content": prompt}],
            response_format="text",
            temperature=0.5,
        )
        
        expanded = response.get("output", body.content)
        word_count = len(expanded.split())
        
        return ExpandResponse(
            original=body.content,
            expanded=expanded,
            word_count=word_count,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Expansion failed: {str(e)}"
        )





