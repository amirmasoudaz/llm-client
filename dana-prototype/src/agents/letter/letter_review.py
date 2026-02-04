# src/agents/letter/schema/review.py

from typing import List, Optional
from pydantic import BaseModel, Field


class ScoreDimension(BaseModel):
    """Individual scoring dimension with evidence-based justification."""
    score: int = Field(minimum=1, maximum=10, description="Score from 1 (poor) to 10 (excellent)")
    justification: str = Field(..., description="Detailed explanation of why this score was assigned")
    evidence: List[str] = Field(
        default_factory=list,
        description="Exact quotes from the letter supporting this score"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Specific, actionable improvements for this dimension"
    )


class LetterReviewDimensions(BaseModel):
    # SCORING DIMENSIONS (1-10 scale)

    specificity: ScoreDimension = Field(
        ...,
        description="How specific vs generic the letter is. High scores have concrete details, project names, metrics."
    )

    research_fit: ScoreDimension = Field(
        ...,
        description="Alignment with target program/lab. High scores explicitly connect applicant's work to target research."
    )

    evidence_quality: ScoreDimension = Field(
        ...,
        description="Quality of supporting evidence. High scores cite metrics, publications, concrete outcomes."
    )

    structure_flow: ScoreDimension = Field(
        ...,
        description="Logical organization and transitions. High scores have clear structure and smooth flow."
    )

    academic_tone: ScoreDimension = Field(
        ...,
        description="Appropriateness of scholarly voice. High scores are professional, authentic, confident without arrogance."
    )

    technical_depth: ScoreDimension = Field(
        ...,
        description="Adequate technical detail for the field. High scores mention methodologies, tools, frameworks appropriately."
    )

    overall_strength: ScoreDimension = Field(
        ...,
        description="Holistic assessment of letter quality. Considers all factors together."
    )


class LetterReviewRespSchema(BaseModel):
    """Complete letter review with multi-dimensional scoring and feedback."""

    dimensions: LetterReviewDimensions = Field(..., description="Multi-dimensional scoring and feedback")

    # SUMMARY FEEDBACK

    strengths: List[str] = Field(
        ...,
        min_items=3,
        max_items=5,
        description="Top 3-5 strengths of the letter with specific examples"
    )

    weaknesses: List[str] = Field(
        ...,
        min_items=3,
        max_items=5,
        description="Top 3-5 weaknesses of the letter with specific examples"
    )

    priority_improvements: List[str] = Field(
        ...,
        min_items=3,
        max_items=5,
        description="Most important changes needed, in priority order"
    )

    # OPTIMIZATION CONTEXT (Optional)

    optimization_suggestions: Optional[str] = Field(
        default=None,
        description=(
            "Suggested feedback for optimization_context if letter needs revision. "
            "Formatted as actionable instructions."
        )
    )
