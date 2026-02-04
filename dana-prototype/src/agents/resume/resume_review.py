# src/agents/resume/schema/review.py

from typing import List, Tuple, Literal
from pydantic import BaseModel, Field


class ScoreDimension(BaseModel):
    """Individual scoring dimension with evidence-based justification."""
    score: int = Field(..., ge=1, le=10, description="Score from 1 (poor) to 10 (excellent)")
    justification: str = Field(..., description="Detailed explanation of why this score was assigned")
    evidence: List[str] = Field(
        default_factory=list,
        description="Exact quotes or specific items from the CV supporting this score"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Specific, actionable improvements for this dimension"
    )


class CVReviewDimensions(BaseModel):
    """Multi-dimensional CV scoring across academic quality metrics."""

    content_completeness: ScoreDimension = Field(
        ...,
        description="Coverage of standard academic CV sections: education, research, publications, skills, etc."
    )

    research_presentation: ScoreDimension = Field(
        ...,
        description="Quality of research experience descriptions: methodology, outcomes, impact, specificity."
    )

    technical_depth: ScoreDimension = Field(
        ...,
        description="Adequate coverage of skills, tools, methodologies appropriate for the field."
    )

    publication_quality: ScoreDimension = Field(
        ...,
        description="Citation format, author list clarity, venue quality, and publication status."
    )

    structure_clarity: ScoreDimension = Field(
        ...,
        description="Logical organization, section ordering, visual hierarchy, and overall readability."
    )

    target_alignment: ScoreDimension = Field(
        ...,
        description="Alignment with target position/program if context provided; emphasis on relevant experience."
    )

    overall_strength: ScoreDimension = Field(
        ...,
        description="Holistic assessment of CV quality; considers all factors together."
    )


class CVReviewRespSchema(BaseModel):
    """Complete CV review with multi-dimensional scoring and feedback."""

    dimensions: CVReviewDimensions = Field(..., description="Multi-dimensional scoring and feedback")

    # Summary feedback
    strengths: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Top 3-5 strengths of the CV with specific examples"
    )

    weaknesses: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Top 3-5 weaknesses of the CV with specific examples"
    )

    priority_improvements: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Most important changes needed, in priority order"
    )

    # Optimization context (optional)
    optimization_suggestions: str = Field(
        default="",
        description="Suggested feedback for optimization if CV needs revision. Formatted as actionable instructions."
    )
