# src/agents/schemas/email_review.py

from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


class ScoreDimension(BaseModel):
    """Individual scoring dimension with evidence-based justification."""
    score: int = Field(..., ge=1, le=10, description="Score from 1 (poor) to 10 (excellent)")
    justification: str = Field(..., description="Detailed explanation of why this score was assigned")
    evidence: List[str] = Field(
        default_factory=list,
        description="Exact quotes from the email supporting this score"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Specific, actionable improvements for this dimension"
    )


class EmailReviewDimensions(BaseModel):
    """Multi-dimensional scoring for professor outreach emails."""

    subject_quality: ScoreDimension = Field(
        ...,
        description="Quality of subject line. High scores are specific, compelling, and professional (not generic like 'Research Opportunity')."
    )

    research_fit: ScoreDimension = Field(
        ...,
        description="Alignment with professor's research. High scores explicitly connect sender's work to professor's specific research areas, papers, or projects."
    )

    evidence_quality: ScoreDimension = Field(
        ...,
        description="Quality of supporting evidence. High scores cite specific projects, publications, metrics, or concrete achievements."
    )

    tone_appropriateness: ScoreDimension = Field(
        ...,
        description="Professional yet personable email tone. High scores balance formality with genuine enthusiasm; avoid overly casual or stiff language."
    )

    length_efficiency: ScoreDimension = Field(
        ...,
        description="Appropriate length and conciseness. High scores are 150-300 words with no filler; every sentence adds value."
    )

    call_to_action: ScoreDimension = Field(
        ...,
        description="Clarity of next steps or meeting request. High scores have clear, specific asks (e.g., '15-minute call next week to discuss collaboration')."
    )

    overall_strength: ScoreDimension = Field(
        ...,
        description="Holistic assessment of email quality. Considers all factors together and likelihood of positive response."
    )


class EmailReviewRespSchema(BaseModel):
    """Complete email review with multi-dimensional scoring and feedback."""

    dimensions: EmailReviewDimensions = Field(..., description="Multi-dimensional scoring and feedback")

    # SUMMARY FEEDBACK

    strengths: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Top 3-5 strengths of the email with specific examples"
    )

    weaknesses: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Top 3-5 weaknesses of the email with specific examples"
    )

    priority_improvements: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Most important changes needed, in priority order"
    )

    # OPTIMIZATION CONTEXT (Optional)

    optimization_suggestions: Optional[str] = Field(
        default=None,
        description=(
            "Suggested feedback for optimization_context if email needs revision. "
            "Formatted as actionable instructions."
        )
    )
