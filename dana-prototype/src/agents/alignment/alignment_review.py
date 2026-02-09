# src/agents/schemas/alignment_review.py

from typing import List, Literal, Tuple, Dict
from pydantic import BaseModel, Field


class QAResult(BaseModel):
    category_id: str
    question_id: str
    answer: Literal["Yes","No"]
    intensity: float
    justification: str


class LLMQuestionOutputs(BaseModel):
    results: List[QAResult]


class Diagnostics(BaseModel):
    coverage_high_conf_count: float
    coverage_high_conf_rate: float
    mean_intensity: float
    required_yes_rate: float
    required_yes_count: int
    required_total: int


class QuestionScore(BaseModel):
    category_id: str
    question_id: str
    weight: float
    answer: str
    intensity: float
    signed: float = Field(..., description="+intensity if Yes, -intensity if No")
    justification: str


class CategoryScore(BaseModel):
    category_id: str
    name: str
    score_0_100: float
    details: List[QuestionScore]


class AlignmentScore(BaseModel):
    score_0_100: float
    label: str
    reasons: List[str] = Field(default_factory=list)


class AlignmentReport(BaseModel):
    request_id: str
    categories: List[CategoryScore]
    overall: AlignmentScore
    question_results: List[QuestionScore]
    diagnostics: Diagnostics


class Judgment(BaseModel):
    answer: str
    intensity: float
    justification: str
