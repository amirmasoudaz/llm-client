from pydantic import BaseModel
from typing import List, Literal, Optional


class TopicSchema(BaseModel):
    tag: str
    expertise: Literal["basic", "intermediate", "advanced"]


class RespSchema(BaseModel):
    primary_topics: List[TopicSchema]
    secondary_topics: List[TopicSchema]
    confidence: float
    sources_used: List[str]
    notes: Optional[str] = None
