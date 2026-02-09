# src/api/schemas/logic.py

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    tags: List[str] = Field(..., min_items=0, description="List of tags selected by user")
    page_size: Optional[int] = Field(10, ge=1, le=100, description="Number of results per page")
    page_number: Optional[int] = Field(1, ge=1, description="Page number (1-based)")
    institute_name: Optional[str] = Field(None, description="If provided, filter results to this institute")
    country: Optional[str] = Field(None, description="If provided, filter results to this country")
    professor_name: Optional[str] = Field(None, description="Optional free-text search over professor names inside the filtered set")
    department_ids: Optional[List[int]] = Field(None, description="If provided, filter results to these department IDs")
    expand_related: bool = Field(True, description="If true, expand each input tag by related neighbors")
    domains: Optional[List[str]] = Field(None, description="If provided, restrict matches to these domains")
    subfields: Optional[List[str]] = Field(None, description="If provided, restrict matches to these subfields")

class QueryItem(BaseModel):
    tag: str = Field(..., description="The resolved tag.")
    subfield: str = Field(..., description="The subfield this tag belongs to.")
    domain: str = Field(..., description="The top-level domain this tag belongs to.")
    doc_text: str = Field(
        ...,
        description="Concatenated string showing tag, subfield, and domain."
    )

class Suggestion(BaseModel):
    tag: str
    domain: str
    subfield: str
    score: float
    prof_count: int

class QueryResponse(BaseModel):
    status: bool = Field(..., examples=[True])
    q: str
    suggestions: List[Suggestion]

class RecommendResponse(BaseModel):
    status: bool
    professor_ids: List[int]
    explanations: List[Dict[str, Any]]
