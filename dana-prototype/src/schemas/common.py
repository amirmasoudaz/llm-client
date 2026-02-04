# src/schemas/common.py
"""Common Pydantic schemas."""

from typing import Any, Generic, List, Optional, TypeVar
from datetime import datetime

from pydantic import BaseModel, Field


T = TypeVar("T")


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[dict] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""
    items: List[T]
    total: int
    limit: int
    offset: int
    has_more: bool


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class UsageInfo(BaseModel):
    """Token and cost usage information."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0


class JobInfo(BaseModel):
    """Job information."""
    job_id: int
    job_type: str
    status: str
    progress: int = 0
    model: Optional[str] = None
    usage: Optional[UsageInfo] = None





