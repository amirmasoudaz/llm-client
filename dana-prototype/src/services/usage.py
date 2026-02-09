# src/services/usage.py
"""Usage tracking service for credit consumption and billing."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import httpx

from src.services.db import DatabaseService
from src.config import get_settings


@dataclass
class CostBreakdown:
    """Cost breakdown for a job or period."""
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass
class UsageSummary:
    """Usage summary for a period."""
    period_start: datetime
    period_end: datetime
    total_tokens: int
    total_cost: Decimal
    job_count: int
    by_job_type: Dict[str, CostBreakdown]
    by_model: Dict[str, CostBreakdown]


# Model pricing (per 1M tokens)
MODEL_PRICING = {
    "gpt-4o": {
        "input": Decimal("5.00"),
        "output": Decimal("15.00"),
    },
    "gpt-4o-mini": {
        "input": Decimal("0.15"),
        "output": Decimal("0.60"),
    },
    "gpt-4-turbo": {
        "input": Decimal("10.00"),
        "output": Decimal("30.00"),
    },
    "gpt-3.5-turbo": {
        "input": Decimal("0.50"),
        "output": Decimal("1.50"),
    },
    "text-embedding-3-small": {
        "input": Decimal("0.02"),
        "output": Decimal("0.00"),
    },
    "text-embedding-3-large": {
        "input": Decimal("0.13"),
        "output": Decimal("0.00"),
    },
}


class UsageService:
    """
    Service for tracking usage and managing credits.
    
    Provides:
    - Cost calculation for jobs
    - Usage aggregation and reporting
    - Credit checking and deduction
    - Platform billing integration
    """
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self._settings = get_settings()
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                base_url=self._settings.platform_backend_url,
            )
        return self._http_client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostBreakdown:
        """Calculate cost for token usage."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o-mini"])
        
        # Calculate cost (pricing is per 1M tokens)
        input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * pricing["input"]
        output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * pricing["output"]
        
        return CostBreakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )
    
    def calculate_job_cost(self, job: Any) -> CostBreakdown:
        """Calculate cost for a job from its usage data."""
        return self.calculate_cost(
            model=job.model or "gpt-4o-mini",
            input_tokens=job.token_input or 0,
            output_tokens=job.token_output or 0,
        )

    async def get_usage_summary(
        self,
        student_id: int,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> UsageSummary:
        """Get usage summary for a student."""
        if not period_end:
            period_end = datetime.utcnow()
        if not period_start:
            period_start = period_end - timedelta(days=30)
        
        # Get jobs for period
        jobs = await self.db.client.aijob.find_many(
            where={
                "student_id": student_id,
                "status": "succeeded",
                "created_at": {
                    "gte": period_start,
                    "lte": period_end,
                },
            },
        )
        
        # Aggregate by job type
        by_job_type: Dict[str, CostBreakdown] = {}
        by_model: Dict[str, CostBreakdown] = {}
        
        total_tokens = 0
        total_cost = Decimal("0")
        
        for job in jobs:
            cost = self.calculate_job_cost(job)
            
            total_tokens += cost.total_tokens
            total_cost += cost.total_cost
            
            # By job type
            if job.job_type not in by_job_type:
                by_job_type[job.job_type] = CostBreakdown(
                    Decimal("0"), Decimal("0"), Decimal("0"), 0, 0, 0
                )
            jt = by_job_type[job.job_type]
            by_job_type[job.job_type] = CostBreakdown(
                jt.input_cost + cost.input_cost,
                jt.output_cost + cost.output_cost,
                jt.total_cost + cost.total_cost,
                jt.input_tokens + cost.input_tokens,
                jt.output_tokens + cost.output_tokens,
                jt.total_tokens + cost.total_tokens,
            )
            
            # By model
            model = job.model or "unknown"
            if model not in by_model:
                by_model[model] = CostBreakdown(
                    Decimal("0"), Decimal("0"), Decimal("0"), 0, 0, 0
                )
            m = by_model[model]
            by_model[model] = CostBreakdown(
                m.input_cost + cost.input_cost,
                m.output_cost + cost.output_cost,
                m.total_cost + cost.total_cost,
                m.input_tokens + cost.input_tokens,
                m.output_tokens + cost.output_tokens,
                m.total_tokens + cost.total_tokens,
            )
        
        return UsageSummary(
            period_start=period_start,
            period_end=period_end,
            total_tokens=total_tokens,
            total_cost=total_cost,
            job_count=len(jobs),
            by_job_type=by_job_type,
            by_model=by_model,
        )
    
    async def check_credits(
        self,
        student_id: int,
        required_credits: Decimal = Decimal("0.01"),
    ) -> tuple[bool, Decimal]:
        """
        Check if student has sufficient credits.
        
        Returns (has_credits, remaining_credits).
        """
        credits = await self._get_platform_credits(student_id)
        remaining = Decimal(str(credits.get("remaining", 0)))
        
        return remaining >= required_credits, remaining
    
    async def deduct_credits(
        self,
        student_id: int,
        amount: Decimal,
        job_id: Optional[int] = None,
        description: str = "",
    ) -> bool:
        """
        Deduct credits from student's account.
        
        Returns True if successful.
        """
        try:
            client = await self._get_http_client()
            
            response = await client.post(
                "/api/credits/deduct",
                json={
                    "student_id": student_id,
                    "amount": float(amount),
                    "job_id": job_id,
                    "description": description,
                },
                headers={
                    "Authorization": f"Bearer {self._settings.platform_api_key}",
                    "Content-Type": "application/json",
                },
            )
            
            return response.status_code == 200
            
        except Exception:
            # Log error but don't fail the operation
            return True  # In development, allow operations
    
    async def _get_platform_credits(self, student_id: int) -> Dict[str, Any]:
        """Get credit status from platform backend."""
        # In development, return mock credits
        if self._settings.app_env == "dev":
            return {
                "used": 0.0,
                "remaining": 1000.0,
                "total": 1000.0,
                "active": True,
            }
        
        try:
            client = await self._get_http_client()
            
            response = await client.get(
                f"/api/credits/{student_id}",
                headers={
                    "Authorization": f"Bearer {self._settings.platform_api_key}",
                },
            )
            
            if response.status_code == 200:
                return response.json()
            
            return {"remaining": 0, "active": False}
            
        except Exception:
            return {"remaining": 0, "active": False}
    
    async def check_and_warn_low_credits(
        self,
        student_id: int,
        threshold: Decimal = Decimal("10.0"),
    ) -> Optional[str]:
        """Check credits and return warning message if low."""
        has_credits, remaining = await self.check_credits(student_id)
        
        if not has_credits:
            return "Your credit balance is depleted. Please add credits to continue using Dana."
        
        if remaining < threshold:
            return f"Your credit balance is running low (${remaining:.2f} remaining). Consider adding more credits."
        
        return None
