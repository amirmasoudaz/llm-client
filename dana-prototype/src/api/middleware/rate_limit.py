# src/api/middleware/rate_limit.py
"""Rate limiting middleware for API requests."""

import time
from collections import defaultdict
from typing import Dict, Tuple
import asyncio

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import get_settings


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens. Returns True if successful."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Refill tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    @property
    def retry_after(self) -> float:
        """Seconds until tokens are available."""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Implements per-IP and per-user rate limiting using token bucket algorithm.
    """
    
    def __init__(self, app):
        super().__init__(app)
        settings = get_settings()
        
        # Rate limits (requests per second, burst capacity)
        self.ip_rate = 10.0
        self.ip_capacity = 50.0
        self.user_rate = 20.0
        self.user_capacity = 100.0
        
        # Buckets storage
        self.ip_buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.ip_rate, self.ip_capacity)
        )
        self.user_buckets: Dict[int, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.user_rate, self.user_capacity)
        )
        
        # Cleanup task
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.monotonic()
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _get_student_id(self, request: Request) -> int | None:
        """Extract student ID from request headers."""
        student_id = request.headers.get("X-Student-ID")
        if student_id:
            try:
                return int(student_id)
            except ValueError:
                pass
        return None
    
    async def _cleanup_old_buckets(self):
        """Remove old buckets to prevent memory leaks."""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        threshold = now - 3600  # Remove buckets not used for 1 hour
        
        # Clean IP buckets
        to_remove = [
            ip for ip, bucket in self.ip_buckets.items()
            if bucket.last_update < threshold
        ]
        for ip in to_remove:
            del self.ip_buckets[ip]
        
        # Clean user buckets
        to_remove = [
            user_id for user_id, bucket in self.user_buckets.items()
            if bucket.last_update < threshold
        ]
        for user_id in to_remove:
            del self.user_buckets[user_id]
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/live"]:
            return await call_next(request)
        
        # Cleanup old buckets periodically
        await self._cleanup_old_buckets()
        
        client_ip = self._get_client_ip(request)
        student_id = self._get_student_id(request)
        
        # Check IP rate limit
        ip_bucket = self.ip_buckets[client_ip]
        if not await ip_bucket.consume():
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests from this IP",
                    "retry_after": ip_bucket.retry_after,
                },
                headers={"Retry-After": str(int(ip_bucket.retry_after) + 1)}
            )
        
        # Check user rate limit if authenticated
        if student_id:
            user_bucket = self.user_buckets[student_id]
            if not await user_bucket.consume():
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Too many requests from this user",
                        "retry_after": user_bucket.retry_after,
                    },
                    headers={"Retry-After": str(int(user_bucket.retry_after) + 1)}
                )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(int(self.ip_capacity))
        response.headers["X-RateLimit-Remaining"] = str(int(ip_bucket.tokens))
        
        return response





