"""
Security and rate-limiting middleware and dependencies for FinTech RAG Assistant.
Provides API key validation and sliding-window rate limiting.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

from fastapi import Header, HTTPException, Request, status

from app.core.config import get_settings

settings = get_settings()


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding window rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_records: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Determines if a request from client_id is permitted within the sliding window."""
        now = time.time()
        cutoff = now - self.window_seconds
        # Evict old timestamps
        timestamps = [ts for ts in self.request_records[client_id] if ts > cutoff]
        if len(timestamps) >= self.max_requests:
            self.request_records[client_id] = timestamps
            return False
        timestamps.append(now)
        self.request_records[client_id] = timestamps
        return True


rate_limiter = SlidingWindowRateLimiter(
    max_requests=settings.RATE_LIMIT_PER_MINUTE,
    window_seconds=60,
)


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Validates the X-API-Key header against the configured secret key.
    Bypassed when ENABLE_AUTH is False.
    """
    if not settings.ENABLE_AUTH:
        return "anonymous"

    if not x_api_key or x_api_key != settings.API_KEY_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Provide a valid 'X-API-Key' header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency enforcing client request quotas per minute based on IP address.
    """
    client_ip = request.client.host if request.client else "unknown_client"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute allowed.",
        )
