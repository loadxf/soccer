"""
Rate limiting utilities for the Soccer Prediction System API.
Implements rate limiting using Redis to track and limit API requests.
"""

import time
from typing import Optional, Tuple, Callable, Dict, Any

from fastapi import Request, HTTPException, status, Depends
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from src.utils.logger import get_logger
from src.utils.db import get_redis_client
from src.utils.auth import get_current_user, User

# Setup logger
logger = get_logger("rate_limit")

# Rate limit settings - can be moved to config file
RATE_LIMIT_ENABLED = True
DEFAULT_RATE_LIMIT = 100  # requests per window
DEFAULT_WINDOW = 3600  # window in seconds (1 hour)
AUTHENTICATED_RATE_LIMIT = 500  # higher limit for authenticated users
ADMIN_RATE_LIMIT = 1000  # higher limit for admin users


class RateLimiter:
    """Redis-based rate limiter for API endpoints."""
    
    def __init__(self, prefix: str = "rate_limit"):
        """Initialize the rate limiter with a Redis prefix."""
        self.prefix = prefix
        self.redis = None
    
    def _get_redis(self):
        """Get Redis client if not already initialized."""
        if self.redis is None:
            self.redis = get_redis_client()
        return self.redis
    
    def _get_key(self, identifier: str, window: int) -> str:
        """Generate a Redis key for the rate limit."""
        # Include the window in the key to support different window sizes
        return f"{self.prefix}:{identifier}:{window}"
    
    def is_rate_limited(self, identifier: str, limit: int, window: int) -> Tuple[bool, int, int]:
        """
        Check if a request should be rate limited.
        
        Args:
            identifier: Unique identifier for the requester (IP address, API key, user ID)
            limit: Maximum number of requests allowed in the window
            window: Time window in seconds
            
        Returns:
            Tuple of (is_limited, current_count, reset_time)
        """
        if not RATE_LIMIT_ENABLED:
            return False, 0, 0
        
        redis = self._get_redis()
        key = self._get_key(identifier, window)
        
        # Get current count
        current_count = redis.get(key)
        if current_count is None:
            current_count = 0
        else:
            current_count = int(current_count)
        
        # Check if over limit
        is_limited = current_count >= limit
        
        # Get time to expiry
        ttl = redis.ttl(key)
        if ttl < 0:
            # Key doesn't exist or has no expiry, set to window
            ttl = window
            reset_time = int(time.time()) + window
        else:
            reset_time = int(time.time()) + ttl
        
        # Only increment if not limited
        if not is_limited:
            # Use pipeline for atomic operations
            pipe = redis.pipeline()
            pipe.incr(key)
            
            # Set expiry if key is new
            if current_count == 0:
                pipe.expire(key, window)
            
            pipe.execute()
            current_count += 1
        
        return is_limited, current_count, reset_time
    
    def reset(self, identifier: str, window: int) -> None:
        """Reset rate limit for an identifier."""
        redis = self._get_redis()
        key = self._get_key(identifier, window)
        redis.delete(key)


# Initialize global rate limiter
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for global rate limiting based on IP address."""
    
    def __init__(
        self, 
        app: ASGIApp, 
        limit: int = DEFAULT_RATE_LIMIT, 
        window: int = DEFAULT_WINDOW
    ):
        super().__init__(app)
        self.limit = limit
        self.window = window
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip certain paths from rate limiting
        path = request.url.path
        if path == "/api/v1/health" or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)
        
        # Check rate limit
        is_limited, current_count, reset_time = rate_limiter.is_rate_limited(
            client_ip, self.limit, self.window
        )
        
        # If limited, return 429 Too Many Requests
        if is_limited:
            logger.warning(f"Rate limit exceeded for IP {client_ip}")
            return Response(
                content="Rate limit exceeded. Please try again later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(reset_time - int(time.time())),
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = max(0, self.limit - current_count)
        response.headers["X-RateLimit-Limit"] = str(self.limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


def rate_limit(
    limit: int = DEFAULT_RATE_LIMIT, 
    window: int = DEFAULT_WINDOW,
    user_based: bool = True
):
    """
    Dependency for endpoint-specific rate limiting.
    
    Args:
        limit: Maximum number of requests in window
        window: Time window in seconds
        user_based: If True, use user ID for authenticated users, otherwise use IP
    """
    
    async def rate_limit_dependency(
        request: Request, 
        current_user: Optional[User] = None
    ):
        if not RATE_LIMIT_ENABLED:
            return
        
        # Determine identifier and applicable limit
        if user_based and current_user:
            # Use user ID for authenticated users
            identifier = f"user:{current_user.get('username', 'unknown')}"
            
            # Adjust limit based on user role
            user_limit = limit
            if current_user.get("role") == "admin":
                user_limit = ADMIN_RATE_LIMIT
            else:
                user_limit = AUTHENTICATED_RATE_LIMIT
        else:
            # Use IP address for anonymous users
            identifier = f"ip:{request.client.host if request.client else 'unknown'}"
            user_limit = limit
        
        # Check rate limit
        is_limited, current, reset_time = rate_limiter.is_rate_limited(
            identifier, user_limit, window
        )
        
        # If limited, raise exception
        if is_limited:
            logger.warning(f"Rate limit exceeded for {identifier}")
            headers = {
                "Retry-After": str(reset_time - int(time.time())),
                "X-RateLimit-Limit": str(user_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time)
            }
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers=headers
            )
    
    # For user-based limiting, also depend on get_current_user but make it optional
    if user_based:
        async def dependency(
            request: Request,
            current_user: Optional[User] = Depends(get_current_user_optional)
        ):
            await rate_limit_dependency(request, current_user)
        return dependency
    else:
        async def dependency(request: Request):
            await rate_limit_dependency(request)
        return dependency


async def get_current_user_optional(request: Request) -> Optional[User]:
    """
    Try to get the current user, but don't require authentication.
    
    This is used for rate limiting to apply different limits for
    authenticated vs. anonymous users.
    """
    try:
        # Extract token from authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.replace("Bearer ", "")
        
        # Use the regular authentication function to get user
        from src.utils.auth import get_current_user
        return await get_current_user(token)
    except Exception:
        # If any error occurs, just return None
        return None 