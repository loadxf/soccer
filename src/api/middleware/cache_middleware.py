"""
Cache middleware for FastAPI.
Provides response caching using Redis and supports various caching strategies.
"""

import json
import time
from typing import Callable, Dict, List, Optional, Set, Union

from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.responses import Response, JSONResponse
from starlette.requests import Request
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.cache import CacheManager, cache_manager
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("cache.middleware")


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware for caching API responses in Redis.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        cache_manager: Optional[CacheManager] = None,
        ttl: Optional[int] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_methods: Optional[List[str]] = None,
        include_query_params: bool = True,
        cache_control_header: bool = True,
    ):
        """
        Initialize the cache middleware.
        
        Args:
            app: ASGI application
            cache_manager: CacheManager instance (uses default if None)
            ttl: Time-to-live for cached responses (overrides cache_manager.settings.ttl)
            exclude_paths: List of path prefixes to exclude from caching
            exclude_methods: List of HTTP methods to exclude from caching
            include_query_params: Whether to include query parameters in cache key
            cache_control_header: Whether to add Cache-Control headers to responses
        """
        super().__init__(app)
        self.cache_manager = cache_manager or cache_manager
        self.ttl = ttl
        self.exclude_paths = set(exclude_paths or [])
        self.exclude_methods = set(exclude_methods or ["POST", "PUT", "DELETE", "PATCH"])
        self.include_query_params = include_query_params
        self.cache_control_header = cache_control_header
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process an incoming request and return a response, with caching.
        
        Args:
            request: FastAPI request
            call_next: Call next middleware
            
        Returns:
            Response object
        """
        # Skip caching for excluded paths or methods
        if self._should_skip_cache(request):
            response = await call_next(request)
            return response
        
        # Create cache key
        cache_key = self.cache_manager.create_key(request)
        
        # Try to get from cache
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for: {request.url.path}")
            
            # Recreate the response from cached data
            try:
                data = json.loads(cached_data)
                return JSONResponse(
                    content=data,
                    headers=self._get_cache_headers(hit=True)
                )
            except Exception as e:
                logger.error(f"Error parsing cached response: {e}")
                # Continue with normal request handling if cache parse fails
        
        # If not in cache or error, call next middleware
        start_time = time.time()
        response = await call_next(request)
        end_time = time.time()
        
        # Only cache successful JSON responses
        if (
            self._is_cacheable_response(response) and 
            hasattr(response, "body") and 
            response.status_code < 400
        ):
            try:
                # Get response data (body is bytes)
                response_content = response.body.decode("utf-8")
                
                # Cache the response
                ttl = self.ttl or self.cache_manager.settings.ttl
                await self.cache_manager.set(cache_key, response_content, ttl=ttl)
                logger.debug(f"Cached response for: {request.url.path} (took {end_time - start_time:.3f}s)")
                
                # Add cache control headers if enabled
                if self.cache_control_header:
                    self._add_cache_headers(response, hit=False)
            except Exception as e:
                logger.error(f"Error caching response: {e}")
        
        return response
    
    def _should_skip_cache(self, request: Request) -> bool:
        """
        Determine if caching should be skipped for this request.
        
        Args:
            request: FastAPI request
            
        Returns:
            True if caching should be skipped, False otherwise
        """
        # Skip if caching is disabled
        if not self.cache_manager.settings.enabled:
            return True
        
        # Skip if method is excluded
        if request.method in self.exclude_methods:
            return True
        
        # Skip if path is excluded
        for path in self.exclude_paths:
            if request.url.path.startswith(path):
                return True
        
        # Check for cache control headers
        if "no-cache" in request.headers.get("Cache-Control", ""):
            return True
        
        return False
    
    def _is_cacheable_response(self, response: Response) -> bool:
        """
        Determine if a response is cacheable.
        
        Args:
            response: FastAPI response
            
        Returns:
            True if response is cacheable, False otherwise
        """
        # Check for JSON Content-Type
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("application/json"):
            return False
        
        # Don't cache streaming responses
        if response.headers.get("Transfer-Encoding", "") == "chunked":
            return False
        
        return True
    
    def _get_cache_headers(self, hit: bool) -> Dict[str, str]:
        """
        Get headers for cached responses.
        
        Args:
            hit: Whether this was a cache hit
            
        Returns:
            Headers dictionary
        """
        headers = {
            "X-Cache": "HIT" if hit else "MISS",
        }
        
        if self.cache_control_header:
            ttl = self.ttl or self.cache_manager.settings.ttl
            headers["Cache-Control"] = f"max-age={ttl}"
        
        return headers
    
    def _add_cache_headers(self, response: Response, hit: bool) -> None:
        """
        Add cache headers to a response.
        
        Args:
            response: FastAPI response
            hit: Whether this was a cache hit
        """
        response.headers["X-Cache"] = "HIT" if hit else "MISS"
        
        if self.cache_control_header:
            ttl = self.ttl or self.cache_manager.settings.ttl
            response.headers["Cache-Control"] = f"max-age={ttl}" 