"""
Caching utility for the Soccer Prediction System.
Implements a Redis-based caching system for API responses.
"""

import json
import hashlib
import inspect
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, TypeVar, cast

from fastapi import Request, Response
from pydantic import BaseModel

from src.utils.db import get_redis_client
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("cache")

# Type variables for generic functions
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

# Default cache settings
DEFAULT_CACHE_TTL = 60 * 5  # 5 minutes
DEFAULT_CACHE_PREFIX = "soccer_api_cache:"


class CacheSettings:
    """Settings for the caching system."""
    
    def __init__(
        self, 
        ttl: int = DEFAULT_CACHE_TTL,
        prefix: str = DEFAULT_CACHE_PREFIX,
        ignore_query_params: Optional[List[str]] = None,
        enabled: bool = True
    ):
        """
        Initialize cache settings.
        
        Args:
            ttl: Time-to-live in seconds for cached items
            prefix: Redis key prefix for cached items
            ignore_query_params: Query parameters to ignore when creating cache keys
            enabled: Whether caching is enabled
        """
        self.ttl = ttl
        self.prefix = prefix
        self.ignore_query_params = set(ignore_query_params or [])
        self.enabled = enabled


class CacheManager:
    """Manager for the Redis-based cache system."""
    
    def __init__(self, settings: Optional[CacheSettings] = None):
        """
        Initialize the cache manager with settings.
        
        Args:
            settings: Cache settings
        """
        self.settings = settings or CacheSettings()
        self._redis = None
    
    @property
    def redis(self):
        """Get Redis client if not already initialized."""
        if self._redis is None:
            self._redis = get_redis_client()
        return self._redis
    
    def create_key(self, request: Request) -> str:
        """
        Create a cache key from the request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Cache key string
        """
        # Start with the path
        key_parts = [request.url.path]
        
        # Add query parameters (sorted for consistency)
        if request.query_params:
            query_items = list(request.query_params.items())
            # Filter out ignored query parameters
            if self.settings.ignore_query_params:
                query_items = [
                    (k, v) for k, v in query_items 
                    if k not in self.settings.ignore_query_params
                ]
            # Sort for consistent key generation
            query_items.sort()
            key_parts.append("?")
            key_parts.append("&".join(f"{k}={v}" for k, v in query_items))
        
        # Create a hash of the full URL with query parameters
        key = "".join(key_parts)
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        
        # Prefixed with the cache prefix
        return f"{self.settings.prefix}{hashed_key}"
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.settings.ttl
            
        try:
            self.redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
            
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern
            
        Returns:
            Number of keys deleted
        """
        try:
            # Get all keys matching the pattern
            full_pattern = f"{self.settings.prefix}{pattern}"
            keys = self.redis.keys(full_pattern)
            
            if not keys:
                return 0
                
            # Delete all matched keys
            self.redis.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.error(f"Error deleting keys from cache: {e}")
            return 0
            
    async def clear_all(self) -> int:
        """
        Clear all cache entries with this prefix.
        
        Returns:
            Number of keys deleted
        """
        return await self.delete_pattern("*")


# Create default cache manager
cache_manager = CacheManager()


def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    ignore_query_params: Optional[List[str]] = None
) -> Callable[[F], F]:
    """
    Decorator to cache API endpoint responses.
    
    Args:
        ttl: Override default cache TTL
        key_prefix: Add prefix to the cache key
        ignore_query_params: Query parameters to ignore when creating cache keys
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if caching is enabled
            if not cache_manager.settings.enabled:
                return await func(*args, **kwargs)
                
            # Get request object from args (FastAPI passes it as first arg to route handlers)
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
                    
            # If no request found in positional args, look in kwargs
            if request is None:
                request = kwargs.get('request')
                
            # If still no request, can't cache
            if request is None:
                logger.warning(f"No request object found for {func.__name__}, can't cache")
                return await func(*args, **kwargs)
            
            # Create cache key
            cache_key = cache_manager.create_key(request)
            if key_prefix:
                cache_key = f"{key_prefix}:{cache_key}"
                
            # Try to get from cache
            cached_response = await cache_manager.get(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for {request.url.path}")
                # Parse the cached response
                cached_data = json.loads(cached_response)
                return cached_data
                
            # If not in cache, call the function
            response = await func(*args, **kwargs)
            
            # Cache the response
            await cache_manager.set(
                cache_key, 
                json.dumps(response), 
                ttl=ttl or cache_manager.settings.ttl
            )
            logger.debug(f"Cached response for {request.url.path}")
            
            return response
            
        return cast(F, wrapper)
    return decorator


async def clear_cache_for_prefix(prefix: str) -> int:
    """
    Clear all cache entries with a specific prefix.
    
    Args:
        prefix: Cache key prefix
        
    Returns:
        Number of keys deleted
    """
    return await cache_manager.delete_pattern(f"{prefix}*")


async def clear_all_cache() -> int:
    """
    Clear all cache entries.
    
    Returns:
        Number of keys deleted
    """
    return await cache_manager.clear_all() 