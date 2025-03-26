"""
Middleware components for the Soccer Prediction System API.
"""

from src.api.middleware.cache_middleware import CacheMiddleware
from src.api.middleware.cache_config import (
    CACHE_ENABLED,
    CACHE_PREFIX,
    CACHE_EXCLUDE_PATHS,
    CACHE_EXCLUDE_METHODS,
    CACHE_IGNORE_QUERY_PARAMS,
    DEFAULT_CACHE_TTL,
    get_ttl_for_path
)

__all__ = [
    "CacheMiddleware",
    "CACHE_ENABLED",
    "CACHE_PREFIX",
    "CACHE_EXCLUDE_PATHS",
    "CACHE_EXCLUDE_METHODS",
    "CACHE_IGNORE_QUERY_PARAMS",
    "DEFAULT_CACHE_TTL",
    "get_ttl_for_path",
] 