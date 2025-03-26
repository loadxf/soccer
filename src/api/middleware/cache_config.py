"""
Configuration module for the caching system.
Defines sensible cache TTL values for different endpoint types.
"""

import os
from typing import Dict, List, Set

# API route paths (prefixes)
API_PREFIX = "/api/v1"
HEALTH_PATH = f"{API_PREFIX}/health"
TEAMS_PATH = f"{API_PREFIX}/teams"
MATCHES_PATH = f"{API_PREFIX}/matches"
PREDICTIONS_PATH = f"{API_PREFIX}/predictions"
ADMIN_PATH = f"{API_PREFIX}/admin"
AUTH_PATH = f"{API_PREFIX}/auth"

# Define cache settings

# Cache time to live (TTL) in seconds
# Short cache (10 seconds) - for rapidly changing data
SHORT_CACHE_TTL = int(os.getenv("SHORT_CACHE_TTL", 10))

# Medium cache (5 minutes) - for data that changes moderately
MEDIUM_CACHE_TTL = int(os.getenv("MEDIUM_CACHE_TTL", 300))  # 5 minutes

# Long cache (1 hour) - for relatively static data
LONG_CACHE_TTL = int(os.getenv("LONG_CACHE_TTL", 3600))  # 1 hour

# Extra long cache (1 day) - for static data
EXTRA_LONG_CACHE_TTL = int(os.getenv("EXTRA_LONG_CACHE_TTL", 86400))  # 24 hours

# Default cache TTL
DEFAULT_CACHE_TTL = MEDIUM_CACHE_TTL

# Cache TTL for specific paths (prefix matching)
PATH_TTL_MAP = {
    HEALTH_PATH: SHORT_CACHE_TTL,
    f"{TEAMS_PATH}/": LONG_CACHE_TTL,  # Team listings change infrequently
    f"{MATCHES_PATH}/": MEDIUM_CACHE_TTL,  # Match data changes more often
    f"{PREDICTIONS_PATH}/models": LONG_CACHE_TTL,  # Model list changes infrequently
    f"{PREDICTIONS_PATH}/match/": MEDIUM_CACHE_TTL,  # Predictions can change
    f"{PREDICTIONS_PATH}/history": MEDIUM_CACHE_TTL,  # History doesn't change much
}

# Paths to exclude from caching
CACHE_EXCLUDE_PATHS = [
    f"{AUTH_PATH}/",  # Never cache auth endpoints
    f"{ADMIN_PATH}/",  # Never cache admin endpoints
    f"{PREDICTIONS_PATH}/custom",  # Don't cache custom predictions
    f"{PREDICTIONS_PATH}/batch",  # Don't cache batch predictions
]

# Methods to exclude from caching (defaults to ["POST", "PUT", "DELETE", "PATCH"])
CACHE_EXCLUDE_METHODS = ["POST", "PUT", "DELETE", "PATCH"]

# Query parameters to ignore when creating cache keys
CACHE_IGNORE_QUERY_PARAMS = [
    "api_key",  # Don't separate cache by API key
    "token",    # Don't separate cache by token
]

# Cache enabled flag (can be disabled in development or testing)
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ["true", "1", "yes"]

# Cache prefix for Redis keys
CACHE_PREFIX = os.getenv("CACHE_PREFIX", "soccer_api_cache:")

# Function to determine TTL for a given path
def get_ttl_for_path(path: str) -> int:
    """
    Get the appropriate TTL for a given path.
    
    Args:
        path: The request path
        
    Returns:
        TTL in seconds
    """
    # First, try exact match
    if path in PATH_TTL_MAP:
        return PATH_TTL_MAP[path]
    
    # Then, try prefix match
    for prefix, ttl in PATH_TTL_MAP.items():
        if path.startswith(prefix):
            return ttl
    
    # Default
    return DEFAULT_CACHE_TTL 