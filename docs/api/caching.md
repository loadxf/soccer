# API Caching System

This document describes the caching system implemented for the Soccer Prediction System API.

## Overview

The caching system uses Redis to store API responses and serve them directly without re-processing identical requests. This reduces server load, database queries, and response times for frequently accessed endpoints.

## How Caching Works

1. **Request Processing**: When a request arrives, the caching middleware checks if it should be cached.
2. **Cache Key Generation**: A unique cache key is created based on the request path and query parameters.
3. **Cache Lookup**: The middleware checks if a response for this key exists in Redis.
4. **Cache Hit**: If found, the cached response is returned immediately with an `X-Cache: HIT` header.
5. **Cache Miss**: If not found, the request is processed normally, and the response is cached before being returned with an `X-Cache: MISS` header.

## Configuration

The caching system is highly configurable through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CACHE_ENABLED` | Enable/disable caching | `true` |
| `SHORT_CACHE_TTL` | TTL for rapidly changing data (seconds) | `10` |
| `MEDIUM_CACHE_TTL` | TTL for moderately changing data (seconds) | `300` (5 minutes) |
| `LONG_CACHE_TTL` | TTL for relatively static data (seconds) | `3600` (1 hour) |
| `EXTRA_LONG_CACHE_TTL` | TTL for static data (seconds) | `86400` (24 hours) |
| `CACHE_PREFIX` | Prefix for Redis cache keys | `soccer_api_cache:` |
| `REDIS_HOST` | Redis server hostname | `127.0.0.1` |
| `REDIS_PORT` | Redis server port | `6379` |
| `REDIS_DB` | Redis database number | `0` |

## Default TTL Values by Endpoint

Different types of data have different appropriate caching times:

| Endpoint | TTL | Reason |
|----------|-----|--------|
| `/api/v1/health` | 10s | Health status can change quickly |
| `/api/v1/teams/` | 1h | Team data changes infrequently |
| `/api/v1/matches/` | 5m | Match data changes more often |
| `/api/v1/predictions/models` | 1h | Model list changes infrequently |
| `/api/v1/predictions/match/` | 5m | Match predictions can change |

## Excluded Endpoints

Some endpoints are never cached:

- All `POST`, `PUT`, `DELETE`, and `PATCH` requests
- Authentication endpoints (`/api/v1/auth/`)
- Admin endpoints (`/api/v1/admin/`)
- Custom prediction endpoints (`/api/v1/predictions/custom`)
- Batch prediction endpoints (`/api/v1/predictions/batch`)

## Implementation Details

The caching system consists of the following components:

- **CacheManager**: Handles the Redis connection and key operations
- **CacheMiddleware**: FastAPI middleware that intercepts requests and serves from cache
- **@cached Decorator**: Function decorator for caching specific endpoints

## Usage Examples

### Middleware-Based Caching

The middleware automatically caches responses based on request path and method:

```python
# Add caching middleware to FastAPI app
app.add_middleware(
    CacheMiddleware,
    exclude_paths=["/api/v1/auth/"],
    exclude_methods=["POST", "PUT", "DELETE", "PATCH"],
)
```

### Decorator-Based Caching

For more fine-grained control, use the `@cached` decorator:

```python
from src.utils.cache import cached

@app.get("/api/v1/teams")
@cached(ttl=3600)  # Cache for 1 hour
async def get_teams(request: Request):
    # Process request
    return {"teams": teams}
```

### Cache Control Headers

The system adds Cache-Control headers to all cached responses:

- `X-Cache: HIT|MISS` - Indicates whether the response was served from cache
- `Cache-Control: max-age=<ttl>` - Indicates how long the response can be cached

### Clearing Cache

There are several ways to clear the cache:

1. **Admin API endpoint** (requires admin role):
   ```
   POST /api/v1/admin/reset-cache
   {
     "scope": "all|teams|matches|predictions"
   }
   ```

2. **Programmatically**:
   ```python
   from src.utils.cache import clear_all_cache, clear_cache_for_prefix
   
   # Clear all cache
   await clear_all_cache()
   
   # Clear cache for specific prefix
   await clear_cache_for_prefix("teams")
   ```

## Performance Considerations

- **Memory Usage**: Monitor Redis memory usage in production environments
- **TTL Values**: Adjust TTL values based on data volatility and API traffic patterns
- **Key Explosion**: Be careful with highly variable query parameters to avoid cache key explosion

## Monitoring

To monitor cache performance:

1. Check the `X-Cache` header in responses to see hit/miss ratio
2. Set logging level to DEBUG to see detailed cache operations
3. Use Redis monitoring tools to observe key counts and memory usage 