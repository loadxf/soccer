"""
Unit tests for the rate limiting utility.
"""

import os
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import Request, Response, HTTPException
from starlette.types import Scope

# Import project components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.rate_limit import (
    RateLimiter, RateLimitMiddleware, rate_limit,
    DEFAULT_RATE_LIMIT, DEFAULT_WINDOW, AUTHENTICATED_RATE_LIMIT, ADMIN_RATE_LIMIT
)


class TestRateLimiter(unittest.TestCase):
    """Test cases for the RateLimiter class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create patcher for Redis client
        self.redis_patcher = patch('src.utils.rate_limit.get_redis_client')
        self.mock_redis_getter = self.redis_patcher.start()
        self.mock_redis = MagicMock()
        self.mock_redis_getter.return_value = self.mock_redis
        
        # Create rate limiter instance
        self.rate_limiter = RateLimiter(prefix="test_rate_limit")
        
        # Set system time reference
        self.current_time = time.time()
        self.time_patcher = patch('time.time')
        self.mock_time = self.time_patcher.start()
        self.mock_time.return_value = self.current_time
        
    def tearDown(self):
        """Clean up test environment."""
        self.redis_patcher.stop()
        self.time_patcher.stop()
    
    def test_init(self):
        """Test initialization of RateLimiter."""
        self.assertEqual(self.rate_limiter.prefix, "test_rate_limit")
        self.assertIsNone(self.rate_limiter.redis)
    
    def test_get_redis(self):
        """Test getting Redis client."""
        redis = self.rate_limiter._get_redis()
        self.assertEqual(redis, self.mock_redis)
        self.mock_redis_getter.assert_called_once()
        
        # Second call should reuse the client
        redis2 = self.rate_limiter._get_redis()
        self.assertEqual(redis2, self.mock_redis)
        self.mock_redis_getter.assert_called_once()
    
    def test_get_key(self):
        """Test key generation for identifiers."""
        identifier = "test_user"
        window = 60
        expected_key = f"test_rate_limit:{identifier}:{window}"
        key = self.rate_limiter._get_key(identifier, window)
        self.assertEqual(key, expected_key)
    
    def test_is_rate_limited_first_request(self):
        """Test rate limiting for first request."""
        # Setup Redis mock
        self.mock_redis.get.return_value = None
        self.mock_redis.ttl.return_value = -2  # Key doesn't exist
        self.mock_redis.pipeline.return_value = self.mock_redis
        
        # Test rate limiting
        identifier = "test_user"
        limit = 5
        window = 60
        
        is_limited, current_count, reset_time = self.rate_limiter.is_rate_limited(
            identifier, limit, window
        )
        
        # Verify results
        self.assertFalse(is_limited)
        self.assertEqual(current_count, 1)
        self.assertEqual(reset_time, int(self.current_time) + window)
        
        # Verify Redis calls
        self.mock_redis.get.assert_called_once_with(f"test_rate_limit:{identifier}:{window}")
        self.mock_redis.incr.assert_called_once()
        self.mock_redis.expire.assert_called_once_with(f"test_rate_limit:{identifier}:{window}", window)
        self.mock_redis.execute.assert_called_once()
    
    def test_is_rate_limited_under_limit(self):
        """Test rate limiting for subsequent requests under limit."""
        # Setup Redis mock
        self.mock_redis.get.return_value = "3"  # Already 3 requests
        self.mock_redis.ttl.return_value = 30  # 30 seconds left in window
        self.mock_redis.pipeline.return_value = self.mock_redis
        
        # Test rate limiting
        identifier = "test_user"
        limit = 5
        window = 60
        
        is_limited, current_count, reset_time = self.rate_limiter.is_rate_limited(
            identifier, limit, window
        )
        
        # Verify results
        self.assertFalse(is_limited)
        self.assertEqual(current_count, 4)
        self.assertEqual(reset_time, int(self.current_time) + 30)
        
        # Verify Redis calls
        self.mock_redis.get.assert_called_once_with(f"test_rate_limit:{identifier}:{window}")
        self.mock_redis.incr.assert_called_once()
        self.mock_redis.expire.assert_not_called()  # Shouldn't set expire for existing key
        self.mock_redis.execute.assert_called_once()
    
    def test_is_rate_limited_over_limit(self):
        """Test rate limiting when limit is exceeded."""
        # Setup Redis mock
        self.mock_redis.get.return_value = "5"  # Already 5 requests
        self.mock_redis.ttl.return_value = 30  # 30 seconds left in window
        
        # Test rate limiting
        identifier = "test_user"
        limit = 5
        window = 60
        
        is_limited, current_count, reset_time = self.rate_limiter.is_rate_limited(
            identifier, limit, window
        )
        
        # Verify results
        self.assertTrue(is_limited)
        self.assertEqual(current_count, 5)
        self.assertEqual(reset_time, int(self.current_time) + 30)
        
        # Verify Redis calls (shouldn't increment when limited)
        self.mock_redis.get.assert_called_once_with(f"test_rate_limit:{identifier}:{window}")
        self.mock_redis.incr.assert_not_called()
        self.mock_redis.pipeline.assert_not_called()
        self.mock_redis.execute.assert_not_called()
    
    def test_reset(self):
        """Test resetting rate limit."""
        # Test reset
        identifier = "test_user"
        window = 60
        
        self.rate_limiter.reset(identifier, window)
        
        # Verify Redis calls
        self.mock_redis.delete.assert_called_once_with(f"test_rate_limit:{identifier}:{window}")


class TestRateLimitMiddleware(unittest.TestCase):
    """Test cases for the RateLimitMiddleware."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock app
        self.mock_app = MagicMock()
        
        # Create patcher for rate_limiter
        self.limiter_patcher = patch('src.utils.rate_limit.rate_limiter')
        self.mock_rate_limiter = self.limiter_patcher.start()
        
        # Create middleware
        self.middleware = RateLimitMiddleware(self.mock_app, limit=10, window=60)
        
    def tearDown(self):
        """Clean up test environment."""
        self.limiter_patcher.stop()
    
    def create_mock_request(self, path="/api/v1/test", client_host="127.0.0.1"):
        """Helper to create a mock request."""
        mock_client = MagicMock()
        mock_client.host = client_host
        
        mock_url = MagicMock()
        mock_url.path = path
        
        mock_request = MagicMock(spec=Request)
        mock_request.client = mock_client
        mock_request.url = mock_url
        
        return mock_request
    
    async def test_dispatch_excluded_path(self):
        """Test that certain paths are excluded from rate limiting."""
        # Create request for excluded path
        mock_request = self.create_mock_request(path="/api/v1/health")
        
        # Mock call_next function
        mock_response = MagicMock(spec=Response)
        mock_call_next = AsyncMock(return_value=mock_response)
        
        # Call dispatch
        response = await self.middleware.dispatch(mock_request, mock_call_next)
        
        # Verify rate limiter wasn't called
        self.mock_rate_limiter.is_rate_limited.assert_not_called()
        
        # Verify call_next was called with request
        mock_call_next.assert_called_once_with(mock_request)
        
        # Verify response headers weren't modified
        self.assertEqual(len(mock_response.headers.items()), 0)
    
    async def test_dispatch_under_limit(self):
        """Test middleware when request is under rate limit."""
        # Create request
        mock_request = self.create_mock_request()
        
        # Mock rate limiter response
        self.mock_rate_limiter.is_rate_limited.return_value = (False, 5, 1000)
        
        # Mock call_next function
        mock_response = MagicMock(spec=Response)
        mock_response.headers = {}
        mock_call_next = AsyncMock(return_value=mock_response)
        
        # Call dispatch
        response = await self.middleware.dispatch(mock_request, mock_call_next)
        
        # Verify rate limiter was called
        self.mock_rate_limiter.is_rate_limited.assert_called_once_with(
            "127.0.0.1", 10, 60
        )
        
        # Verify call_next was called with request
        mock_call_next.assert_called_once_with(mock_request)
        
        # Verify response headers were added
        self.assertEqual(response.headers["X-RateLimit-Limit"], "10")
        self.assertEqual(response.headers["X-RateLimit-Remaining"], "5")
        self.assertEqual(response.headers["X-RateLimit-Reset"], "1000")


if __name__ == "__main__":
    unittest.main() 