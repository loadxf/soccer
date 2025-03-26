"""
Tests for the caching middleware.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from src.api.middleware.cache_middleware import CacheMiddleware
from src.utils.cache import CacheManager, CacheSettings


class TestCacheMiddleware(unittest.TestCase):
    """Tests for CacheMiddleware."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a FastAPI app with a simple endpoint
        self.app = FastAPI()
        
        @self.app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        @self.app.get("/test/{item_id}")
        async def test_item_endpoint(item_id: int):
            return {"item_id": item_id, "message": "test item"}
        
        @self.app.post("/test")
        async def test_post_endpoint():
            return {"message": "test post"}
        
        # Create cache manager with testing settings
        self.cache_settings = CacheSettings(
            ttl=60,
            prefix="test_middleware_cache:",
            enabled=True
        )
        self.cache_manager = CacheManager(self.cache_settings)
        
        # Patch the cache manager methods
        self.cache_get_patcher = patch.object(self.cache_manager, 'get')
        self.mock_cache_get = self.cache_get_patcher.start()
        self.mock_cache_get.return_value = None  # Cache miss by default
        
        self.cache_set_patcher = patch.object(self.cache_manager, 'set')
        self.mock_cache_set = self.cache_set_patcher.start()
        self.mock_cache_set.return_value = True
        
        # Add middleware to app
        self.app.add_middleware(
            CacheMiddleware,
            cache_manager=self.cache_manager,
            exclude_paths=["/no-cache"],
            exclude_methods=["POST", "PUT", "DELETE"],
        )
        
        # Create test client
        self.client = TestClient(self.app)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.cache_get_patcher.stop()
        self.cache_set_patcher.stop()
    
    def test_cache_miss(self):
        """Test middleware with cache miss."""
        # Set up cache miss
        self.mock_cache_get.return_value = None
        
        # Make request
        response = self.client.get("/test")
        
        # Verify cache was checked and response was set in cache
        self.mock_cache_get.assert_called_once()
        self.mock_cache_set.assert_called_once()
        
        # Verify response headers
        self.assertEqual(response.headers["X-Cache"], "MISS")
        self.assertEqual(response.headers["Cache-Control"], f"max-age={self.cache_settings.ttl}")
        
        # Verify response content
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "test"})
    
    def test_cache_hit(self):
        """Test middleware with cache hit."""
        # Set up cache hit
        cached_response = json.dumps({"message": "cached"})
        self.mock_cache_get.return_value = cached_response
        
        # Make request
        response = self.client.get("/test")
        
        # Verify cache was checked but not set
        self.mock_cache_get.assert_called_once()
        self.mock_cache_set.assert_not_called()
        
        # Verify response headers
        self.assertEqual(response.headers["X-Cache"], "HIT")
        self.assertEqual(response.headers["Cache-Control"], f"max-age={self.cache_settings.ttl}")
        
        # Verify response content came from cache
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "cached"})
    
    def test_excluded_method(self):
        """Test middleware skips excluded HTTP methods."""
        # Make POST request (excluded method)
        response = self.client.post("/test")
        
        # Verify cache was not used
        self.mock_cache_get.assert_not_called()
        self.mock_cache_set.assert_not_called()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "test post"})
        
        # Verify no cache headers added
        self.assertNotIn("X-Cache", response.headers)
        self.assertNotIn("Cache-Control", response.headers)
    
    def test_path_parameters(self):
        """Test middleware with path parameters."""
        # Make request with path parameter
        response = self.client.get("/test/123")
        
        # Verify cache was checked and response was set in cache
        self.mock_cache_get.assert_called_once()
        self.mock_cache_set.assert_called_once()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"item_id": 123, "message": "test item"})
    
    def test_query_parameters(self):
        """Test middleware with query parameters."""
        # Make request with query parameters
        response = self.client.get("/test?param1=value1&param2=value2")
        
        # Verify cache was checked and response was set in cache
        self.mock_cache_get.assert_called_once()
        self.mock_cache_set.assert_called_once()
        
        # Make same request again
        self.mock_cache_get.reset_mock()
        self.mock_cache_set.reset_mock()
        
        # Set up cache hit for second request
        cached_response = json.dumps({"message": "test"})
        self.mock_cache_get.return_value = cached_response
        
        response = self.client.get("/test?param1=value1&param2=value2")
        
        # Verify cache was checked but not set
        self.mock_cache_get.assert_called_once()
        self.mock_cache_set.assert_not_called()
    
    def test_different_query_order(self):
        """Test middleware with different query parameter order."""
        # Set up a new test client with a patched create_key method
        with patch.object(self.cache_manager, 'create_key') as mock_create_key:
            # Make these requests use the same cache key regardless of param order
            mock_create_key.return_value = "same_key_for_both"
            
            test_client = TestClient(self.app)
            
            # First request with one query parameter order
            test_client.get("/test?param1=value1&param2=value2")
            
            # Second request with different order
            test_client.get("/test?param2=value2&param1=value1")
            
            # Verify create_key was called with both requests
            self.assertEqual(mock_create_key.call_count, 2)
    
    def test_error_response_not_cached(self):
        """Test middleware doesn't cache error responses."""
        # Add error endpoint to app
        @self.app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")
        
        @self.app.exception_handler(ValueError)
        async def value_error_handler(request, exc):
            return JSONResponse(
                status_code=400,
                content={"detail": str(exc)}
            )
        
        # Create new client with updated app
        client = TestClient(self.app)
        
        # Make request that triggers error
        response = client.get("/error")
        
        # Verify cache was checked but error response was not cached
        self.mock_cache_get.assert_called_once()
        self.mock_cache_set.assert_not_called()
        
        # Verify error response
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Test error"})
    
    def test_caching_disabled(self):
        """Test middleware when caching is disabled."""
        # Disable caching
        self.cache_settings.enabled = False
        
        # Make request
        response = self.client.get("/test")
        
        # Verify cache was not used
        self.mock_cache_get.assert_not_called()
        self.mock_cache_set.assert_not_called()
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "test"})


if __name__ == "__main__":
    unittest.main() 