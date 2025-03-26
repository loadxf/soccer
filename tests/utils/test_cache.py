"""
Unit tests for the caching utility.
"""

import unittest
import json
from unittest.mock import patch, MagicMock

from fastapi import Request
from starlette.datastructures import URL

from src.utils.cache import (
    CacheSettings, 
    CacheManager, 
    cached, 
    clear_all_cache, 
    clear_cache_for_prefix
)

class TestCacheSettings(unittest.TestCase):
    """Tests for the CacheSettings class."""
    
    def test_default_settings(self):
        """Test default cache settings."""
        settings = CacheSettings()
        self.assertEqual(settings.ttl, 300)  # 5 minutes default
        self.assertEqual(settings.prefix, "soccer_api_cache:")
        self.assertEqual(settings.ignore_query_params, set())
        self.assertTrue(settings.enabled)
    
    def test_custom_settings(self):
        """Test custom cache settings."""
        settings = CacheSettings(
            ttl=60,
            prefix="test_prefix:",
            ignore_query_params=["token", "api_key"],
            enabled=False
        )
        self.assertEqual(settings.ttl, 60)
        self.assertEqual(settings.prefix, "test_prefix:")
        self.assertEqual(settings.ignore_query_params, {"token", "api_key"})
        self.assertFalse(settings.enabled)


class TestCacheManager(unittest.TestCase):
    """Tests for the CacheManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.settings = CacheSettings(
            ttl=60,
            prefix="test_cache:",
            ignore_query_params=["token"]
        )
        self.cache_manager = CacheManager(self.settings)
        
        # Create patcher for Redis client
        self.redis_patcher = patch('src.utils.cache.get_redis_client')
        self.mock_redis_getter = self.redis_patcher.start()
        self.mock_redis = MagicMock()
        self.mock_redis_getter.return_value = self.mock_redis
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.redis_patcher.stop()
    
    def test_create_key(self):
        """Test cache key creation."""
        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.url = URL("http://testserver/api/v1/teams")
        mock_request.query_params = {}
        
        # Test basic key creation
        key = self.cache_manager.create_key(mock_request)
        self.assertTrue(key.startswith("test_cache:"))
        
        # Test with query params
        mock_request.url = URL("http://testserver/api/v1/teams?limit=10&offset=0")
        mock_request.query_params = {"limit": "10", "offset": "0"}
        key = self.cache_manager.create_key(mock_request)
        self.assertTrue(key.startswith("test_cache:"))
        
        # Test with ignored query params
        mock_request.url = URL("http://testserver/api/v1/teams?limit=10&token=abc123")
        mock_request.query_params = {"limit": "10", "token": "abc123"}
        key = self.cache_manager.create_key(mock_request)
        # The key should be different from one with just limit param
        key2 = self.cache_manager.create_key(mock_request)
        self.assertEqual(key, key2)
        
        # Different query param order should result in same key
        mock_request.url = URL("http://testserver/api/v1/teams?token=abc123&limit=10")
        mock_request.query_params = {"token": "abc123", "limit": "10"}
        key3 = self.cache_manager.create_key(mock_request)
        self.assertEqual(key, key3)
    
    async def test_get(self):
        """Test getting value from cache."""
        # Setup Redis mock
        self.mock_redis.get.return_value = json.dumps({"data": "test"})
        
        # Get value from cache
        value = await self.cache_manager.get("test_key")
        self.assertEqual(value, json.dumps({"data": "test"}))
        self.mock_redis.get.assert_called_once_with("test_key")
        
        # Test with exception
        self.mock_redis.get.side_effect = Exception("Redis error")
        value = await self.cache_manager.get("test_key")
        self.assertIsNone(value)
    
    async def test_set(self):
        """Test setting value in cache."""
        # Set value in cache
        result = await self.cache_manager.set("test_key", json.dumps({"data": "test"}))
        self.assertTrue(result)
        self.mock_redis.setex.assert_called_once_with(
            "test_key", 
            self.settings.ttl,
            json.dumps({"data": "test"})
        )
        
        # Test with custom TTL
        result = await self.cache_manager.set("test_key", json.dumps({"data": "test"}), ttl=120)
        self.assertTrue(result)
        self.mock_redis.setex.assert_called_with(
            "test_key", 
            120,
            json.dumps({"data": "test"})
        )
        
        # Test with exception
        self.mock_redis.setex.side_effect = Exception("Redis error")
        result = await self.cache_manager.set("test_key", json.dumps({"data": "test"}))
        self.assertFalse(result)
    
    async def test_delete(self):
        """Test deleting value from cache."""
        # Delete value from cache
        result = await self.cache_manager.delete("test_key")
        self.assertTrue(result)
        self.mock_redis.delete.assert_called_once_with("test_key")
        
        # Test with exception
        self.mock_redis.delete.side_effect = Exception("Redis error")
        result = await self.cache_manager.delete("test_key")
        self.assertFalse(result)
    
    async def test_delete_pattern(self):
        """Test deleting values matching pattern."""
        # Set up Redis mock
        self.mock_redis.keys.return_value = ["test_cache:key1", "test_cache:key2"]
        
        # Delete values matching pattern
        count = await self.cache_manager.delete_pattern("key*")
        self.assertEqual(count, 2)
        self.mock_redis.keys.assert_called_once_with("test_cache:key*")
        self.mock_redis.delete.assert_called_once_with("test_cache:key1", "test_cache:key2")
        
        # Test with no matching keys
        self.mock_redis.keys.return_value = []
        count = await self.cache_manager.delete_pattern("nonexistent*")
        self.assertEqual(count, 0)
        
        # Test with exception
        self.mock_redis.keys.side_effect = Exception("Redis error")
        count = await self.cache_manager.delete_pattern("key*")
        self.assertEqual(count, 0)
    
    async def test_clear_all(self):
        """Test clearing all cache."""
        # Mock delete_pattern method
        with patch.object(self.cache_manager, 'delete_pattern', return_value=5) as mock_delete_pattern:
            count = await self.cache_manager.clear_all()
            self.assertEqual(count, 5)
            mock_delete_pattern.assert_called_once_with("*")


class TestCachedDecorator(unittest.TestCase):
    """Tests for the cached decorator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patch for cache_manager
        self.cache_manager_patcher = patch('src.utils.cache.cache_manager')
        self.mock_cache_manager = self.cache_manager_patcher.start()
        
        # Setup standard mock behaviors
        self.mock_cache_manager.settings.enabled = True
        self.mock_cache_manager.create_key.return_value = "test_key"
        self.mock_cache_manager.get.return_value = None  # Cache miss by default
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.cache_manager_patcher.stop()
    
    async def test_cache_hit(self):
        """Test cached decorator with cache hit."""
        # Setup cache hit
        self.mock_cache_manager.get.return_value = json.dumps({"data": "cached"})
        
        # Create decorated function
        @cached()
        async def test_func(request):
            return {"data": "original"}
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        
        # Call function
        result = await test_func(mock_request)
        
        # Verify result is from cache
        self.assertEqual(result, {"data": "cached"})
        self.mock_cache_manager.get.assert_called_once_with("test_key")
        self.mock_cache_manager.set.assert_not_called()
    
    async def test_cache_miss(self):
        """Test cached decorator with cache miss."""
        # Setup cache miss
        self.mock_cache_manager.get.return_value = None
        
        # Create decorated function
        @cached()
        async def test_func(request):
            return {"data": "original"}
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        
        # Call function
        result = await test_func(mock_request)
        
        # Verify result is from function and cache was set
        self.assertEqual(result, {"data": "original"})
        self.mock_cache_manager.get.assert_called_once_with("test_key")
        self.mock_cache_manager.set.assert_called_once_with(
            "test_key",
            json.dumps({"data": "original"}),
            ttl=None  # Uses default TTL
        )
    
    async def test_custom_ttl(self):
        """Test cached decorator with custom TTL."""
        # Setup cache miss
        self.mock_cache_manager.get.return_value = None
        
        # Create decorated function with custom TTL
        @cached(ttl=120)
        async def test_func(request):
            return {"data": "original"}
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        
        # Call function
        result = await test_func(mock_request)
        
        # Verify custom TTL was used
        self.mock_cache_manager.set.assert_called_once_with(
            "test_key",
            json.dumps({"data": "original"}),
            ttl=120
        )
    
    async def test_key_prefix(self):
        """Test cached decorator with key prefix."""
        # Setup
        self.mock_cache_manager.get.return_value = None
        
        # Create decorated function with key prefix
        @cached(key_prefix="prefix")
        async def test_func(request):
            return {"data": "original"}
        
        # Mock behavior for prefixed key
        self.mock_cache_manager.create_key.return_value = "base_key"
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        
        # Call function
        result = await test_func(mock_request)
        
        # Verify key was prefixed
        self.mock_cache_manager.set.assert_called_once_with(
            "prefix:base_key",
            json.dumps({"data": "original"}),
            ttl=None
        )
    
    async def test_caching_disabled(self):
        """Test cached decorator with caching disabled."""
        # Setup caching disabled
        self.mock_cache_manager.settings.enabled = False
        
        # Create decorated function
        @cached()
        async def test_func(request):
            return {"data": "original"}
        
        # Create mock request
        mock_request = MagicMock(spec=Request)
        
        # Call function
        result = await test_func(mock_request)
        
        # Verify cache was not used
        self.assertEqual(result, {"data": "original"})
        self.mock_cache_manager.get.assert_not_called()
        self.mock_cache_manager.set.assert_not_called()
    
    async def test_no_request_parameter(self):
        """Test cached decorator with no request parameter."""
        # Create decorated function without request parameter
        @cached()
        async def test_func(param1, param2):
            return {"param1": param1, "param2": param2}
        
        # Call function
        result = await test_func("value1", "value2")
        
        # Verify function was called but caching was skipped
        self.assertEqual(result, {"param1": "value1", "param2": "value2"})
        self.mock_cache_manager.get.assert_not_called()
        self.mock_cache_manager.set.assert_not_called()


class TestClearCacheFunctions(unittest.TestCase):
    """Tests for the clear_cache functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patch for cache_manager
        self.cache_manager_patcher = patch('src.utils.cache.cache_manager')
        self.mock_cache_manager = self.cache_manager_patcher.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.cache_manager_patcher.stop()
    
    async def test_clear_cache_for_prefix(self):
        """Test clear_cache_for_prefix function."""
        # Setup
        self.mock_cache_manager.delete_pattern.return_value = 3
        
        # Call function
        count = await clear_cache_for_prefix("test")
        
        # Verify
        self.assertEqual(count, 3)
        self.mock_cache_manager.delete_pattern.assert_called_once_with("test*")
    
    async def test_clear_all_cache(self):
        """Test clear_all_cache function."""
        # Setup
        self.mock_cache_manager.clear_all.return_value = 10
        
        # Call function
        count = await clear_all_cache()
        
        # Verify
        self.assertEqual(count, 10)
        self.mock_cache_manager.clear_all.assert_called_once()


if __name__ == "__main__":
    unittest.main() 