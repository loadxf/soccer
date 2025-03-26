"""
Tests for the Soccer Prediction System API rate limiting functionality.
"""

import os
import sys
import unittest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add the project root to the path so that imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the FastAPI app and rate limiting components
from src.api.server import app
from src.utils.auth import create_access_token, UserInDB, users_db
from src.utils.rate_limit import RateLimitMiddleware, rate_limit

# Create test client
client = TestClient(app)


class TestAPIRateLimiting(unittest.TestCase):
    """Test cases for API rate limiting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Store original environment variables
        self.original_secret_key = os.environ.get("SECRET_KEY")
        self.original_expire_minutes = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")
        
        # Set environment variables for testing
        os.environ["SECRET_KEY"] = "test_secret_key"
        os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "15"
        
        # Test user data
        self.test_username = "testuser"
        self.test_password = "testpassword"
        self.test_email = "test@example.com"
        
        # Add test user to users_db with admin role
        from src.utils.auth import get_password_hash
        users_db[self.test_username] = {
            "username": self.test_username,
            "email": self.test_email,
            "hashed_password": get_password_hash(self.test_password),
            "disabled": False,
            "roles": ["user", "admin"]
        }
        
        # Create access token for tests
        token_data = {"sub": self.test_username}
        self.access_token = create_access_token(token_data)
        self.headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Set up test middleware with low limits for testing
        self.test_rate_limit = 3  # 3 requests allowed
        self.test_window = 5  # 5 seconds window
        
        # Apply patch to rate limit settings
        self.rate_limit_patcher = patch('src.utils.rate_limit.RATE_LIMIT', self.test_rate_limit)
        self.window_patcher = patch('src.utils.rate_limit.WINDOW', self.test_window)
        self.rate_limit_patcher.start()
        self.window_patcher.start()
        
    def tearDown(self):
        """Clean up test environment."""
        # Reset environment variables
        if self.original_secret_key:
            os.environ["SECRET_KEY"] = self.original_secret_key
        else:
            os.environ.pop("SECRET_KEY", None)
            
        if self.original_expire_minutes:
            os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = self.original_expire_minutes
        else:
            os.environ.pop("ACCESS_TOKEN_EXPIRE_MINUTES", None)
        
        # Clean up test user if added
        if self.test_username in users_db:
            del users_db[self.test_username]
            
        # Stop patches
        self.rate_limit_patcher.stop()
        self.window_patcher.stop()
        
        # Clear rate limit cache
        from src.utils.rate_limit import _rate_limit_cache
        _rate_limit_cache.clear()
    
    def test_rate_limit_exceeded(self):
        """Test that API endpoints enforce rate limits."""
        # Create a new client for this test to ensure clean state
        test_client = TestClient(app)
        
        # Make multiple requests up to the limit
        for i in range(self.test_rate_limit):
            response = test_client.get("/api/v1/health", headers=self.headers)
            self.assertEqual(response.status_code, 200, f"Request {i+1} should succeed")
            
            # Check rate limit headers
            self.assertIn("X-RateLimit-Limit", response.headers)
            self.assertIn("X-RateLimit-Remaining", response.headers)
            self.assertIn("X-RateLimit-Reset", response.headers)
            
            limit = int(response.headers["X-RateLimit-Limit"])
            remaining = int(response.headers["X-RateLimit-Remaining"])
            
            self.assertEqual(limit, self.test_rate_limit)
            self.assertEqual(remaining, self.test_rate_limit - (i + 1))
        
        # Next request should be rate limited
        response = test_client.get("/api/v1/health", headers=self.headers)
        self.assertEqual(response.status_code, 429)  # Too Many Requests
        
        # Check error response
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Rate limit exceeded", data["detail"])
        
        # Check rate limit headers in error response
        self.assertIn("X-RateLimit-Limit", response.headers)
        self.assertIn("X-RateLimit-Remaining", response.headers)
        self.assertIn("X-RateLimit-Reset", response.headers)
        self.assertEqual(int(response.headers["X-RateLimit-Remaining"]), 0)
        
        # Retry-After header should be present
        self.assertIn("Retry-After", response.headers)
    
    def test_rate_limit_reset(self):
        """Test that rate limits reset after the window period."""
        # This test requires actual waiting time to allow the rate limit window to pass
        test_client = TestClient(app)
        
        # Make requests up to the limit
        for i in range(self.test_rate_limit):
            response = test_client.get("/api/v1/health", headers=self.headers)
            self.assertEqual(response.status_code, 200)
        
        # Next request should fail
        response = test_client.get("/api/v1/health", headers=self.headers)
        self.assertEqual(response.status_code, 429)
        
        # Wait for the rate limit window to expire
        print(f"Waiting {self.test_window} seconds for rate limit window to expire...")
        time.sleep(self.test_window + 1)
        
        # After waiting, we should be able to make requests again
        response = test_client.get("/api/v1/health", headers=self.headers)
        self.assertEqual(response.status_code, 200)
    
    def test_different_endpoints_same_limit(self):
        """Test that rate limits apply across different endpoints for the same user."""
        test_client = TestClient(app)
        
        # Make requests to different endpoints
        response = test_client.get("/api/v1/health", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        response = test_client.get("/api/v1/", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        response = test_client.get("/api/v1/matches/1", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        # Fourth request should hit the rate limit
        response = test_client.get("/api/v1/health", headers=self.headers)
        self.assertEqual(response.status_code, 429)
    
    def test_different_users_different_limits(self):
        """Test that rate limits are applied per-user and not globally."""
        test_client = TestClient(app)
        
        # Create a second user
        second_username = "testuser2"
        from src.utils.auth import get_password_hash
        users_db[second_username] = {
            "username": second_username,
            "email": "test2@example.com",
            "hashed_password": get_password_hash("password2"),
            "disabled": False,
            "roles": ["user"]
        }
        
        # Create token for second user
        token_data = {"sub": second_username}
        second_token = create_access_token(token_data)
        second_headers = {"Authorization": f"Bearer {second_token}"}
        
        # Make requests up to the limit with first user
        for i in range(self.test_rate_limit):
            response = test_client.get("/api/v1/health", headers=self.headers)
            self.assertEqual(response.status_code, 200)
        
        # First user should now be rate limited
        response = test_client.get("/api/v1/health", headers=self.headers)
        self.assertEqual(response.status_code, 429)
        
        # Second user should still be able to make requests
        for i in range(self.test_rate_limit):
            response = test_client.get("/api/v1/health", headers=second_headers)
            self.assertEqual(response.status_code, 200)
        
        # Clean up second user
        del users_db[second_username]


if __name__ == "__main__":
    unittest.main() 