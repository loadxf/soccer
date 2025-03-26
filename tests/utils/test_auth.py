"""
Unit tests for the authentication utility.
"""

import os
import unittest
import jwt
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import project components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.auth import (
    verify_password, get_password_hash, get_user, authenticate_user,
    create_access_token, UserInDB, User, users_db, ALGORITHM
)


class TestAuth(unittest.TestCase):
    """Test cases for authentication utility."""
    
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
    
    def test_user_in_db_class(self):
        """Test the UserInDB class."""
        # Test successful initialization
        user_data = {
            "username": self.test_username,
            "email": self.test_email,
            "hashed_password": "hashed_" + self.test_password,
            "disabled": False
        }
        user = UserInDB(**user_data)
        
        # Check all fields were set
        for key, value in user_data.items():
            self.assertEqual(user[key], value)
        
        # Test initialization with missing required fields
        with self.assertRaises(ValueError):
            UserInDB(username=self.test_username)
            
    def test_user_class(self):
        """Test the User class."""
        # Test that hashed_password is filtered out
        user_data = {
            "username": self.test_username,
            "email": self.test_email,
            "hashed_password": "hashed_" + self.test_password,
            "disabled": False
        }
        user = User(**user_data)
        
        # Check hashed_password is not in the user dict
        self.assertNotIn("hashed_password", user)
        
        # Check other fields were set
        self.assertEqual(user["username"], self.test_username)
        self.assertEqual(user["email"], self.test_email)
        self.assertEqual(user["disabled"], False)
        
    def test_password_hashing(self):
        """Test password hashing and verification."""
        # Test password hashing
        hashed_password = get_password_hash(self.test_password)
        self.assertNotEqual(hashed_password, self.test_password)
        
        # Test password verification
        self.assertTrue(verify_password(self.test_password, hashed_password))
        self.assertFalse(verify_password("wrong_password", hashed_password))
        
    def test_get_user(self):
        """Test getting user by username."""
        # Test with non-existent user
        self.assertIsNone(get_user("nonexistent_user"))
        
        # Add test user to users_db
        users_db[self.test_username] = {
            "username": self.test_username,
            "email": self.test_email,
            "hashed_password": get_password_hash(self.test_password),
            "disabled": False
        }
        
        # Test with existing user
        user = get_user(self.test_username)
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], self.test_username)
        self.assertEqual(user["email"], self.test_email)
        
    def test_authenticate_user(self):
        """Test user authentication."""
        # Add test user to users_db
        hashed_password = get_password_hash(self.test_password)
        users_db[self.test_username] = {
            "username": self.test_username,
            "email": self.test_email,
            "hashed_password": hashed_password,
            "disabled": False
        }
        
        # Test with correct credentials
        user = authenticate_user(self.test_username, self.test_password)
        self.assertIsNotNone(user)
        self.assertEqual(user["username"], self.test_username)
        
        # Test with incorrect password
        user = authenticate_user(self.test_username, "wrong_password")
        self.assertIsNone(user)
        
        # Test with non-existent user
        user = authenticate_user("nonexistent_user", self.test_password)
        self.assertIsNone(user)
        
    def test_create_access_token(self):
        """Test JWT token creation."""
        # Test token creation with default expiry
        data = {"sub": self.test_username}
        token = create_access_token(data)
        
        # Decode token and check claims
        payload = jwt.decode(token, os.environ["SECRET_KEY"], algorithms=[ALGORITHM])
        self.assertEqual(payload["sub"], self.test_username)
        self.assertIn("exp", payload)
        
        # Test token creation with custom expiry
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta=expires_delta)
        
        # Decode token and check expiry is approximately 30 minutes from now
        payload = jwt.decode(token, os.environ["SECRET_KEY"], algorithms=[ALGORITHM])
        exp_time = datetime.fromtimestamp(payload["exp"])
        now = datetime.utcnow()
        self.assertAlmostEqual(
            (exp_time - now).total_seconds(), 
            expires_delta.total_seconds(),
            delta=10  # Allow for a small difference due to execution time
        )


if __name__ == "__main__":
    unittest.main() 