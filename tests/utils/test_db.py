"""
Unit tests for the database utilities.
"""

import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile

# Import project components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.db import (
    Base, engine, metadata, db_session,
    get_mongo_client, get_mongo_db, get_redis_client
)


class TestDatabaseUtils(unittest.TestCase):
    """Test cases for database utilities."""
    
    def setUp(self):
        """Set up test environment."""
        # Store original environment variables
        self.original_db_type = os.environ.get("DB_TYPE")
        self.original_db_host = os.environ.get("DB_HOST")
        self.original_db_port = os.environ.get("DB_PORT")
        self.original_db_name = os.environ.get("DB_NAME")
        self.original_db_user = os.environ.get("DB_USER")
        self.original_db_password = os.environ.get("DB_PASSWORD")
        self.original_mongo_uri = os.environ.get("MONGO_URI")
        self.original_redis_host = os.environ.get("REDIS_HOST")
        self.original_redis_port = os.environ.get("REDIS_PORT")
        self.original_redis_db = os.environ.get("REDIS_DB")
        
        # Set environment variables for testing
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_NAME"] = "test_db"
        os.environ["MONGO_URI"] = "mongodb://127.0.0.1:27017/test_db"
        os.environ["REDIS_HOST"] = "127.0.0.1"
        os.environ["REDIS_PORT"] = "6379"
        os.environ["REDIS_DB"] = "0"
        
    def tearDown(self):
        """Clean up test environment."""
        # Reset environment variables
        self._reset_env_var("DB_TYPE", self.original_db_type)
        self._reset_env_var("DB_HOST", self.original_db_host)
        self._reset_env_var("DB_PORT", self.original_db_port)
        self._reset_env_var("DB_NAME", self.original_db_name)
        self._reset_env_var("DB_USER", self.original_db_user)
        self._reset_env_var("DB_PASSWORD", self.original_db_password)
        self._reset_env_var("MONGO_URI", self.original_mongo_uri)
        self._reset_env_var("REDIS_HOST", self.original_redis_host)
        self._reset_env_var("REDIS_PORT", self.original_redis_port)
        self._reset_env_var("REDIS_DB", self.original_redis_db)
    
    def _reset_env_var(self, var_name, original_value):
        """Helper to reset an environment variable."""
        if original_value is not None:
            os.environ[var_name] = original_value
        else:
            os.environ.pop(var_name, None)
            
    @patch('sqlalchemy.create_engine')
    def test_sqlalchemy_config(self, mock_create_engine):
        """Test SQLAlchemy configuration."""
        # Re-import db to trigger engine creation with test env vars
        with patch.dict('sys.modules', {'src.utils.db': None}):
            from importlib import reload
            import src.utils.db
            reload(src.utils.db)
            
        # Check that create_engine was called with the right connection string
        call_args = mock_create_engine.call_args[0][0]
        self.assertTrue(call_args.startswith("sqlite:///"))
        self.assertTrue(call_args.endswith("test_db.db"))
    
    @patch('src.utils.db.Session')
    def test_db_session_context_manager(self, mock_session):
        """Test database session context manager."""
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Use context manager
        with db_session() as session:
            # Do something with session
            session.query("test")
        
        # Verify session was used correctly
        mock_session_instance.query.assert_called_once_with("test")
        mock_session_instance.commit.assert_called_once()
        mock_session_instance.close.assert_called_once()
        
    @patch('src.utils.db.Session')
    def test_db_session_exception_handling(self, mock_session):
        """Test database session exception handling."""
        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        
        # Simulate an exception
        mock_session_instance.query.side_effect = Exception("Test exception")
        
        # Use context manager with exception
        with self.assertRaises(Exception):
            with db_session() as session:
                session.query("test")
        
        # Verify rollback was called
        mock_session_instance.rollback.assert_called_once()
        mock_session_instance.close.assert_called_once()
    
    @patch('src.utils.db.MongoClient')
    def test_get_mongo_client(self, mock_mongo_client):
        """Test getting MongoDB client."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        
        # Reset module-level variables to ensure clean test
        import src.utils.db
        src.utils.db._mongo_client = None
        
        # Call function
        client = get_mongo_client()
        
        # Verify client was created with correct URI
        mock_mongo_client.assert_called_once_with(os.environ["MONGO_URI"])
        
        # Verify ping was called to test connection
        mock_client_instance.admin.command.assert_called_once_with('ping')
        
        # Call again to test singleton pattern
        client2 = get_mongo_client()
        
        # Verify MongoClient was only created once
        mock_mongo_client.assert_called_once()
        self.assertEqual(client, client2)
    
    @patch('src.utils.db.get_mongo_client')
    def test_get_mongo_db(self, mock_get_mongo_client):
        """Test getting MongoDB database."""
        # Setup mock
        mock_client = MagicMock()
        mock_get_mongo_client.return_value = mock_client
        
        # Call function with explicit db name
        db_name = "test_db"
        db = get_mongo_db(db_name)
        
        # Verify correct db was accessed
        mock_client.__getitem__.assert_called_once_with(db_name)
        
        # Call function with implicit db name (from URI)
        get_mongo_db()
        
        # Verify db name was extracted from URI
        mock_client.__getitem__.assert_called_with("test_db")
    
    @patch('src.utils.db.redis.Redis')
    def test_get_redis_client(self, mock_redis):
        """Test getting Redis client."""
        # Setup mock
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        
        # Reset module-level variables to ensure clean test
        import src.utils.db
        src.utils.db._redis_client = None
        
        # Call function
        client = get_redis_client()
        
        # Verify Redis client was created with correct parameters
        mock_redis.assert_called_once_with(
            host=os.environ["REDIS_HOST"],
            port=int(os.environ["REDIS_PORT"]),
            db=int(os.environ["REDIS_DB"]),
            decode_responses=True
        )
        
        # Verify ping was called to test connection
        mock_redis_instance.ping.assert_called_once()
        
        # Call again to test singleton pattern
        client2 = get_redis_client()
        
        # Verify Redis client was only created once
        mock_redis.assert_called_once()
        self.assertEqual(client, client2)


if __name__ == "__main__":
    unittest.main() 