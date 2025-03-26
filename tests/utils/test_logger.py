"""
Unit tests for the logger utility.
"""

import os
import unittest
import logging
import tempfile
from pathlib import Path

# Import project components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger, get_logger


class TestLogger(unittest.TestCase):
    """Test cases for logger utility."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for log files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = os.path.join(self.temp_dir.name, "test.log")
        
        # Store original environment variables
        self.original_log_file = os.environ.get("LOG_FILE")
        self.original_log_level = os.environ.get("LOG_LEVEL")
        
        # Set environment variables for testing
        os.environ["LOG_FILE"] = self.log_file
        os.environ["LOG_LEVEL"] = "DEBUG"
        
    def tearDown(self):
        """Clean up test environment."""
        # Reset environment variables
        if self.original_log_file:
            os.environ["LOG_FILE"] = self.original_log_file
        else:
            os.environ.pop("LOG_FILE", None)
            
        if self.original_log_level:
            os.environ["LOG_LEVEL"] = self.original_log_level
        else:
            os.environ.pop("LOG_LEVEL", None)
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
        
        # Remove all handlers from the root logger to avoid affecting other tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def test_setup_logger(self):
        """Test that setup_logger correctly configures a logger."""
        logger_name = "test_setup_logger"
        logger = setup_logger(logger_name)
        
        # Test logger level
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Test handlers count (file and console)
        self.assertEqual(len(logger.handlers), 2)
        
        # Test handler types
        handler_types = [type(h) for h in logger.handlers]
        self.assertIn(logging.handlers.RotatingFileHandler, handler_types)
        self.assertIn(logging.StreamHandler, handler_types)
        
        # Test file handler configuration
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                self.assertEqual(handler.baseFilename, self.log_file)
                self.assertEqual(handler.maxBytes, 10*1024*1024)  # 10MB
                self.assertEqual(handler.backupCount, 5)
    
    def test_get_logger(self):
        """Test that get_logger returns a correctly named logger."""
        logger_name = "test_component"
        logger = get_logger(logger_name)
        
        # Test logger name
        self.assertEqual(logger.name, f"soccer_prediction.{logger_name}")
        
    def test_logger_functionality(self):
        """Test that logger correctly writes logs to file."""
        logger_name = "test_functionality"
        logger = setup_logger(logger_name)
        
        # Log some messages
        test_message = "This is a test log message"
        logger.debug(test_message)
        logger.info(test_message)
        logger.warning(test_message)
        logger.error(test_message)
        
        # Check that log file exists and contains messages
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            
        # Check all levels were logged
        self.assertIn("DEBUG", log_content)
        self.assertIn("INFO", log_content)
        self.assertIn("WARNING", log_content)
        self.assertIn("ERROR", log_content)
        self.assertIn(test_message, log_content)
    
    def test_multiple_logger_instances(self):
        """Test that multiple logger instances with the same name share handlers."""
        logger_name = "test_multiple"
        
        # Create first logger instance
        logger1 = setup_logger(logger_name)
        handler_count1 = len(logger1.handlers)
        
        # Create second logger instance with same name
        logger2 = setup_logger(logger_name)
        handler_count2 = len(logger2.handlers)
        
        # Check that handlers weren't duplicated
        self.assertEqual(handler_count1, handler_count2)
        self.assertIs(logger1, logger2)


if __name__ == "__main__":
    unittest.main() 