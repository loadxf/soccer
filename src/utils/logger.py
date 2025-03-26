"""
Logging configuration for the Soccer Prediction System.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Try to import configuration
try:
    from config.default_config import LOG_LEVEL, LOG_FILE, BASE_DIR
except ImportError:
    # Fallback to defaults if config is not available
    LOG_LEVEL = "INFO"
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    LOG_FILE = os.path.join(BASE_DIR, "app.log")

# Ensure log directory exists
log_dir = os.path.dirname(LOG_FILE)
# Fix: Check if log_dir is not empty before trying to create it
if log_dir:
    os.makedirs(log_dir, exist_ok=True)
else:
    # If log_dir is empty, use a default logs directory
    log_dir = os.path.join(str(BASE_DIR), "logs")
    os.makedirs(log_dir, exist_ok=True)
    LOG_FILE = os.path.join(log_dir, "app.log")

# Configure logging
def setup_logger(name="soccer_prediction"):
    """Set up and configure a logger with the given name."""
    logger = logging.getLogger(name)
    
    # Only set up handlers if none exist yet
    if not logger.handlers:
        # Set level based on config or environment variable
        log_level = getattr(logging, os.getenv("LOG_LEVEL", LOG_LEVEL).upper())
        logger.setLevel(log_level)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create file handler
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger

# Create a default logger
logger = setup_logger()

def get_logger(name):
    """Get a child logger with the given name."""
    return logging.getLogger(f"soccer_prediction.{name}") 