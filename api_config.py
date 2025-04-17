"""
Shared configuration for the Soccer Prediction System.
This file provides centralized configuration for both API and UI components.
"""

import os
import platform
import socket
from typing import List

# Base configuration
API_VERSION = "v1"
# Docker uses port 8000, not 8080
API_PORT = 8000  # Must match docker-compose port mapping
REQUEST_TIMEOUT = 10  # seconds

# Determine the best hostname to use based on system capabilities
def get_preferred_hostname() -> str:
    """
    Determine the best hostname to use based on system capabilities.
    Returns 'localhost' or '127.0.0.1' depending on what works best on this system.
    """
    # Check if we're running in Docker
    if os.path.exists('/.dockerenv'):
        # In Docker, use 0.0.0.0 to allow container networking
        return '0.0.0.0'
    
    # Try to connect to localhost first
    try:
        socket.gethostbyname('localhost')
        # If this works, localhost is properly configured
        return 'localhost'
    except socket.gaierror:
        # Fall back to IP address if localhost isn't resolving properly
        return '127.0.0.1'

# Set API host using the preferred hostname
API_HOST = get_preferred_hostname()

# Set API base URL - remove '/api/v1' since the API doesn't use this path structure
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# UI configuration
UI_PORT = 8501
UI_BASE_URL = f"http://{API_HOST}:{UI_PORT}"

# CORS configuration
CORS_ORIGINS: List[str] = [
    f"http://localhost:{UI_PORT}",
    f"http://127.0.0.1:{UI_PORT}",
    f"http://localhost:{API_PORT}",
    f"http://127.0.0.1:{API_PORT}",
    f"http://app:{API_PORT}",  # Add Docker service name
    f"http://ui:{UI_PORT}",    # Add Docker service name
    f"http://frontend:80",     # Add frontend container
    f"http://localhost:3000",  # Add mapped frontend port
    "null"  # Allow file:// protocol requests
]

# API server configuration
API_TITLE = "Soccer Prediction System API"
API_DESCRIPTION = "REST API for soccer match predictions"

# Feature flags
ENABLE_DEBUG = True
ENABLE_LOGGING = True
ENABLE_FALLBACK_DATA = True

# Define different configurations based on environment
ENV = os.environ.get("APP_ENV", "development")

if ENV == "production":
    ENABLE_DEBUG = False
    # In production, we might use a different host/port
    # API_HOST = "api.example.com"
    # API_PORT = 443
    # API_BASE_URL = f"https://{API_HOST}/api/{API_VERSION}"

# Helper functions for hostname resolution
def try_connect(hostname: str, port: int) -> bool:
    """Try to connect to a hostname:port to check if it's available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # 2 second timeout
        result = sock.connect_ex((hostname, port))
        sock.close()
        return result == 0  # 0 means connection succeeded
    except:
        return False

def get_alternative_hostname(hostname: str) -> str:
    """Get the alternative hostname (localhost vs 127.0.0.1)."""
    return '127.0.0.1' if hostname == 'localhost' else 'localhost'

def get_working_api_url() -> str:
    """
    Attempt to find a working API URL by trying both localhost and 127.0.0.1.
    Returns the first URL that works, or the default if none work.
    """
    if try_connect(API_HOST, API_PORT):
        return API_BASE_URL
    
    # Try the alternative hostname
    alt_host = get_alternative_hostname(API_HOST)
    if try_connect(alt_host, API_PORT):
        return f"http://{alt_host}:{API_PORT}"
    
    # If neither works, return the default
    return API_BASE_URL 