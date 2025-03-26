"""
API Configuration Module

This module provides centralized configuration for API connection settings,
including hostname resolution with fallback mechanisms.
"""

import logging
import socket
import requests
from typing import Tuple, Optional
import time

# Constants
DEFAULT_API_HOST = "localhost"
FALLBACK_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8080
DEFAULT_API_TIMEOUT = 10  # seconds

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_config")

class ApiConfig:
    """Centralized API configuration with fallback mechanisms"""
    
    def __init__(self):
        self.api_host = DEFAULT_API_HOST
        self.api_port = DEFAULT_API_PORT
        self.api_timeout = DEFAULT_API_TIMEOUT
        self.fallback_mode = False
        self.session_id = int(time.time())  # Unique identifier for current session
    
    @property
    def api_base_url(self) -> str:
        """Return the base URL for API calls."""
        return f"http://{self.api_host}:{self.api_port}/api/v1"
    
    @property
    def api_health_url(self) -> str:
        """Return the health check URL."""
        return f"{self.api_base_url}/health"
    
    def get_best_hostname(self) -> str:
        """Determine the best hostname to use, testing both options if needed."""
        if self._test_connection(DEFAULT_API_HOST):
            logger.info(f"Primary hostname '{DEFAULT_API_HOST}' is accessible")
            self.api_host = DEFAULT_API_HOST
            self.fallback_mode = False
            return DEFAULT_API_HOST
            
        logger.warning(f"Primary hostname '{DEFAULT_API_HOST}' failed, trying fallback '{FALLBACK_API_HOST}'")
        if self._test_connection(FALLBACK_API_HOST):
            logger.info(f"Fallback hostname '{FALLBACK_API_HOST}' is accessible")
            self.api_host = FALLBACK_API_HOST
            self.fallback_mode = True
            return FALLBACK_API_HOST
            
        logger.error("Both primary and fallback hostnames failed")
        self.fallback_mode = True
        return DEFAULT_API_HOST  # Return the default even though it failed
    
    def _test_connection(self, hostname: str) -> bool:
        """Test if the API is accessible using the given hostname."""
        try:
            # First test if the hostname resolves
            socket.gethostbyname(hostname)
            
            # Then test if the API responds
            response = requests.get(
                f"http://{hostname}:{self.api_port}/api/v1/health",
                timeout=self.api_timeout
            )
            return response.status_code == 200
        except (socket.gaierror, requests.RequestException) as e:
            logger.warning(f"Connection test failed for {hostname}: {str(e)}")
            return False
    
    def get_request_url(self, endpoint: str) -> str:
        """Get the full URL for an API endpoint with current hostname."""
        # Ensure endpoint doesn't start with a slash
        endpoint = endpoint.lstrip('/')
        return f"{self.api_base_url}/{endpoint}"
    
    def reset(self) -> None:
        """Reset the configuration to defaults."""
        self.api_host = DEFAULT_API_HOST
        self.api_port = DEFAULT_API_PORT
        self.api_timeout = DEFAULT_API_TIMEOUT
        self.fallback_mode = False
        self.session_id = int(time.time())

# Create a singleton instance
config = ApiConfig()

def test_api_connection() -> Tuple[bool, str, Optional[dict]]:
    """
    Test the API connection using both hostnames and return the status.
    
    Returns:
        Tuple[bool, str, Optional[dict]]: 
            - Success status
            - Message describing the result
            - Response data if successful, None otherwise
    """
    # Try primary hostname
    try:
        response = requests.get(
            f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/api/v1/health",
            timeout=DEFAULT_API_TIMEOUT
        )
        if response.status_code == 200:
            return True, f"API accessible via {DEFAULT_API_HOST}", response.json()
    except requests.RequestException:
        pass
    
    # Try fallback hostname
    try:
        response = requests.get(
            f"http://{FALLBACK_API_HOST}:{DEFAULT_API_PORT}/api/v1/health",
            timeout=DEFAULT_API_TIMEOUT
        )
        if response.status_code == 200:
            return True, f"API accessible via {FALLBACK_API_HOST} (fallback)", response.json()
    except requests.RequestException:
        pass
    
    return False, "API is not accessible", None

def reset_configuration():
    """Reset the API configuration to defaults and redetermine best hostname."""
    config.reset()
    config.get_best_hostname()
    logger.info(f"API configuration reset. Using {config.api_host}:{config.api_port}") 