"""
API Configuration Module

This module provides centralized configuration for API connection settings,
including hostname resolution with fallback mechanisms.
"""

import logging
import socket
import requests
import os
from typing import Tuple, Optional
import time

# Constants
DEFAULT_API_HOST = "localhost"
FALLBACK_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000  # Updated from 8080 to 8000
DEFAULT_API_TIMEOUT = 10  # seconds

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_config")

class ApiConfig:
    """Centralized API configuration with fallback mechanisms"""
    
    def __init__(self):
        # Check if we're running in Docker
        self.is_docker = os.path.exists('/.dockerenv')
        
        # In Docker, use the service name
        if self.is_docker:
            self.api_host = "app"
        else:
            # Try to get host from environment variable if available
            self.api_host = os.environ.get('API_HOST', DEFAULT_API_HOST)
            
        self.api_port = int(os.environ.get('API_PORT', DEFAULT_API_PORT))
        self.api_timeout = DEFAULT_API_TIMEOUT
        self.fallback_mode = False
        self.session_id = int(time.time())  # Unique identifier for current session
    
    @property
    def api_base_url(self) -> str:
        """Return the base URL for API calls."""
        # No longer using /api/v1 prefix
        return f"http://{self.api_host}:{self.api_port}"
    
    @property
    def api_health_url(self) -> str:
        """Return the health check URL."""
        # Direct health endpoint
        return f"{self.api_base_url}/health"
    
    def get_best_hostname(self) -> str:
        """Determine the best hostname to use, testing both options if needed."""
        # Check for remote host in environment
        remote_host = os.environ.get('REMOTE_API_HOST')
        if remote_host and self._test_connection(remote_host):
            logger.info(f"Remote host '{remote_host}' is accessible")
            self.api_host = remote_host
            self.fallback_mode = False
            return remote_host
        
        # In Docker, always use the container service name first
        if self.is_docker and self._test_connection("app"):
            logger.info("Docker service name 'app' is accessible")
            self.api_host = "app"
            self.fallback_mode = False
            return "app"
            
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
            # First test if the hostname resolves (skip for container names and IP addresses)
            if hostname not in ["app"] and not self._is_ip_address(hostname):
                socket.gethostbyname(hostname)
            
            # Then test if the API responds with direct health endpoint
            response = requests.get(
                f"http://{hostname}:{self.api_port}/health",
                timeout=self.api_timeout
            )
            return response.status_code == 200
        except (socket.gaierror, requests.RequestException) as e:
            logger.warning(f"Connection test failed for {hostname}: {str(e)}")
            return False
    
    def _is_ip_address(self, hostname: str) -> bool:
        """Check if the hostname is an IP address."""
        try:
            socket.inet_aton(hostname)
            return True
        except socket.error:
            return False
    
    def get_request_url(self, endpoint: str) -> str:
        """Get the full URL for an API endpoint with current hostname."""
        # Ensure endpoint doesn't start with a slash
        endpoint = endpoint.lstrip('/')
        return f"{self.api_base_url}/{endpoint}"
    
    def reset(self) -> None:
        """Reset the configuration to defaults."""
        # Check for remote host in environment
        remote_host = os.environ.get('REMOTE_API_HOST')
        if remote_host:
            self.api_host = remote_host
        elif self.is_docker:
            self.api_host = "app"
        else:
            self.api_host = DEFAULT_API_HOST
        self.api_port = int(os.environ.get('API_PORT', DEFAULT_API_PORT))
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
    # Check if we're running in Docker
    is_docker = os.path.exists('/.dockerenv')
    
    # Check for remote host in environment
    remote_host = os.environ.get('REMOTE_API_HOST')
    if remote_host:
        try:
            response = requests.get(
                f"http://{remote_host}:{DEFAULT_API_PORT}/health",
                timeout=DEFAULT_API_TIMEOUT
            )
            if response.status_code == 200:
                return True, f"API accessible via remote host ({remote_host})", response.json()
        except requests.RequestException:
            pass
    
    # Try Docker service name first if in Docker
    if is_docker:
        try:
            response = requests.get(
                f"http://app:{DEFAULT_API_PORT}/health",
                timeout=DEFAULT_API_TIMEOUT
            )
            if response.status_code == 200:
                return True, "API accessible via Docker service name", response.json()
        except requests.RequestException:
            pass
    
    # Try primary hostname
    try:
        response = requests.get(
            f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/health",
            timeout=DEFAULT_API_TIMEOUT
        )
        if response.status_code == 200:
            return True, f"API accessible via {DEFAULT_API_HOST}", response.json()
    except requests.RequestException:
        pass
    
    # Try fallback hostname
    try:
        response = requests.get(
            f"http://{FALLBACK_API_HOST}:{DEFAULT_API_PORT}/health",
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