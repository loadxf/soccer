"""
Integration service for the Soccer Prediction System API.
This module handles communication with the backend API.
"""

import os
import json
import requests
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from functools import lru_cache
import logging
import sys
import socket

# Add project root to path to allow importing api_config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_service")

# Try to import shared configuration
try:
    from api_config import (
        API_BASE_URL, REQUEST_TIMEOUT, 
        get_working_api_url, get_alternative_hostname
    )
    
    # Use the configuration from api_config
    BASE_URL = get_working_api_url()  # Try to find a working URL
    logger.info(f"Using shared configuration: {BASE_URL}")
    has_config = True
except ImportError:
    # Check if we're running in Docker
    is_docker = os.path.exists('/.dockerenv')
    
    # Check for remote host in environment
    remote_host = os.environ.get('REMOTE_API_HOST')
    if remote_host:
        API_HOST = remote_host
        logger.info(f"Using remote host from environment: {API_HOST}")
    # Define API base URL using standard hostname and port
    elif is_docker:
        API_HOST = "app"  # Use the service name in Docker
    else:
        API_HOST = "localhost"  # Use localhost for non-Docker environments
        
    API_PORT = int(os.environ.get('API_PORT', 8000))  # Get port from environment or use 8000
    # No /api/v1 prefix
    API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
    BASE_URL = API_BASE_URL
    REQUEST_TIMEOUT = int(os.environ.get('API_TIMEOUT', 10))  # Timeout in seconds
    logger.info(f"Using default configuration: {API_BASE_URL}")
    has_config = False

    # Define our own fallback functions if api_config is not available
    def get_alternative_hostname(hostname):
        if hostname == "app":
            return "app"  # Always use the service name in Docker
        # Don't modify remote hosts
        if hostname == os.environ.get('REMOTE_API_HOST'):
            return hostname
        return '127.0.0.1' if hostname == 'localhost' else 'localhost'

# Cache configuration
CACHE_TTL = 300  # 5 minutes cache TTL

# Log current network details to help with troubleshooting
def log_network_details():
    """Log network details for troubleshooting remote connections"""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"Current machine hostname: {hostname}")
        logger.info(f"Current machine local IP: {local_ip}")
        
        # Log environment variables related to API connection
        for env_var in ['REMOTE_API_HOST', 'API_PORT', 'API_HOST']:
            value = os.environ.get(env_var)
            if value:
                logger.info(f"Environment variable {env_var}={value}")
    except Exception as e:
        logger.warning(f"Could not log network details: {str(e)}")

# Log network details at module load time to help with debugging
log_network_details()

def get_api_data(endpoint: str, params: Optional[Dict] = None, cache: bool = True) -> Any:
    """
    Get data from the API endpoint with enhanced error handling.
    
    Args:
        endpoint: API endpoint to call
        params: Optional query parameters
        cache: Whether to cache the result
        
    Returns:
        API response data or fallback data if API is unavailable
    """
    # Ensure endpoint doesn't start with a slash
    endpoint = endpoint.lstrip('/')
    url = f"{BASE_URL}/{endpoint}"
    logger.info(f"Making GET request to: {url}")
    
    # Use cached version if available and cache is enabled
    if cache:
        cache_key = f"{url}_{json.dumps(params) if params else ''}"
        cached_result = _get_cached_data(cache_key)
        if cached_result is not None:
            logger.info(f"Using cached data for: {url}")
            return cached_result
    
    # Try with the primary URL first
    result = _make_api_request(url, params)
    if result is not None:
        return result
    
    # Check if we're using a remote host
    remote_host = os.environ.get('REMOTE_API_HOST')
    if remote_host and remote_host in url:
        # If remote host is failing, log a warning and try localhost as a fallback
        logger.warning(f"Remote API host ({remote_host}) is unavailable. Check network connection and server status.")
        logger.warning("Possible remote connection issues:")
        logger.warning("1. Check if the API server is running on the remote host")
        logger.warning("2. Check firewall rules - make sure port 8000 is open")
        logger.warning("3. Verify the remote host is reachable via ping or telnet")
        logger.warning("4. Run ./test_api.sh script to diagnose connectivity issues")
        
        fallback_url = url.replace(remote_host, "localhost")
        logger.info(f"Trying localhost fallback: {fallback_url}")
        result = _make_api_request(fallback_url, params)
        if result is not None:
            return result
        
        fallback_url = url.replace(remote_host, "127.0.0.1")
        logger.info(f"Trying 127.0.0.1 fallback: {fallback_url}")
        result = _make_api_request(fallback_url, params)
        if result is not None:
            return result
    else:
        # If primary URL fails and it's using localhost, try with 127.0.0.1
        if "localhost" in BASE_URL:
            fallback_url = url.replace("localhost", "127.0.0.1")
            logger.info(f"Primary URL failed, trying fallback URL: {fallback_url}")
            result = _make_api_request(fallback_url, params)
            if result is not None:
                return result
        
        # If primary URL fails and it's using 127.0.0.1, try with localhost
        if "127.0.0.1" in BASE_URL:
            fallback_url = url.replace("127.0.0.1", "localhost")
            logger.info(f"Primary URL failed, trying fallback URL: {fallback_url}")
            result = _make_api_request(fallback_url, params)
            if result is not None:
                return result
            
    # If both URLs fail, return fallback data
    logger.error(f"All API request attempts failed for: {endpoint}")
    return _get_fallback_data(endpoint)

def _make_api_request(url: str, params: Optional[Dict] = None) -> Optional[Any]:
    """Make a request to the API with error handling."""
    try:
        # Add browser-like headers to avoid potential CORS issues
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Origin': url.rsplit('/', 1)[0],  # Set origin to base URL
            'Referer': url.rsplit('/', 1)[0]  # Set referrer to base URL
        }
        
        logger.info(f"Starting API request to {url}")
        start_time = time.time()
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        elapsed_time = time.time() - start_time
        logger.info(f"API request completed in {elapsed_time:.2f} seconds with status {response.status_code}")
        
        if response.status_code >= 500:
            logger.error(f"API server error: {url} returned {response.status_code}")
            logger.error(f"Response content: {response.text[:200]}")
            return None
            
        response.raise_for_status()
        
        # Try to parse the response as JSON
        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from API: {url}")
            logger.error(f"Response content: {response.text[:200]}")
            return None
        
        logger.info(f"API request successful: {url}")
        
        # Cache successful results
        cache_key = f"{url}_{json.dumps(params) if params else ''}"
        _cache_data(cache_key, data)
        
        return data
    except requests.exceptions.ConnectTimeout:
        logger.error(f"Connection timeout when connecting to {url}")
        logger.error(f"Possible causes: API server not running, network connectivity issues, firewall blocking")
        return None
    except requests.exceptions.ReadTimeout:
        logger.error(f"Read timeout when reading from {url}")
        logger.error(f"Possible causes: API server overloaded or unresponsive")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error to {url}: {str(e)}")
        logger.error(f"Possible causes: API server not running, network issues, incorrect hostname")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {url} - {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during API request: {url} - {str(e)}")
        return None

def post_api_data(endpoint: str, data: Dict, cache: bool = False) -> Any:
    """
    Post data to the API endpoint with enhanced error handling.
    
    Args:
        endpoint: API endpoint to call
        data: Data to post
        cache: Whether to cache the result
        
    Returns:
        API response data or fallback data if API is unavailable
    """
    url = f"{BASE_URL}/{endpoint}"
    
    try:
        response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        
        # Cache successful results if requested
        if cache:
            cache_key = f"{url}_{json.dumps(data)}"
            _cache_data(cache_key, result)
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"API POST request failed: {url} - {str(e)}")
        return _get_fallback_data(endpoint, is_post=True, post_data=data)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response from API: {url}")
        return _get_fallback_data(endpoint, is_post=True, post_data=data)
    except Exception as e:
        logger.error(f"Unexpected error during API POST request: {url} - {str(e)}")
        return _get_fallback_data(endpoint, is_post=True, post_data=data)

@lru_cache(maxsize=100)
def _get_cached_data(cache_key: str) -> Optional[Any]:
    """Get cached data if not expired."""
    # This is a simple in-memory cache with LRU policy
    # In a production environment, this could be replaced with Redis or another caching solution
    return None  # Currently, caching is handled by the lru_cache decorator

def _cache_data(cache_key: str, data: Any) -> None:
    """Cache data with expiration."""
    # This would be used to insert data into the cache
    # Currently handled by the lru_cache decorator
    pass

def _get_fallback_data(endpoint: str, is_post: bool = False, post_data: Optional[Dict] = None, 
                      server_error: bool = False, status_code: Optional[int] = None,
                      invalid_json: bool = False, exception: Optional[str] = None) -> Any:
    """
    Get fallback data for critical endpoints when the API is unavailable.
    
    Args:
        endpoint: API endpoint
        is_post: Whether this is a POST request fallback
        post_data: Data that was being posted (for context)
        server_error: Whether a server error occurred (5xx status code)
        status_code: The HTTP status code if there was an error
        invalid_json: Whether the response couldn't be parsed as JSON
        exception: Exception message if an exception occurred
        
    Returns:
        Fallback data
    """
    # Include error information in the fallback response
    error_info = {
        "offline_mode": True,
        "reason": "unknown"
    }
    
    if server_error:
        error_info["reason"] = "server_error"
        error_info["status_code"] = status_code
    elif invalid_json:
        error_info["reason"] = "invalid_json"
    elif exception:
        error_info["reason"] = "exception"
        error_info["message"] = exception
    
    # Base fallbacks for common endpoints
    fallbacks = {
        "health": {"status": "offline", "message": "API is currently unavailable", **error_info},
        "teams": [
            {"id": 1, "name": "Sample Team A", "country": "Country A"},
            {"id": 2, "name": "Sample Team B", "country": "Country B"},
        ],
        "matches": [
            {
                "id": 1, 
                "home_team": "Sample Team A", 
                "away_team": "Sample Team B",
                "date": "2025-01-01", 
                "competition": "Sample League", 
                "season": "2024/2025"
            }
        ],
        "predictions/models": [
            {"id": 1, "name": "Ensemble (Offline)", "type": "ensemble", "version": "1.0"},
            {"id": 2, "name": "Logistic Regression (Offline)", "type": "baseline", "version": "1.0"},
        ],
        "predictions/history": [],
        "command": {"status": "error", "message": "API unavailable for command execution"}
    }
    
    # Handle nested endpoints like team details or match details
    if endpoint.startswith("teams/") and len(endpoint.split("/")) == 2:
        team_id = int(endpoint.split("/")[1])
        return {"id": team_id, "name": f"Team {team_id} (Offline)", "country": "Unknown", **error_info}
    
    if endpoint.startswith("matches/") and len(endpoint.split("/")) == 2:
        match_id = int(endpoint.split("/")[1])
        return {
            "id": match_id,
            "home_team": "Home Team (Offline)",
            "away_team": "Away Team (Offline)",
            "date": "Unknown",
            "competition": "Unknown",
            "season": "Unknown",
            **error_info
        }
    
    if endpoint.startswith("predictions/match/"):
        return {
            "home_win": 0.33,
            "draw": 0.34,
            "away_win": 0.33,
            "status": "offline_fallback",
            "model": "Offline Fallback",
            **error_info
        }
    
    if endpoint == "predictions/custom" and is_post and post_data:
        return {
            "home_win": 0.33,
            "draw": 0.34,
            "away_win": 0.33,
            "home_team_id": post_data.get("home_team_id", 0),
            "away_team_id": post_data.get("away_team_id", 0),
            "status": "offline_fallback",
            "model": post_data.get("model", "Offline Fallback"),
            **error_info
        }
    
    # Return the fallback if available, otherwise a generic error
    fallback = fallbacks.get(endpoint, {"error": "API unavailable", "endpoint": endpoint})
    
    # If it's a list, we can't just add the error_info dict
    if isinstance(fallback, list):
        # For lists, return the fallback with an extra element indicating offline mode
        if len(fallback) > 0 and isinstance(fallback[0], dict):
            # Mark first item as offline
            fallback[0] = {**fallback[0], **error_info} 
        return fallback
    else:
        # For dictionaries, merge in the error info
        return {**fallback, **error_info}


class SoccerPredictionAPI:
    """Interface to the Soccer Prediction System API"""
    
    @staticmethod
    def check_health() -> Dict:
        """Check if the API is healthy"""
        # Try multiple endpoint patterns to improve reliability
        endpoints_to_try = [
            "/health",            # Direct health endpoint without prefix
            "/api/v1/health",     # Health endpoint with API v1 prefix
            "/api/health"         # Health endpoint with alternative prefix
        ]
        
        last_error = None
        
        # Try each endpoint in sequence
        for endpoint in endpoints_to_try:
            url = f"{BASE_URL}{endpoint}"
            logger.info(f"Checking API health at: {url}")
            
            try:
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                status_code = response.status_code
                logger.info(f"Health check status code: {status_code}")
                
                if status_code == 200:
                    try:
                        data = response.json()
                        logger.info(f"Health check response: {data}")
                        return {
                            "status": "online", 
                            "message": "API is available", 
                            "details": data,
                            "endpoint": endpoint
                        }
                    except json.JSONDecodeError:
                        logger.warning(f"Health check endpoint {endpoint} returned non-JSON response")
                        continue
                
                logger.warning(f"API returned non-200 status code for {endpoint}: {status_code}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error during health check for {endpoint}: {str(e)}")
                last_error = e
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout during health check for {endpoint}: {str(e)}")
                last_error = e
            except Exception as e:
                logger.warning(f"Unexpected error during health check for {endpoint}: {str(e)}")
                last_error = e
        
        # If we get here, all endpoints failed
        if last_error:
            logger.error(f"All health check endpoints failed. Last error: {str(last_error)}")
            return {"status": "offline", "message": f"Could not connect to API server: {str(last_error)}"}
        else:
            logger.error("All health check endpoints failed with non-200 responses")
            return {"status": "error", "message": "API server is not responding correctly"}
    
    @staticmethod
    def get_teams() -> List[Dict]:
        """Get list of teams"""
        return get_api_data("teams")
    
    @staticmethod
    def get_team_details(team_id: int) -> Dict:
        """Get details for a specific team"""
        return get_api_data(f"teams/{team_id}")
    
    @staticmethod
    def get_matches() -> List[Dict]:
        """Get list of matches"""
        return get_api_data("matches")
    
    @staticmethod
    def get_match_details(match_id: int) -> Dict:
        """Get details for a specific match"""
        return get_api_data(f"matches/{match_id}")
    
    @staticmethod
    def get_available_models() -> List[Dict]:
        """Get list of available prediction models"""
        return get_api_data("predictions/models")
    
    @staticmethod
    def predict_match(match_id: int, model_name: Optional[str] = None) -> Dict:
        """Get prediction for a specific match"""
        params = {}
        if model_name:
            params["model"] = model_name
        return get_api_data(f"predictions/match/{match_id}", params)
    
    @staticmethod
    def predict_custom_match(home_team_id: int, away_team_id: int, model_name: Optional[str] = None) -> Dict:
        """Make prediction for a custom match"""
        data = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id
        }
        if model_name:
            data["model"] = model_name
        return post_api_data("predictions/custom", data)
    
    @staticmethod
    def get_prediction_history() -> List[Dict]:
        """Get historical predictions"""
        return get_api_data("predictions/history")

# Command execution helpers
def run_command(command: str, **kwargs) -> Dict:
    """Run a command via the API"""
    data = {"command": command, **kwargs}
    return post_api_data("command", data)

def train_model(model_type: str, dataset: str = "all", feature_set: str = "all", **kwargs) -> Dict:
    """Train a model"""
    return run_command(
        "train",
        model_type=model_type,
        dataset=dataset,
        feature_set=feature_set,
        **kwargs
    )
    
def predict_with_model(model_type: str, home_team_id: int, away_team_id: int) -> Dict:
    """Make a prediction using a model"""
    return run_command(
        "predict",
        model_type=model_type,
        home_team_id=home_team_id,
        away_team_id=away_team_id
    )
    
def evaluate_model(model_path: str, **kwargs) -> Dict:
    """Evaluate a model"""
    return run_command(
        "evaluate",
        model_path=model_path,
        **kwargs
    )
    
def explain_model(model_path: str, methods: List[str], **kwargs) -> Dict:
    """Generate model explanations"""
    return run_command(
        "explain",
        model_path=model_path,
        methods=methods,
        **kwargs
    ) 