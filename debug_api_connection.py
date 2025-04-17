import requests
import sys
import os
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

# Import the API service module
try:
    from ui.api_service import SoccerPredictionAPI, API_BASE_URL, REQUEST_TIMEOUT
    
    print(f"API Configuration:")
    print(f"API_BASE_URL: {API_BASE_URL}")
    print(f"REQUEST_TIMEOUT: {REQUEST_TIMEOUT}")
    print()
    
    # Try direct health check using requests with both endpoint patterns
    print("Testing direct API connection with different endpoints:")
    
    # Test the /health endpoint
    try:
        direct_url = "http://127.0.0.1:8000/health"
        response = requests.get(direct_url, timeout=5)
        print(f"Direct connection to {direct_url}: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Direct connection to /health failed: {e}")
    
    # Test the /api/v1/health endpoint
    try:
        api_v1_url = "http://127.0.0.1:8000/api/v1/health"
        response = requests.get(api_v1_url, timeout=5)
        print(f"Direct connection to {api_v1_url}: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Direct connection to /api/v1/health failed: {e}")
    print()
    
    # Try health check using the API service
    print("Testing API service health check:")
    try:
        api_health = SoccerPredictionAPI.check_health()
        print(f"API health check result: {api_health}")
    except Exception as e:
        print(f"API health check failed: {e}")
    print()
    
    # Try to get teams using the API service
    print("Testing API service get_teams:")
    try:
        teams = SoccerPredictionAPI.get_teams()
        print(f"Got {len(teams)} teams")
        print(f"First team: {teams[0]}")
    except Exception as e:
        print(f"API get_teams failed: {e}")
        
except ImportError as e:
    print(f"Failed to import API service: {e}")
    sys.exit(1) 