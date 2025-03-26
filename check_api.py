"""
Simple script to check if the API server is running.
"""

import requests
import sys

def check_api():
    """Check if the API server is running and accessible."""
    url = "http://127.0.0.1:8080/api/v1/health"
    
    try:
        response = requests.get(url, timeout=5)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(response.json())
        return True
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to {url}")
        print("The API server might not be running.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Checking API status...")
    if check_api():
        print("API is running!")
    else:
        print("API check failed.")
        sys.exit(1) 