"""
Test script for downloading 2024-2025 season data.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from src.data.pipeline import download_football_data
from src.data.football_api_manager import FootballDataAPI

def test_download_2425_season():
    """Test downloading the 2024-2025 Premier League data."""
    print("Testing direct API URL generation...")
    api = FootballDataAPI()
    url = api.get_url_for_season_league("2425", "E0")
    print(f"URL: {url}")
    
    # Test if the URL is accessible
    import requests
    response = requests.head(url)
    print(f"URL accessible: {response.status_code == 200}")
    
    print("\nTesting pipeline download function...")
    result = download_football_data('football_data', custom_seasons=['2425'], custom_leagues=['E0'])
    print(f"Download successful: {result}")
    
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("Testing 2024-2025 season download")
    print("=" * 60)
    success = test_download_2425_season()
    print("\nTest result:", "✅ PASSED" if success else "❌ FAILED") 