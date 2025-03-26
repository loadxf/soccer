"""
Football API Manager Module

This module provides functions for managing football data from football-data.co.uk.
It handles fetching, validation, and storage of football match data.
"""

import os
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import time
import logging
import json

# Import project components
from src.utils.logger import get_logger

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("data.football_api")

# Define data directory paths
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
FOOTBALL_DATA_DIR = os.path.join(RAW_DATA_DIR, "football_data")

# Define football-data.co.uk constants
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
AVAILABLE_LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "Conference",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "I1": "Serie A",
    "I2": "Serie B",
    "D1": "Bundesliga",
    "D2": "Bundesliga 2",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Belgian First Division",
    "P1": "Portuguese Liga",
    "T1": "Turkish Super Lig",
    "G1": "Greek Super League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two"
}

# Define available seasons (from 2010-2011 up to 2024-2025)
AVAILABLE_SEASONS = [f"{year}{year+1}" for year in range(2010, 2025)] + ["2425"]

class FootballDataAPI:
    """
    Client for interacting with football-data.co.uk API.
    Provides methods for fetching and processing football match data.
    """
    
    def __init__(self, base_url: str = FOOTBALL_DATA_BASE_URL):
        """
        Initialize the Football Data API client.
        
        Args:
            base_url: Base URL for the football-data.co.uk API
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.retry_count = 3
        self.retry_delay = 2  # seconds
        
        # Ensure storage directories exist
        os.makedirs(FOOTBALL_DATA_DIR, exist_ok=True)
    
    def get_url_for_season_league(self, season: str, league: str) -> str:
        """
        Construct URL for a specific season and league.
        
        Args:
            season: Season in format YYYYYYYY (e.g., "20232024")
            league: League code (e.g., "E0" for Premier League)
            
        Returns:
            str: Complete URL for the CSV data
        """
        # Special case for 2024-2025 season 
        if season == "2425" or season == "20242025":
            return f"{self.base_url}/2425/{league}.csv"
        return f"{self.base_url}/{season}/{league}.csv"
    
    def download_data(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """
        Download data from a URL with retry logic.
        
        Args:
            url: URL to download data from
            max_retries: Maximum number of retry attempts
            
        Returns:
            Optional[bytes]: Downloaded data or None if failed
        """
        retries = 0
        
        while retries < max_retries:
            try:
                logger.info(f"Downloading data from {url}")
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 200:
                    return response.content
                elif response.status_code == 404:
                    logger.warning(f"Resource not found (404): {url}")
                    return None
                else:
                    logger.warning(f"Failed to download data: HTTP {response.status_code}")
            
            except requests.RequestException as e:
                logger.error(f"Request error: {e}")
            
            retries += 1
            if retries < max_retries:
                wait_time = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds... (Attempt {retries+1}/{max_retries})")
                time.sleep(wait_time)
        
        logger.error(f"Failed to download data after {max_retries} attempts: {url}")
        return None
    
    def fetch_season_data(self, season: str, leagues: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch data for a specific season and set of leagues.
        
        Args:
            season: Season in format YYYYYYYY (e.g., "20232024")
            leagues: List of league codes to fetch (if None, fetch all available leagues)
            
        Returns:
            Dict[str, Any]: Dictionary with downloaded data information
        """
        if season not in AVAILABLE_SEASONS:
            logger.warning(f"Unsupported season: {season}")
            return {"status": "error", "message": f"Unsupported season: {season}", "files": []}
        
        if leagues is None:
            leagues = list(AVAILABLE_LEAGUES.keys())
        
        # Create season directory
        season_dir = os.path.join(FOOTBALL_DATA_DIR, season)
        os.makedirs(season_dir, exist_ok=True)
        
        results = {
            "status": "success",
            "season": season,
            "files": [],
            "errors": []
        }
        
        for league in leagues:
            if league not in AVAILABLE_LEAGUES:
                logger.warning(f"Unknown league code: {league}")
                results["errors"].append(f"Unknown league code: {league}")
                continue
            
            url = self.get_url_for_season_league(season, league)
            data = self.download_data(url)
            
            if data:
                # Save the file
                filename = f"{league}.csv"
                file_path = os.path.join(season_dir, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Validate data
                if self._validate_csv_data(file_path):
                    results["files"].append({
                        "league": league,
                        "league_name": AVAILABLE_LEAGUES.get(league, "Unknown"),
                        "filename": filename,
                        "path": file_path,
                        "size": len(data),
                        "download_date": datetime.now().isoformat()
                    })
                else:
                    logger.warning(f"Downloaded file failed validation: {file_path}")
                    results["errors"].append(f"Data validation failed for {league}")
            else:
                results["errors"].append(f"Failed to download {league}")
        
        # Update status if there were any errors
        if results["errors"] and not results["files"]:
            results["status"] = "error"
        elif results["errors"]:
            results["status"] = "partial"
        
        return results
    
    def _validate_csv_data(self, file_path: str) -> bool:
        """
        Validate downloaded CSV data.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Try to read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if file is empty
            if df.empty:
                logger.warning(f"Empty CSV file: {file_path}")
                return False
            
            # Check for minimum expected columns
            expected_columns = ['Date', 'HomeTeam', 'AwayTeam']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing expected columns in {file_path}: {missing_columns}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating CSV data: {e}")
            return False
    
    def fetch_all_seasons(self, seasons: Optional[List[str]] = None, leagues: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch data for multiple seasons and leagues.
        
        Args:
            seasons: List of seasons to fetch (if None, fetch all available seasons)
            leagues: List of league codes to fetch (if None, fetch all available leagues)
            
        Returns:
            Dict[str, Any]: Dictionary with downloaded data information
        """
        if seasons is None:
            seasons = AVAILABLE_SEASONS
        
        results = {
            "status": "success",
            "seasons": [],
            "total_files": 0,
            "errors": []
        }
        
        for season in seasons:
            season_result = self.fetch_season_data(season, leagues)
            results["seasons"].append({
                "season": season,
                "status": season_result["status"],
                "files_count": len(season_result["files"])
            })
            
            results["total_files"] += len(season_result["files"])
            if "errors" in season_result and season_result["errors"]:
                results["errors"].extend([f"{season}/{err}" for err in season_result["errors"]])
        
        # Update overall status
        if results["errors"] and results["total_files"] == 0:
            results["status"] = "error"
        elif results["errors"]:
            results["status"] = "partial"
        
        return results
    
    def get_upcoming_fixtures(self) -> pd.DataFrame:
        """
        Get upcoming fixtures from the latest available data.
        
        Returns:
            pd.DataFrame: DataFrame containing upcoming fixtures
        """
        # Find most recent season data
        recent_seasons = sorted(AVAILABLE_SEASONS, reverse=True)
        
        for season in recent_seasons:
            season_dir = os.path.join(FOOTBALL_DATA_DIR, season)
            if os.path.exists(season_dir):
                # Get all CSV files in the season directory
                csv_files = [f for f in os.listdir(season_dir) if f.endswith('.csv')]
                
                if csv_files:
                    # Combine all leagues for this season
                    dfs = []
                    
                    for csv_file in csv_files:
                        file_path = os.path.join(season_dir, csv_file)
                        try:
                            df = pd.read_csv(file_path)
                            
                            # Add league information
                            league_code = csv_file.replace('.csv', '')
                            df['League'] = league_code
                            df['LeagueName'] = AVAILABLE_LEAGUES.get(league_code, "Unknown")
                            
                            # Convert date if needed
                            if 'Date' in df.columns:
                                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                            
                            # Filter to only include upcoming matches (no result yet)
                            if 'FTHG' in df.columns and 'FTAG' in df.columns:
                                df = df[(df['FTHG'].isna()) | (df['FTAG'].isna())]
                            
                            dfs.append(df)
                        except Exception as e:
                            logger.error(f"Error reading fixtures from {file_path}: {e}")
                    
                    if dfs:
                        # Combine all leagues
                        fixtures_df = pd.concat(dfs, ignore_index=True)
                        
                        # Sort by date
                        if 'Date' in fixtures_df.columns:
                            fixtures_df = fixtures_df.sort_values('Date')
                        
                        return fixtures_df
        
        # If no data found, return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam', 'League', 'LeagueName'])
    
    def get_team_data(self, team_name: str, recent_seasons: int = 2) -> pd.DataFrame:
        """
        Get historical data for a specific team from recent seasons.
        
        Args:
            team_name: Name of the team
            recent_seasons: Number of recent seasons to include
            
        Returns:
            pd.DataFrame: DataFrame containing team's historical data
        """
        recent_season_list = sorted(AVAILABLE_SEASONS, reverse=True)[:recent_seasons]
        
        dfs = []
        for season in recent_season_list:
            season_dir = os.path.join(FOOTBALL_DATA_DIR, season)
            if os.path.exists(season_dir):
                for csv_file in os.listdir(season_dir):
                    if csv_file.endswith('.csv'):
                        file_path = os.path.join(season_dir, csv_file)
                        try:
                            df = pd.read_csv(file_path)
                            
                            # Add league and season information
                            league_code = csv_file.replace('.csv', '')
                            df['League'] = league_code
                            df['LeagueName'] = AVAILABLE_LEAGUES.get(league_code, "Unknown")
                            df['Season'] = season
                            
                            # Filter for the team
                            team_df = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
                            
                            if not team_df.empty:
                                dfs.append(team_df)
                        except Exception as e:
                            logger.error(f"Error getting team data from {file_path}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Convert date
            if 'Date' in combined_df.columns:
                combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
                combined_df = combined_df.sort_values('Date')
            
            return combined_df
        
        # Return empty DataFrame if no data found
        return pd.DataFrame()


# Entry point for CLI usage
if __name__ == "__main__":
    if len(sys.argv) > 1:
        api = FootballDataAPI()
        
        if sys.argv[1] == "download":
            # Download data for specific season
            if len(sys.argv) > 2:
                season = sys.argv[2]
                leagues = sys.argv[3].split(",") if len(sys.argv) > 3 else None
                
                print(f"Downloading data for season {season}...")
                result = api.fetch_season_data(season, leagues)
                
                print(f"Status: {result['status']}")
                print(f"Files downloaded: {len(result['files'])}")
                
                if result['errors']:
                    print("Errors:")
                    for error in result['errors']:
                        print(f"  - {error}")
            else:
                print("Please specify a season (e.g., '20232024')")
        
        elif sys.argv[1] == "download_all":
            # Download all seasons
            print("Downloading data for all available seasons...")
            result = api.fetch_all_seasons()
            
            print(f"Status: {result['status']}")
            print(f"Total files downloaded: {result['total_files']}")
            
            if result['errors']:
                print(f"Errors: {len(result['errors'])}")
        
        elif sys.argv[1] == "upcoming":
            # Get upcoming fixtures
            print("Getting upcoming fixtures...")
            fixtures = api.get_upcoming_fixtures()
            
            if fixtures.empty:
                print("No upcoming fixtures found.")
            else:
                print(f"Found {len(fixtures)} upcoming fixtures.")
                print(fixtures.head())
        
        elif sys.argv[1] == "team":
            # Get team data
            if len(sys.argv) > 2:
                team_name = sys.argv[2]
                print(f"Getting data for {team_name}...")
                
                team_data = api.get_team_data(team_name)
                
                if team_data.empty:
                    print(f"No data found for {team_name}.")
                else:
                    print(f"Found {len(team_data)} matches for {team_name}.")
                    print(team_data.head())
            else:
                print("Please specify a team name.")
        
        else:
            print("Unknown command. Available commands: download, download_all, upcoming, team")
    else:
        print("Available commands: download, download_all, upcoming, team") 