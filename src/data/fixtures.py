"""
Fixtures Module

This module provides functions for fetching and managing upcoming match fixtures.
It interfaces with the football-data.co.uk API and provides data to the prediction system.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import json

# Import project components
from src.utils.logger import get_logger
from src.data.football_api_manager import FootballDataAPI

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("data.fixtures")

# Define fixtures directory path
FIXTURES_DIR = os.path.join(DATA_DIR, "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

class FixtureManager:
    """
    Manager for upcoming match fixtures.
    Provides methods for fetching, storing, and retrieving fixture data.
    """
    
    def __init__(self, cache_file: str = "fixtures_cache.json"):
        """
        Initialize the fixture manager.
        
        Args:
            cache_file: Filename for the fixtures cache
        """
        self.football_api = FootballDataAPI()
        self.cache_file = os.path.join(FIXTURES_DIR, cache_file)
        self.fixtures = None
        self.last_updated = None
        
        # Load cached fixtures if available
        self._load_cache()
    
    def _load_cache(self):
        """Load fixtures from cache file if it exists and is recent."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                if "last_updated" in cache_data:
                    last_updated = datetime.fromisoformat(cache_data["last_updated"])
                    age = datetime.now() - last_updated
                    
                    # Use cache if less than 24 hours old
                    if age.total_seconds() < 86400:  # 24 hours in seconds
                        self.last_updated = last_updated
                        
                        if "fixtures" in cache_data:
                            # Convert JSON fixtures to DataFrame
                            self.fixtures = pd.DataFrame(cache_data["fixtures"])
                            
                            # Convert date columns back to datetime
                            if "Date" in self.fixtures.columns:
                                self.fixtures["Date"] = pd.to_datetime(self.fixtures["Date"])
                            
                            logger.info(f"Loaded {len(self.fixtures)} fixtures from cache (updated {age.total_seconds()/3600:.1f} hours ago)")
                    else:
                        logger.info(f"Cache is {age.total_seconds()/3600:.1f} hours old, will refresh fixtures")
            except Exception as e:
                logger.error(f"Error loading fixtures cache: {e}")
    
    def _save_cache(self):
        """Save current fixtures to cache file."""
        if self.fixtures is None:
            return
        
        try:
            # Convert DataFrame to dict for JSON serialization
            fixtures_list = self.fixtures.to_dict(orient="records")
            
            cache_data = {
                "last_updated": datetime.now().isoformat(),
                "fixtures": fixtures_list
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved {len(self.fixtures)} fixtures to cache")
        except Exception as e:
            logger.error(f"Error saving fixtures to cache: {e}")
    
    def refresh_fixtures(self, force: bool = False) -> pd.DataFrame:
        """
        Refresh fixtures from the football data API.
        
        Args:
            force: Force refresh even if cache is recent
            
        Returns:
            pd.DataFrame: DataFrame of upcoming fixtures
        """
        # Check if we need to refresh
        if not force and self.fixtures is not None and self.last_updated is not None:
            age = datetime.now() - self.last_updated
            if age.total_seconds() < 86400:  # 24 hours in seconds
                logger.info(f"Using cached fixtures (updated {age.total_seconds()/3600:.1f} hours ago)")
                return self.fixtures
        
        # Fetch upcoming fixtures from the football API
        try:
            logger.info("Fetching upcoming fixtures from football-data.co.uk")
            fixtures_df = self.football_api.get_upcoming_fixtures()
            
            if fixtures_df.empty:
                logger.warning("No upcoming fixtures found from API")
                return pd.DataFrame()
            
            # Store the fixtures
            self.fixtures = fixtures_df
            self.last_updated = datetime.now()
            
            # Save to cache
            self._save_cache()
            
            return fixtures_df
        except Exception as e:
            logger.error(f"Error refreshing fixtures: {e}")
            return pd.DataFrame()
    
    def get_fixtures(self, days_ahead: int = 30, team: Optional[str] = None, 
                    league: Optional[str] = None) -> pd.DataFrame:
        """
        Get upcoming fixtures with optional filtering.
        
        Args:
            days_ahead: Number of days ahead to include
            team: Filter by team name
            league: Filter by league code
            
        Returns:
            pd.DataFrame: Filtered DataFrame of upcoming fixtures
        """
        # Refresh fixtures if needed
        if self.fixtures is None:
            self.refresh_fixtures()
        
        if self.fixtures is None or self.fixtures.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        fixtures = self.fixtures.copy()
        
        # Filter by date
        if "Date" in fixtures.columns:
            end_date = datetime.now() + timedelta(days=days_ahead)
            fixtures = fixtures[fixtures["Date"] <= end_date]
        
        # Filter by team if specified
        if team:
            fixtures = fixtures[(fixtures["HomeTeam"] == team) | (fixtures["AwayTeam"] == team)]
        
        # Filter by league if specified
        if league:
            fixtures = fixtures[fixtures["League"] == league]
        
        return fixtures
    
    def get_next_fixture(self, team: str) -> Optional[Dict[str, Any]]:
        """
        Get the next fixture for a specific team.
        
        Args:
            team: Team name
            
        Returns:
            Optional[Dict[str, Any]]: Next fixture or None if no fixtures found
        """
        # Get fixtures for the team
        team_fixtures = self.get_fixtures(team=team)
        
        if team_fixtures.empty:
            return None
        
        # Sort by date and get the first one
        if "Date" in team_fixtures.columns:
            team_fixtures = team_fixtures.sort_values("Date")
        
        next_match = team_fixtures.iloc[0].to_dict()
        
        # Add a flag to indicate whether the team is home or away
        next_match["is_home"] = next_match["HomeTeam"] == team
        
        return next_match
    
    def get_team_fixtures(self, team: str, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get upcoming fixtures for a specific team.
        
        Args:
            team: Team name
            count: Maximum number of fixtures to return
            
        Returns:
            List[Dict[str, Any]]: List of upcoming fixtures
        """
        # Get fixtures for the team
        team_fixtures = self.get_fixtures(team=team)
        
        if team_fixtures.empty:
            return []
        
        # Sort by date
        if "Date" in team_fixtures.columns:
            team_fixtures = team_fixtures.sort_values("Date")
        
        # Convert to list of dicts and add home/away flag
        fixtures_list = []
        for _, row in team_fixtures.head(count).iterrows():
            fixture = row.to_dict()
            fixture["is_home"] = fixture["HomeTeam"] == team
            fixtures_list.append(fixture)
        
        return fixtures_list
    
    def get_fixtures_by_date(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get fixtures within a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD) or None for today
            end_date: End date (YYYY-MM-DD) or None for 7 days after start
            
        Returns:
            pd.DataFrame: Fixtures within the date range
        """
        # Refresh fixtures if needed
        if self.fixtures is None:
            self.refresh_fixtures()
        
        if self.fixtures is None or self.fixtures.empty:
            return pd.DataFrame()
        
        # Parse dates
        if start_date is None:
            start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_dt = datetime.fromisoformat(start_date)
        
        if end_date is None:
            end_dt = start_dt + timedelta(days=7)
        else:
            end_dt = datetime.fromisoformat(end_date)
        
        # Filter fixtures by date range
        fixtures = self.fixtures.copy()
        
        if "Date" in fixtures.columns:
            fixtures = fixtures[(fixtures["Date"] >= start_dt) & (fixtures["Date"] <= end_dt)]
            fixtures = fixtures.sort_values("Date")
        
        return fixtures
    
    def prepare_fixture_features(self, home_team: str, away_team: str, 
                                match_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare feature data for a specific fixture for prediction.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date (YYYY-MM-DD) or None for current date
            
        Returns:
            Dict[str, Any]: Feature dictionary for the fixture
        """
        logger.info(f"Preparing features for {home_team} vs {away_team}")
        
        # Get team data for both teams
        home_data = self.football_api.get_team_data(home_team)
        away_data = self.football_api.get_team_data(away_team)
        
        if home_data.empty or away_data.empty:
            logger.warning("Insufficient historical data for teams")
            return {}
        
        # Parse match date
        if match_date:
            match_dt = datetime.fromisoformat(match_date)
        else:
            match_dt = datetime.now()
        
        # Calculate basic stats
        features = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_dt.isoformat(),
            "features": {}
        }
        
        # Recent form (last 5 matches)
        home_form = self._calculate_form(home_data, home_team)
        away_form = self._calculate_form(away_data, away_team)
        
        features["features"]["home_form"] = home_form
        features["features"]["away_form"] = away_form
        
        # Head-to-head
        h2h_matches = pd.concat([
            home_data[(home_data["HomeTeam"] == home_team) & (home_data["AwayTeam"] == away_team)],
            home_data[(home_data["HomeTeam"] == away_team) & (home_data["AwayTeam"] == home_team)]
        ])
        
        if not h2h_matches.empty:
            h2h_matches = h2h_matches.sort_values("Date", ascending=False)
            features["features"]["h2h_matches"] = len(h2h_matches)
            
            # Get results of last H2H matches
            h2h_results = []
            for _, match in h2h_matches.head(5).iterrows():
                if "FTHG" in match and "FTAG" in match:
                    if match["HomeTeam"] == home_team:
                        if match["FTHG"] > match["FTAG"]:
                            h2h_results.append("H")
                        elif match["FTHG"] < match["FTAG"]:
                            h2h_results.append("A")
                        else:
                            h2h_results.append("D")
                    else:
                        if match["FTHG"] < match["FTAG"]:
                            h2h_results.append("H")
                        elif match["FTHG"] > match["FTAG"]:
                            h2h_results.append("A")
                        else:
                            h2h_results.append("D")
            
            features["features"]["h2h_results"] = h2h_results
        else:
            features["features"]["h2h_matches"] = 0
            features["features"]["h2h_results"] = []
        
        # Average goals
        if "FTHG" in home_data.columns and "FTAG" in home_data.columns:
            # Home team scoring at home
            home_scoring = home_data[home_data["HomeTeam"] == home_team]["FTHG"].mean()
            # Away team scoring away
            away_scoring = away_data[away_data["AwayTeam"] == away_team]["FTAG"].mean()
            
            features["features"]["home_avg_goals_for"] = home_scoring if not np.isnan(home_scoring) else 0
            features["features"]["away_avg_goals_for"] = away_scoring if not np.isnan(away_scoring) else 0
            
            # Home team conceding at home
            home_conceding = home_data[home_data["HomeTeam"] == home_team]["FTAG"].mean()
            # Away team conceding away
            away_conceding = away_data[away_data["AwayTeam"] == away_team]["FTHG"].mean()
            
            features["features"]["home_avg_goals_against"] = home_conceding if not np.isnan(home_conceding) else 0
            features["features"]["away_avg_goals_against"] = away_conceding if not np.isnan(away_conceding) else 0
        
        return features
    
    def _calculate_form(self, team_data: pd.DataFrame, team_name: str) -> List[str]:
        """
        Calculate recent form for a team based on last 5 matches.
        
        Args:
            team_data: DataFrame with team's match data
            team_name: Team name
            
        Returns:
            List[str]: List of results (W/D/L)
        """
        if team_data.empty:
            return []
        
        # Sort matches by date (most recent first)
        recent_matches = team_data.sort_values("Date", ascending=False)
        
        # Calculate results
        results = []
        for _, match in recent_matches.head(5).iterrows():
            if "FTHG" not in match or "FTAG" not in match:
                continue
                
            if match["HomeTeam"] == team_name:
                if match["FTHG"] > match["FTAG"]:
                    results.append("W")
                elif match["FTHG"] < match["FTAG"]:
                    results.append("L")
                else:
                    results.append("D")
            else:  # Away team
                if match["FTHG"] < match["FTAG"]:
                    results.append("W")
                elif match["FTHG"] > match["FTAG"]:
                    results.append("L")
                else:
                    results.append("D")
        
        return results


# Singleton instance
fixture_manager = FixtureManager()

def get_fixture_manager() -> FixtureManager:
    """
    Get the global fixture manager instance.
    
    Returns:
        FixtureManager: The global fixture manager
    """
    return fixture_manager

def get_upcoming_fixtures(days_ahead: int = 30, team: Optional[str] = None, 
                        league: Optional[str] = None) -> pd.DataFrame:
    """
    Get upcoming fixtures with optional filtering.
    
    Args:
        days_ahead: Number of days ahead to include
        team: Filter by team name
        league: Filter by league code
        
    Returns:
        pd.DataFrame: Filtered DataFrame of upcoming fixtures
    """
    return fixture_manager.get_fixtures(days_ahead, team, league)

def prepare_match_features(home_team: str, away_team: str, 
                         match_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Prepare feature data for a specific match.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Match date (YYYY-MM-DD) or None for current date
        
    Returns:
        Dict[str, Any]: Feature dictionary for the match
    """
    return fixture_manager.prepare_fixture_features(home_team, away_team, match_date)

def refresh_fixture_data(force: bool = False) -> int:
    """
    Refresh fixture data from the API.
    
    Args:
        force: Force refresh even if cache is recent
        
    Returns:
        int: Number of fixtures loaded
    """
    fixtures = fixture_manager.refresh_fixtures(force)
    return len(fixtures) if fixtures is not None else 0 