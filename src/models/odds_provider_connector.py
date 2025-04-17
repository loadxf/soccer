"""
Odds provider connector for fetching betting odds from various sources.

This module provides a standardized interface for fetching odds from 
different bookmakers and odds providers. It handles the connection, 
data retrieval, and parsing of odds data into a consistent format.
"""

from typing import Dict, List, Union, Optional, Any, Callable
import pandas as pd
import numpy as np
import logging
import requests
import json
from abc import ABC, abstractmethod
import time
from datetime import datetime, timedelta
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class OddsProviderConnector(ABC):
    """
    Abstract base class for odds provider connectors.
    
    This class defines the interface that all odds provider connectors
    must implement to fetch and process betting odds from different sources.
    """
    
    @abstractmethod
    def fetch_odds(self, competition_id: str, start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch odds for a specific competition/league.
        
        Args:
            competition_id: ID of the competition to fetch odds for
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            DataFrame containing odds data
        """
        pass
    
    @abstractmethod
    def fetch_match_odds(self, match_id: str) -> pd.DataFrame:
        """
        Fetch odds for a specific match.
        
        Args:
            match_id: Unique identifier for the match
            
        Returns:
            DataFrame containing odds data for the match
        """
        pass
    
    @abstractmethod
    def get_available_bookmakers(self) -> List[str]:
        """
        Get a list of available bookmakers.
        
        Returns:
            List of bookmaker IDs/names
        """
        pass
    
    @abstractmethod
    def get_available_markets(self) -> Dict[str, str]:
        """
        Get available betting markets.
        
        Returns:
            Dictionary mapping market IDs to market names
        """
        pass
    
    @abstractmethod
    def get_available_competitions(self) -> Dict[str, str]:
        """
        Get available competitions/leagues.
        
        Returns:
            Dictionary mapping competition IDs to competition names
        """
        pass


class APIConnector(OddsProviderConnector):
    """
    Base class for API-based odds providers.
    
    This class implements common functionality for connectors
    that fetch odds data from REST APIs.
    """
    
    def __init__(self, api_key: str, base_url: str, request_timeout: int = 30,
                 request_delay: float = 1.0, max_retries: int = 3):
        """
        Initialize the API connector.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL for API requests
            request_timeout: Timeout in seconds for API requests
            request_delay: Delay between API requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make an API request with retries.
        
        Args:
            endpoint: API endpoint path
            params: Optional query parameters
            headers: Optional headers
            
        Returns:
            JSON response data
            
        Raises:
            RuntimeError: If the request fails after all retries
        """
        if headers is None:
            headers = {}
        
        if params is None:
            params = {}
        
        # Add API key to params or headers depending on the API design
        params['api_key'] = self.api_key
        
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.request_timeout
                )
                
                # Introduce delay to respect API rate limits
                time.sleep(self.request_delay)
                
                response.raise_for_status()  # Raise for 4xx/5xx responses
                
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to make request to {url} after {self.max_retries} attempts")
                    raise RuntimeError(f"API request failed: {str(e)}")
                
                # Exponential backoff
                time.sleep((2 ** attempt) * self.request_delay)
    
    def _transform_response(self, response_data: Dict[str, Any], 
                          transform_function: Callable) -> pd.DataFrame:
        """
        Transform API response data into a DataFrame.
        
        Args:
            response_data: Raw API response data
            transform_function: Function to transform the data
            
        Returns:
            Transformed DataFrame
        """
        try:
            return transform_function(response_data)
        except Exception as e:
            logger.error(f"Error transforming response data: {str(e)}")
            raise


class OddsAPIConnector(APIConnector):
    """
    Connector for the Odds API service (odds-api.com).
    
    This connector fetches odds from the Odds API service, which provides
    access to odds from multiple bookmakers.
    """
    
    def __init__(self, api_key: str, request_timeout: int = 30,
                 request_delay: float = 1.0, max_retries: int = 3,
                 cache_dir: Optional[Union[str, Path]] = None,
                 cache_duration: int = 3600):
        """
        Initialize the Odds API connector.
        
        Args:
            api_key: API key for authentication
            request_timeout: Timeout in seconds for API requests
            request_delay: Delay between API requests in seconds
            max_retries: Maximum number of retries for failed requests
            cache_dir: Optional directory to cache API responses
            cache_duration: Cache duration in seconds (default: 1 hour)
        """
        super().__init__(
            api_key=api_key,
            base_url="https://api.the-odds-api.com/v4",
            request_timeout=request_timeout,
            request_delay=request_delay,
            max_retries=max_retries
        )
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_duration = cache_duration
        
        # Create cache directory if needed
        if self.cache_dir and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize bookmakers and sports data
        self._bookmakers = None
        self._markets = None
        self._sports = None
    
    def _get_cache_path(self, key: str) -> Optional[Path]:
        """
        Get the cache file path for a given key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file or None if caching is disabled
        """
        if not self.cache_dir:
            return None
        
        return self.cache_dir / f"{key}.json"
    
    def _read_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Read data from cache if it exists and is still valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not available
        """
        cache_path = self._get_cache_path(key)
        if not cache_path or not cache_path.exists():
            return None
        
        current_time = time.time()
        file_modified_time = os.path.getmtime(cache_path)
        
        # Check if cache is still valid
        if current_time - file_modified_time > self.cache_duration:
            logger.debug(f"Cache expired for {key}")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading cache for {key}: {str(e)}")
            return None
    
    def _write_to_cache(self, key: str, data: Dict[str, Any]) -> None:
        """
        Write data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        cache_path = self._get_cache_path(key)
        if not cache_path:
            return
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing cache for {key}: {str(e)}")
    
    def _fetch_with_cache(self, endpoint: str, params: Optional[Dict[str, Any]] = None,
                        cache_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch data with caching.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            cache_key: Custom cache key, will be generated if not provided
            
        Returns:
            API response data
        """
        if params is None:
            params = {}
        
        # Generate cache key if not provided
        if cache_key is None:
            param_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            cache_key = f"{endpoint}_{param_str}"
        
        # Check cache first
        cached_data = self._read_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Using cached data for {cache_key}")
            return cached_data
        
        # If not in cache, fetch from API
        response_data = self._make_request(endpoint, params)
        
        # Cache the response
        self._write_to_cache(cache_key, response_data)
        
        return response_data
    
    def get_available_bookmakers(self) -> List[str]:
        """
        Get a list of available bookmakers.
        
        Returns:
            List of bookmaker IDs
        """
        if self._bookmakers is not None:
            return self._bookmakers
        
        # First, get any odds data to extract bookmakers
        sports = self.get_available_competitions()
        if not sports:
            return []
        
        sport_key = list(sports.keys())[0]
        
        try:
            odds_data = self._fetch_with_cache(
                endpoint="sports/" + sport_key + "/odds",
                params={"regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
                cache_key="bookmakers_list"
            )
            
            # Extract unique bookmaker IDs
            bookmakers = set()
            for event in odds_data:
                if 'bookmakers' in event:
                    for bookmaker in event['bookmakers']:
                        if 'key' in bookmaker:
                            bookmakers.add(bookmaker['key'])
            
            self._bookmakers = list(bookmakers)
            return self._bookmakers
            
        except Exception as e:
            logger.error(f"Failed to fetch bookmakers: {str(e)}")
            return []
    
    def get_available_markets(self) -> Dict[str, str]:
        """
        Get available betting markets.
        
        Returns:
            Dictionary mapping market IDs to market names
        """
        if self._markets is not None:
            return self._markets
        
        # Odds API markets are predefined
        self._markets = {
            "h2h": "Moneyline/1X2",
            "spreads": "Point Spread/Handicap",
            "totals": "Totals/Over-Under",
            "outrights": "Outrights/Futures",
            "h2h_lay": "Moneyline/1X2 (Lay)",
            "alternate_spreads": "Alternate Spreads",
            "alternate_totals": "Alternate Totals",
            "btts": "Both Teams to Score",
            "draw_no_bet": "Draw No Bet",
            "asian_handicap": "Asian Handicap"
        }
        
        return self._markets
    
    def get_available_competitions(self) -> Dict[str, str]:
        """
        Get available competitions/leagues.
        
        Returns:
            Dictionary mapping competition IDs to competition names
        """
        if self._sports is not None:
            return self._sports
        
        try:
            response_data = self._fetch_with_cache(
                endpoint="sports",
                cache_key="sports_list"
            )
            
            sports = {}
            for sport in response_data:
                if 'key' in sport and 'title' in sport:
                    sports[sport['key']] = sport['title']
            
            self._sports = sports
            return sports
            
        except Exception as e:
            logger.error(f"Failed to fetch sports: {str(e)}")
            return {}
    
    def fetch_odds(self, competition_id: str, start_date: Optional[str] = None, 
                 end_date: Optional[str] = None, markets: Optional[List[str]] = None,
                 bookmakers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch odds for a specific competition/league.
        
        Args:
            competition_id: ID of the competition to fetch odds for
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            markets: List of markets to fetch (default: all)
            bookmakers: List of bookmakers to include (default: all)
            
        Returns:
            DataFrame containing odds data
        """
        if markets is None:
            # Default to main markets
            markets = ["h2h", "spreads", "totals"]
        
        params = {
            "regions": "us,uk,eu",  # Include major regions
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }
        
        all_odds_data = []
        
        # Fetch each market separately
        for market in markets:
            try:
                params["markets"] = market
                
                response_data = self._fetch_with_cache(
                    endpoint=f"sports/{competition_id}/odds",
                    params=params,
                    cache_key=f"odds_{competition_id}_{market}_{start_date}_{end_date}"
                )
                
                # Transform response to DataFrame
                odds_df = self._transform_odds_response(response_data, market, bookmakers)
                
                if not odds_df.empty:
                    all_odds_data.append(odds_df)
                
            except Exception as e:
                logger.error(f"Error fetching odds for {competition_id} ({market}): {str(e)}")
        
        if not all_odds_data:
            return pd.DataFrame()
        
        # Combine all markets
        combined_df = pd.concat(all_odds_data, ignore_index=True)
        
        # Apply date filters if provided
        if start_date:
            combined_df = combined_df[combined_df['commence_time'] >= start_date]
        
        if end_date:
            combined_df = combined_df[combined_df['commence_time'] <= end_date]
        
        return combined_df
    
    def fetch_match_odds(self, match_id: str, markets: Optional[List[str]] = None,
                       bookmakers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetch odds for a specific match.
        
        Args:
            match_id: Unique identifier for the match
            markets: List of markets to fetch (default: all)
            bookmakers: List of bookmakers to include (default: all)
            
        Returns:
            DataFrame containing odds data for the match
        """
        # For Odds API, we need to know the sport key
        # This is a limitation, but we'll handle it by searching through all sports
        
        sports = self.get_available_competitions()
        
        all_odds_data = []
        
        for sport_key in sports.keys():
            try:
                odds_df = self.fetch_odds(sport_key, markets=markets, bookmakers=bookmakers)
                
                # Filter for the specific match ID
                match_odds = odds_df[odds_df['match_id'] == match_id]
                
                if not match_odds.empty:
                    all_odds_data.append(match_odds)
                    break  # Found the match, no need to check other sports
            
            except Exception as e:
                logger.debug(f"No match found in {sport_key}: {str(e)}")
        
        if not all_odds_data:
            logger.warning(f"No odds found for match ID {match_id}")
            return pd.DataFrame()
        
        return pd.concat(all_odds_data, ignore_index=True)
    
    def _transform_odds_response(self, response_data: List[Dict[str, Any]], 
                               market_type: str, 
                               filter_bookmakers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform the API response into a structured DataFrame.
        
        Args:
            response_data: Raw API response
            market_type: Market type (h2h, spreads, totals)
            filter_bookmakers: Optional list of bookmakers to include
            
        Returns:
            DataFrame with structured odds data
        """
        if not response_data:
            return pd.DataFrame()
        
        records = []
        
        for event in response_data:
            match_id = event.get('id')
            home_team = event.get('home_team')
            away_team = event.get('away_team')
            sport_key = event.get('sport_key')
            commence_time = event.get('commence_time')
            
            # Process each bookmaker's odds
            if 'bookmakers' in event:
                for bookmaker in event['bookmakers']:
                    bookmaker_id = bookmaker.get('key')
                    bookmaker_name = bookmaker.get('title')
                    
                    # Skip if not in the filter list
                    if filter_bookmakers and bookmaker_id not in filter_bookmakers:
                        continue
                    
                    last_update = bookmaker.get('last_update')
                    
                    # Process markets for this bookmaker
                    if 'markets' in bookmaker:
                        for market_data in bookmaker['markets']:
                            market_key = market_data.get('key')
                            
                            # Skip if not the requested market
                            if market_key != market_type:
                                continue
                            
                            # Process outcomes
                            if 'outcomes' in market_data:
                                for outcome in market_data['outcomes']:
                                    name = outcome.get('name')
                                    price = outcome.get('price')
                                    point = outcome.get('point')  # For spreads and totals
                                    
                                    # Determine outcome type
                                    outcome_type = None
                                    if market_key == 'h2h':
                                        if name == home_team:
                                            outcome_type = 'home'
                                        elif name == away_team:
                                            outcome_type = 'away'
                                        else:
                                            outcome_type = 'draw'
                                    elif market_key in ['spreads', 'alternate_spreads']:
                                        if name == home_team:
                                            outcome_type = 'home_spread'
                                        else:
                                            outcome_type = 'away_spread'
                                    elif market_key in ['totals', 'alternate_totals']:
                                        outcome_type = name.lower()  # 'over' or 'under'
                                    elif market_key == 'btts':
                                        outcome_type = name.lower()  # 'yes' or 'no'
                                    
                                    record = {
                                        'match_id': match_id,
                                        'sport_key': sport_key,
                                        'home_team': home_team,
                                        'away_team': away_team,
                                        'commence_time': commence_time,
                                        'bookmaker_id': bookmaker_id,
                                        'bookmaker_name': bookmaker_name,
                                        'last_update': last_update,
                                        'market': market_key,
                                        'outcome_name': name,
                                        'outcome_type': outcome_type,
                                        'odds': price
                                    }
                                    
                                    # Add point value for spreads and totals
                                    if point is not None:
                                        record['point'] = point
                                    
                                    records.append(record)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Convert timestamps to datetime
        for col in ['commence_time', 'last_update']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df


class FileOddsConnector(OddsProviderConnector):
    """
    Connector for reading odds data from files.
    
    This connector is useful for testing, historical analysis, or
    when working with pre-downloaded or exported odds data.
    """
    
    def __init__(self, data_dir: Union[str, Path], file_format: str = 'csv'):
        """
        Initialize the file-based odds connector.
        
        Args:
            data_dir: Directory containing odds data files
            file_format: File format ('csv', 'parquet', or 'json')
        """
        self.data_dir = Path(data_dir)
        self.file_format = file_format.lower()
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
        
        # Available competitions and metadata
        self._competitions = None
        self._bookmakers = None
        self._markets = None
        
        # Cache of loaded data
        self._loaded_data = {}
    
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a data file into a DataFrame.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame containing the loaded data
        """
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist")
            return pd.DataFrame()
        
        try:
            if self.file_format == 'csv':
                return pd.read_csv(file_path)
            elif self.file_format == 'parquet':
                return pd.read_parquet(file_path)
            elif self.file_format == 'json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {self.file_format}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def _scan_competitions(self) -> Dict[str, str]:
        """
        Scan the data directory for available competitions.
        
        Returns:
            Dictionary mapping competition IDs to competition names
        """
        if self._competitions is not None:
            return self._competitions
        
        competitions = {}
        metadata_file = self.data_dir / f"competitions.{self.file_format}"
        
        # Try to load from metadata file first
        if metadata_file.exists():
            try:
                df = self._load_file(metadata_file)
                if not df.empty and 'id' in df.columns and 'name' in df.columns:
                    for _, row in df.iterrows():
                        competitions[row['id']] = row['name']
            except Exception as e:
                logger.error(f"Error loading competitions metadata: {str(e)}")
        
        # If no metadata or failed, try to infer from directory structure
        if not competitions:
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    competitions[item.name] = item.name
        
        self._competitions = competitions
        return competitions
    
    def get_available_competitions(self) -> Dict[str, str]:
        """
        Get available competitions/leagues.
        
        Returns:
            Dictionary mapping competition IDs to competition names
        """
        return self._scan_competitions()
    
    def _scan_bookmakers(self) -> List[str]:
        """
        Scan the data files for available bookmakers.
        
        Returns:
            List of bookmaker IDs
        """
        if self._bookmakers is not None:
            return self._bookmakers
        
        bookmakers = set()
        
        # Try to load from metadata file first
        metadata_file = self.data_dir / f"bookmakers.{self.file_format}"
        if metadata_file.exists():
            try:
                df = self._load_file(metadata_file)
                if not df.empty and 'id' in df.columns:
                    bookmakers.update(df['id'].unique())
            except Exception as e:
                logger.error(f"Error loading bookmakers metadata: {str(e)}")
        
        # If no metadata or failed, try to infer from a sample data file
        if not bookmakers:
            competitions = self.get_available_competitions()
            if competitions:
                comp_id = list(competitions.keys())[0]
                sample_file = self.data_dir / comp_id / f"odds.{self.file_format}"
                
                if sample_file.exists():
                    df = self._load_file(sample_file)
                    if not df.empty and 'bookmaker_id' in df.columns:
                        bookmakers.update(df['bookmaker_id'].unique())
        
        self._bookmakers = list(bookmakers)
        return self._bookmakers
    
    def get_available_bookmakers(self) -> List[str]:
        """
        Get a list of available bookmakers.
        
        Returns:
            List of bookmaker IDs
        """
        return self._scan_bookmakers()
    
    def _scan_markets(self) -> Dict[str, str]:
        """
        Scan the data files for available markets.
        
        Returns:
            Dictionary mapping market IDs to market names
        """
        if self._markets is not None:
            return self._markets
        
        markets = {}
        
        # Try to load from metadata file first
        metadata_file = self.data_dir / f"markets.{self.file_format}"
        if metadata_file.exists():
            try:
                df = self._load_file(metadata_file)
                if not df.empty and 'id' in df.columns and 'name' in df.columns:
                    for _, row in df.iterrows():
                        markets[row['id']] = row['name']
            except Exception as e:
                logger.error(f"Error loading markets metadata: {str(e)}")
        
        # If no metadata or failed, use some standard markets
        if not markets:
            markets = {
                "1x2": "1X2 (Home/Draw/Away)",
                "h2h": "Moneyline/Head to Head",
                "asian_handicap": "Asian Handicap",
                "over_under": "Over/Under",
                "btts": "Both Teams to Score",
                "correct_score": "Correct Score",
                "double_chance": "Double Chance",
                "draw_no_bet": "Draw No Bet"
            }
        
        self._markets = markets
        return markets
    
    def get_available_markets(self) -> Dict[str, str]:
        """
        Get available betting markets.
        
        Returns:
            Dictionary mapping market IDs to market names
        """
        return self._scan_markets()
    
    def fetch_odds(self, competition_id: str, start_date: Optional[str] = None, 
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch odds for a specific competition/league.
        
        Args:
            competition_id: ID of the competition to fetch odds for
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            
        Returns:
            DataFrame containing odds data
        """
        # Check if competition exists
        competitions = self.get_available_competitions()
        if competition_id not in competitions:
            logger.warning(f"Competition {competition_id} not found")
            return pd.DataFrame()
        
        # Check if data is already loaded
        cache_key = f"{competition_id}_{start_date}_{end_date}"
        if cache_key in self._loaded_data:
            return self._loaded_data[cache_key]
        
        # Find and load the data file
        data_file = self.data_dir / competition_id / f"odds.{self.file_format}"
        if not data_file.exists():
            data_file = self.data_dir / f"{competition_id}.{self.file_format}"
        
        if not data_file.exists():
            logger.warning(f"No odds data found for competition {competition_id}")
            return pd.DataFrame()
        
        df = self._load_file(data_file)
        
        # Apply date filters if provided
        if start_date and 'commence_time' in df.columns:
            df = df[df['commence_time'] >= start_date]
        
        if end_date and 'commence_time' in df.columns:
            df = df[df['commence_time'] <= end_date]
        
        # Cache the loaded data
        self._loaded_data[cache_key] = df
        
        return df
    
    def fetch_match_odds(self, match_id: str) -> pd.DataFrame:
        """
        Fetch odds for a specific match.
        
        Args:
            match_id: Unique identifier for the match
            
        Returns:
            DataFrame containing odds data for the match
        """
        # Try to find the match in all competitions
        for comp_id in self.get_available_competitions():
            df = self.fetch_odds(comp_id)
            
            if 'match_id' in df.columns:
                match_df = df[df['match_id'] == match_id]
                if not match_df.empty:
                    return match_df
        
        # If not found, check if there's a match-specific file
        match_file = self.data_dir / "matches" / f"{match_id}.{self.file_format}"
        if match_file.exists():
            return self._load_file(match_file)
        
        logger.warning(f"No odds found for match {match_id}")
        return pd.DataFrame()


class OddsConnectorFactory:
    """
    Factory class for creating odds provider connectors.
    
    This class provides methods for creating and configuring
    different types of odds provider connectors.
    """
    
    @staticmethod
    def create_connector(connector_type: str, **kwargs) -> OddsProviderConnector:
        """
        Create a new odds provider connector.
        
        Args:
            connector_type: Type of connector ('api', 'file', etc.)
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured odds provider connector
            
        Raises:
            ValueError: If the connector type is unsupported
        """
        if connector_type.lower() == 'api' or connector_type.lower() == 'odds_api':
            if 'api_key' not in kwargs:
                raise ValueError("API key is required for OddsAPIConnector")
            
            return OddsAPIConnector(
                api_key=kwargs['api_key'],
                request_timeout=kwargs.get('request_timeout', 30),
                request_delay=kwargs.get('request_delay', 1.0),
                max_retries=kwargs.get('max_retries', 3),
                cache_dir=kwargs.get('cache_dir'),
                cache_duration=kwargs.get('cache_duration', 3600)
            )
        
        elif connector_type.lower() == 'file':
            if 'data_dir' not in kwargs:
                raise ValueError("Data directory is required for FileOddsConnector")
            
            return FileOddsConnector(
                data_dir=kwargs['data_dir'],
                file_format=kwargs.get('file_format', 'csv')
            )
        
        else:
            raise ValueError(f"Unsupported connector type: {connector_type}")
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> OddsProviderConnector:
        """
        Create a connector from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured odds provider connector
        """
        if 'type' not in config:
            raise ValueError("Connector type must be specified in configuration")
        
        connector_type = config['type']
        params = {k: v for k, v in config.items() if k != 'type'}
        
        return OddsConnectorFactory.create_connector(connector_type, **params)


# Example usage functions

def get_best_odds(connectors: List[OddsProviderConnector], match_id: str, 
                 market: str = 'h2h') -> pd.DataFrame:
    """
    Get the best available odds for a match across multiple providers.
    
    Args:
        connectors: List of odds provider connectors
        match_id: ID of the match
        market: Betting market to analyze
        
    Returns:
        DataFrame with best odds for each outcome
    """
    all_odds = []
    
    for connector in connectors:
        try:
            odds_df = connector.fetch_match_odds(match_id)
            if not odds_df.empty:
                all_odds.append(odds_df)
        except Exception as e:
            logger.error(f"Error fetching odds from connector: {str(e)}")
    
    if not all_odds:
        logger.warning(f"No odds found for match {match_id}")
        return pd.DataFrame()
    
    # Combine all odds data
    combined_df = pd.concat(all_odds, ignore_index=True)
    
    # Filter for the specified market
    market_df = combined_df[combined_df['market'] == market]
    
    if market_df.empty:
        logger.warning(f"No odds found for market {market}")
        return pd.DataFrame()
    
    # Find best odds for each outcome
    best_odds = market_df.groupby('outcome_type')['odds'].max().reset_index()
    
    # Join with details of the bookmaker offering each best price
    result = []
    for _, row in best_odds.iterrows():
        outcome = row['outcome_type']
        best_price = row['odds']
        
        # Find bookmaker with this price
        best_row = market_df[
            (market_df['outcome_type'] == outcome) & 
            (market_df['odds'] == best_price)
        ].iloc[0]
        
        result.append({
            'match_id': match_id,
            'outcome_type': outcome,
            'odds': best_price,
            'bookmaker': best_row['bookmaker_name']
        })
    
    return pd.DataFrame(result)


def get_odds_history(connector: OddsProviderConnector, match_id: str, 
                   market: str = 'h2h', bookmaker: Optional[str] = None) -> pd.DataFrame:
    """
    Get the odds history for a match.
    
    Args:
        connector: Odds provider connector
        match_id: ID of the match
        market: Betting market to analyze
        bookmaker: Optional bookmaker to filter for
        
    Returns:
        DataFrame with odds history
    """
    odds_df = connector.fetch_match_odds(match_id)
    
    if odds_df.empty:
        logger.warning(f"No odds found for match {match_id}")
        return pd.DataFrame()
    
    # Filter for the specified market
    market_df = odds_df[odds_df['market'] == market]
    
    if market_df.empty:
        logger.warning(f"No odds found for market {market}")
        return pd.DataFrame()
    
    # Filter for a specific bookmaker if provided
    if bookmaker:
        market_df = market_df[market_df['bookmaker_id'] == bookmaker]
        
        if market_df.empty:
            logger.warning(f"No odds found for bookmaker {bookmaker}")
            return pd.DataFrame()
    
    # Sort by last update time
    market_df = market_df.sort_values('last_update')
    
    return market_df[['match_id', 'outcome_type', 'odds', 'bookmaker_id', 'last_update']] 