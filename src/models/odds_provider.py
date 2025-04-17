"""
Odds provider interface for the betting system.

This module provides standardized interfaces for connecting to
various odds data providers and retrieving odds data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import requests
import json
import time

logger = logging.getLogger(__name__)

class OddsProviderConnector(ABC):
    """
    Abstract interface for connecting to odds data providers.
    
    This interface ensures that different odds data sources can be used
    interchangeably throughout the betting system.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_timeout: int = 300):
        """
        Initialize the odds provider connector.
        
        Args:
            api_key: Optional API key for the odds provider
            cache_timeout: Time in seconds to cache odds data
        """
        self._api_key = api_key
        self._cache_timeout = cache_timeout
        self._cache = {}
        self._last_update = {}
    
    @property
    def provider_name(self) -> str:
        """Get the name of the odds provider."""
        return self.__class__.__name__
    
    @property
    @abstractmethod
    def supported_markets(self) -> List[str]:
        """Get the list of markets supported by this odds provider."""
        pass
    
    @property
    @abstractmethod
    def supported_bookmakers(self) -> List[str]:
        """Get the list of bookmakers supported by this odds provider."""
        pass
    
    @abstractmethod
    def get_odds(self, 
                match_id: str,
                market: str,
                bookmaker: Optional[str] = None,
                use_cache: bool = True) -> Dict[str, Any]:
        """
        Get current odds for a specific match and market.
        
        Args:
            match_id: Unique identifier for the match
            market: Market to get odds for (e.g. '1X2', 'over_under_2.5')
            bookmaker: Optional bookmaker name to filter by
            use_cache: Whether to use cached data if available
            
        Returns:
            Dict with odds information
        """
        pass
    
    @abstractmethod
    def get_odds_history(self,
                        match_id: str,
                        market: str,
                        bookmaker: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical odds for a specific match and market.
        
        Args:
            match_id: Unique identifier for the match
            market: Market to get odds for
            bookmaker: Optional bookmaker name to filter by
            start_time: Optional start time for historical data
            end_time: Optional end time for historical data
            
        Returns:
            DataFrame with historical odds data
        """
        pass
    
    @abstractmethod
    def get_matches(self, 
                   league_id: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get upcoming or past matches with available odds.
        
        Args:
            league_id: Optional league identifier to filter by
            start_date: Optional start date for matches
            end_date: Optional end date for matches
            
        Returns:
            DataFrame with match information
        """
        pass
    
    @abstractmethod
    def search_matches(self, query: str) -> pd.DataFrame:
        """
        Search for matches by team name or other criteria.
        
        Args:
            query: Search query string
            
        Returns:
            DataFrame with matching matches
        """
        pass
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self._cache or cache_key not in self._last_update:
            return False
        
        elapsed = time.time() - self._last_update[cache_key]
        return elapsed < self._cache_timeout
    
    def _add_to_cache(self, cache_key: str, data: Any) -> None:
        """
        Add data to the cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        self._cache[cache_key] = data
        self._last_update[cache_key] = time.time()
        
    def clear_cache(self) -> None:
        """Clear the odds cache."""
        self._cache = {}
        self._last_update = {}


class TheOddsAPIConnector(OddsProviderConnector):
    """
    Connector for The Odds API (https://the-odds-api.com).
    
    This connector retrieves odds data from The Odds API for various sports and markets.
    """
    
    API_BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Market mapping from API format to internal format
    MARKET_MAPPING = {
        "h2h": "1X2",
        "spreads": "asian_handicap",
        "totals": "over_under",
        "outrights": "outright",
        "btts": "btts"
    }
    
    # Reverse market mapping
    REVERSE_MARKET_MAPPING = {v: k for k, v in MARKET_MAPPING.items()}
    
    def __init__(self, api_key: str, cache_timeout: int = 300):
        """
        Initialize The Odds API connector.
        
        Args:
            api_key: API key for The Odds API
            cache_timeout: Time in seconds to cache odds data
        """
        super().__init__(api_key, cache_timeout)
        
        if not api_key:
            raise ValueError("API key is required for The Odds API")
        
        self._supported_sports = None
    
    @property
    def supported_markets(self) -> List[str]:
        """Get the list of markets supported by The Odds API."""
        return list(self.MARKET_MAPPING.values())
    
    @property
    def supported_bookmakers(self) -> List[str]:
        """Get the list of bookmakers supported by The Odds API."""
        # This list should be refreshed periodically as it may change
        return [
            "bet365", "betfair", "betmgm", "betonline", "betway", "caesars", 
            "draftkings", "fanduel", "pinnacle", "unibet", "williamhill"
        ]
    
    def _get_api_sport_key(self, league_id: str) -> str:
        """
        Convert internal league ID to API sport key.
        
        Args:
            league_id: Internal league identifier
            
        Returns:
            Sport key for the API
        """
        # This is a simplified mapping - would need to be expanded in a real implementation
        sport_key_mapping = {
            "epl": "soccer_epl",
            "la_liga": "soccer_spain_la_liga",
            "bundesliga": "soccer_germany_bundesliga",
            "serie_a": "soccer_italy_serie_a",
            "ligue_1": "soccer_france_ligue_one",
            "champions_league": "soccer_uefa_champs_league"
        }
        
        return sport_key_mapping.get(league_id, league_id)
    
    def _get_internal_league_id(self, sport_key: str) -> str:
        """
        Convert API sport key to internal league ID.
        
        Args:
            sport_key: Sport key from the API
            
        Returns:
            Internal league identifier
        """
        # Reverse of the mapping in _get_api_sport_key
        league_id_mapping = {
            "soccer_epl": "epl",
            "soccer_spain_la_liga": "la_liga",
            "soccer_germany_bundesliga": "bundesliga",
            "soccer_italy_serie_a": "serie_a",
            "soccer_france_ligue_one": "ligue_1",
            "soccer_uefa_champs_league": "champions_league"
        }
        
        return league_id_mapping.get(sport_key, sport_key)
    
    def _api_market_to_internal(self, api_market: str) -> str:
        """
        Convert API market name to internal market name.
        
        Args:
            api_market: Market name from the API
            
        Returns:
            Internal market name
        """
        return self.MARKET_MAPPING.get(api_market, api_market)
    
    def _internal_market_to_api(self, internal_market: str) -> str:
        """
        Convert internal market name to API market name.
        
        Args:
            internal_market: Internal market name
            
        Returns:
            API market name
        """
        return self.REVERSE_MARKET_MAPPING.get(internal_market, internal_market)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to The Odds API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.API_BASE_URL}/{endpoint}"
        
        # Add API key to params
        params["apiKey"] = self._api_key
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {str(e)}")
            if hasattr(response, 'text'):
                logger.error(f"Response: {response.text}")
            raise
    
    def get_supported_sports(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get list of supported sports from the API.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            List of sport information dictionaries
        """
        cache_key = "supported_sports"
        
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            data = self._make_request("sports", {})
            self._add_to_cache(cache_key, data)
            self._supported_sports = data
            return data
        except Exception as e:
            logger.error(f"Error getting supported sports: {str(e)}")
            # Return empty list on error
            return []
    
    def get_odds(self, 
                match_id: str,
                market: str,
                bookmaker: Optional[str] = None,
                use_cache: bool = True) -> Dict[str, Any]:
        """
        Get current odds for a specific match and market.
        
        Args:
            match_id: Unique identifier for the match in format "sport_key:event_id"
            market: Market to get odds for (e.g. '1X2', 'over_under')
            bookmaker: Optional bookmaker name to filter by
            use_cache: Whether to use cached data if available
            
        Returns:
            Dict with odds information
        """
        # Parse match_id to get sport_key and event_id
        if ":" in match_id:
            sport_key, event_id = match_id.split(":", 1)
        else:
            # If match_id doesn't contain sport_key, handle appropriately
            logger.error(f"Invalid match_id format: {match_id}")
            return {}
        
        # Convert internal market to API market
        api_market = self._internal_market_to_api(market)
        if not api_market:
            logger.error(f"Unsupported market: {market}")
            return {}
        
        cache_key = f"odds:{match_id}:{market}"
        if bookmaker:
            cache_key += f":{bookmaker}"
        
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        params = {
            "sport": sport_key,
            "regions": "us,uk,eu",  # Adjust as needed
            "markets": api_market
        }
        
        if bookmaker:
            params["bookmakers"] = bookmaker
        
        try:
            data = self._make_request("sports/{sport}/odds", params)
            
            # Find the specific event
            event_data = {}
            for event in data:
                if str(event.get("id")) == event_id:
                    event_data = event
                    break
            
            # Format the response to match our internal structure
            result = self._format_odds_response(event_data, market, bookmaker)
            
            self._add_to_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Error getting odds for {match_id}, market {market}: {str(e)}")
            return {}
    
    def _format_odds_response(self, 
                             event_data: Dict[str, Any],
                             market: str,
                             bookmaker: Optional[str] = None) -> Dict[str, Any]:
        """
        Format API response to our internal structure.
        
        Args:
            event_data: Event data from the API
            market: Internal market name
            bookmaker: Optional bookmaker name to filter by
            
        Returns:
            Formatted odds data
        """
        if not event_data:
            return {}
        
        # Extract basic event information
        result = {
            "match_id": f"{event_data.get('sport_key')}:{event_data.get('id')}",
            "home_team": event_data.get("home_team"),
            "away_team": event_data.get("away_team"),
            "start_time": event_data.get("commence_time"),
            "market": market,
            "bookmakers": []
        }
        
        # Extract bookmaker odds
        bookmakers_data = event_data.get("bookmakers", [])
        if bookmaker:
            bookmakers_data = [b for b in bookmakers_data if b.get("key") == bookmaker]
        
        for bm_data in bookmakers_data:
            bm_key = bm_data.get("key")
            bm_name = bm_data.get("title")
            
            # Find markets data
            markets_data = bm_data.get("markets", [])
            api_market = self._internal_market_to_api(market)
            market_data = next((m for m in markets_data if m.get("key") == api_market), None)
            
            if not market_data:
                continue
            
            outcomes = []
            for outcome in market_data.get("outcomes", []):
                outcomes.append({
                    "name": outcome.get("name"),
                    "price": outcome.get("price"),
                    "point": outcome.get("point")
                })
            
            result["bookmakers"].append({
                "key": bm_key,
                "name": bm_name,
                "last_update": bm_data.get("last_update"),
                "outcomes": outcomes
            })
        
        return result
    
    def get_odds_history(self,
                        match_id: str,
                        market: str,
                        bookmaker: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical odds for a specific match and market.
        
        Note: The Odds API doesn't provide historical odds directly, so this is a simplified
        implementation. In a real-world scenario, you would need to store odds data over time.
        
        Args:
            match_id: Unique identifier for the match
            market: Market to get odds for
            bookmaker: Optional bookmaker name to filter by
            start_time: Optional start time for historical data
            end_time: Optional end time for historical data
            
        Returns:
            DataFrame with historical odds data
        """
        # This is a simplified implementation as The Odds API doesn't provide historical data
        # In a real implementation, you would store odds data in a database over time
        logger.warning("Historical odds not available directly from The Odds API")
        
        # Return an empty DataFrame with the expected structure
        columns = [
            "match_id", "market", "bookmaker", "outcome", "price", 
            "point", "timestamp", "home_team", "away_team"
        ]
        
        return pd.DataFrame(columns=columns)
    
    def get_matches(self, 
                   league_id: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get upcoming matches with available odds.
        
        Args:
            league_id: Optional league identifier to filter by
            start_date: Optional start date for matches (not used in this implementation)
            end_date: Optional end date for matches (not used in this implementation)
            
        Returns:
            DataFrame with match information
        """
        # Convert league_id to API sport key if provided
        sport_key = None
        if league_id:
            sport_key = self._get_api_sport_key(league_id)
        
        # Get all sports if sport_key not specified
        if not sport_key:
            # Ensure we have supported sports data
            if not self._supported_sports:
                self.get_supported_sports()
            
            all_matches = []
            for sport in self._supported_sports:
                sport_key = sport.get("key")
                if "soccer" in sport_key:  # Filter for soccer if needed
                    matches = self._get_matches_for_sport(sport_key)
                    all_matches.extend(matches)
            
            matches_data = all_matches
        else:
            # Get matches for specific sport
            matches_data = self._get_matches_for_sport(sport_key)
        
        # Convert to DataFrame
        matches = []
        for match in matches_data:
            sport_key = match.get("sport_key")
            league_id = self._get_internal_league_id(sport_key)
            
            matches.append({
                "match_id": f"{sport_key}:{match.get('id')}",
                "league_id": league_id,
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "start_time": match.get("commence_time"),
                "has_odds": bool(match.get("bookmakers"))
            })
        
        matches_df = pd.DataFrame(matches)
        
        # Filter by date if provided
        if start_date or end_date:
            if "start_time" in matches_df.columns:
                matches_df["start_time"] = pd.to_datetime(matches_df["start_time"])
                
                if start_date:
                    matches_df = matches_df[matches_df["start_time"] >= start_date]
                
                if end_date:
                    matches_df = matches_df[matches_df["start_time"] <= end_date]
        
        return matches_df
    
    def _get_matches_for_sport(self, sport_key: str) -> List[Dict[str, Any]]:
        """
        Get upcoming matches for a specific sport.
        
        Args:
            sport_key: Sport key for the API
            
        Returns:
            List of match data dictionaries
        """
        cache_key = f"matches:{sport_key}"
        
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        params = {
            "sport": sport_key,
            "regions": "us,uk,eu",  # Adjust as needed
            "odds_format": "decimal"
        }
        
        try:
            data = self._make_request(f"sports/{sport_key}/odds", params)
            self._add_to_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error getting matches for sport {sport_key}: {str(e)}")
            return []
    
    def search_matches(self, query: str) -> pd.DataFrame:
        """
        Search for matches by team name.
        
        Args:
            query: Search query string
            
        Returns:
            DataFrame with matching matches
        """
        # Get all matches first
        all_matches = self.get_matches()
        
        # Filter by query
        if query and not all_matches.empty:
            query = query.lower()
            mask = (
                all_matches["home_team"].str.lower().str.contains(query, na=False) |
                all_matches["away_team"].str.lower().str.contains(query, na=False)
            )
            return all_matches[mask]
        
        return all_matches


class MockOddsConnector(OddsProviderConnector):
    """
    Mock odds provider for testing and development.
    
    This connector generates synthetic odds data for testing without requiring
    an external API connection.
    """
    
    def __init__(self, cache_timeout: int = 300):
        """
        Initialize the mock odds provider.
        
        Args:
            cache_timeout: Time in seconds to cache odds data
        """
        super().__init__(None, cache_timeout)
        
        # Pre-populated test matches
        self._test_matches = self._generate_test_matches()
    
    @property
    def supported_markets(self) -> List[str]:
        """Get the list of markets supported by this mock provider."""
        return ["1X2", "over_under", "btts", "asian_handicap", "correct_score"]
    
    @property
    def supported_bookmakers(self) -> List[str]:
        """Get the list of bookmakers supported by this mock provider."""
        return ["bet365", "betfair", "pinnacle", "williamhill", "marathon"]
    
    def _generate_test_matches(self) -> List[Dict[str, Any]]:
        """
        Generate test match data.
        
        Returns:
            List of test match dictionaries
        """
        # Sample leagues and teams
        leagues = {
            "epl": {"name": "English Premier League", "teams": [
                "Arsenal", "Chelsea", "Liverpool", "Manchester City", 
                "Manchester United", "Tottenham", "Leicester", "Everton"
            ]},
            "la_liga": {"name": "Spanish La Liga", "teams": [
                "Barcelona", "Real Madrid", "Atletico Madrid", "Sevilla",
                "Valencia", "Villarreal", "Real Sociedad", "Athletic Bilbao"
            ]},
            "bundesliga": {"name": "German Bundesliga", "teams": [
                "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
                "Wolfsburg", "Gladbach", "Eintracht Frankfurt", "Schalke 04"
            ]}
        }
        
        # Generate matches
        import random
        from datetime import datetime, timedelta
        
        matches = []
        match_id = 1000
        
        for league_id, league_info in leagues.items():
            teams = league_info["teams"]
            
            # Create matches between each pair of teams
            for i in range(len(teams)):
                for j in range(i+1, len(teams)):
                    home_team = teams[i]
                    away_team = teams[j]
                    
                    # Random start time in the next 14 days
                    days_ahead = random.randint(1, 14)
                    hours_ahead = random.randint(0, 23)
                    minutes_ahead = random.choice([0, 15, 30, 45])
                    
                    start_time = (datetime.now() + 
                                  timedelta(days=days_ahead, 
                                           hours=hours_ahead, 
                                           minutes=minutes_ahead))
                    
                    matches.append({
                        "match_id": f"{league_id}:{match_id}",
                        "league_id": league_id,
                        "league_name": league_info["name"],
                        "home_team": home_team,
                        "away_team": away_team,
                        "start_time": start_time.isoformat(),
                        "has_odds": True
                    })
                    
                    match_id += 1
        
        return matches
    
    def _generate_odds_for_match(self, 
                                match_id: str,
                                market: str,
                                bookmaker: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate synthetic odds for a match and market.
        
        Args:
            match_id: Match identifier
            market: Market to generate odds for
            bookmaker: Optional bookmaker to generate odds for
            
        Returns:
            Dictionary with generated odds data
        """
        import random
        
        # Find the match
        match = next((m for m in self._test_matches if m["match_id"] == match_id), None)
        if not match:
            return {}
        
        # Bookmakers to generate data for
        bookmakers_list = [bookmaker] if bookmaker else self.supported_bookmakers
        
        # Base result
        result = {
            "match_id": match_id,
            "home_team": match["home_team"],
            "away_team": match["away_team"],
            "start_time": match["start_time"],
            "market": market,
            "bookmakers": []
        }
        
        # Generate odds for each bookmaker
        for bm in bookmakers_list:
            # Slight variation in odds between bookmakers
            variation = random.uniform(0.95, 1.05)
            
            if market == "1X2":
                # Generate 1X2 odds
                home_bias = random.uniform(0.9, 1.1)  # Random bias towards home team
                home_strength = random.uniform(0.8, 1.2)  # Random team strength
                away_strength = random.uniform(0.8, 1.2)
                
                # Base probabilities (will be adjusted to sum to >1 for margin)
                home_prob = 0.45 * home_bias * home_strength
                draw_prob = 0.25
                away_prob = 0.3 / home_bias * away_strength
                
                # Adjust to ensure they sum to more than 1 (bookmaker margin)
                total_prob = home_prob + draw_prob + away_prob
                margin = random.uniform(1.05, 1.15)  # 5-15% margin
                
                home_prob = home_prob / total_prob * margin
                draw_prob = draw_prob / total_prob * margin
                away_prob = away_prob / total_prob * margin
                
                # Convert to decimal odds
                home_odds = round(1 / home_prob * variation, 2)
                draw_odds = round(1 / draw_prob * variation, 2)
                away_odds = round(1 / away_prob * variation, 2)
                
                outcomes = [
                    {"name": match["home_team"], "price": home_odds, "point": None},
                    {"name": "Draw", "price": draw_odds, "point": None},
                    {"name": match["away_team"], "price": away_odds, "point": None}
                ]
            
            elif market == "over_under":
                # Generate over/under odds for multiple points
                points = [1.5, 2.5, 3.5]
                outcomes = []
                
                for point in points:
                    # Probabilities depend on the point value
                    if point == 1.5:
                        over_prob = random.uniform(0.65, 0.75)
                    elif point == 2.5:
                        over_prob = random.uniform(0.45, 0.55)
                    else:  # 3.5
                        over_prob = random.uniform(0.25, 0.35)
                    
                    under_prob = 1.05 - over_prob  # Slight margin
                    
                    over_odds = round(1 / over_prob * variation, 2)
                    under_odds = round(1 / under_prob * variation, 2)
                    
                    outcomes.extend([
                        {"name": "Over", "price": over_odds, "point": point},
                        {"name": "Under", "price": under_odds, "point": point}
                    ])
            
            elif market == "btts":
                # Both teams to score
                yes_prob = random.uniform(0.55, 0.65)
                no_prob = 1.05 - yes_prob  # Slight margin
                
                yes_odds = round(1 / yes_prob * variation, 2)
                no_odds = round(1 / no_prob * variation, 2)
                
                outcomes = [
                    {"name": "Yes", "price": yes_odds, "point": None},
                    {"name": "No", "price": no_odds, "point": None}
                ]
            
            elif market == "asian_handicap":
                # Asian handicap with different lines
                handicaps = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
                outcomes = []
                
                for handicap in handicaps:
                    # Home probability depends on handicap
                    home_prob = 0.5 + handicap * -0.1  # Adjust probability based on handicap
                    home_prob = max(0.1, min(0.9, home_prob))  # Keep within reasonable bounds
                    
                    away_prob = 1.05 - home_prob  # Slight margin
                    
                    home_odds = round(1 / home_prob * variation, 2)
                    away_odds = round(1 / away_prob * variation, 2)
                    
                    outcomes.extend([
                        {"name": match["home_team"], "price": home_odds, "point": handicap},
                        {"name": match["away_team"], "price": away_odds, "point": -handicap}
                    ])
            
            else:
                # Default market with random outcomes
                outcomes = [
                    {"name": "Outcome 1", "price": round(random.uniform(1.5, 3.0), 2), "point": None},
                    {"name": "Outcome 2", "price": round(random.uniform(1.5, 3.0), 2), "point": None},
                ]
            
            # Add to bookmakers list
            result["bookmakers"].append({
                "key": bm.lower(),
                "name": bm,
                "last_update": datetime.now().isoformat(),
                "outcomes": outcomes
            })
        
        return result
    
    def get_odds(self, 
                match_id: str,
                market: str,
                bookmaker: Optional[str] = None,
                use_cache: bool = True) -> Dict[str, Any]:
        """
        Get current odds for a specific match and market.
        
        Args:
            match_id: Unique identifier for the match
            market: Market to get odds for
            bookmaker: Optional bookmaker name to filter by
            use_cache: Whether to use cached data if available
            
        Returns:
            Dict with odds information
        """
        cache_key = f"odds:{match_id}:{market}"
        if bookmaker:
            cache_key += f":{bookmaker}"
        
        if use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        # Generate odds data
        result = self._generate_odds_for_match(match_id, market, bookmaker)
        
        self._add_to_cache(cache_key, result)
        return result
    
    def get_odds_history(self,
                        match_id: str,
                        market: str,
                        bookmaker: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get historical odds for a specific match and market.
        
        For the mock connector, this generates synthetic historical data.
        
        Args:
            match_id: Unique identifier for the match
            market: Market to get odds for
            bookmaker: Optional bookmaker name to filter by
            start_time: Optional start time for historical data
            end_time: Optional end time for historical data
            
        Returns:
            DataFrame with historical odds data
        """
        import random
        
        # Find the match
        match = next((m for m in self._test_matches if m["match_id"] == match_id), None)
        if not match:
            return pd.DataFrame()
        
        # Generate synthetic historical data
        history_data = []
        
        # Bookmakers to generate data for
        bookmakers_list = [bookmaker] if bookmaker else self.supported_bookmakers[:3]
        
        # Date range
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        # Generate data points for each day
        current_time = start_time
        while current_time <= end_time:
            # Current odds
            current_odds = self._generate_odds_for_match(match_id, market)
            
            for bm_data in current_odds.get("bookmakers", []):
                if bookmaker and bm_data["key"] != bookmaker.lower():
                    continue
                
                for outcome in bm_data.get("outcomes", []):
                    # Add some random variation to historical prices
                    variation = random.uniform(0.95, 1.05)
                    historical_price = round(outcome["price"] * variation, 2)
                    
                    history_data.append({
                        "match_id": match_id,
                        "market": market,
                        "bookmaker": bm_data["key"],
                        "outcome": outcome["name"],
                        "price": historical_price,
                        "point": outcome["point"],
                        "timestamp": current_time.isoformat(),
                        "home_team": match["home_team"],
                        "away_team": match["away_team"]
                    })
            
            # Move to next time point (every 3 hours)
            current_time += timedelta(hours=3)
        
        return pd.DataFrame(history_data)
    
    def get_matches(self, 
                   league_id: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get upcoming or past matches with available odds.
        
        Args:
            league_id: Optional league identifier to filter by
            start_date: Optional start date for matches
            end_date: Optional end date for matches
            
        Returns:
            DataFrame with match information
        """
        # Filter matches
        matches = self._test_matches.copy()
        
        if league_id:
            matches = [m for m in matches if m["league_id"] == league_id]
        
        if start_date or end_date:
            filtered_matches = []
            
            for match in matches:
                match_time = datetime.fromisoformat(match["start_time"])
                
                if start_date and match_time < start_date:
                    continue
                
                if end_date and match_time > end_date:
                    continue
                
                filtered_matches.append(match)
            
            matches = filtered_matches
        
        return pd.DataFrame(matches)
    
    def search_matches(self, query: str) -> pd.DataFrame:
        """
        Search for matches by team name or other criteria.
        
        Args:
            query: Search query string
            
        Returns:
            DataFrame with matching matches
        """
        if not query:
            return pd.DataFrame(self._test_matches)
        
        # Filter matches by query
        query = query.lower()
        matches = [
            m for m in self._test_matches if
            query in m["home_team"].lower() or
            query in m["away_team"].lower() or
            query in m.get("league_name", "").lower()
        ]
        
        return pd.DataFrame(matches) 