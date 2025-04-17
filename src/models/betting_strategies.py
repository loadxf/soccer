"""
Betting Strategies Module

This module implements various betting strategies for soccer matches, including:
- Value betting
- Kelly criterion
- Poisson distribution betting
- Model ensemble approaches
- Market movement analysis
- Specialized strategies (Asian handicap, Draw No Bet)

Each strategy can be used individually or combined with bankroll management
techniques to optimize betting performance.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Type, Protocol, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json
from enum import Enum, auto
import random
from collections import defaultdict
import scipy.stats as poisson
from scipy.stats import norm
import time
import math

# Import project components
try:
    from src.utils.logger import get_logger
    logger = get_logger("models.betting_strategies")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("models.betting_strategies")

try:
    from src.models.soccer_distributions import create_value_bets, DixonColesModel, BivariatePoissonModel
except ImportError:
    logger.warning("soccer_distributions module not available. Some strategies may not work.")

try:
    from src.data.soccer_features import extract_betting_features
except ImportError:
    logger.warning("soccer_features module not available. Some strategies may not work.")

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Define paths
MODELS_DIR = os.path.join(DATA_DIR, "models")
BETTING_STRATEGIES_DIR = os.path.join(MODELS_DIR, "betting_strategies")
os.makedirs(BETTING_STRATEGIES_DIR, exist_ok=True)

# Odds conversion utility functions
def decimal_to_fractional(decimal_odds: float) -> str:
    """
    Convert decimal odds to fractional odds as a string.
    
    Args:
        decimal_odds: Decimal odds (e.g., 2.5)
        
    Returns:
        str: Fractional odds (e.g., "3/2")
    """
    if decimal_odds <= 1:
        return "0/1"
    
    # Decimal odds = (numerator / denominator) + 1
    decimal_minus_one = decimal_odds - 1
    
    # Find the simplest fraction that approximates decimal_minus_one
    precision = 1e-6
    for denominator in range(1, 101):  # Limit to reasonable fractions
        numerator = round(decimal_minus_one * denominator)
        if abs(decimal_minus_one - numerator / denominator) < precision:
            from math import gcd
            common_divisor = gcd(numerator, denominator)
            return f"{numerator // common_divisor}/{denominator // common_divisor}"
    
    # If no simple fraction found, return the rounded approximation
    denominator = 100
    numerator = round(decimal_minus_one * denominator)
    from math import gcd
    common_divisor = gcd(numerator, denominator)
    return f"{numerator // common_divisor}/{denominator // common_divisor}"

def decimal_to_american(decimal_odds: float) -> int:
    """
    Convert decimal odds to American odds.
    
    Args:
        decimal_odds: Decimal odds (e.g., 2.5)
        
    Returns:
        int: American odds (e.g., +150 or -200)
    """
    if decimal_odds == 1:
        return 0
    elif decimal_odds >= 2:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))

def american_to_decimal(american_odds: int) -> float:
    """
    Convert American odds to decimal odds.
    
    Args:
        american_odds: American odds (e.g., +150 or -200)
        
    Returns:
        float: Decimal odds (e.g., 2.5)
    """
    if american_odds == 0:
        return 1.0
    elif american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))

def implied_probability(decimal_odds: float) -> float:
    """
    Calculate implied probability from decimal odds.
    
    Args:
        decimal_odds: Decimal odds (e.g., 2.5)
        
    Returns:
        float: Implied probability (e.g., 0.4)
    """
    return 1 / decimal_odds

class BetType(Enum):
    """Enumeration of bet types."""
    HOME = "Home"
    AWAY = "Away"
    DRAW = "Draw"
    OVER = "Over"
    UNDER = "Under"
    HOME_AH = "Home AH"  # Asian Handicap
    AWAY_AH = "Away AH"  # Asian Handicap
    BTTS_YES = "BTTS Yes"  # Both Teams To Score
    BTTS_NO = "BTTS No"   # Both Teams To Score
    OTHER = "Other"

@dataclass
class BettingStrategyResult:
    """
    Dataclass for standardizing betting strategy results.
    
    This provides a consistent interface for different betting strategies
    to return their evaluation results.
    """
    match_id: Union[str, int]
    home_team: Any
    away_team: Any
    date: Optional[datetime] = None
    bet_type: Optional[BetType] = None
    bet_description: Optional[str] = None
    odds: Optional[float] = None
    predicted_probability: Optional[float] = None
    implied_probability: Optional[float] = None
    edge: Optional[float] = None
    expected_value: Optional[float] = None
    recommended_stake: Optional[float] = None
    potential_profit: Optional[float] = None
    confidence_score: Optional[float] = None
    strategy_name: Optional[str] = None
    model_name: Optional[str] = None
    timestamp: datetime = datetime.now()
    extra: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.extra is None:
            self.extra = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            'match_id': self.match_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'date': self.date.isoformat() if self.date else None,
            'bet_type': self.bet_type.value if self.bet_type else None,
            'bet_description': self.bet_description,
            'odds': self.odds,
            'predicted_probability': self.predicted_probability,
            'implied_probability': self.implied_probability,
            'edge': self.edge,
            'expected_value': self.expected_value,
            'recommended_stake': self.recommended_stake,
            'potential_profit': self.potential_profit,
            'confidence_score': self.confidence_score,
            'strategy_name': self.strategy_name,
            'model_name': self.model_name,
            'timestamp': self.timestamp.isoformat(),
            'extra': self.extra
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BettingStrategyResult':
        """Create a BettingStrategyResult from a dictionary."""
        # Convert timestamp string back to datetime
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert date string back to datetime
        if 'date' in data and data['date']:
            data['date'] = datetime.fromisoformat(data['date'])
        
        # Convert bet_type string to enum
        if 'bet_type' in data and data['bet_type']:
            data['bet_type'] = BetType(data['bet_type'])
        
        return cls(**data)

class BettingStrategy(ABC):
    """
    Abstract base class for all betting strategies.
    
    This provides a common interface and shared functionality
    for different betting strategy implementations.
    """
    
    def __init__(self, 
                name: str,
                min_edge: float = 0.05,
                min_odds: float = 1.1,
                max_odds: float = 10.0,
                stake_percentage: float = 1.0,
                confidence_threshold: float = 0.6):
        """
        Initialize the betting strategy.
        
        Args:
            name: Name of the strategy
            min_edge: Minimum edge (difference between predicted and implied probability)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            stake_percentage: Percentage of standard stake to use (0-100)
            confidence_threshold: Minimum confidence score to place a bet (0-1)
        """
        self.name = name
        self.min_edge = min_edge
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.stake_percentage = stake_percentage / 100 if stake_percentage > 1 else stake_percentage
        self.confidence_threshold = confidence_threshold
        self.results = []
    
    @abstractmethod
    def evaluate_bet(self, 
                    match_id: Union[str, int],
                    home_team: Any,
                    away_team: Any,
                    odds: Dict[str, float],
                    predictions: Dict[str, float],
                    **kwargs) -> Optional[BettingStrategyResult]:
        """
        Evaluate a potential bet based on the strategy.
        
        Args:
            match_id: Identifier for the match
            home_team: Home team identifier
            away_team: Away team identifier
            odds: Dictionary of odds (e.g., {'home': 2.0, 'draw': 3.2, 'away': 4.5})
            predictions: Dictionary of predicted probabilities (e.g., {'home': 0.55, 'draw': 0.25, 'away': 0.2})
            **kwargs: Additional parameters specific to the strategy
            
        Returns:
            Optional[BettingStrategyResult]: Bet evaluation result or None if no bet recommended
        """
        pass
    
    def calculate_edge(self, predicted_probability: float, decimal_odds: float) -> float:
        """
        Calculate the edge for a bet.
        
        Edge = predicted probability - implied probability
        
        Args:
            predicted_probability: Predicted probability of the outcome
            decimal_odds: Decimal odds offered
            
        Returns:
            float: Edge value
        """
        implied_prob = implied_probability(decimal_odds)
        return predicted_probability - implied_prob
    
    def calculate_expected_value(self, predicted_probability: float, decimal_odds: float) -> float:
        """
        Calculate the expected value of a bet.
        
        EV = (probability of winning × amount won per bet) - (probability of losing × amount lost per bet)
        
        Args:
            predicted_probability: Predicted probability of the outcome
            decimal_odds: Decimal odds offered
            
        Returns:
            float: Expected value (positive is favorable)
        """
        return predicted_probability * (decimal_odds - 1) - (1 - predicted_probability)
    
    def calculate_stake(self, 
                      predicted_probability: float, 
                      decimal_odds: float, 
                      bankroll: float = 100.0,
                      method: str = "flat",
                      **kwargs) -> float:
        """
        Calculate the recommended stake for a bet.
        
        Args:
            predicted_probability: Predicted probability of the outcome
            decimal_odds: Decimal odds offered
            bankroll: Current bankroll
            method: Staking method ('flat', 'kelly', 'fractional_kelly')
            **kwargs: Additional parameters for specific staking methods
            
        Returns:
            float: Recommended stake
        """
        if method == "flat":
            # Simple flat staking (percentage of bankroll)
            base_stake = bankroll * (kwargs.get('flat_percentage', 0.02))
            return base_stake * self.stake_percentage
            
        elif method == "kelly":
            # Kelly Criterion: (bp - q) / b
            # where b = decimal odds - 1, p = probability of win, q = probability of loss
            b = decimal_odds - 1
            p = predicted_probability
            q = 1 - p
            
            kelly_stake = (b * p - q) / b if b > 0 else 0
            kelly_stake = max(0, min(1, kelly_stake))  # Constrain between 0 and 1
            
            return kelly_stake * bankroll * self.stake_percentage
            
        elif method == "fractional_kelly":
            # Fractional Kelly uses a fraction of the full Kelly stake
            fraction = kwargs.get('kelly_fraction', 0.5)
            
            b = decimal_odds - 1
            p = predicted_probability
            q = 1 - p
            
            kelly_stake = (b * p - q) / b if b > 0 else 0
            kelly_stake = max(0, min(1, kelly_stake))  # Constrain between 0 and 1
            
            return kelly_stake * bankroll * fraction * self.stake_percentage
        
        else:
            raise ValueError(f"Unknown staking method: {method}")
    
    def log_result(self, result: BettingStrategyResult) -> None:
        """Log a betting result."""
        self.results.append(result)
        logger.info(f"[{self.name}] {result.bet_description} - Edge: {result.edge:.4f}, EV: {result.expected_value:.4f}") 

class ValueBettingStrategy(BettingStrategy):
    """
    A simple value betting strategy that focuses on finding mispriced odds.
    
    This strategy:
    1. Uses historical closing line values to identify value
    2. Implements progressive staking based on edge
    3. Works with any predictive model that can produce probabilities
    4. Can be filtered by market, league, and minimum edge
    """
    
    def __init__(self, 
                name: str = "Value Betting",
                min_edge: float = 0.05,
                min_odds: float = 1.5,
                max_odds: float = 7.0,
                stake_percentage: float = 1.0,
                confidence_threshold: float = 0.7,
                market_filter: Optional[List[str]] = None,
                league_filter: Optional[List[str]] = None,
                progressive_staking: bool = True,
                max_bet_percentage: float = 0.05,
                closing_line_factor: float = 0.8):
        """
        Initialize the value betting strategy.
        
        Args:
            name: Name of the strategy
            min_edge: Minimum edge (difference between predicted and implied probability)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            stake_percentage: Percentage of standard stake to use (0-100)
            confidence_threshold: Minimum confidence score to place a bet (0-1)
            market_filter: List of markets to consider (e.g., ['1X2', 'over_under'])
            league_filter: List of leagues to consider
            progressive_staking: Whether to scale stake based on edge size
            max_bet_percentage: Maximum percentage of bankroll to bet
            closing_line_factor: Weight given to closing line vs. current odds
        """
        super().__init__(
            name=name,
            min_edge=min_edge,
            min_odds=min_odds,
            max_odds=max_odds,
            stake_percentage=stake_percentage,
            confidence_threshold=confidence_threshold
        )
        self.market_filter = market_filter or ['1X2', 'over_under', 'btts']
        self.league_filter = league_filter
        self.progressive_staking = progressive_staking
        self.max_bet_percentage = max_bet_percentage
        self.closing_line_factor = closing_line_factor
        
        # Historical performance tracking
        self.performance_history = []
        self.roi_by_market = {}
        self.roi_by_league = {}
    
    def calculate_value_rating(self, 
                             edge: float, 
                             odds: float, 
                             confidence: float) -> float:
        """
        Calculate a value rating for the bet based on edge, odds, and confidence.
        
        Args:
            edge: Edge percentage
            odds: Decimal odds
            confidence: Confidence score
            
        Returns:
            float: Value rating (higher is better)
        """
        # Simple weighted formula that prioritizes edge but considers odds and confidence
        return (edge * 0.7) + (confidence * 0.2) + (0.1 / odds if odds > 0 else 0)
    
    def calculate_progressive_stake(self, 
                                  edge: float, 
                                  confidence: float, 
                                  base_stake_percentage: float,
                                  bankroll: float) -> float:
        """
        Calculate stake using progressive staking based on edge size.
        
        Args:
            edge: Edge percentage
            confidence: Confidence score
            base_stake_percentage: Base stake percentage
            bankroll: Current bankroll
            
        Returns:
            float: Recommended stake amount
        """
        if not self.progressive_staking:
            return bankroll * base_stake_percentage
        
        # Calculate stake multiplier based on edge and confidence
        # Edge ranges from min_edge (e.g., 0.05) to potentially 0.2+ for great value
        # We want to scale from 1x to 3x the base stake
        
        # Normalize edge to a 0-1 scale for typical edge values
        normalized_edge = min(1.0, (edge - self.min_edge) / 0.15)
        
        # Calculate multiplier combining edge and confidence
        multiplier = 1.0 + (normalized_edge * 2.0) * (confidence / self.confidence_threshold)
        
        # Calculate stake percentage (capped at max_bet_percentage)
        stake_pct = min(self.max_bet_percentage, base_stake_percentage * multiplier)
        
        return bankroll * stake_pct
    
    def is_market_allowed(self, market: str) -> bool:
        """
        Check if a betting market is allowed by the filter.
        
        Args:
            market: Market identifier
            
        Returns:
            bool: Whether the market is allowed
        """
        if not self.market_filter:
            return True
        
        # Map market to category
        category = None
        if market in ['home', 'draw', 'away']:
            category = '1X2'
        elif market.startswith('over_') or market.startswith('under_'):
            category = 'over_under'
        elif market.startswith('btts_'):
            category = 'btts'
        elif market.startswith('cs_'):
            category = 'correct_score'
        else:
            category = market
            
        return category in self.market_filter
    
    def is_league_allowed(self, league: Optional[str]) -> bool:
        """
        Check if a league is allowed by the filter.
        
        Args:
            league: League identifier
            
        Returns:
            bool: Whether the league is allowed
        """
        if not self.league_filter or league is None:
            return True
            
        return league in self.league_filter
    
    def evaluate_bet(self, 
                   match_id: Union[str, int],
                   market: str,
                   predicted_probability: float,
                   current_odds: float,
                   closing_odds: Optional[float] = None,
                   home_team: Optional[str] = None,
                   away_team: Optional[str] = None,
                   league: Optional[str] = None,
                   bankroll: float = 1000.0,
                   date: Optional[datetime] = None,
                   confidence: float = 0.8,
                   model_name: Optional[str] = None) -> Optional[BettingStrategyResult]:
        """
        Evaluate a potential value bet.
        
        Args:
            match_id: Identifier for the match
            market: Market identifier (e.g., 'home', 'over_2.5')
            predicted_probability: Model's predicted probability
            current_odds: Current decimal odds
            closing_odds: Closing decimal odds (if available)
            home_team: Home team identifier
            away_team: Away team identifier
            league: League identifier
            bankroll: Current bankroll
            date: Match date
            confidence: Confidence score
            model_name: Name of the model used for prediction
            
        Returns:
            Optional[BettingStrategyResult]: Bet evaluation result or None if not a value bet
        """
        # Skip if league not allowed
        if not self.is_league_allowed(league):
            return None
            
        # Skip if market not allowed
        if not self.is_market_allowed(market):
            return None
            
        # Skip if odds outside acceptable range
        if current_odds < self.min_odds or current_odds > self.max_odds:
            return None
            
        # Calculate implied probability from current odds
        implied_prob = implied_probability(current_odds)
        
        # If closing odds available, blend with current odds
        effective_odds = current_odds
        if closing_odds is not None:
            closing_implied = implied_probability(closing_odds)
            blended_implied = (implied_prob * (1 - self.closing_line_factor) + 
                              closing_implied * self.closing_line_factor)
            effective_odds = 1 / blended_implied if blended_implied > 0 else current_odds
        
        # Calculate edge
        edge = self.calculate_edge(predicted_probability, effective_odds)
        
        # Skip if edge too small
        if edge < self.min_edge:
            return None
            
        # Skip if confidence too low
        if confidence < self.confidence_threshold:
            return None
            
        # Calculate expected value
        ev = self.calculate_expected_value(predicted_probability, effective_odds)
        
        # Calculate recommended stake
        if self.progressive_staking:
            recommended_stake = self.calculate_progressive_stake(
                edge=edge,
                confidence=confidence,
                base_stake_percentage=self.stake_percentage,
                bankroll=bankroll
            )
        else:
            recommended_stake = bankroll * self.stake_percentage
        
        # Calculate value rating
        value_rating = self.calculate_value_rating(edge, current_odds, confidence)
        
        # Map market to BetType
        bet_type = BetType.OTHER
        if market == 'home':
            bet_type = BetType.HOME
        elif market == 'draw':
            bet_type = BetType.DRAW
        elif market == 'away':
            bet_type = BetType.AWAY
        elif market.startswith('over_'):
            bet_type = BetType.OVER
        elif market.startswith('under_'):
            bet_type = BetType.UNDER
        elif market == 'btts_yes':
            bet_type = BetType.BTTS_YES
        elif market == 'btts_no':
            bet_type = BetType.BTTS_NO
        
        # Create result
        result = BettingStrategyResult(
            match_id=match_id,
            home_team=home_team or "",
            away_team=away_team or "",
            date=date,
            bet_type=bet_type,
            bet_description=f"{market.replace('_', ' ').title()} @ {current_odds:.2f}",
            odds=current_odds,
            predicted_probability=predicted_probability,
            implied_probability=implied_prob,
            edge=edge,
            expected_value=ev,
            recommended_stake=recommended_stake,
            potential_profit=recommended_stake * (current_odds - 1),
            confidence_score=confidence,
            strategy_name=self.name,
            model_name=model_name or self.name,
            extra={'value_rating': value_rating, 'closing_odds': closing_odds}
        )
        
        # Log the result
        self.log_result(result)
        
        return result

    @classmethod
    def from_model_predictions(cls,
                             predictions_df: pd.DataFrame,
                             odds_df: pd.DataFrame,
                             use_closing_line: bool = False,
                             league_column: Optional[str] = None,
                             bankroll: float = 1000.0,
                             **kwargs) -> List[BettingStrategyResult]:
        """
        Create bets from model predictions and odds dataframes.
        
        Args:
            predictions_df: DataFrame with columns 'match_id', market columns (e.g., 'home', 'draw', 'away')
                           containing predicted probabilities
            odds_df: DataFrame with columns 'match_id', odds columns matching prediction columns
            use_closing_line: Whether to use closing line value
            league_column: Column name for league information
            bankroll: Current bankroll
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            List[BettingStrategyResult]: List of bet evaluation results
        """
        # Initialize strategy
        strategy = cls(**kwargs)
        
        # Check required columns
        required_columns = ['match_id']
        if not all(col in predictions_df.columns for col in required_columns):
            logger.error(f"Predictions DataFrame missing required columns: {required_columns}")
            return []
        
        if not all(col in odds_df.columns for col in required_columns):
            logger.error(f"Odds DataFrame missing required columns: {required_columns}")
            return []
        
        # Merge dataframes
        merged_df = pd.merge(predictions_df, odds_df, on='match_id', suffixes=('_pred', '_odds'))
        
        # Get market columns from predictions
        market_columns = [col for col in predictions_df.columns 
                         if col not in ['match_id', 'home_team', 'away_team', 'date', 'league']]
        
        # List to store results
        results = []
        
        # Process each match
        for _, row in merged_df.iterrows():
            match_id = row['match_id']
            
            # Get home/away teams if available
            home_team = row.get('home_team_pred', row.get('home_team_odds', None))
            away_team = row.get('away_team_pred', row.get('away_team_odds', None))
            
            # Get date if available
            date = row.get('date_pred', row.get('date_odds', None))
            
            # Get league if available
            league = None
            if league_column and league_column in row:
                league = row[league_column]
            
            # Process each market
            for market in market_columns:
                pred_col = f"{market}_pred" if f"{market}_pred" in row else market
                odds_col = f"{market}_odds" if f"{market}_odds" in row else f"{market}"
                
                # Skip if market not in odds
                if odds_col not in row:
                    continue
                
                # Get prediction and odds
                pred_prob = row[pred_col]
                current_odds = row[odds_col]
                
                # Get closing odds if available
                closing_odds = None
                if use_closing_line and f"{market}_closing" in row:
                    closing_odds = row[f"{market}_closing"]
                
                # Get confidence if available
                confidence = row.get(f"{market}_confidence", 0.8)  # Default confidence if not provided
                
                # Evaluate bet
                result = strategy.evaluate_bet(
                    match_id=match_id,
                    market=market,
                    predicted_probability=pred_prob,
                    current_odds=current_odds,
                    closing_odds=closing_odds,
                    home_team=home_team,
                    away_team=away_team,
                    league=league,
                    bankroll=bankroll,
                    date=date,
                    confidence=confidence,
                    model_name="Model Predictions"
                )
                
                if result:
                    results.append(result)
        
        return results
    
    @classmethod
    def from_odds_movement(cls,
                         historical_odds_df: pd.DataFrame,
                         current_odds_df: pd.DataFrame,
                         min_movement_threshold: float = 0.1,
                         bankroll: float = 1000.0,
                         **kwargs) -> List[BettingStrategyResult]:
        """
        Create bets based on odds movement (steam betting).
        
        Args:
            historical_odds_df: DataFrame with historical odds (columns: match_id, market, odds, timestamp)
            current_odds_df: DataFrame with current odds (columns: match_id, market, odds)
            min_movement_threshold: Minimum odds movement to consider as significant
            bankroll: Current bankroll
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            List[BettingStrategyResult]: List of bet evaluation results
        """
        # Initialize strategy
        strategy = cls(**kwargs)
        
        # Verify required columns
        required_hist_cols = ['match_id', 'market', 'odds', 'timestamp']
        required_curr_cols = ['match_id', 'market', 'odds']
        
        if not all(col in historical_odds_df.columns for col in required_hist_cols):
            logger.error(f"Historical odds DataFrame missing required columns: {required_hist_cols}")
            return []
        
        if not all(col in current_odds_df.columns for col in required_curr_cols):
            logger.error(f"Current odds DataFrame missing required columns: {required_curr_cols}")
            return []
        
        # Group historical odds by match_id and market, get the earliest and latest odds
        earliest_odds = historical_odds_df.sort_values('timestamp').groupby(['match_id', 'market']).first().reset_index()
        latest_odds = historical_odds_df.sort_values('timestamp').groupby(['match_id', 'market']).last().reset_index()
        
        # Merge with current odds
        merged_df = pd.merge(
            latest_odds,
            current_odds_df,
            on=['match_id', 'market'],
            suffixes=('_hist', '_curr')
        )
        
        # Also merge with earliest odds
        merged_df = pd.merge(
            merged_df,
            earliest_odds[['match_id', 'market', 'odds']],
            on=['match_id', 'market'],
            suffixes=('', '_earliest')
        )
        
        # List to store results
        results = []
        
        # Process each row
        for _, row in merged_df.iterrows():
            match_id = row['match_id']
            market = row['market']
            earliest_odds = row['odds_earliest']
            latest_hist_odds = row['odds_hist']
            current_odds = row['odds_curr']
            
            # Calculate odds movement
            movement = latest_hist_odds / earliest_odds - 1 if earliest_odds > 0 else 0
            
            # Skip if movement below threshold
            if abs(movement) < min_movement_threshold:
                continue
            
            # Negative movement means odds are shortening (probability increasing)
            is_shortening = movement < 0
            
            # Calculate predicted probability based on odds movement trend
            # If odds are shortening, we assume market is efficient and closing odds will be lower
            if is_shortening:
                # Calculate implied probability from current odds
                current_implied = implied_probability(current_odds)
                # Adjust probability upward based on movement
                movement_factor = abs(movement) * 1.5  # Amplify the movement effect
                predicted_probability = min(0.95, current_implied * (1 + movement_factor))
            else:
                # If odds are drifting, we skip as it's likely negative value
                continue
            
            # Get home/away teams if available
            home_team = row.get('home_team', None)
            away_team = row.get('away_team', None)
            
            # Get date if available
            date = row.get('date', None)
            
            # Get league if available
            league = row.get('league', None)
            
            # Evaluate bet
            result = strategy.evaluate_bet(
                match_id=match_id,
                market=market,
                predicted_probability=predicted_probability,
                current_odds=current_odds,
                closing_odds=latest_hist_odds,
                home_team=home_team,
                away_team=away_team,
                league=league,
                bankroll=bankroll,
                date=date,
                confidence=0.7,  # Lower confidence for steam betting
                model_name="Odds Movement"
            )
            
            if result:
                results.append(result)
        
        return results
    
    def update_performance_tracking(self, 
                                  bet_result: Dict[str, Any]) -> None:
        """
        Update performance tracking metrics after bet settlement.
        
        Args:
            bet_result: Dictionary with bet result information
        """
        # Add result to history
        self.performance_history.append(bet_result)
        
        # Extract information
        market = bet_result.get('market')
        league = bet_result.get('league')
        profit = bet_result.get('profit', 0)
        stake = bet_result.get('stake', 0)
        
        # Update ROI by market
        if market:
            if market not in self.roi_by_market:
                self.roi_by_market[market] = {'profit': 0, 'stake': 0}
            self.roi_by_market[market]['profit'] += profit
            self.roi_by_market[market]['stake'] += stake
        
        # Update ROI by league
        if league:
            if league not in self.roi_by_league:
                self.roi_by_league[league] = {'profit': 0, 'stake': 0}
            self.roi_by_league[league]['profit'] += profit
            self.roi_by_league[league]['stake'] += stake
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        # Calculate overall performance
        total_bets = len(self.performance_history)
        if total_bets == 0:
            return {'total_bets': 0, 'roi': 0, 'profit': 0}
            
        total_stake = sum(bet.get('stake', 0) for bet in self.performance_history)
        total_profit = sum(bet.get('profit', 0) for bet in self.performance_history)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Calculate win rate
        wins = sum(1 for bet in self.performance_history if bet.get('profit', 0) > 0)
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Calculate ROI by market
        roi_by_market = {}
        for market, data in self.roi_by_market.items():
            roi_by_market[market] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by league
        roi_by_league = {}
        for league, data in self.roi_by_league.items():
            roi_by_league[league] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'stake': total_stake,
            'roi_by_market': roi_by_market,
            'roi_by_league': roi_by_league
        }

class DrawNoBetStrategy(BettingStrategy):
    """
    Draw No Bet (DNB) strategy that removes the draw option from betting considerations.
    
    This strategy:
    1. Converts standard 1X2 markets into Draw No Bet markets
    2. Refunds stakes when a match ends in a draw
    3. Provides better odds protection at the cost of lower potential returns
    4. Works well for teams with low draw probability but good win chances
    """
    
    def __init__(self, 
                name: str = "Draw No Bet",
                min_edge: float = 0.04,
                min_odds: float = 1.3,
                max_odds: float = 5.0,
                stake_percentage: float = 1.0,
                confidence_threshold: float = 0.65,
                min_win_probability: float = 0.4,
                max_draw_probability: float = 0.3,
                league_filter: Optional[List[str]] = None):
        """
        Initialize the Draw No Bet strategy.
        
        Args:
            name: Name of the strategy
            min_edge: Minimum edge (difference between predicted and implied probability)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            stake_percentage: Percentage of standard stake to use (0-100)
            confidence_threshold: Minimum confidence score to place a bet (0-1)
            min_win_probability: Minimum probability for team to win outright
            max_draw_probability: Maximum probability of a draw to consider the bet
            league_filter: List of leagues to consider
        """
        super().__init__(
            name=name,
            min_edge=min_edge,
            min_odds=min_odds,
            max_odds=max_odds,
            stake_percentage=stake_percentage,
            confidence_threshold=confidence_threshold
        )
        self.min_win_probability = min_win_probability
        self.max_draw_probability = max_draw_probability
        self.league_filter = league_filter
        
        # Performance tracking
        self.performance_history = []
        self.roi_by_league = {}
        self.roi_by_team = {}
    
    def is_league_allowed(self, league: Optional[str]) -> bool:
        """
        Check if a league is allowed by the filter.
        
        Args:
            league: League identifier
            
        Returns:
            bool: Whether the league is allowed
        """
        if not self.league_filter or league is None:
            return True
            
        return league in self.league_filter
    
    def calculate_dnb_odds(self, 
                         win_odds: float, 
                         draw_odds: float) -> float:
        """
        Calculate Draw No Bet odds from 1X2 market odds.
        
        Formula: DNB odds = (win_odds * draw_odds) / (win_odds + draw_odds - 1)
        
        Args:
            win_odds: Decimal odds for team to win
            draw_odds: Decimal odds for a draw
            
        Returns:
            float: Calculated DNB odds
        """
        try:
            # Standard formula for converting 1X2 to Draw No Bet
            denominator = win_odds + draw_odds - 1
            if denominator <= 0:
                return win_odds  # Fallback to win odds if calculation fails
                
            dnb_odds = (win_odds * draw_odds) / denominator
            return dnb_odds
        except (ZeroDivisionError, TypeError):
            return win_odds  # Fallback to win odds if calculation fails
    
    def evaluate_bet(self, 
                   match_id: Union[str, int],
                   home_team: Any,
                   away_team: Any,
                   odds: Dict[str, float],
                   predictions: Dict[str, float],
                   bankroll: float = 1000.0,
                   date: Optional[datetime] = None,
                   league: Optional[str] = None,
                   confidence: Optional[Dict[str, float]] = None,
                   model_name: Optional[str] = None) -> Optional[BettingStrategyResult]:
        """
        Evaluate a potential Draw No Bet opportunity.
        
        Args:
            match_id: Identifier for the match
            home_team: Home team identifier
            away_team: Away team identifier
            odds: Dictionary of odds (e.g., {'home': 2.0, 'draw': 3.2, 'away': 4.5})
            predictions: Dictionary of predicted probabilities (e.g., {'home': 0.55, 'draw': 0.25, 'away': 0.2})
            bankroll: Current bankroll
            date: Match date
            league: League identifier
            confidence: Dictionary of confidence scores
            model_name: Name of the model used for prediction
            
        Returns:
            Optional[BettingStrategyResult]: Bet evaluation result or None if no bet recommended
        """
        # Skip if league not allowed
        if not self.is_league_allowed(league):
            return None
            
        # Check required data is available
        if not all(k in odds for k in ['home', 'draw', 'away']):
            logger.warning(f"Missing required odds for match {match_id}")
            return None
            
        if not all(k in predictions for k in ['home', 'draw', 'away']):
            logger.warning(f"Missing required predictions for match {match_id}")
            return None
        
        # Get confidence scores if available
        home_confidence = confidence.get('home', 0.7) if confidence else 0.7
        away_confidence = confidence.get('away', 0.7) if confidence else 0.7
        
        # Check home team opportunity
        home_win_prob = predictions['home']
        draw_prob = predictions['draw']
        
        if (home_win_prob >= self.min_win_probability and 
            draw_prob <= self.max_draw_probability and
            home_confidence >= self.confidence_threshold):
            
            # Calculate DNB odds
            home_dnb_odds = self.calculate_dnb_odds(odds['home'], odds['draw'])
            
            # Calculate adjusted probability (remove draw)
            adjusted_home_prob = home_win_prob / (home_win_prob + predictions['away'])
            
            # Calculate edge
            home_implied_prob = implied_probability(home_dnb_odds)
            home_edge = adjusted_home_prob - home_implied_prob
            
            # Check if meets edge requirement
            if (home_edge >= self.min_edge and 
                home_dnb_odds >= self.min_odds and 
                home_dnb_odds <= self.max_odds):
                
                # Calculate expected value
                home_ev = self.calculate_expected_value(adjusted_home_prob, home_dnb_odds)
                
                # Calculate recommended stake
                home_stake = bankroll * self.stake_percentage
                
                # Create result for home DNB
                home_result = BettingStrategyResult(
                    match_id=match_id,
                    home_team=home_team,
                    away_team=away_team,
                    date=date,
                    bet_type=BetType.HOME,
                    bet_description=f"Home DNB: {home_team} vs {away_team} @ {home_dnb_odds:.2f}",
                    odds=home_dnb_odds,
                    predicted_probability=adjusted_home_prob,
                    implied_probability=home_implied_prob,
                    edge=home_edge,
                    expected_value=home_ev,
                    recommended_stake=home_stake,
                    potential_profit=home_stake * (home_dnb_odds - 1),
                    confidence_score=home_confidence,
                    strategy_name=f"{self.name} (Home)",
                    model_name=model_name or self.name,
                    extra={
                        'original_win_odds': odds['home'],
                        'draw_odds': odds['draw'],
                        'original_win_prob': home_win_prob,
                        'draw_prob': draw_prob,
                        'team': home_team
                    }
                )
                
                # Log the result
                self.log_result(home_result)
                return home_result
        
        # Check away team opportunity
        away_win_prob = predictions['away']
        
        if (away_win_prob >= self.min_win_probability and 
            draw_prob <= self.max_draw_probability and
            away_confidence >= self.confidence_threshold):
            
            # Calculate DNB odds
            away_dnb_odds = self.calculate_dnb_odds(odds['away'], odds['draw'])
            
            # Calculate adjusted probability (remove draw)
            adjusted_away_prob = away_win_prob / (predictions['home'] + away_win_prob)
            
            # Calculate edge
            away_implied_prob = implied_probability(away_dnb_odds)
            away_edge = adjusted_away_prob - away_implied_prob
            
            # Check if meets edge requirement
            if (away_edge >= self.min_edge and 
                away_dnb_odds >= self.min_odds and 
                away_dnb_odds <= self.max_odds):
                
                # Calculate expected value
                away_ev = self.calculate_expected_value(adjusted_away_prob, away_dnb_odds)
                
                # Calculate recommended stake
                away_stake = bankroll * self.stake_percentage
                
                # Create result for away DNB
                away_result = BettingStrategyResult(
                    match_id=match_id,
                    home_team=home_team,
                    away_team=away_team,
                    date=date,
                    bet_type=BetType.AWAY,
                    bet_description=f"Away DNB: {away_team} @ {home_team} @ {away_dnb_odds:.2f}",
                    odds=away_dnb_odds,
                    predicted_probability=adjusted_away_prob,
                    implied_probability=away_implied_prob,
                    edge=away_edge,
                    expected_value=away_ev,
                    recommended_stake=away_stake,
                    potential_profit=away_stake * (away_dnb_odds - 1),
                    confidence_score=away_confidence,
                    strategy_name=f"{self.name} (Away)",
                    model_name=model_name or self.name,
                    extra={
                        'original_win_odds': odds['away'],
                        'draw_odds': odds['draw'],
                        'original_win_prob': away_win_prob,
                        'draw_prob': draw_prob,
                        'team': away_team
                    }
                )
                
                # Log the result
                self.log_result(away_result)
                return away_result
        
        # No bet recommendation
        return None
    
    @classmethod
    def from_model_predictions(cls,
                            predictions_df: pd.DataFrame,
                            odds_df: pd.DataFrame,
                            league_column: Optional[str] = None,
                            bankroll: float = 1000.0,
                            confidence_column_prefix: Optional[str] = None,
                            **kwargs) -> List[BettingStrategyResult]:
        """
        Create bets from model predictions and odds dataframes.
        
        Args:
            predictions_df: DataFrame with columns 'match_id', 'home', 'draw', 'away'
                          containing predicted probabilities
            odds_df: DataFrame with columns 'match_id', 'home', 'draw', 'away'
                    containing decimal odds
            league_column: Column name for league information
            bankroll: Current bankroll
            confidence_column_prefix: Prefix for confidence columns (e.g., 'confidence_')
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            List[BettingStrategyResult]: List of bet evaluation results
        """
        # Initialize strategy
        strategy = cls(**kwargs)
        
        # Check required columns
        required_pred_cols = ['match_id', 'home', 'draw', 'away']
        required_odds_cols = ['match_id', 'home', 'draw', 'away']
        
        if not all(col in predictions_df.columns for col in required_pred_cols):
            logger.error(f"Predictions DataFrame missing required columns: {required_pred_cols}")
            return []
        
        if not all(col in odds_df.columns for col in required_odds_cols):
            logger.error(f"Odds DataFrame missing required columns: {required_odds_cols}")
            return []
        
        # Merge dataframes
        merged_df = pd.merge(predictions_df, odds_df, on='match_id', suffixes=('_pred', '_odds'))
        
        # List to store results
        results = []
        
        # Process each match
        for _, row in merged_df.iterrows():
            match_id = row['match_id']
            
            # Get home/away teams if available
            home_team = row.get('home_team_pred', row.get('home_team_odds', f"Home Team {match_id}"))
            away_team = row.get('away_team_pred', row.get('away_team_odds', f"Away Team {match_id}"))
            
            # Get date if available
            date = row.get('date_pred', row.get('date_odds', None))
            
            # Get league if available
            league = None
            if league_column and league_column in row:
                league = row[league_column]
            
            # Get predictions
            predictions = {
                'home': row['home_pred'],
                'draw': row['draw_pred'],
                'away': row['away_pred']
            }
            
            # Get odds
            odds = {
                'home': row['home_odds'],
                'draw': row['draw_odds'],
                'away': row['away_odds']
            }
            
            # Get confidence if available
            confidence = None
            if confidence_column_prefix:
                confidence = {
                    'home': row.get(f"{confidence_column_prefix}home", 0.7),
                    'draw': row.get(f"{confidence_column_prefix}draw", 0.7),
                    'away': row.get(f"{confidence_column_prefix}away", 0.7)
                }
            
            # Evaluate bet
            result = strategy.evaluate_bet(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                odds=odds,
                predictions=predictions,
                bankroll=bankroll,
                date=date,
                league=league,
                confidence=confidence,
                model_name="Model Predictions"
            )
            
            if result:
                results.append(result)
        
        return results
    
    def update_performance_tracking(self, 
                                  bet_result: Dict[str, Any]) -> None:
        """
        Update performance tracking metrics after bet settlement.
        
        Args:
            bet_result: Dictionary with bet result information
        """
        # Add result to history
        self.performance_history.append(bet_result)
        
        # Extract information
        league = bet_result.get('league')
        team = bet_result.get('extra', {}).get('team')
        profit = bet_result.get('profit', 0)
        stake = bet_result.get('stake', 0)
        
        # Update ROI by league
        if league:
            if league not in self.roi_by_league:
                self.roi_by_league[league] = {'profit': 0, 'stake': 0}
            self.roi_by_league[league]['profit'] += profit
            self.roi_by_league[league]['stake'] += stake
        
        # Update ROI by team
        if team:
            if team not in self.roi_by_team:
                self.roi_by_team[team] = {'profit': 0, 'stake': 0}
            self.roi_by_team[team]['profit'] += profit
            self.roi_by_team[team]['stake'] += stake
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        # Calculate overall performance
        total_bets = len(self.performance_history)
        if total_bets == 0:
            return {'total_bets': 0, 'roi': 0, 'profit': 0}
            
        total_stake = sum(bet.get('stake', 0) for bet in self.performance_history)
        total_profit = sum(bet.get('profit', 0) for bet in self.performance_history)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Calculate win rate (excluding draws which are refunded)
        wins = sum(1 for bet in self.performance_history 
                  if bet.get('profit', 0) > 0 and bet.get('result') != 'draw')
        losses = sum(1 for bet in self.performance_history 
                    if bet.get('profit', 0) < 0 and bet.get('result') != 'draw')
        draws = sum(1 for bet in self.performance_history 
                   if bet.get('result') == 'draw')
        
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Calculate ROI by league
        roi_by_league = {}
        for league, data in self.roi_by_league.items():
            roi_by_league[league] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by team
        roi_by_team = {}
        for team, data in self.roi_by_team.items():
            roi_by_team[team] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'stake': total_stake,
            'roi_by_league': roi_by_league,
            'roi_by_team': roi_by_team
        }

class AsianHandicapStrategy(BettingStrategy):
    """
    Asian Handicap betting strategy that provides a handicap advantage or disadvantage to teams.
    
    This strategy:
    1. Evaluates Asian Handicap markets where teams are given virtual head starts/deficits
    2. Provides better odds than standard 1X2 markets by eliminating draw possibility
    3. Offers half-win/half-loss scenarios for half-handicaps
    4. Works well for matches with clear favorites but uncertainty about margin of victory
    """
    
    def __init__(self, 
                name: str = "Asian Handicap",
                min_edge: float = 0.04,
                min_odds: float = 1.7,
                max_odds: float = 3.0,
                stake_percentage: float = 1.0,
                confidence_threshold: float = 0.7,
                handicap_ranges: Optional[Dict[str, List[float]]] = None,
                min_ev: float = 0.05,
                league_filter: Optional[List[str]] = None,
                team_strength_adjustment: bool = True):
        """
        Initialize the Asian Handicap strategy.
        
        Args:
            name: Name of the strategy
            min_edge: Minimum edge (difference between predicted and implied probability)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            stake_percentage: Percentage of standard stake to use (0-100)
            confidence_threshold: Minimum confidence score to place a bet (0-1)
            handicap_ranges: Dictionary mapping league names to lists of handicap values to consider
            min_ev: Minimum expected value to place a bet
            league_filter: List of leagues to consider
            team_strength_adjustment: Whether to adjust handicap selection based on team strength
        """
        super().__init__(
            name=name,
            min_edge=min_edge,
            min_odds=min_odds,
            max_odds=max_odds,
            stake_percentage=stake_percentage,
            confidence_threshold=confidence_threshold
        )
        # Default handicap ranges if none provided
        self.handicap_ranges = handicap_ranges or {
            'default': [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5]
        }
        self.min_ev = min_ev
        self.league_filter = league_filter
        self.team_strength_adjustment = team_strength_adjustment
        
        # Performance tracking
        self.performance_history = []
        self.roi_by_league = {}
        self.roi_by_handicap = {}
    
    def is_league_allowed(self, league: Optional[str]) -> bool:
        """
        Check if a league is allowed by the filter.
        
        Args:
            league: League identifier
            
        Returns:
            bool: Whether the league is allowed
        """
        if not self.league_filter or league is None:
            return True
            
        return league in self.league_filter
    
    def get_handicap_range(self, league: Optional[str]) -> List[float]:
        """
        Get the appropriate handicap range for a league.
        
        Args:
            league: League identifier
            
        Returns:
            List[float]: List of handicap values to consider
        """
        if league in self.handicap_ranges:
            return self.handicap_ranges[league]
        else:
            return self.handicap_ranges['default']
    
    def adjust_handicap_range(self, 
                           home_strength: float, 
                           away_strength: float,
                           handicap_range: List[float]) -> List[float]:
        """
        Adjust handicap range based on team strength difference.
        
        Args:
            home_strength: Home team strength score (0-1)
            away_strength: Away team strength score (0-1)
            handicap_range: Base handicap range
            
        Returns:
            List[float]: Adjusted handicap range
        """
        if not self.team_strength_adjustment:
            return handicap_range
            
        # Calculate strength difference
        diff = home_strength - away_strength
        
        # Bias the handicap range based on difference
        if diff > 0.3:  # Home team much stronger
            return [h for h in handicap_range if h <= 0]  # Away team needs handicap
        elif diff < -0.3:  # Away team much stronger
            return [h for h in handicap_range if h >= 0]  # Home team needs handicap
        else:
            return handicap_range  # Keep full range
    
    def calculate_handicap_win_probability(self,
                                        expected_home_goals: float,
                                        expected_away_goals: float,
                                        handicap: float) -> Tuple[float, float]:
        """
        Calculate win probabilities for a given Asian Handicap.
        
        Args:
            expected_home_goals: Expected goals for home team
            expected_away_goals: Expected goals for away team
            handicap: Asian Handicap line (positive favors away team)
            
        Returns:
            Tuple[float, float]: (Home win probability, Away win probability)
        """
        # Adjust goal expectations based on handicap
        adjusted_home_goals = expected_home_goals
        adjusted_away_goals = expected_away_goals + handicap
        
        # Goal difference distribution parameters
        mean_diff = adjusted_home_goals - adjusted_away_goals
        # Variance of difference is sum of individual variances
        # Assume variance of goals follows Poisson with mean = variance
        var_diff = expected_home_goals + expected_away_goals
        std_diff = np.sqrt(var_diff)
        
        # Use normal approximation for goal difference
        if handicap % 1 == 0:  # Whole number handicap
            # Home wins if diff > 0, away wins if diff < 0, draw if diff = 0
            home_win_prob = 1 - norm.cdf(0, mean_diff, std_diff)
            away_win_prob = norm.cdf(0, mean_diff, std_diff)
            return home_win_prob, away_win_prob
        else:  # Half number handicap (no draw possibility)
            split_point = 0
            home_win_prob = 1 - norm.cdf(split_point, mean_diff, std_diff)
            away_win_prob = 1 - home_win_prob
            return home_win_prob, away_win_prob
    
    def evaluate_bet(self, 
                    match_id: Union[str, int],
                    home_team: Any,
                    away_team: Any,
                    expected_goals: Dict[str, float],
                    handicap_odds: Dict[float, Dict[str, float]],
                    bankroll: float = 1000.0,
                    date: Optional[datetime] = None,
                    league: Optional[str] = None,
                    home_strength: float = 0.5,
                    away_strength: float = 0.5,
                    confidence: float = 0.7,
                    model_name: Optional[str] = None) -> List[BettingStrategyResult]:
        """
        Evaluate potential Asian Handicap betting opportunities.
        
        Args:
            match_id: Identifier for the match
            home_team: Home team identifier
            away_team: Away team identifier
            expected_goals: Dictionary with expected goals (e.g., {'home': 1.5, 'away': 1.2})
            handicap_odds: Dictionary mapping handicaps to dictionaries of home/away odds
                          e.g., {0.5: {'home': 1.8, 'away': 2.1}, -0.5: {'home': 2.2, 'away': 1.7}}
            bankroll: Current bankroll
            date: Match date
            league: League identifier
            home_strength: Home team strength score (0-1)
            away_strength: Away team strength score (0-1)
            confidence: Confidence score for predictions
            model_name: Name of the model used for predictions
            
        Returns:
            List[BettingStrategyResult]: List of bet evaluation results
        """
        # Skip if league not allowed
        if not self.is_league_allowed(league):
            return []
            
        # Check required data is available
        if not all(k in expected_goals for k in ['home', 'away']):
            logger.warning(f"Missing expected goals for match {match_id}")
            return []
            
        if not handicap_odds:
            logger.warning(f"No handicap odds provided for match {match_id}")
            return []
        
        # Get expected goals
        home_xg = expected_goals['home']
        away_xg = expected_goals['away']
        
        # Get handicap range for this league
        handicap_range = self.get_handicap_range(league)
        
        # Adjust range based on team strength if enabled
        handicap_range = self.adjust_handicap_range(home_strength, away_strength, handicap_range)
        
        # Evaluate each handicap
        results = []
        
        for handicap in handicap_range:
            # Skip if odds not available for this handicap
            if handicap not in handicap_odds:
                continue
                
            # Get odds for this handicap
            handicap_odds_dict = handicap_odds[handicap]
            
            if not all(k in handicap_odds_dict for k in ['home', 'away']):
                continue
                
            # Calculate win probabilities for this handicap
            home_win_prob, away_win_prob = self.calculate_handicap_win_probability(
                expected_home_goals=home_xg,
                expected_away_goals=away_xg,
                handicap=handicap
            )
            
            # Check home team opportunity
            home_odds = handicap_odds_dict['home']
            if (home_odds >= self.min_odds and home_odds <= self.max_odds):
                # Calculate edge
                home_implied_prob = implied_probability(home_odds)
                home_edge = home_win_prob - home_implied_prob
                
                # Check if meets edge requirement
                if home_edge >= self.min_edge and confidence >= self.confidence_threshold:
                    # Calculate expected value
                    home_ev = self.calculate_expected_value(home_win_prob, home_odds)
                    
                    # Skip if EV too low
                    if home_ev < self.min_ev:
                        continue
                        
                    # Calculate recommended stake
                    home_stake = bankroll * self.stake_percentage
                    
                    # Format handicap for display
                    handicap_display = f"+{handicap}" if handicap > 0 else str(handicap)
                    
                    # Create result for home Asian Handicap
                    home_result = BettingStrategyResult(
                        match_id=match_id,
                        home_team=home_team,
                        away_team=away_team,
                        date=date,
                        bet_type=BetType.HOME_AH,
                        bet_description=f"Home AH{handicap_display}: {home_team} vs {away_team} @ {home_odds:.2f}",
                        odds=home_odds,
                        predicted_probability=home_win_prob,
                        implied_probability=home_implied_prob,
                        edge=home_edge,
                        expected_value=home_ev,
                        recommended_stake=home_stake,
                        potential_profit=home_stake * (home_odds - 1),
                        confidence_score=confidence,
                        strategy_name=f"{self.name} (Home {handicap_display})",
                        model_name=model_name or self.name,
                        extra={
                            'handicap': handicap,
                            'expected_home_goals': home_xg,
                            'expected_away_goals': away_xg,
                            'home_strength': home_strength,
                            'away_strength': away_strength
                        }
                    )
                    
                    # Log the result
                    self.log_result(home_result)
                    results.append(home_result)
            
            # Check away team opportunity
            away_odds = handicap_odds_dict['away']
            if (away_odds >= self.min_odds and away_odds <= self.max_odds):
                # Calculate edge
                away_implied_prob = implied_probability(away_odds)
                away_edge = away_win_prob - away_implied_prob
                
                # Check if meets edge requirement
                if away_edge >= self.min_edge and confidence >= self.confidence_threshold:
                    # Calculate expected value
                    away_ev = self.calculate_expected_value(away_win_prob, away_odds)
                    
                    # Skip if EV too low
                    if away_ev < self.min_ev:
                        continue
                        
                    # Calculate recommended stake
                    away_stake = bankroll * self.stake_percentage
                    
                    # Format handicap for display
                    handicap_display = f"{handicap}" if handicap > 0 else f"{-handicap}"
                    
                    # Create result for away Asian Handicap
                    away_result = BettingStrategyResult(
                        match_id=match_id,
                        home_team=home_team,
                        away_team=away_team,
                        date=date,
                        bet_type=BetType.AWAY_AH,
                        bet_description=f"Away AH{handicap_display}: {away_team} @ {home_team} @ {away_odds:.2f}",
                        odds=away_odds,
                        predicted_probability=away_win_prob,
                        implied_probability=away_implied_prob,
                        edge=away_edge,
                        expected_value=away_ev,
                        recommended_stake=away_stake,
                        potential_profit=away_stake * (away_odds - 1),
                        confidence_score=confidence,
                        strategy_name=f"{self.name} (Away {handicap_display})",
                        model_name=model_name or self.name,
                        extra={
                            'handicap': handicap,
                            'expected_home_goals': home_xg,
                            'expected_away_goals': away_xg,
                            'home_strength': home_strength,
                            'away_strength': away_strength
                        }
                    )
                    
                    # Log the result
                    self.log_result(away_result)
                    results.append(away_result)
        
        return results
    
    @classmethod
    def from_model_predictions(cls,
                            predictions_df: pd.DataFrame,
                            handicap_odds_df: pd.DataFrame,
                            league_column: Optional[str] = None,
                            strength_columns: Optional[Tuple[str, str]] = None,
                            bankroll: float = 1000.0,
                            confidence_column: Optional[str] = None,
                            **kwargs) -> List[BettingStrategyResult]:
        """
        Create bets from model predictions and handicap odds dataframes.
        
        Args:
            predictions_df: DataFrame with columns 'match_id', 'home_xg', 'away_xg'
                          containing expected goals
            handicap_odds_df: DataFrame with columns 'match_id', 'handicap', 'home_odds', 'away_odds'
            league_column: Column name for league information
            strength_columns: Tuple of column names for (home_strength, away_strength)
            bankroll: Current bankroll
            confidence_column: Column name for confidence score
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            List[BettingStrategyResult]: List of bet evaluation results
        """
        # Initialize strategy
        strategy = cls(**kwargs)
        
        # Check required columns
        required_pred_cols = ['match_id', 'home_xg', 'away_xg']
        required_odds_cols = ['match_id', 'handicap', 'home_odds', 'away_odds']
        
        if not all(col in predictions_df.columns for col in required_pred_cols):
            logger.error(f"Predictions DataFrame missing required columns: {required_pred_cols}")
            return []
        
        if not all(col in handicap_odds_df.columns for col in required_odds_cols):
            logger.error(f"Handicap odds DataFrame missing required columns: {required_odds_cols}")
            return []
        
        # Merge predictions with unique matches
        unique_matches = predictions_df[['match_id', 'home_xg', 'away_xg']].drop_duplicates()
        
        if league_column and league_column in predictions_df.columns:
            unique_matches[league_column] = predictions_df[league_column]
        
        if confidence_column and confidence_column in predictions_df.columns:
            unique_matches[confidence_column] = predictions_df[confidence_column]
        
        if strength_columns and all(col in predictions_df.columns for col in strength_columns):
            unique_matches[strength_columns[0]] = predictions_df[strength_columns[0]]
            unique_matches[strength_columns[1]] = predictions_df[strength_columns[1]]
        
        # Get home/away teams if available
        if 'home_team' in predictions_df.columns and 'away_team' in predictions_df.columns:
            unique_matches['home_team'] = predictions_df['home_team']
            unique_matches['away_team'] = predictions_df['away_team']
        
        if 'date' in predictions_df.columns:
            unique_matches['date'] = predictions_df['date']
        
        # Group handicap odds by match
        grouped_odds = handicap_odds_df.groupby('match_id')
        
        # List to store results
        results = []
        
        # Process each match
        for _, row in unique_matches.iterrows():
            match_id = row['match_id']
            
            # Skip if no handicap odds for this match
            if match_id not in grouped_odds.groups:
                continue
            
            # Get match odds
            match_odds = grouped_odds.get_group(match_id)
            
            # Convert to dictionary format
            handicap_odds = {}
            for _, odds_row in match_odds.iterrows():
                handicap = odds_row['handicap']
                handicap_odds[handicap] = {
                    'home': odds_row['home_odds'],
                    'away': odds_row['away_odds']
                }
            
            # Get expected goals
            expected_goals = {
                'home': row['home_xg'],
                'away': row['away_xg']
            }
            
            # Get home/away teams if available
            home_team = row.get('home_team', f"Home Team {match_id}")
            away_team = row.get('away_team', f"Away Team {match_id}")
            
            # Get date if available
            date = row.get('date', None)
            
            # Get league if available
            league = None
            if league_column and league_column in row:
                league = row[league_column]
            
            # Get team strengths if available
            home_strength = 0.5
            away_strength = 0.5
            if strength_columns and all(col in row for col in strength_columns):
                home_strength = row[strength_columns[0]]
                away_strength = row[strength_columns[1]]
            
            # Get confidence if available
            confidence = 0.7
            if confidence_column and confidence_column in row:
                confidence = row[confidence_column]
            
            # Evaluate bets
            match_results = strategy.evaluate_bet(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                expected_goals=expected_goals,
                handicap_odds=handicap_odds,
                bankroll=bankroll,
                date=date,
                league=league,
                home_strength=home_strength,
                away_strength=away_strength,
                confidence=confidence,
                model_name="Model Predictions"
            )
            
            results.extend(match_results)
        
        return results
    
    def update_performance_tracking(self, 
                                  bet_result: Dict[str, Any]) -> None:
        """
        Update performance tracking metrics after bet settlement.
        
        Args:
            bet_result: Dictionary with bet result information
        """
        # Add result to history
        self.performance_history.append(bet_result)
        
        # Extract information
        league = bet_result.get('league')
        handicap = bet_result.get('extra', {}).get('handicap')
        profit = bet_result.get('profit', 0)
        stake = bet_result.get('stake', 0)
        
        # Update ROI by league
        if league:
            if league not in self.roi_by_league:
                self.roi_by_league[league] = {'profit': 0, 'stake': 0}
            self.roi_by_league[league]['profit'] += profit
            self.roi_by_league[league]['stake'] += stake
        
        # Update ROI by handicap
        if handicap is not None:
            handicap_key = str(handicap)
            if handicap_key not in self.roi_by_handicap:
                self.roi_by_handicap[handicap_key] = {'profit': 0, 'stake': 0}
            self.roi_by_handicap[handicap_key]['profit'] += profit
            self.roi_by_handicap[handicap_key]['stake'] += stake
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        # Calculate overall performance
        total_bets = len(self.performance_history)
        if total_bets == 0:
            return {'total_bets': 0, 'roi': 0, 'profit': 0}
            
        total_stake = sum(bet.get('stake', 0) for bet in self.performance_history)
        total_profit = sum(bet.get('profit', 0) for bet in self.performance_history)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Calculate win rate
        wins = sum(1 for bet in self.performance_history if bet.get('profit', 0) > 0)
        losses = sum(1 for bet in self.performance_history if bet.get('profit', 0) < 0)
        half_wins = sum(1 for bet in self.performance_history 
                       if bet.get('profit', 0) > 0 and bet.get('half_win', False))
        half_losses = sum(1 for bet in self.performance_history 
                         if bet.get('profit', 0) < 0 and bet.get('half_loss', False))
        
        win_rate = (wins + (half_wins * 0.5)) / total_bets if total_bets > 0 else 0
        
        # Calculate ROI by league
        roi_by_league = {}
        for league, data in self.roi_by_league.items():
            roi_by_league[league] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by handicap
        roi_by_handicap = {}
        for handicap, data in self.roi_by_handicap.items():
            roi_by_handicap[handicap] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Sort handicaps by ROI
        best_handicaps = sorted(
            [(h, r) for h, r in roi_by_handicap.items() if self.roi_by_handicap[h]['stake'] > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'half_wins': half_wins,
            'half_losses': half_losses,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'stake': total_stake,
            'roi_by_league': roi_by_league,
            'roi_by_handicap': roi_by_handicap,
            'best_handicaps': best_handicaps[:5] if best_handicaps else []
        } 

class ModelEnsembleStrategy(BettingStrategy):
    """
    Model Ensemble Strategy that combines predictions from multiple models.
    
    This strategy:
    1. Combines predictions from multiple models with different weighting schemes
    2. Improves prediction accuracy by leveraging diverse model strengths
    3. Reduces variance in predictions compared to single models
    4. Dynamically adjusts model weights based on recent performance
    """
    
    def __init__(self, 
                name: str = "Model Ensemble",
                min_edge: float = 0.04,
                min_odds: float = 1.5,
                max_odds: float = 7.0,
                stake_percentage: float = 1.0,
                confidence_threshold: float = 0.7,
                models: List[Dict[str, Any]] = None,
                weighting_method: str = "equal",
                market_filter: Optional[List[str]] = None,
                league_filter: Optional[List[str]] = None,
                adaptive_weights: bool = False,
                weight_update_frequency: int = 50):
        """
        Initialize the Model Ensemble strategy.
        
        Args:
            name: Name of the strategy
            min_edge: Minimum edge (difference between predicted and implied probability)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            stake_percentage: Percentage of standard stake to use (0-100)
            confidence_threshold: Minimum confidence score to place a bet (0-1)
            models: List of model dictionaries, each containing 'name', 'weight', and optional 'league_weights'
            weighting_method: Method to combine model predictions ('equal', 'weighted', 'performance')
            market_filter: List of markets to consider
            league_filter: List of leagues to consider
            adaptive_weights: Whether to adapt weights based on model performance
            weight_update_frequency: Number of bets after which to update weights
        """
        super().__init__(
            name=name,
            min_edge=min_edge,
            min_odds=min_odds,
            max_odds=max_odds,
            stake_percentage=stake_percentage,
            confidence_threshold=confidence_threshold
        )
        # Default to equal weights if no models provided
        self.models = models or [{'name': 'default', 'weight': 1.0}]
        self.weighting_method = weighting_method
        self.market_filter = market_filter or ['1X2', 'over_under', 'btts']
        self.league_filter = league_filter
        self.adaptive_weights = adaptive_weights
        self.weight_update_frequency = weight_update_frequency
        
        # Normalize weights
        self._normalize_weights()
        
        # Performance tracking
        self.performance_history = []
        self.model_performance = {model['name']: {'correct': 0, 'incorrect': 0} for model in self.models}
        self.roi_by_market = {}
        self.roi_by_league = {}
        self.roi_by_model = {model['name']: {'profit': 0, 'stake': 0} for model in self.models}
        self.bets_since_update = 0
    
    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1.0"""
        total_weight = sum(model['weight'] for model in self.models)
        if total_weight > 0:
            for model in self.models:
                model['weight'] = model['weight'] / total_weight
    
    def is_market_allowed(self, market: str) -> bool:
        """
        Check if a betting market is allowed by the filter.
        
        Args:
            market: Market identifier
            
        Returns:
            bool: Whether the market is allowed
        """
        if not self.market_filter:
            return True
        
        # Map market to category
        category = None
        if market in ['home', 'draw', 'away']:
            category = '1X2'
        elif market.startswith('over_') or market.startswith('under_'):
            category = 'over_under'
        elif market.startswith('btts_'):
            category = 'btts'
        elif market.startswith('cs_'):
            category = 'correct_score'
        else:
            category = market
            
        return category in self.market_filter
    
    def is_league_allowed(self, league: Optional[str]) -> bool:
        """
        Check if a league is allowed by the filter.
        
        Args:
            league: League identifier
            
        Returns:
            bool: Whether the league is allowed
        """
        if not self.league_filter or league is None:
            return True
            
        return league in self.league_filter
    
    def get_model_weight(self, model_name: str, league: Optional[str] = None) -> float:
        """
        Get the weight for a specific model, optionally adjusting for league.
        
        Args:
            model_name: Name of the model
            league: League identifier
            
        Returns:
            float: Model weight
        """
        # Find the model
        for model in self.models:
            if model['name'] == model_name:
                # If league-specific weights exist and this league is included
                if league and 'league_weights' in model and league in model['league_weights']:
                    return model['weight'] * model['league_weights'][league]
                else:
                    return model['weight']
        
        # Default to zero weight if model not found
        return 0.0
    
    def combine_predictions(self, 
                          model_predictions: Dict[str, Dict[str, float]], 
                          league: Optional[str] = None) -> Tuple[Dict[str, float], float]:
        """
        Combine predictions from multiple models.
        
        Args:
            model_predictions: Dictionary mapping model names to prediction dictionaries
            league: League identifier
            
        Returns:
            Tuple[Dict[str, float], float]: Combined predictions and confidence score
        """
        if not model_predictions:
            return {}, 0.0
        
        # Initialize combined predictions
        combined = {}
        total_weight = 0.0
        
        # Equal weighting
        if self.weighting_method == "equal":
            for model_name, predictions in model_predictions.items():
                model_weight = 1.0 / len(model_predictions)
                total_weight += model_weight
                
                for market, prob in predictions.items():
                    if market not in combined:
                        combined[market] = 0.0
                    combined[market] += prob * model_weight
        
        # Weighted average
        elif self.weighting_method in ["weighted", "performance"]:
            for model_name, predictions in model_predictions.items():
                model_weight = self.get_model_weight(model_name, league)
                
                if model_weight > 0:
                    total_weight += model_weight
                    
                    for market, prob in predictions.items():
                        if market not in combined:
                            combined[market] = 0.0
                        combined[market] += prob * model_weight
        
        # Normalize if needed
        if total_weight > 0 and total_weight != 1.0:
            for market in combined:
                combined[market] /= total_weight
        
        # Calculate confidence based on model agreement
        confidence = 0.7  # Default confidence value
        
        if len(model_predictions) > 1:
            # Calculate variance of predictions as a measure of agreement
            variances = {}
            for market in combined:
                predictions = [preds.get(market, 0) for preds in model_predictions.values()]
                if len(predictions) > 1:
                    variances[market] = np.var(predictions)
            
            # Average variance across all markets
            if variances:
                avg_variance = sum(variances.values()) / len(variances)
                # Convert to confidence: low variance = high confidence
                confidence = max(0.5, min(0.95, 1.0 - avg_variance))
        
        return combined, confidence
    
    def update_model_weights(self) -> None:
        """Update model weights based on recent performance."""
        if not self.adaptive_weights or self.bets_since_update < self.weight_update_frequency:
            return
        
        # Reset counter
        self.bets_since_update = 0
        
        # Calculate performance scores for each model
        scores = {}
        for model_name, perf in self.model_performance.items():
            total = perf['correct'] + perf['incorrect']
            if total > 0:
                scores[model_name] = perf['correct'] / total
            else:
                scores[model_name] = 0.5  # Default score
        
        # Update weights
        for model in self.models:
            model_name = model['name']
            if model_name in scores:
                # Adjust weight based on performance score
                model['weight'] = 0.3 + (scores[model_name] * 0.7)  # Scale between 0.3 and 1.0
        
        # Normalize weights
        self._normalize_weights()
        
        # Log weight update
        logger.info(f"Updated model weights: {[(m['name'], m['weight']) for m in self.models]}")
    
    def evaluate_bet(self, 
                   match_id: Union[str, int],
                   market: str,
                   model_predictions: Dict[str, Dict[str, float]],
                   current_odds: float,
                   home_team: Optional[str] = None,
                   away_team: Optional[str] = None,
                   league: Optional[str] = None,
                   bankroll: float = 1000.0,
                   date: Optional[datetime] = None,
                   model_confidences: Optional[Dict[str, float]] = None) -> Optional[BettingStrategyResult]:
        """
        Evaluate a potential bet using ensemble model predictions.
        
        Args:
            match_id: Identifier for the match
            market: Market identifier (e.g., 'home', 'over_2.5')
            model_predictions: Dictionary mapping model names to prediction dictionaries
            current_odds: Current decimal odds
            home_team: Home team identifier
            away_team: Away team identifier
            league: League identifier
            bankroll: Current bankroll
            date: Match date
            model_confidences: Dictionary mapping model names to confidence scores
            
        Returns:
            Optional[BettingStrategyResult]: Bet evaluation result or None if not a value bet
        """
        # Skip if league not allowed
        if not self.is_league_allowed(league):
            return None
            
        # Skip if market not allowed
        if not self.is_market_allowed(market):
            return None
            
        # Skip if odds outside acceptable range
        if current_odds < self.min_odds or current_odds > self.max_odds:
            return None
        
        # Combine model predictions
        combined_predictions, ensemble_confidence = self.combine_predictions(
            model_predictions, league
        )
        
        # Skip if no predictions for this market
        if market not in combined_predictions:
            return None
        
        # Get combined probability
        predicted_probability = combined_predictions[market]
        
        # Calculate implied probability from current odds
        implied_prob = implied_probability(current_odds)
        
        # Calculate edge
        edge = self.calculate_edge(predicted_probability, current_odds)
        
        # Skip if edge too small
        if edge < self.min_edge:
            return None
            
        # Skip if confidence too low
        if ensemble_confidence < self.confidence_threshold:
            return None
            
        # Calculate expected value
        ev = self.calculate_expected_value(predicted_probability, current_odds)
        
        # Calculate recommended stake
        recommended_stake = bankroll * self.stake_percentage
        
        # Map market to BetType
        bet_type = BetType.OTHER
        if market == 'home':
            bet_type = BetType.HOME
        elif market == 'draw':
            bet_type = BetType.DRAW
        elif market == 'away':
            bet_type = BetType.AWAY
        elif market.startswith('over_'):
            bet_type = BetType.OVER
        elif market.startswith('under_'):
            bet_type = BetType.UNDER
        elif market == 'btts_yes':
            bet_type = BetType.BTTS_YES
        elif market == 'btts_no':
            bet_type = BetType.BTTS_NO
        
        # Create result
        result = BettingStrategyResult(
            match_id=match_id,
            home_team=home_team or "",
            away_team=away_team or "",
            date=date,
            bet_type=bet_type,
            bet_description=f"{market.replace('_', ' ').title()} @ {current_odds:.2f}",
            odds=current_odds,
            predicted_probability=predicted_probability,
            implied_probability=implied_prob,
            edge=edge,
            expected_value=ev,
            recommended_stake=recommended_stake,
            potential_profit=recommended_stake * (current_odds - 1),
            confidence_score=ensemble_confidence,
            strategy_name=self.name,
            model_name="Ensemble",
            extra={
                'model_predictions': {k: v.get(market, 0) for k, v in model_predictions.items()},
                'model_confidences': model_confidences,
                'weighting_method': self.weighting_method
            }
        )
        
        # Log the result
        self.log_result(result)
        
        # Update counter for weight adjustment
        self.bets_since_update += 1
        
        # Check if we need to update weights
        if self.adaptive_weights and self.bets_since_update >= self.weight_update_frequency:
            self.update_model_weights()
        
        return result
    
    @classmethod
    def from_model_predictions(cls,
                             model_predictions_dict: Dict[str, pd.DataFrame],
                             odds_df: pd.DataFrame,
                             league_column: Optional[str] = None,
                             confidence_columns: Optional[Dict[str, str]] = None,
                             bankroll: float = 1000.0,
                             **kwargs) -> List[BettingStrategyResult]:
        """
        Create bets from multiple model prediction dataframes.
        
        Args:
            model_predictions_dict: Dictionary mapping model names to prediction dataframes
            odds_df: DataFrame with columns 'match_id', market columns matching prediction columns
            league_column: Column name for league information
            confidence_columns: Dictionary mapping model names to confidence column names
            bankroll: Current bankroll
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            List[BettingStrategyResult]: List of bet evaluation results
        """
        # Initialize strategy
        strategy = cls(**kwargs)
        
        # Check required columns in odds dataframe
        required_odds_cols = ['match_id']
        if not all(col in odds_df.columns for col in required_odds_cols):
            logger.error(f"Odds DataFrame missing required columns: {required_odds_cols}")
            return []
        
        # Check if we have any model predictions
        if not model_predictions_dict:
            logger.error("No model predictions provided")
            return []
        
        # Get common match IDs across all model predictions and odds
        all_match_ids = set(odds_df['match_id'])
        for model_name, pred_df in model_predictions_dict.items():
            if 'match_id' not in pred_df.columns:
                logger.error(f"Model predictions for {model_name} missing 'match_id' column")
                return []
            all_match_ids = all_match_ids.intersection(set(pred_df['match_id']))
        
        if not all_match_ids:
            logger.warning("No common match IDs found across all datasets")
            return []
        
        # Get market columns (use the first model as reference)
        first_model_name = next(iter(model_predictions_dict))
        first_model_df = model_predictions_dict[first_model_name]
        market_columns = [col for col in first_model_df.columns 
                         if col not in ['match_id', 'home_team', 'away_team', 'date', 'league']]
        
        # List to store results
        results = []
        
        # Process each match
        for match_id in all_match_ids:
            # Get odds row for this match
            odds_row = odds_df[odds_df['match_id'] == match_id].iloc[0]
            
            # Skip if no odds row found
            if odds_row.empty:
                continue
            
            # Get predictions for this match from all models
            match_predictions = {}
            for model_name, pred_df in model_predictions_dict.items():
                pred_row = pred_df[pred_df['match_id'] == match_id]
                
                if pred_row.empty:
                    continue
                
                pred_row = pred_row.iloc[0]
                
                # Extract predictions for all markets
                match_predictions[model_name] = {
                    market: pred_row[market] for market in market_columns 
                    if market in pred_row and not pd.isna(pred_row[market])
                }
            
            # Get home/away teams if available
            home_team = None
            away_team = None
            for model_name, pred_df in model_predictions_dict.items():
                if 'home_team' in pred_df.columns and 'away_team' in pred_df.columns:
                    pred_row = pred_df[pred_df['match_id'] == match_id]
                    if not pred_row.empty:
                        home_team = pred_row['home_team'].iloc[0]
                        away_team = pred_row['away_team'].iloc[0]
                        break
            
            # Use odds dataframe if teams not found in predictions
            if home_team is None and 'home_team' in odds_df.columns:
                home_team = odds_row['home_team']
            if away_team is None and 'away_team' in odds_df.columns:
                away_team = odds_row['away_team']
            
            # Get date if available
            date = None
            for model_name, pred_df in model_predictions_dict.items():
                if 'date' in pred_df.columns:
                    pred_row = pred_df[pred_df['match_id'] == match_id]
                    if not pred_row.empty:
                        date = pred_row['date'].iloc[0]
                        break
            
            # Use odds dataframe if date not found in predictions
            if date is None and 'date' in odds_df.columns:
                date = odds_row['date']
            
            # Get league if available
            league = None
            if league_column:
                for model_name, pred_df in model_predictions_dict.items():
                    if league_column in pred_df.columns:
                        pred_row = pred_df[pred_df['match_id'] == match_id]
                        if not pred_row.empty:
                            league = pred_row[league_column].iloc[0]
                            break
                
                # Use odds dataframe if league not found in predictions
                if league is None and league_column in odds_df.columns:
                    league = odds_row[league_column]
            
            # Get confidences if available
            model_confidences = {}
            if confidence_columns:
                for model_name, conf_col in confidence_columns.items():
                    if model_name in model_predictions_dict:
                        pred_df = model_predictions_dict[model_name]
                        if conf_col in pred_df.columns:
                            pred_row = pred_df[pred_df['match_id'] == match_id]
                            if not pred_row.empty:
                                model_confidences[model_name] = pred_row[conf_col].iloc[0]
            
            # Process each market
            for market in market_columns:
                # Skip if market not in odds
                if market not in odds_row:
                    continue
                
                # Get current odds for this market
                current_odds = odds_row[market]
                
                # Skip if no valid odds
                if pd.isna(current_odds) or current_odds <= 1.0:
                    continue
                
                # Evaluate bet
                result = strategy.evaluate_bet(
                    match_id=match_id,
                    market=market,
                    model_predictions=match_predictions,
                    current_odds=current_odds,
                    home_team=home_team,
                    away_team=away_team,
                    league=league,
                    bankroll=bankroll,
                    date=date,
                    model_confidences=model_confidences
                )
                
                if result:
                    results.append(result)
        
        return results
    
    def update_model_performance(self, 
                               bet_result: Dict[str, Any], 
                               actual_outcome: bool) -> None:
        """
        Update model performance tracking after bet settlement.
        
        Args:
            bet_result: Dictionary with bet result information
            actual_outcome: Whether the bet was a winner
        """
        # Get model predictions for this bet
        model_predictions = bet_result.get('extra', {}).get('model_predictions', {})
        
        # Threshold for "correct" prediction (e.g., > 0.5 for binary markets)
        threshold = 0.5
        
        # Update each model's performance
        for model_name, predicted_prob in model_predictions.items():
            # Model predicted a win if probability > threshold
            model_predicted_win = predicted_prob > threshold
            
            # Update correct/incorrect counts
            if model_name in self.model_performance:
                if model_predicted_win == actual_outcome:
                    self.model_performance[model_name]['correct'] += 1
                else:
                    self.model_performance[model_name]['incorrect'] += 1
    
    def update_performance_tracking(self, 
                                  bet_result: Dict[str, Any]) -> None:
        """
        Update performance tracking metrics after bet settlement.
        
        Args:
            bet_result: Dictionary with bet result information
        """
        # Add result to history
        self.performance_history.append(bet_result)
        
        # Extract information
        market = bet_result.get('market')
        league = bet_result.get('league')
        profit = bet_result.get('profit', 0)
        stake = bet_result.get('stake', 0)
        actual_outcome = bet_result.get('result', False)
        
        # Update model performance if result is available
        if 'result' in bet_result:
            self.update_model_performance(bet_result, bet_result['result'] == 'win')
        
        # Update ROI by market
        if market:
            if market not in self.roi_by_market:
                self.roi_by_market[market] = {'profit': 0, 'stake': 0}
            self.roi_by_market[market]['profit'] += profit
            self.roi_by_market[market]['stake'] += stake
        
        # Update ROI by league
        if league:
            if league not in self.roi_by_league:
                self.roi_by_league[league] = {'profit': 0, 'stake': 0}
            self.roi_by_league[league]['profit'] += profit
            self.roi_by_league[league]['stake'] += stake
        
        # Update ROI by model
        model_predictions = bet_result.get('extra', {}).get('model_predictions', {})
        total_contrib = sum(model_predictions.values())
        
        if total_contrib > 0:
            for model_name, pred_value in model_predictions.items():
                model_contrib = pred_value / total_contrib
                model_profit = profit * model_contrib
                model_stake = stake * model_contrib
                
                if model_name not in self.roi_by_model:
                    self.roi_by_model[model_name] = {'profit': 0, 'stake': 0}
                
                self.roi_by_model[model_name]['profit'] += model_profit
                self.roi_by_model[model_name]['stake'] += model_stake
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        # Calculate overall performance
        total_bets = len(self.performance_history)
        if total_bets == 0:
            return {'total_bets': 0, 'roi': 0, 'profit': 0}
            
        total_stake = sum(bet.get('stake', 0) for bet in self.performance_history)
        total_profit = sum(bet.get('profit', 0) for bet in self.performance_history)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Calculate win rate
        wins = sum(1 for bet in self.performance_history if bet.get('profit', 0) > 0)
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Calculate ROI by market
        roi_by_market = {}
        for market, data in self.roi_by_market.items():
            roi_by_market[market] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by league
        roi_by_league = {}
        for league, data in self.roi_by_league.items():
            roi_by_league[league] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by model
        roi_by_model = {}
        for model_name, data in self.roi_by_model.items():
            roi_by_model[model_name] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate model accuracy
        model_accuracy = {}
        for model_name, perf in self.model_performance.items():
            total = perf['correct'] + perf['incorrect']
            if total > 0:
                model_accuracy[model_name] = perf['correct'] / total
            else:
                model_accuracy[model_name] = 0
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'stake': total_stake,
            'roi_by_market': roi_by_market,
            'roi_by_league': roi_by_league,
            'roi_by_model': roi_by_model,
            'model_accuracy': model_accuracy,
            'model_weights': {model['name']: model['weight'] for model in self.models}
        }

class MarketMovementStrategy(BettingStrategy):
    """
    Market Movement Strategy that identifies value based on significant odds movements.
    
    This strategy:
    1. Tracks odds movements over time to identify significant market shifts
    2. Capitalizes on "steam moves" where sharp money causes rapid odds changes
    3. Can operate in reverse to find value against overreactions
    4. Works well in volatile markets with significant line movements
    """
    
    def __init__(self, 
                name: str = "Market Movement",
                min_movement_pct: float = 0.05,
                max_movement_pct: float = 0.20,
                min_odds: float = 1.5,
                max_odds: float = 7.0,
                stake_percentage: float = 1.0,
                confidence_threshold: float = 0.65,
                market_filter: Optional[List[str]] = None,
                league_filter: Optional[List[str]] = None,
                movement_window: str = "1h",
                follow_steam: bool = True,
                min_liquidity: float = 1000.0,
                time_sensitivity: float = 0.5):
        """
        Initialize the Market Movement strategy.
        
        Args:
            name: Name of the strategy
            min_movement_pct: Minimum percentage movement to consider significant
            max_movement_pct: Maximum percentage movement to consider (avoid outliers)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            stake_percentage: Percentage of standard stake to use (0-100)
            confidence_threshold: Minimum confidence score to place a bet (0-1)
            market_filter: List of markets to consider
            league_filter: List of leagues to consider
            movement_window: Time window to consider for movement ('30m', '1h', '3h', '1d')
            follow_steam: Whether to follow steam moves (True) or fade them (False)
            min_liquidity: Minimum market liquidity to consider (estimated from volume/movement)
            time_sensitivity: Weight given to more recent movements (0-1)
        """
        super().__init__(
            name=name,
            min_edge=min_movement_pct/2,  # Edge is derived from movement
            min_odds=min_odds,
            max_odds=max_odds,
            stake_percentage=stake_percentage,
            confidence_threshold=confidence_threshold
        )
        self.min_movement_pct = min_movement_pct
        self.max_movement_pct = max_movement_pct
        self.market_filter = market_filter or ['1X2', 'over_under', 'btts']
        self.league_filter = league_filter
        self.movement_window = movement_window
        self.follow_steam = follow_steam
        self.min_liquidity = min_liquidity
        self.time_sensitivity = time_sensitivity
        
        # Map time window to seconds
        self.window_seconds = {
            '30m': 1800,
            '1h': 3600,
            '3h': 10800,
            '6h': 21600,
            '12h': 43200,
            '1d': 86400
        }.get(movement_window, 3600)
        
        # Performance tracking
        self.performance_history = []
        self.roi_by_market = {}
        self.roi_by_league = {}
        self.roi_by_movement = {}  # ROI by movement percentage ranges
    
    def is_market_allowed(self, market: str) -> bool:
        """
        Check if a betting market is allowed by the filter.
        
        Args:
            market: Market identifier
            
        Returns:
            bool: Whether the market is allowed
        """
        if not self.market_filter:
            return True
        
        # Map market to category
        category = None
        if market in ['home', 'draw', 'away']:
            category = '1X2'
        elif market.startswith('over_') or market.startswith('under_'):
            category = 'over_under'
        elif market.startswith('btts_'):
            category = 'btts'
        elif market.startswith('cs_'):
            category = 'correct_score'
        else:
            category = market
            
        return category in self.market_filter
    
    def is_league_allowed(self, league: Optional[str]) -> bool:
        """
        Check if a league is allowed by the filter.
        
        Args:
            league: League identifier
            
        Returns:
            bool: Whether the league is allowed
        """
        if not self.league_filter or league is None:
            return True
            
        return league in self.league_filter
    
    def calculate_movement_score(self, 
                              movements: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """
        Calculate a movement score based on a series of odds movements.
        
        Args:
            movements: List of tuples (odds_change_pct, timestamp_seconds_ago, liquidity)
            
        Returns:
            Tuple[float, float]: (Movement score, confidence score)
        """
        if not movements:
            return 0.0, 0.0
        
        # Sort by timestamp (most recent first)
        movements.sort(key=lambda x: x[1])
        
        # Calculate weighted movement score
        total_weight = 0.0
        movement_score = 0.0
        total_liquidity = 0.0
        
        for change_pct, seconds_ago, liquidity in movements:
            # More recent movements have higher weight
            recency_weight = max(0.1, 1.0 - (seconds_ago / self.window_seconds) * self.time_sensitivity)
            
            # Higher liquidity has higher weight
            liquidity_weight = min(1.0, liquidity / self.min_liquidity)
            
            # Combine weights
            weight = recency_weight * liquidity_weight
            total_weight += weight
            
            # Add to score
            movement_score += change_pct * weight
            total_liquidity += liquidity
        
        # Normalize score
        if total_weight > 0:
            movement_score /= total_weight
        
        # Calculate confidence based on liquidity and number of movements
        liquidity_factor = min(1.0, total_liquidity / (self.min_liquidity * 3))
        movement_count_factor = min(1.0, len(movements) / 5)
        confidence = 0.5 + (liquidity_factor * 0.3) + (movement_count_factor * 0.2)
        
        return movement_score, confidence
    
    def estimate_edge_from_movement(self, 
                                  movement_score: float, 
                                  current_odds: float) -> float:
        """
        Estimate betting edge from market movement.
        
        Args:
            movement_score: Calculated movement score
            current_odds: Current decimal odds
            
        Returns:
            float: Estimated edge
        """
        # If following steam, negative movement (shortening odds) is positive edge
        if self.follow_steam:
            # Convert negative movement to positive edge
            edge = -movement_score
        else:
            # When fading steam, positive movement (lengthening odds) is positive edge
            edge = movement_score
        
        # Scale edge based on current odds (higher odds need higher edge)
        odds_factor = max(1.0, np.log(current_odds) + 0.5)
        adjusted_edge = edge / odds_factor
        
        return adjusted_edge
    
    def evaluate_market_movement(self, 
                              match_id: Union[str, int],
                              market: str,
                              odds_history: List[Tuple[float, datetime, Optional[float]]],
                              current_odds: float,
                              home_team: Optional[str] = None,
                              away_team: Optional[str] = None,
                              league: Optional[str] = None,
                              bankroll: float = 1000.0,
                              match_date: Optional[datetime] = None) -> Optional[BettingStrategyResult]:
        """
        Evaluate a potential bet based on market movement.
        
        Args:
            match_id: Identifier for the match
            market: Market identifier (e.g., 'home', 'over_2.5')
            odds_history: List of tuples (odds, timestamp, optional liquidity value)
            current_odds: Current decimal odds
            home_team: Home team identifier
            away_team: Away team identifier
            league: League identifier
            bankroll: Current bankroll
            match_date: Match date
            
        Returns:
            Optional[BettingStrategyResult]: Bet evaluation result or None if not a value bet
        """
        # Skip if league not allowed
        if not self.is_league_allowed(league):
            return None
            
        # Skip if market not allowed
        if not self.is_market_allowed(market):
            return None
            
        # Skip if odds outside acceptable range
        if current_odds < self.min_odds or current_odds > self.max_odds:
            return None
            
        # Skip if no odds history
        if len(odds_history) < 2:
            return None
        
        # Get current time
        now = datetime.now()
        
        # Filter odds history to within our window
        window_start = now - timedelta(seconds=self.window_seconds)
        recent_odds = [(odds, timestamp, liquidity) for odds, timestamp, liquidity in odds_history 
                      if timestamp >= window_start]
        
        # Skip if not enough recent odds
        if len(recent_odds) < 2:
            return None
        
        # Calculate movements
        movements = []
        
        # Sort by timestamp (oldest first)
        recent_odds.sort(key=lambda x: x[1])
        
        # Default liquidity if not provided
        default_liquidity = self.min_liquidity
        
        for i in range(1, len(recent_odds)):
            prev_odds, prev_time, prev_liq = recent_odds[i-1]
            curr_odds, curr_time, curr_liq = recent_odds[i]
            
            # Calculate percentage change
            pct_change = (curr_odds / prev_odds) - 1.0
            
            # Skip insignificant movements
            if abs(pct_change) < self.min_movement_pct or abs(pct_change) > self.max_movement_pct:
                continue
            
            # Calculate seconds ago
            seconds_ago = (now - curr_time).total_seconds()
            
            # Use provided liquidity or default
            liquidity = curr_liq if curr_liq is not None else default_liquidity
            
            movements.append((pct_change, seconds_ago, liquidity))
        
        # Skip if no significant movements
        if not movements:
            return None
        
        # Calculate movement score and confidence
        movement_score, confidence = self.calculate_movement_score(movements)
        
        # Skip if confidence too low
        if confidence < self.confidence_threshold:
            return None
        
        # Estimate edge from movement
        edge = self.estimate_edge_from_movement(movement_score, current_odds)
        
        # Skip if edge too small
        if edge < self.min_edge:
            return None
        
        # Calculate implied probability from current odds
        implied_prob = implied_probability(current_odds)
        
        # Estimated "true" probability based on movement
        estimated_prob = implied_prob * (1.0 + edge)
        
        # Calculate expected value
        ev = self.calculate_expected_value(estimated_prob, current_odds)
        
        # Calculate recommended stake
        recommended_stake = bankroll * self.stake_percentage
        
        # Map market to BetType
        bet_type = BetType.OTHER
        if market == 'home':
            bet_type = BetType.HOME
        elif market == 'draw':
            bet_type = BetType.DRAW
        elif market == 'away':
            bet_type = BetType.AWAY
        elif market.startswith('over_'):
            bet_type = BetType.OVER
        elif market.startswith('under_'):
            bet_type = BetType.UNDER
        elif market == 'btts_yes':
            bet_type = BetType.BTTS_YES
        elif market == 'btts_no':
            bet_type = BetType.BTTS_NO
        
        # Create result
        result = BettingStrategyResult(
            match_id=match_id,
            home_team=home_team or "",
            away_team=away_team or "",
            date=match_date,
            bet_type=bet_type,
            bet_description=f"{market.replace('_', ' ').title()} @ {current_odds:.2f} (Movement: {movement_score:.2%})",
            odds=current_odds,
            predicted_probability=estimated_prob,
            implied_probability=implied_prob,
            edge=edge,
            expected_value=ev,
            recommended_stake=recommended_stake,
            potential_profit=recommended_stake * (current_odds - 1),
            confidence_score=confidence,
            strategy_name=f"{self.name} ({'Follow' if self.follow_steam else 'Fade'} Steam)",
            model_name="Market Movement",
            extra={
                'movement_score': movement_score,
                'movement_count': len(movements),
                'first_odds': recent_odds[0][0],
                'last_odds': recent_odds[-1][0],
                'total_pct_change': (current_odds / recent_odds[0][0]) - 1.0,
                'window': self.movement_window,
                'follow_steam': self.follow_steam
            }
        )
        
        # Log the result
        self.log_result(result)
        
        return result
    
    @classmethod
    def from_odds_history(cls,
                        odds_history_df: pd.DataFrame,
                        current_odds_df: pd.DataFrame,
                        bankroll: float = 1000.0,
                        league_column: Optional[str] = None,
                        **kwargs) -> List[BettingStrategyResult]:
        """
        Create bets from odds history and current odds dataframes.
        
        Args:
            odds_history_df: DataFrame with columns 'match_id', 'market', 'odds', 'timestamp', 'liquidity'
            current_odds_df: DataFrame with columns 'match_id', market columns
            bankroll: Current bankroll
            league_column: Column name for league information
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            List[BettingStrategyResult]: List of bet evaluation results
        """
        # Initialize strategy
        strategy = cls(**kwargs)
        
        # Check required columns in odds history
        required_hist_cols = ['match_id', 'market', 'odds', 'timestamp']
        if not all(col in odds_history_df.columns for col in required_hist_cols):
            logger.error(f"Odds history DataFrame missing required columns: {required_hist_cols}")
            return []
        
        # Check required columns in current odds
        required_curr_cols = ['match_id']
        if not all(col in current_odds_df.columns for col in required_curr_cols):
            logger.error(f"Current odds DataFrame missing required columns: {required_curr_cols}")
            return []
        
        # Get common match IDs
        hist_match_ids = set(odds_history_df['match_id'])
        curr_match_ids = set(current_odds_df['match_id'])
        match_ids = hist_match_ids.intersection(curr_match_ids)
        
        if not match_ids:
            logger.warning("No common match IDs found between datasets")
            return []
        
        # Get list of markets from odds history
        markets = odds_history_df['market'].unique()
        
        # Get team names from current odds if available
        has_team_names = ('home_team' in current_odds_df.columns and 'away_team' in current_odds_df.columns)
        
        # Get match dates from current odds if available
        has_dates = 'date' in current_odds_df.columns
        
        # List to store results
        results = []
        
        # Process each match
        for match_id in match_ids:
            # Get team names
            home_team = None
            away_team = None
            if has_team_names:
                match_row = current_odds_df[current_odds_df['match_id'] == match_id].iloc[0]
                home_team = match_row['home_team']
                away_team = match_row['away_team']
            
            # Get match date
            match_date = None
            if has_dates:
                match_row = current_odds_df[current_odds_df['match_id'] == match_id].iloc[0]
                match_date = match_row['date']
            
            # Get league if available
            league = None
            if league_column and league_column in current_odds_df.columns:
                match_row = current_odds_df[current_odds_df['match_id'] == match_id].iloc[0]
                league = match_row[league_column]
            
            # Get current odds for this match
            curr_odds_row = current_odds_df[current_odds_df['match_id'] == match_id].iloc[0]
            
            # Get odds history for this match
            match_history = odds_history_df[odds_history_df['match_id'] == match_id]
            
            # Process each market
            for market in markets:
                # Skip if market not in odds history for this match
                if market not in match_history['market'].values:
                    continue
                
                # Get market history
                market_history = match_history[match_history['market'] == market]
                
                # Convert to list of tuples
                history_list = []
                for _, row in market_history.iterrows():
                    odds = row['odds']
                    timestamp = row['timestamp']
                    liquidity = row.get('liquidity', None)
                    history_list.append((odds, timestamp, liquidity))
                
                # Get current odds for this market
                if market in curr_odds_row:
                    current_odds = curr_odds_row[market]
                    
                    # Skip if no valid odds
                    if pd.isna(current_odds) or current_odds <= 1.0:
                        continue
                    
                    # Evaluate market movement
                    result = strategy.evaluate_market_movement(
                        match_id=match_id,
                        market=market,
                        odds_history=history_list,
                        current_odds=current_odds,
                        home_team=home_team,
                        away_team=away_team,
                        league=league,
                        bankroll=bankroll,
                        match_date=match_date
                    )
                    
                    if result:
                        results.append(result)
        
        return results
    
    def update_performance_tracking(self, 
                                  bet_result: Dict[str, Any]) -> None:
        """
        Update performance tracking metrics after bet settlement.
        
        Args:
            bet_result: Dictionary with bet result information
        """
        # Add result to history
        self.performance_history.append(bet_result)
        
        # Extract information
        market = bet_result.get('market')
        league = bet_result.get('league')
        movement = bet_result.get('extra', {}).get('movement_score', 0)
        profit = bet_result.get('profit', 0)
        stake = bet_result.get('stake', 0)
        
        # Update ROI by market
        if market:
            if market not in self.roi_by_market:
                self.roi_by_market[market] = {'profit': 0, 'stake': 0}
            self.roi_by_market[market]['profit'] += profit
            self.roi_by_market[market]['stake'] += stake
        
        # Update ROI by league
        if league:
            if league not in self.roi_by_league:
                self.roi_by_league[league] = {'profit': 0, 'stake': 0}
            self.roi_by_league[league]['profit'] += profit
            self.roi_by_league[league]['stake'] += stake
        
        # Update ROI by movement range
        if movement:
            # Categorize movement into ranges
            abs_movement = abs(movement)
            if abs_movement < 0.05:
                range_key = "<5%"
            elif abs_movement < 0.1:
                range_key = "5-10%"
            elif abs_movement < 0.15:
                range_key = "10-15%"
            else:
                range_key = ">15%"
            
            # Add direction
            direction = "Shortening" if movement < 0 else "Lengthening"
            range_key = f"{direction} {range_key}"
            
            if range_key not in self.roi_by_movement:
                self.roi_by_movement[range_key] = {'profit': 0, 'stake': 0}
            
            self.roi_by_movement[range_key]['profit'] += profit
            self.roi_by_movement[range_key]['stake'] += stake
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        # Calculate overall performance
        total_bets = len(self.performance_history)
        if total_bets == 0:
            return {'total_bets': 0, 'roi': 0, 'profit': 0}
            
        total_stake = sum(bet.get('stake', 0) for bet in self.performance_history)
        total_profit = sum(bet.get('profit', 0) for bet in self.performance_history)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Calculate win rate
        wins = sum(1 for bet in self.performance_history if bet.get('profit', 0) > 0)
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Calculate ROI by market
        roi_by_market = {}
        for market, data in self.roi_by_market.items():
            roi_by_market[market] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by league
        roi_by_league = {}
        for league, data in self.roi_by_league.items():
            roi_by_league[league] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by movement
        roi_by_movement = {}
        for range_key, data in self.roi_by_movement.items():
            roi_by_movement[range_key] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Sort movement ranges by ROI
        best_movements = sorted(
            [(r, v) for r, v in roi_by_movement.items() if self.roi_by_movement[r]['stake'] > 0],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'stake': total_stake,
            'roi_by_market': roi_by_market,
            'roi_by_league': roi_by_league,
            'roi_by_movement': roi_by_movement,
            'best_movements': best_movements[:5] if best_movements else [],
            'strategy_type': 'Follow Steam' if self.follow_steam else 'Fade Steam',
            'movement_window': self.movement_window
        }

class ExpectedGoalsStrategy(BettingStrategy):
    """
    A betting strategy based on expected goals (xG) that leverages advanced match statistics.
    
    This strategy:
    1. Uses expected goals (xG) models to identify value bets
    2. Focuses on over/under and both teams to score markets
    3. Takes into account team form, attacking and defensive strengths
    4. Provides confidence-based staking adjustments
    """
    
    def __init__(self, 
                name: str = "Expected Goals",
                min_edge: float = 0.05,
                min_odds: float = 1.5,
                max_odds: float = 4.0,
                stake_percentage: float = 1.0,
                confidence_threshold: float = 0.7,
                xg_weight: float = 0.7,
                historical_weight: float = 0.3,
                form_factor: float = 0.2,
                min_xg_difference: float = 0.3,
                market_filter: Optional[List[str]] = None,
                league_filter: Optional[List[str]] = None,
                min_matches: int = 5):
        """
        Initialize the Expected Goals strategy.
        
        Args:
            name: Name of the strategy
            min_edge: Minimum edge (difference between predicted and implied probability)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            stake_percentage: Percentage of standard stake to use (0-100)
            confidence_threshold: Minimum confidence score to place a bet (0-1)
            xg_weight: Weight given to the xG model (0-1)
            historical_weight: Weight given to historical results (0-1)
            form_factor: Weight adjustment based on recent form (0-1)
            min_xg_difference: Minimum xG difference to consider for certain markets
            market_filter: List of markets to consider (e.g., ['over_under', 'btts'])
            league_filter: List of leagues to consider
            min_matches: Minimum number of matches required for reliable xG data
        """
        super().__init__(
            name=name,
            min_edge=min_edge,
            min_odds=min_odds,
            max_odds=max_odds,
            stake_percentage=stake_percentage,
            confidence_threshold=confidence_threshold
        )
        self.xg_weight = xg_weight
        self.historical_weight = historical_weight
        self.form_factor = form_factor
        self.min_xg_difference = min_xg_difference
        self.market_filter = market_filter or ['over_under', 'btts', '1X2']
        self.league_filter = league_filter
        self.min_matches = min_matches
        
        # Performance tracking
        self.performance_history = []
        self.roi_by_market = {}
        self.roi_by_league = {}
        self.roi_by_xg_range = {}  # ROI by expected goals ranges
    
    def is_market_allowed(self, market: str) -> bool:
        """
        Check if a betting market is allowed by the filter.
        
        Args:
            market: Market identifier
            
        Returns:
            bool: Whether the market is allowed
        """
        if not self.market_filter:
            return True
        
        # Map market to category
        category = None
        if market in ['home', 'draw', 'away']:
            category = '1X2'
        elif market.startswith('over_') or market.startswith('under_'):
            category = 'over_under'
        elif market.startswith('btts_'):
            category = 'btts'
        else:
            category = market
            
        return category in self.market_filter
    
    def is_league_allowed(self, league: Optional[str]) -> bool:
        """
        Check if a league is allowed by the filter.
        
        Args:
            league: League identifier
            
        Returns:
            bool: Whether the league is allowed
        """
        if not self.league_filter or league is None:
            return True
            
        return league in self.league_filter
    
    def calculate_goal_probabilities(self, 
                                 home_xg: float, 
                                 away_xg: float,
                                 home_defense: float = 1.0,
                                 away_defense: float = 1.0,
                                 home_form: float = 1.0,
                                 away_form: float = 1.0) -> Dict[str, float]:
        """
        Calculate goal-related probabilities based on expected goals.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            home_defense: Home team defensive strength factor (1.0 is average)
            away_defense: Away team defensive strength factor (1.0 is average)
            home_form: Home team form factor (1.0 is average)
            away_form: Away team form factor (1.0 is average)
        """
        # Calculate probabilities based on xG models
        home_prob = home_xg / (home_xg + away_xg)
        away_prob = away_xg / (home_xg + away_xg)
        
        # Adjust probabilities based on team form
        home_prob *= home_form
        away_prob *= away_form
        
        # Calculate defensive adjustments
        home_defense_adj = home_defense / (home_defense + away_defense)
        away_defense_adj = away_defense / (home_defense + away_defense)
        
        # Apply defensive adjustments
        home_prob *= home_defense_adj
        away_prob *= away_defense_adj
        
        return {
            'home': home_prob,
            'away': away_prob
        }
    
    def evaluate_bet(self, 
                   match_id: Union[str, int],
                   market: str,
                   home_xg: float,
                   away_xg: float,
                   home_defense: float = 1.0,
                   away_defense: float = 1.0,
                   home_form: float = 1.0,
                   away_form: float = 1.0,
                   bankroll: float = 1000.0,
                   date: Optional[datetime] = None,
                   model_name: Optional[str] = None) -> Optional[BettingStrategyResult]:
        """
        Evaluate a potential bet based on expected goals.
        
        Args:
            match_id: Identifier for the match
            market: Market identifier (e.g., 'over_under', 'btts')
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            home_defense: Home team defensive strength factor (1.0 is average)
            away_defense: Away team defensive strength factor (1.0 is average)
            home_form: Home team form factor (1.0 is average)
            away_form: Away team form factor (1.0 is average)
            bankroll: Current bankroll
            date: Match date
            model_name: Name of the model used for predictions
            
        Returns:
            Optional[BettingStrategyResult]: Bet evaluation result or None if no bet recommended
        """
        # Skip if league not allowed
        if not self.is_league_allowed(None):
            return None
            
        # Skip if market not allowed
        if market not in self.market_filter:
            return None
            
        # Calculate probabilities based on xG models
        probabilities = self.calculate_goal_probabilities(
            home_xg=home_xg,
            away_xg=away_xg,
            home_defense=home_defense,
            away_defense=away_defense,
            home_form=home_form,
            away_form=away_form
        )
        
        # Calculate implied probabilities
        home_prob = probabilities['home']
        away_prob = probabilities['away']
        
        # Calculate edge
        edge = self.calculate_edge(home_prob, 1/home_prob)
        
        # Skip if edge too small
        if edge < self.min_edge:
            return None
            
        # Calculate expected value
        ev = self.calculate_expected_value(home_prob, 1/home_prob)
        
        # Calculate recommended stake
        stake = self.calculate_stake(
            predicted_probability=home_prob,
            decimal_odds=1/home_prob,
            bankroll=bankroll,
            method="fractional_kelly",
            kelly_fraction=0.3  # More conservative for in-play
        )
        
        # Determine bet type
        bet_type = None
        if market == 'over_under':
            if home_xg + away_xg > self.min_xg_difference:
                bet_type = BetType.OVER
            else:
                bet_type = BetType.UNDER
        elif market == 'btts':
            if home_xg > 0 and away_xg > 0:
                bet_type = BetType.BTTS_YES
            else:
                bet_type = BetType.BTTS_NO
        elif market == '1X2':
            if home_prob > away_prob:
                bet_type = BetType.HOME
            elif home_prob < away_prob:
                bet_type = BetType.AWAY
            else:
                bet_type = BetType.DRAW
        
        # Create result
        result = BettingStrategyResult(
            match_id=match_id,
            home_team=None,
            away_team=None,
            date=date,
            bet_type=bet_type,
            bet_description=f"{market.replace('_', ' ').title()} - {home_xg:.2f}xG vs {away_xg:.2f}xG",
            odds=1/home_prob,
            predicted_probability=home_prob,
            implied_probability=implied_probability(1/home_prob),
            edge=edge,
            expected_value=ev,
            recommended_stake=stake,
            potential_profit=stake * (1/home_prob - 1),
            confidence_score=self.confidence_threshold,
            strategy_name=self.name,
            model_name=model_name or self.name,
            extra={
                'home_xg': home_xg,
                'away_xg': away_xg,
                'home_defense': home_defense,
                'away_defense': away_defense,
                'home_form': home_form,
                'away_form': away_form,
                'market': market
            }
        )
        
        # Log the result
        self.log_result(result)
        return result
    
    @classmethod
    def from_model_predictions(cls,
                             model_predictions_dict: Dict[str, pd.DataFrame],
                             odds_df: pd.DataFrame,
                             league_column: Optional[str] = None,
                             confidence_columns: Optional[Dict[str, str]] = None,
                             bankroll: float = 1000.0,
                             **kwargs) -> List[BettingStrategyResult]:
        """
        Create bets from multiple model prediction dataframes.
        
        Args:
            model_predictions_dict: Dictionary mapping model names to prediction dataframes
            odds_df: DataFrame with columns 'match_id', market columns matching prediction columns
            league_column: Column name for league information
            confidence_columns: Dictionary mapping model names to confidence column names
            bankroll: Current bankroll
            **kwargs: Additional parameters for strategy initialization
            
        Returns:
            List[BettingStrategyResult]: List of bet evaluation results
        """
        # Initialize strategy
        strategy = cls(**kwargs)
        
        # Check required columns in odds dataframe
        required_odds_cols = ['match_id']
        if not all(col in odds_df.columns for col in required_odds_cols):
            logger.error(f"Odds DataFrame missing required columns: {required_odds_cols}")
            return []
        
        # Check if we have any model predictions
        if not model_predictions_dict:
            logger.error("No model predictions provided")
            return []
        
        # Get common match IDs across all model predictions and odds
        all_match_ids = set(odds_df['match_id'])
        for model_name, pred_df in model_predictions_dict.items():
            if 'match_id' not in pred_df.columns:
                logger.error(f"Model predictions for {model_name} missing 'match_id' column")
                return []
            all_match_ids = all_match_ids.intersection(set(pred_df['match_id']))
        
        if not all_match_ids:
            logger.warning("No common match IDs found across all datasets")
            return []
        
        # Get market columns (use the first model as reference)
        first_model_name = next(iter(model_predictions_dict))
        first_model_df = model_predictions_dict[first_model_name]
        market_columns = [col for col in first_model_df.columns 
                         if col not in ['match_id', 'home_team', 'away_team', 'date', 'league']]
        
        # List to store results
        results = []
        
        # Process each match
        for match_id in all_match_ids:
            # Get odds row for this match
            odds_row = odds_df[odds_df['match_id'] == match_id].iloc[0]
            
            # Skip if no odds row found
            if odds_row.empty:
                continue
            
            # Get predictions for this match from all models
            match_predictions = {}
            for model_name, pred_df in model_predictions_dict.items():
                pred_row = pred_df[pred_df['match_id'] == match_id]
                
                if pred_row.empty:
                    continue
                
                pred_row = pred_row.iloc[0]
                
                # Extract predictions for all markets
                match_predictions[model_name] = {
                    market: pred_row[market] for market in market_columns 
                    if market in pred_row and not pd.isna(pred_row[market])
                }
            
            # Get home/away teams if available
            home_team = None
            away_team = None
            for model_name, pred_df in model_predictions_dict.items():
                if 'home_team' in pred_df.columns and 'away_team' in pred_df.columns:
                    pred_row = pred_df[pred_df['match_id'] == match_id]
                    if not pred_row.empty:
                        home_team = pred_row['home_team'].iloc[0]
                        away_team = pred_row['away_team'].iloc[0]
                        break
            
            # Use odds dataframe if teams not found in predictions
            if home_team is None and 'home_team' in odds_df.columns:
                home_team = odds_row['home_team']
            if away_team is None and 'away_team' in odds_df.columns:
                away_team = odds_row['away_team']
            
            # Get date if available
            date = None
            for model_name, pred_df in model_predictions_dict.items():
                if 'date' in pred_df.columns:
                    pred_row = pred_df[pred_df['match_id'] == match_id]
                    if not pred_row.empty:
                        date = pred_row['date'].iloc[0]
                        break
            
            # Use odds dataframe if date not found in predictions
            if date is None and 'date' in odds_df.columns:
                date = odds_row['date']
            
            # Get league if available
            league = None
            if league_column:
                for model_name, pred_df in model_predictions_dict.items():
                    if league_column in pred_df.columns:
                        pred_row = pred_df[pred_df['match_id'] == match_id]
                        if not pred_row.empty:
                            league = pred_row[league_column].iloc[0]
                            break
                
                # Use odds dataframe if league not found in predictions
                if league is None and league_column in odds_df.columns:
                    league = odds_row[league_column]
            
            # Get confidences if available
            model_confidences = {}
            if confidence_columns:
                for model_name, conf_col in confidence_columns.items():
                    if model_name in model_predictions_dict:
                        pred_df = model_predictions_dict[model_name]
                        if conf_col in pred_df.columns:
                            pred_row = pred_df[pred_df['match_id'] == match_id]
                            if not pred_row.empty:
                                model_confidences[model_name] = pred_row[conf_col].iloc[0]
            
            # Process each market
            for market in market_columns:
                # Skip if market not in odds
                if market not in odds_row:
                    continue
                
                # Get current odds for this market
                current_odds = odds_row[market]
                
                # Skip if no valid odds
                if pd.isna(current_odds) or current_odds <= 1.0:
                    continue
                
                # Evaluate bet
                result = strategy.evaluate_bet(
                    match_id=match_id,
                    market=market,
                    model_predictions=match_predictions,
                    current_odds=current_odds,
                    home_team=home_team,
                    away_team=away_team,
                    league=league,
                    bankroll=bankroll,
                    date=date,
                    model_confidences=model_confidences
                )
                
                if result:
                    results.append(result)
        
        return results
    
    def update_model_performance(self, 
                               bet_result: Dict[str, Any], 
                               actual_outcome: bool) -> None:
        """
        Update model performance tracking after bet settlement.
        
        Args:
            bet_result: Dictionary with bet result information
            actual_outcome: Whether the bet was a winner
        """
        # Get model predictions for this bet
        model_predictions = bet_result.get('extra', {}).get('model_predictions', {})
        
        # Threshold for "correct" prediction (e.g., > 0.5 for binary markets)
        threshold = 0.5
        
        # Update each model's performance
        for model_name, predicted_prob in model_predictions.items():
            # Model predicted a win if probability > threshold
            model_predicted_win = predicted_prob > threshold
            
            # Update correct/incorrect counts
            if model_name in self.model_performance:
                if model_predicted_win == actual_outcome:
                    self.model_performance[model_name]['correct'] += 1
                else:
                    self.model_performance[model_name]['incorrect'] += 1
    
    def update_performance_tracking(self, 
                                  bet_result: Dict[str, Any]) -> None:
        """
        Update performance tracking metrics after bet settlement.
        
        Args:
            bet_result: Dictionary with bet result information
        """
        # Add result to history
        self.performance_history.append(bet_result)
        
        # Extract information
        market = bet_result.get('market')
        league = bet_result.get('league')
        profit = bet_result.get('profit', 0)
        stake = bet_result.get('stake', 0)
        actual_outcome = bet_result.get('result', False)
        
        # Update model performance if result is available
        if 'result' in bet_result:
            self.update_model_performance(bet_result, bet_result['result'] == 'win')
        
        # Update ROI by market
        if market:
            if market not in self.roi_by_market:
                self.roi_by_market[market] = {'profit': 0, 'stake': 0}
            self.roi_by_market[market]['profit'] += profit
            self.roi_by_market[market]['stake'] += stake
        
        # Update ROI by league
        if league:
            if league not in self.roi_by_league:
                self.roi_by_league[league] = {'profit': 0, 'stake': 0}
            self.roi_by_league[league]['profit'] += profit
            self.roi_by_league[league]['stake'] += stake
        
        # Update ROI by model
        model_predictions = bet_result.get('extra', {}).get('model_predictions', {})
        total_contrib = sum(model_predictions.values())
        
        if total_contrib > 0:
            for model_name, pred_value in model_predictions.items():
                model_contrib = pred_value / total_contrib
                model_profit = profit * model_contrib
                model_stake = stake * model_contrib
                
                if model_name not in self.roi_by_model:
                    self.roi_by_model[model_name] = {'profit': 0, 'stake': 0}
                
                self.roi_by_model[model_name]['profit'] += model_profit
                self.roi_by_model[model_name]['stake'] += model_stake
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        # Calculate overall performance
        total_bets = len(self.performance_history)
        if total_bets == 0:
            return {'total_bets': 0, 'roi': 0, 'profit': 0}
            
        total_stake = sum(bet.get('stake', 0) for bet in self.performance_history)
        total_profit = sum(bet.get('profit', 0) for bet in self.performance_history)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Calculate win rate
        wins = sum(1 for bet in self.performance_history if bet.get('profit', 0) > 0)
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Calculate ROI by market
        roi_by_market = {}
        for market, data in self.roi_by_market.items():
            roi_by_market[market] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by league
        roi_by_league = {}
        for league, data in self.roi_by_league.items():
            roi_by_league[league] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by model
        roi_by_model = {}
        for model_name, data in self.roi_by_model.items():
            roi_by_model[model_name] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate model accuracy
        model_accuracy = {}
        for model_name, perf in self.model_performance.items():
            total = perf['correct'] + perf['incorrect']
            if total > 0:
                model_accuracy[model_name] = perf['correct'] / total
            else:
                model_accuracy[model_name] = 0
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'stake': total_stake,
            'roi_by_market': roi_by_market,
            'roi_by_league': roi_by_league,
            'roi_by_model': roi_by_model,
            'model_accuracy': model_accuracy,
            'model_weights': {model['name']: model['weight'] for model in self.models}
        }

class InPlayBettingStrategy(BettingStrategy):
    """
    A strategy for in-play (live) betting that leverages real-time match states.
    
    This strategy:
    1. Focuses on betting opportunities that emerge during live matches
    2. Uses game state, momentum, and score information to find value
    3. Can be applied to various markets (goals, scorelines, next team to score)
    4. Incorporates time-based probability adjustments
    """
    
    def __init__(self, 
                name: str = "In-Play Betting",
                min_edge: float = 0.06,
                min_odds: float = 1.5,
                max_odds: float = 8.0,
                stake_percentage: float = 0.8,
                confidence_threshold: float = 0.7,
                time_windows: Optional[List[Tuple[int, int]]] = None,
                market_filter: Optional[List[str]] = None,
                league_filter: Optional[List[str]] = None,
                momentum_threshold: float = 0.6,
                max_goal_difference: int = 2,
                min_time_elapsed: int = 10,
                max_time_elapsed: int = 80):
        """
        Initialize the In-Play Betting strategy.
        
        Args:
            name: Name of the strategy
            min_edge: Minimum edge (difference between predicted and implied probability)
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            stake_percentage: Percentage of standard stake to use (0-100)
            confidence_threshold: Minimum confidence score to place a bet (0-1)
            time_windows: List of time windows (start, end) in minutes to consider bets
            market_filter: List of markets to consider
            league_filter: List of leagues to consider
            momentum_threshold: Minimum momentum score to consider a bet (0-1)
            max_goal_difference: Maximum absolute goal difference to consider a bet
            min_time_elapsed: Minimum match time elapsed (minutes) to consider a bet
            max_time_elapsed: Maximum match time elapsed (minutes) to consider a bet
        """
        super().__init__(
            name=name,
            min_edge=min_edge,
            min_odds=min_odds,
            max_odds=max_odds,
            stake_percentage=stake_percentage,
            confidence_threshold=confidence_threshold
        )
        self.time_windows = time_windows or [(10, 30), (30, 60), (60, 80)]
        self.market_filter = market_filter or ['next_goal', 'total_goals', 'result']
        self.league_filter = league_filter
        self.momentum_threshold = momentum_threshold
        self.max_goal_difference = max_goal_difference
        self.min_time_elapsed = min_time_elapsed
        self.max_time_elapsed = max_time_elapsed
        
        # Performance tracking
        self.performance_history = []
        self.roi_by_market = {}
        self.roi_by_time_window = {}
        self.roi_by_score_state = {}
    
    def is_market_allowed(self, market: str) -> bool:
        """
        Check if a betting market is allowed by the filter.
        
        Args:
            market: Market identifier
            
        Returns:
            bool: Whether the market is allowed
        """
        if not self.market_filter:
            return True
        
        # Map market to category
        category = None
        if market in ['home_next', 'away_next', 'no_more_goals']:
            category = 'next_goal'
        elif market.startswith('over_') or market.startswith('under_'):
            category = 'total_goals'
        elif market in ['home', 'draw', 'away']:
            category = 'result'
        else:
            category = market
            
        return category in self.market_filter
    
    def is_league_allowed(self, league: Optional[str]) -> bool:
        """
        Check if a league is allowed by the filter.
        
        Args:
            league: League identifier
            
        Returns:
            bool: Whether the league is allowed
        """
        if not self.league_filter or league is None:
            return True
            
        return league in self.league_filter
    
    def is_time_window_allowed(self, time_elapsed: int) -> bool:
        """
        Check if the current time window is allowed.
        
        Args:
            time_elapsed: Minutes elapsed in the match
            
        Returns:
            bool: Whether the time window is allowed
        """
        if time_elapsed < self.min_time_elapsed or time_elapsed > self.max_time_elapsed:
            return False
            
        for start, end in self.time_windows:
            if start <= time_elapsed <= end:
                return True
                
        return False
    
    def calculate_momentum_score(self, 
                             stats: Dict[str, Any],
                             current_score: Tuple[int, int],
                             time_elapsed: int) -> Tuple[float, float]:
        """
        Calculate momentum scores for both teams based on match statistics.
        
        Args:
            stats: Dictionary of match statistics (shots, possession, etc.)
            current_score: Tuple of (home_goals, away_goals)
            time_elapsed: Minutes elapsed in the match
            
        Returns:
            Tuple[float, float]: (Home momentum, Away momentum) scores (0-1)
        """
        # Extract relevant statistics
        home_shots = stats.get('home_shots', 0)
        away_shots = stats.get('away_shots', 0)
        home_shots_on_target = stats.get('home_shots_on_target', 0)
        away_shots_on_target = stats.get('away_shots_on_target', 0)
        home_possession = stats.get('home_possession', 50)
        away_possession = stats.get('away_possession', 50)
        home_corners = stats.get('home_corners', 0)
        away_corners = stats.get('away_corners', 0)
        
        # Recent events (last 10 minutes)
        home_recent_shots = stats.get('home_recent_shots', 0)
        away_recent_shots = stats.get('away_recent_shots', 0)
        home_recent_corners = stats.get('home_recent_corners', 0)
        away_recent_corners = stats.get('away_recent_corners', 0)
        
        # Calculate base momentum from overall stats
        total_shots = home_shots + away_shots
        shot_factor_home = home_shots / total_shots if total_shots > 0 else 0.5
        
        total_shots_on_target = home_shots_on_target + away_shots_on_target
        shot_on_target_factor_home = home_shots_on_target / total_shots_on_target if total_shots_on_target > 0 else 0.5
        
        possession_factor_home = home_possession / 100
        
        total_corners = home_corners + away_corners
        corner_factor_home = home_corners / total_corners if total_corners > 0 else 0.5
        
        # Calculate recency momentum (more weight to recent events)
        total_recent_shots = home_recent_shots + away_recent_shots
        recent_shot_factor_home = home_recent_shots / total_recent_shots if total_recent_shots > 0 else 0.5
        
        total_recent_corners = home_recent_corners + away_recent_corners
        recent_corner_factor_home = home_recent_corners / total_recent_corners if total_recent_corners > 0 else 0.5
        
        # Score factor (teams trailing often have more momentum)
        home_score, away_score = current_score
        score_diff = home_score - away_score
        
        # If home team is trailing, increase their momentum
        score_factor_home = 0.5
        if score_diff < 0:
            score_factor_home = 0.5 + (min(abs(score_diff), 2) * 0.1)
        elif score_diff > 0:
            score_factor_home = 0.5 - (min(score_diff, 2) * 0.1)
        
        # Time factor (momentum swings more in later stages)
        time_factor = min(1.0, time_elapsed / 90)
        
        # Weight the factors
        base_weights = {
            'shots': 0.15,
            'shots_on_target': 0.2,
            'possession': 0.1,
            'corners': 0.1,
            'recent_shots': 0.2,
            'recent_corners': 0.15,
            'score': 0.1
        }
        
        # Adjust weights based on time
        if time_elapsed < 30:
            # Early game - recent events less important
            weights = {
                'shots': 0.2,
                'shots_on_target': 0.25,
                'possession': 0.15,
                'corners': 0.15,
                'recent_shots': 0.1,
                'recent_corners': 0.05,
                'score': 0.1
            }
        elif time_elapsed < 60:
            # Mid game - balanced weights
            weights = base_weights
        else:
            # Late game - recent events and score more important
            weights = {
                'shots': 0.1,
                'shots_on_target': 0.15,
                'possession': 0.05,
                'corners': 0.05,
                'recent_shots': 0.25,
                'recent_corners': 0.2,
                'score': 0.2
            }
        
        # Calculate home momentum
        home_momentum = (
            weights['shots'] * shot_factor_home +
            weights['shots_on_target'] * shot_on_target_factor_home +
            weights['possession'] * possession_factor_home +
            weights['corners'] * corner_factor_home +
            weights['recent_shots'] * recent_shot_factor_home +
            weights['recent_corners'] * recent_corner_factor_home +
            weights['score'] * score_factor_home
        )
        
        # Away momentum is inverse of home for some factors, but calculated similarly
        shot_factor_away = 1 - shot_factor_home
        shot_on_target_factor_away = 1 - shot_on_target_factor_home
        possession_factor_away = 1 - possession_factor_home
        corner_factor_away = 1 - corner_factor_home
        recent_shot_factor_away = 1 - recent_shot_factor_home
        recent_corner_factor_away = 1 - recent_corner_factor_home
        
        # If away team is trailing, increase their momentum
        score_factor_away = 0.5
        if score_diff > 0:
            score_factor_away = 0.5 + (min(score_diff, 2) * 0.1)
        elif score_diff < 0:
            score_factor_away = 0.5 - (min(abs(score_diff), 2) * 0.1)
        
        # Calculate away momentum
        away_momentum = (
            weights['shots'] * shot_factor_away +
            weights['shots_on_target'] * shot_on_target_factor_away +
            weights['possession'] * possession_factor_away +
            weights['corners'] * corner_factor_away +
            weights['recent_shots'] * recent_shot_factor_away +
            weights['recent_corners'] * recent_corner_factor_away +
            weights['score'] * score_factor_away
        )
        
        return home_momentum, away_momentum
    
    def calculate_time_adjusted_probabilities(self,
                                         base_probabilities: Dict[str, float],
                                         current_score: Tuple[int, int],
                                         time_elapsed: int,
                                         home_momentum: float,
                                         away_momentum: float) -> Dict[str, float]:
        """
        Adjust probabilities based on match time and score.
        
        Args:
            base_probabilities: Dictionary of base probabilities for different markets
            current_score: Tuple of (home_goals, away_goals)
            time_elapsed: Minutes elapsed in the match
            home_momentum: Home team momentum score (0-1)
            away_momentum: Away team momentum score (0-1)
            
        Returns:
            Dict[str, float]: Adjusted probabilities
        """
        # Calculate time factor (0-1)
        time_factor = time_elapsed / 90
        
        # Extract current score
        home_score, away_score = current_score
        score_diff = home_score - away_score
        total_goals = home_score + away_score
        
        # Adjust probabilities for each market
        adjusted_probs = {}
        
        for market, base_prob in base_probabilities.items():
            adjustment = 0.0
            
            if market == 'home_next':
                # Home team next goal probability
                # Adjust based on momentum
                momentum_adj = (home_momentum - 0.5) * 0.3
                
                # Score state adjustment
                if score_diff < 0:
                    # Trailing teams push harder
                    score_adj = min(abs(score_diff), 2) * 0.05
                else:
                    score_adj = 0
                    
                # Time adjustment - later in game, stronger teams assert themselves
                time_adj = 0
                if time_factor > 0.7 and home_momentum > 0.6:
                    time_adj = (time_factor - 0.7) * 0.1
                
                adjustment = momentum_adj + score_adj + time_adj
                
            elif market == 'away_next':
                # Away team next goal probability
                # Similar to home but for away team
                momentum_adj = (away_momentum - 0.5) * 0.3
                
                if score_diff > 0:
                    score_adj = min(score_diff, 2) * 0.05
                else:
                    score_adj = 0
                    
                time_adj = 0
                if time_factor > 0.7 and away_momentum > 0.6:
                    time_adj = (time_factor - 0.7) * 0.1
                
                adjustment = momentum_adj + score_adj + time_adj
                
            elif market == 'no_more_goals':
                # No more goals probability increases with time
                time_adj = time_factor * 0.2
                
                # And decreases with momentum
                momentum_adj = -0.1 * max(home_momentum, away_momentum)
                
                # Few goals so far suggests low-scoring game
                if total_goals <= 1:
                    score_adj = 0.1
                else:
                    score_adj = -0.05 * total_goals
                
                adjustment = time_adj + momentum_adj + score_adj
                
            elif market.startswith('over_'):
                # Over X.5 goals
                goals_threshold = float(market.split('_')[1])
                
                # Remaining goals needed
                goals_needed = goals_threshold - total_goals
                
                # If already exceeded, probability is 1
                if goals_needed <= 0:
                    adjusted_probs[market] = 1.0
                    continue
                    
                # Adjust based on time remaining and momentum
                time_remaining = 1 - time_factor
                
                # Average goals per minute in soccer is roughly 0.03
                avg_goals_per_min = 0.03
                expected_remaining_goals = time_remaining * 90 * avg_goals_per_min
                
                # Adjust for high momentum
                momentum_factor = max(home_momentum, away_momentum)
                if momentum_factor > 0.6:
                    expected_remaining_goals *= (1 + (momentum_factor - 0.6) * 0.5)
                
                # Probability of scoring at least goals_needed
                # Using Poisson approximation
                prob = 1 - poisson.cdf(goals_needed - 1, expected_remaining_goals)
                
                adjusted_probs[market] = max(0, min(1, prob))
                continue
                
            elif market.startswith('under_'):
                # Under X.5 goals
                goals_threshold = float(market.split('_')[1])
                
                # If already exceeded, probability is 0
                if total_goals > goals_threshold:
                    adjusted_probs[market] = 0.0
                    continue
                    
                # Remaining goals allowed
                goals_allowed = goals_threshold - total_goals
                
                # Adjust based on time remaining and momentum
                time_remaining = 1 - time_factor
                
                # Average goals per minute
                avg_goals_per_min = 0.03
                expected_remaining_goals = time_remaining * 90 * avg_goals_per_min
                
                # Adjust for high momentum
                momentum_factor = max(home_momentum, away_momentum)
                if momentum_factor > 0.6:
                    expected_remaining_goals *= (1 + (momentum_factor - 0.6) * 0.5)
                
                # Probability of scoring fewer than goals_allowed
                prob = poisson.cdf(goals_allowed, expected_remaining_goals)
                
                adjusted_probs[market] = max(0, min(1, prob))
                continue
                
            elif market in ['home', 'draw', 'away']:
                # Match result markets
                # Calculate expected remaining goals
                time_remaining = 1 - time_factor
                avg_goals_per_min = 0.03
                expected_remaining_goals = time_remaining * 90 * avg_goals_per_min
                
                # Split based on momentum
                expected_home_goals = expected_remaining_goals * home_momentum
                expected_away_goals = expected_remaining_goals * away_momentum
                
                # Current score
                home_score, away_score = current_score
                
                # Home win probability
                if market == 'home':
                    # Home team already ahead
                    if score_diff > 0:
                        adjustment = time_factor * 0.2
                    # Home team behind
                    elif score_diff < 0:
                        goals_needed = abs(score_diff) + 1
                        # Probability of home scoring at least goals_needed more than away
                        # This is simplified and could be improved
                        if expected_home_goals > expected_away_goals:
                            prob_diff = (expected_home_goals - expected_away_goals)
                            adjustment = -0.2 + prob_diff * 0.1
                        else:
                            adjustment = -0.3
                
                # Draw probability
                elif market == 'draw':
                    # Currently tied
                    if score_diff == 0:
                        # More likely as time progresses
                        adjustment = time_factor * 0.25
                    else:
                        # Need exact comeback
                        adjustment = -0.1 + (time_factor * 0.1)
                
                # Away win probability
                elif market == 'away':
                    # Away team already ahead
                    if score_diff < 0:
                        adjustment = time_factor * 0.2
                    # Away team behind
                    elif score_diff > 0:
                        goals_needed = score_diff + 1
                        # Probability of away scoring at least goals_needed more than home
                        if expected_away_goals > expected_home_goals:
                            prob_diff = (expected_away_goals - expected_home_goals)
                            adjustment = -0.2 + prob_diff * 0.1
                        else:
                            adjustment = -0.3
            
            # Apply adjustment to base probability
            adjusted_prob = base_prob + adjustment
            
            # Ensure probability is between 0 and 1
            adjusted_probs[market] = max(0, min(1, adjusted_prob))
        
        return adjusted_probs
    
    def evaluate_bet(self, 
                   match_id: Union[str, int],
                   market: str,
                   current_odds: float,
                   base_probabilities: Dict[str, float],
                   match_stats: Dict[str, Any],
                   home_team: Optional[str] = None,
                   away_team: Optional[str] = None,
                   league: Optional[str] = None,
                   bankroll: float = 1000.0,
                   date: Optional[datetime] = None,
                   model_name: Optional[str] = None) -> Optional[BettingStrategyResult]:
        """
        Evaluate a potential in-play bet.
        
        Args:
            match_id: Identifier for the match
            market: Market identifier
            current_odds: Current decimal odds
            base_probabilities: Dictionary of base probabilities
            match_stats: Dictionary of match statistics and state
            home_team: Home team identifier
            away_team: Away team identifier
            league: League identifier
            bankroll: Current bankroll
            date: Match date
            model_name: Name of the model providing base probabilities
            
        Returns:
            Optional[BettingStrategyResult]: Bet evaluation result or None if no bet recommended
        """
        # Skip if league not allowed
        if not self.is_league_allowed(league):
            return None
            
        # Skip if market not allowed
        if not self.is_market_allowed(market):
            return None
            
        # Extract current match state
        time_elapsed = match_stats.get('time_elapsed', 0)
        current_score = (match_stats.get('home_score', 0), match_stats.get('away_score', 0))
        home_score, away_score = current_score
        
        # Skip if time window not allowed
        if not self.is_time_window_allowed(time_elapsed):
            return None
            
        # Skip if goal difference exceeds maximum
        if abs(home_score - away_score) > self.max_goal_difference:
            return None
            
        # Calculate momentum
        home_momentum, away_momentum = self.calculate_momentum_score(
            stats=match_stats,
            current_score=current_score,
            time_elapsed=time_elapsed
        )
        
        # Skip if momentum below threshold (for next goal markets)
        if market in ['home_next', 'away_next']:
            team_momentum = home_momentum if market == 'home_next' else away_momentum
            if team_momentum < self.momentum_threshold:
                return None
        
        # Calculate time-adjusted probabilities
        adjusted_probs = self.calculate_time_adjusted_probabilities(
            base_probabilities=base_probabilities,
            current_score=current_score,
            time_elapsed=time_elapsed,
            home_momentum=home_momentum,
            away_momentum=away_momentum
        )
        
        # Skip if market not in adjusted probabilities
        if market not in adjusted_probs:
            return None
            
        # Get predicted probability
        predicted_probability = adjusted_probs[market]
        
        # Calculate edge
        edge = self.calculate_edge(predicted_probability, current_odds)
        
        # Skip if edge is below threshold
        if edge < self.min_edge:
            return None
            
        # Skip if odds outside allowed range
        if current_odds < self.min_odds or current_odds > self.max_odds:
            return None
        
        # Calculate expected value
        expected_value = self.calculate_expected_value(predicted_probability, current_odds)
        
        # Calculate confidence based on time and state
        # Higher confidence later in game and with clearer patterns
        base_confidence = self.confidence_threshold
        time_conf_factor = time_elapsed / 90  # Later in game = more confidence
        
        # Momentum clarity (how clearly one team has momentum)
        momentum_diff = abs(home_momentum - away_momentum)
        momentum_clarity = momentum_diff * 0.2
        
        # Score clarity
        score_clarity = 0.1 if abs(home_score - away_score) >= 2 else 0
        
        confidence = min(0.95, base_confidence + (time_conf_factor * 0.15) + momentum_clarity + score_clarity)
        
        # Skip if confidence below threshold
        if confidence < self.confidence_threshold:
            return None
        
        # Calculate stake (using fractional Kelly)
        stake = self.calculate_stake(
            predicted_probability=predicted_probability,
            decimal_odds=current_odds,
            bankroll=bankroll,
            method="fractional_kelly",
            kelly_fraction=0.3  # More conservative for in-play
        )
        
        # Reduce stake as the game progresses (higher variance)
        time_factor = time_elapsed / 90
        stake_adjustment = 1 - (time_factor * 0.3)
        stake *= stake_adjustment * self.stake_percentage
        
        # Determine bet type
        bet_type = None
        if market == 'home_next':
            bet_type = BetType.HOME
        elif market == 'away_next':
            bet_type = BetType.AWAY
        elif market == 'no_more_goals':
            bet_type = BetType.UNDER
        elif market.startswith('over_'):
            bet_type = BetType.OVER
        elif market.startswith('under_'):
            bet_type = BetType.UNDER
        elif market == 'home':
            bet_type = BetType.HOME
        elif market == 'draw':
            bet_type = BetType.DRAW
        elif market == 'away':
            bet_type = BetType.AWAY
        else:
            bet_type = BetType.OTHER
        
        # Create description
        time_str = f"{time_elapsed}'"
        score_str = f"{home_score}-{away_score}"
        description = f"[{time_str} {score_str}] {market.replace('_', ' ').title()} - {home_team} vs {away_team} @ {current_odds:.2f}"
        
        # Create result
        result = BettingStrategyResult(
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            date=date,
            bet_type=bet_type,
            bet_description=description,
            odds=current_odds,
            predicted_probability=predicted_probability,
            implied_probability=implied_probability(current_odds),
            edge=edge,
            expected_value=expected_value,
            recommended_stake=stake,
            potential_profit=stake * (current_odds - 1),
            confidence_score=confidence,
            strategy_name=f"{self.name} ({market})",
            model_name=model_name or self.name,
            extra={
                'time_elapsed': time_elapsed,
                'current_score': current_score,
                'home_momentum': home_momentum,
                'away_momentum': away_momentum,
                'market': market,
                'base_probability': base_probabilities.get(market, 0),
                'adjusted_probability': predicted_probability
            }
        )
        
        # Log the result
        self.log_result(result)
        return result
    
    def update_performance_tracking(self, 
                              bet_result: Dict[str, Any]) -> None:
        """
        Update performance tracking with a bet result.
        
        Args:
            bet_result: Dictionary with bet result details
        """
        # Extract data
        market = bet_result.get('extra', {}).get('market', 'unknown')
        time_elapsed = bet_result.get('extra', {}).get('time_elapsed', 0)
        current_score = bet_result.get('extra', {}).get('current_score', (0, 0))
        
        # Determine time window
        if time_elapsed < 30:
            time_window = '0-30'
        elif time_elapsed < 60:
            time_window = '30-60'
        elif time_elapsed < 75:
            time_window = '60-75'
        else:
            time_window = '75+'
        
        # Determine score state
        home_score, away_score = current_score
        if home_score > away_score:
            if home_score - away_score == 1:
                score_state = 'home_leading_1'
            else:
                score_state = 'home_leading_2+'
        elif away_score > home_score:
            if away_score - home_score == 1:
                score_state = 'away_leading_1'
            else:
                score_state = 'away_leading_2+'
        else:
            score_state = 'tied'
        
        # Add to performance history
        self.performance_history.append(bet_result)
        
        # Update ROI by market
        if market not in self.roi_by_market:
            self.roi_by_market[market] = {'profit': 0, 'stake': 0}
            
        self.roi_by_market[market]['profit'] += bet_result.get('profit', 0)
        self.roi_by_market[market]['stake'] += bet_result.get('stake', 0)
        
        # Update ROI by time window
        if time_window not in self.roi_by_time_window:
            self.roi_by_time_window[time_window] = {'profit': 0, 'stake': 0}
            
        self.roi_by_time_window[time_window]['profit'] += bet_result.get('profit', 0)
        self.roi_by_time_window[time_window]['stake'] += bet_result.get('stake', 0)
        
        # Update ROI by score state
        if score_state not in self.roi_by_score_state:
            self.roi_by_score_state[score_state] = {'profit': 0, 'stake': 0}
            
        self.roi_by_score_state[score_state]['profit'] += bet_result.get('profit', 0)
        self.roi_by_score_state[score_state]['stake'] += bet_result.get('stake', 0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of strategy performance.
        
        Returns:
            Dict[str, Any]: Performance summary statistics
        """
        if not self.performance_history:
            return {'bets': 0, 'win_rate': 0, 'roi': 0, 'profit': 0}
        
        # Calculate overall performance
        total_bets = len(self.performance_history)
        wins = sum(1 for bet in self.performance_history if bet.get('profit', 0) > 0)
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        total_profit = sum(bet.get('profit', 0) for bet in self.performance_history)
        total_stake = sum(bet.get('stake', 0) for bet in self.performance_history)
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Calculate ROI by market
        roi_by_market = {}
        for market, data in self.roi_by_market.items():
            roi_by_market[market] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by time window
        roi_by_time_window = {}
        for time_window, data in self.roi_by_time_window.items():
            roi_by_time_window[time_window] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Calculate ROI by score state
        roi_by_score_state = {}
        for score_state, data in self.roi_by_score_state.items():
            roi_by_score_state[score_state] = data['profit'] / data['stake'] if data['stake'] > 0 else 0
        
        # Find best performing metrics
        best_markets = sorted(
            [(market, roi) for market, roi in roi_by_market.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        best_time_windows = sorted(
            [(time_window, roi) for time_window, roi in roi_by_time_window.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        best_score_states = sorted(
            [(score_state, roi) for score_state, roi in roi_by_score_state.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'bets': total_bets,
            'wins': wins,
            'win_rate': win_rate,
            'roi': roi,
            'profit': total_profit,
            'stake': total_stake,
            'roi_by_market': roi_by_market,
            'roi_by_time_window': roi_by_time_window,
            'roi_by_score_state': roi_by_score_state,
            'best_markets': best_markets[:3] if best_markets else [],
            'best_time_windows': best_time_windows if best_time_windows else [],
            'best_score_states': best_score_states if best_score_states else []
        }
