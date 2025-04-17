"""
Pi-Rating System for Soccer Prediction

This module implements the Pi-rating system, an enhanced rating system for soccer teams.
Based on the paper by Constantinou & Fenton (2013), Pi-ratings improve upon traditional
Elo ratings by incorporating separate offensive and defensive ratings and adjusting
based on in-match goal differences.

Reference: Constantinou, A.C., Fenton, N.E. (2013): Determining the level of ability of 
           football teams by dynamic ratings based on the relative discrepancies in scores 
           between adversaries. Journal of Quantitative Analysis in Sports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import os
from pathlib import Path
from scipy.stats import norm

# Import project components
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("data.pi_ratings")

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Define paths
RATINGS_DIR = os.path.join(DATA_DIR, "ratings")
PI_RATINGS_DIR = os.path.join(RATINGS_DIR, "pi")
os.makedirs(PI_RATINGS_DIR, exist_ok=True)


class PiRatingSystem:
    """
    Implementation of the Pi-rating system for soccer team ratings.
    
    Pi-ratings incorporate:
    1. Separate offensive and defensive ratings
    2. Dynamic rating adjustments based on goal differences
    3. Importance weighting for different competitions
    4. Time decay for ratings
    """
    
    def __init__(self, 
                initial_offensive_rating: float = 0.0,
                initial_defensive_rating: float = 0.0,
                offensive_k: float = 0.12,  # Offensive learning rate
                defensive_k: float = 0.08,  # Defensive learning rate
                home_advantage: float = 0.2,  # Home advantage in expected goal difference
                time_decay_factor: float = 0.005  # Rating decay per day
                ):
        """
        Initialize the Pi-rating system.
        
        Args:
            initial_offensive_rating: Starting offensive rating for new teams
            initial_defensive_rating: Starting defensive rating for new teams
            offensive_k: Offensive rating adjustment factor
            defensive_k: Defensive rating adjustment factor
            home_advantage: Home advantage in expected goal difference
            time_decay_factor: Daily decay factor for ratings
        """
        self.initial_offensive_rating = initial_offensive_rating
        self.initial_defensive_rating = initial_defensive_rating
        self.offensive_k = offensive_k
        self.defensive_k = defensive_k
        self.home_advantage = home_advantage
        self.time_decay_factor = time_decay_factor
        
        # Team ratings
        self.team_ratings = {}
        self.team_last_match_date = {}
        
        # Rating history
        self.store_history = True
        self.rating_history = {}
    
    def _initialize_team(self, team_id: Any):
        """
        Initialize ratings for a team if not already present.
        
        Args:
            team_id: Team identifier
        """
        if team_id not in self.team_ratings:
            self.team_ratings[team_id] = {
                'offensive': self.initial_offensive_rating,
                'defensive': self.initial_defensive_rating,
                'overall': self.initial_offensive_rating - self.initial_defensive_rating
            }
            
            if self.store_history:
                self.rating_history[team_id] = {
                    'offensive': [],
                    'defensive': [],
                    'overall': [],
                    'dates': []
                }
    
    def _apply_time_decay(self, team_id: Any, current_date: datetime):
        """
        Apply time decay to a team's ratings based on time since last match.
        
        Args:
            team_id: Team identifier
            current_date: Current match date
        """
        if team_id not in self.team_last_match_date:
            self.team_last_match_date[team_id] = current_date
            return
        
        # Calculate days since last match
        last_match_date = self.team_last_match_date[team_id]
        days_since_last_match = (current_date - last_match_date).days
        
        if days_since_last_match <= 0:
            return
        
        # Calculate decay factor
        decay_factor = 1.0 - (self.time_decay_factor * days_since_last_match)
        decay_factor = max(0.7, decay_factor)  # Cap decay at 30%
        
        # Apply decay to offensive and defensive ratings
        self.team_ratings[team_id]['offensive'] *= decay_factor
        self.team_ratings[team_id]['defensive'] *= decay_factor
        
        # Update overall rating
        self.team_ratings[team_id]['overall'] = (
            self.team_ratings[team_id]['offensive'] - 
            self.team_ratings[team_id]['defensive']
        )
        
        # Update last match date
        self.team_last_match_date[team_id] = current_date
    
    def _expected_goal_difference(self, home_team_id: Any, away_team_id: Any) -> float:
        """
        Calculate the expected goal difference between two teams.
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            
        Returns:
            float: Expected goal difference (positive means home team advantage)
        """
        # Get team ratings
        home_offensive = self.team_ratings[home_team_id]['offensive']
        home_defensive = self.team_ratings[home_team_id]['defensive']
        away_offensive = self.team_ratings[away_team_id]['offensive']
        away_defensive = self.team_ratings[away_team_id]['defensive']
        
        # Calculate expected goal difference
        # Expected goals home = home offensive - away defensive + home advantage
        # Expected goals away = away offensive - home defensive
        expected_home_goals = home_offensive - away_defensive + self.home_advantage
        expected_away_goals = away_offensive - home_defensive
        
        return expected_home_goals - expected_away_goals
    
    def _calculate_k_factor(self, goal_difference: int, match_importance: float = 1.0) -> Tuple[float, float]:
        """
        Calculate the dynamic K-factor for rating adjustments based on goal difference.
        
        Args:
            goal_difference: Absolute goal difference in the match
            match_importance: Importance weighting for the match (higher for more important matches)
            
        Returns:
            Tuple[float, float]: Adjusted (offensive_k, defensive_k)
        """
        # Base K factors
        base_offensive_k = self.offensive_k
        base_defensive_k = self.defensive_k
        
        # Goal difference multiplier (diminishing returns for large goal differences)
        if goal_difference <= 1:
            goal_multiplier = 1.0
        elif goal_difference == 2:
            goal_multiplier = 1.5
        elif goal_difference == 3:
            goal_multiplier = 1.75
        else:
            goal_multiplier = 2.0  # Cap at 2.0 for goal differences >= 4
        
        # Apply multipliers
        adjusted_offensive_k = base_offensive_k * goal_multiplier * match_importance
        adjusted_defensive_k = base_defensive_k * goal_multiplier * match_importance
        
        return adjusted_offensive_k, adjusted_defensive_k
    
    def update_ratings(self, 
                     home_team_id: Any, 
                     away_team_id: Any, 
                     home_goals: int, 
                     away_goals: int, 
                     match_date: datetime,
                     match_importance: float = 1.0) -> Dict[str, Any]:
        """
        Update team ratings based on a match result.
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_date: Date of the match
            match_importance: Importance factor for the match (1.0 = regular match)
            
        Returns:
            Dict: Updated ratings information
        """
        # Initialize teams if necessary
        self._initialize_team(home_team_id)
        self._initialize_team(away_team_id)
        
        # Apply time decay
        self._apply_time_decay(home_team_id, match_date)
        self._apply_time_decay(away_team_id, match_date)
        
        # Get expected goal difference
        expected_goal_diff = self._expected_goal_difference(home_team_id, away_team_id)
        
        # Actual goal difference
        actual_goal_diff = home_goals - away_goals
        
        # Goal difference discrepancy
        goal_diff_discrepancy = actual_goal_diff - expected_goal_diff
        
        # Calculate dynamic K factors
        abs_goal_diff = abs(actual_goal_diff)
        offensive_k, defensive_k = self._calculate_k_factor(abs_goal_diff, match_importance)
        
        # Calculate rating adjustments
        # If home team outperforms:
        #   Home offensive rating increases, Away defensive rating increases
        #   Home defensive rating decreases, Away offensive rating decreases
        # If away team outperforms:
        #   Away offensive rating increases, Home defensive rating increases
        #   Away defensive rating decreases, Home offensive rating decreases
        
        # Store original ratings
        original_home_ratings = self.team_ratings[home_team_id].copy()
        original_away_ratings = self.team_ratings[away_team_id].copy()
        
        # Apply rating adjustments
        if goal_diff_discrepancy > 0:  # Home team outperformed expectations
            # Offensive adjustment for home team (positive)
            self.team_ratings[home_team_id]['offensive'] += offensive_k * goal_diff_discrepancy
            
            # Defensive adjustment for away team (positive = worse defense)
            self.team_ratings[away_team_id]['defensive'] += defensive_k * goal_diff_discrepancy
            
        elif goal_diff_discrepancy < 0:  # Away team outperformed expectations
            # Defensive adjustment for home team (positive = worse defense)
            self.team_ratings[home_team_id]['defensive'] += defensive_k * abs(goal_diff_discrepancy)
            
            # Offensive adjustment for away team (positive)
            self.team_ratings[away_team_id]['offensive'] += offensive_k * abs(goal_diff_discrepancy)
        
        # Update overall ratings
        self.team_ratings[home_team_id]['overall'] = (
            self.team_ratings[home_team_id]['offensive'] - 
            self.team_ratings[home_team_id]['defensive']
        )
        
        self.team_ratings[away_team_id]['overall'] = (
            self.team_ratings[away_team_id]['offensive'] - 
            self.team_ratings[away_team_id]['defensive']
        )
        
        # Update last match dates
        self.team_last_match_date[home_team_id] = match_date
        self.team_last_match_date[away_team_id] = match_date
        
        # Store ratings history if enabled
        if self.store_history:
            for team_id, ratings in [(home_team_id, self.team_ratings[home_team_id]), 
                                     (away_team_id, self.team_ratings[away_team_id])]:
                self.rating_history[team_id]['offensive'].append(ratings['offensive'])
                self.rating_history[team_id]['defensive'].append(ratings['defensive'])
                self.rating_history[team_id]['overall'].append(ratings['overall'])
                self.rating_history[team_id]['dates'].append(match_date)
        
        # Return updated ratings and adjustments
        return {
            'home_team': {
                'id': home_team_id,
                'original_ratings': original_home_ratings,
                'updated_ratings': self.team_ratings[home_team_id],
                'adjustment': {
                    'offensive': self.team_ratings[home_team_id]['offensive'] - original_home_ratings['offensive'],
                    'defensive': self.team_ratings[home_team_id]['defensive'] - original_home_ratings['defensive'],
                    'overall': self.team_ratings[home_team_id]['overall'] - original_home_ratings['overall']
                }
            },
            'away_team': {
                'id': away_team_id,
                'original_ratings': original_away_ratings,
                'updated_ratings': self.team_ratings[away_team_id],
                'adjustment': {
                    'offensive': self.team_ratings[away_team_id]['offensive'] - original_away_ratings['offensive'],
                    'defensive': self.team_ratings[away_team_id]['defensive'] - original_away_ratings['defensive'],
                    'overall': self.team_ratings[away_team_id]['overall'] - original_away_ratings['overall']
                }
            },
            'match': {
                'date': match_date,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'expected_goal_diff': expected_goal_diff,
                'actual_goal_diff': actual_goal_diff,
                'goal_diff_discrepancy': goal_diff_discrepancy
            }
        }
    
    def predict_match(self, home_team_id: Any, away_team_id: Any) -> Dict[str, Any]:
        """
        Predict the outcome of a match between two teams.
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            
        Returns:
            Dict: Prediction results
        """
        # Initialize teams if necessary
        self._initialize_team(home_team_id)
        self._initialize_team(away_team_id)
        
        # Get expected goal difference
        expected_goal_diff = self._expected_goal_difference(home_team_id, away_team_id)
        
        # Translate to match outcome probabilities using a normal distribution
        # The standard deviation is set to 1.5 goals, which is a common value in soccer
        std_dev = 1.5
        
        # Probability of home win: P(goal_diff > 0)
        p_home_win = 1 - norm.cdf(0, expected_goal_diff, std_dev)
        
        # Probability of draw: P(goal_diff = 0)
        # Approximated as the probability of goal diff between -0.5 and 0.5
        p_draw = norm.cdf(0.5, expected_goal_diff, std_dev) - norm.cdf(-0.5, expected_goal_diff, std_dev)
        
        # Probability of away win: P(goal_diff < 0)
        p_away_win = norm.cdf(0, expected_goal_diff, std_dev)
        
        # Calculate expected goals
        home_offensive = self.team_ratings[home_team_id]['offensive']
        home_defensive = self.team_ratings[home_team_id]['defensive']
        away_offensive = self.team_ratings[away_team_id]['offensive']
        away_defensive = self.team_ratings[away_team_id]['defensive']
        
        expected_home_goals = home_offensive - away_defensive + self.home_advantage
        expected_away_goals = away_offensive - home_defensive
        
        # Ensure expected goals are non-negative
        expected_home_goals = max(0.1, expected_home_goals)
        expected_away_goals = max(0.1, expected_away_goals)
        
        return {
            'home_team': {
                'id': home_team_id,
                'ratings': self.team_ratings[home_team_id],
                'expected_goals': expected_home_goals
            },
            'away_team': {
                'id': away_team_id,
                'ratings': self.team_ratings[away_team_id],
                'expected_goals': expected_away_goals
            },
            'prediction': {
                'expected_goal_diff': expected_goal_diff,
                'p_home_win': p_home_win,
                'p_draw': p_draw,
                'p_away_win': p_away_win
            }
        }
    
    def get_ratings(self, as_of_date: Optional[datetime] = None) -> Dict[Any, Dict[str, float]]:
        """
        Get ratings for all teams, optionally as of a specific date.
        
        Args:
            as_of_date: Optional date to get historical ratings (if history is stored)
            
        Returns:
            Dict: Team ratings dictionary
        """
        if as_of_date is None or not self.store_history:
            return self.team_ratings
        
        # Get historical ratings
        historical_ratings = {}
        
        for team_id in self.rating_history:
            # Find the closest date before as_of_date
            dates = np.array(self.rating_history[team_id]['dates'])
            if len(dates) == 0:
                continue
                
            valid_dates = [d for d in dates if d <= as_of_date]
            if not valid_dates:
                continue
                
            # Get index of the latest valid date
            latest_date = max(valid_dates)
            idx = self.rating_history[team_id]['dates'].index(latest_date)
            
            # Get ratings at that date
            historical_ratings[team_id] = {
                'offensive': self.rating_history[team_id]['offensive'][idx],
                'defensive': self.rating_history[team_id]['defensive'][idx],
                'overall': self.rating_history[team_id]['overall'][idx]
            }
        
        return historical_ratings
    
    def get_rating_history(self, team_id: Any) -> Dict[str, List]:
        """
        Get the rating history for a specific team.
        
        Args:
            team_id: Team identifier
            
        Returns:
            Dict: Dictionary with rating history
        """
        if not self.store_history:
            raise ValueError("Rating history storage is disabled")
        
        if team_id not in self.rating_history:
            raise ValueError(f"No history found for team {team_id}")
        
        return {
            'offensive': self.rating_history[team_id]['offensive'],
            'defensive': self.rating_history[team_id]['defensive'],
            'overall': self.rating_history[team_id]['overall'],
            'dates': [d.strftime("%Y-%m-%d") for d in self.rating_history[team_id]['dates']]
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the rating system to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            str: Path to the saved file
        """
        # Default path if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                PI_RATINGS_DIR, 
                f"pi_ratings_{timestamp}.pkl"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        import joblib
        with open(filepath, "wb") as f:
            joblib.dump(self, f)
        
        logger.info(f"Pi-rating system saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "PiRatingSystem":
        """
        Load a rating system from a file.
        
        Args:
            filepath: Path to the saved file
            
        Returns:
            PiRatingSystem: Loaded rating system
        """
        import joblib
        with open(filepath, "rb") as f:
            system = joblib.load(f)
        
        logger.info(f"Pi-rating system loaded from {filepath}")
        
        return system


def calculate_pi_ratings(
    matches_df: pd.DataFrame,
    offensive_k: float = 0.12,
    defensive_k: float = 0.08,
    home_advantage: float = 0.2,
    time_decay_factor: float = 0.005,
    competition_importance: Optional[Dict[str, float]] = None
) -> Tuple[PiRatingSystem, pd.DataFrame]:
    """
    Calculate Pi-ratings for all teams in a match dataset.
    
    Args:
        matches_df: DataFrame containing match data (requires home_team, away_team, home_goals, away_goals, date columns)
        offensive_k: Offensive rating adjustment factor
        defensive_k: Defensive rating adjustment factor
        home_advantage: Home advantage in expected goal difference
        time_decay_factor: Daily decay factor for ratings
        competition_importance: Optional dictionary mapping competition names to importance factors
        
    Returns:
        Tuple[PiRatingSystem, pd.DataFrame]: Rating system and DataFrame with match data and ratings
    """
    # Check required columns
    required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals', 'date']
    
    # If column names are different, try to map them
    col_mapping = {}
    
    if not all(col in matches_df.columns for col in required_cols):
        # Try to map columns automatically
        possible_mappings = {
            'home_team': ['home_club_id', 'home_team_id', 'home_id'],
            'away_team': ['away_club_id', 'away_team_id', 'away_id'],
            'home_goals': ['home_club_goals', 'home_score', 'home_team_goals'],
            'away_goals': ['away_club_goals', 'away_score', 'away_team_goals'],
            'date': ['match_date', 'game_date', 'date_time']
        }
        
        for req_col, alternates in possible_mappings.items():
            for alt_col in alternates:
                if alt_col in matches_df.columns:
                    col_mapping[req_col] = alt_col
                    break
    
    # Create a copy of the dataframe with correct column names
    matches = matches_df.copy()
    for req_col, alt_col in col_mapping.items():
        matches[req_col] = matches_df[alt_col]
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_dtype(matches['date']):
        matches['date'] = pd.to_datetime(matches['date'])
    
    # Sort matches by date
    matches = matches.sort_values('date')
    
    # Initialize Pi-rating system
    pi_system = PiRatingSystem(
        offensive_k=offensive_k,
        defensive_k=defensive_k,
        home_advantage=home_advantage,
        time_decay_factor=time_decay_factor
    )
    
    # Default competition importance
    if competition_importance is None:
        competition_importance = {
            'league': 1.0,
            'cup': 1.1,
            'champions_league': 1.2,
            'international': 1.3
        }
    
    # Process each match in chronological order
    match_results = []
    
    for idx, match in matches.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        match_date = match['date']
        
        # Determine match importance
        if 'competition' in match:
            competition = match['competition']
            importance = competition_importance.get(competition, 1.0)
        else:
            importance = 1.0
        
        # Update ratings
        rating_update = pi_system.update_ratings(
            home_team_id=home_team,
            away_team_id=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            match_date=match_date,
            match_importance=importance
        )
        
        # Store match result with ratings
        match_result = {
            'match_id': idx,
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'expected_goal_diff': rating_update['match']['expected_goal_diff'],
            'goal_diff_discrepancy': rating_update['match']['goal_diff_discrepancy'],
            'home_offensive_rating_pre': rating_update['home_team']['original_ratings']['offensive'],
            'home_defensive_rating_pre': rating_update['home_team']['original_ratings']['defensive'],
            'home_overall_rating_pre': rating_update['home_team']['original_ratings']['overall'],
            'away_offensive_rating_pre': rating_update['away_team']['original_ratings']['offensive'],
            'away_defensive_rating_pre': rating_update['away_team']['original_ratings']['defensive'],
            'away_overall_rating_pre': rating_update['away_team']['original_ratings']['overall'],
            'home_offensive_rating_post': rating_update['home_team']['updated_ratings']['offensive'],
            'home_defensive_rating_post': rating_update['home_team']['updated_ratings']['defensive'],
            'home_overall_rating_post': rating_update['home_team']['updated_ratings']['overall'],
            'away_offensive_rating_post': rating_update['away_team']['updated_ratings']['offensive'],
            'away_defensive_rating_post': rating_update['away_team']['updated_ratings']['defensive'],
            'away_overall_rating_post': rating_update['away_team']['updated_ratings']['overall']
        }
        
        match_results.append(match_result)
    
    # Create DataFrame with match results
    results_df = pd.DataFrame(match_results)
    
    return pi_system, results_df


def get_latest_pi_ratings(
    matches_df: pd.DataFrame,
    cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
    **kwargs
) -> Dict[Any, Dict[str, float]]:
    """
    Get the latest Pi-ratings for all teams based on match data.
    
    Args:
        matches_df: DataFrame containing match data
        cutoff_date: Optional cutoff date for matches to include
        **kwargs: Additional arguments to pass to calculate_pi_ratings
        
    Returns:
        Dict: Dictionary of team ratings
    """
    # Apply cutoff date if provided
    if cutoff_date is not None:
        if isinstance(cutoff_date, str):
            cutoff_date = pd.to_datetime(cutoff_date)
        matches_df = matches_df[matches_df['date'] <= cutoff_date].copy()
    
    # Calculate ratings
    pi_system, _ = calculate_pi_ratings(matches_df, **kwargs)
    
    # Get final ratings
    if cutoff_date is not None:
        return pi_system.get_ratings(as_of_date=cutoff_date)
    else:
        return pi_system.get_ratings()


def generate_pi_rating_features(matches_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Generate Pi-rating features for a match dataset.
    
    Args:
        matches_df: DataFrame containing match data
        **kwargs: Additional arguments to pass to calculate_pi_ratings
        
    Returns:
        pd.DataFrame: DataFrame with Pi-rating features
    """
    # Calculate ratings
    _, results_df = calculate_pi_ratings(matches_df, **kwargs)
    
    # Select relevant columns for features
    feature_cols = [
        'match_id', 'date', 'home_team', 'away_team',
        'home_offensive_rating_pre', 'home_defensive_rating_pre', 'home_overall_rating_pre',
        'away_offensive_rating_pre', 'away_defensive_rating_pre', 'away_overall_rating_pre',
        'expected_goal_diff'
    ]
    
    return results_df[feature_cols]


def predict_with_pi_ratings(
    pi_system: PiRatingSystem,
    fixtures_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict match outcomes using Pi-ratings.
    
    Args:
        pi_system: Trained Pi-rating system
        fixtures_df: DataFrame containing fixtures to predict
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Check required columns
    required_cols = ['home_team', 'away_team']
    missing_cols = [col for col in required_cols if col not in fixtures_df.columns]
    
    if missing_cols:
        # Try to map columns automatically
        possible_mappings = {
            'home_team': ['home_club_id', 'home_team_id', 'home_id'],
            'away_team': ['away_club_id', 'away_team_id', 'away_id']
        }
        
        for req_col in missing_cols:
            for alt_col in possible_mappings[req_col]:
                if alt_col in fixtures_df.columns:
                    fixtures_df[req_col] = fixtures_df[alt_col]
                    break
    
    # Make predictions
    predictions = []
    
    for idx, fixture in fixtures_df.iterrows():
        home_team = fixture['home_team']
        away_team = fixture['away_team']
        
        # Predict match
        prediction = pi_system.predict_match(home_team, away_team)
        
        # Extract prediction information
        pred_info = {
            'fixture_id': idx,
            'home_team': home_team,
            'away_team': away_team,
            'home_offensive_rating': prediction['home_team']['ratings']['offensive'],
            'home_defensive_rating': prediction['home_team']['ratings']['defensive'],
            'home_overall_rating': prediction['home_team']['ratings']['overall'],
            'away_offensive_rating': prediction['away_team']['ratings']['offensive'],
            'away_defensive_rating': prediction['away_team']['ratings']['defensive'],
            'away_overall_rating': prediction['away_team']['ratings']['overall'],
            'expected_goal_diff': prediction['prediction']['expected_goal_diff'],
            'expected_home_goals': prediction['home_team']['expected_goals'],
            'expected_away_goals': prediction['away_team']['expected_goals'],
            'p_home_win': prediction['prediction']['p_home_win'],
            'p_draw': prediction['prediction']['p_draw'],
            'p_away_win': prediction['prediction']['p_away_win']
        }
        
        predictions.append(pred_info)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df 