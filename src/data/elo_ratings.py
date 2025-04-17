"""
Elo Rating System for Soccer Prediction

This module implements a time-decaying Elo rating system for soccer teams.
Elo ratings provide a way to estimate team strength based on match outcomes,
with ratings updated after each match based on the expected vs. actual result.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

# Import project components
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("data.elo_ratings")


def expected_result(rating_a: float, rating_b: float, home_advantage: float = 100) -> float:
    """
    Calculate the expected result (win probability) for team A against team B.
    
    Args:
        rating_a: Elo rating of team A
        rating_b: Elo rating of team B
        home_advantage: Rating points added for home advantage
        
    Returns:
        Expected outcome as a probability (0-1) of team A winning
    """
    # Apply home advantage to home team
    effective_rating_a = rating_a + home_advantage
    
    # Calculate expected result using the Elo formula
    expected = 1.0 / (1.0 + 10 ** ((rating_b - effective_rating_a) / 400))
    
    return expected


def update_rating(rating: float, expected: float, actual: float, k_factor: float = 32) -> float:
    """
    Update Elo rating based on match outcome.
    
    Args:
        rating: Current Elo rating
        expected: Expected match outcome (0-1)
        actual: Actual match outcome (1 for win, 0.5 for draw, 0 for loss)
        k_factor: K-factor determining how much ratings change (higher = more volatile)
        
    Returns:
        Updated Elo rating
    """
    return rating + k_factor * (actual - expected)


def calculate_match_outcome(home_goals: int, away_goals: int) -> Tuple[float, float]:
    """
    Calculate match outcome as Elo scores.
    
    Args:
        home_goals: Goals scored by home team
        away_goals: Goals scored by away team
        
    Returns:
        Tuple of (home_result, away_result) where each is 1 for win, 0.5 for draw, 0 for loss
    """
    if home_goals > away_goals:
        return 1.0, 0.0
    elif home_goals < away_goals:
        return 0.0, 1.0
    else:
        return 0.5, 0.5


def calculate_dynamic_k_factor(
    base_k: float = 32, 
    goal_diff: int = 0, 
    match_importance: float = 1.0, 
    max_k: float = 64
) -> float:
    """
    Calculate a dynamic K-factor based on match characteristics.
    
    Args:
        base_k: Base K-factor value
        goal_diff: Absolute goal difference in the match
        match_importance: Importance multiplier (e.g., 1.5 for tournaments, 1.0 for league)
        max_k: Maximum allowed K-factor
        
    Returns:
        Dynamic K-factor value
    """
    # Adjust K-factor based on goal difference (more decisive victories have more impact)
    goal_factor = 1.0 + (0.1 * min(goal_diff, 3))  # Cap at 3 goals difference
    
    # Calculate final K-factor
    k_factor = base_k * goal_factor * match_importance
    
    # Ensure it doesn't exceed the maximum
    return min(k_factor, max_k)


def apply_time_decay(
    ratings: Dict[int, float], 
    decay_factor: float = 0.5, 
    reference_days: int = 365
) -> Dict[int, float]:
    """
    Apply time decay to Elo ratings (ratings revert to the mean over time).
    
    Args:
        ratings: Dictionary of team IDs to current Elo ratings
        decay_factor: How much to decay (0 = no decay, 1 = full decay to mean)
        reference_days: Number of days over which decay_factor is applied
        
    Returns:
        Dictionary of team IDs to adjusted Elo ratings
    """
    # The mean rating that ratings will decay toward
    mean_rating = 1500
    
    # Apply decay to each rating
    decayed_ratings = {}
    for team_id, rating in ratings.items():
        # Decay formula: new_rating = rating + decay_amount * (mean_rating - rating)
        decayed_ratings[team_id] = rating + decay_factor * (mean_rating - rating)
    
    return decayed_ratings


def calculate_elo_ratings(
    matches_df: pd.DataFrame,
    initial_rating: float = 1500,
    base_k: float = 32,
    home_advantage: float = 100,
    include_dynamic_k: bool = True,
    include_time_decay: bool = True,
    decay_factor: float = 0.1,
    reference_days: int = 120
) -> pd.DataFrame:
    """
    Calculate Elo ratings for all teams in a match dataset.
    Updates ratings after each match in chronological order.
    
    Args:
        matches_df: DataFrame containing match data
        initial_rating: Initial Elo rating for new teams
        base_k: Base K-factor for rating updates
        home_advantage: Rating advantage for home team
        include_dynamic_k: Whether to use dynamic K-factor
        include_time_decay: Whether to apply time decay
        decay_factor: Time decay factor
        reference_days: Reference days for time decay
        
    Returns:
        DataFrame with match data and Elo ratings before and after each match
    """
    # Ensure matches are sorted by date
    matches_df = matches_df.copy()
    if 'date' in matches_df.columns and not pd.api.types.is_datetime64_dtype(matches_df['date']):
        matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    matches_df = matches_df.sort_values('date')
    
    # Initialize team ratings
    team_ratings = {}
    
    # Track the last match date for each team (for time decay)
    team_last_match = {}
    
    # Columns to add to the result DataFrame
    result_columns = [
        'home_elo_pre', 'away_elo_pre',  # Ratings before the match
        'home_elo_post', 'away_elo_post',  # Ratings after the match
        'home_elo_expected', 'away_elo_expected',  # Expected result
        'elo_k_factor'  # K-factor used for the match
    ]
    for col in result_columns:
        matches_df[col] = np.nan
    
    # Process each match in chronological order
    for idx, match in matches_df.iterrows():
        home_team_id = match['home_club_id']
        away_team_id = match['away_club_id']
        
        # Get current ratings (or initial rating if not seen before)
        home_rating = team_ratings.get(home_team_id, initial_rating)
        away_rating = team_ratings.get(away_team_id, initial_rating)
        
        # Apply time decay if enabled
        if include_time_decay:
            if home_team_id in team_last_match:
                days_since_last_match = (match['date'] - team_last_match[home_team_id]).days
                if days_since_last_match > 0:
                    decay = decay_factor * (days_since_last_match / reference_days)
                    home_rating = home_rating + decay * (initial_rating - home_rating)
            
            if away_team_id in team_last_match:
                days_since_last_match = (match['date'] - team_last_match[away_team_id]).days
                if days_since_last_match > 0:
                    decay = decay_factor * (days_since_last_match / reference_days)
                    away_rating = away_rating + decay * (initial_rating - away_rating)
        
        # Calculate expected results
        home_expected = expected_result(home_rating, away_rating, home_advantage)
        away_expected = 1.0 - home_expected
        
        # Calculate actual results
        home_goals = match['home_club_goals']
        away_goals = match['away_club_goals']
        home_actual, away_actual = calculate_match_outcome(home_goals, away_goals)
        
        # Calculate K-factor
        k_factor = base_k
        if include_dynamic_k:
            goal_diff = abs(home_goals - away_goals)
            match_importance = 1.0  # Could be derived from match type or competition
            k_factor = calculate_dynamic_k_factor(base_k, goal_diff, match_importance)
        
        # Update ratings
        new_home_rating = update_rating(home_rating, home_expected, home_actual, k_factor)
        new_away_rating = update_rating(away_rating, away_expected, away_actual, k_factor)
        
        # Store updated ratings
        team_ratings[home_team_id] = new_home_rating
        team_ratings[away_team_id] = new_away_rating
        
        # Update last match date
        team_last_match[home_team_id] = match['date']
        team_last_match[away_team_id] = match['date']
        
        # Store ratings and expected results in the result DataFrame
        matches_df.at[idx, 'home_elo_pre'] = home_rating
        matches_df.at[idx, 'away_elo_pre'] = away_rating
        matches_df.at[idx, 'home_elo_post'] = new_home_rating
        matches_df.at[idx, 'away_elo_post'] = new_away_rating
        matches_df.at[idx, 'home_elo_expected'] = home_expected
        matches_df.at[idx, 'away_elo_expected'] = away_expected
        matches_df.at[idx, 'elo_k_factor'] = k_factor
    
    return matches_df


def get_latest_team_ratings(
    matches_df: pd.DataFrame,
    cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
    **kwargs
) -> Dict[int, float]:
    """
    Get the latest Elo ratings for all teams as of a given date.
    
    Args:
        matches_df: DataFrame containing match data
        cutoff_date: Only include matches up to this date
        **kwargs: Additional arguments to pass to calculate_elo_ratings
        
    Returns:
        Dictionary mapping team IDs to their latest Elo ratings
    """
    # Filter matches by cutoff date if provided
    if cutoff_date is not None:
        if isinstance(cutoff_date, str):
            cutoff_date = pd.to_datetime(cutoff_date)
        matches_df = matches_df[matches_df['date'] <= cutoff_date].copy()
    
    # Calculate Elo ratings
    elo_df = calculate_elo_ratings(matches_df, **kwargs)
    
    # Get the latest ratings for each team
    team_ratings = {}
    
    # First get all home teams' latest ratings
    home_latest = elo_df.groupby('home_club_id').apply(lambda x: x.sort_values('date').iloc[-1])
    for team_id, row in home_latest.iterrows():
        team_ratings[team_id] = row['home_elo_post']
    
    # Then check away teams' latest ratings
    away_latest = elo_df.groupby('away_club_id').apply(lambda x: x.sort_values('date').iloc[-1])
    for team_id, row in away_latest.iterrows():
        latest_date = row['date']
        # If the team's latest match was as an away team, or if they're not in home_latest
        if team_id not in team_ratings or home_latest.loc[team_id]['date'] < latest_date:
            team_ratings[team_id] = row['away_elo_post']
    
    return team_ratings


def generate_elo_features(matches_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Generate Elo-based features for match prediction.
    
    Args:
        matches_df: DataFrame containing match data
        **kwargs: Additional arguments for Elo calculation
        
    Returns:
        DataFrame with Elo features added
    """
    # Calculate Elo ratings for all matches
    elo_df = calculate_elo_ratings(matches_df, **kwargs)
    
    # Calculate additional derived features
    elo_df['elo_diff'] = elo_df['home_elo_pre'] - elo_df['away_elo_pre']
    elo_df['elo_sum'] = elo_df['home_elo_pre'] + elo_df['away_elo_pre']
    elo_df['elo_avg'] = elo_df['elo_sum'] / 2
    
    # Calculate win probability based on pre-match Elo
    elo_df['home_win_probability'] = elo_df['home_elo_expected']
    elo_df['away_win_probability'] = elo_df['away_elo_expected']
    elo_df['draw_probability'] = 1 - (elo_df['home_win_probability'] + elo_df['away_win_probability'])
    
    # Calculate Elo "surprise" (difference between actual and expected outcome)
    home_actual = np.where(elo_df['home_club_goals'] > elo_df['away_club_goals'], 1.0,
                          np.where(elo_df['home_club_goals'] < elo_df['away_club_goals'], 0.0, 0.5))
    elo_df['elo_surprise'] = np.abs(home_actual - elo_df['home_elo_expected'])
    
    return elo_df 