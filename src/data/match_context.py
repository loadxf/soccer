"""
Match Context Features Module

This module provides functions to generate context-specific features for soccer matches,
including rest days, travel distance, match importance, and derby/rivalry detection.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import math
from pathlib import Path

# Import project components
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("data.match_context")

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")


def calculate_rest_days(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the number of days since the last match for both home and away teams.
    
    Args:
        matches_df: DataFrame containing match data
        
    Returns:
        DataFrame with rest days features added
    """
    # Ensure matches are sorted by date
    matches_df = matches_df.copy()
    if 'date' in matches_df.columns and not pd.api.types.is_datetime64_dtype(matches_df['date']):
        matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    matches_df = matches_df.sort_values('date')
    
    # Initialize rest days columns
    matches_df['home_rest_days'] = np.nan
    matches_df['away_rest_days'] = np.nan
    
    # Track the last match date for each team
    team_last_match = {}
    
    # Process each match
    for idx, match in matches_df.iterrows():
        home_team_id = match['home_club_id']
        away_team_id = match['away_club_id']
        match_date = match['date']
        
        # Calculate rest days for home team
        if home_team_id in team_last_match:
            home_last_match = team_last_match[home_team_id]
            home_rest_days = (match_date - home_last_match).days
            matches_df.at[idx, 'home_rest_days'] = home_rest_days
        else:
            # First match of the season for this team
            matches_df.at[idx, 'home_rest_days'] = 7  # Default to 7 days for first match
        
        # Calculate rest days for away team
        if away_team_id in team_last_match:
            away_last_match = team_last_match[away_team_id]
            away_rest_days = (match_date - away_last_match).days
            matches_df.at[idx, 'away_rest_days'] = away_rest_days
        else:
            # First match of the season for this team
            matches_df.at[idx, 'away_rest_days'] = 7  # Default to 7 days for first match
        
        # Update last match date for both teams
        team_last_match[home_team_id] = match_date
        team_last_match[away_team_id] = match_date
    
    # Calculate rest advantage (positive means home team is more rested)
    matches_df['rest_advantage'] = matches_df['home_rest_days'] - matches_df['away_rest_days']
    
    # Create categorical features for quick interpretation
    matches_df['home_team_rested'] = (matches_df['home_rest_days'] >= 6).astype(int)
    matches_df['away_team_rested'] = (matches_df['away_rest_days'] >= 6).astype(int)
    matches_df['home_team_congested'] = (matches_df['home_rest_days'] <= 3).astype(int)
    matches_df['away_team_congested'] = (matches_df['away_rest_days'] <= 3).astype(int)
    
    return matches_df


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Args:
        lat1, lon1: Coordinates of point 1
        lat2, lon2: Coordinates of point 2
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r


def calculate_travel_distance(matches_df: pd.DataFrame, clubs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the travel distance for the away team.
    
    Args:
        matches_df: DataFrame containing match data
        clubs_df: DataFrame containing club location data
        
    Returns:
        DataFrame with travel distance features added
    """
    # Check if location data is available
    if 'latitude' not in clubs_df.columns or 'longitude' not in clubs_df.columns:
        logger.warning("Location data not available in clubs_df, cannot calculate travel distance")
        return matches_df
    
    matches_df = matches_df.copy()
    
    # Create a lookup dictionary for club locations
    club_locations = {}
    for _, club in clubs_df.iterrows():
        club_id = club['club_id']
        if pd.notna(club['latitude']) and pd.notna(club['longitude']):
            club_locations[club_id] = (club['latitude'], club['longitude'])
    
    # Calculate travel distance for each match
    matches_df['travel_distance_km'] = np.nan
    
    for idx, match in matches_df.iterrows():
        home_id = match['home_club_id']
        away_id = match['away_club_id']
        
        if home_id in club_locations and away_id in club_locations:
            home_lat, home_lon = club_locations[home_id]
            away_lat, away_lon = club_locations[away_id]
            
            distance = haversine_distance(home_lat, home_lon, away_lat, away_lon)
            matches_df.at[idx, 'travel_distance_km'] = distance
    
    # Create categorical features based on travel distance
    if 'travel_distance_km' in matches_df:
        matches_df['short_travel'] = (matches_df['travel_distance_km'] < 100).astype(int)
        matches_df['medium_travel'] = ((matches_df['travel_distance_km'] >= 100) & 
                                     (matches_df['travel_distance_km'] < 300)).astype(int)
        matches_df['long_travel'] = (matches_df['travel_distance_km'] >= 300).astype(int)
    
    return matches_df


def identify_derbies(matches_df: pd.DataFrame, clubs_df: pd.DataFrame, 
                    derby_distance_km: float = 50,
                    derby_pairs: Optional[List[Tuple[int, int]]] = None) -> pd.DataFrame:
    """
    Identify derby/rivalry matches based on geographic proximity or specified pairs.
    
    Args:
        matches_df: DataFrame containing match data
        clubs_df: DataFrame containing club data
        derby_distance_km: Maximum distance to consider as a local derby
        derby_pairs: List of team ID pairs that are considered rivals
        
    Returns:
        DataFrame with derby indicator added
    """
    matches_df = matches_df.copy()
    matches_df['is_derby'] = 0
    
    # Detect derbies based on distance if location data is available
    if 'travel_distance_km' in matches_df.columns:
        matches_df.loc[matches_df['travel_distance_km'] <= derby_distance_km, 'is_derby'] = 1
    
    # Add manually specified derby pairs
    if derby_pairs:
        for home_id, away_id in derby_pairs:
            # Mark matches between these teams as derbies
            derby_matches = ((matches_df['home_club_id'] == home_id) & 
                            (matches_df['away_club_id'] == away_id)) | \
                           ((matches_df['home_club_id'] == away_id) & 
                            (matches_df['away_club_id'] == home_id))
            matches_df.loc[derby_matches, 'is_derby'] = 1
    
    return matches_df


def calculate_match_importance(matches_df: pd.DataFrame, 
                              standings_df: Optional[pd.DataFrame] = None,
                              season_progress: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Calculate match importance based on table position, season progress, and implications.
    
    Args:
        matches_df: DataFrame containing match data
        standings_df: DataFrame containing current standings before each match
        season_progress: Series with match date index and % of season completed
        
    Returns:
        DataFrame with match importance features added
    """
    matches_df = matches_df.copy()
    
    # Initialize importance columns
    matches_df['match_importance'] = 1.0  # Default importance
    matches_df['is_relegation_battle'] = 0
    matches_df['is_title_race'] = 0
    matches_df['is_promotion_battle'] = 0
    matches_df['is_europa_battle'] = 0
    matches_df['is_champions_league_battle'] = 0
    
    # If we don't have standings data, use a simple model based on season progress
    if standings_df is None or season_progress is None:
        if 'date' in matches_df.columns and 'season' in matches_df.columns:
            # Group by season and calculate season progress
            season_groups = matches_df.groupby('season')
            
            for season, season_matches in season_groups:
                # Get first and last date of the season
                season_start = season_matches['date'].min()
                season_end = season_matches['date'].max()
                season_length = (season_end - season_start).days
                
                if season_length > 0:
                    # Calculate progress for each match in this season
                    for idx in season_matches.index:
                        match_date = matches_df.loc[idx, 'date']
                        days_elapsed = (match_date - season_start).days
                        progress = days_elapsed / season_length
                        
                        # Importance increases as season progresses
                        # Follow a U-shaped curve: important early, less important mid-season, very important late
                        if progress < 0.1:  # First 10% of season
                            importance = 1.2  # Early season excitement
                        elif progress > 0.8:  # Last 20% of season
                            importance = 1.5  # End of season is crucial
                        else:  # Mid-season
                            importance = 1.0
                        
                        matches_df.at[idx, 'match_importance'] = importance
                        
                        # Late season matches more likely to be crucial battles
                        if progress > 0.7:
                            matches_df.at[idx, 'is_relegation_battle'] = 1
                            matches_df.at[idx, 'is_title_race'] = 1
    
    # If we have standings data, use it for more accurate importance calculation
    else:
        for idx, match in matches_df.iterrows():
            match_date = match['date']
            home_id = match['home_club_id']
            away_id = match['away_club_id']
            
            # Get current standings before this match
            current_standings = standings_df[standings_df['date'] < match_date]
            
            if not current_standings.empty:
                # Use the most recent standings
                latest_date = current_standings['date'].max()
                latest_standings = current_standings[current_standings['date'] == latest_date]
                
                # Get positions of both teams
                try:
                    home_pos = latest_standings[latest_standings['club_id'] == home_id]['position'].iloc[0]
                    away_pos = latest_standings[latest_standings['club_id'] == away_id]['position'].iloc[0]
                    
                    # Position difference affects importance
                    pos_diff = abs(home_pos - away_pos)
                    
                    # Close teams in standings = more important match
                    if pos_diff <= 3:
                        matches_df.at[idx, 'match_importance'] *= 1.3
                    
                    # Title race
                    if home_pos <= 3 or away_pos <= 3:
                        matches_df.at[idx, 'is_title_race'] = 1
                        
                        # Title showdown (both teams in top 3)
                        if home_pos <= 3 and away_pos <= 3:
                            matches_df.at[idx, 'match_importance'] *= 1.5
                    
                    # Relegation battle
                    max_pos = latest_standings['position'].max()
                    if (home_pos >= max_pos - 5) or (away_pos >= max_pos - 5):
                        matches_df.at[idx, 'is_relegation_battle'] = 1
                        
                        # Relegation six-pointer
                        if (home_pos >= max_pos - 5) and (away_pos >= max_pos - 5):
                            matches_df.at[idx, 'match_importance'] *= 1.5
                    
                    # European spots battle
                    if (3 < home_pos <= 7) or (3 < away_pos <= 7):
                        matches_df.at[idx, 'is_europa_battle'] = 1
                        matches_df.at[idx, 'is_champions_league_battle'] = 1
                
                except (IndexError, KeyError):
                    # Team not found in standings
                    pass
    
    # Apply season progress importance modifier if available
    if season_progress is not None:
        for idx, match in matches_df.iterrows():
            match_date = match['date']
            if match_date in season_progress.index:
                progress = season_progress[match_date]
                
                # Last 10% of season is most important
                if progress > 0.9:
                    matches_df.at[idx, 'match_importance'] *= 1.5
                # Last 25% of season is quite important
                elif progress > 0.75:
                    matches_df.at[idx, 'match_importance'] *= 1.3
    
    # Derbies are always important
    if 'is_derby' in matches_df.columns:
        matches_df.loc[matches_df['is_derby'] == 1, 'match_importance'] *= 1.4
    
    return matches_df


def generate_match_context_features(
    matches_df: pd.DataFrame,
    clubs_df: Optional[pd.DataFrame] = None,
    standings_df: Optional[pd.DataFrame] = None,
    derby_pairs: Optional[List[Tuple[int, int]]] = None
) -> pd.DataFrame:
    """
    Generate all match context features.
    
    Args:
        matches_df: DataFrame containing match data
        clubs_df: DataFrame containing club data (optional)
        standings_df: DataFrame containing standings data (optional)
        derby_pairs: List of team ID pairs that are considered rivals (optional)
        
    Returns:
        DataFrame with all match context features added
    """
    # Make a copy to avoid modifying the original
    result_df = matches_df.copy()
    
    # Calculate rest days
    result_df = calculate_rest_days(result_df)
    logger.info("Calculated rest days features")
    
    # Calculate travel distance if club data is available
    if clubs_df is not None:
        result_df = calculate_travel_distance(result_df, clubs_df)
        logger.info("Calculated travel distance features")
    
        # Identify derby matches
        result_df = identify_derbies(result_df, clubs_df, derby_pairs=derby_pairs)
        logger.info("Identified derby matches")
    
    # Calculate match importance
    result_df = calculate_match_importance(result_df, standings_df)
    logger.info("Calculated match importance features")
    
    return result_df 