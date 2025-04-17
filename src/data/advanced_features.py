"""
Advanced Feature Engineering for Soccer Prediction

This module implements specialized feature engineering techniques for soccer match prediction:
- Team performance differentials
- Match importance metrics
- Player availability impact
- Form with exponential decay
- Context-specific features

These advanced features are designed to capture nuanced patterns in soccer data
that simple statistical features might miss.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import os

# Import project components
try:
    from src.utils.logger import get_logger
    logger = get_logger("data.advanced_features")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("data.advanced_features")

try:
    from src.data.soccer_features import calculate_exponential_team_form
except ImportError:
    logger.warning("soccer_features module not available. Some functions may not work.")

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Define paths
FEATURES_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEATURES_DIR, exist_ok=True)


def calculate_team_differential_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate differential features between home and away teams.
    
    These features capture the relative strength difference between teams,
    which research has shown to be more predictive than absolute metrics.
    
    Args:
        df: DataFrame with team features
        
    Returns:
        DataFrame with differential features added
    """
    logger.info("Calculating team differential features")
    
    # Make a copy of the dataframe
    result_df = df.copy()
    
    # Identify home and away feature columns
    home_cols = [col for col in df.columns if col.startswith('home_') and not col.startswith('home_team')]
    away_cols = [col for col in df.columns if col.startswith('away_') and not col.startswith('away_team')]
    
    # Match home and away feature columns
    paired_cols = []
    for home_col in home_cols:
        feature_name = home_col[5:]  # Remove 'home_' prefix
        away_col = f'away_{feature_name}'
        
        if away_col in away_cols:
            paired_cols.append((home_col, away_col, feature_name))
    
    # Calculate differential features
    for home_col, away_col, feature_name in paired_cols:
        # Only create differential for numeric columns
        if pd.api.types.is_numeric_dtype(df[home_col]) and pd.api.types.is_numeric_dtype(df[away_col]):
            # Raw difference (home - away)
            result_df[f'diff_{feature_name}'] = df[home_col] - df[away_col]
            
            # Ratio (home / away) with handling for division by zero
            divisor = df[away_col].replace(0, np.nan)
            ratio = df[home_col] / divisor
            result_df[f'ratio_{feature_name}'] = ratio.fillna(1.0)  # Fill NaN with 1.0
            
            # Absolute difference (helpful for features where direction doesn't matter)
            result_df[f'abs_diff_{feature_name}'] = abs(df[home_col] - df[away_col])
        
    return result_df


def calculate_match_importance(df: pd.DataFrame, league_standings: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate match importance features based on league standings and season context.
    
    Research shows teams perform differently based on what's at stake (relegation battles,
    championship races, etc.). These features capture such contextual importance.
    
    Args:
        df: DataFrame with match data
        league_standings: Optional DataFrame with league standings information
        
    Returns:
        DataFrame with match importance features added
    """
    logger.info("Calculating match importance features")
    
    # Make a copy of the dataframe
    result_df = df.copy()
    
    # If league standings are provided, use them for advanced importance calculations
    if league_standings is not None:
        # Logic to join standings with match data and calculate importance metrics
        # This requires league standings to have team_id, points, position, etc.
        pass
    
    # Even without standings, we can infer some importance from match date and sequence
    if 'match_date' in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_dtype(df['match_date']):
            result_df['match_date'] = pd.to_datetime(df['match_date'])
            
        # Extract month and day of week
        result_df['month'] = result_df['match_date'].dt.month
        result_df['day_of_week'] = result_df['match_date'].dt.dayofweek
        
        # Calculate season progress (0.0 to 1.0)
        # First, identify seasons based on date ranges
        result_df['season_year'] = result_df['match_date'].dt.year
        result_df.loc[result_df['match_date'].dt.month < 7, 'season_year'] -= 1
        
        # Group by season and calculate progress within season
        result_df = result_df.sort_values('match_date')
        seasons = result_df.groupby('season_year')
        
        # Temporary series to hold season progress
        season_progress = pd.Series(index=result_df.index)
        
        for season, season_df in seasons:
            season_start = season_df['match_date'].min()
            season_end = season_df['match_date'].max()
            season_length = (season_end - season_start).days
            
            if season_length > 0:
                # Calculate progress as days from season start / season length
                progress = ((season_df['match_date'] - season_start).dt.days / season_length)
                season_progress.loc[season_df.index] = progress
        
        result_df['season_progress'] = season_progress
        
        # Create importance based on season progress
        # Matches become more important as season progresses, with end-of-season matches most crucial
        result_df['match_importance'] = (result_df['season_progress'] * 2) ** 2
        
        # Cap importance at 1.0
        result_df['match_importance'] = result_df['match_importance'].clip(0, 1)
    
    return result_df


def calculate_player_availability_impact(
    df: pd.DataFrame, 
    player_data: pd.DataFrame, 
    importance_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Calculate features based on player availability and importance.
    
    Research shows missing key players significantly impacts team performance.
    This function quantifies that impact based on player importance metrics.
    
    Args:
        df: DataFrame with match data
        player_data: DataFrame with player availability and importance data
        importance_threshold: Threshold to determine key players
        
    Returns:
        DataFrame with player availability impact features added
    """
    logger.info("Calculating player availability impact features")
    
    # Make a copy of the dataframe
    result_df = df.copy()
    
    # This requires player_data to have match_id, team_id, player_id, available, importance
    if player_data is not None and not player_data.empty:
        # Identify key players based on importance threshold
        key_players = player_data[player_data['importance'] >= importance_threshold]
        
        # Aggregate to match level
        match_stats = key_players.groupby(['match_id', 'team_id']).agg(
            total_key_players=('player_id', 'count'),
            available_key_players=('available', 'sum'),
            avg_key_importance=('importance', 'mean')
        ).reset_index()
        
        # Calculate unavailable key players and percentage
        match_stats['unavailable_key_players'] = match_stats['total_key_players'] - match_stats['available_key_players']
        match_stats['pct_key_available'] = match_stats['available_key_players'] / match_stats['total_key_players']
        
        # Replace NaN with 1.0 (if no key players, assume 100% availability)
        match_stats['pct_key_available'] = match_stats['pct_key_available'].fillna(1.0)
        
        # Split into home and away features
        home_stats = match_stats.copy()
        away_stats = match_stats.copy()
        
        # Create match_id to join on
        if 'match_id' not in result_df.columns and 'home_team_id' in result_df.columns:
            # Try to create a match_id from other columns if it doesn't exist
            result_df['match_id'] = (
                result_df['match_date'].astype(str) + '_' + 
                result_df['home_team_id'].astype(str) + '_' + 
                result_df['away_team_id'].astype(str)
            )
        
        # Join home team stats
        if 'home_team_id' in result_df.columns:
            home_stats = home_stats.rename(columns={
                'team_id': 'home_team_id',
                'total_key_players': 'home_total_key_players',
                'available_key_players': 'home_available_key_players',
                'unavailable_key_players': 'home_unavailable_key_players',
                'pct_key_available': 'home_pct_key_available',
                'avg_key_importance': 'home_avg_key_importance'
            })
            
            result_df = pd.merge(
                result_df, 
                home_stats, 
                on=['match_id', 'home_team_id'], 
                how='left'
            )
        
        # Join away team stats
        if 'away_team_id' in result_df.columns:
            away_stats = away_stats.rename(columns={
                'team_id': 'away_team_id',
                'total_key_players': 'away_total_key_players',
                'available_key_players': 'away_available_key_players',
                'unavailable_key_players': 'away_unavailable_key_players',
                'pct_key_available': 'away_pct_key_available',
                'avg_key_importance': 'away_avg_key_importance'
            })
            
            result_df = pd.merge(
                result_df, 
                away_stats, 
                on=['match_id', 'away_team_id'], 
                how='left'
            )
            
        # Calculate differential features
        if 'home_pct_key_available' in result_df.columns and 'away_pct_key_available' in result_df.columns:
            result_df['diff_pct_key_available'] = result_df['home_pct_key_available'] - result_df['away_pct_key_available']
        
        if 'home_unavailable_key_players' in result_df.columns and 'away_unavailable_key_players' in result_df.columns:
            result_df['diff_unavailable_key_players'] = result_df['home_unavailable_key_players'] - result_df['away_unavailable_key_players']
        
        # Fill missing values
        player_cols = [col for col in result_df.columns if 'key_players' in col or 'key_available' in col]
        for col in player_cols:
            if col in result_df.columns:
                if 'pct' in col:
                    result_df[col] = result_df[col].fillna(1.0)  # Assume full availability if missing
                else:
                    result_df[col] = result_df[col].fillna(0)  # Assume zero if missing
    
    return result_df


def calculate_advanced_form_metrics(df: pd.DataFrame, half_life_days: int = 30) -> pd.DataFrame:
    """
    Calculate advanced form metrics with sophisticated temporal weighting.
    
    Research shows recency bias is critical in sports prediction. This implements
    multiple decay functions to capture both short and long-term form.
    
    Args:
        df: DataFrame with match data
        half_life_days: Half-life in days for exponential decay
        
    Returns:
        DataFrame with advanced form metrics added
    """
    logger.info("Calculating advanced form metrics")
    
    # Import function from soccer_features if available
    try:
        from src.data.soccer_features import calculate_exponential_team_form
        
        # Calculate form with different half-lives to capture multiple time horizons
        short_form = calculate_exponential_team_form(df, half_life_days=14)
        medium_form = calculate_exponential_team_form(df, half_life_days=30)
        long_form = calculate_exponential_team_form(df, half_life_days=60)
        
        # Combine forms with different time horizons
        short_form_cols = [col for col in short_form.columns if 'exp_form' in col]
        for col in short_form_cols:
            short_form = short_form.rename(columns={col: f'short_{col}'})
        
        long_form_cols = [col for col in long_form.columns if 'exp_form' in col]
        for col in long_form_cols:
            long_form = long_form.rename(columns={col: f'long_{col}'})
        
        # Merge forms
        form_df = pd.merge(
            medium_form,
            short_form[['team_id', 'date'] + [f'short_{col}' for col in short_form_cols]],
            on=['team_id', 'date'],
            how='left'
        )
        
        form_df = pd.merge(
            form_df,
            long_form[['team_id', 'date'] + [f'long_{col}' for col in long_form_cols]],
            on=['team_id', 'date'],
            how='left'
        )
        
        # Create form differences (short-term vs long-term)
        for col in short_form_cols:
            if f'short_{col}' in form_df.columns and col in form_df.columns and f'long_{col}' in form_df.columns:
                # Positive values indicate improving form, negative values indicate declining form
                form_df[f'improving_{col}'] = form_df[f'short_{col}'] - form_df[f'long_{col}']
        
        # Return the combined form dataframe
        return form_df
    
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to calculate advanced form metrics: {e}")
        return df


def create_all_advanced_features(
    matches_df: pd.DataFrame,
    player_data: Optional[pd.DataFrame] = None,
    league_standings: Optional[pd.DataFrame] = None,
    half_life_days: int = 30
) -> pd.DataFrame:
    """
    Create all advanced features in one pipeline.
    
    Args:
        matches_df: DataFrame with match data
        player_data: Optional DataFrame with player data
        league_standings: Optional DataFrame with league standings
        half_life_days: Half-life in days for form calculations
        
    Returns:
        DataFrame with all advanced features added
    """
    logger.info("Creating all advanced features")
    
    # Apply feature engineering steps sequentially
    df = matches_df.copy()
    
    # Step 1: Calculate team differential features
    df = calculate_team_differential_features(df)
    
    # Step 2: Calculate match importance features
    df = calculate_match_importance(df, league_standings)
    
    # Step 3: Calculate player availability impact
    if player_data is not None and not player_data.empty:
        df = calculate_player_availability_impact(df, player_data)
    
    # Step 4: Calculate advanced form metrics
    try:
        form_df = calculate_advanced_form_metrics(df, half_life_days)
        
        # Merge form metrics with main dataframe if successful
        if form_df is not None and not form_df.empty:
            common_cols = list(set(df.columns) & set(form_df.columns))
            form_cols = [col for col in form_df.columns if col not in common_cols]
            
            if common_cols and form_cols:
                df = pd.merge(df, form_df[common_cols + form_cols], on=common_cols, how='left')
    except Exception as e:
        logger.warning(f"Failed to add advanced form metrics: {e}")
    
    return df


def select_features_for_model(
    df: pd.DataFrame, 
    model_type: str,
    target_col: str = 'result'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select optimal features based on model type.
    Different models work better with different feature subsets.
    
    Args:
        df: DataFrame with features
        model_type: Type of model (e.g., 'lightgbm', 'nn', 'bayesian')
        target_col: Target column name
        
    Returns:
        Tuple of (dataframe with selected features, list of selected features)
    """
    logger.info(f"Selecting features for {model_type}")
    
    # Basic features that work well for all models
    base_features = [col for col in df.columns if col != target_col and not col.endswith('_id') 
                    and col != 'match_date' and col != 'match_id']
    
    # Model-specific feature selections
    if model_type in ['lightgbm', 'xgboost', 'catboost']:
        # Gradient boosting models handle missing values and categoricals well
        selected_features = base_features
        
    elif model_type in ['neural_network', 'deep_ensemble', 'transformer']:
        # Neural networks work better with standardized continuous features 
        # Exclude categorical features unless they're binary
        selected_features = [f for f in base_features if pd.api.types.is_numeric_dtype(df[f])]
        
        # Add one-hot encoded categoricals
        cat_cols = [f for f in base_features if (not pd.api.types.is_numeric_dtype(df[f]) or 
                                              (pd.api.types.is_numeric_dtype(df[f]) and df[f].nunique() < 15))]
        
        # Don't include columns that are already selected
        cat_cols = [c for c in cat_cols if c not in selected_features]
        
        # One-hot encode them if needed
        for col in cat_cols:
            if df[col].nunique() < 15:  # Only one-hot encode if not too many categories
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                selected_features.extend(dummies.columns.tolist())
        
    elif model_type in ['logistic', 'random_forest']:
        # These models need more preprocessing for categoricals and missing values
        selected_features = [f for f in base_features if pd.api.types.is_numeric_dtype(df[f])]
        
    elif model_type in ['bayesian', 'dixon_coles']:
        # Select features particularly suitable for Bayesian models
        # Prefer interpretable features with clear football meaning
        preferred_patterns = ['_goals', '_points', '_wins', 'diff_', 'form_', 'elo_']
        selected_features = [f for f in base_features if any(pattern in f for pattern in preferred_patterns)]
    
    else:
        # Default to all features
        selected_features = base_features
    
    # Filter out features with too many missing values
    missing_pct = df[selected_features].isnull().mean()
    high_missing = missing_pct[missing_pct > 0.3].index.tolist()
    selected_features = [f for f in selected_features if f not in high_missing]
    
    # Return selected features and dataframe with those features
    return df[selected_features + [target_col]], selected_features 