"""
Feature Engineering Module

This module provides functions for generating features from processed data.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
import json
from pathlib import Path
import time
from tqdm import tqdm
import joblib

# Import project utilities
try:
    from src.utils.logger import get_logger
    logger = get_logger("data.features")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("data.features")

# Define required columns for different feature types
REQUIRED_COLUMNS = {
    "team_form": ["team_id", "opponent_id", "match_date", "result", "goals_scored", "goals_conceded"],
    "player_form": ["player_id", "team_id", "match_date", "minutes_played", "goals", "assists"],
    "match_features": ["home_team_id", "away_team_id", "match_date", "home_goals", "away_goals"],
    "basic_features": ["match_date", "team_id", "opponent_id", "score"]
}

# Feature engineering configuration with defaults
DEFAULT_CONFIG = {
    "window_sizes": [3, 5, 10],
    "include_team_stats": True,
    "include_opponent_stats": True,
    "include_time_features": True,
    "include_categorical_encoding": True,
    "outlier_threshold": 3.0,  # Standard deviations for outlier detection
    "fill_missing": "mean",    # Strategy for handling missing values: mean, median, zero, none
    "min_matches_required": 3  # Minimum matches required for reliable form calculation
}

# Get data directories
try:
    from config.default_config import DATA_DIR
    # Convert DATA_DIR to Path object if it's a string
    if isinstance(DATA_DIR, str):
        DATA_DIR = Path(DATA_DIR)
except ImportError:
    # Fallback default if config is not available
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Define paths
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def validate_dataframe(
    df: pd.DataFrame, 
    feature_type: str, 
    raise_error: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate a dataframe for feature generation.
    
    Args:
        df: DataFrame to validate
        feature_type: Type of features to generate
        raise_error: Whether to raise an error if validation fails
        
    Returns:
        Tuple of (is_valid, list of missing columns)
    """
    if feature_type not in REQUIRED_COLUMNS:
        if raise_error:
            raise ValueError(f"Unknown feature type: {feature_type}")
        return False, [f"Unknown feature type: {feature_type}"]
    
    required_cols = REQUIRED_COLUMNS[feature_type]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        error_msg = f"Missing required columns for {feature_type}: {', '.join(missing_cols)}"
        logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return False, missing_cols
    
    # Check for empty dataframe
    if len(df) == 0:
        error_msg = "DataFrame is empty"
        logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return False, ["DataFrame is empty"]
    
    # Check data types and basic quality
    type_errors = []
    
    # Date column validation
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        if not pd.api.types.is_datetime64_dtype(df[col]):
            try:
                # Try to convert to datetime
                pd.to_datetime(df[col])
            except:
                type_errors.append(f"Column {col} cannot be converted to datetime")
    
    # Numeric column validation for essential metrics
    numeric_cols = [col for col in df.columns if 'goals' in col.lower() or 
                   'score' in col.lower() or 'points' in col.lower()]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                # Try to convert to numeric
                pd.to_numeric(df[col])
            except:
                type_errors.append(f"Column {col} cannot be converted to numeric")
    
    if type_errors:
        error_msg = f"Data type issues: {', '.join(type_errors)}"
        logger.error(error_msg)
        if raise_error:
            raise ValueError(error_msg)
        return False, type_errors
    
    # Basic quality checks
    quality_errors = []
    
    # Check for too many missing values
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.3].index.tolist()
    if high_missing:
        quality_errors.append(f"High missing values (>30%) in columns: {', '.join(high_missing)}")
    
    if quality_errors:
        logger.warning(f"Data quality issues: {', '.join(quality_errors)}")
        # Don't fail validation for quality warnings, just log them
    
    return len(missing_cols) == 0 and len(type_errors) == 0, missing_cols + type_errors

def preprocess_data(
    df: pd.DataFrame, 
    feature_type: str,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Preprocess data for feature generation.
    
    Args:
        df: Input DataFrame
        feature_type: Type of features to generate
        config: Configuration dictionary
        
    Returns:
        Preprocessed DataFrame
    """
    # Use default config with any overrides
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Merge with defaults for any missing config
        default_config = DEFAULT_CONFIG.copy()
        default_config.update(config)
        config = default_config
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert date columns to datetime
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        if not pd.api.types.is_datetime64_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Sort by date
    if date_cols:
        df = df.sort_values(by=date_cols[0])
    
    # Convert numeric columns
    numeric_cols = [col for col in df.columns if 'goals' in col.lower() or 
                   'score' in col.lower() or 'points' in col.lower()]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values based on config
    if config['fill_missing'] != 'none':
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if config['fill_missing'] == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif config['fill_missing'] == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif config['fill_missing'] == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Handle outliers if configured
    if config.get('handle_outliers', False):
        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            threshold = config['outlier_threshold']
            outliers = abs(df[col] - mean) > threshold * std
            
            if outliers.sum() > 0:
                logger.info(f"Capping {outliers.sum()} outliers in column {col}")
                df.loc[df[col] > mean + threshold * std, col] = mean + threshold * std
                df.loc[df[col] < mean - threshold * std, col] = mean - threshold * std
    
    return df

def calculate_team_form(
    df: pd.DataFrame, 
    config: Dict[str, Any] = None,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Calculate team form features.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        show_progress: Whether to show a progress bar
        
    Returns:
        DataFrame with form features
    """
    # Validate input data
    is_valid, errors = validate_dataframe(df, "team_form")
    if not is_valid:
        logger.error(f"Invalid input data for team form calculation: {errors}")
        return pd.DataFrame()
    
    # Use default config with any overrides
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        # Merge with defaults for any missing config
        default_config = DEFAULT_CONFIG.copy()
        default_config.update(config)
        config = default_config
    
    # Preprocess data
    df = preprocess_data(df, "team_form", config)
    
    # Prepare result dataframe
    teams = df['team_id'].unique()
    result_df = pd.DataFrame()
    
    # Create progress bar if requested
    teams_iter = tqdm(teams, desc="Calculating team form") if show_progress else teams
    
    for team in teams_iter:
        # Get team matches
        team_df = df[df['team_id'] == team].sort_values('match_date')
        
        if len(team_df) < config['min_matches_required']:
            logger.warning(f"Team {team} has fewer than {config['min_matches_required']} matches, skipping")
            continue
        
        # Calculate basic form
        team_df['points'] = team_df['result'].map({'W': 3, 'D': 1, 'L': 0})
        
        # Calculate features for each window size
        for window in config['window_sizes']:
            if len(team_df) <= window:
                # Skip if not enough matches
                continue
                
            # Rolling averages
            team_df[f'form_{window}_matches'] = team_df['points'].rolling(window=window, min_periods=1).mean()
            team_df[f'goals_scored_{window}_matches'] = team_df['goals_scored'].rolling(window=window, min_periods=1).mean()
            team_df[f'goals_conceded_{window}_matches'] = team_df['goals_conceded'].rolling(window=window, min_periods=1).mean()
            
            # Weighted form (more recent matches have higher weight)
            weights = np.exp(np.linspace(-1, 0, window))
            weights = weights / weights.sum()
            
            def weighted_rolling(series, window, weights):
                result = np.full(len(series), np.nan)
                for i in range(window - 1, len(series)):
                    result[i] = np.sum(series.iloc[i-window+1:i+1].values * weights)
                return pd.Series(result, index=series.index)
            
            if len(team_df) >= window:
                team_df[f'weighted_form_{window}_matches'] = weighted_rolling(team_df['points'], window, weights)
        
        # Add to result
        result_df = pd.concat([result_df, team_df])
    
    # Generate quality report
    feature_quality_report = check_feature_quality(result_df)
    
    # Log quality issues
    for issue in feature_quality_report.get('issues', []):
        logger.warning(f"Feature quality issue: {issue}")
    
    return result_df

def calculate_player_form(
    df: pd.DataFrame,
    config: Dict[str, Any] = None,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Calculate player form features.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        show_progress: Whether to show a progress bar
        
    Returns:
        DataFrame with player form features
    """
    # Validate input data
    is_valid, errors = validate_dataframe(df, "player_form")
    if not is_valid:
        logger.error(f"Invalid input data for player form calculation: {errors}")
        return pd.DataFrame()
    
    # Use default config with any overrides
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        default_config = DEFAULT_CONFIG.copy()
        default_config.update(config)
        config = default_config
    
    # Preprocess data
    df = preprocess_data(df, "player_form", config)
    
    # Prepare result dataframe
    players = df['player_id'].unique()
    result_df = pd.DataFrame()
    
    # Create progress bar if requested
    players_iter = tqdm(players, desc="Calculating player form") if show_progress else players
    
    for player in players_iter:
        # Get player matches
        player_df = df[df['player_id'] == player].sort_values('match_date')
        
        if len(player_df) < config['min_matches_required']:
            logger.warning(f"Player {player} has fewer than {config['min_matches_required']} matches, skipping")
            continue
        
        # Calculate performance metrics
        player_df['goals_per_90'] = player_df['goals'] * 90 / player_df['minutes_played'].clip(lower=1)
        player_df['assists_per_90'] = player_df['assists'] * 90 / player_df['minutes_played'].clip(lower=1)
        player_df['goal_contributions'] = player_df['goals'] + player_df['assists']
        
        # Calculate features for each window size
        for window in config['window_sizes']:
            if len(player_df) <= window:
                # Skip if not enough matches
                continue
                
            # Rolling averages
            player_df[f'goals_per_90_{window}_matches'] = player_df['goals_per_90'].rolling(window=window, min_periods=1).mean()
            player_df[f'assists_per_90_{window}_matches'] = player_df['assists_per_90'].rolling(window=window, min_periods=1).mean()
            player_df[f'minutes_{window}_matches'] = player_df['minutes_played'].rolling(window=window, min_periods=1).mean()
            
            # Weighted form (more recent matches have higher weight)
            weights = np.exp(np.linspace(-1, 0, window))
            weights = weights / weights.sum()
            
            def weighted_rolling(series, window, weights):
                result = np.full(len(series), np.nan)
                for i in range(window - 1, len(series)):
                    result[i] = np.sum(series.iloc[i-window+1:i+1].values * weights)
                return pd.Series(result, index=series.index)
            
            if len(player_df) >= window:
                player_df[f'weighted_goals_{window}_matches'] = weighted_rolling(player_df['goals_per_90'], window, weights)
                player_df[f'weighted_assists_{window}_matches'] = weighted_rolling(player_df['assists_per_90'], window, weights)
        
        # Add to result
        result_df = pd.concat([result_df, player_df])
    
    # Generate quality report
    feature_quality_report = check_feature_quality(result_df)
    
    # Log quality issues
    for issue in feature_quality_report.get('issues', []):
        logger.warning(f"Feature quality issue: {issue}")
    
    return result_df

def create_match_features(
    df: pd.DataFrame, 
    config: Dict[str, Any] = None,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Create features for match prediction.
    
    Args:
        df: Input DataFrame with match data
        config: Configuration dictionary
        show_progress: Whether to show a progress bar
        
    Returns:
        DataFrame with match features
    """
    # Validate input data
    is_valid, errors = validate_dataframe(df, "match_features")
    if not is_valid:
        logger.error(f"Invalid input data for match feature creation: {errors}")
        return pd.DataFrame()
    
    # Use default config with any overrides
    if config is None:
        config = DEFAULT_CONFIG.copy()
    else:
        default_config = DEFAULT_CONFIG.copy()
        default_config.update(config)
        config = default_config
    
    # Preprocess data
    df = preprocess_data(df, "match_features", config)
    
    # Ensure match_date is datetime
    df['match_date'] = pd.to_datetime(df['match_date'])
    
    # Create home/away versions of the data
    home_df = df.copy()
    home_df['team_id'] = home_df['home_team_id']
    home_df['opponent_id'] = home_df['away_team_id']
    home_df['goals_scored'] = home_df['home_goals']
    home_df['goals_conceded'] = home_df['away_goals']
    home_df['is_home'] = 1
    
    away_df = df.copy()
    away_df['team_id'] = away_df['away_team_id']
    away_df['opponent_id'] = away_df['home_team_id']
    away_df['goals_scored'] = away_df['away_goals']
    away_df['goals_conceded'] = away_df['home_goals']
    away_df['is_home'] = 0
    
    # Combine and determine result
    team_df = pd.concat([home_df, away_df]).sort_values('match_date')
    team_df['result'] = np.where(team_df['goals_scored'] > team_df['goals_conceded'], 'W',
                            np.where(team_df['goals_scored'] < team_df['goals_conceded'], 'L', 'D'))
    
    # Calculate team form
    form_df = calculate_team_form(team_df, config, show_progress)
    
    # Create match features by merging home and away team stats
    match_features = df.copy()
    
    # Add progress bar if requested
    matches_iter = tqdm(range(len(match_features)), desc="Creating match features") if show_progress else range(len(match_features))
    
    # Get the last form stats before each match
    for i in matches_iter:
        match = match_features.iloc[i]
        match_date = match['match_date']
        home_team = match['home_team_id']
        away_team = match['away_team_id']
        
        # Get form stats for home team before this match
        home_form = form_df[(form_df['team_id'] == home_team) & 
                           (form_df['match_date'] < match_date)].sort_values('match_date')
        
        if len(home_form) > 0:
            # Get the most recent form
            last_home_form = home_form.iloc[-1]
            
            # Add form features to match
            for col in home_form.columns:
                if col.startswith('form_') or col.startswith('goals_') or col.startswith('weighted_'):
                    match_features.loc[i, f'home_{col}'] = last_home_form[col]
        
        # Get form stats for away team before this match
        away_form = form_df[(form_df['team_id'] == away_team) & 
                           (form_df['match_date'] < match_date)].sort_values('match_date')
        
        if len(away_form) > 0:
            # Get the most recent form
            last_away_form = away_form.iloc[-1]
            
            # Add form features to match
            for col in away_form.columns:
                if col.startswith('form_') or col.startswith('goals_') or col.startswith('weighted_'):
                    match_features.loc[i, f'away_{col}'] = last_away_form[col]
    
    # Generate quality report
    feature_quality_report = check_feature_quality(match_features)
    
    # Log quality issues
    for issue in feature_quality_report.get('issues', []):
        logger.warning(f"Feature quality issue: {issue}")
    
    return match_features

def check_feature_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check the quality of generated features.
    
    Args:
        df: DataFrame with generated features
        
    Returns:
        Dictionary with quality metrics and issues
    """
    quality_report = {
        "timestamp": datetime.now().isoformat(),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "issues": []
    }
    
    if len(df) == 0:
        quality_report["issues"].append("Empty DataFrame")
        return quality_report
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    quality_report["missing_stats"] = {
        "total_missing_cells": int(missing.sum()),
        "avg_missing_pct": float(missing_pct.mean()),
        "columns_with_missing": int((missing > 0).sum())
    }
    
    # Log columns with high missing values
    high_missing = missing_pct[missing_pct > 30].index.tolist()
    if high_missing:
        for col in high_missing:
            quality_report["issues"].append(f"High missing values in {col}: {missing_pct[col]:.1f}%")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        quality_report["issues"].append(f"Constant columns: {', '.join(constant_cols)}")
    
    # Check for high correlations between features
    try:
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [(i, j, corr.loc[i, j]) for i, j in zip(*np.where(upper > 0.95))]
            
            if high_corr:
                for i, j, v in high_corr[:5]:  # Report only first 5 to avoid verbosity
                    quality_report["issues"].append(
                        f"High correlation between {numeric_df.columns[i]} and {numeric_df.columns[j]}: {v:.2f}"
                    )
                
                if len(high_corr) > 5:
                    quality_report["issues"].append(f"{len(high_corr) - 5} more high correlations found")
    except Exception as e:
        quality_report["issues"].append(f"Error checking correlations: {str(e)}")
    
    # Check for outliers
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue  # Skip constant columns
                
            z_scores = (df[col] - mean) / std
            outliers_count = (abs(z_scores) > 3).sum()
            outliers_pct = outliers_count / len(df) * 100
            
            if outliers_pct > 5:
                quality_report["issues"].append(
                    f"High number of outliers in {col}: {outliers_count} ({outliers_pct:.1f}%)"
                )
    except Exception as e:
        quality_report["issues"].append(f"Error checking outliers: {str(e)}")
    
    # Add feature statistics
    quality_report["feature_stats"] = {}
    
    try:
        for col in df.select_dtypes(include=['number']).columns:
            quality_report["feature_stats"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median())
            }
    except Exception as e:
        quality_report["issues"].append(f"Error calculating feature stats: {str(e)}")
    
    return quality_report

def save_feature_metadata(
    feature_df: pd.DataFrame, 
    output_path: str, 
    source_dataset: str,
    quality_report: Dict[str, Any],
    config: Dict[str, Any]
) -> str:
    """
    Save metadata about generated features.
    
    Args:
        feature_df: DataFrame with generated features
        output_path: Path where features were saved
        source_dataset: Name or ID of source dataset
        quality_report: Feature quality report
        config: Configuration used for feature generation
        
    Returns:
        Path to metadata file
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source_dataset": source_dataset,
        "output_path": output_path,
        "num_samples": len(feature_df),
        "num_features": len(feature_df.columns),
        "feature_names": list(feature_df.columns),
        "quality_report": quality_report,
        "configuration": config
    }
    
    # Create metadata file path
    metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save feature metadata: {e}")
    
    return metadata_path

def generate_features_for_dataset(
    input_path: str,
    output_path: str,
    feature_type: str = "match_features",
    config: Dict[str, Any] = None,
    show_progress: bool = False
) -> Tuple[bool, str]:
    """
    Generate features for a dataset file.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to save generated features
        feature_type: Type of features to generate
        config: Configuration for feature generation
        show_progress: Whether to show progress bars
        
    Returns:
        Tuple of (success, message)
    """
    logger.info(f"Generating {feature_type} for {input_path}")
    start_time = time.time()
    
    try:
        # Load the data
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_path)
        elif input_path.endswith('.json'):
            df = pd.read_json(input_path)
        else:
            return False, f"Unsupported file format: {os.path.splitext(input_path)[1]}"
        
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Validate the data
        is_valid, errors = validate_dataframe(df, feature_type)
        if not is_valid:
            return False, f"Invalid dataset for {feature_type}: {', '.join(errors)}"
        
        # Generate features based on type
        if feature_type == "team_form":
            features_df = calculate_team_form(df, config, show_progress)
        elif feature_type == "player_form":
            features_df = calculate_player_form(df, config, show_progress)
        elif feature_type == "match_features":
            features_df = create_match_features(df, config, show_progress)
        else:
            return False, f"Unknown feature type: {feature_type}"
        
        # Check if we got any features
        if len(features_df) == 0:
            return False, "No features generated"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save features
        features_df.to_csv(output_path, index=False)
        
        # Generate and save quality report
        quality_report = check_feature_quality(features_df)
        metadata_path = save_feature_metadata(
            features_df, 
            output_path, 
            os.path.basename(input_path),
            quality_report,
            config or DEFAULT_CONFIG
        )
        
        duration = time.time() - start_time
        logger.info(f"Generated {len(features_df)} samples with {len(features_df.columns)} features in {duration:.2f}s")
        
        return True, f"Generated {len(features_df)} samples with {len(features_df.columns)} features"
    
    except Exception as e:
        logger.exception(f"Error generating features: {str(e)}")
        return False, f"Error: {str(e)}"

def load_processed_data(dataset_name: str = "matches", version: str = "latest") -> pd.DataFrame:
    """Load processed data from the processed directory."""
    logger.info(f"Loading processed data: {dataset_name}, version: {version}")
    
    if version == "latest":
        # Find the latest version
        files = list(PROCESSED_DIR.glob(f"{dataset_name}_*.csv"))
        if not files:
            logger.error(f"No processed data found for {dataset_name}")
            return pd.DataFrame()
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        file_path = files[0]
    else:
        file_path = PROCESSED_DIR / f"{dataset_name}_{version}.csv"
    
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    return pd.read_csv(file_path)

def create_feature_datasets(processed_data: pd.DataFrame, features_config: Dict = None) -> Dict[str, pd.DataFrame]:
    """Create feature datasets from processed data according to configuration."""
    logger.info("Creating feature datasets")
    
    if features_config is None:
        features_config = {
            "basic": ["team_home", "team_away", "league", "season"],
            "match_stats": ["home_goals", "away_goals", "home_shots", "away_shots"],
            "form": ["home_form", "away_form", "home_wins_last5", "away_wins_last5"],
        }
    
    feature_datasets = {}
    
    # Create datasets for each feature group
    for group_name, columns in features_config.items():
        # Filter only columns that exist in the data
        valid_columns = [col for col in columns if col in processed_data.columns]
        
        if not valid_columns:
            logger.warning(f"No valid columns found for group {group_name}")
            continue
            
        # Create the dataset
        feature_datasets[group_name] = processed_data[valid_columns].copy()
        logger.info(f"Created feature group {group_name} with {len(valid_columns)} features")
        
    return feature_datasets

def load_feature_pipeline(pipeline_name: str, version: str = "latest") -> Any:
    """Load a feature transformation pipeline from disk."""
    logger.info(f"Loading feature pipeline: {pipeline_name}, version: {version}")
    
    if version == "latest":
        # Find the latest version
        files = list(FEATURES_DIR.glob(f"pipeline_{pipeline_name}_*.joblib"))
        if not files:
            logger.warning(f"No pipeline found for {pipeline_name}, returning None")
            return None
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        file_path = files[0]
    else:
        file_path = FEATURES_DIR / f"pipeline_{pipeline_name}_{version}.joblib"
    
    logger.info(f"Loading pipeline from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Pipeline file not found: {file_path}")
        return None
    
    try:
        return joblib.load(file_path)
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        return None

def apply_feature_pipeline(data: pd.DataFrame, pipeline_name: str, version: str = "latest") -> pd.DataFrame:
    """Apply a feature transformation pipeline to data."""
    logger.info(f"Applying feature pipeline {pipeline_name} to data of shape {data.shape}")
    
    pipeline = load_feature_pipeline(pipeline_name, version)
    if pipeline is None:
        logger.warning("Pipeline not found, returning original data")
        return data
    
    try:
        transformed_data = pipeline.transform(data)
        logger.info(f"Data transformed successfully, new shape: {transformed_data.shape}")
        return transformed_data
    except Exception as e:
        logger.error(f"Error applying pipeline: {e}")
        return data

def save_feature_dataset(data: pd.DataFrame, name: str, version: str = None) -> str:
    """Save a feature dataset to disk."""
    if version is None:
        version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    file_path = FEATURES_DIR / f"{name}_{version}.csv"
    logger.info(f"Saving feature dataset to {file_path}")
    
    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Dataset saved successfully: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return "" 