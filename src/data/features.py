"""
Feature engineering module for the Soccer Prediction System.
Contains functions to create features from processed data.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Import project components
from src.utils.logger import get_logger
try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("data.features")

# Define paths
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

# Ensure features directory exists
os.makedirs(FEATURES_DIR, exist_ok=True)


def load_processed_data(dataset_name: str) -> Dict[str, pd.DataFrame]:
    """
    Load processed data for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of dataframes for the dataset
    """
    dataset_dir = os.path.join(PROCESSED_DATA_DIR, dataset_name)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Processed data directory not found: {dataset_dir}")
    
    result = {}
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")
    
    # Load each CSV file
    for file in csv_files:
        name = os.path.splitext(file)[0]
        file_path = os.path.join(dataset_dir, file)
        result[name] = pd.read_csv(file_path)
        logger.info(f"Loaded {file} with {len(result[name])} rows")
    
    return result


def calculate_team_form(games_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculate team form based on recent match results.
    
    Args:
        games_df: DataFrame containing match data
        window: Number of previous matches to consider for form
        
    Returns:
        pd.DataFrame: DataFrame with team form features
    """
    # Ensure games are sorted by date
    games_df = games_df.sort_values('date')
    
    # Create separate dataframes for home and away performance
    home_team_results = games_df[['date', 'home_club_id', 'home_club_goals', 'away_club_goals']].copy()
    away_team_results = games_df[['date', 'away_club_id', 'home_club_goals', 'away_club_goals']].copy()
    
    # Rename columns for consistency
    home_team_results.columns = ['date', 'team_id', 'goals_scored', 'goals_conceded']
    away_team_results.columns = ['date', 'team_id', 'goals_conceded', 'goals_scored']
    
    # Combine home and away results
    team_results = pd.concat([home_team_results, away_team_results])
    team_results = team_results.sort_values(['team_id', 'date'])
    
    # Calculate match outcome (win=3, draw=1, loss=0)
    team_results['points'] = 1  # Default for draw
    team_results.loc[team_results['goals_scored'] > team_results['goals_conceded'], 'points'] = 3
    team_results.loc[team_results['goals_scored'] < team_results['goals_conceded'], 'points'] = 0
    
    # Calculate additional stats
    team_results['goal_diff'] = team_results['goals_scored'] - team_results['goals_conceded']
    team_results['win'] = (team_results['points'] == 3).astype(int)
    team_results['draw'] = (team_results['points'] == 1).astype(int)
    team_results['loss'] = (team_results['points'] == 0).astype(int)
    
    # Calculate rolling stats
    form_features = []
    for team_id in team_results['team_id'].unique():
        team_data = team_results[team_results['team_id'] == team_id].copy()
        
        # Calculate rolling stats
        team_data[f'last_{window}_points'] = team_data['points'].rolling(window=window, min_periods=1).sum()
        team_data[f'last_{window}_goals_scored'] = team_data['goals_scored'].rolling(window=window, min_periods=1).sum()
        team_data[f'last_{window}_goals_conceded'] = team_data['goals_conceded'].rolling(window=window, min_periods=1).sum()
        team_data[f'last_{window}_goal_diff'] = team_data['goal_diff'].rolling(window=window, min_periods=1).sum()
        team_data[f'last_{window}_wins'] = team_data['win'].rolling(window=window, min_periods=1).sum()
        team_data[f'last_{window}_draws'] = team_data['draw'].rolling(window=window, min_periods=1).sum()
        team_data[f'last_{window}_losses'] = team_data['loss'].rolling(window=window, min_periods=1).sum()
        
        # Calculate averages
        team_data[f'avg_{window}_goals_scored'] = team_data['goals_scored'].rolling(window=window, min_periods=1).mean()
        team_data[f'avg_{window}_goals_conceded'] = team_data['goals_conceded'].rolling(window=window, min_periods=1).mean()
        team_data[f'avg_{window}_goal_diff'] = team_data['goal_diff'].rolling(window=window, min_periods=1).mean()
        
        # Calculate streaks
        team_data['win_streak'] = team_data['win'].groupby((team_data['win'] != team_data['win'].shift()).cumsum()).cumsum()
        team_data['loss_streak'] = team_data['loss'].groupby((team_data['loss'] != team_data['loss'].shift()).cumsum()).cumsum()
        team_data['unbeaten_streak'] = team_data['win_or_draw'] = ((team_data['win'] + team_data['draw']) > 0).astype(int)
        team_data['unbeaten_streak'] = team_data['win_or_draw'].groupby((team_data['win_or_draw'] != team_data['win_or_draw'].shift()).cumsum()).cumsum()
        
        # Calculate win percentage
        team_data[f'win_pct_{window}'] = team_data['win'].rolling(window=window, min_periods=1).mean()
        
        # Add to form features
        form_features.append(team_data)
    
    # Combine all team form features
    form_df = pd.concat(form_features)
    
    # Clean up temporary columns
    form_df = form_df.drop(['win', 'draw', 'loss', 'win_or_draw'], axis=1)
    
    return form_df


def calculate_player_form(appearances_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calculate player form based on recent performances.
    
    Args:
        appearances_df: DataFrame containing player appearance data
        window: Number of previous matches to consider for form
        
    Returns:
        pd.DataFrame: DataFrame with player form features
    """
    # Ensure appearances are sorted by date
    appearances_df = appearances_df.sort_values(['player_id', 'date'])
    
    player_features = []
    
    for player_id in appearances_df['player_id'].unique():
        player_data = appearances_df[appearances_df['player_id'] == player_id].copy()
        
        # Skip players with too few appearances
        if len(player_data) < 2:
            continue
        
        # Calculate basic stats for the player
        player_data['goals_per_minute'] = player_data['goals'] / player_data['minutes_played'].replace(0, 90)
        player_data['assists_per_minute'] = player_data['assists'] / player_data['minutes_played'].replace(0, 90)
        
        # Calculate rolling averages
        player_data[f'last_{window}_goals'] = player_data['goals'].rolling(window=window, min_periods=1).sum()
        player_data[f'last_{window}_assists'] = player_data['assists'].rolling(window=window, min_periods=1).sum()
        player_data[f'last_{window}_minutes'] = player_data['minutes_played'].rolling(window=window, min_periods=1).sum()
        
        # Calculate averages per 90 minutes
        player_data[f'last_{window}_goals_per_90'] = (player_data[f'last_{window}_goals'] / player_data[f'last_{window}_minutes']) * 90
        player_data[f'last_{window}_assists_per_90'] = (player_data[f'last_{window}_assists'] / player_data[f'last_{window}_minutes']) * 90
        
        # Calculate form trend (recent performances vs overall average)
        overall_goals_per_90 = (player_data['goals'].sum() / player_data['minutes_played'].sum()) * 90
        player_data['goals_form_trend'] = player_data[f'last_{window}_goals_per_90'] - overall_goals_per_90
        
        overall_assists_per_90 = (player_data['assists'].sum() / player_data['minutes_played'].sum()) * 90
        player_data['assists_form_trend'] = player_data[f'last_{window}_assists_per_90'] - overall_assists_per_90
        
        # Add player_data to player_features
        player_features.append(player_data)
    
    if not player_features:
        logger.warning("No player features calculated - insufficient data")
        return pd.DataFrame()
    
    # Combine all player features
    player_form_df = pd.concat(player_features)
    
    # Fill NaN values with 0
    numeric_cols = player_form_df.select_dtypes(include=[np.number]).columns
    player_form_df[numeric_cols] = player_form_df[numeric_cols].fillna(0)
    
    return player_form_df


def create_match_feature_dataset(games_df: pd.DataFrame, team_form_df: pd.DataFrame,
                                player_form_df: Optional[pd.DataFrame] = None,
                                lookback_days: int = 30) -> pd.DataFrame:
    """
    Create a feature dataset for match prediction.
    
    Args:
        games_df: DataFrame containing match data
        team_form_df: DataFrame containing team form data
        player_form_df: Optional DataFrame containing player form data
        lookback_days: Maximum number of days to look back for form data
        
    Returns:
        pd.DataFrame: Feature dataset for match prediction
    """
    # Create a copy of games dataframe
    matches = games_df.copy()
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_dtype(matches['date']):
        matches['date'] = pd.to_datetime(matches['date'])
    
    # Create features for each match
    match_features = []
    
    for _, match in matches.iterrows():
        match_date = match['date']
        home_team_id = match['home_club_id']
        away_team_id = match['away_club_id']
        
        # Get team form data before the match
        home_form = team_form_df[(team_form_df['team_id'] == home_team_id) & 
                                (team_form_df['date'] < match_date) &
                                (team_form_df['date'] >= match_date - pd.Timedelta(days=lookback_days))]
        
        away_form = team_form_df[(team_form_df['team_id'] == away_team_id) & 
                                (team_form_df['date'] < match_date) &
                                (team_form_df['date'] >= match_date - pd.Timedelta(days=lookback_days))]
        
        # Skip matches with insufficient form data
        if home_form.empty or away_form.empty:
            continue
        
        # Get the most recent form data for each team
        home_form = home_form.sort_values('date', ascending=False).iloc[0]
        away_form = away_form.sort_values('date', ascending=False).iloc[0]
        
        # Create feature dict for this match
        feature_dict = {
            'match_id': match['game_id'],
            'date': match_date,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'competition_id': match.get('competition_id', None)
        }
        
        # Add form features for home team (with prefix)
        for col in home_form.index:
            if col not in ['date', 'team_id']:
                feature_dict[f'home_{col}'] = home_form[col]
        
        # Add form features for away team (with prefix)
        for col in away_form.index:
            if col not in ['date', 'team_id']:
                feature_dict[f'away_{col}'] = away_form[col]
        
        # Add head-to-head features if available
        h2h_matches = matches[(matches['date'] < match_date) &
                            (matches['date'] >= match_date - pd.Timedelta(days=365*2)) &  # Last 2 years
                            (((matches['home_club_id'] == home_team_id) & (matches['away_club_id'] == away_team_id)) |
                            ((matches['home_club_id'] == away_team_id) & (matches['away_club_id'] == home_team_id)))]
        
        if not h2h_matches.empty:
            home_wins = 0
            away_wins = 0
            draws = 0
            
            for _, h2h in h2h_matches.iterrows():
                if h2h['home_club_id'] == home_team_id:
                    if h2h['home_club_goals'] > h2h['away_club_goals']:
                        home_wins += 1
                    elif h2h['home_club_goals'] < h2h['away_club_goals']:
                        away_wins += 1
                    else:
                        draws += 1
                else:  # away_team was home in this h2h match
                    if h2h['home_club_goals'] > h2h['away_club_goals']:
                        away_wins += 1
                    elif h2h['home_club_goals'] < h2h['away_club_goals']:
                        home_wins += 1
                    else:
                        draws += 1
            
            feature_dict['h2h_home_wins'] = home_wins
            feature_dict['h2h_away_wins'] = away_wins
            feature_dict['h2h_draws'] = draws
            feature_dict['h2h_total'] = len(h2h_matches)
            
            # Calculate win percentages
            feature_dict['h2h_home_win_pct'] = home_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0
            feature_dict['h2h_away_win_pct'] = away_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0
            feature_dict['h2h_draw_pct'] = draws / len(h2h_matches) if len(h2h_matches) > 0 else 0
        else:
            # No head-to-head history
            feature_dict['h2h_home_wins'] = 0
            feature_dict['h2h_away_wins'] = 0
            feature_dict['h2h_draws'] = 0
            feature_dict['h2h_total'] = 0
            feature_dict['h2h_home_win_pct'] = 0
            feature_dict['h2h_away_win_pct'] = 0
            feature_dict['h2h_draw_pct'] = 0
        
        # Add target variables if available
        feature_dict['home_goals'] = match['home_club_goals']
        feature_dict['away_goals'] = match['away_club_goals']
        
        if match['home_club_goals'] > match['away_club_goals']:
            feature_dict['result'] = 'home_win'
        elif match['home_club_goals'] < match['away_club_goals']:
            feature_dict['result'] = 'away_win'
        else:
            feature_dict['result'] = 'draw'
        
        # Add to match features
        match_features.append(feature_dict)
    
    # Create DataFrame from features
    match_features_df = pd.DataFrame(match_features)
    
    # Return the features DataFrame
    return match_features_df


def create_feature_datasets(dataset_name: str = "transfermarkt") -> Dict[str, pd.DataFrame]:
    """
    Create feature datasets from processed data.
    
    Args:
        dataset_name: Name of the dataset to process
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of feature dataframes
    """
    logger.info(f"Creating feature datasets for {dataset_name}")
    
    # Load processed data
    try:
        data_dict = load_processed_data(dataset_name)
    except FileNotFoundError as e:
        logger.error(f"Error loading processed data: {e}")
        return {}
    
    dataset_dir = os.path.join(FEATURES_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    result = {}
    
    if dataset_name == "transfermarkt":
        # Calculate team form
        games_df = data_dict.get("games_processed")
        if games_df is not None:
            team_form_df = calculate_team_form(games_df)
            team_form_df.to_csv(os.path.join(dataset_dir, "team_form.csv"), index=False)
            result["team_form"] = team_form_df
            logger.info(f"Created team form features with {len(team_form_df)} rows")
            
            # Calculate player form if appearance data is available
            appearances_df = data_dict.get("appearances_processed")
            if appearances_df is not None:
                player_form_df = calculate_player_form(appearances_df)
                player_form_df.to_csv(os.path.join(dataset_dir, "player_form.csv"), index=False)
                result["player_form"] = player_form_df
                logger.info(f"Created player form features with {len(player_form_df)} rows")
            
            # Create match features dataset
            match_features_df = create_match_feature_dataset(games_df, team_form_df)
            match_features_df.to_csv(os.path.join(dataset_dir, "match_features.csv"), index=False)
            result["match_features"] = match_features_df
            logger.info(f"Created match features dataset with {len(match_features_df)} rows")
    
    return result


def create_feature_pipeline(df: pd.DataFrame, 
                          numeric_cols: List[str],
                          categorical_cols: List[str],
                          target_col: Optional[str] = None) -> Tuple[ColumnTransformer, Optional[LabelEncoder]]:
    """
    Create a scikit-learn pipeline for feature transformation.
    
    Args:
        df: DataFrame containing the features
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        target_col: Optional target column name
        
    Returns:
        Tuple[ColumnTransformer, Optional[LabelEncoder]]: Feature pipeline and target encoder
    """
    # Numeric features pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Column transformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Fit the preprocessor
    preprocessor.fit(df)
    
    # Process target if provided
    target_encoder = None
    if target_col is not None and target_col in df.columns:
        if df[target_col].dtype == 'object':
            target_encoder = LabelEncoder()
            target_encoder.fit(df[target_col].dropna())
    
    return preprocessor, target_encoder


def save_feature_pipeline(pipeline: ColumnTransformer, 
                         target_encoder: Optional[LabelEncoder],
                         dataset_name: str,
                         feature_type: str) -> str:
    """
    Save a feature pipeline to disk.
    
    Args:
        pipeline: The feature transformation pipeline
        target_encoder: Optional encoder for the target variable
        dataset_name: Name of the dataset
        feature_type: Type of features
        
    Returns:
        str: Path to the saved pipeline
    """
    dataset_dir = os.path.join(FEATURES_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save pipeline
    pipeline_path = os.path.join(dataset_dir, f"{feature_type}_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    
    # Save target encoder if available
    if target_encoder is not None:
        encoder_path = os.path.join(dataset_dir, f"{feature_type}_target_encoder.joblib")
        joblib.dump(target_encoder, encoder_path)
    
    logger.info(f"Saved feature pipeline to {pipeline_path}")
    return pipeline_path


def load_feature_pipeline(dataset_name: str, feature_type: str) -> Tuple[Optional[ColumnTransformer], Optional[LabelEncoder]]:
    """
    Load a feature pipeline from disk.
    
    Args:
        dataset_name: Name of the dataset
        feature_type: Type of features
        
    Returns:
        Tuple[Optional[ColumnTransformer], Optional[LabelEncoder]]: The loaded pipeline and target encoder
    """
    dataset_dir = os.path.join(FEATURES_DIR, dataset_name)
    
    # Load pipeline
    pipeline_path = os.path.join(dataset_dir, f"{feature_type}_pipeline.joblib")
    pipeline = None
    if os.path.exists(pipeline_path):
        pipeline = joblib.load(pipeline_path)
    
    # Load target encoder if available
    encoder_path = os.path.join(dataset_dir, f"{feature_type}_target_encoder.joblib")
    target_encoder = None
    if os.path.exists(encoder_path):
        target_encoder = joblib.load(encoder_path)
    
    return pipeline, target_encoder


def apply_feature_pipeline(df: pd.DataFrame, 
                         pipeline: ColumnTransformer,
                         target_encoder: Optional[LabelEncoder] = None,
                         target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply a feature pipeline to transform data.
    
    Args:
        df: DataFrame to transform
        pipeline: Feature transformation pipeline
        target_encoder: Optional encoder for the target variable
        target_col: Optional target column name
        
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Transformed features and target
    """
    # Transform features
    X = pipeline.transform(df)
    
    # Transform target if provided
    y = None
    if target_col is not None and target_col in df.columns and target_encoder is not None:
        y = target_encoder.transform(df[target_col].dropna())
    
    return X, y


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Soccer Prediction Feature Engineering")
    parser.add_argument("--dataset", type=str, default="transfermarkt", 
                        help="Dataset to process (default: transfermarkt)")
    parser.add_argument("--feature-type", type=str, default="match_features",
                        help="Type of features to create (default: match_features)")
    
    args = parser.parse_args()
    
    # Create feature datasets
    feature_datasets = create_feature_datasets(args.dataset)
    
    if args.feature_type in feature_datasets:
        df = feature_datasets[args.feature_type]
        
        # Example feature pipeline creation
        if args.feature_type == "match_features":
            # Example numeric and categorical columns (adjust as needed)
            numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'result']
            
            # Remove non-feature columns
            for col in ['match_id', 'date', 'home_team_id', 'away_team_id', 'home_goals', 'away_goals', 'result']:
                if col in numeric_cols:
                    numeric_cols.remove(col)
                if col in categorical_cols:
                    categorical_cols.remove(col)
            
            # Create and save pipeline
            pipeline, target_encoder = create_feature_pipeline(
                df, numeric_cols, categorical_cols, "result"
            )
            
            save_feature_pipeline(pipeline, target_encoder, args.dataset, args.feature_type)
            
            # Example transformation
            X, y = apply_feature_pipeline(df, pipeline, target_encoder, "result")
            logger.info(f"Transformed features shape: {X.shape}")
            if y is not None:
                logger.info(f"Transformed target shape: {y.shape}")
        
        else:
            logger.info(f"Pipeline creation not implemented for feature type: {args.feature_type}")
    
    else:
        logger.warning(f"No features of type {args.feature_type} created") 