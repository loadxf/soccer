"""
Data augmentation techniques for the Soccer Prediction System.
Contains functions to augment data through synthetic data generation,
resampling, and other techniques to improve model performance.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import random
from datetime import datetime, timedelta
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Import project components
from src.utils.logger import get_logger
try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("data.augmentation")

# Define paths
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented")

# Ensure augmented directory exists
os.makedirs(AUGMENTED_DIR, exist_ok=True)


def oversample_minority_class(df: pd.DataFrame, target_col: str, random_state: int = 42) -> pd.DataFrame:
    """
    Oversample minority classes to balance the dataset.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Balanced dataset
    """
    # Count class frequencies
    class_counts = df[target_col].value_counts()
    majority_class = class_counts.index[0]
    majority_count = class_counts.iloc[0]
    
    logger.info(f"Original class distribution: {class_counts.to_dict()}")
    
    # Create a list to store data for each class
    balanced_data = []
    
    # Process each class
    for class_value, count in class_counts.items():
        # Get samples for this class
        class_data = df[df[target_col] == class_value]
        
        # If this is a minority class, oversample it
        if count < majority_count:
            oversampled_data = resample(
                class_data,
                replace=True,
                n_samples=majority_count,
                random_state=random_state
            )
            balanced_data.append(oversampled_data)
        else:
            balanced_data.append(class_data)
    
    # Combine all balanced classes
    balanced_df = pd.concat(balanced_data)
    
    logger.info(f"Balanced class distribution: {balanced_df[target_col].value_counts().to_dict()}")
    
    return balanced_df


def undersample_majority_class(df: pd.DataFrame, target_col: str, random_state: int = 42) -> pd.DataFrame:
    """
    Undersample majority classes to balance the dataset.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        random_state: Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Balanced dataset
    """
    # Count class frequencies
    class_counts = df[target_col].value_counts()
    minority_class = class_counts.index[-1]
    minority_count = class_counts.iloc[-1]
    
    logger.info(f"Original class distribution: {class_counts.to_dict()}")
    
    # Create a list to store data for each class
    balanced_data = []
    
    # Process each class
    for class_value, count in class_counts.items():
        # Get samples for this class
        class_data = df[df[target_col] == class_value]
        
        # If this is a majority class, undersample it
        if count > minority_count:
            undersampled_data = resample(
                class_data,
                replace=False,
                n_samples=minority_count,
                random_state=random_state
            )
            balanced_data.append(undersampled_data)
        else:
            balanced_data.append(class_data)
    
    # Combine all balanced classes
    balanced_df = pd.concat(balanced_data)
    
    logger.info(f"Balanced class distribution: {balanced_df[target_col].value_counts().to_dict()}")
    
    return balanced_df


def generate_synthetic_matches(games_df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """
    Generate synthetic match data based on real match patterns.
    
    Args:
        games_df: DataFrame containing real match data
        n_samples: Number of synthetic samples to generate
        
    Returns:
        pd.DataFrame: DataFrame with synthetic match data
    """
    # Make a copy of the input dataframe
    df = games_df.copy()
    
    # Ensure games are sorted by date
    if 'date' in df.columns:
        df = df.sort_values('date')
    
    # Extract necessary columns
    required_cols = ['home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']
    
    # Add match_date if available
    date_col = None
    for col in ['date', 'match_date']:
        if col in df.columns:
            date_col = col
            required_cols.append(col)
            break
    
    # Skip if required columns are missing
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Required column {col} missing for synthetic data generation")
            return pd.DataFrame()
    
    # Get unique team IDs
    team_ids = pd.concat([df['home_club_id'], df['away_club_id']]).unique()
    logger.info(f"Found {len(team_ids)} unique teams for synthetic data generation")
    
    # Get goal distributions
    home_goals_dist = df['home_club_goals'].value_counts(normalize=True).to_dict()
    away_goals_dist = df['away_club_goals'].value_counts(normalize=True).to_dict()
    
    # Generate synthetic data
    synthetic_data = []
    
    for i in range(n_samples):
        # Sample two different teams
        home_team = np.random.choice(team_ids)
        away_team = np.random.choice([t for t in team_ids if t != home_team])
        
        # Sample goals from distribution
        home_goals = np.random.choice(
            list(home_goals_dist.keys()),
            p=list(home_goals_dist.values())
        )
        away_goals = np.random.choice(
            list(away_goals_dist.keys()),
            p=list(away_goals_dist.values())
        )
        
        # Create match record
        match_data = {
            'home_club_id': home_team,
            'away_club_id': away_team,
            'home_club_goals': home_goals,
            'away_club_goals': away_goals,
            'is_synthetic': True
        }
        
        # Add date if needed
        if date_col:
            # Generate a random date within the range of the dataset
            if pd.api.types.is_datetime64_dtype(df[date_col]):
                min_date = df[date_col].min()
                max_date = df[date_col].max()
                date_range = (max_date - min_date).days
                random_days = np.random.randint(0, date_range)
                match_data[date_col] = min_date + timedelta(days=random_days)
            else:
                # Try to parse as string date
                try:
                    dates = pd.to_datetime(df[date_col])
                    min_date = dates.min()
                    max_date = dates.max()
                    date_range = (max_date - min_date).days
                    random_days = np.random.randint(0, date_range)
                    match_data[date_col] = (min_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
                except Exception as e:
                    logger.warning(f"Could not parse dates for synthetic data: {e}")
                    match_data[date_col] = df[date_col].sample(1).iloc[0]
        
        synthetic_data.append(match_data)
    
    # Convert to dataframe
    synthetic_df = pd.DataFrame(synthetic_data)
    
    logger.info(f"Generated {len(synthetic_df)} synthetic matches")
    
    return synthetic_df


def add_gaussian_noise(df: pd.DataFrame, columns: List[str], noise_level: float = 0.05) -> pd.DataFrame:
    """
    Add Gaussian noise to numerical columns.
    
    Args:
        df: DataFrame containing the data
        columns: List of numerical columns to add noise to
        noise_level: Standard deviation of the noise relative to column standard deviation
        
    Returns:
        pd.DataFrame: DataFrame with added noise
    """
    # Make a copy of the input dataframe
    noisy_df = df.copy()
    
    # Add noise to each column
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in dataframe, skipping")
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column {col} is not numeric, skipping")
            continue
        
        # Calculate standard deviation for the column
        col_std = df[col].std()
        
        # Generate Gaussian noise
        noise = np.random.normal(0, col_std * noise_level, size=len(df))
        
        # Add noise to the column
        noisy_df[col] = df[col] + noise
        
        logger.info(f"Added noise to column {col} (std: {col_std:.4f}, noise level: {noise_level:.4f})")
    
    return noisy_df


def augment_time_series(df: pd.DataFrame, time_col: str, value_cols: List[str], 
                        shift_values: List[int], window_sizes: List[int]) -> pd.DataFrame:
    """
    Augment time series data with lagged and rolling features.
    
    Args:
        df: DataFrame containing the time series data
        time_col: Column containing the timestamp
        value_cols: Columns containing the values to augment
        shift_values: List of time shifts to create lagged features
        window_sizes: List of window sizes for rolling statistics
        
    Returns:
        pd.DataFrame: Augmented dataframe with additional features
    """
    # Make a copy of the input dataframe
    augmented_df = df.copy()
    
    # Ensure time column is in datetime format
    if not pd.api.types.is_datetime64_dtype(augmented_df[time_col]):
        try:
            augmented_df[time_col] = pd.to_datetime(augmented_df[time_col])
        except Exception as e:
            logger.error(f"Could not convert {time_col} to datetime: {e}")
            return df
    
    # Sort by time column
    augmented_df = augmented_df.sort_values(time_col)
    
    # Create group columns if they exist (like team_id, player_id)
    group_cols = []
    for col in ['team_id', 'player_id', 'club_id', 'home_club_id', 'away_club_id']:
        if col in augmented_df.columns:
            group_cols.append(col)
    
    # Add lagged features
    for col in value_cols:
        if col not in augmented_df.columns:
            logger.warning(f"Column {col} not found in dataframe, skipping")
            continue
            
        if not pd.api.types.is_numeric_dtype(augmented_df[col]):
            logger.warning(f"Column {col} is not numeric, skipping")
            continue
            
        # Create lagged features
        for shift in shift_values:
            new_col = f"{col}_lag_{shift}"
            
            if group_cols:
                # Apply lag within each group
                augmented_df[new_col] = augmented_df.groupby(group_cols)[col].shift(shift)
            else:
                # Apply lag to entire dataset
                augmented_df[new_col] = augmented_df[col].shift(shift)
                
            logger.info(f"Created lagged feature {new_col}")
            
        # Create rolling statistics
        for window in window_sizes:
            # Rolling mean
            mean_col = f"{col}_roll_mean_{window}"
            if group_cols:
                augmented_df[mean_col] = augmented_df.groupby(group_cols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            else:
                augmented_df[mean_col] = augmented_df[col].rolling(window, min_periods=1).mean()
                
            # Rolling standard deviation
            std_col = f"{col}_roll_std_{window}"
            if group_cols:
                augmented_df[std_col] = augmented_df.groupby(group_cols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
            else:
                augmented_df[std_col] = augmented_df[col].rolling(window, min_periods=1).std()
                
            logger.info(f"Created rolling statistics {mean_col} and {std_col}")
    
    # Fill NaN values
    augmented_df = augmented_df.fillna(0)
    
    return augmented_df


def save_augmented_data(df: pd.DataFrame, dataset_name: str, augmentation_type: str) -> str:
    """
    Save augmented data to file.
    
    Args:
        df: DataFrame containing the augmented data
        dataset_name: Name of the dataset
        augmentation_type: Type of augmentation applied
        
    Returns:
        str: Path to the saved file
    """
    # Create directory for the dataset
    dataset_dir = os.path.join(AUGMENTED_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{augmentation_type}_{timestamp}.csv"
    file_path = os.path.join(dataset_dir, filename)
    
    # Save the dataframe
    df.to_csv(file_path, index=False)
    logger.info(f"Saved augmented data to {file_path}")
    
    return file_path


def run_augmentation(dataset_name: str, augmentation_type: str, **kwargs) -> pd.DataFrame:
    """
    Run a specific augmentation technique on a dataset.
    
    Args:
        dataset_name: Name of the dataset to augment
        augmentation_type: Type of augmentation to apply
        **kwargs: Additional parameters for the augmentation function
        
    Returns:
        pd.DataFrame: Augmented dataframe
    """
    # Load processed data
    from src.data.features import load_processed_data
    
    try:
        data_dict = load_processed_data(dataset_name)
    except Exception as e:
        logger.error(f"Error loading processed data for {dataset_name}: {e}")
        return pd.DataFrame()
    
    # Select the appropriate dataframe based on the augmentation type
    if augmentation_type in ['oversample', 'undersample']:
        # For class balancing, use match data if available
        if 'matches' in data_dict:
            df = data_dict['matches']
            logger.info(f"Using matches dataframe for {augmentation_type}")
        elif 'games' in data_dict:
            df = data_dict['games']
            logger.info(f"Using games dataframe for {augmentation_type}")
        else:
            # Use the first available dataframe
            name = list(data_dict.keys())[0]
            df = data_dict[name]
            logger.info(f"Using {name} dataframe for {augmentation_type}")
    elif augmentation_type == 'synthetic':
        # For synthetic data generation, use match/game data
        if 'matches' in data_dict:
            df = data_dict['matches']
        elif 'games' in data_dict:
            df = data_dict['games']
        else:
            logger.error(f"No match/game data found for synthetic data generation")
            return pd.DataFrame()
    elif augmentation_type == 'noise':
        # For adding noise, use any dataset with numerical columns
        name = list(data_dict.keys())[0]
        df = data_dict[name]
    elif augmentation_type == 'time_series':
        # For time series augmentation, use match/game data or any time series data
        if 'matches' in data_dict:
            df = data_dict['matches']
        elif 'games' in data_dict:
            df = data_dict['games']
        else:
            # Use the first available dataframe
            name = list(data_dict.keys())[0]
            df = data_dict[name]
    else:
        logger.error(f"Unknown augmentation type: {augmentation_type}")
        return pd.DataFrame()
    
    # Apply the augmentation technique
    if augmentation_type == 'oversample':
        if 'target_col' not in kwargs:
            logger.error("Missing required parameter 'target_col' for oversampling")
            return pd.DataFrame()
        augmented_df = oversample_minority_class(df, kwargs['target_col'], kwargs.get('random_state', 42))
    
    elif augmentation_type == 'undersample':
        if 'target_col' not in kwargs:
            logger.error("Missing required parameter 'target_col' for undersampling")
            return pd.DataFrame()
        augmented_df = undersample_majority_class(df, kwargs['target_col'], kwargs.get('random_state', 42))
    
    elif augmentation_type == 'synthetic':
        n_samples = kwargs.get('n_samples', 100)
        augmented_df = generate_synthetic_matches(df, n_samples)
    
    elif augmentation_type == 'noise':
        if 'columns' not in kwargs:
            logger.error("Missing required parameter 'columns' for adding noise")
            return pd.DataFrame()
        noise_level = kwargs.get('noise_level', 0.05)
        augmented_df = add_gaussian_noise(df, kwargs['columns'], noise_level)
    
    elif augmentation_type == 'time_series':
        if 'time_col' not in kwargs or 'value_cols' not in kwargs:
            logger.error("Missing required parameters for time series augmentation")
            return pd.DataFrame()
        shift_values = kwargs.get('shift_values', [1, 2, 3])
        window_sizes = kwargs.get('window_sizes', [3, 5, 10])
        augmented_df = augment_time_series(df, kwargs['time_col'], kwargs['value_cols'], 
                                           shift_values, window_sizes)
    
    # Save the augmented data if requested
    if kwargs.get('save', True):
        save_augmented_data(augmented_df, dataset_name, augmentation_type)
    
    return augmented_df 