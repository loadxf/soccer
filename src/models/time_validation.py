"""
Time-Based Cross-Validation Module

This module implements time-aware cross-validation strategies for soccer prediction models.
Proper temporal validation is critical for soccer predictions to avoid data leakage.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Optional, Tuple, Union, List
from datetime import datetime, timedelta
from sklearn.model_selection import BaseCrossValidator
import math

# Import project components
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("models.time_validation")


class TimeSeriesSplit(BaseCrossValidator):
    """
    Time Series cross-validator that respects temporal order of observations.
    
    Provides train/test indices to split time series data samples such that 
    all training samples occur before validation samples. This cross-validation
    object is designed to respect the temporal order of soccer matches and 
    prevent data leakage from future matches.
    
    Parameters:
        n_splits: Number of splits. Must be at least 2.
        test_size: Number of samples in the test set. If a float is given,
                   it is interpreted as a fraction (0.0 to 1.0) of the total samples.
        gap: Number of samples to exclude between training and test sets.
             This is important to simulate real prediction scenarios in soccer
             where you typically predict multiple days/weeks in advance.
        max_train_size: Maximum size for a single training set. Can be absolute or
                        relative (if float < 1.0).
    """
    
    def __init__(self, n_splits: int = 5, test_size: Union[int, float] = 0.2, 
                 gap: int = 0, max_train_size: Optional[Union[int, float]] = None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
    
    def split(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None, 
              groups: Optional[Union[np.ndarray, pd.Series]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Parameters:
            X: Training data. Time series samples.
            y: Target variable (ignored, included for API consistency).
            groups: Sample dates/times for time-based splitting. If not provided,
                   assumes X is sorted chronologically.
                   
        Yields:
            train_index, test_index: arrays of indices for training and testing.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        # Handle test_size as a fraction
        if isinstance(self.test_size, float) and 0.0 < self.test_size <= 1.0:
            test_size = max(1, int(n_samples * self.test_size))
        else:
            test_size = self.test_size
        
        # Handle max_train_size as a fraction
        max_train_size = self.max_train_size
        if isinstance(max_train_size, float) and max_train_size < 1.0:
            max_train_size = int(n_samples * max_train_size)
        
        # Make sure we have enough samples
        if test_size + self.gap >= n_samples:
            raise ValueError(
                f"Too few samples ({n_samples}) to split into train and test sets "
                f"with test_size={test_size} and gap={self.gap}."
            )
        
        # Calculate fold sizes
        n_splits = self.n_splits
        
        # Ensure there's enough data for the requested splits
        if (test_size + self.gap) * n_splits > n_samples:
            logger.warning(
                f"Reducing number of splits from {n_splits} to "
                f"{n_samples // (test_size + self.gap)} due to insufficient data."
            )
            n_splits = max(2, n_samples // (test_size + self.gap))
        
        # Determine split points
        split_points = []
        
        # If groups is provided (dates), use dates to determine splits
        if groups is not None:
            # Convert groups to datetime if it's not already
            if not isinstance(groups, pd.Series):
                groups = pd.Series(groups)
            
            if not pd.api.types.is_datetime64_dtype(groups):
                groups = pd.to_datetime(groups)
            
            # Sort unique dates
            unique_dates = groups.sort_values().unique()
            n_dates = len(unique_dates)
            
            # Calculate test periods
            date_test_size = max(1, int(n_dates * (test_size / n_samples)))
            
            # Create evenly spaced splits across the date range
            date_splits = []
            for i in range(n_splits):
                # Get end of this test set
                if i == n_splits - 1:
                    test_end_idx = n_dates - 1
                else:
                    test_end_idx = n_dates - 1 - i * (n_dates // n_splits)
                
                # Get start of this test set
                test_start_idx = max(0, test_end_idx - date_test_size + 1)
                
                # Get end of train set (accounting for gap)
                gap_idx = max(0, min(test_start_idx - 1, int(test_start_idx * self.gap / test_size)))
                train_end_idx = max(0, test_start_idx - 1 - gap_idx)
                
                # Get start of train set
                if max_train_size is not None:
                    date_train_size = max(1, int(n_dates * (max_train_size / n_samples)))
                    train_start_idx = max(0, train_end_idx - date_train_size + 1)
                else:
                    train_start_idx = 0
                
                # Add split points
                date_splits.append({
                    'train_start': unique_dates[train_start_idx],
                    'train_end': unique_dates[train_end_idx],
                    'test_start': unique_dates[test_start_idx],
                    'test_end': unique_dates[test_end_idx]
                })
            
            # Convert date splits to indices
            for split in date_splits:
                train_mask = (groups >= split['train_start']) & (groups <= split['train_end'])
                test_mask = (groups >= split['test_start']) & (groups <= split['test_end'])
                
                train_indices = indices[train_mask]
                test_indices = indices[test_mask]
                
                # Skip if either set is empty
                if len(train_indices) == 0 or len(test_indices) == 0:
                    continue
                    
                yield train_indices, test_indices
        
        else:
            # Time-based splits without date information
            for i in range(n_splits):
                # Get end of this test set
                if i == n_splits - 1:
                    test_end = n_samples - 1
                else:
                    test_end = n_samples - 1 - i * (n_samples // n_splits)
                
                # Get start of this test set
                test_start = max(0, test_end - test_size + 1)
                
                # Get end of train set (with gap)
                train_end = max(0, test_start - self.gap - 1)
                
                # Get start of train set
                if max_train_size is not None:
                    train_start = max(0, train_end - max_train_size + 1)
                else:
                    train_start = 0
                
                yield indices[train_start:train_end + 1], indices[test_start:test_end + 1]
    
    def get_n_splits(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                    y: Optional[Union[np.ndarray, pd.Series]] = None, 
                    groups: Optional[Union[np.ndarray, pd.Series]] = None) -> int:
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


class SeasonBasedSplit(BaseCrossValidator):
    """
    Season-based cross-validator for soccer predictions.
    
    This cross-validator uses soccer seasons to define train/test splits,
    which is a common and intuitive approach in soccer prediction. It ensures
    all training samples come from past seasons relative to test samples.
    
    Parameters:
        test_seasons: Number of seasons to use for testing. Default is 1.
        max_train_seasons: Maximum number of past seasons to use for training. 
                          If None, all available past seasons are used.
        rolling: If True, generate rolling forecasts, otherwise single train-test split.
        season_column: Name of the column containing season information.
    """
    
    def __init__(self, test_seasons: int = 1, max_train_seasons: Optional[int] = None,
                 rolling: bool = True, season_column: str = 'season'):
        self.test_seasons = test_seasons
        self.max_train_seasons = max_train_seasons
        self.rolling = rolling
        self.season_column = season_column
    
    def split(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None, 
              groups: Optional[Union[np.ndarray, pd.Series]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set by season.
        
        Parameters:
            X: Training data. If DataFrame, must contain season_column.
            y: Target variable (ignored, included for API consistency).
            groups: Optional season labels. If not provided, X must be a DataFrame 
                   with season_column.
                   
        Yields:
            train_index, test_index: arrays of indices for training and testing.
        """
        # Get season information
        if groups is not None:
            seasons = groups
        elif isinstance(X, pd.DataFrame) and self.season_column in X.columns:
            seasons = X[self.season_column]
        else:
            raise ValueError(
                f"Season information must be provided either through groups parameter "
                f"or as a column '{self.season_column}' in X DataFrame."
            )
        
        # Convert to Series if needed
        if not isinstance(seasons, pd.Series):
            seasons = pd.Series(seasons)
        
        # Get unique seasons in chronological order
        unique_seasons = sorted(seasons.unique())
        n_seasons = len(unique_seasons)
        
        if n_seasons < self.test_seasons + 1:
            raise ValueError(
                f"At least {self.test_seasons + 1} seasons required for splitting, "
                f"but only {n_seasons} seasons found."
            )
        
        # Create indices array
        indices = np.arange(len(seasons))
        
        # Generate splits
        if self.rolling:
            # Generate rolling forecast splits
            for i in range(n_seasons - self.test_seasons):
                # Test seasons
                test_season_idx = unique_seasons[i + 1:i + 1 + self.test_seasons]
                test_mask = seasons.isin(test_season_idx)
                test_indices = indices[test_mask]
                
                # Train seasons (all previous seasons up to max_train_seasons)
                if self.max_train_seasons is not None:
                    train_season_idx = unique_seasons[max(0, i + 1 - self.max_train_seasons):i + 1]
                else:
                    train_season_idx = unique_seasons[:i + 1]
                
                train_mask = seasons.isin(train_season_idx)
                train_indices = indices[train_mask]
                
                yield train_indices, test_indices
        else:
            # Single train-test split with most recent seasons as test
            test_season_idx = unique_seasons[-self.test_seasons:]
            test_mask = seasons.isin(test_season_idx)
            test_indices = indices[test_mask]
            
            # Train on older seasons
            if self.max_train_seasons is not None:
                train_season_idx = unique_seasons[-self.test_seasons - self.max_train_seasons:-self.test_seasons]
            else:
                train_season_idx = unique_seasons[:-self.test_seasons]
            
            train_mask = seasons.isin(train_season_idx)
            train_indices = indices[train_mask]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                    y: Optional[Union[np.ndarray, pd.Series]] = None, 
                    groups: Optional[Union[np.ndarray, pd.Series]] = None) -> int:
        """Returns the number of splitting iterations in the cross-validator"""
        if not self.rolling:
            return 1
        
        # Get season information to determine number of splits
        if groups is not None:
            seasons = groups
        elif X is not None and isinstance(X, pd.DataFrame) and self.season_column in X.columns:
            seasons = X[self.season_column]
        else:
            # Without data, return default behavior
            return max(1, self._n_seasons - self.test_seasons) if hasattr(self, '_n_seasons') else 1
        
        # Convert to Series if needed
        if not isinstance(seasons, pd.Series):
            seasons = pd.Series(seasons)
        
        # Get unique seasons
        n_seasons = len(set(seasons))
        
        return max(1, n_seasons - self.test_seasons)


class MatchDayBasedSplit(BaseCrossValidator):
    """
    Match-day based cross-validator for soccer predictions.
    
    This cross-validator splits data based on match days within a season,
    allowing for more granular testing scenarios that simulate predicting
    each match day using only information from previous match days.
    
    Parameters:
        n_future_match_days: Number of match days to predict at once.
        n_test_periods: Number of test periods to generate.
        min_train_match_days: Minimum number of match days needed for training.
        date_column: Name of the column containing match dates.
        season_column: Name of the column containing season information.
    """
    
    def __init__(self, n_future_match_days: int = 1, n_test_periods: int = 10,
                 min_train_match_days: int = 3, date_column: str = 'date',
                 season_column: str = 'season'):
        self.n_future_match_days = n_future_match_days
        self.n_test_periods = n_test_periods
        self.min_train_match_days = min_train_match_days
        self.date_column = date_column
        self.season_column = season_column
    
    def split(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None, 
              groups: Optional[Union[np.ndarray, pd.Series]] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set by match day.
        
        Parameters:
            X: Training data. Must be a DataFrame with date_column and season_column.
            y: Target variable (ignored, included for API consistency).
            groups: Optional match dates. If not provided, X must have date_column.
                   
        Yields:
            train_index, test_index: arrays of indices for training and testing.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame for MatchDayBasedSplit")
        
        # Get date and season information
        if self.date_column not in X.columns:
            raise ValueError(f"X must contain a column named '{self.date_column}'")
        
        if self.season_column not in X.columns:
            raise ValueError(f"X must contain a column named '{self.season_column}'")
        
        # Create indices array
        indices = np.arange(len(X))
        
        # Process each season separately
        for season in X[self.season_column].unique():
            season_mask = X[self.season_column] == season
            season_data = X[season_mask].copy()
            
            # Get unique match dates for this season, sorted chronologically
            match_dates = sorted(season_data[self.date_column].unique())
            
            if len(match_dates) < self.min_train_match_days + self.n_future_match_days:
                # Skip seasons with insufficient data
                logger.warning(f"Season {season} has insufficient match days ({len(match_dates)}), skipping.")
                continue
            
            # Determine number of test periods for this season
            n_possible_splits = len(match_dates) - self.min_train_match_days - self.n_future_match_days + 1
            if n_possible_splits <= 0:
                continue
                
            n_test_periods = min(self.n_test_periods, n_possible_splits)
            
            # Calculate step size for evenly distributed test periods
            if n_test_periods > 1:
                step = max(1, n_possible_splits // n_test_periods)
            else:
                step = 1
            
            # Generate splits
            for i in range(0, n_possible_splits, step):
                if i // step >= n_test_periods:
                    break
                    
                # Split point is the first day of the test period
                split_idx = self.min_train_match_days + i
                
                # Training data: all matches on or before the split date
                train_dates = match_dates[:split_idx]
                train_mask = season_data[self.date_column].isin(train_dates)
                train_indices = indices[season_mask][train_mask.values]
                
                # Test data: next n_future_match_days after the split
                test_dates = match_dates[split_idx:split_idx + self.n_future_match_days]
                test_mask = season_data[self.date_column].isin(test_dates)
                test_indices = indices[season_mask][test_mask.values]
                
                # Skip if either set is empty
                if len(train_indices) == 0 or len(test_indices) == 0:
                    continue
                    
                yield train_indices, test_indices
    
    def get_n_splits(self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None, 
                    y: Optional[Union[np.ndarray, pd.Series]] = None, 
                    groups: Optional[Union[np.ndarray, pd.Series]] = None) -> int:
        """
        Returns the maximum number of splitting iterations in the cross-validator.
        The actual number may be less depending on the data.
        """
        # Without data, we can only return the requested number of test periods
        if X is None:
            return self.n_test_periods
        
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            return self.n_test_periods
        
        # Check if we have the necessary columns
        if self.date_column not in X.columns or self.season_column not in X.columns:
            return self.n_test_periods
        
        # Count splits across all seasons
        n_splits = 0
        for season in X[self.season_column].unique():
            season_mask = X[self.season_column] == season
            season_data = X[season_mask]
            
            # Get unique match dates for this season
            match_dates = sorted(season_data[self.date_column].unique())
            
            # Calculate possible splits for this season
            n_possible_splits = len(match_dates) - self.min_train_match_days - self.n_future_match_days + 1
            if n_possible_splits <= 0:
                continue
                
            # Add splits for this season (capped by n_test_periods)
            n_splits += min(self.n_test_periods, n_possible_splits)
        
        return max(1, n_splits) 