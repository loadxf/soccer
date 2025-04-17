"""
Tests for time-based cross-validation strategies.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the modules to test
from src.models.time_validation import TimeSeriesSplit, SeasonBasedSplit, MatchDayBasedSplit


class TestTimeSeriesSplit(unittest.TestCase):
    """Test cases for the TimeSeriesSplit class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a test dataset with a sequence of dates
        self.n_samples = 100
        self.X = np.random.rand(self.n_samples, 5)  # 5 features
        self.y = np.random.choice([0, 1, 2], size=self.n_samples)  # 3 classes
        
        # Create dates for temporal validation
        start_date = datetime(2023, 1, 1)
        self.dates = pd.date_range(start=start_date, periods=self.n_samples, freq='D')
    
    def test_basic_split(self):
        """Test basic functionality of TimeSeriesSplit."""
        # Create a splitter
        cv = TimeSeriesSplit(n_splits=3, test_size=20)
        
        # Generate splits
        splits = list(cv.split(self.X, self.y))
        
        # Check the number of splits
        self.assertEqual(len(splits), 3)
        
        # Verify that each split respects temporal order
        for train_idx, test_idx in splits:
            # Check that indices are valid
            self.assertTrue(np.all(train_idx < len(self.X)))
            self.assertTrue(np.all(test_idx < len(self.X)))
            
            # Check that train and test don't overlap
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            
            # Check that all train samples come before test samples
            self.assertTrue(np.all(np.max(train_idx) < np.min(test_idx)))
            
            # Check test size
            self.assertLessEqual(len(test_idx), 20)
    
    def test_gap_parameter(self):
        """Test the gap parameter in TimeSeriesSplit."""
        # Create a splitter with a gap
        gap = 5
        cv = TimeSeriesSplit(n_splits=3, test_size=20, gap=gap)
        
        # Generate splits
        splits = list(cv.split(self.X, self.y))
        
        # Verify that gap is respected
        for train_idx, test_idx in splits:
            # Check that there's a gap between train and test
            self.assertGreaterEqual(np.min(test_idx) - np.max(train_idx), gap + 1)
    
    def test_date_based_split(self):
        """Test TimeSeriesSplit with date groups."""
        # Create a splitter
        cv = TimeSeriesSplit(n_splits=3, test_size=0.2)
        
        # Generate splits using dates
        splits = list(cv.split(self.X, self.y, groups=self.dates))
        
        # Check the number of splits
        self.assertEqual(len(splits), 3)
        
        # Verify that splits respect dates
        for train_idx, test_idx in splits:
            train_dates = self.dates[train_idx]
            test_dates = self.dates[test_idx]
            
            # Check that dates are in order
            self.assertTrue(train_dates.max() < test_dates.min())


class TestSeasonBasedSplit(unittest.TestCase):
    """Test cases for the SeasonBasedSplit class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a test dataset with seasons
        seasons = []
        for season in range(2018, 2023):  # 5 seasons
            n_matches = 38  # Typical season length
            seasons.extend([season] * n_matches)
        
        self.n_samples = len(seasons)
        self.X = np.random.rand(self.n_samples, 5)  # 5 features
        self.y = np.random.choice([0, 1, 2], size=self.n_samples)  # 3 classes
        self.seasons = pd.Series(seasons)
        
        # Create a DataFrame with season column
        self.df = pd.DataFrame({
            'feature1': np.random.rand(self.n_samples),
            'feature2': np.random.rand(self.n_samples),
            'season': self.seasons,
            'result': self.y
        })
    
    def test_rolling_split(self):
        """Test rolling forecasts in SeasonBasedSplit."""
        # Create a splitter with rolling forecasts
        cv = SeasonBasedSplit(test_seasons=1, rolling=True)
        
        # Generate splits
        splits = list(cv.split(self.df, self.y, groups=self.seasons))
        
        # Check the number of splits (should be seasons - test_seasons)
        self.assertEqual(len(splits), 4)  # 5 seasons - 1 test season
        
        # Verify that splits respect seasons
        for i, (train_idx, test_idx) in enumerate(splits):
            train_seasons = self.seasons[train_idx].unique()
            test_seasons = self.seasons[test_idx].unique()
            
            # Test set should be a single season
            self.assertEqual(len(test_seasons), 1)
            
            # Check that training is on past seasons relative to test
            self.assertTrue(np.all(train_seasons < test_seasons))
            
            # Season should advance each split
            if i > 0:
                prev_test = self.seasons[list(cv.split(self.df, self.y, groups=self.seasons))[i-1][1]].unique()[0]
                curr_test = test_seasons[0]
                self.assertEqual(prev_test + 1, curr_test)
    
    def test_non_rolling_split(self):
        """Test single train-test split in SeasonBasedSplit."""
        # Create a splitter without rolling forecasts
        cv = SeasonBasedSplit(test_seasons=1, rolling=False)
        
        # Generate splits
        splits = list(cv.split(self.df, self.y, groups=self.seasons))
        
        # Check that there's only one split
        self.assertEqual(len(splits), 1)
        
        # Verify that the split is correct
        train_idx, test_idx = splits[0]
        train_seasons = self.seasons[train_idx].unique()
        test_seasons = self.seasons[test_idx].unique()
        
        # Test set should be the last season
        self.assertEqual(len(test_seasons), 1)
        self.assertEqual(test_seasons[0], self.seasons.max())
        
        # Training set should be all other seasons
        self.assertEqual(len(train_seasons), 4)  # 5 total - 1 test
    
    def test_max_train_seasons(self):
        """Test the max_train_seasons parameter."""
        # Create a splitter with limited training seasons
        max_train = 2
        cv = SeasonBasedSplit(test_seasons=1, max_train_seasons=max_train, rolling=True)
        
        # Generate splits
        splits = list(cv.split(self.df, self.y, groups=self.seasons))
        
        # Verify that max_train_seasons is respected
        for train_idx, test_idx in splits:
            train_seasons = self.seasons[train_idx].unique()
            self.assertLessEqual(len(train_seasons), max_train)


class TestMatchDayBasedSplit(unittest.TestCase):
    """Test cases for the MatchDayBasedSplit class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a test dataset with match days within seasons
        data = []
        for season in range(2020, 2023):  # 3 seasons
            for match_day in range(1, 39):  # 38 match days per season
                # Each match day has 10 matches
                for match in range(10):
                    date = datetime(season, 8, 1) + timedelta(days=(match_day-1)*7 + match//5)
                    data.append({
                        'season': season,
                        'date': date,
                        'match_day': match_day,
                        'feature1': np.random.rand(),
                        'feature2': np.random.rand(),
                        'result': np.random.choice([0, 1, 2])
                    })
        
        self.df = pd.DataFrame(data)
    
    def test_match_day_split(self):
        """Test basic functionality of MatchDayBasedSplit."""
        # Create a splitter
        cv = MatchDayBasedSplit(n_future_match_days=1, n_test_periods=5, min_train_match_days=3)
        
        # Generate splits
        splits = list(cv.split(self.df))
        
        # Check that we got splits
        self.assertGreater(len(splits), 0)
        
        # Verify that splits respect match days
        for train_idx, test_idx in splits:
            train_dates = self.df.iloc[train_idx]['date'].unique()
            test_dates = self.df.iloc[test_idx]['date'].unique()
            
            # Check that there's no overlap
            self.assertEqual(len(set(train_dates) & set(test_dates)), 0)
            
            # Check that training dates are before test dates
            self.assertTrue(train_dates.max() < test_dates.min())
            
            # Check that each season is handled separately
            train_seasons = self.df.iloc[train_idx]['season'].unique()
            test_seasons = self.df.iloc[test_idx]['season'].unique()
            
            # Training and test should be in the same season
            self.assertEqual(set(train_seasons), set(test_seasons))
    
    def test_n_future_match_days(self):
        """Test the n_future_match_days parameter."""
        # Create a splitter with multiple future match days
        n_future = 3
        cv = MatchDayBasedSplit(n_future_match_days=n_future, n_test_periods=5, min_train_match_days=3)
        
        # Generate splits
        splits = list(cv.split(self.df))
        
        # Verify that n_future_match_days is respected
        for train_idx, test_idx in splits:
            # Get unique dates in test set
            test_dates = self.df.iloc[test_idx]['date'].unique()
            test_match_days = self.df.iloc[test_idx]['match_day'].unique()
            
            # Check that we're testing at most n_future match days
            self.assertLessEqual(len(test_match_days), n_future)


if __name__ == '__main__':
    unittest.main() 