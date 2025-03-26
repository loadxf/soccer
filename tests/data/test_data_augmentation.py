"""
Tests for data augmentation module.
"""

import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Import project components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.augmentation import (
    oversample_minority_class,
    undersample_majority_class,
    generate_synthetic_matches,
    add_gaussian_noise,
    augment_time_series,
    run_augmentation
)
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("tests.data_augmentation")


class TestDataAugmentation(unittest.TestCase):
    """Test cases for data augmentation methods."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample match dataset
        self.match_data = pd.DataFrame({
            'match_id': list(range(100)),
            'date': [datetime.now() - timedelta(days=i) for i in range(100)],
            'home_club_id': np.random.randint(1, 21, 100),
            'away_club_id': np.random.randint(1, 21, 100),
            'home_club_goals': np.random.randint(0, 6, 100),
            'away_club_goals': np.random.randint(0, 4, 100),
        })
        
        # Add a result column (for classification tasks)
        self.match_data['result'] = np.where(
            self.match_data['home_club_goals'] > self.match_data['away_club_goals'], 
            'H', 
            np.where(
                self.match_data['home_club_goals'] < self.match_data['away_club_goals'],
                'A',
                'D'
            )
        )
        
        # Create imbalanced result distribution
        # Make 'D' less common
        draw_indices = self.match_data[self.match_data['result'] == 'D'].index
        if len(draw_indices) > 10:
            self.match_data = self.match_data.drop(draw_indices[10:])
    
    def test_oversample_minority_class(self):
        """Test oversampling of minority class."""
        # Count original class distribution
        original_counts = self.match_data['result'].value_counts()
        
        # Apply oversampling
        oversampled_df = oversample_minority_class(self.match_data, 'result')
        
        # Check that all classes have the same count
        oversampled_counts = oversampled_df['result'].value_counts()
        
        self.assertEqual(
            len(set(oversampled_counts.values)), 
            1, 
            "After oversampling, all classes should have the same count"
        )
        
        # Check that minority class has been increased
        minority_class = original_counts.index[-1]
        self.assertGreater(
            oversampled_counts[minority_class],
            original_counts[minority_class],
            "Minority class count should increase after oversampling"
        )
        
        # Check that majority class remains unchanged
        majority_class = original_counts.index[0]
        self.assertEqual(
            oversampled_counts[majority_class],
            original_counts[majority_class],
            "Majority class count should remain unchanged after oversampling"
        )
    
    def test_undersample_majority_class(self):
        """Test undersampling of majority class."""
        # Count original class distribution
        original_counts = self.match_data['result'].value_counts()
        
        # Apply undersampling
        undersampled_df = undersample_majority_class(self.match_data, 'result')
        
        # Check that all classes have the same count
        undersampled_counts = undersampled_df['result'].value_counts()
        
        self.assertEqual(
            len(set(undersampled_counts.values)), 
            1, 
            "After undersampling, all classes should have the same count"
        )
        
        # Check that majority class has been decreased
        majority_class = original_counts.index[0]
        self.assertLess(
            undersampled_counts[majority_class],
            original_counts[majority_class],
            "Majority class count should decrease after undersampling"
        )
        
        # Check that minority class remains unchanged
        minority_class = original_counts.index[-1]
        self.assertEqual(
            undersampled_counts[minority_class],
            original_counts[minority_class],
            "Minority class count should remain unchanged after undersampling"
        )
    
    def test_generate_synthetic_matches(self):
        """Test generation of synthetic match data."""
        # Generate synthetic matches
        n_samples = 50
        synthetic_df = generate_synthetic_matches(self.match_data, n_samples)
        
        # Check that the correct number of samples were generated
        self.assertEqual(len(synthetic_df), n_samples, f"Should generate {n_samples} synthetic matches")
        
        # Check that required columns are present
        required_cols = ['home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals', 'is_synthetic']
        for col in required_cols:
            self.assertIn(col, synthetic_df.columns, f"Column {col} should be in synthetic data")
        
        # Check that home and away teams are different
        self.assertTrue(
            all(synthetic_df['home_club_id'] != synthetic_df['away_club_id']),
            "Home and away teams should be different in synthetic matches"
        )
        
        # Check that the synthetic flag is set
        self.assertTrue(all(synthetic_df['is_synthetic']), "is_synthetic flag should be True for all synthetic matches")
    
    def test_add_gaussian_noise(self):
        """Test adding Gaussian noise to numerical columns."""
        # Select columns to add noise to
        columns = ['home_club_goals', 'away_club_goals']
        
        # Get original values
        original_values = self.match_data[columns].copy()
        
        # Add noise
        noise_level = 0.1
        noisy_df = add_gaussian_noise(self.match_data, columns, noise_level)
        
        # Check that values have changed
        for col in columns:
            self.assertFalse(
                all(noisy_df[col] == original_values[col]),
                f"Column {col} should have changed after adding noise"
            )
        
        # Check that the mean hasn't changed significantly
        for col in columns:
            self.assertAlmostEqual(
                noisy_df[col].mean(),
                original_values[col].mean(),
                delta=1.0,  # Allow some difference due to random noise
                msg=f"Mean of {col} should not change significantly after adding noise"
            )
    
    def test_augment_time_series(self):
        """Test time series augmentation."""
        # Define parameters
        time_col = 'date'
        value_cols = ['home_club_goals', 'away_club_goals']
        shift_values = [1, 2]
        window_sizes = [3, 5]
        
        # Apply time series augmentation
        augmented_df = augment_time_series(
            self.match_data, time_col, value_cols, shift_values, window_sizes
        )
        
        # Check that new columns have been created
        expected_columns = []
        
        # Lagged columns
        for col in value_cols:
            for shift in shift_values:
                expected_columns.append(f"{col}_lag_{shift}")
        
        # Rolling statistics columns
        for col in value_cols:
            for window in window_sizes:
                expected_columns.append(f"{col}_roll_mean_{window}")
                expected_columns.append(f"{col}_roll_std_{window}")
        
        for col in expected_columns:
            self.assertIn(col, augmented_df.columns, f"Column {col} should be created during time series augmentation")
        
        # Check that original columns are preserved
        for col in self.match_data.columns:
            self.assertIn(col, augmented_df.columns, f"Original column {col} should be preserved")
    
    def test_run_augmentation(self):
        """Test the augmentation runner (integration test)."""
        # This is more of an integration test that would require real data
        # For unit testing, we'll just test the function signature and basic functionality
        
        # Create a mock implementation that doesn't need to load data from disk
        from unittest.mock import patch, MagicMock
        
        with patch('src.data.features.load_processed_data') as mock_load:
            # Setup mock to return our test data
            mock_data_dict = {'matches': self.match_data}
            mock_load.return_value = mock_data_dict
            
            # Test oversampling
            result = run_augmentation(
                'mock_dataset', 
                'oversample', 
                target_col='result',
                save=False  # Don't save to disk during testing
            )
            
            # Check result
            self.assertIsInstance(result, pd.DataFrame, "Result should be a DataFrame")
            self.assertGreater(len(result), 0, "Result should not be empty")
            
            # Verify the mock was called
            mock_load.assert_called_once()


if __name__ == '__main__':
    unittest.main() 