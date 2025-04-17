"""
Tests for the Elo rating system implementation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the module to test
from src.data.elo_ratings import (
    expected_result,
    update_rating,
    calculate_match_outcome,
    calculate_dynamic_k_factor,
    apply_time_decay,
    calculate_elo_ratings,
    get_latest_team_ratings,
    generate_elo_features
)


class TestEloRatings(unittest.TestCase):
    """Test cases for Elo rating functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create a sample dataset of matches
        self.matches_df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10, freq='W'),
            'home_club_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'away_club_id': [2, 3, 4, 5, 1, 3, 4, 5, 1, 2],
            'home_club_goals': [2, 1, 0, 2, 1, 3, 2, 1, 0, 2],
            'away_club_goals': [1, 1, 2, 0, 0, 0, 2, 1, 1, 1]
        })
    
    def test_expected_result(self):
        """Test expected result calculation."""
        # Equal teams with home advantage
        self.assertAlmostEqual(expected_result(1500, 1500, 100), 0.6400, places=4)
        
        # Stronger home team
        self.assertAlmostEqual(expected_result(1600, 1500, 100), 0.7597, places=4)
        
        # Stronger away team
        self.assertAlmostEqual(expected_result(1500, 1700, 100), 0.2689, places=4)
        
        # No home advantage
        self.assertAlmostEqual(expected_result(1500, 1500, 0), 0.5000, places=4)
    
    def test_update_rating(self):
        """Test rating update calculation."""
        # Win when expected to win
        self.assertEqual(update_rating(1500, 0.75, 1.0, 32), 1508.0)
        
        # Loss when expected to lose
        self.assertEqual(update_rating(1500, 0.25, 0.0, 32), 1492.0)
        
        # Draw when evenly matched
        self.assertEqual(update_rating(1500, 0.5, 0.5, 32), 1500.0)
        
        # Unexpected win (big rating change)
        self.assertEqual(update_rating(1500, 0.1, 1.0, 32), 1528.8)
        
        # Unexpected loss (big rating change)
        self.assertEqual(update_rating(1500, 0.9, 0.0, 32), 1471.2)
    
    def test_calculate_match_outcome(self):
        """Test match outcome calculation."""
        # Home win
        self.assertEqual(calculate_match_outcome(2, 1), (1.0, 0.0))
        
        # Away win
        self.assertEqual(calculate_match_outcome(0, 3), (0.0, 1.0))
        
        # Draw
        self.assertEqual(calculate_match_outcome(1, 1), (0.5, 0.5))
    
    def test_calculate_dynamic_k_factor(self):
        """Test dynamic K-factor calculation."""
        # Base case
        self.assertEqual(calculate_dynamic_k_factor(32, 0, 1.0), 32.0)
        
        # Goal difference
        self.assertEqual(calculate_dynamic_k_factor(32, 2, 1.0), 38.4)
        
        # Goal difference capped at 3
        self.assertEqual(calculate_dynamic_k_factor(32, 5, 1.0), 41.6)
        
        # Match importance
        self.assertEqual(calculate_dynamic_k_factor(32, 0, 1.5), 48.0)
        
        # Combined effects
        self.assertEqual(calculate_dynamic_k_factor(32, 2, 1.5), 57.6)
        
        # Maximum K limit
        self.assertEqual(calculate_dynamic_k_factor(32, 5, 2.0, max_k=60), 60.0)
    
    def test_apply_time_decay(self):
        """Test time decay application."""
        # Sample ratings
        ratings = {1: 1600, 2: 1400, 3: 1500}
        
        # No decay
        no_decay = apply_time_decay(ratings, decay_factor=0)
        self.assertEqual(no_decay[1], 1600)
        self.assertEqual(no_decay[2], 1400)
        self.assertEqual(no_decay[3], 1500)
        
        # Full decay (should all be 1500)
        full_decay = apply_time_decay(ratings, decay_factor=1)
        self.assertEqual(full_decay[1], 1500)
        self.assertEqual(full_decay[2], 1500)
        self.assertEqual(full_decay[3], 1500)
        
        # Partial decay
        partial_decay = apply_time_decay(ratings, decay_factor=0.5)
        self.assertEqual(partial_decay[1], 1550)  # 1600 + 0.5 * (1500 - 1600)
        self.assertEqual(partial_decay[2], 1450)  # 1400 + 0.5 * (1500 - 1400)
        self.assertEqual(partial_decay[3], 1500)  # 1500 + 0.5 * (1500 - 1500)
    
    def test_calculate_elo_ratings(self):
        """Test Elo rating calculation for match data."""
        # Calculate ratings
        elo_df = calculate_elo_ratings(
            self.matches_df,
            initial_rating=1500,
            base_k=32,
            home_advantage=100,
            include_dynamic_k=True,
            include_time_decay=True
        )
        
        # Check that the output dataframe has the correct columns
        required_columns = [
            'home_elo_pre', 'away_elo_pre',
            'home_elo_post', 'away_elo_post',
            'home_elo_expected', 'away_elo_expected',
            'elo_k_factor'
        ]
        for col in required_columns:
            self.assertIn(col, elo_df.columns)
        
        # Check that values are calculated
        self.assertFalse(elo_df['home_elo_pre'].isna().any())
        self.assertFalse(elo_df['away_elo_pre'].isna().any())
        self.assertFalse(elo_df['home_elo_post'].isna().any())
        self.assertFalse(elo_df['away_elo_post'].isna().any())
        
        # Check initial ratings (first match for each team)
        first_match = elo_df.iloc[0]
        self.assertEqual(first_match['home_elo_pre'], 1500)
        self.assertEqual(first_match['away_elo_pre'], 1500)
        
        # Ratings should update after matches
        self.assertNotEqual(first_match['home_elo_post'], 1500)
        self.assertNotEqual(first_match['away_elo_post'], 1500)
        
        # Check that later matches use updated ratings
        second_match = elo_df.iloc[1]
        self.assertEqual(second_match['home_elo_pre'], 1500)  # Team 2 first appearance as home
        self.assertNotEqual(second_match['away_elo_pre'], 1500)  # Team 3 should have rating from previous match
    
    def test_get_latest_team_ratings(self):
        """Test retrieving latest team ratings."""
        # Get ratings after all matches
        all_ratings = get_latest_team_ratings(self.matches_df)
        
        # Should have ratings for all 5 teams
        self.assertEqual(len(all_ratings), 5)
        for team_id in range(1, 6):
            self.assertIn(team_id, all_ratings)
        
        # Ratings with cutoff date
        cutoff_date = self.matches_df['date'].iloc[4]  # After 5 matches
        partial_ratings = get_latest_team_ratings(self.matches_df, cutoff_date=cutoff_date)
        
        # Should still have all 5 teams
        self.assertEqual(len(partial_ratings), 5)
        
        # Ratings should be different from final ratings
        for team_id in range(1, 6):
            self.assertNotEqual(all_ratings[team_id], partial_ratings[team_id])
    
    def test_generate_elo_features(self):
        """Test generation of Elo-based features."""
        # Generate features
        features_df = generate_elo_features(self.matches_df)
        
        # Check that all derived features are present
        derived_features = [
            'elo_diff', 'elo_sum', 'elo_avg',
            'home_win_probability', 'away_win_probability', 'draw_probability',
            'elo_surprise'
        ]
        for feature in derived_features:
            self.assertIn(feature, features_df.columns)
        
        # Check that win probabilities sum to approximately 1
        prob_sum = features_df['home_win_probability'] + features_df['away_win_probability'] + features_df['draw_probability']
        for p in prob_sum:
            self.assertAlmostEqual(p, 1.0, places=5)
        
        # Check that elo_diff is correctly calculated
        for i, row in features_df.iterrows():
            self.assertEqual(row['elo_diff'], row['home_elo_pre'] - row['away_elo_pre'])
            self.assertEqual(row['elo_sum'], row['home_elo_pre'] + row['away_elo_pre'])
            self.assertEqual(row['elo_avg'], row['elo_sum'] / 2)


if __name__ == '__main__':
    unittest.main() 