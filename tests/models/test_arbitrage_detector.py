"""
Unit tests for arbitrage detector.

This module contains tests for the ArbitrageDetector implementation.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import logging

from src.models.arbitrage_detector import (
    ArbitrageDetector,
    ArbitrageOpportunity,
    BookmakerMarginCalculator,
    CorrelationHandler
)

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestBookmakerMarginCalculator(unittest.TestCase):
    """Test the BookmakerMarginCalculator utility class."""
    
    def test_calculate_margin(self):
        """Test margin calculation from odds."""
        # Test 1X2 market with fair odds (no margin)
        fair_odds = {'home': 3.0, 'draw': 3.0, 'away': 3.0}
        self.assertAlmostEqual(
            BookmakerMarginCalculator.calculate_margin(fair_odds), 
            0.0, 
            places=2
        )
        
        # Test 1X2 market with typical margin
        typical_odds = {'home': 2.5, 'draw': 3.3, 'away': 2.9}
        self.assertGreater(
            BookmakerMarginCalculator.calculate_margin(typical_odds), 
            0.0
        )
        
        # Test binary market with 5% margin
        binary_odds = {'yes': 1.9, 'no': 1.9}  # Fair odds would be 2.0 each
        self.assertAlmostEqual(
            BookmakerMarginCalculator.calculate_margin(binary_odds), 
            5.26, 
            places=2
        )
    
    def test_calculate_fair_odds(self):
        """Test fair odds calculation."""
        # Test with 5% margin
        odds_with_margin = {'home': 2.5, 'draw': 3.3, 'away': 3.0}
        fair_odds = BookmakerMarginCalculator.calculate_fair_odds(odds_with_margin)
        
        # Fair odds should sum to 1.0 when converted to probabilities
        implied_probs = {k: 1/v for k, v in fair_odds.items()}
        self.assertAlmostEqual(sum(implied_probs.values()), 1.0, places=5)
        
        # Fair odds should be higher than original odds (less margin)
        for outcome in odds_with_margin:
            self.assertGreaterEqual(fair_odds[outcome], odds_with_margin[outcome])
    
    def test_identify_best_value(self):
        """Test identifying best value across bookmakers."""
        odds_by_bookmaker = {
            'Bookmaker1': {'home': 2.5, 'draw': 3.3, 'away': 3.0},
            'Bookmaker2': {'home': 2.6, 'draw': 3.2, 'away': 2.9},
            'Bookmaker3': {'home': 2.4, 'draw': 3.4, 'away': 3.1}
        }
        
        best_values = BookmakerMarginCalculator.identify_best_value(odds_by_bookmaker)
        
        # Check that best values are correctly identified
        self.assertEqual(best_values['home'][0], 'Bookmaker2')
        self.assertEqual(best_values['home'][1], 2.6)
        
        self.assertEqual(best_values['draw'][0], 'Bookmaker3')
        self.assertEqual(best_values['draw'][1], 3.4)
        
        self.assertEqual(best_values['away'][0], 'Bookmaker3')
        self.assertEqual(best_values['away'][1], 3.1)


class TestArbitrageDetector(unittest.TestCase):
    """Test the ArbitrageDetector implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ArbitrageDetector(
            min_profit_percent=0.5,
            max_bookmaker_margin=7.0
        )
        
        # Create mock odds data
        self.match_id = "m123"
        self.home_team = "Home FC"
        self.away_team = "Away United"
        
        # Create arbitrage opportunity in 1X2 market
        self.arbitrage_1x2_data = pd.DataFrame({
            'match_id': [self.match_id] * 6,
            'market': ['1X2'] * 6,
            'bookmaker_name': ['Bookie1', 'Bookie2', 'Bookie1', 'Bookie2', 'Bookie1', 'Bookie2'],
            'outcome_type': ['home', 'home', 'draw', 'draw', 'away', 'away'],
            'odds': [3.0, 3.1, 3.5, 3.6, 3.0, 3.1],
            'home_team': [self.home_team] * 6,
            'away_team': [self.away_team] * 6
        })
        
        # Create non-arbitrage opportunity in 1X2 market
        self.non_arbitrage_1x2_data = pd.DataFrame({
            'match_id': [self.match_id] * 6,
            'market': ['1X2'] * 6,
            'bookmaker_name': ['Bookie1', 'Bookie2', 'Bookie1', 'Bookie2', 'Bookie1', 'Bookie2'],
            'outcome_type': ['home', 'home', 'draw', 'draw', 'away', 'away'],
            'odds': [2.0, 2.1, 3.0, 3.1, 2.9, 3.0],
            'home_team': [self.home_team] * 6,
            'away_team': [self.away_team] * 6
        })
        
        # Create arbitrage opportunity in BTTS market
        self.arbitrage_btts_data = pd.DataFrame({
            'match_id': [self.match_id] * 4,
            'market': ['btts'] * 4,
            'bookmaker_name': ['Bookie1', 'Bookie2', 'Bookie1', 'Bookie2'],
            'outcome_type': ['yes', 'yes', 'no', 'no'],
            'odds': [2.0, 2.1, 2.0, 2.1],
            'home_team': [self.home_team] * 4,
            'away_team': [self.away_team] * 4
        })
    
    def test_get_implied_probabilities(self):
        """Test calculation of implied probabilities from odds."""
        odds = {'home': 3.0, 'draw': 3.0, 'away': 3.0}
        implied_probs = self.detector.get_implied_probabilities(odds)
        
        # Each implied probability should be 1/3
        for value in implied_probs.values():
            self.assertAlmostEqual(value, 1/3, places=5)
        
        # Sum should be 1.0 for fair odds
        self.assertAlmostEqual(sum(implied_probs.values()), 1.0, places=5)
    
    def test_calculate_optimal_stakes(self):
        """Test optimal stake calculation."""
        odds = {'home': 3.0, 'draw': 3.0, 'away': 3.0}
        stakes = self.detector.calculate_optimal_stakes(odds, 300.0)
        
        # Each stake should be equal for equal odds
        for value in stakes.values():
            self.assertAlmostEqual(value, 100.0, places=2)
        
        # Total should match input stake
        self.assertAlmostEqual(sum(stakes.values()), 300.0, places=2)
        
        # Test with different odds
        odds = {'home': 2.0, 'draw': 3.0, 'away': 6.0}
        stakes = self.detector.calculate_optimal_stakes(odds, 100.0)
        
        # Total should still match input stake
        self.assertAlmostEqual(sum(stakes.values()), 100.0, places=2)
        
        # Higher odds should have lower stakes
        self.assertGreater(stakes['home'], stakes['draw'])
        self.assertGreater(stakes['draw'], stakes['away'])
    
    def test_is_valid_arbitrage(self):
        """Test arbitrage validation."""
        # Valid arbitrage (sum < 1)
        valid_probs = {'home': 0.3, 'draw': 0.3, 'away': 0.3}
        self.assertTrue(
            self.detector.is_valid_arbitrage(valid_probs, min_profit_percent=0.5)
        )
        
        # Invalid arbitrage (sum > 1)
        invalid_probs = {'home': 0.4, 'draw': 0.3, 'away': 0.4}
        self.assertFalse(
            self.detector.is_valid_arbitrage(invalid_probs, min_profit_percent=0.5)
        )
        
        # Valid arbitrage but below profit threshold
        marginal_probs = {'home': 0.333, 'draw': 0.333, 'away': 0.333}
        self.assertFalse(
            self.detector.is_valid_arbitrage(marginal_probs, min_profit_percent=1.0)
        )
    
    def test_check_1x2_arbitrage(self):
        """Test 1X2 arbitrage detection."""
        # Test with arbitrage opportunity
        opportunity = self.detector.check_1x2_arbitrage(
            match_id=self.match_id,
            home_team=self.home_team,
            away_team=self.away_team,
            odds_data=self.arbitrage_1x2_data
        )
        
        self.assertIsNotNone(opportunity)
        self.assertIsInstance(opportunity, ArbitrageOpportunity)
        self.assertEqual(opportunity.market_type, "1X2")
        self.assertGreater(opportunity.arbitrage_profit_percent, 0)
        
        # Test with non-arbitrage opportunity
        opportunity = self.detector.check_1x2_arbitrage(
            match_id=self.match_id,
            home_team=self.home_team,
            away_team=self.away_team,
            odds_data=self.non_arbitrage_1x2_data
        )
        
        self.assertIsNone(opportunity)


class TestCorrelationHandler(unittest.TestCase):
    """Test the CorrelationHandler implementation."""
    
    def test_get_correlation(self):
        """Test correlation coefficient retrieval."""
        # Test known correlation
        correlation = CorrelationHandler.get_correlation(
            "1X2", "home", "btts", "no"
        )
        self.assertEqual(correlation, 0.4)
        
        # Test reverse lookup
        reverse_correlation = CorrelationHandler.get_correlation(
            "btts", "no", "1X2", "home"
        )
        self.assertEqual(reverse_correlation, 0.4)
        
        # Test unknown correlation
        unknown_correlation = CorrelationHandler.get_correlation(
            "1X2", "home", "correct_score", "1-0"
        )
        self.assertEqual(unknown_correlation, 0.0)
    
    def test_is_correlated(self):
        """Test correlation checking."""
        # Test correlated markets
        self.assertTrue(
            CorrelationHandler.is_correlated("btts", "yes", "over_under_2.5", "over")
        )
        
        # Test uncorrelated markets
        self.assertFalse(
            CorrelationHandler.is_correlated("1X2", "home", "correct_score", "0-0")
        )
        
        # Test with custom threshold
        self.assertTrue(
            CorrelationHandler.is_correlated(
                "1X2", "home", "btts", "no", threshold=0.3
            )
        )
        self.assertFalse(
            CorrelationHandler.is_correlated(
                "1X2", "home", "btts", "no", threshold=0.5
            )
        )
    
    def test_calculate_joint_probability(self):
        """Test joint probability calculation."""
        # Independent events
        self.assertAlmostEqual(
            CorrelationHandler.calculate_joint_probability(0.5, 0.5, 0),
            0.25,
            places=5
        )
        
        # Perfect positive correlation
        self.assertAlmostEqual(
            CorrelationHandler.calculate_joint_probability(0.7, 0.3, 1.0),
            0.3,
            places=5
        )
        
        # Perfect negative correlation
        self.assertAlmostEqual(
            CorrelationHandler.calculate_joint_probability(0.7, 0.4, -1.0),
            0.1,
            places=5
        )
        
        # Partial positive correlation
        joint_prob = CorrelationHandler.calculate_joint_probability(0.6, 0.4, 0.5)
        self.assertGreater(joint_prob, 0.6 * 0.4)  # Should be higher than independent case
        
        # Partial negative correlation
        joint_prob = CorrelationHandler.calculate_joint_probability(0.6, 0.4, -0.5)
        self.assertLess(joint_prob, 0.6 * 0.4)  # Should be lower than independent case


if __name__ == '__main__':
    unittest.main() 