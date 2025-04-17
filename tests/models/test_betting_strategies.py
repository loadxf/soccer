"""
Unit tests for betting strategies.

This module contains tests for the betting strategies implemented in src/models/betting_strategies.py.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import logging

from src.models.betting_strategies import (
    BetType,
    BettingStrategyResult,
    BettingStrategy,
    ValueBettingStrategy,
    DrawNoBetStrategy,
    AsianHandicapStrategy,
    ModelEnsembleStrategy,
    MarketMovementStrategy,
    ExpectedGoalsStrategy,
    InPlayBettingStrategy,
    implied_probability
)

# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestBettingStrategyResult(unittest.TestCase):
    """Test the BettingStrategyResult data class."""
    
    def test_initialization(self):
        """Test that BettingStrategyResult initializes correctly."""
        result = BettingStrategyResult(
            match_id=123,
            home_team="Home Team",
            away_team="Away Team",
            bet_type=BetType.HOME,
            odds=2.0
        )
        
        self.assertEqual(result.match_id, 123)
        self.assertEqual(result.home_team, "Home Team")
        self.assertEqual(result.away_team, "Away Team")
        self.assertEqual(result.bet_type, BetType.HOME)
        self.assertEqual(result.odds, 2.0)
        self.assertIsNotNone(result.timestamp)
        self.assertIsNotNone(result.extra)
        
    def test_to_dict(self):
        """Test the to_dict method."""
        result = BettingStrategyResult(
            match_id=123,
            home_team="Home Team",
            away_team="Away Team",
            bet_type=BetType.HOME,
            odds=2.0,
            predicted_probability=0.6
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["match_id"], 123)
        self.assertEqual(result_dict["home_team"], "Home Team")
        self.assertEqual(result_dict["away_team"], "Away Team")
        self.assertEqual(result_dict["bet_type"], "Home")
        self.assertEqual(result_dict["odds"], 2.0)
        self.assertEqual(result_dict["predicted_probability"], 0.6)
        
    def test_from_dict(self):
        """Test the from_dict class method."""
        data = {
            "match_id": 123,
            "home_team": "Home Team",
            "away_team": "Away Team",
            "bet_type": "Home",
            "odds": 2.0,
            "predicted_probability": 0.6,
            "timestamp": datetime.now().isoformat(),
            "date": (datetime.now() - timedelta(days=1)).isoformat()
        }
        
        result = BettingStrategyResult.from_dict(data)
        self.assertEqual(result.match_id, 123)
        self.assertEqual(result.home_team, "Home Team")
        self.assertEqual(result.away_team, "Away Team")
        self.assertEqual(result.bet_type, BetType.HOME)
        self.assertEqual(result.odds, 2.0)
        self.assertEqual(result.predicted_probability, 0.6)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertIsInstance(result.date, datetime)

class TestValueBettingStrategy(unittest.TestCase):
    """Test the ValueBettingStrategy class."""
    
    def setUp(self):
        """Set up common test variables."""
        self.strategy = ValueBettingStrategy(
            min_edge=0.05,
            min_odds=1.5,
            max_odds=5.0
        )
        
    def test_initialization(self):
        """Test that ValueBettingStrategy initializes correctly."""
        self.assertEqual(self.strategy.name, "Value Betting")
        self.assertEqual(self.strategy.min_edge, 0.05)
        self.assertEqual(self.strategy.min_odds, 1.5)
        self.assertEqual(self.strategy.max_odds, 5.0)
        self.assertEqual(self.strategy.market_filter, ['1X2', 'over_under', 'btts'])
        
    def test_calculate_value_rating(self):
        """Test the calculate_value_rating method."""
        value_rating = self.strategy.calculate_value_rating(
            edge=0.1,
            odds=2.0,
            confidence=0.8
        )
        # Value rating should increase with edge, odds, and confidence
        self.assertGreater(value_rating, 0)
        
        # Test that higher edge gives higher rating
        higher_edge_rating = self.strategy.calculate_value_rating(
            edge=0.2,
            odds=2.0,
            confidence=0.8
        )
        self.assertGreater(higher_edge_rating, value_rating)
        
    def test_is_market_allowed(self):
        """Test the is_market_allowed method."""
        self.assertTrue(self.strategy.is_market_allowed('home'))
        self.assertTrue(self.strategy.is_market_allowed('draw'))
        self.assertTrue(self.strategy.is_market_allowed('away'))
        self.assertTrue(self.strategy.is_market_allowed('over_2.5'))
        self.assertTrue(self.strategy.is_market_allowed('under_2.5'))
        self.assertTrue(self.strategy.is_market_allowed('btts_yes'))
        
        # Test with custom market filter
        strategy_custom = ValueBettingStrategy(
            market_filter=['1X2']
        )
        self.assertTrue(strategy_custom.is_market_allowed('home'))
        self.assertFalse(strategy_custom.is_market_allowed('over_2.5'))
        
    def test_is_league_allowed(self):
        """Test the is_league_allowed method."""
        # No league filter means all leagues allowed
        self.assertTrue(self.strategy.is_league_allowed('Premier League'))
        
        # Test with league filter
        strategy_league = ValueBettingStrategy(
            league_filter=['Premier League', 'La Liga']
        )
        self.assertTrue(strategy_league.is_league_allowed('Premier League'))
        self.assertFalse(strategy_league.is_league_allowed('Bundesliga'))
        
    def test_evaluate_bet(self):
        """Test the evaluate_bet method."""
        result = self.strategy.evaluate_bet(
            match_id=123,
            market='home',
            predicted_probability=0.6,
            current_odds=2.0,
            home_team="Home Team",
            away_team="Away Team",
            confidence=0.8
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.match_id, 123)
        self.assertEqual(result.bet_type, BetType.HOME)
        self.assertEqual(result.odds, 2.0)
        self.assertEqual(result.predicted_probability, 0.6)
        self.assertAlmostEqual(result.edge, 0.6 - 0.5, places=6)  # Edge = pred_prob - implied_prob
        
        # Test bet rejection due to insufficient edge
        result_no_edge = self.strategy.evaluate_bet(
            match_id=123,
            market='home',
            predicted_probability=0.51,  # Just above implied probability
            current_odds=2.0,
            home_team="Home Team",
            away_team="Away Team",
            confidence=0.8
        )
        
        self.assertIsNone(result_no_edge)  # Should be rejected

class TestDrawNoBetStrategy(unittest.TestCase):
    """Test the DrawNoBetStrategy class."""
    
    def setUp(self):
        """Set up common test variables."""
        self.strategy = DrawNoBetStrategy(
            min_edge=0.04,
            min_odds=1.3,
            max_odds=3.0
        )
        
    def test_initialization(self):
        """Test that DrawNoBetStrategy initializes correctly."""
        self.assertEqual(self.strategy.name, "Draw No Bet")
        self.assertEqual(self.strategy.min_edge, 0.04)
        self.assertEqual(self.strategy.min_odds, 1.3)
        self.assertEqual(self.strategy.max_odds, 3.0)
        
    def test_calculate_dnb_odds(self):
        """Test the calculate_dnb_odds method."""
        # Test standard case
        dnb_odds = self.strategy.calculate_dnb_odds(win_odds=2.0, draw_odds=3.0)
        self.assertAlmostEqual(dnb_odds, 1.5, places=6)
        
        # Test edge case with very low odds
        dnb_odds_low = self.strategy.calculate_dnb_odds(win_odds=1.1, draw_odds=3.0)
        self.assertGreater(dnb_odds_low, 1.0)
        
    def test_evaluate_bet(self):
        """Test the evaluate_bet method."""
        result = self.strategy.evaluate_bet(
            match_id=123,
            home_team="Home Team",
            away_team="Away Team",
            odds={'home': 2.0, 'draw': 3.0, 'away': 4.0},
            predictions={'home': 0.55, 'draw': 0.2, 'away': 0.25},
            confidence={'home': 0.8, 'away': 0.7}
        )
        
        # Should be a valid DNB bet for home team
        self.assertIsNotNone(result)
        self.assertEqual(result.match_id, 123)
        self.assertEqual(result.bet_type, BetType.HOME)
        
        # Test with unfavorable probabilities
        result_unfavorable = self.strategy.evaluate_bet(
            match_id=123,
            home_team="Home Team",
            away_team="Away Team",
            odds={'home': 2.0, 'draw': 3.0, 'away': 4.0},
            predictions={'home': 0.35, 'draw': 0.4, 'away': 0.25},  # Low home win prob, high draw prob
            confidence={'home': 0.8, 'away': 0.7}
        )
        
        # Should reject the bet
        self.assertIsNone(result_unfavorable)

class TestAsianHandicapStrategy(unittest.TestCase):
    """Test the AsianHandicapStrategy class."""
    
    def setUp(self):
        """Set up common test variables."""
        self.strategy = AsianHandicapStrategy()
        
    def test_initialization(self):
        """Test that AsianHandicapStrategy initializes correctly."""
        self.assertEqual(self.strategy.name, "Asian Handicap")
        self.assertIsNotNone(self.strategy.handicap_ranges)
        
    def test_get_handicap_range(self):
        """Test the get_handicap_range method."""
        # Default range
        default_range = self.strategy.get_handicap_range(league=None)
        self.assertIsInstance(default_range, list)
        self.assertIn(-1.0, default_range)
        self.assertIn(1.0, default_range)
        
        # Custom range
        strategy_custom = AsianHandicapStrategy(
            handicap_ranges={
                'Premier League': [-1.0, -0.5, 0, 0.5, 1.0],
                'default': [-2.0, -1.0, 0, 1.0, 2.0]
            }
        )
        
        pl_range = strategy_custom.get_handicap_range(league='Premier League')
        self.assertEqual(pl_range, [-1.0, -0.5, 0, 0.5, 1.0])
        
        other_range = strategy_custom.get_handicap_range(league='Bundesliga')
        self.assertEqual(other_range, [-2.0, -1.0, 0, 1.0, 2.0])  # Should use default
        
    def test_calculate_handicap_win_probability(self):
        """Test the calculate_handicap_win_probability method."""
        # Test positive handicap
        home_prob, away_prob = self.strategy.calculate_handicap_win_probability(
            expected_home_goals=1.5,
            expected_away_goals=1.0,
            handicap=0.5  # Advantage to away team
        )
        
        # Home team still favored to win but probability reduced
        self.assertGreater(home_prob, 0.4)
        self.assertLess(home_prob, 0.6)
        self.assertAlmostEqual(home_prob + away_prob, 1.0, places=6)
        
        # Test negative handicap
        home_prob, away_prob = self.strategy.calculate_handicap_win_probability(
            expected_home_goals=1.5,
            expected_away_goals=1.0,
            handicap=-0.5  # Advantage to home team
        )
        
        # Home team more favored to win
        self.assertGreater(home_prob, 0.5)
        self.assertLess(away_prob, 0.5)
        self.assertAlmostEqual(home_prob + away_prob, 1.0, places=6)

class TestModelEnsembleStrategy(unittest.TestCase):
    """Test the ModelEnsembleStrategy class."""
    
    def setUp(self):
        """Set up common test variables."""
        self.strategy = ModelEnsembleStrategy(
            models=[
                {'name': 'model1', 'weight': 0.6},
                {'name': 'model2', 'weight': 0.4}
            ]
        )
        
    def test_initialization(self):
        """Test that ModelEnsembleStrategy initializes correctly."""
        self.assertEqual(self.strategy.name, "Model Ensemble")
        self.assertEqual(len(self.strategy.models), 2)
        self.assertEqual(self.strategy.models[0]['name'], 'model1')
        
        # Test weight normalization
        self.assertAlmostEqual(self.strategy.models[0]['weight'] + self.strategy.models[1]['weight'], 1.0)
        
    def test_get_model_weight(self):
        """Test the get_model_weight method."""
        weight1 = self.strategy.get_model_weight('model1')
        weight2 = self.strategy.get_model_weight('model2')
        
        self.assertAlmostEqual(weight1, 0.6)
        self.assertAlmostEqual(weight2, 0.4)
        
        # Test with league-specific weights
        strategy_league = ModelEnsembleStrategy(
            models=[
                {
                    'name': 'model1', 
                    'weight': 0.6,
                    'league_weights': {'Premier League': 0.8, 'La Liga': 0.5}
                },
                {'name': 'model2', 'weight': 0.4}
            ]
        )
        
        weight_pl = strategy_league.get_model_weight('model1', league='Premier League')
        weight_ll = strategy_league.get_model_weight('model1', league='La Liga')
        
        # League weights are applied as multipliers
        self.assertAlmostEqual(weight_pl, 0.6 * 0.8)
        self.assertAlmostEqual(weight_ll, 0.6 * 0.5)
        
    def test_combine_predictions(self):
        """Test the combine_predictions method."""
        model_predictions = {
            'model1': {'home': 0.5, 'draw': 0.25, 'away': 0.25},
            'model2': {'home': 0.4, 'draw': 0.3, 'away': 0.3}
        }
        
        combined, confidence = self.strategy.combine_predictions(model_predictions)
        
        # Weighted average: (0.5*0.6 + 0.4*0.4) = 0.46
        self.assertAlmostEqual(combined['home'], 0.46, places=6)
        self.assertAlmostEqual(combined['draw'] + combined['away'] + combined['home'], 1.0, places=6)
        self.assertGreater(confidence, 0.5)  # Should have reasonable confidence

class TestExpectedGoalsStrategy(unittest.TestCase):
    """Test the ExpectedGoalsStrategy class."""
    
    def setUp(self):
        """Set up common test variables."""
        self.strategy = ExpectedGoalsStrategy()
        
    def test_initialization(self):
        """Test that ExpectedGoalsStrategy initializes correctly."""
        self.assertEqual(self.strategy.name, "Expected Goals")
        self.assertGreater(self.strategy.xg_weight, 0.5)  # xG should be weighted more heavily
        self.assertLess(self.strategy.historical_weight, 0.5)
        
    def test_calculate_goal_probabilities(self):
        """Test the calculate_goal_probabilities method."""
        probabilities = self.strategy.calculate_goal_probabilities(
            home_xg=1.5,
            away_xg=1.0
        )
        
        # Check for required markets
        self.assertIn('over_2.5', probabilities)
        self.assertIn('under_2.5', probabilities)
        self.assertIn('btts_yes', probabilities)
        self.assertIn('btts_no', probabilities)
        self.assertIn('home', probabilities)
        self.assertIn('draw', probabilities)
        self.assertIn('away', probabilities)
        
        # Check probability consistency
        self.assertAlmostEqual(probabilities['home'] + probabilities['draw'] + probabilities['away'], 1.0, places=6)
        self.assertAlmostEqual(probabilities['over_2.5'] + probabilities['under_2.5'], 1.0, places=6)
        self.assertAlmostEqual(probabilities['btts_yes'] + probabilities['btts_no'], 1.0, places=6)
        
        # Home team has higher xG, should be favored
        self.assertGreater(probabilities['home'], probabilities['away'])
        
    def test_calculate_confidence(self):
        """Test the calculate_confidence method."""
        home_xg_data = {
            'mean': 1.5,
            'std': 0.3,
            'matches': 20,
            'scoring_rate': 0.7
        }
        
        away_xg_data = {
            'mean': 1.0,
            'std': 0.4,
            'matches': 15,
            'scoring_rate': 0.6
        }
        
        # Test for different markets
        conf_home = self.strategy.calculate_confidence(home_xg_data, away_xg_data, 'home')
        conf_over = self.strategy.calculate_confidence(home_xg_data, away_xg_data, 'over_2.5')
        conf_btts = self.strategy.calculate_confidence(home_xg_data, away_xg_data, 'btts_yes')
        
        # All should be above threshold
        self.assertGreater(conf_home, self.strategy.confidence_threshold)
        self.assertGreater(conf_over, self.strategy.confidence_threshold)
        self.assertGreater(conf_btts, self.strategy.confidence_threshold)
        
        # Low data should reduce confidence
        home_xg_low_data = {
            'mean': 1.5,
            'std': 0.3,
            'matches': 2,  # Very few matches
            'scoring_rate': 0.7
        }
        
        conf_low_data = self.strategy.calculate_confidence(home_xg_low_data, away_xg_data, 'home')
        self.assertLess(conf_low_data, conf_home)

class TestInPlayBettingStrategy(unittest.TestCase):
    """Test the InPlayBettingStrategy class."""
    
    def setUp(self):
        """Set up common test variables."""
        self.strategy = InPlayBettingStrategy()
        
    def test_initialization(self):
        """Test that InPlayBettingStrategy initializes correctly."""
        self.assertEqual(self.strategy.name, "In-Play Betting")
        self.assertIsNotNone(self.strategy.time_windows)
        self.assertIn('next_goal', self.strategy.market_filter)
        
    def test_is_time_window_allowed(self):
        """Test the is_time_window_allowed method."""
        # Test typical time windows
        self.assertTrue(self.strategy.is_time_window_allowed(15))  # Early game
        self.assertTrue(self.strategy.is_time_window_allowed(45))  # Mid game
        self.assertTrue(self.strategy.is_time_window_allowed(70))  # Late game
        
        # Test boundaries
        self.assertFalse(self.strategy.is_time_window_allowed(5))   # Too early
        self.assertFalse(self.strategy.is_time_window_allowed(85))  # Too late
        
        # Test custom time windows
        strategy_custom = InPlayBettingStrategy(
            time_windows=[(15, 30), (70, 85)]
        )
        
        self.assertTrue(strategy_custom.is_time_window_allowed(20))
        self.assertFalse(strategy_custom.is_time_window_allowed(45))
        self.assertTrue(strategy_custom.is_time_window_allowed(75))
        
    def test_calculate_momentum_score(self):
        """Test the calculate_momentum_score method."""
        # Test balanced stats
        stats_balanced = {
            'home_shots': 5,
            'away_shots': 5,
            'home_shots_on_target': 2,
            'away_shots_on_target': 2,
            'home_possession': 50,
            'away_possession': 50,
            'home_corners': 3,
            'away_corners': 3,
            'home_recent_shots': 1,
            'away_recent_shots': 1,
            'home_recent_corners': 1,
            'away_recent_corners': 1
        }
        
        home_momentum, away_momentum = self.strategy.calculate_momentum_score(
            stats=stats_balanced,
            current_score=(0, 0),
            time_elapsed=45
        )
        
        # Should be roughly equal
        self.assertAlmostEqual(home_momentum, 0.5, delta=0.1)
        self.assertAlmostEqual(away_momentum, 0.5, delta=0.1)
        
        # Test home team dominance
        stats_home_dominant = {
            'home_shots': 10,
            'away_shots': 2,
            'home_shots_on_target': 5,
            'away_shots_on_target': 1,
            'home_possession': 65,
            'away_possession': 35,
            'home_corners': 6,
            'away_corners': 1,
            'home_recent_shots': 3,
            'away_recent_shots': 0,
            'home_recent_corners': 2,
            'away_recent_corners': 0
        }
        
        home_momentum, away_momentum = self.strategy.calculate_momentum_score(
            stats=stats_home_dominant,
            current_score=(1, 0),
            time_elapsed=45
        )
        
        # Home team should have much more momentum
        self.assertGreater(home_momentum, 0.7)
        self.assertLess(away_momentum, 0.3)
        
        # Test trailing team effect
        home_momentum_trailing, away_momentum_trailing = self.strategy.calculate_momentum_score(
            stats=stats_home_dominant,
            current_score=(0, 2),  # Home team trailing despite dominance
            time_elapsed=45
        )
        
        # Home team should get slight momentum boost from trailing
        self.assertGreater(home_momentum_trailing, home_momentum)
        
    def test_calculate_time_adjusted_probabilities(self):
        """Test the calculate_time_adjusted_probabilities method."""
        base_probs = {
            'home': 0.45,
            'draw': 0.25,
            'away': 0.3,
            'over_2.5': 0.6,
            'under_2.5': 0.4,
            'home_next': 0.55,
            'away_next': 0.45,
            'no_more_goals': 0.2
        }
        
        # Test early game adjustment
        adjusted_early = self.strategy.calculate_time_adjusted_probabilities(
            base_probabilities=base_probs,
            current_score=(0, 0),
            time_elapsed=15,
            home_momentum=0.5,
            away_momentum=0.5
        )
        
        # Draw and no more goals should have lower probability early
        self.assertLess(adjusted_early['draw'], base_probs['draw'])
        self.assertLess(adjusted_early['no_more_goals'], base_probs['no_more_goals'])
        
        # Test late game adjustment
        adjusted_late = self.strategy.calculate_time_adjusted_probabilities(
            base_probabilities=base_probs,
            current_score=(0, 0),
            time_elapsed=75,
            home_momentum=0.5,
            away_momentum=0.5
        )
        
        # Draw and no more goals should have higher probability late
        self.assertGreater(adjusted_late['draw'], base_probs['draw'])
        self.assertGreater(adjusted_late['no_more_goals'], base_probs['no_more_goals'])
        
        # Test with team leading
        adjusted_leading = self.strategy.calculate_time_adjusted_probabilities(
            base_probabilities=base_probs,
            current_score=(1, 0),
            time_elapsed=75,
            home_momentum=0.6,
            away_momentum=0.4
        )
        
        # Home win probability should be higher with lead late in game
        self.assertGreater(adjusted_leading['home'], base_probs['home'])
        
    def test_evaluate_bet(self):
        """Test the evaluate_bet method."""
        match_stats = {
            'time_elapsed': 60,
            'home_score': 1,
            'away_score': 1,
            'home_shots': 8,
            'away_shots': 4,
            'home_shots_on_target': 3,
            'away_shots_on_target': 2,
            'home_possession': 55,
            'away_possession': 45,
            'home_corners': 5,
            'away_corners': 2,
            'home_recent_shots': 3,
            'away_recent_shots': 1,
            'home_recent_corners': 2,
            'away_recent_corners': 0
        }
        
        base_probabilities = {
            'home': 0.45,
            'draw': 0.25,
            'away': 0.3,
            'over_2.5': 0.6,
            'under_2.5': 0.4,
            'home_next': 0.55,
            'away_next': 0.45,
            'no_more_goals': 0.2
        }
        
        # Test a good value bet
        result = self.strategy.evaluate_bet(
            match_id=123,
            market='home_next',
            current_odds=2.2,  # Good value for ~0.55 probability
            base_probabilities=base_probabilities,
            match_stats=match_stats,
            home_team="Home Team",
            away_team="Away Team"
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.match_id, 123)
        self.assertEqual(result.bet_type, BetType.HOME)
        
        # Test market rejection
        result_rejected = self.strategy.evaluate_bet(
            match_id=123,
            market='home_next',
            current_odds=1.5,  # Poor value
            base_probabilities=base_probabilities,
            match_stats=match_stats,
            home_team="Home Team",
            away_team="Away Team"
        )
        
        self.assertIsNone(result_rejected)
        
        # Test time window rejection
        match_stats_early = dict(match_stats)
        match_stats_early['time_elapsed'] = 5  # Too early
        
        result_early = self.strategy.evaluate_bet(
            match_id=123,
            market='home_next',
            current_odds=2.2,
            base_probabilities=base_probabilities,
            match_stats=match_stats_early,
            home_team="Home Team",
            away_team="Away Team"
        )
        
        self.assertIsNone(result_early)
        
        # Test goal difference rejection
        match_stats_big_lead = dict(match_stats)
        match_stats_big_lead['home_score'] = 3
        match_stats_big_lead['away_score'] = 0
        
        result_big_lead = self.strategy.evaluate_bet(
            match_id=123,
            market='home_next',
            current_odds=2.2,
            base_probabilities=base_probabilities,
            match_stats=match_stats_big_lead,
            home_team="Home Team",
            away_team="Away Team"
        )
        
        self.assertIsNone(result_big_lead)

if __name__ == '__main__':
    unittest.main() 