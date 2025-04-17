"""
Integration tests for betting strategies.

This module contains tests for how betting strategies interact with each other
and with other components like bankroll management.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import logging
import tempfile
import os

from src.models.betting_strategies import (
    BetType,
    BettingStrategyResult,
    ValueBettingStrategy,
    DrawNoBetStrategy,
    AsianHandicapStrategy,
    ModelEnsembleStrategy,
    MarketMovementStrategy,
    ExpectedGoalsStrategy,
    InPlayBettingStrategy
)

from src.models.bankroll_management import (
    KellyCriterionManager,
    StopLossManager
)

# Import the BacktestEngine if it exists
try:
    from src.models.backtesting import BacktestEngine
    BACKTEST_ENGINE_AVAILABLE = True
except ImportError:
    BACKTEST_ENGINE_AVAILABLE = False
    
# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestStrategyIntegration(unittest.TestCase):
    """Test the integration between different betting strategies."""
    
    def setUp(self):
        """Set up common test variables."""
        # Create a set of test predictions
        self.test_predictions = pd.DataFrame({
            'match_id': [1, 1, 1, 2, 2, 2],
            'date': [datetime.now(), datetime.now(), datetime.now(), 
                    datetime.now(), datetime.now(), datetime.now()],
            'market': ['home', 'draw', 'away', 'home', 'draw', 'away'],
            'probability': [0.55, 0.25, 0.2, 0.3, 0.3, 0.4],
            'home_team': ['Team A', 'Team A', 'Team A', 'Team C', 'Team C', 'Team C'],
            'away_team': ['Team B', 'Team B', 'Team B', 'Team D', 'Team D', 'Team D'],
            'league': ['League 1', 'League 1', 'League 1', 'League 2', 'League 2', 'League 2'],
            'confidence': [0.8, 0.7, 0.6, 0.5, 0.8, 0.9]
        })
        
        # Create test odds data
        self.test_odds = pd.DataFrame({
            'match_id': [1, 1, 1, 2, 2, 2],
            'date': [datetime.now(), datetime.now(), datetime.now(), 
                    datetime.now(), datetime.now(), datetime.now()],
            'market': ['home', 'draw', 'away', 'home', 'draw', 'away'],
            'odds': [1.8, 3.5, 4.5, 3.0, 3.2, 2.4],
            'home_team': ['Team A', 'Team A', 'Team A', 'Team C', 'Team C', 'Team C'],
            'away_team': ['Team B', 'Team B', 'Team B', 'Team D', 'Team D', 'Team D'],
            'league': ['League 1', 'League 1', 'League 1', 'League 2', 'League 2', 'League 2']
        })
        
        # Create test results data
        self.test_results = pd.DataFrame({
            'match_id': [1, 2],
            'date': [datetime.now(), datetime.now()],
            'home_score': [2, 0],
            'away_score': [0, 2],
            'home_team': ['Team A', 'Team C'],
            'away_team': ['Team B', 'Team D'],
            'league': ['League 1', 'League 2']
        })
        
        # Initialize strategies
        self.value_strategy = ValueBettingStrategy(
            min_edge=0.05,
            min_odds=1.5,
            max_odds=5.0
        )
        
        self.dnb_strategy = DrawNoBetStrategy(
            min_edge=0.04,
            min_odds=1.3,
            max_odds=3.0
        )
        
        # Initialize bankroll manager
        self.kelly_manager = KellyCriterionManager(
            initial_bankroll=1000.0,
            fraction=0.5  # Conservative Kelly
        )
        
        self.stop_loss_manager = StopLossManager(
            initial_bankroll=1000.0,
            daily_stop_loss_percentage=0.05
        )
    
    def test_value_strategy_with_kelly_manager(self):
        """Test ValueBettingStrategy with KellyCriterionManager."""
        # Get bets from strategy
        strategy_results = self.value_strategy.from_model_predictions(
            predictions_df=self.test_predictions.rename(columns={'probability': 'predicted_probability'}),
            odds_df=self.test_odds,
            bankroll=self.kelly_manager.current_bankroll
        )
        
        # Process each bet with Kelly manager
        placed_bets = []
        for bet in strategy_results:
            bet_data = bet.to_dict()
            stake = self.kelly_manager.calculate_recommended_stake(bet_data)
            
            # Place bet
            bet_id = f"bet_{len(placed_bets) + 1}"
            success = self.kelly_manager.place_bet(
                amount=stake,
                match_id=bet.match_id,
                strategy_name="Value Betting",
                bet_description=bet.bet_description,
                bet_id=bet_id,
                extra=bet_data
            )
            
            if success:
                placed_bets.append({
                    'bet_id': bet_id,
                    'strategy_result': bet,
                    'stake': stake
                })
        
        # Verify bets were placed
        self.assertGreater(len(placed_bets), 0)
        
        # Verify stakes were calculated appropriately
        for bet in placed_bets:
            self.assertGreater(bet['stake'], 0)
            # Stake should be less than max_bet_percentage * bankroll
            self.assertLessEqual(bet['stake'], self.kelly_manager.current_bankroll * self.kelly_manager.max_bet_percentage)
            
        # Settle bets based on results
        for bet in placed_bets:
            match_id = bet['strategy_result'].match_id
            bet_type = bet['strategy_result'].bet_type
            
            # Find match result
            match_result = self.test_results[self.test_results['match_id'] == match_id].iloc[0]
            
            # Determine outcome
            if bet_type == BetType.HOME:
                outcome = 'win' if match_result['home_score'] > match_result['away_score'] else 'loss'
            elif bet_type == BetType.AWAY:
                outcome = 'win' if match_result['away_score'] > match_result['home_score'] else 'loss'
            elif bet_type == BetType.DRAW:
                outcome = 'win' if match_result['home_score'] == match_result['away_score'] else 'loss'
            else:
                outcome = 'loss'  # Default for more complex bet types not handled here
                
            # Calculate win amount for wins
            win_amount = None
            if outcome == 'win':
                win_amount = bet['stake'] * bet['strategy_result'].odds
                
            # Settle the bet
            self.kelly_manager.settle_bet(
                bet_id=bet['bet_id'],
                outcome=outcome,
                win_amount=win_amount
            )
            
        # Verify bankroll was updated
        self.assertNotEqual(self.kelly_manager.current_bankroll, 1000.0)
        
        # Get performance summary
        performance = self.kelly_manager.get_performance_summary()
        
        # Verify performance metrics
        self.assertEqual(performance['bets_placed'], len(placed_bets))
        self.assertEqual(performance['bets_settled'], len(placed_bets))
        self.assertIn('roi', performance)
    
    def test_multiple_strategies_with_stop_loss(self):
        """Test multiple strategies with StopLossManager."""
        # Get bets from multiple strategies
        value_results = self.value_strategy.from_model_predictions(
            predictions_df=self.test_predictions.rename(columns={'probability': 'predicted_probability'}),
            odds_df=self.test_odds,
            bankroll=self.stop_loss_manager.current_bankroll
        )
        
        predictions_1x2 = self.test_predictions.copy()
        # Convert to format needed for DNB
        home_rows = predictions_1x2[predictions_1x2['market'] == 'home'].copy()
        draw_rows = predictions_1x2[predictions_1x2['market'] == 'draw'].copy()
        away_rows = predictions_1x2[predictions_1x2['market'] == 'away'].copy()
        
        # Prepare 1X2 market data for each match
        match_data = []
        match_ids = predictions_1x2['match_id'].unique()
        
        for match_id in match_ids:
            home_prob = home_rows[home_rows['match_id'] == match_id]['probability'].values[0]
            draw_prob = draw_rows[draw_rows['match_id'] == match_id]['probability'].values[0]
            away_prob = away_rows[away_rows['match_id'] == match_id]['probability'].values[0]
            
            home_odds = self.test_odds[(self.test_odds['match_id'] == match_id) & 
                                      (self.test_odds['market'] == 'home')]['odds'].values[0]
            draw_odds = self.test_odds[(self.test_odds['match_id'] == match_id) & 
                                      (self.test_odds['market'] == 'draw')]['odds'].values[0]
            away_odds = self.test_odds[(self.test_odds['match_id'] == match_id) & 
                                      (self.test_odds['market'] == 'away')]['odds'].values[0]
            
            home_team = home_rows[home_rows['match_id'] == match_id]['home_team'].values[0]
            away_team = home_rows[home_rows['match_id'] == match_id]['away_team'].values[0]
            league = home_rows[home_rows['match_id'] == match_id]['league'].values[0]
            
            match_data.append({
                'match_id': match_id,
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'odds': {'home': home_odds, 'draw': draw_odds, 'away': away_odds},
                'predictions': {'home': home_prob, 'draw': draw_prob, 'away': away_prob},
                'date': datetime.now()
            })
        
        # Evaluate each match with DNB strategy
        dnb_results = []
        for match in match_data:
            result = self.dnb_strategy.evaluate_bet(**match)
            if result:
                dnb_results.append(result)
        
        # Combine results from both strategies
        all_results = value_results + dnb_results
        
        # Process bets with stop loss manager
        placed_bets = []
        for bet in all_results:
            bet_data = bet.to_dict()
            stake = self.stop_loss_manager.calculate_recommended_stake(bet_data)
            
            # Place bet
            bet_id = f"bet_{len(placed_bets) + 1}"
            success = self.stop_loss_manager.place_bet(
                amount=stake,
                match_id=bet.match_id,
                strategy_name=bet.strategy_name,
                bet_description=bet.bet_description,
                bet_id=bet_id,
                extra=bet_data
            )
            
            if success:
                placed_bets.append({
                    'bet_id': bet_id,
                    'strategy_result': bet,
                    'stake': stake
                })
        
        # Verify bets were placed
        self.assertGreater(len(placed_bets), 0)
        
        # Settle bets based on results
        for bet in placed_bets:
            match_id = bet['strategy_result'].match_id
            bet_type = bet['strategy_result'].bet_type
            
            # Find match result
            match_result = self.test_results[self.test_results['match_id'] == match_id].iloc[0]
            
            # Determine outcome
            if bet_type == BetType.HOME:
                outcome = 'win' if match_result['home_score'] > match_result['away_score'] else 'loss'
            elif bet_type == BetType.AWAY:
                outcome = 'win' if match_result['away_score'] > match_result['home_score'] else 'loss'
            elif bet_type == BetType.DRAW:
                outcome = 'win' if match_result['home_score'] == match_result['away_score'] else 'loss'
            else:
                outcome = 'loss'  # Default for more complex bet types not handled here
                
            # Calculate win amount for wins
            win_amount = None
            if outcome == 'win':
                win_amount = bet['stake'] * bet['strategy_result'].odds
                
            # Settle the bet
            self.stop_loss_manager.settle_bet(
                bet_id=bet['bet_id'],
                outcome=outcome,
                win_amount=win_amount
            )
            
        # Get stop loss status
        stop_status = self.stop_loss_manager.get_stop_status()
        
        # Verify stop loss tracking is working
        self.assertIn('is_stopped', stop_status)
        self.assertIn('current_streak', stop_status)
        self.assertIn('daily_loss', stop_status)
        
        # Get performance summary
        performance = self.stop_loss_manager.get_performance_summary()
        
        # Verify performance metrics
        self.assertEqual(performance['bets_placed'], len(placed_bets))
        self.assertEqual(performance['bets_settled'], len(placed_bets))
        self.assertIn('roi', performance)
        self.assertIn('stop_loss_status', performance)


@unittest.skipIf(not BACKTEST_ENGINE_AVAILABLE, "BacktestEngine not available")
class TestBacktestEngineIntegration(unittest.TestCase):
    """Test BacktestEngine integration with betting strategies."""
    
    def setUp(self):
        """Set up common test variables."""
        # Create a temporary directory for test data and results
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self.temp_dir.name, 'data')
        self.results_dir = os.path.join(self.temp_dir.name, 'results')
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create test data
        self.create_test_data()
        
        # Initialize strategies
        self.value_strategy = ValueBettingStrategy(
            min_edge=0.05,
            min_odds=1.5,
            max_odds=5.0
        )
        
        self.dnb_strategy = DrawNoBetStrategy(
            min_edge=0.04,
            min_odds=1.3,
            max_odds=3.0
        )
        
        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(
            strategies=[self.value_strategy, self.dnb_strategy],
            initial_bankroll=1000.0,
            data_dir=self.data_dir,
            result_dir=self.results_dir,
            verbose=False
        )
    
    def tearDown(self):
        """Clean up temporary files after test."""
        self.temp_dir.cleanup()
    
    def create_test_data(self):
        """Create test data files for backtesting."""
        # Create historical odds data
        odds_data = []
        
        # Generate 10 days of data for 5 matches per day
        for day in range(10):
            date = (datetime.now() - timedelta(days=10-day)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            for match_id in range(day*5 + 1, day*5 + 6):
                home_odds = np.random.uniform(1.5, 3.0)
                draw_odds = np.random.uniform(3.0, 4.5)
                away_odds = np.random.uniform(2.0, 5.0)
                
                # Home market
                odds_data.append({
                    'match_id': match_id,
                    'date': date,
                    'market': 'home',
                    'odds': home_odds,
                    'home_team': f"Team {match_id*2-1}",
                    'away_team': f"Team {match_id*2}",
                    'league': f"League {(match_id % 3) + 1}"
                })
                
                # Draw market
                odds_data.append({
                    'match_id': match_id,
                    'date': date,
                    'market': 'draw',
                    'odds': draw_odds,
                    'home_team': f"Team {match_id*2-1}",
                    'away_team': f"Team {match_id*2}",
                    'league': f"League {(match_id % 3) + 1}"
                })
                
                # Away market
                odds_data.append({
                    'match_id': match_id,
                    'date': date,
                    'market': 'away',
                    'odds': away_odds,
                    'home_team': f"Team {match_id*2-1}",
                    'away_team': f"Team {match_id*2}",
                    'league': f"League {(match_id % 3) + 1}"
                })
        
        # Create predictions data
        predictions_data = []
        
        for entry in odds_data:
            # Convert odds to probabilities with some noise
            implied_prob = 1 / entry['odds']
            predicted_prob = min(0.95, max(0.05, implied_prob + np.random.uniform(-0.1, 0.1)))
            
            predictions_data.append({
                'match_id': entry['match_id'],
                'date': entry['date'],
                'market': entry['market'],
                'predicted_probability': predicted_prob,
                'home_team': entry['home_team'],
                'away_team': entry['away_team'],
                'league': entry['league'],
                'confidence': np.random.uniform(0.6, 0.9)
            })
        
        # Create results data
        results_data = []
        match_ids = set(entry['match_id'] for entry in odds_data)
        
        for match_id in match_ids:
            # Get match details from first entry
            match_entry = next(entry for entry in odds_data if entry['match_id'] == match_id)
            
            # Generate random scores
            home_score = np.random.randint(0, 4)
            away_score = np.random.randint(0, 4)
            
            results_data.append({
                'match_id': match_id,
                'date': match_entry['date'],
                'home_score': home_score,
                'away_score': away_score,
                'home_team': match_entry['home_team'],
                'away_team': match_entry['away_team'],
                'league': match_entry['league'],
                'total_goals': home_score + away_score
            })
        
        # Save to CSV files
        pd.DataFrame(odds_data).to_csv(os.path.join(self.data_dir, 'historical_odds.csv'), index=False)
        pd.DataFrame(predictions_data).to_csv(os.path.join(self.data_dir, 'model_predictions.csv'), index=False)
        pd.DataFrame(results_data).to_csv(os.path.join(self.data_dir, 'match_results.csv'), index=False)
    
    def test_backtest_run(self):
        """Test running a backtest with the engine."""
        # Load data
        self.assertTrue(self.backtest_engine.load_data())
        
        # Run backtest
        results = self.backtest_engine.run_backtest()
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn('total_bets', results)
        self.assertIn('final_bankroll', results)
        
        # Check that bets were placed
        self.assertGreater(results.get('total_bets', 0), 0)
        
        # Check that we have daily balance data
        self.assertGreater(len(self.backtest_engine.daily_balance), 0)
        
        # Check that we have strategy performance data
        self.assertIn('strategy_performance', results)
        self.assertIn('Value Betting', results['strategy_performance'])
        
        # Test saving results
        self.assertTrue(self.backtest_engine.save_results())
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.results_dir, 'performance_metrics.json')))
        self.assertTrue(os.path.exists(os.path.join(self.results_dir, 'bets.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.results_dir, 'daily_balance.csv')))
    
    def test_multiple_strategies(self):
        """Test backtesting with multiple strategies."""
        # Add a third strategy
        market_movement_strategy = MarketMovementStrategy(
            min_movement_pct=0.05,
            min_odds=1.5,
            max_odds=5.0
        )
        
        self.backtest_engine.strategies.append(market_movement_strategy)
        
        # Load data
        self.assertTrue(self.backtest_engine.load_data())
        
        # Run backtest
        results = self.backtest_engine.run_backtest()
        
        # Verify results for all strategies
        self.assertIn('strategy_performance', results)
        strategy_names = self.backtest_engine.strategies
        
        for strategy in strategy_names:
            self.assertIn(strategy.name, results['strategy_performance'])
            
            # Check that each strategy has metrics
            strategy_metrics = results['strategy_performance'][strategy.name]
            self.assertIn('bets', strategy_metrics)
            self.assertIn('roi', strategy_metrics)
        
        # Generate and save report
        report = self.backtest_engine.generate_report(
            report_path=os.path.join(self.results_dir, 'backtest_report.json'),
            include_plots=True
        )
        
        self.assertIsInstance(report, dict)
        self.assertIn('summary_metrics', report)
        self.assertIn('strategies_tested', report)
        

if __name__ == '__main__':
    unittest.main() 