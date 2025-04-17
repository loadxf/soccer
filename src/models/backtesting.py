"""
Backtesting engine for betting strategies.

This module provides tools for backtesting and evaluating betting strategies
using historical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import warnings
from tqdm import tqdm

from src.models.betting_strategies import (
    BettingStrategy, 
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
    BankrollManager,
    KellyCriterionManager,
    StopLossManager,
    BankrollTracker
)

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting betting strategies against historical data.
    
    This class allows comprehensive backtesting of betting strategies
    using historical market data and prediction models.
    """
    
    def __init__(self, 
                 strategies: List[BettingStrategy],
                 bankroll_manager: Optional[BankrollManager] = None,
                 initial_bankroll: float = 1000.0,
                 test_start_date: Optional[datetime] = None,
                 test_end_date: Optional[datetime] = None,
                 data_dir: Optional[str] = None,
                 league_filter: Optional[List[str]] = None,
                 market_filter: Optional[List[str]] = None,
                 result_dir: Optional[str] = None,
                 parallel: bool = False,
                 verbose: bool = True):
        """
        Initialize the backtesting engine.
        
        Args:
            strategies: List of betting strategies to test
            bankroll_manager: Optional bankroll manager (default creates KellyCriterionManager)
            initial_bankroll: Starting bankroll
            test_start_date: Start date for backtest period
            test_end_date: End date for backtest period
            data_dir: Directory containing historical data
            league_filter: Optional list of leagues to include
            market_filter: Optional list of markets to include
            result_dir: Directory to save results
            parallel: Whether to use parallel processing
            verbose: Whether to display progress information
        """
        self.strategies = strategies
        self.initial_bankroll = initial_bankroll
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.data_dir = data_dir or Path("data/backtest")
        self.league_filter = league_filter
        self.market_filter = market_filter
        self.result_dir = result_dir or Path("results/backtest")
        self.parallel = parallel
        self.verbose = verbose
        
        # Create result directory if it doesn't exist
        if self.result_dir:
            Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        
        # Create or use bankroll manager
        if bankroll_manager:
            self.bankroll_manager = bankroll_manager
        else:
            self.bankroll_manager = KellyCriterionManager(
                initial_bankroll=initial_bankroll,
                name="Backtest Kelly Manager",
                fraction=0.5  # Conservative Kelly
            )
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.bets: List[Dict[str, Any]] = []
        self.daily_balance: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Data caching
        self._odds_data = None
        self._prediction_data = None
        self._results_data = None
    
    def load_data(self, 
                 odds_file: Optional[str] = None,
                 predictions_file: Optional[str] = None,
                 results_file: Optional[str] = None) -> bool:
        """
        Load historical data for backtesting.
        
        Args:
            odds_file: Path to historical odds data CSV
            predictions_file: Path to model predictions CSV
            results_file: Path to match results CSV
            
        Returns:
            bool: Whether loading was successful
        """
        # Determine file paths
        data_dir = Path(self.data_dir)
        odds_path = odds_file or data_dir / "historical_odds.csv"
        predictions_path = predictions_file or data_dir / "model_predictions.csv"
        results_path = results_file or data_dir / "match_results.csv"
        
        # Check if files exist
        files_exist = all(Path(p).exists() for p in [odds_path, predictions_path, results_path])
        if not files_exist:
            logger.error(f"Missing data files. Please ensure all required files exist.")
            return False
        
        try:
            # Load odds data
            self._odds_data = pd.read_csv(odds_path)
            
            # Convert date columns to datetime
            if 'date' in self._odds_data.columns:
                self._odds_data['date'] = pd.to_datetime(self._odds_data['date'])
                
            # Load predictions data
            self._prediction_data = pd.read_csv(predictions_path)
            if 'date' in self._prediction_data.columns:
                self._prediction_data['date'] = pd.to_datetime(self._prediction_data['date'])
                
            # Load results data
            self._results_data = pd.read_csv(results_path)
            if 'date' in self._results_data.columns:
                self._results_data['date'] = pd.to_datetime(self._results_data['date'])
                
            # Apply date filtering if specified
            if self.test_start_date:
                self._odds_data = self._odds_data[self._odds_data['date'] >= self.test_start_date]
                self._prediction_data = self._prediction_data[self._prediction_data['date'] >= self.test_start_date]
                self._results_data = self._results_data[self._results_data['date'] >= self.test_start_date]
                
            if self.test_end_date:
                self._odds_data = self._odds_data[self._odds_data['date'] <= self.test_end_date]
                self._prediction_data = self._prediction_data[self._prediction_data['date'] <= self.test_end_date]
                self._results_data = self._results_data[self._results_data['date'] <= self.test_end_date]
                
            # Apply league filtering if specified
            if self.league_filter and 'league' in self._odds_data.columns:
                self._odds_data = self._odds_data[self._odds_data['league'].isin(self.league_filter)]
                
            if self.league_filter and 'league' in self._prediction_data.columns:
                self._prediction_data = self._prediction_data[self._prediction_data['league'].isin(self.league_filter)]
                
            if self.league_filter and 'league' in self._results_data.columns:
                self._results_data = self._results_data[self._results_data['league'].isin(self.league_filter)]
                
            # Apply market filtering if specified
            if self.market_filter and 'market' in self._odds_data.columns:
                self._odds_data = self._odds_data[self._odds_data['market'].isin(self.market_filter)]
                
            if self.market_filter and 'market' in self._prediction_data.columns:
                self._prediction_data = self._prediction_data[self._prediction_data['market'].isin(self.market_filter)]
                
            logger.info(f"Successfully loaded {len(self._odds_data)} odds records, "
                       f"{len(self._prediction_data)} prediction records, and "
                       f"{len(self._results_data)} result records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def run_backtest(self, 
                    strategy_params: Optional[Dict[str, Dict[str, Any]]] = None,
                    max_active_bets: int = 50,
                    min_odds: float = 1.1,
                    max_odds: float = 10.0) -> Dict[str, Any]:
        """
        Run backtest simulation with loaded data.
        
        Args:
            strategy_params: Optional parameters for each strategy
            max_active_bets: Maximum number of active bets at once
            min_odds: Minimum odds to consider
            max_odds: Maximum odds to consider
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        if self._odds_data is None or self._prediction_data is None or self._results_data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return {}
            
        # Reset results
        self.results = []
        self.bets = []
        self.daily_balance = []
        self.performance_metrics = {}
        
        # Prepare backtest
        all_dates = sorted(self._odds_data['date'].unique())
        
        # Make sure we have match IDs for joining data
        if 'match_id' not in self._odds_data.columns or 'match_id' not in self._prediction_data.columns:
            logger.error("Data must contain 'match_id' column for joining")
            return {}
            
        # Configure strategies with params if provided
        if strategy_params:
            for strategy in self.strategies:
                if strategy.name in strategy_params:
                    for param, value in strategy_params[strategy.name].items():
                        setattr(strategy, param, value)
        
        # Reset bankroll manager
        self.bankroll_manager.reset(self.initial_bankroll)
        
        # Track active bets
        active_bets = {}
        
        # Run backtest day by day
        if self.verbose:
            date_iterator = tqdm(all_dates, desc="Backtesting")
        else:
            date_iterator = all_dates
            
        for current_date in date_iterator:
            # Get odds and predictions for current date
            daily_odds = self._odds_data[self._odds_data['date'] == current_date]
            daily_predictions = self._prediction_data[self._prediction_data['date'] == current_date]
            
            # Process each strategy
            for strategy in self.strategies:
                strategy_bets = []
                
                try:
                    # Use the from_model_predictions class method if available
                    if hasattr(strategy, 'from_model_predictions'):
                        strategy_results = strategy.from_model_predictions(
                            predictions_df=daily_predictions,
                            odds_df=daily_odds,
                            bankroll=self.bankroll_manager.available_balance()
                        )
                        strategy_bets.extend(strategy_results)
                except Exception as e:
                    logger.warning(f"Error running strategy {strategy.name}: {str(e)}")
                    continue
                
                # Filter bets by odds range
                strategy_bets = [bet for bet in strategy_bets 
                                if bet.odds and min_odds <= bet.odds <= max_odds]
                
                # Limit number of active bets if needed
                if len(active_bets) >= max_active_bets:
                    continue
                    
                # Process each bet
                for bet in strategy_bets:
                    # Skip if we have too many active bets
                    if len(active_bets) >= max_active_bets:
                        break
                        
                    # Calculate recommended stake
                    bet_data = bet.to_dict()
                    recommended_stake = self.bankroll_manager.calculate_recommended_stake(bet_data)
                    
                    # Place bet if stake is sufficient
                    if recommended_stake >= 1.0:  # Minimum $1 bet
                        bet_id = f"{bet.match_id}_{bet.bet_type.value}_{datetime.now().timestamp()}"
                        
                        # Place the bet
                        success = self.bankroll_manager.place_bet(
                            amount=recommended_stake,
                            match_id=bet.match_id,
                            strategy_name=strategy.name,
                            bet_description=bet.bet_description,
                            bet_id=bet_id,
                            extra=bet_data
                        )
                        
                        if success:
                            # Add to active bets
                            active_bets[bet_id] = {
                                'bet_id': bet_id,
                                'match_id': bet.match_id,
                                'bet_type': bet.bet_type.value,
                                'stake': recommended_stake,
                                'odds': bet.odds,
                                'strategy': strategy.name,
                                'placed_date': current_date,
                                'settled': False
                            }
                            
                            # Log the bet
                            self.bets.append(active_bets[bet_id])
            
            # Settle bets for matches that have results
            daily_results = self._results_data[self._results_data['date'] <= current_date]
            
            bets_to_remove = []
            for bet_id, bet in active_bets.items():
                if bet['settled']:
                    continue
                    
                # Find match result
                match_results = daily_results[daily_results['match_id'] == bet['match_id']]
                
                if not match_results.empty:
                    match_result = match_results.iloc[0]
                    
                    # Determine bet outcome
                    outcome = self._determine_bet_outcome(bet, match_result)
                    
                    # Calculate win amount if bet won
                    win_amount = None
                    if outcome == 'win':
                        win_amount = bet['stake'] * bet['odds']
                    
                    # Settle the bet
                    self.bankroll_manager.settle_bet(
                        bet_id=bet_id,
                        outcome=outcome,
                        win_amount=win_amount
                    )
                    
                    # Update bet record
                    bet['settled'] = True
                    bet['outcome'] = outcome
                    bet['settlement_date'] = current_date
                    bet['win_amount'] = win_amount if outcome == 'win' else 0
                    bet['profit'] = (win_amount - bet['stake']) if outcome == 'win' else -bet['stake']
                    
                    # Mark for removal from active bets
                    bets_to_remove.append(bet_id)
                    
                    # Update strategy performance if possible
                    for strategy in self.strategies:
                        if strategy.name == bet['strategy'] and hasattr(strategy, 'update_performance_tracking'):
                            strategy.update_performance_tracking(bet)
            
            # Remove settled bets from active list
            for bet_id in bets_to_remove:
                del active_bets[bet_id]
                
            # Record daily balance
            self.daily_balance.append({
                'date': current_date,
                'balance': self.bankroll_manager.current_bankroll,
                'active_bets': len(active_bets),
                'bets_placed': len([b for b in self.bets if b['placed_date'] == current_date]),
                'bets_settled': len(bets_to_remove)
            })
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
        return self.performance_metrics
    
    def _determine_bet_outcome(self, bet: Dict[str, Any], match_result: pd.Series) -> str:
        """
        Determine the outcome of a bet based on match result.
        
        Args:
            bet: Bet information
            match_result: Match result data
            
        Returns:
            str: Outcome ('win', 'loss', or 'push')
        """
        bet_type = bet['bet_type']
        
        # Handle different bet types
        if bet_type == 'Home':
            if 'home_score' in match_result and 'away_score' in match_result:
                return 'win' if match_result['home_score'] > match_result['away_score'] else 'loss'
                
        elif bet_type == 'Away':
            if 'home_score' in match_result and 'away_score' in match_result:
                return 'win' if match_result['away_score'] > match_result['home_score'] else 'loss'
                
        elif bet_type == 'Draw':
            if 'home_score' in match_result and 'away_score' in match_result:
                return 'win' if match_result['home_score'] == match_result['away_score'] else 'loss'
                
        elif bet_type == 'Over':
            if 'total_goals' in match_result and 'line' in bet:
                return 'win' if match_result['total_goals'] > bet['line'] else 'loss'
            elif 'home_score' in match_result and 'away_score' in match_result and 'line' in bet:
                total = match_result['home_score'] + match_result['away_score']
                return 'win' if total > bet['line'] else 'loss'
                
        elif bet_type == 'Under':
            if 'total_goals' in match_result and 'line' in bet:
                return 'win' if match_result['total_goals'] < bet['line'] else 'loss'
            elif 'home_score' in match_result and 'away_score' in match_result and 'line' in bet:
                total = match_result['home_score'] + match_result['away_score']
                return 'win' if total < bet['line'] else 'loss'
                
        elif bet_type == 'BTTS Yes':
            if 'home_score' in match_result and 'away_score' in match_result:
                return 'win' if match_result['home_score'] > 0 and match_result['away_score'] > 0 else 'loss'
                
        elif bet_type == 'BTTS No':
            if 'home_score' in match_result and 'away_score' in match_result:
                return 'win' if match_result['home_score'] == 0 or match_result['away_score'] == 0 else 'loss'
        
        # For more complex types like Asian Handicap, would need additional logic
                
        # Default to loss if can't determine
        logger.warning(f"Could not determine outcome for bet type {bet_type}")
        return 'loss'
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate and store performance metrics for the backtest."""
        metrics = {}
        
        # Basic metrics
        settled_bets = [b for b in self.bets if b.get('settled', False)]
        winning_bets = [b for b in settled_bets if b.get('outcome') == 'win']
        losing_bets = [b for b in settled_bets if b.get('outcome') == 'loss']
        push_bets = [b for b in settled_bets if b.get('outcome') == 'push']
        
        total_bets = len(settled_bets)
        
        metrics['total_bets'] = total_bets
        metrics['winning_bets'] = len(winning_bets)
        metrics['losing_bets'] = len(losing_bets)
        metrics['push_bets'] = len(push_bets)
        
        if total_bets > 0:
            metrics['win_rate'] = len(winning_bets) / total_bets
        else:
            metrics['win_rate'] = 0
            
        # Profit metrics
        total_stakes = sum(b.get('stake', 0) for b in settled_bets)
        total_returns = sum(b.get('win_amount', 0) for b in winning_bets)
        total_profit = total_returns - total_stakes
        
        metrics['total_stakes'] = total_stakes
        metrics['total_returns'] = total_returns
        metrics['total_profit'] = total_profit
        
        if total_stakes > 0:
            metrics['roi'] = total_profit / total_stakes
        else:
            metrics['roi'] = 0
            
        # Final bankroll
        metrics['initial_bankroll'] = self.initial_bankroll
        metrics['final_bankroll'] = self.bankroll_manager.current_bankroll
        metrics['bankroll_growth'] = metrics['final_bankroll'] - metrics['initial_bankroll']
        metrics['bankroll_growth_pct'] = (metrics['bankroll_growth'] / self.initial_bankroll) * 100
        
        # Strategy-specific metrics
        metrics['strategy_performance'] = {}
        
        for strategy in self.strategies:
            strategy_bets = [b for b in settled_bets if b.get('strategy') == strategy.name]
            
            if not strategy_bets:
                continue
                
            strategy_wins = [b for b in strategy_bets if b.get('outcome') == 'win']
            strategy_stakes = sum(b.get('stake', 0) for b in strategy_bets)
            strategy_returns = sum(b.get('win_amount', 0) for b in strategy_wins)
            strategy_profit = strategy_returns - strategy_stakes
            
            strategy_metrics = {
                'bets': len(strategy_bets),
                'wins': len(strategy_wins),
                'win_rate': len(strategy_wins) / len(strategy_bets) if strategy_bets else 0,
                'stakes': strategy_stakes,
                'returns': strategy_returns,
                'profit': strategy_profit,
                'roi': strategy_profit / strategy_stakes if strategy_stakes > 0 else 0
            }
            
            # Get advanced metrics from strategy if available
            if hasattr(strategy, 'get_performance_summary'):
                strategy_summary = strategy.get_performance_summary()
                strategy_metrics.update(strategy_summary)
                
            metrics['strategy_performance'][strategy.name] = strategy_metrics
        
        # Calculate metrics by time period
        daily_metrics = self._calculate_period_metrics('daily')
        weekly_metrics = self._calculate_period_metrics('weekly')
        monthly_metrics = self._calculate_period_metrics('monthly')
        
        metrics['daily_metrics'] = daily_metrics
        metrics['weekly_metrics'] = weekly_metrics
        metrics['monthly_metrics'] = monthly_metrics
        
        # Store metrics
        self.performance_metrics = metrics
    
    def _calculate_period_metrics(self, period: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate metrics for specific time periods.
        
        Args:
            period: Period type ('daily', 'weekly', 'monthly')
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Metrics by period
        """
        if not self.daily_balance:
            return {}
            
        # Convert to DataFrame for easier grouping
        balance_df = pd.DataFrame(self.daily_balance)
        balance_df['date'] = pd.to_datetime(balance_df['date'])
        
        # Create period grouper
        if period == 'daily':
            balance_df['period'] = balance_df['date'].dt.date
        elif period == 'weekly':
            balance_df['period'] = balance_df['date'].dt.to_period('W').apply(lambda x: x.start_time.date())
        elif period == 'monthly':
            balance_df['period'] = balance_df['date'].dt.to_period('M').apply(lambda x: x.start_time.date())
        else:
            return {}
            
        # Group by period
        grouped = balance_df.groupby('period')
        
        # Calculate metrics
        period_metrics = []
        
        for period_date, group in grouped:
            period_dict = {
                'period': period_date.isoformat() if hasattr(period_date, 'isoformat') else str(period_date),
                'start_balance': group['balance'].iloc[0] if not group.empty else 0,
                'end_balance': group['balance'].iloc[-1] if not group.empty else 0,
                'bets_placed': group['bets_placed'].sum(),
                'bets_settled': group['bets_settled'].sum()
            }
            
            period_dict['profit'] = period_dict['end_balance'] - period_dict['start_balance']
            period_dict['growth_pct'] = (period_dict['profit'] / period_dict['start_balance'] * 100) if period_dict['start_balance'] > 0 else 0
            
            period_metrics.append(period_dict)
            
        return period_metrics
    
    def plot_equity_curve(self, 
                        save_path: Optional[str] = None,
                        show_figure: bool = True) -> None:
        """
        Plot equity curve showing bankroll over time.
        
        Args:
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        if not self.daily_balance:
            logger.warning("No backtest data available for plotting")
            return
            
        # Convert to DataFrame
        balance_df = pd.DataFrame(self.daily_balance)
        balance_df['date'] = pd.to_datetime(balance_df['date'])
        
        # Set up plot
        plt.figure(figsize=(12, 6))
        sns.set_style('whitegrid')
        
        # Plot equity curve
        plt.plot(balance_df['date'], balance_df['balance'], linewidth=2, color='#1f77b4')
        
        # Add initial bankroll reference line
        plt.axhline(y=self.initial_bankroll, linestyle='--', alpha=0.7, color='red', label='Initial Bankroll')
        
        # Format plot
        plt.title('Equity Curve', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Bankroll', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        def currency_fmt(x, pos):
            return f"${x:,.0f}"
            
        plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_fmt))
        
        # Rotate date labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show if requested
        if show_figure:
            plt.show()
        else:
            plt.close()
            
    def plot_strategy_comparison(self,
                              metric: str = 'roi',
                              save_path: Optional[str] = None,
                              show_figure: bool = True) -> None:
        """
        Plot strategy comparison by performance metric.
        
        Args:
            metric: Metric to compare ('roi', 'win_rate', 'profit')
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        if not self.performance_metrics or 'strategy_performance' not in self.performance_metrics:
            logger.warning("No strategy metrics available for plotting")
            return
            
        # Extract metrics
        strategies = []
        values = []
        
        for strategy_name, metrics in self.performance_metrics['strategy_performance'].items():
            if metric in metrics:
                strategies.append(strategy_name)
                values.append(metrics[metric])
                
        if not strategies:
            logger.warning(f"No strategy data available for metric: {metric}")
            return
            
        # Create DataFrame
        strategy_df = pd.DataFrame({
            'Strategy': strategies,
            'Value': values
        })
        
        # Sort by value
        strategy_df = strategy_df.sort_values('Value', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.set_style('whitegrid')
        
        # Create bar plot
        ax = sns.barplot(x='Strategy', y='Value', data=strategy_df)
        
        # Format plot
        metric_name = metric.replace('_', ' ').title()
        plt.title(f'Strategy Comparison by {metric_name}', fontsize=16)
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(rotation=45)
        
        # Format y-axis based on metric
        from matplotlib.ticker import FuncFormatter
        
        if metric == 'roi':
            def percentage_fmt(x, pos):
                return f"{x*100:.1f}%"
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_fmt))
        elif metric == 'win_rate':
            def percentage_fmt(x, pos):
                return f"{x*100:.1f}%"
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_fmt))
        elif metric in ['profit', 'stakes', 'returns']:
            def currency_fmt(x, pos):
                return f"${x:,.0f}"
            ax.yaxis.set_major_formatter(FuncFormatter(currency_fmt))
            
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show if requested
        if show_figure:
            plt.show()
        else:
            plt.close()
    
    def generate_report(self, 
                      report_path: Optional[str] = None,
                      include_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report.
        
        Args:
            report_path: Optional path to save HTML report
            include_plots: Whether to include plots in HTML report
            
        Returns:
            Dict[str, Any]: Report data
        """
        if not self.performance_metrics:
            logger.warning("No backtest results available for reporting")
            return {}
            
        report = {
            'title': 'Betting Strategy Backtest Report',
            'generated_at': datetime.now().isoformat(),
            'test_period': {
                'start': self.test_start_date.isoformat() if self.test_start_date else None,
                'end': self.test_end_date.isoformat() if self.test_end_date else None
            },
            'summary_metrics': {
                'total_bets': self.performance_metrics.get('total_bets', 0),
                'win_rate': self.performance_metrics.get('win_rate', 0),
                'initial_bankroll': self.performance_metrics.get('initial_bankroll', 0),
                'final_bankroll': self.performance_metrics.get('final_bankroll', 0),
                'profit': self.performance_metrics.get('total_profit', 0),
                'roi': self.performance_metrics.get('roi', 0)
            },
            'strategies_tested': [s.name for s in self.strategies],
            'strategy_metrics': self.performance_metrics.get('strategy_performance', {})
        }
        
        # Generate plots if requested
        if include_plots and report_path:
            # Create report directory
            report_dir = Path(report_path).parent
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate plots
            plots_dir = report_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Equity curve
            equity_path = plots_dir / 'equity_curve.png'
            self.plot_equity_curve(save_path=str(equity_path), show_figure=False)
            report['plots'] = {'equity_curve': str(equity_path)}
            
            # Strategy comparison plots
            for metric in ['roi', 'win_rate', 'profit']:
                metric_path = plots_dir / f'strategy_{metric}.png'
                self.plot_strategy_comparison(
                    metric=metric,
                    save_path=str(metric_path),
                    show_figure=False
                )
                report['plots'][f'strategy_{metric}'] = str(metric_path)
                
        # Save report if path provided
        if report_path:
            try:
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Report saved to {report_path}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")
                
        return report
    
    def save_results(self, 
                   output_dir: Optional[str] = None) -> bool:
        """
        Save backtest results to files.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            bool: Whether saving was successful
        """
        if not self.performance_metrics:
            logger.warning("No backtest results to save")
            return False
            
        # Determine output directory
        output_path = Path(output_dir or self.result_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save performance metrics
            metrics_path = output_path / 'performance_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
                
            # Save bets as CSV
            bets_path = output_path / 'bets.csv'
            bets_df = pd.DataFrame(self.bets)
            bets_df.to_csv(bets_path, index=False)
            
            # Save daily balance as CSV
            balance_path = output_path / 'daily_balance.csv'
            balance_df = pd.DataFrame(self.daily_balance)
            balance_df.to_csv(balance_path, index=False)
            
            # Generate summary report
            report_path = output_path / 'backtest_report.json'
            self.generate_report(report_path=str(report_path), include_plots=True)
            
            logger.info(f"Results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            return False
            
    def reset(self) -> None:
        """Reset the backtest engine for a new test."""
        self.results = []
        self.bets = []
        self.daily_balance = []
        self.performance_metrics = {}
        
        if self.bankroll_manager:
            self.bankroll_manager.reset(self.initial_bankroll) 