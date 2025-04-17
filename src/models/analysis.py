"""
Analysis tools for betting strategies.

This module provides classes for analyzing betting strategy performance,
risk assessment, and strategy comparison.
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
import statsmodels.api as sm
from scipy import stats

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculate and analyze betting strategy performance metrics.
    
    This class provides comprehensive performance metrics for evaluating
    betting strategies, including ROI, profit factor, sharpe ratio, etc.
    """
    
    def __init__(self, bets_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None):
        """
        Initialize the PerformanceMetrics calculator.
        
        Args:
            bets_data: Optional betting history data
        """
        self.bets = None
        
        if bets_data is not None:
            self.load_data(bets_data)
            
    def load_data(self, bets_data: Union[List[Dict[str, Any]], pd.DataFrame]) -> None:
        """
        Load betting history data.
        
        Args:
            bets_data: Betting history data
        """
        if isinstance(bets_data, list):
            self.bets = pd.DataFrame(bets_data)
        elif isinstance(bets_data, pd.DataFrame):
            self.bets = bets_data.copy()
        else:
            raise ValueError("bets_data must be a list of dictionaries or a pandas DataFrame")
            
        # Ensure required columns exist
        required_cols = ['stake', 'odds']
        if not all(col in self.bets.columns for col in required_cols):
            raise ValueError(f"Bets data must contain columns: {required_cols}")
            
        # Convert date columns to datetime if present
        date_cols = ['placed_date', 'settlement_date', 'date']
        for col in date_cols:
            if col in self.bets.columns:
                if self.bets[col].dtype == 'object':
                    self.bets[col] = pd.to_datetime(self.bets[col])
                    
        # Add calculated columns if not present
        if 'profit' not in self.bets.columns:
            # Check if we have outcome and win_amount
            if 'outcome' in self.bets.columns and 'win_amount' in self.bets.columns:
                self.bets['profit'] = np.where(
                    self.bets['outcome'] == 'win',
                    self.bets['win_amount'] - self.bets['stake'],
                    -self.bets['stake']
                )
            elif 'win_amount' in self.bets.columns:
                self.bets['profit'] = self.bets['win_amount'] - self.bets['stake']
                
        # Add implied probability if not present
        if 'implied_probability' not in self.bets.columns and 'odds' in self.bets.columns:
            self.bets['implied_probability'] = 1 / self.bets['odds']
    
    def calculate_basic_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic performance metrics.
        
        Returns:
            Dict[str, Any]: Basic metrics
        """
        if self.bets is None or len(self.bets) == 0:
            return {}
            
        metrics = {}
        
        # Count metrics
        metrics['total_bets'] = len(self.bets)
        
        if 'outcome' in self.bets.columns:
            metrics['wins'] = (self.bets['outcome'] == 'win').sum()
            metrics['losses'] = (self.bets['outcome'] == 'loss').sum()
            metrics['pushes'] = (self.bets['outcome'] == 'push').sum()
            
            if metrics['total_bets'] > 0:
                metrics['win_rate'] = metrics['wins'] / metrics['total_bets']
            else:
                metrics['win_rate'] = 0
                
        # Financial metrics
        metrics['total_stakes'] = self.bets['stake'].sum()
        
        if 'profit' in self.bets.columns:
            metrics['total_profit'] = self.bets['profit'].sum()
            
            if metrics['total_stakes'] > 0:
                metrics['roi'] = metrics['total_profit'] / metrics['total_stakes']
            else:
                metrics['roi'] = 0
                
            # Profit metrics
            if 'outcome' in self.bets.columns:
                winning_bets = self.bets[self.bets['outcome'] == 'win']
                losing_bets = self.bets[self.bets['outcome'] == 'loss']
                
                metrics['gross_winnings'] = winning_bets['profit'].sum() if len(winning_bets) > 0 else 0
                metrics['gross_losses'] = abs(losing_bets['profit'].sum()) if len(losing_bets) > 0 else 0
                
                if metrics['gross_losses'] > 0:
                    metrics['profit_factor'] = metrics['gross_winnings'] / metrics['gross_losses']
                else:
                    metrics['profit_factor'] = float('inf') if metrics['gross_winnings'] > 0 else 0
        
        # Average metrics
        metrics['avg_odds'] = self.bets['odds'].mean()
        metrics['avg_stake'] = self.bets['stake'].mean()
        
        if 'profit' in self.bets.columns:
            metrics['avg_profit'] = self.bets['profit'].mean()
            
        # Volatility metrics
        if 'profit' in self.bets.columns:
            metrics['profit_std'] = self.bets['profit'].std()
            metrics['profit_volatility'] = metrics['profit_std'] / metrics['avg_stake'] if metrics['avg_stake'] > 0 else 0
        
        return metrics
    
    def calculate_advanced_metrics(self) -> Dict[str, Any]:
        """
        Calculate advanced performance metrics.
        
        Returns:
            Dict[str, Any]: Advanced metrics
        """
        if self.bets is None or len(self.bets) < 5:  # Need enough bets for meaningful analysis
            return {}
            
        metrics = {}
        
        # Calculate drawdown
        if 'profit' in self.bets.columns:
            # Sort by date if available
            if 'placed_date' in self.bets.columns:
                sorted_bets = self.bets.sort_values('placed_date')
            elif 'date' in self.bets.columns:
                sorted_bets = self.bets.sort_values('date')
            else:
                sorted_bets = self.bets
                
            # Calculate cumulative profit
            sorted_bets['cumulative_profit'] = sorted_bets['profit'].cumsum()
            
            # Calculate running maximum
            sorted_bets['running_max'] = sorted_bets['cumulative_profit'].cummax()
            
            # Calculate drawdown
            sorted_bets['drawdown'] = sorted_bets['running_max'] - sorted_bets['cumulative_profit']
            
            # Get maximum drawdown
            metrics['max_drawdown'] = sorted_bets['drawdown'].max()
            
            # Calculate average drawdown
            metrics['avg_drawdown'] = sorted_bets['drawdown'].mean()
            
            # Calculate drawdown duration
            if 'placed_date' in sorted_bets.columns:
                # Find periods of drawdown
                drawdown_periods = []
                in_drawdown = False
                start_date = None
                
                for i, row in sorted_bets.iterrows():
                    if row['drawdown'] > 0 and not in_drawdown:
                        in_drawdown = True
                        start_date = row['placed_date']
                    elif row['drawdown'] == 0 and in_drawdown:
                        in_drawdown = False
                        end_date = row['placed_date']
                        drawdown_periods.append((start_date, end_date))
                        
                # Calculate max drawdown duration in days
                if drawdown_periods:
                    drawdown_durations = [(end - start).days for start, end in drawdown_periods]
                    metrics['max_drawdown_duration'] = max(drawdown_durations)
                    metrics['avg_drawdown_duration'] = sum(drawdown_durations) / len(drawdown_durations)
            
            # Calculate Sharpe ratio (annualized)
            if 'profit' in self.bets.columns and len(self.bets) > 1:
                avg_return = self.bets['profit'].mean()
                std_return = self.bets['profit'].std()
                
                if std_return > 0:
                    # Assume 252 trading days per year
                    metrics['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252)
                else:
                    metrics['sharpe_ratio'] = 0
                    
            # Calculate Sortino ratio (only considers downside risk)
            if 'profit' in self.bets.columns and len(self.bets) > 1:
                avg_return = self.bets['profit'].mean()
                downside_returns = self.bets.loc[self.bets['profit'] < 0, 'profit']
                
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        # Assume 252 trading days per year
                        metrics['sortino_ratio'] = (avg_return / downside_std) * np.sqrt(252)
                    else:
                        metrics['sortino_ratio'] = float('inf') if avg_return > 0 else 0
                else:
                    metrics['sortino_ratio'] = float('inf') if avg_return > 0 else 0
                    
            # Calculate winning and losing streaks
            if 'outcome' in self.bets.columns:
                # Sort by date if available
                if 'placed_date' in self.bets.columns:
                    streak_bets = self.bets.sort_values('placed_date')
                elif 'date' in self.bets.columns:
                    streak_bets = self.bets.sort_values('date')
                else:
                    streak_bets = self.bets
                    
                # Initialize streak tracking
                current_streak = 0
                max_win_streak = 0
                max_loss_streak = 0
                
                for outcome in streak_bets['outcome']:
                    if outcome == 'win':
                        if current_streak > 0:
                            current_streak += 1
                        else:
                            current_streak = 1
                    elif outcome == 'loss':
                        if current_streak < 0:
                            current_streak -= 1
                        else:
                            current_streak = -1
                    else:  # push or other outcome
                        continue
                        
                    # Update max streaks
                    if current_streak > max_win_streak:
                        max_win_streak = current_streak
                    elif current_streak < -max_loss_streak:
                        max_loss_streak = -current_streak
                        
                metrics['max_win_streak'] = max_win_streak
                metrics['max_loss_streak'] = max_loss_streak
        
        return metrics
    
    def calculate_betting_efficiency(self) -> Dict[str, Any]:
        """
        Calculate betting efficiency metrics.
        
        Returns:
            Dict[str, Any]: Efficiency metrics
        """
        if self.bets is None or len(self.bets) == 0:
            return {}
            
        metrics = {}
        
        # Check if we have predicted probability or similar column
        prob_cols = ['predicted_probability', 'pred_prob', 'probability']
        pred_prob_col = next((col for col in prob_cols if col in self.bets.columns), None)
        
        if pred_prob_col and 'outcome' in self.bets.columns:
            # Group bets into probability buckets
            self.bets['prob_bucket'] = pd.cut(
                self.bets[pred_prob_col], 
                bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
            )
            
            # Calculate win rate by probability bucket
            win_rates = self.bets.groupby('prob_bucket')['outcome'].apply(
                lambda x: (x == 'win').mean()
            ).to_dict()
            
            metrics['win_rate_by_probability'] = win_rates
            
            # Calculate calibration error
            calibration_error = 0
            bucket_counts = self.bets.groupby('prob_bucket').size()
            total_bets = len(self.bets)
            
            for bucket, win_rate in win_rates.items():
                # Get midpoint of bucket (e.g., 0.15 for '10-20%')
                bucket_str = bucket.split('-')[0]
                bucket_midpoint = (float(bucket_str) + 5) / 100
                bucket_count = bucket_counts.get(bucket, 0)
                
                # Weight by number of bets in bucket
                calibration_error += abs(win_rate - bucket_midpoint) * (bucket_count / total_bets)
                
            metrics['calibration_error'] = calibration_error
            
            # Check if we have value rating
            if 'value_rating' in self.bets.columns:
                # Group by value rating buckets
                self.bets['value_bucket'] = pd.qcut(
                    self.bets['value_rating'], 
                    q=5, 
                    duplicates='drop',
                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                )
                
                # Calculate ROI by value bucket
                if 'profit' in self.bets.columns and 'stake' in self.bets.columns:
                    roi_by_value = self.bets.groupby('value_bucket').apply(
                        lambda x: x['profit'].sum() / x['stake'].sum() if x['stake'].sum() > 0 else 0
                    ).to_dict()
                    
                    metrics['roi_by_value_rating'] = roi_by_value
                    
        # Calculate closing line value if possible
        if all(col in self.bets.columns for col in ['odds', 'closing_odds']):
            # Calculate closing line value
            self.bets['closing_line_value'] = (self.bets['odds'] / self.bets['closing_odds']) - 1
            
            metrics['avg_closing_line_value'] = self.bets['closing_line_value'].mean()
            
            # Calculate correlation between CLV and profit
            if 'profit' in self.bets.columns:
                clv_profit_corr = self.bets[['closing_line_value', 'profit']].corr().iloc[0, 1]
                metrics['clv_profit_correlation'] = clv_profit_corr
                
            # Calculate ROI by CLV buckets
            if 'profit' in self.bets.columns and 'stake' in self.bets.columns:
                self.bets['clv_bucket'] = pd.qcut(
                    self.bets['closing_line_value'], 
                    q=5, 
                    duplicates='drop',
                    labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
                )
                
                roi_by_clv = self.bets.groupby('clv_bucket').apply(
                    lambda x: x['profit'].sum() / x['stake'].sum() if x['stake'].sum() > 0 else 0
                ).to_dict()
                
                metrics['roi_by_clv'] = roi_by_clv
        
        return metrics
    
    def segment_performance(self, segment_by: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Segment performance by various factors.
        
        Args:
            segment_by: List of columns to segment by
            
        Returns:
            Dict[str, Dict[str, Any]]: Segmented metrics
        """
        if self.bets is None or len(self.bets) == 0:
            return {}
            
        segmented_metrics = {}
        
        for segment in segment_by:
            if segment not in self.bets.columns:
                continue
                
            segment_values = self.bets[segment].unique()
            
            segmented_metrics[segment] = {}
            
            for value in segment_values:
                segment_bets = self.bets[self.bets[segment] == value]
                
                if len(segment_bets) == 0:
                    continue
                    
                # Create temporary PerformanceMetrics object
                temp_metrics = PerformanceMetrics(segment_bets)
                
                # Calculate metrics
                basic_metrics = temp_metrics.calculate_basic_metrics()
                
                segmented_metrics[segment][value] = basic_metrics
        
        return segmented_metrics
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all performance metrics.
        
        Returns:
            Dict[str, Any]: Complete metrics
        """
        if self.bets is None or len(self.bets) == 0:
            return {}
            
        all_metrics = {}
        
        # Get basic metrics
        basic_metrics = self.calculate_basic_metrics()
        all_metrics.update(basic_metrics)
        
        # Get advanced metrics
        advanced_metrics = self.calculate_advanced_metrics()
        all_metrics.update(advanced_metrics)
        
        # Get efficiency metrics
        efficiency_metrics = self.calculate_betting_efficiency()
        all_metrics.update(efficiency_metrics)
        
        # Add segments if common fields are present
        segment_fields = ['strategy', 'market', 'league', 'bet_type']
        available_segments = [field for field in segment_fields if field in self.bets.columns]
        
        if available_segments:
            segmented_metrics = self.segment_performance(available_segments)
            all_metrics['segments'] = segmented_metrics
            
        return all_metrics
    
    def plot_profit_curve(self,
                       save_path: Optional[str] = None,
                       show_figure: bool = True) -> None:
        """
        Plot cumulative profit curve.
        
        Args:
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        if self.bets is None or len(self.bets) == 0 or 'profit' not in self.bets.columns:
            logger.warning("No suitable data for plotting profit curve")
            return
            
        # Sort by date if available
        if 'placed_date' in self.bets.columns:
            sorted_bets = self.bets.sort_values('placed_date')
        elif 'date' in self.bets.columns:
            sorted_bets = self.bets.sort_values('date')
        else:
            sorted_bets = self.bets
            
        # Calculate cumulative profit
        sorted_bets['cumulative_profit'] = sorted_bets['profit'].cumsum()
        
        # Set up plot
        plt.figure(figsize=(12, 6))
        sns.set_style('whitegrid')
        
        # Date column for x-axis
        date_col = next((col for col in ['placed_date', 'date'] if col in sorted_bets.columns), None)
        
        if date_col:
            plt.plot(sorted_bets[date_col], sorted_bets['cumulative_profit'], linewidth=2)
            plt.gcf().autofmt_xdate()  # Rotate date labels
        else:
            # Use bet index if no date column
            plt.plot(range(len(sorted_bets)), sorted_bets['cumulative_profit'], linewidth=2)
            plt.xlabel('Bet Number')
            
        # Add reference line at 0
        plt.axhline(y=0, linestyle='--', alpha=0.7, color='red')
        
        # Format plot
        plt.title('Cumulative Profit Over Time', fontsize=16)
        plt.ylabel('Profit', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        from matplotlib.ticker import FuncFormatter
        def currency_fmt(x, pos):
            return f"${x:,.0f}"
            
        plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_fmt))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show if requested
        if show_figure:
            plt.show()
        else:
            plt.close()
    
    def plot_roi_by_factor(self,
                        factor: str,
                        save_path: Optional[str] = None,
                        show_figure: bool = True) -> None:
        """
        Plot ROI by a segmentation factor.
        
        Args:
            factor: Column to segment by
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        if self.bets is None or len(self.bets) == 0:
            logger.warning("No data for plotting ROI by factor")
            return
            
        if factor not in self.bets.columns:
            logger.warning(f"Factor '{factor}' not found in data")
            return
            
        if 'profit' not in self.bets.columns or 'stake' not in self.bets.columns:
            logger.warning("Profit and stake columns required for ROI calculation")
            return
            
        # Calculate ROI by factor
        roi_by_factor = self.bets.groupby(factor).apply(
            lambda x: x['profit'].sum() / x['stake'].sum() if x['stake'].sum() > 0 else 0
        ).reset_index()
        roi_by_factor.columns = [factor, 'roi']
        
        # Sort by ROI
        roi_by_factor = roi_by_factor.sort_values('roi', ascending=False)
        
        # Set up plot
        plt.figure(figsize=(10, 6))
        sns.set_style('whitegrid')
        
        # Create bar plot
        ax = sns.barplot(x=factor, y='roi', data=roi_by_factor)
        
        # Format plot
        factor_name = factor.replace('_', ' ').title()
        plt.title(f'ROI by {factor_name}', fontsize=16)
        plt.xlabel(factor_name, fontsize=12)
        plt.ylabel('ROI', fontsize=12)
        plt.xticks(rotation=45 if len(roi_by_factor) > 5 else 0)
        
        # Format y-axis as percentage
        from matplotlib.ticker import FuncFormatter
        def percentage_fmt(x, pos):
            return f"{x*100:.1f}%"
            
        ax.yaxis.set_major_formatter(FuncFormatter(percentage_fmt))
        
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
                      output_path: Optional[str] = None,
                      include_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            output_path: Path to save report
            include_plots: Whether to include plots
            
        Returns:
            Dict[str, Any]: Report data
        """
        if self.bets is None or len(self.bets) == 0:
            logger.warning("No data for generating report")
            return {}
            
        # Get all metrics
        all_metrics = self.get_all_metrics()
        
        # Create report structure
        report = {
            'title': 'Betting Performance Report',
            'generated_at': datetime.now().isoformat(),
            'metrics': all_metrics,
            'summary': self._generate_summary(all_metrics)
        }
        
        # Generate plots if requested
        if include_plots and output_path:
            plots_dir = Path(output_path).parent / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Profit curve
            profit_plot_path = plots_dir / 'profit_curve.png'
            self.plot_profit_curve(save_path=str(profit_plot_path), show_figure=False)
            
            # ROI by segments
            for factor in ['strategy', 'market', 'league', 'bet_type']:
                if factor in self.bets.columns:
                    factor_plot_path = plots_dir / f'roi_by_{factor}.png'
                    self.plot_roi_by_factor(
                        factor=factor,
                        save_path=str(factor_plot_path),
                        show_figure=False
                    )
                    
            report['plots'] = {
                'profit_curve': str(profit_plot_path)
            }
            
        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
                logger.info(f"Performance report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")
                
        return report
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a text summary of performance metrics.
        
        Args:
            metrics: Performance metrics
            
        Returns:
            str: Summary text
        """
        if not metrics:
            return "No performance data available."
            
        total_bets = metrics.get('total_bets', 0)
        roi = metrics.get('roi', 0) * 100  # Convert to percentage
        profit = metrics.get('total_profit', 0)
        win_rate = metrics.get('win_rate', 0) * 100  # Convert to percentage
        
        summary = [
            f"Performance Summary ({total_bets} bets)",
            f"Total Profit: ${profit:.2f}",
            f"ROI: {roi:.2f}%",
            f"Win Rate: {win_rate:.2f}%"
        ]
        
        # Add additional metrics if available
        if 'profit_factor' in metrics:
            summary.append(f"Profit Factor: {metrics['profit_factor']:.2f}")
            
        if 'max_drawdown' in metrics:
            summary.append(f"Maximum Drawdown: ${metrics['max_drawdown']:.2f}")
            
        if 'sharpe_ratio' in metrics:
            summary.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            
        if 'max_win_streak' in metrics and 'max_loss_streak' in metrics:
            summary.append(f"Max Win Streak: {metrics['max_win_streak']}, Max Loss Streak: {metrics['max_loss_streak']}")
            
        # Add closing line value if available
        if 'avg_closing_line_value' in metrics:
            clv_pct = metrics['avg_closing_line_value'] * 100
            summary.append(f"Avg Closing Line Value: {clv_pct:.2f}%")
            
        return "\n".join(summary)

class RiskAnalyzer:
    """
    Analyze risk in betting strategies and bankroll management.
    
    This class provides tools for assessing and managing risk in betting
    strategies, including drawdown analysis, Kelly criterion calculation,
    and risk-adjusted returns.
    """
    
    def __init__(self, 
                bets_data: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
                initial_bankroll: float = 1000.0):
        """
        Initialize the RiskAnalyzer.
        
        Args:
            bets_data: Optional betting history data
            initial_bankroll: Initial bankroll amount
        """
        self.bets = None
        self.initial_bankroll = initial_bankroll
        
        if bets_data is not None:
            self.load_data(bets_data)
            
    def load_data(self, bets_data: Union[List[Dict[str, Any]], pd.DataFrame]) -> None:
        """
        Load betting history data.
        
        Args:
            bets_data: Betting history data
        """
        if isinstance(bets_data, list):
            self.bets = pd.DataFrame(bets_data)
        elif isinstance(bets_data, pd.DataFrame):
            self.bets = bets_data.copy()
        else:
            raise ValueError("bets_data must be a list of dictionaries or a pandas DataFrame")
            
        # Ensure required columns exist
        required_cols = ['stake']
        if not all(col in self.bets.columns for col in required_cols):
            raise ValueError(f"Bets data must contain columns: {required_cols}")
            
        # Convert date columns to datetime if present
        date_cols = ['placed_date', 'settlement_date', 'date']
        for col in date_cols:
            if col in self.bets.columns:
                if self.bets[col].dtype == 'object':
                    self.bets[col] = pd.to_datetime(self.bets[col])
    
    def analyze_drawdown(self) -> Dict[str, Any]:
        """
        Analyze drawdown characteristics.
        
        Returns:
            Dict[str, Any]: Drawdown metrics
        """
        if self.bets is None or len(self.bets) < 2 or 'profit' not in self.bets.columns:
            return {}
            
        # Sort by date if available
        if 'placed_date' in self.bets.columns:
            sorted_bets = self.bets.sort_values('placed_date')
        elif 'date' in self.bets.columns:
            sorted_bets = self.bets.sort_values('date')
        else:
            sorted_bets = self.bets
            
        # Calculate cumulative profit and bankroll
        sorted_bets['cumulative_profit'] = sorted_bets['profit'].cumsum()
        sorted_bets['bankroll'] = self.initial_bankroll + sorted_bets['cumulative_profit']
        
        # Calculate running maximum
        sorted_bets['running_max'] = sorted_bets['bankroll'].cummax()
        
        # Calculate drawdown in monetary terms
        sorted_bets['drawdown'] = sorted_bets['running_max'] - sorted_bets['bankroll']
        
        # Calculate drawdown as percentage
        sorted_bets['drawdown_pct'] = sorted_bets['drawdown'] / sorted_bets['running_max']
        
        # Find maximum drawdown
        max_dd = sorted_bets['drawdown'].max()
        max_dd_pct = sorted_bets['drawdown_pct'].max()
        
        # Find when max drawdown occurred
        max_dd_idx = sorted_bets['drawdown'].idxmax()
        
        # Calculate drawdown duration
        drawdown_periods = []
        current_period = None
        
        for i, row in sorted_bets.iterrows():
            if row['drawdown'] > 0 and current_period is None:
                # Start of drawdown period
                current_period = {
                    'start_idx': i,
                    'start_bankroll': row['running_max'],
                    'lowest_bankroll': row['bankroll'],
                    'lowest_idx': i
                }
            elif row['drawdown'] > 0 and current_period is not None:
                # Continuing drawdown period
                if row['bankroll'] < current_period['lowest_bankroll']:
                    # Update lowest point
                    current_period['lowest_bankroll'] = row['bankroll']
                    current_period['lowest_idx'] = i
            elif row['drawdown'] == 0 and current_period is not None:
                # End of drawdown period
                current_period['end_idx'] = i
                current_period['end_bankroll'] = row['bankroll']
                current_period['drawdown'] = current_period['start_bankroll'] - current_period['lowest_bankroll']
                current_period['drawdown_pct'] = current_period['drawdown'] / current_period['start_bankroll']
                
                # Add duration if date column available
                if 'placed_date' in sorted_bets.columns:
                    start_date = sorted_bets.loc[current_period['start_idx'], 'placed_date']
                    end_date = sorted_bets.loc[current_period['end_idx'], 'placed_date']
                    duration = (end_date - start_date).days
                    current_period['duration_days'] = duration
                    
                # Add to periods list
                drawdown_periods.append(current_period)
                current_period = None
                
        # Handle if we're still in a drawdown at the end
        if current_period is not None:
            current_period['end_idx'] = sorted_bets.index[-1]
            current_period['end_bankroll'] = sorted_bets.iloc[-1]['bankroll']
            current_period['drawdown'] = current_period['start_bankroll'] - current_period['lowest_bankroll']
            current_period['drawdown_pct'] = current_period['drawdown'] / current_period['start_bankroll']
            
            # Add duration if date column available
            if 'placed_date' in sorted_bets.columns:
                start_date = sorted_bets.loc[current_period['start_idx'], 'placed_date']
                end_date = sorted_bets.loc[current_period['end_idx'], 'placed_date']
                duration = (end_date - start_date).days
                current_period['duration_days'] = duration
                
            drawdown_periods.append(current_period)
        
        # Calculate metrics for all drawdown periods
        metrics = {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'average_drawdown': sorted_bets['drawdown'].mean(),
            'average_drawdown_pct': sorted_bets['drawdown_pct'].mean(),
            'drawdown_periods_count': len(drawdown_periods)
        }
        
        # Get max drawdown period
        if drawdown_periods:
            max_dd_period = max(drawdown_periods, key=lambda x: x['drawdown'])
            metrics['max_drawdown_period'] = max_dd_period
            
            # Calculate average duration
            if 'placed_date' in sorted_bets.columns:
                durations = [p.get('duration_days', 0) for p in drawdown_periods if 'duration_days' in p]
                if durations:
                    metrics['average_drawdown_duration'] = sum(durations) / len(durations)
                    metrics['max_drawdown_duration'] = max(durations)
        
        return metrics
    
    def calculate_kelly_criterion(self, edge: Optional[float] = None, odds: Optional[float] = None) -> float:
        """
        Calculate Kelly criterion optimal stake percentage.
        
        Args:
            edge: Optional edge (can be calculated from data if not provided)
            odds: Optional odds (can be calculated from data if not provided)
            
        Returns:
            float: Kelly criterion stake percentage
        """
        # Use provided edge and odds, or calculate from data
        if edge is None or odds is None:
            if self.bets is None or len(self.bets) == 0:
                return 0.0
                
            # Calculate average win rate
            if 'outcome' in self.bets.columns:
                win_rate = (self.bets['outcome'] == 'win').mean()
            else:
                # If no outcome column, try to infer from profit
                if 'profit' in self.bets.columns:
                    win_rate = (self.bets['profit'] > 0).mean()
                else:
                    return 0.0
                    
            # Calculate average odds
            if 'odds' in self.bets.columns:
                avg_odds = self.bets['odds'].mean()
            else:
                return 0.0
                
            # Calculate edge
            implied_prob = 1 / avg_odds
            edge = win_rate - implied_prob
        
        # Basic Kelly formula: f = (bp - q) / b
        # where:
        # - f is the fraction of the bankroll to wager
        # - b is the decimal odds minus 1
        # - p is the probability of winning
        # - q is the probability of losing (1 - p)
        
        if odds <= 1.0:
            return 0.0
            
        b = odds - 1.0
        p = edge + (1 / odds)  # Convert edge back to win probability
        q = 1 - p
        
        # Calculate Kelly fraction
        if b > 0 and p > 0:
            kelly = (b * p - q) / b
        else:
            kelly = 0.0
            
        # Kelly can be negative for negative edge bets
        return max(0.0, kelly)
    
    def optimize_staking_plan(self, 
                            method: str = 'kelly', 
                            risk_fraction: float = 1.0) -> Dict[str, Any]:
        """
        Optimize staking plan based on historical bets.
        
        Args:
            method: Staking method ('kelly', 'flat', 'proportional')
            risk_fraction: Fraction of optimal Kelly to use (0-1)
            
        Returns:
            Dict[str, Any]: Optimized staking plan
        """
        if self.bets is None or len(self.bets) < 10:
            return {'error': 'Insufficient data for optimization'}
            
        # Group by relevant segments
        segment_columns = ['bet_type', 'strategy', 'market', 'league']
        available_segments = [col for col in segment_columns if col in self.bets.columns]
        
        results = {
            'method': method,
            'risk_fraction': risk_fraction,
            'overall': {},
            'segments': {}
        }
        
        # Calculate overall optimal staking
        kelly = self.calculate_kelly_criterion()
        optimal_stake_pct = kelly * risk_fraction
        
        results['overall'] = {
            'kelly_criterion': kelly,
            'optimal_stake_pct': optimal_stake_pct,
            'optimal_stake': self.initial_bankroll * optimal_stake_pct
        }
        
        # Calculate for segments if available
        for segment in available_segments:
            results['segments'][segment] = {}
            
            for value in self.bets[segment].unique():
                segment_bets = self.bets[self.bets[segment] == value]
                
                if len(segment_bets) < 5:
                    continue
                    
                # Calculate win rate for this segment
                if 'outcome' in segment_bets.columns:
                    win_rate = (segment_bets['outcome'] == 'win').mean()
                else:
                    # If no outcome column, try to infer from profit
                    if 'profit' in segment_bets.columns:
                        win_rate = (segment_bets['profit'] > 0).mean()
                    else:
                        continue
                        
                # Calculate average odds for this segment
                if 'odds' in segment_bets.columns:
                    avg_odds = segment_bets['odds'].mean()
                else:
                    continue
                    
                # Calculate edge
                implied_prob = 1 / avg_odds
                edge = win_rate - implied_prob
                
                # Calculate Kelly for this segment
                segment_kelly = self.calculate_kelly_criterion(edge, avg_odds)
                segment_optimal_stake_pct = segment_kelly * risk_fraction
                
                results['segments'][segment][value] = {
                    'win_rate': win_rate,
                    'avg_odds': avg_odds,
                    'edge': edge,
                    'kelly_criterion': segment_kelly,
                    'optimal_stake_pct': segment_optimal_stake_pct,
                    'optimal_stake': self.initial_bankroll * segment_optimal_stake_pct,
                    'sample_size': len(segment_bets)
                }
        
        return results
    
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate various risk-adjusted performance metrics.
        
        Returns:
            Dict[str, Any]: Risk-adjusted metrics
        """
        if self.bets is None or len(self.bets) < 10 or 'profit' not in self.bets.columns:
            return {}
            
        # Calculate basic statistics
        avg_profit = self.bets['profit'].mean()
        std_profit = self.bets['profit'].std()
        
        # Calculate total stake and ROI
        total_stake = self.bets['stake'].sum()
        total_profit = self.bets['profit'].sum()
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        # Get drawdown metrics
        drawdown = self.analyze_drawdown()
        max_drawdown = drawdown.get('max_drawdown', 0)
        max_drawdown_pct = drawdown.get('max_drawdown_pct', 0)
        
        metrics = {
            'avg_profit': avg_profit,
            'std_profit': std_profit,
            'total_profit': total_profit,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct
        }
        
        # Calculate risk-adjusted metrics
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if std_profit > 0:
            metrics['sharpe_ratio'] = avg_profit / std_profit
        else:
            metrics['sharpe_ratio'] = float('inf') if avg_profit > 0 else 0
            
        # Sortino ratio (only considers downside risk)
        downside_returns = self.bets.loc[self.bets['profit'] < 0, 'profit']
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                metrics['sortino_ratio'] = avg_profit / downside_std
            else:
                metrics['sortino_ratio'] = float('inf') if avg_profit > 0 else 0
        else:
            metrics['sortino_ratio'] = float('inf') if avg_profit > 0 else 0
            
        # Calmar ratio (return / max drawdown)
        if max_drawdown_pct > 0:
            metrics['calmar_ratio'] = roi / max_drawdown_pct
        else:
            metrics['calmar_ratio'] = float('inf') if roi > 0 else 0
            
        # Risk of ruin
        # Simple estimate based on Kelly criterion
        kelly = self.calculate_kelly_criterion()
        if kelly > 0:
            metrics['risk_of_ruin'] = np.exp(-2 * self.initial_bankroll * kelly / std_profit)
        else:
            metrics['risk_of_ruin'] = 1.0
        
        return metrics
    
    def simulate_bankroll_paths(self, 
                              n_simulations: int = 1000, 
                              n_bets: int = 100,
                              stake_method: str = 'flat',
                              stake_pct: float = 0.02) -> Dict[str, Any]:
        """
        Simulate future bankroll paths using Monte Carlo simulation.
        
        Args:
            n_simulations: Number of simulations to run
            n_bets: Number of bets per simulation
            stake_method: Staking method ('flat', 'kelly', 'proportional')
            stake_pct: Percentage of bankroll to stake
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        if self.bets is None or len(self.bets) < 10:
            return {'error': 'Insufficient data for simulation'}
            
        # Calculate win rate and average odds
        if 'outcome' in self.bets.columns:
            win_rate = (self.bets['outcome'] == 'win').mean()
        else:
            # Try to infer from profit
            if 'profit' in self.bets.columns:
                win_rate = (self.bets['profit'] > 0).mean()
            else:
                return {'error': 'Cannot determine win rate from data'}
                
        # Get odds distribution if available
        if 'odds' in self.bets.columns:
            odds_mean = self.bets['odds'].mean()
            odds_std = self.bets['odds'].std()
            odds_min = self.bets['odds'].min()
        else:
            return {'error': 'Odds data required for simulation'}
            
        # Prepare for simulation
        simulation_results = []
        
        for sim in range(n_simulations):
            # Start with initial bankroll
            bankroll = self.initial_bankroll
            bankroll_path = [bankroll]
            
            for bet in range(n_bets):
                if bankroll <= 0:
                    # Bankroll depleted
                    break
                    
                # Generate random odds (truncated normal distribution)
                odds = max(odds_min, np.random.normal(odds_mean, odds_std))
                
                # Calculate stake based on method
                if stake_method == 'flat':
                    stake = self.initial_bankroll * stake_pct
                elif stake_method == 'proportional':
                    stake = bankroll * stake_pct
                elif stake_method == 'kelly':
                    kelly = self.calculate_kelly_criterion()
                    stake = bankroll * kelly * stake_pct  # Fractional Kelly
                else:
                    stake = bankroll * stake_pct
                    
                # Ensure stake doesn't exceed bankroll
                stake = min(stake, bankroll)
                
                # Determine outcome
                outcome = np.random.random() < win_rate
                
                # Update bankroll
                if outcome:
                    bankroll += stake * (odds - 1)
                else:
                    bankroll -= stake
                    
                # Record bankroll
                bankroll_path.append(bankroll)
                
            # Add padding if path shorter than n_bets
            bankroll_path.extend([bankroll] * (n_bets + 1 - len(bankroll_path)))
                
            # Add to results
            simulation_results.append(bankroll_path)
            
        # Calculate statistics
        final_bankrolls = [path[-1] for path in simulation_results]
        
        mean_final = np.mean(final_bankrolls)
        median_final = np.median(final_bankrolls)
        std_final = np.std(final_bankrolls)
        
        percentiles = np.percentile(final_bankrolls, [5, 10, 25, 50, 75, 90, 95])
        
        # Calculate probability of ruin
        ruin_count = sum(1 for b in final_bankrolls if b <= 0.1 * self.initial_bankroll)
        prob_ruin = ruin_count / n_simulations
        
        # Calculate probability of target return
        target_return = 2 * self.initial_bankroll  # Double initial bankroll
        target_count = sum(1 for b in final_bankrolls if b >= target_return)
        prob_target = target_count / n_simulations
        
        # Convert paths to numpy array for percentile calculations
        paths_array = np.array(simulation_results)
        
        # Calculate path percentiles
        median_path = np.median(paths_array, axis=0)
        lower_5pct_path = np.percentile(paths_array, 5, axis=0)
        lower_25pct_path = np.percentile(paths_array, 25, axis=0)
        upper_75pct_path = np.percentile(paths_array, 75, axis=0)
        upper_95pct_path = np.percentile(paths_array, 95, axis=0)
        
        return {
            'initial_bankroll': self.initial_bankroll,
            'win_rate': win_rate,
            'avg_odds': odds_mean,
            'simulation_count': n_simulations,
            'bets_per_simulation': n_bets,
            'stake_method': stake_method,
            'stake_pct': stake_pct,
            'final_bankroll_mean': mean_final,
            'final_bankroll_median': median_final,
            'final_bankroll_std': std_final,
            'percentiles': {
                'p5': percentiles[0],
                'p10': percentiles[1],
                'p25': percentiles[2],
                'p50': percentiles[3],
                'p75': percentiles[4],
                'p90': percentiles[5],
                'p95': percentiles[6]
            },
            'probability_of_ruin': prob_ruin,
            'probability_of_target_return': prob_target,
            'median_path': median_path.tolist(),
            'lower_5pct_path': lower_5pct_path.tolist(),
            'lower_25pct_path': lower_25pct_path.tolist(),
            'upper_75pct_path': upper_75pct_path.tolist(),
            'upper_95pct_path': upper_95pct_path.tolist()
        }
    
    def plot_drawdown_analysis(self,
                             save_path: Optional[str] = None,
                             show_figure: bool = True) -> None:
        """
        Plot drawdown analysis.
        
        Args:
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        if self.bets is None or len(self.bets) < 2 or 'profit' not in self.bets.columns:
            logger.warning("Insufficient data for drawdown analysis")
            return
            
        # Sort by date if available
        if 'placed_date' in self.bets.columns:
            sorted_bets = self.bets.sort_values('placed_date')
        elif 'date' in self.bets.columns:
            sorted_bets = self.bets.sort_values('date')
        else:
            sorted_bets = self.bets
            
        # Calculate cumulative profit and bankroll
        sorted_bets['cumulative_profit'] = sorted_bets['profit'].cumsum()
        sorted_bets['bankroll'] = self.initial_bankroll + sorted_bets['cumulative_profit']
        
        # Calculate running maximum and drawdown
        sorted_bets['running_max'] = sorted_bets['bankroll'].cummax()
        sorted_bets['drawdown'] = sorted_bets['running_max'] - sorted_bets['bankroll']
        sorted_bets['drawdown_pct'] = sorted_bets['drawdown'] / sorted_bets['running_max']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Date column for x-axis
        date_col = next((col for col in ['placed_date', 'date'] if col in sorted_bets.columns), None)
        
        # Plot bankroll and running maximum
        if date_col:
            ax1.plot(sorted_bets[date_col], sorted_bets['bankroll'], label='Bankroll')
            ax1.plot(sorted_bets[date_col], sorted_bets['running_max'], label='Running Maximum', linestyle='--')
        else:
            ax1.plot(range(len(sorted_bets)), sorted_bets['bankroll'], label='Bankroll')
            ax1.plot(range(len(sorted_bets)), sorted_bets['running_max'], label='Running Maximum', linestyle='--')
        
        # Plot drawdown
        if date_col:
            ax2.fill_between(sorted_bets[date_col], 0, sorted_bets['drawdown_pct'] * 100, color='red', alpha=0.3)
        else:
            ax2.fill_between(range(len(sorted_bets)), 0, sorted_bets['drawdown_pct'] * 100, color='red', alpha=0.3)
        
        # Format plots
        ax1.set_title('Bankroll and Running Maximum', fontsize=14)
        ax1.set_ylabel('Bankroll ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Drawdown Percentage', fontsize=14)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
        
        if date_col:
            plt.gcf().autofmt_xdate()  # Rotate date labels
        else:
            ax2.set_xlabel('Bet Number', fontsize=12)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show if requested
        if show_figure:
            plt.show()
        else:
            plt.close()
    
    def plot_monte_carlo_simulation(self,
                                  n_simulations: int = 100,
                                  n_bets: int = 100,
                                  stake_method: str = 'flat',
                                  stake_pct: float = 0.02,
                                  save_path: Optional[str] = None,
                                  show_figure: bool = True) -> None:
        """
        Plot Monte Carlo simulation of future bankroll paths.
        
        Args:
            n_simulations: Number of simulations to run
            n_bets: Number of bets per simulation
            stake_method: Staking method ('flat', 'kelly', 'proportional')
            stake_pct: Percentage of bankroll to stake
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        # Run simulation
        sim_results = self.simulate_bankroll_paths(
            n_simulations=n_simulations,
            n_bets=n_bets,
            stake_method=stake_method,
            stake_pct=stake_pct
        )
        
        if 'error' in sim_results:
            logger.warning(f"Error in simulation: {sim_results['error']}")
            return
            
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot median and percentile paths
        x = range(n_bets + 1)
        
        plt.plot(x, sim_results['median_path'], label='Median Path', linewidth=2, color='blue')
        plt.plot(x, sim_results['upper_95pct_path'], label='95th Percentile', linewidth=1.5, color='green')
        plt.plot(x, sim_results['upper_75pct_path'], label='75th Percentile', linewidth=1.5, color='lightgreen', linestyle='--')
        plt.plot(x, sim_results['lower_25pct_path'], label='25th Percentile', linewidth=1.5, color='orange', linestyle='--')
        plt.plot(x, sim_results['lower_5pct_path'], label='5th Percentile', linewidth=1.5, color='red')
        
        # Shade the area between 5th and 95th percentiles
        plt.fill_between(x, sim_results['lower_5pct_path'], sim_results['upper_95pct_path'], alpha=0.1, color='blue')
        
        # Add reference line for initial bankroll
        plt.axhline(y=self.initial_bankroll, linestyle='--', alpha=0.7, color='gray', label='Initial Bankroll')
        
        # Format plot
        plt.title(f'Monte Carlo Simulation: {n_simulations} Paths over {n_bets} Bets', fontsize=16)
        plt.xlabel('Number of Bets', fontsize=12)
        plt.ylabel('Bankroll ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add annotation with key statistics
        text = (
            f"Win Rate: {sim_results['win_rate']:.2%}\n"
            f"Avg Odds: {sim_results['avg_odds']:.2f}\n"
            f"Staking: {sim_results['stake_method']} ({sim_results['stake_pct']:.1%})\n"
            f"Final Bankroll (Median): ${sim_results['final_bankroll_median']:.2f}\n"
            f"Prob. of 50%+ Loss: {sim_results['probability_of_ruin']:.2%}\n"
            f"Prob. of 100%+ Gain: {sim_results['probability_of_target_return']:.2%}"
        )
        
        plt.annotate(text, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                    fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show if requested
        if show_figure:
            plt.show()
        else:
            plt.close()

class StrategyComparator:
    """
    Compare multiple betting strategies across various performance metrics.
    
    This class provides tools for comparing and evaluating multiple betting
    strategies to determine which perform best under different conditions.
    """
    
    def __init__(self, strategies_data: Dict[str, Union[List[Dict[str, Any]], pd.DataFrame]]):
        """
        Initialize the StrategyComparator.
        
        Args:
            strategies_data: Dictionary mapping strategy names to betting data
        """
        self.strategies = {}
        self.metrics = {}
        
        for name, data in strategies_data.items():
            self.add_strategy(name, data)
    
    def add_strategy(self, name: str, data: Union[List[Dict[str, Any]], pd.DataFrame]) -> None:
        """
        Add a strategy to the comparison.
        
        Args:
            name: Strategy name
            data: Strategy betting data
        """
        if isinstance(data, list):
            self.strategies[name] = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.strategies[name] = data.copy()
        else:
            raise ValueError("Strategy data must be a list of dictionaries or a pandas DataFrame")
            
        # Ensure required columns exist
        required_cols = ['stake']
        if not all(col in self.strategies[name].columns for col in required_cols):
            raise ValueError(f"Strategy data must contain columns: {required_cols}")
            
        # Calculate metrics
        self._calculate_metrics(name)
    
    def _calculate_metrics(self, strategy_name: str) -> None:
        """
        Calculate performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy to calculate metrics for
        """
        data = self.strategies[strategy_name]
        
        metrics = {}
        
        # Basic metrics
        metrics['total_bets'] = len(data)
        metrics['total_stake'] = data['stake'].sum()
        
        # Calculate win rate if outcome is available
        if 'outcome' in data.columns:
            metrics['wins'] = (data['outcome'] == 'win').sum()
            metrics['losses'] = (data['outcome'] == 'loss').sum()
            metrics['pushes'] = (data['outcome'] == 'push').sum()
            
            if metrics['total_bets'] > 0:
                metrics['win_rate'] = metrics['wins'] / metrics['total_bets']
            else:
                metrics['win_rate'] = 0
        
        # Calculate profit and ROI if available
        if 'profit' in data.columns:
            metrics['total_profit'] = data['profit'].sum()
            
            if metrics['total_stake'] > 0:
                metrics['roi'] = metrics['total_profit'] / metrics['total_stake']
            else:
                metrics['roi'] = 0
        
        # Calculate average odds if available
        if 'odds' in data.columns:
            metrics['avg_odds'] = data['odds'].mean()
        
        # Calculate standard deviation of returns
        if 'profit' in data.columns:
            metrics['profit_std'] = data['profit'].std()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            if metrics['profit_std'] > 0:
                metrics['sharpe_ratio'] = data['profit'].mean() / metrics['profit_std']
            else:
                metrics['sharpe_ratio'] = float('inf') if data['profit'].mean() > 0 else 0
                
        # Store metrics
        self.metrics[strategy_name] = metrics
    
    def compare_basic_metrics(self) -> pd.DataFrame:
        """
        Compare basic performance metrics across strategies.
        
        Returns:
            pd.DataFrame: Comparison of basic metrics
        """
        if not self.metrics:
            return pd.DataFrame()
            
        # Determine common metrics
        common_metrics = set.intersection(*[set(metrics.keys()) for metrics in self.metrics.values()])
        
        # Create DataFrame
        comparison = {}
        
        for metric in common_metrics:
            comparison[metric] = {name: metrics[metric] for name, metrics in self.metrics.items()}
            
        return pd.DataFrame(comparison)
    
    def compare_strategies(self, 
                         metrics: List[str] = None, 
                         sort_by: str = 'roi') -> pd.DataFrame:
        """
        Compare strategies across selected metrics.
        
        Args:
            metrics: List of metrics to compare (default is key performance indicators)
            sort_by: Metric to sort by
            
        Returns:
            pd.DataFrame: Comparison DataFrame
        """
        if not self.metrics:
            return pd.DataFrame()
            
        # Default metrics if not specified
        if metrics is None:
            metrics = ['total_bets', 'win_rate', 'roi', 'total_profit', 'sharpe_ratio']
            
        # Filter to metrics available for all strategies
        common_metrics = set.intersection(*[set(metrics_dict.keys()) for metrics_dict in self.metrics.values()])
        available_metrics = [m for m in metrics if m in common_metrics]
        
        if not available_metrics:
            logger.warning("No common metrics found across strategies")
            return pd.DataFrame()
            
        # Create DataFrame
        data = {
            'strategy': list(self.metrics.keys())
        }
        
        for metric in available_metrics:
            data[metric] = [self.metrics[strategy][metric] for strategy in self.metrics.keys()]
            
        df = pd.DataFrame(data)
        
        # Sort by specified metric if available
        if sort_by in available_metrics:
            df = df.sort_values(sort_by, ascending=False)
            
        return df
    
    def calculate_statistical_significance(self, 
                                         metric: str = 'profit', 
                                         alpha: float = 0.05) -> pd.DataFrame:
        """
        Calculate statistical significance of differences between strategies.
        
        Args:
            metric: Metric to compare (e.g., 'profit')
            alpha: Significance level
            
        Returns:
            pd.DataFrame: Matrix of p-values for strategy comparisons
        """
        if not self.strategies or metric not in next(iter(self.strategies.values())).columns:
            logger.warning(f"Metric '{metric}' not available for statistical testing")
            return pd.DataFrame()
            
        strategy_names = list(self.strategies.keys())
        p_values = np.zeros((len(strategy_names), len(strategy_names)))
        
        # Calculate p-values for each pair of strategies
        for i, strategy1 in enumerate(strategy_names):
            data1 = self.strategies[strategy1][metric]
            
            for j, strategy2 in enumerate(strategy_names):
                if i == j:
                    p_values[i, j] = 1.0  # Same strategy
                    continue
                    
                data2 = self.strategies[strategy2][metric]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                p_values[i, j] = p_value
        
        # Create DataFrame
        p_value_df = pd.DataFrame(p_values, index=strategy_names, columns=strategy_names)
        
        # Mark significant differences
        significance_df = p_value_df.applymap(lambda p: 'Significant' if p < alpha else 'Not Significant')
        
        return significance_df
    
    def plot_performance_comparison(self,
                                 metric: str = 'roi',
                                 save_path: Optional[str] = None,
                                 show_figure: bool = True) -> None:
        """
        Plot performance comparison between strategies.
        
        Args:
            metric: Metric to compare
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        if not self.metrics:
            logger.warning("No strategies to compare")
            return
            
        # Check if metric is available for all strategies
        if not all(metric in metrics for metrics in self.metrics.values()):
            logger.warning(f"Metric '{metric}' not available for all strategies")
            return
            
        # Get data
        strategies = list(self.metrics.keys())
        values = [self.metrics[strategy][metric] for strategy in strategies]
        
        # Create DataFrame
        df = pd.DataFrame({'Strategy': strategies, 'Value': values})
        df = df.sort_values('Value', ascending=False)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.set_style('whitegrid')
        
        # Create bar plot
        ax = sns.barplot(x='Strategy', y='Value', data=df)
        
        # Format metric name for display
        metric_display = ' '.join(word.capitalize() for word in metric.split('_'))
        
        # Format plot
        plt.title(f'Strategy Comparison: {metric_display}', fontsize=16)
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel(metric_display, fontsize=12)
        plt.xticks(rotation=45)
        
        # Format y-axis based on metric
        from matplotlib.ticker import FuncFormatter
        
        if metric in ['roi', 'win_rate']:
            def percentage_fmt(x, pos):
                return f"{x*100:.1f}%"
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_fmt))
        elif metric in ['total_profit', 'total_stake']:
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
    
    def plot_cumulative_returns(self,
                             save_path: Optional[str] = None,
                             show_figure: bool = True) -> None:
        """
        Plot cumulative returns for each strategy.
        
        Args:
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        if not self.strategies:
            logger.warning("No strategies to compare")
            return
            
        # Check for required columns
        if not all('profit' in df.columns for df in self.strategies.values()):
            logger.warning("Profit data required for all strategies")
            return
            
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.set_style('whitegrid')
        
        # Plot cumulative returns for each strategy
        for name, data in self.strategies.items():
            # Sort by date if available
            if 'placed_date' in data.columns:
                sorted_data = data.sort_values('placed_date')
            elif 'date' in data.columns:
                sorted_data = data.sort_values('date')
            else:
                sorted_data = data
                
            # Calculate cumulative profits
            cumulative = sorted_data['profit'].cumsum()
            
            # Plot
            if 'placed_date' in sorted_data.columns:
                plt.plot(sorted_data['placed_date'], cumulative, label=name)
            elif 'date' in sorted_data.columns:
                plt.plot(sorted_data['date'], cumulative, label=name)
            else:
                plt.plot(range(len(cumulative)), cumulative, label=name)
                
        # Format plot
        plt.title('Cumulative Returns by Strategy', fontsize=16)
        
        if 'placed_date' in next(iter(self.strategies.values())).columns or 'date' in next(iter(self.strategies.values())).columns:
            plt.xlabel('Date', fontsize=12)
            plt.gcf().autofmt_xdate()  # Rotate date labels
        else:
            plt.xlabel('Bet Number', fontsize=12)
            
        plt.ylabel('Cumulative Profit ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add reference line at 0
        plt.axhline(y=0, linestyle='--', alpha=0.7, color='red')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show if requested
        if show_figure:
            plt.show()
        else:
            plt.close()
    
    def plot_win_rate_by_odds(self,
                           bins: int = 5,
                           save_path: Optional[str] = None,
                           show_figure: bool = True) -> None:
        """
        Plot win rate by odds range for each strategy.
        
        Args:
            bins: Number of odds bins
            save_path: Optional path to save the figure
            show_figure: Whether to display the figure
        """
        if not self.strategies:
            logger.warning("No strategies to compare")
            return
            
        # Check for required columns
        required_cols = ['odds', 'outcome']
        if not all(all(col in df.columns for col in required_cols) for df in self.strategies.values()):
            logger.warning(f"All strategies must have columns: {required_cols}")
            return
            
        # Create plot
        plt.figure(figsize=(12, 6))
        sns.set_style('whitegrid')
        
        # Define colors for each strategy
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.strategies)))
        
        # Process each strategy
        for i, (name, data) in enumerate(self.strategies.items()):
            # Create odds bins
            odds_min = data['odds'].min()
            odds_max = min(10.0, data['odds'].max())  # Cap at 10.0 for better visualization
            bin_edges = np.linspace(odds_min, odds_max, bins + 1)
            
            # Create bin labels
            bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(bins)]
            
            # Assign bins
            data['odds_bin'] = pd.cut(
                data['odds'], 
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True
            )
            
            # Calculate win rate by bin
            win_rates = data.groupby('odds_bin')['outcome'].apply(
                lambda x: (x == 'win').mean()
            ).reset_index()
            
            win_rates.columns = ['odds_bin', 'win_rate']
            
            # Convert odds bin to numeric for plotting
            win_rates['bin_center'] = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(bins)]
            
            # Add slight offset for better visualization
            offset = (i - len(self.strategies) / 2) * 0.1
            
            # Plot win rates
            plt.bar(
                win_rates['bin_center'] + offset, 
                win_rates['win_rate'],
                width=0.1,
                label=name,
                color=colors[i],
                alpha=0.7
            )
            
            # Add expected win rate based on fair odds
            expected_rates = [1/((bin_edges[i] + bin_edges[i+1]) / 2) for i in range(bins)]
            plt.plot(win_rates['bin_center'], expected_rates, 'o--', color=colors[i], alpha=0.5)
        
        # Format plot
        plt.title('Win Rate by Odds Range', fontsize=16)
        plt.xlabel('Odds', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format y-axis as percentage
        from matplotlib.ticker import FuncFormatter
        def percentage_fmt(x, pos):
            return f"{x*100:.0f}%"
            
        plt.gca().yaxis.set_major_formatter(FuncFormatter(percentage_fmt))
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        # Show if requested
        if show_figure:
            plt.show()
        else:
            plt.close()
    
    def generate_comparison_report(self,
                               metrics: List[str] = None,
                               include_plots: bool = True,
                               output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Args:
            metrics: List of metrics to include
            include_plots: Whether to include plots
            output_path: Path to save report
            
        Returns:
            Dict[str, Any]: Report data
        """
        if not self.strategies:
            logger.warning("No strategies to compare")
            return {}
            
        # Default metrics if not specified
        if metrics is None:
            metrics = ['total_bets', 'win_rate', 'roi', 'total_profit', 'sharpe_ratio']
            
        # Generate comparison data
        comparison_df = self.compare_strategies(metrics)
        
        # Get statistical significance
        sig_df = None
        if 'profit' in next(iter(self.strategies.values())).columns:
            sig_df = self.calculate_statistical_significance('profit')
            
        # Create report structure
        report = {
            'title': 'Strategy Comparison Report',
            'generated_at': datetime.now().isoformat(),
            'strategies_compared': list(self.strategies.keys()),
            'metrics_compared': metrics,
            'comparison_data': comparison_df.to_dict(orient='records'),
            'statistical_significance': sig_df.to_dict() if sig_df is not None else None
        }
        
        # Generate plots if requested
        if include_plots and output_path:
            plots_dir = Path(output_path).parent / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Performance comparison plots
            for metric in metrics:
                if all(metric in m for m in self.metrics.values()):
                    metric_path = plots_dir / f'comparison_{metric}.png'
                    self.plot_performance_comparison(
                        metric=metric,
                        save_path=str(metric_path),
                        show_figure=False
                    )
                    
            # Cumulative returns plot
            returns_path = plots_dir / 'cumulative_returns.png'
            self.plot_cumulative_returns(
                save_path=str(returns_path),
                show_figure=False
            )
            
            # Win rate by odds plot
            win_rate_path = plots_dir / 'win_rate_by_odds.png'
            self.plot_win_rate_by_odds(
                save_path=str(win_rate_path),
                show_figure=False
            )
            
            report['plots'] = {
                'cumulative_returns': str(returns_path),
                'win_rate_by_odds': str(win_rate_path)
            }
            
            for metric in metrics:
                if all(metric in m for m in self.metrics.values()):
                    report['plots'][f'comparison_{metric}'] = str(plots_dir / f'comparison_{metric}.png')
                    
        # Save report if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
                logger.info(f"Comparison report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")
                
        return report 