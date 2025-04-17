"""
Betting logger for tracking and analyzing betting activities.

This module provides functionality for logging, tracking, and analyzing
betting activities across different strategies and markets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import logging
import json
import os
import uuid
from enum import Enum
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class BetStatus(Enum):
    """Enumeration of possible bet statuses."""
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    VOID = "void"
    PARTIALLY_WON = "partially_won"
    PARTIALLY_LOST = "partially_lost"
    CASHOUT = "cashout"


class BetType(Enum):
    """Enumeration of bet types."""
    HOME = "home"
    AWAY = "away"
    DRAW = "draw"
    OVER = "over"
    UNDER = "under"
    BTTS_YES = "btts_yes"
    BTTS_NO = "btts_no"
    HANDICAP = "handicap"
    CORRECT_SCORE = "correct_score"
    OTHER = "other"


class BetLogger:
    """
    Logger for tracking and analyzing betting activities.
    
    This class provides methods for logging bets, updating their status,
    calculating performance metrics, and generating reports.
    """
    
    def __init__(self, 
                log_dir: str = "./bet_logs",
                starting_bankroll: float = 1000.0,
                currency: str = "USD"):
        """
        Initialize the bet logger.
        
        Args:
            log_dir: Directory to store log files
            starting_bankroll: Initial bankroll amount
            currency: Currency symbol for reports
        """
        self.log_dir = Path(log_dir)
        self.starting_bankroll = starting_bankroll
        self.currency = currency
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DataFrames for tracking bets and performance
        self.bets_df = pd.DataFrame(columns=[
            "bet_id", "timestamp", "match_id", "home_team", "away_team", 
            "league", "market", "selection", "odds", "stake", 
            "expected_value", "predicted_probability", "true_probability",
            "status", "profit_loss", "strategy", "confidence", 
            "settlement_date", "bet_type", "bookmaker"
        ])
        
        self.performance_df = pd.DataFrame(columns=[
            "date", "bankroll", "daily_pl", "bets_count", 
            "win_count", "loss_count", "void_count"
        ])
        
        # Load existing logs if available
        self._load_logs()
    
    def _load_logs(self) -> None:
        """Load existing log files if they exist."""
        bets_file = self.log_dir / "bets_log.csv"
        performance_file = self.log_dir / "performance_log.csv"
        
        try:
            if bets_file.exists():
                self.bets_df = pd.read_csv(bets_file)
                
                # Convert string timestamps to datetime objects
                if "timestamp" in self.bets_df.columns:
                    self.bets_df["timestamp"] = pd.to_datetime(self.bets_df["timestamp"])
                if "settlement_date" in self.bets_df.columns:
                    self.bets_df["settlement_date"] = pd.to_datetime(self.bets_df["settlement_date"])
            
            if performance_file.exists():
                self.performance_df = pd.read_csv(performance_file)
                
                # Convert string dates to datetime objects
                if "date" in self.performance_df.columns:
                    self.performance_df["date"] = pd.to_datetime(self.performance_df["date"])
                
        except Exception as e:
            logger.error(f"Error loading log files: {str(e)}")
            logger.info("Creating new log files")
    
    def _save_logs(self) -> None:
        """Save logs to disk."""
        try:
            # Save bets log
            bets_file = self.log_dir / "bets_log.csv"
            self.bets_df.to_csv(bets_file, index=False)
            
            # Save performance log
            performance_file = self.log_dir / "performance_log.csv"
            self.performance_df.to_csv(performance_file, index=False)
            
        except Exception as e:
            logger.error(f"Error saving log files: {str(e)}")
    
    def log_bet(self, 
               match_id: str,
               home_team: str,
               away_team: str,
               league: str,
               market: str,
               selection: str,
               odds: float,
               stake: float,
               expected_value: Optional[float] = None,
               predicted_probability: Optional[float] = None,
               true_probability: Optional[float] = None,
               status: BetStatus = BetStatus.PENDING,
               profit_loss: float = 0.0,
               strategy: str = "manual",
               confidence: Optional[float] = None,
               bet_type: BetType = BetType.OTHER,
               bookmaker: str = "unknown") -> str:
        """
        Log a new bet.
        
        Args:
            match_id: Unique identifier for the match
            home_team: Home team name
            away_team: Away team name
            league: League name
            market: Betting market (e.g., "1X2", "over_under_2.5")
            selection: Selected outcome
            odds: Betting odds (decimal format)
            stake: Amount staked
            expected_value: Expected value of the bet
            predicted_probability: Predicted probability of the outcome
            true_probability: True probability of the outcome (if known)
            status: Status of the bet (default: PENDING)
            profit_loss: Profit or loss from the bet (0 for pending bets)
            strategy: Strategy used to place the bet
            confidence: Confidence level in the bet (0-1)
            bet_type: Type of bet
            bookmaker: Bookmaker where the bet was placed
            
        Returns:
            Unique bet ID
        """
        # Generate a unique bet ID
        bet_id = str(uuid.uuid4())
        
        # Create a new bet entry
        new_bet = {
            "bet_id": bet_id,
            "timestamp": datetime.now(),
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "league": league,
            "market": market,
            "selection": selection,
            "odds": odds,
            "stake": stake,
            "expected_value": expected_value,
            "predicted_probability": predicted_probability,
            "true_probability": true_probability,
            "status": status.value,
            "profit_loss": profit_loss,
            "strategy": strategy,
            "confidence": confidence,
            "settlement_date": None,
            "bet_type": bet_type.value,
            "bookmaker": bookmaker
        }
        
        # Add to DataFrame
        self.bets_df = pd.concat([self.bets_df, pd.DataFrame([new_bet])], ignore_index=True)
        
        # Save logs
        self._save_logs()
        
        return bet_id
    
    def bulk_log_bets(self, bets: List[Dict[str, Any]]) -> List[str]:
        """
        Log multiple bets at once.
        
        Args:
            bets: List of bet dictionaries with the same fields as log_bet
            
        Returns:
            List of unique bet IDs
        """
        bet_ids = []
        
        for bet in bets:
            # Generate a unique bet ID for each bet
            bet_id = str(uuid.uuid4())
            bet["bet_id"] = bet_id
            bet_ids.append(bet_id)
            
            # Set timestamp if not provided
            if "timestamp" not in bet:
                bet["timestamp"] = datetime.now()
            
            # Convert status to string value if it's an enum
            if "status" in bet and isinstance(bet["status"], BetStatus):
                bet["status"] = bet["status"].value
            
            # Convert bet_type to string value if it's an enum
            if "bet_type" in bet and isinstance(bet["bet_type"], BetType):
                bet["bet_type"] = bet["bet_type"].value
        
        # Add to DataFrame
        new_bets_df = pd.DataFrame(bets)
        self.bets_df = pd.concat([self.bets_df, new_bets_df], ignore_index=True)
        
        # Save logs
        self._save_logs()
        
        return bet_ids
    
    def update_bet_status(self, 
                        bet_id: str,
                        status: BetStatus,
                        profit_loss: Optional[float] = None,
                        settlement_date: Optional[datetime] = None) -> bool:
        """
        Update the status of a bet.
        
        Args:
            bet_id: Unique identifier for the bet
            status: New status of the bet
            profit_loss: Profit or loss from the bet
            settlement_date: Date when the bet was settled
            
        Returns:
            True if the bet was successfully updated, False otherwise
        """
        # Find the bet in the DataFrame
        bet_idx = self.bets_df[self.bets_df["bet_id"] == bet_id].index
        
        if len(bet_idx) == 0:
            logger.warning(f"Bet ID {bet_id} not found")
            return False
        
        # Update the status
        self.bets_df.loc[bet_idx, "status"] = status.value
        
        # Update profit/loss if provided
        if profit_loss is not None:
            self.bets_df.loc[bet_idx, "profit_loss"] = profit_loss
        
        # Update settlement date
        if settlement_date is None:
            settlement_date = datetime.now()
        
        self.bets_df.loc[bet_idx, "settlement_date"] = settlement_date
        
        # Update performance metrics
        self._update_performance()
        
        # Save logs
        self._save_logs()
        
        return True
    
    def bulk_update_bet_status(self, 
                              updates: List[Dict[str, Any]]) -> int:
        """
        Update the status of multiple bets at once.
        
        Args:
            updates: List of dictionaries with bet_id, status, and optionally
                    profit_loss and settlement_date
            
        Returns:
            Number of bets successfully updated
        """
        success_count = 0
        
        for update in updates:
            bet_id = update.get("bet_id")
            status = update.get("status")
            
            if not bet_id or not status:
                continue
            
            # Convert status to enum if it's a string
            if isinstance(status, str):
                try:
                    status = BetStatus(status)
                except ValueError:
                    logger.warning(f"Invalid status value: {status}")
                    continue
            
            profit_loss = update.get("profit_loss")
            settlement_date = update.get("settlement_date", datetime.now())
            
            if self.update_bet_status(bet_id, status, profit_loss, settlement_date):
                success_count += 1
        
        return success_count
    
    def _update_performance(self) -> None:
        """Update performance metrics based on settled bets."""
        # Get today's date
        today = datetime.now().date()
        
        # Get all settled bets
        settled_bets = self.bets_df[self.bets_df["status"].isin([
            BetStatus.WON.value, BetStatus.LOST.value, BetStatus.VOID.value,
            BetStatus.PARTIALLY_WON.value, BetStatus.PARTIALLY_LOST.value,
            BetStatus.CASHOUT.value
        ])]
        
        # Group by settlement date and calculate metrics
        if not settled_bets.empty and "settlement_date" in settled_bets.columns:
            # Convert settlement_date to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(settled_bets["settlement_date"]):
                settled_bets["settlement_date"] = pd.to_datetime(settled_bets["settlement_date"])
            
            # Extract date from datetime
            settled_bets["settlement_date_only"] = settled_bets["settlement_date"].dt.date
            
            # Group by date
            daily_metrics = settled_bets.groupby("settlement_date_only").agg({
                "bet_id": "count",
                "profit_loss": "sum",
                "status": lambda x: (x == BetStatus.WON.value).sum(),
            }).rename(columns={
                "bet_id": "bets_count",
                "status": "win_count"
            })
            
            daily_metrics["loss_count"] = daily_metrics["bets_count"] - daily_metrics["win_count"]
            daily_metrics["void_count"] = settled_bets.groupby("settlement_date_only")["status"].apply(
                lambda x: (x == BetStatus.VOID.value).sum()
            )
            
            # Calculate cumulative bankroll
            daily_metrics["daily_pl"] = daily_metrics["profit_loss"]
            daily_metrics["bankroll"] = self.starting_bankroll + daily_metrics["profit_loss"].cumsum()
            
            # Convert to DataFrame with date as a column
            daily_metrics = daily_metrics.reset_index().rename(columns={"settlement_date_only": "date"})
            
            # Update performance DataFrame
            for _, row in daily_metrics.iterrows():
                date = row["date"]
                
                # Check if this date already exists in performance_df
                date_idx = self.performance_df[self.performance_df["date"] == pd.Timestamp(date)].index
                
                if len(date_idx) > 0:
                    # Update existing entry
                    self.performance_df.loc[date_idx] = row
                else:
                    # Add new entry
                    self.performance_df = pd.concat(
                        [self.performance_df, pd.DataFrame([row])], 
                        ignore_index=True
                    )
            
            # Sort by date
            self.performance_df = self.performance_df.sort_values("date").reset_index(drop=True)
    
    def get_pending_bets(self) -> pd.DataFrame:
        """
        Get all pending bets.
        
        Returns:
            DataFrame with pending bets
        """
        return self.bets_df[self.bets_df["status"] == BetStatus.PENDING.value]
    
    def get_settled_bets(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        strategy: Optional[str] = None,
                        market: Optional[str] = None,
                        league: Optional[str] = None) -> pd.DataFrame:
        """
        Get all settled bets with optional filters.
        
        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            strategy: Optional strategy name for filtering
            market: Optional market for filtering
            league: Optional league for filtering
            
        Returns:
            DataFrame with settled bets
        """
        # Get all settled bets
        settled_bets = self.bets_df[self.bets_df["status"] != BetStatus.PENDING.value].copy()
        
        # Apply filters
        if start_date:
            settled_bets = settled_bets[settled_bets["settlement_date"] >= start_date]
        
        if end_date:
            settled_bets = settled_bets[settled_bets["settlement_date"] <= end_date]
        
        if strategy:
            settled_bets = settled_bets[settled_bets["strategy"] == strategy]
        
        if market:
            settled_bets = settled_bets[settled_bets["market"] == market]
        
        if league:
            settled_bets = settled_bets[settled_bets["league"] == league]
        
        return settled_bets
    
    def get_performance_summary(self,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get a summary of betting performance.
        
        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            Dictionary with performance metrics
        """
        # Filter performance data
        performance = self.performance_df.copy()
        
        if start_date:
            performance = performance[performance["date"] >= pd.Timestamp(start_date)]
        
        if end_date:
            performance = performance[performance["date"] <= pd.Timestamp(end_date)]
        
        if performance.empty:
            return {
                "total_bets": 0,
                "total_stake": 0,
                "total_profit_loss": 0,
                "roi": 0,
                "win_rate": 0,
                "current_bankroll": self.starting_bankroll,
                "starting_bankroll": self.starting_bankroll,
                "profit_factor": 0
            }
        
        # Filter bets data
        bets = self.get_settled_bets(start_date, end_date)
        
        # Calculate metrics
        total_bets = bets.shape[0]
        total_stake = bets["stake"].sum()
        total_profit_loss = bets["profit_loss"].sum()
        roi = (total_profit_loss / total_stake * 100) if total_stake > 0 else 0
        
        win_count = (bets["status"] == BetStatus.WON.value).sum()
        win_rate = (win_count / total_bets * 100) if total_bets > 0 else 0
        
        current_bankroll = performance["bankroll"].iloc[-1]
        
        # Calculate profit factor (sum of winnings / sum of losses)
        winning_bets = bets[bets["profit_loss"] > 0]
        losing_bets = bets[bets["profit_loss"] < 0]
        
        total_winnings = winning_bets["profit_loss"].sum()
        total_losses = abs(losing_bets["profit_loss"].sum())
        
        profit_factor = (total_winnings / total_losses) if total_losses > 0 else 0
        
        return {
            "total_bets": total_bets,
            "total_stake": total_stake,
            "total_profit_loss": total_profit_loss,
            "roi": roi,
            "win_rate": win_rate,
            "current_bankroll": current_bankroll,
            "starting_bankroll": self.starting_bankroll,
            "profit_factor": profit_factor
        }
    
    def get_strategy_performance(self) -> pd.DataFrame:
        """
        Get performance metrics by strategy.
        
        Returns:
            DataFrame with performance metrics for each strategy
        """
        # Get all settled bets
        settled_bets = self.bets_df[self.bets_df["status"] != BetStatus.PENDING.value]
        
        if settled_bets.empty:
            return pd.DataFrame(columns=[
                "strategy", "total_bets", "win_count", "loss_count", 
                "total_stake", "total_profit_loss", "roi", "win_rate"
            ])
        
        # Group by strategy
        strategy_performance = settled_bets.groupby("strategy").agg({
            "bet_id": "count",
            "stake": "sum",
            "profit_loss": "sum",
            "status": lambda x: (x == BetStatus.WON.value).sum()
        }).rename(columns={
            "bet_id": "total_bets",
            "status": "win_count"
        })
        
        # Calculate additional metrics
        strategy_performance["loss_count"] = (
            strategy_performance["total_bets"] - strategy_performance["win_count"]
        )
        
        strategy_performance["roi"] = (
            strategy_performance["profit_loss"] / strategy_performance["stake"] * 100
        )
        
        strategy_performance["win_rate"] = (
            strategy_performance["win_count"] / strategy_performance["total_bets"] * 100
        )
        
        # Reset index to make strategy a column
        return strategy_performance.reset_index()
    
    def get_market_performance(self) -> pd.DataFrame:
        """
        Get performance metrics by market.
        
        Returns:
            DataFrame with performance metrics for each market
        """
        # Get all settled bets
        settled_bets = self.bets_df[self.bets_df["status"] != BetStatus.PENDING.value]
        
        if settled_bets.empty:
            return pd.DataFrame(columns=[
                "market", "total_bets", "win_count", "loss_count", 
                "total_stake", "total_profit_loss", "roi", "win_rate"
            ])
        
        # Group by market
        market_performance = settled_bets.groupby("market").agg({
            "bet_id": "count",
            "stake": "sum",
            "profit_loss": "sum",
            "status": lambda x: (x == BetStatus.WON.value).sum()
        }).rename(columns={
            "bet_id": "total_bets",
            "status": "win_count"
        })
        
        # Calculate additional metrics
        market_performance["loss_count"] = (
            market_performance["total_bets"] - market_performance["win_count"]
        )
        
        market_performance["roi"] = (
            market_performance["profit_loss"] / market_performance["stake"] * 100
        )
        
        market_performance["win_rate"] = (
            market_performance["win_count"] / market_performance["total_bets"] * 100
        )
        
        # Reset index to make market a column
        return market_performance.reset_index()
    
    def get_league_performance(self) -> pd.DataFrame:
        """
        Get performance metrics by league.
        
        Returns:
            DataFrame with performance metrics for each league
        """
        # Get all settled bets
        settled_bets = self.bets_df[self.bets_df["status"] != BetStatus.PENDING.value]
        
        if settled_bets.empty:
            return pd.DataFrame(columns=[
                "league", "total_bets", "win_count", "loss_count", 
                "total_stake", "total_profit_loss", "roi", "win_rate"
            ])
        
        # Group by league
        league_performance = settled_bets.groupby("league").agg({
            "bet_id": "count",
            "stake": "sum",
            "profit_loss": "sum",
            "status": lambda x: (x == BetStatus.WON.value).sum()
        }).rename(columns={
            "bet_id": "total_bets",
            "status": "win_count"
        })
        
        # Calculate additional metrics
        league_performance["loss_count"] = (
            league_performance["total_bets"] - league_performance["win_count"]
        )
        
        league_performance["roi"] = (
            league_performance["profit_loss"] / league_performance["stake"] * 100
        )
        
        league_performance["win_rate"] = (
            league_performance["win_count"] / league_performance["total_bets"] * 100
        )
        
        # Reset index to make league a column
        return league_performance.reset_index()
    
    def plot_bankroll_history(self, 
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             save_path: Optional[str] = None) -> None:
        """
        Plot bankroll history over time.
        
        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            save_path: Optional path to save the plot
        """
        # Filter performance data
        performance = self.performance_df.copy()
        
        if start_date:
            performance = performance[performance["date"] >= pd.Timestamp(start_date)]
        
        if end_date:
            performance = performance[performance["date"] <= pd.Timestamp(end_date)]
        
        if performance.empty:
            logger.warning("No performance data available for plotting")
            return
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(performance["date"], performance["bankroll"], marker='o')
        plt.title("Bankroll History")
        plt.xlabel("Date")
        plt.ylabel(f"Bankroll ({self.currency})")
        plt.grid(True)
        
        # Add reference line for starting bankroll
        plt.axhline(y=self.starting_bankroll, color='r', linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        # Add profit/loss labels
        for i, row in performance.iterrows():
            if row["daily_pl"] != 0:
                color = 'green' if row["daily_pl"] > 0 else 'red'
                plt.annotate(f"{row['daily_pl']:.2f}",
                            (row["date"], row["bankroll"]),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            color=color,
                            fontsize=8)
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def export_to_csv(self, 
                     file_path: str,
                     data_type: str = "bets",
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> bool:
        """
        Export data to CSV file.
        
        Args:
            file_path: Path to save the CSV file
            data_type: Type of data to export ("bets", "performance", "strategy", 
                      "market", or "league")
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            if data_type == "bets":
                data = self.get_settled_bets(start_date, end_date)
            elif data_type == "performance":
                data = self.performance_df
                
                if start_date:
                    data = data[data["date"] >= pd.Timestamp(start_date)]
                
                if end_date:
                    data = data[data["date"] <= pd.Timestamp(end_date)]
            elif data_type == "strategy":
                data = self.get_strategy_performance()
            elif data_type == "market":
                data = self.get_market_performance()
            elif data_type == "league":
                data = self.get_league_performance()
            else:
                logger.error(f"Invalid data type: {data_type}")
                return False
            
            # Save to CSV
            data.to_csv(file_path, index=False)
            return True
        
        except Exception as e:
            logger.error(f"Error exporting data to CSV: {str(e)}")
            return False
    
    def get_bet_results_by_confidence(self, 
                                    bin_width: float = 0.1) -> pd.DataFrame:
        """
        Analyze bet results grouped by confidence level.
        
        Args:
            bin_width: Width of confidence bins (default 0.1 for 10% bins)
            
        Returns:
            DataFrame with performance metrics by confidence level
        """
        # Get settled bets with confidence values
        settled_bets = self.bets_df[
            (self.bets_df["status"] != BetStatus.PENDING.value) & 
            (~self.bets_df["confidence"].isna())
        ]
        
        if settled_bets.empty:
            return pd.DataFrame(columns=[
                "confidence_bin", "total_bets", "win_count", "roi", "win_rate"
            ])
        
        # Create confidence bins
        bins = np.arange(0, 1.01, bin_width)
        labels = [f"{round(low, 1)}-{round(low+bin_width, 1)}" for low in bins[:-1]]
        
        settled_bets["confidence_bin"] = pd.cut(
            settled_bets["confidence"], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        
        # Group by confidence bin
        confidence_performance = settled_bets.groupby("confidence_bin").agg({
            "bet_id": "count",
            "stake": "sum",
            "profit_loss": "sum",
            "status": lambda x: (x == BetStatus.WON.value).sum()
        }).rename(columns={
            "bet_id": "total_bets",
            "status": "win_count"
        })
        
        # Calculate additional metrics
        confidence_performance["roi"] = (
            confidence_performance["profit_loss"] / confidence_performance["stake"] * 100
        )
        
        confidence_performance["win_rate"] = (
            confidence_performance["win_count"] / confidence_performance["total_bets"] * 100
        )
        
        # Reset index to make confidence_bin a column
        return confidence_performance.reset_index()
    
    def get_value_analysis(self) -> pd.DataFrame:
        """
        Analyze the relationship between expected value and actual results.
        
        Returns:
            DataFrame with performance metrics by EV bin
        """
        # Get settled bets with expected value
        settled_bets = self.bets_df[
            (self.bets_df["status"] != BetStatus.PENDING.value) & 
            (~self.bets_df["expected_value"].isna())
        ]
        
        if settled_bets.empty:
            return pd.DataFrame(columns=[
                "ev_bin", "total_bets", "win_count", "roi", "win_rate", "avg_expected_value", "actual_ev"
            ])
        
        # Create EV bins
        bins = [-float('inf'), 0, 0.05, 0.1, 0.15, 0.2, 0.3, float('inf')]
        labels = ["Negative", "0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
        
        settled_bets["ev_bin"] = pd.cut(
            settled_bets["expected_value"],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Group by EV bin
        ev_performance = settled_bets.groupby("ev_bin").agg({
            "bet_id": "count",
            "stake": "sum",
            "profit_loss": "sum",
            "expected_value": "mean",
            "status": lambda x: (x == BetStatus.WON.value).sum()
        }).rename(columns={
            "bet_id": "total_bets",
            "status": "win_count",
            "expected_value": "avg_expected_value"
        })
        
        # Calculate additional metrics
        ev_performance["roi"] = (
            ev_performance["profit_loss"] / ev_performance["stake"] * 100
        )
        
        ev_performance["win_rate"] = (
            ev_performance["win_count"] / ev_performance["total_bets"] * 100
        )
        
        # Calculate actual EV (profit/loss per unit stake)
        ev_performance["actual_ev"] = ev_performance["profit_loss"] / ev_performance["stake"]
        
        # Reset index to make ev_bin a column
        return ev_performance.reset_index()


class CSVBetLogger(BetLogger):
    """
    Implementation of BetLogger that stores data in CSV files.
    
    This is a simple implementation that extends the base BetLogger class
    with the same functionality but ensures data persistence via CSV files.
    """
    
    def __init__(self, 
                log_dir: str = "./bet_logs",
                starting_bankroll: float = 1000.0,
                currency: str = "USD"):
        """
        Initialize the CSV-based bet logger.
        
        Args:
            log_dir: Directory to store log files
            starting_bankroll: Initial bankroll amount
            currency: Currency symbol for reports
        """
        super().__init__(log_dir, starting_bankroll, currency)


class JSONBetLogger(BetLogger):
    """
    Implementation of BetLogger that stores data in JSON files.
    
    This implementation provides the same functionality as the base BetLogger
    but persists data in JSON format instead of CSV.
    """
    
    def __init__(self, 
                log_dir: str = "./bet_logs",
                starting_bankroll: float = 1000.0,
                currency: str = "USD"):
        """
        Initialize the JSON-based bet logger.
        
        Args:
            log_dir: Directory to store log files
            starting_bankroll: Initial bankroll amount
            currency: Currency symbol for reports
        """
        self.log_dir = Path(log_dir)
        self.starting_bankroll = starting_bankroll
        self.currency = currency
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DataFrames for tracking bets and performance
        self.bets_df = pd.DataFrame(columns=[
            "bet_id", "timestamp", "match_id", "home_team", "away_team", 
            "league", "market", "selection", "odds", "stake", 
            "expected_value", "predicted_probability", "true_probability",
            "status", "profit_loss", "strategy", "confidence", 
            "settlement_date", "bet_type", "bookmaker"
        ])
        
        self.performance_df = pd.DataFrame(columns=[
            "date", "bankroll", "daily_pl", "bets_count", 
            "win_count", "loss_count", "void_count"
        ])
        
        # Load existing logs if available
        self._load_logs()
    
    def _load_logs(self) -> None:
        """Load existing log files if they exist."""
        bets_file = self.log_dir / "bets_log.json"
        performance_file = self.log_dir / "performance_log.json"
        
        try:
            if bets_file.exists():
                with open(bets_file, 'r') as f:
                    bets_data = json.load(f)
                
                self.bets_df = pd.DataFrame(bets_data)
                
                # Convert string timestamps to datetime objects
                if "timestamp" in self.bets_df.columns:
                    self.bets_df["timestamp"] = pd.to_datetime(self.bets_df["timestamp"])
                if "settlement_date" in self.bets_df.columns:
                    self.bets_df["settlement_date"] = pd.to_datetime(self.bets_df["settlement_date"])
            
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    performance_data = json.load(f)
                
                self.performance_df = pd.DataFrame(performance_data)
                
                # Convert string dates to datetime objects
                if "date" in self.performance_df.columns:
                    self.performance_df["date"] = pd.to_datetime(self.performance_df["date"])
        
        except Exception as e:
            logger.error(f"Error loading JSON log files: {str(e)}")
            logger.info("Creating new log files")
    
    def _save_logs(self) -> None:
        """Save logs to disk in JSON format."""
        try:
            # Convert timestamps to strings for JSON serialization
            bets_df = self.bets_df.copy()
            if "timestamp" in bets_df.columns:
                bets_df["timestamp"] = bets_df["timestamp"].astype(str)
            if "settlement_date" in bets_df.columns:
                bets_df["settlement_date"] = bets_df["settlement_date"].astype(str)
            
            # Save bets log
            bets_file = self.log_dir / "bets_log.json"
            with open(bets_file, 'w') as f:
                json.dump(bets_df.to_dict(orient="records"), f, indent=2)
            
            # Convert dates to strings for JSON serialization
            performance_df = self.performance_df.copy()
            if "date" in performance_df.columns:
                performance_df["date"] = performance_df["date"].astype(str)
            
            # Save performance log
            performance_file = self.log_dir / "performance_log.json"
            with open(performance_file, 'w') as f:
                json.dump(performance_df.to_dict(orient="records"), f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving JSON log files: {str(e)}") 