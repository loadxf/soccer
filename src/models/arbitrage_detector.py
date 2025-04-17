"""
Arbitrage detector for finding profitable betting opportunities across bookmakers.

This module provides functionality to detect and evaluate arbitrage opportunities
by comparing odds from multiple bookmakers for the same event.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Set
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass, field
import math
from pathlib import Path

from src.models.odds_provider_connector import OddsProviderConnector

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """
    Dataclass for representing an arbitrage betting opportunity.
    
    This class contains all relevant information about an identified arbitrage
    opportunity, including the event details, bookmakers, odds, and expected profit.
    """
    match_id: str
    home_team: str
    away_team: str
    market_type: str
    bookmakers: Dict[str, str]  # Outcome -> bookmaker mapping
    odds: Dict[str, float]  # Outcome -> odds mapping
    implied_probabilities: Dict[str, float]
    total_implied_probability: float
    arbitrage_profit_percent: float
    optimal_stakes: Dict[str, float]
    total_stake: float
    expected_return: float
    date: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the opportunity to a dictionary."""
        return {
            'match_id': self.match_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'market_type': self.market_type,
            'bookmakers': self.bookmakers,
            'odds': self.odds,
            'implied_probabilities': self.implied_probabilities,
            'total_implied_probability': self.total_implied_probability,
            'arbitrage_profit_percent': self.arbitrage_profit_percent,
            'optimal_stakes': self.optimal_stakes,
            'total_stake': self.total_stake,
            'expected_return': self.expected_return,
            'date': self.date.isoformat() if self.date else None,
            'timestamp': self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArbitrageOpportunity':
        """Create an ArbitrageOpportunity from a dictionary."""
        # Convert timestamp string back to datetime
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert date string back to datetime
        if 'date' in data and data['date']:
            data['date'] = datetime.fromisoformat(data['date'])
        
        return cls(**data)


class BookmakerMarginCalculator:
    """
    Calculator for bookmaker margins and fair odds.
    
    This utility class provides methods to calculate bookmaker margins,
    determine fair odds without margin, and analyze the value offered
    by different bookmakers.
    """
    
    @staticmethod
    def calculate_margin(odds: Dict[str, float]) -> float:
        """
        Calculate the bookmaker's margin (overround) for a set of odds.
        
        Args:
            odds: Dictionary mapping outcomes to odds values
            
        Returns:
            Margin as a percentage (e.g., 5.0 for 5%)
        """
        total_implied_prob = sum(1 / odd for odd in odds.values())
        margin = (total_implied_prob - 1) * 100
        return margin
    
    @staticmethod
    def calculate_fair_odds(odds: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate fair odds by removing the bookmaker's margin.
        
        Args:
            odds: Dictionary mapping outcomes to odds values
            
        Returns:
            Dictionary mapping outcomes to fair odds
        """
        # Calculate implied probabilities
        implied_probs = {outcome: 1 / odd for outcome, odd in odds.items()}
        
        # Calculate total implied probability (including margin)
        total_implied_prob = sum(implied_probs.values())
        
        # Normalize probabilities to remove margin
        fair_probs = {outcome: prob / total_implied_prob 
                     for outcome, prob in implied_probs.items()}
        
        # Convert probabilities back to odds
        fair_odds = {outcome: 1 / prob if prob > 0 else float('inf') 
                    for outcome, prob in fair_probs.items()}
        
        return fair_odds
    
    @staticmethod
    def identify_best_value(odds_by_bookmaker: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[str, float]]:
        """
        Identify which bookmaker offers the best value for each outcome.
        
        Args:
            odds_by_bookmaker: Dictionary mapping bookmakers to their odds for each outcome
            
        Returns:
            Dictionary mapping outcomes to (bookmaker, odds) tuples
        """
        best_values = {}
        
        # Organize odds by outcome
        odds_by_outcome = {}
        for bookmaker, odds in odds_by_bookmaker.items():
            for outcome, odd in odds.items():
                if outcome not in odds_by_outcome:
                    odds_by_outcome[outcome] = []
                odds_by_outcome[outcome].append((bookmaker, odd))
        
        # Find best odds for each outcome
        for outcome, bookmaker_odds in odds_by_outcome.items():
            if bookmaker_odds:
                best_bookmaker, best_odd = max(bookmaker_odds, key=lambda x: x[1])
                best_values[outcome] = (best_bookmaker, best_odd)
        
        return best_values
    
    @staticmethod
    def compare_margins(odds_by_bookmaker: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Compare margins across different bookmakers.
        
        Args:
            odds_by_bookmaker: Dictionary mapping bookmakers to their odds for each outcome
            
        Returns:
            Dictionary mapping bookmakers to their margins
        """
        return {
            bookmaker: BookmakerMarginCalculator.calculate_margin(odds)
            for bookmaker, odds in odds_by_bookmaker.items()
        }


class ArbitrageDetector:
    """
    Detector for finding arbitrage opportunities across multiple bookmakers.
    
    This class provides functionality to identify profitable arbitrage opportunities
    by comparing odds from different bookmakers for the same event.
    
    Arbitrage betting (or "sure betting") is a technique where a bettor places bets on
    all possible outcomes of an event at odds that guarantee a profit regardless of the result.
    """
    
    # Define common market types
    MARKET_1X2 = "1X2"
    MARKET_OVER_UNDER = "over_under"
    MARKET_BTTS = "btts"
    MARKET_ASIAN_HANDICAP = "asian_handicap"
    
    def __init__(self, 
                min_profit_percent: float = 1.0,
                max_bookmaker_margin: float = 7.0,
                excluded_bookmakers: Optional[List[str]] = None,
                included_bookmakers: Optional[List[str]] = None,
                markets: Optional[List[str]] = None,
                min_odds: float = 1.1,
                max_odds: float = 15.0,
                stake_amount: float = 100.0,
                max_matches: int = 100):
        """
        Initialize the arbitrage detector.
        
        Args:
            min_profit_percent: Minimum profit percentage required for an opportunity
            max_bookmaker_margin: Maximum allowed bookmaker margin for individual bookmakers
            excluded_bookmakers: List of bookmaker names to exclude from analysis
            included_bookmakers: List of bookmaker names to include in analysis (if None, all except excluded are used)
            markets: List of market types to analyze (if None, all supported markets are used)
            min_odds: Minimum odds value to consider
            max_odds: Maximum odds value to consider
            stake_amount: Default stake amount for calculating optimal bet allocation
            max_matches: Maximum number of matches to analyze in a single run
        """
        self.min_profit_percent = min_profit_percent
        self.max_bookmaker_margin = max_bookmaker_margin
        self.excluded_bookmakers = set(excluded_bookmakers or [])
        self.included_bookmakers = set(included_bookmakers or [])
        self.markets = markets or [self.MARKET_1X2, self.MARKET_OVER_UNDER, self.MARKET_BTTS, self.MARKET_ASIAN_HANDICAP]
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.stake_amount = stake_amount
        self.max_matches = max_matches
        
        # Tracking variables
        self.opportunities_found = 0
        self.matches_analyzed = 0
        self.last_run_timestamp = None
    
    def find_arbitrage_opportunities(self, 
                                   connectors: List[OddsProviderConnector], 
                                   match_ids: Optional[List[str]] = None,
                                   league_id: Optional[str] = None,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities across multiple odds providers.
        
        Args:
            connectors: List of odds provider connector instances
            match_ids: Optional list of specific match IDs to analyze
            league_id: Optional league/competition ID to limit the search
            start_date: Optional start date for match search
            end_date: Optional end date for match search
            
        Returns:
            List of ArbitrageOpportunity objects representing profitable opportunities
        """
        # Track when this run started
        self.last_run_timestamp = datetime.now()
        self.opportunities_found = 0
        self.matches_analyzed = 0
        
        # Get list of matches to analyze
        if match_ids:
            all_match_ids = match_ids[:self.max_matches]
        else:
            # Get matches from the first connector
            if not connectors:
                logger.error("No odds connectors provided")
                return []
            
            try:
                matches_df = connectors[0].get_matches(
                    league_id=league_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # Limit to max_matches
                matches_df = matches_df.head(self.max_matches)
                
                all_match_ids = matches_df['match_id'].tolist()
                
                logger.info(f"Found {len(all_match_ids)} matches to analyze")
            except Exception as e:
                logger.error(f"Error getting matches: {str(e)}")
                return []
        
        # Find arbitrage opportunities for each match
        all_opportunities = []
        
        for match_id in all_match_ids:
            self.matches_analyzed += 1
            
            try:
                opportunities = self.find_arbitrage_for_match(connectors, match_id)
                all_opportunities.extend(opportunities)
                
                if opportunities:
                    logger.info(f"Found {len(opportunities)} arbitrage opportunities for match {match_id}")
                    self.opportunities_found += len(opportunities)
            except Exception as e:
                logger.error(f"Error analyzing match {match_id}: {str(e)}")
        
        logger.info(f"Found {len(all_opportunities)} total arbitrage opportunities across {self.matches_analyzed} matches")
        
        # Sort by profit percentage (highest first)
        all_opportunities.sort(key=lambda x: x.arbitrage_profit_percent, reverse=True)
        
        return all_opportunities
    
    def find_arbitrage_for_match(self,
                              connectors: List[OddsProviderConnector],
                              match_id: str) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities for a specific match.
        
        Args:
            connectors: List of odds provider connector instances
            match_id: ID of the match to analyze
            
        Returns:
            List of ArbitrageOpportunity objects for this match
        """
        opportunities = []
        
        # Get odds from all connectors
        all_odds_data = []
        match_info = {'home_team': None, 'away_team': None, 'date': None}
        
        for connector in connectors:
            try:
                odds_df = connector.fetch_match_odds(match_id)
                
                if not odds_df.empty:
                    # Update match information if not yet set
                    if match_info['home_team'] is None and 'home_team' in odds_df.columns:
                        match_info['home_team'] = odds_df['home_team'].iloc[0]
                    if match_info['away_team'] is None and 'away_team' in odds_df.columns:
                        match_info['away_team'] = odds_df['away_team'].iloc[0]
                    if match_info['date'] is None and 'date' in odds_df.columns:
                        match_info['date'] = odds_df['date'].iloc[0]
                        
                    all_odds_data.append(odds_df)
            except Exception as e:
                logger.error(f"Error fetching odds from connector {connector.__class__.__name__}: {str(e)}")
        
        if not all_odds_data:
            logger.warning(f"No odds data found for match {match_id}")
            return []
        
        # Combine odds data from all connectors
        combined_odds = pd.concat(all_odds_data, ignore_index=True)
        
        # Filter by included/excluded bookmakers
        if self.included_bookmakers:
            combined_odds = combined_odds[combined_odds['bookmaker_name'].isin(self.included_bookmakers)]
        
        if self.excluded_bookmakers:
            combined_odds = combined_odds[~combined_odds['bookmaker_name'].isin(self.excluded_bookmakers)]
        
        # Filter by odds range
        combined_odds = combined_odds[
            (combined_odds['odds'] >= self.min_odds) & 
            (combined_odds['odds'] <= self.max_odds)
        ]
        
        if combined_odds.empty:
            logger.warning(f"No valid odds found for match {match_id} after filtering")
            return []
        
        # Check for arbitrage in different market types
        for market in self.markets:
            if market == self.MARKET_1X2:
                opportunity = self.check_1x2_arbitrage(
                    match_id=match_id,
                    home_team=match_info['home_team'] or "Home Team",
                    away_team=match_info['away_team'] or "Away Team",
                    odds_data=combined_odds,
                    date=match_info['date']
                )
                
                if opportunity:
                    opportunities.append(opportunity)
            
            elif market == self.MARKET_OVER_UNDER:
                over_under_opps = self.check_over_under_arbitrage(
                    match_id=match_id,
                    home_team=match_info['home_team'] or "Home Team",
                    away_team=match_info['away_team'] or "Away Team",
                    odds_data=combined_odds,
                    date=match_info['date']
                )
                
                opportunities.extend(over_under_opps)
            
            elif market == self.MARKET_BTTS:
                btts_opportunity = self.check_btts_arbitrage(
                    match_id=match_id,
                    home_team=match_info['home_team'] or "Home Team",
                    away_team=match_info['away_team'] or "Away Team",
                    odds_data=combined_odds,
                    date=match_info['date']
                )
                
                if btts_opportunity:
                    opportunities.append(btts_opportunity)
            
            elif market == self.MARKET_ASIAN_HANDICAP:
                ah_opportunities = self.check_asian_handicap_arbitrage(
                    match_id=match_id,
                    home_team=match_info['home_team'] or "Home Team",
                    away_team=match_info['away_team'] or "Away Team",
                    odds_data=combined_odds,
                    date=match_info['date']
                )
                
                opportunities.extend(ah_opportunities)
        
        return opportunities 
    
    def check_1x2_arbitrage(self,
                         match_id: str,
                         home_team: str,
                         away_team: str,
                         odds_data: pd.DataFrame,
                         date: Optional[datetime] = None) -> Optional[ArbitrageOpportunity]:
        """
        Check for arbitrage opportunities in 1X2 (home/draw/away) markets.
        
        Args:
            match_id: Match identifier
            home_team: Name of the home team
            away_team: Name of the away team
            odds_data: DataFrame containing odds data from multiple bookmakers
            date: Optional match date
            
        Returns:
            ArbitrageOpportunity if one exists, None otherwise
        """
        # Filter for 1X2 market
        market_odds = odds_data[odds_data['market'] == '1X2'].copy()
        
        if market_odds.empty:
            return None
        
        # Get best odds for each outcome (home, draw, away)
        best_home_row = market_odds[market_odds['outcome_type'] == 'home'].sort_values('odds', ascending=False).iloc[0] if not market_odds[market_odds['outcome_type'] == 'home'].empty else None
        best_draw_row = market_odds[market_odds['outcome_type'] == 'draw'].sort_values('odds', ascending=False).iloc[0] if not market_odds[market_odds['outcome_type'] == 'draw'].empty else None
        best_away_row = market_odds[market_odds['outcome_type'] == 'away'].sort_values('odds', ascending=False).iloc[0] if not market_odds[market_odds['outcome_type'] == 'away'].empty else None
        
        # Skip if any outcome is missing
        if best_home_row is None or best_draw_row is None or best_away_row is None:
            return None
        
        # Create odds dictionary
        best_odds = {
            'home': best_home_row['odds'],
            'draw': best_draw_row['odds'],
            'away': best_away_row['odds']
        }
        
        # Create bookmakers dictionary
        bookmakers = {
            'home': best_home_row['bookmaker_name'],
            'draw': best_draw_row['bookmaker_name'],
            'away': best_away_row['bookmaker_name']
        }
        
        # Calculate implied probabilities
        implied_probabilities = self.get_implied_probabilities(best_odds)
        total_implied_probability = sum(implied_probabilities.values())
        
        # Check if this is an arbitrage opportunity
        if total_implied_probability < 1.0:
            arbitrage_profit_percent = (1 - total_implied_probability) * 100
            
            # Check if profit meets minimum threshold
            if arbitrage_profit_percent >= self.min_profit_percent:
                # Calculate optimal stakes
                optimal_stakes = self.calculate_optimal_stakes(best_odds, self.stake_amount)
                expected_return = self.stake_amount * (1 + arbitrage_profit_percent / 100)
                
                # Create and return opportunity
                return ArbitrageOpportunity(
                    match_id=match_id,
                    home_team=home_team,
                    away_team=away_team,
                    market_type="1X2",
                    bookmakers=bookmakers,
                    odds=best_odds,
                    implied_probabilities=implied_probabilities,
                    total_implied_probability=total_implied_probability,
                    arbitrage_profit_percent=arbitrage_profit_percent,
                    optimal_stakes=optimal_stakes,
                    total_stake=self.stake_amount,
                    expected_return=expected_return,
                    date=date
                )
        
        return None
    
    def check_over_under_arbitrage(self,
                               match_id: str,
                               home_team: str,
                               away_team: str,
                               odds_data: pd.DataFrame,
                               date: Optional[datetime] = None) -> List[ArbitrageOpportunity]:
        """
        Check for arbitrage opportunities in over/under markets.
        
        Args:
            match_id: Match identifier
            home_team: Name of the home team
            away_team: Name of the away team
            odds_data: DataFrame containing odds data from multiple bookmakers
            date: Optional match date
            
        Returns:
            List of ArbitrageOpportunity objects for over/under markets
        """
        # Filter for over/under markets
        market_odds = odds_data[odds_data['market'].str.startswith('over_under_')].copy()
        
        if market_odds.empty:
            return []
        
        opportunities = []
        
        # Group by goal line (e.g., over_under_2.5)
        for goal_line, group in market_odds.groupby('market'):
            # Get best odds for over and under
            best_over_row = group[group['outcome_type'] == 'over'].sort_values('odds', ascending=False).iloc[0] if not group[group['outcome_type'] == 'over'].empty else None
            best_under_row = group[group['outcome_type'] == 'under'].sort_values('odds', ascending=False).iloc[0] if not group[group['outcome_type'] == 'under'].empty else None
            
            # Skip if any outcome is missing
            if best_over_row is None or best_under_row is None:
                continue
            
            # Create odds dictionary
            best_odds = {
                'over': best_over_row['odds'],
                'under': best_under_row['odds']
            }
            
            # Create bookmakers dictionary
            bookmakers = {
                'over': best_over_row['bookmaker_name'],
                'under': best_under_row['bookmaker_name']
            }
            
            # Calculate implied probabilities
            implied_probabilities = self.get_implied_probabilities(best_odds)
            total_implied_probability = sum(implied_probabilities.values())
            
            # Check if this is an arbitrage opportunity
            if total_implied_probability < 1.0:
                arbitrage_profit_percent = (1 - total_implied_probability) * 100
                
                # Check if profit meets minimum threshold
                if arbitrage_profit_percent >= self.min_profit_percent:
                    # Calculate optimal stakes
                    optimal_stakes = self.calculate_optimal_stakes(best_odds, self.stake_amount)
                    expected_return = self.stake_amount * (1 + arbitrage_profit_percent / 100)
                    
                    # Extract goal line value from market name (e.g., "2.5" from "over_under_2.5")
                    goal_line_value = goal_line.split('_')[-1]
                    
                    # Create and append opportunity
                    opportunities.append(ArbitrageOpportunity(
                        match_id=match_id,
                        home_team=home_team,
                        away_team=away_team,
                        market_type=f"Over/Under {goal_line_value}",
                        bookmakers=bookmakers,
                        odds=best_odds,
                        implied_probabilities=implied_probabilities,
                        total_implied_probability=total_implied_probability,
                        arbitrage_profit_percent=arbitrage_profit_percent,
                        optimal_stakes=optimal_stakes,
                        total_stake=self.stake_amount,
                        expected_return=expected_return,
                        date=date
                    ))
        
        return opportunities
    
    def check_btts_arbitrage(self,
                          match_id: str,
                          home_team: str,
                          away_team: str,
                          odds_data: pd.DataFrame,
                          date: Optional[datetime] = None) -> Optional[ArbitrageOpportunity]:
        """
        Check for arbitrage opportunities in Both Teams To Score (BTTS) markets.
        
        Args:
            match_id: Match identifier
            home_team: Name of the home team
            away_team: Name of the away team
            odds_data: DataFrame containing odds data from multiple bookmakers
            date: Optional match date
            
        Returns:
            ArbitrageOpportunity if one exists, None otherwise
        """
        # Filter for BTTS market
        market_odds = odds_data[odds_data['market'] == 'btts'].copy()
        
        if market_odds.empty:
            return None
        
        # Get best odds for each outcome (yes, no)
        best_yes_row = market_odds[market_odds['outcome_type'] == 'yes'].sort_values('odds', ascending=False).iloc[0] if not market_odds[market_odds['outcome_type'] == 'yes'].empty else None
        best_no_row = market_odds[market_odds['outcome_type'] == 'no'].sort_values('odds', ascending=False).iloc[0] if not market_odds[market_odds['outcome_type'] == 'no'].empty else None
        
        # Skip if any outcome is missing
        if best_yes_row is None or best_no_row is None:
            return None
        
        # Create odds dictionary
        best_odds = {
            'yes': best_yes_row['odds'],
            'no': best_no_row['odds']
        }
        
        # Create bookmakers dictionary
        bookmakers = {
            'yes': best_yes_row['bookmaker_name'],
            'no': best_no_row['bookmaker_name']
        }
        
        # Calculate implied probabilities
        implied_probabilities = self.get_implied_probabilities(best_odds)
        total_implied_probability = sum(implied_probabilities.values())
        
        # Check if this is an arbitrage opportunity
        if total_implied_probability < 1.0:
            arbitrage_profit_percent = (1 - total_implied_probability) * 100
            
            # Check if profit meets minimum threshold
            if arbitrage_profit_percent >= self.min_profit_percent:
                # Calculate optimal stakes
                optimal_stakes = self.calculate_optimal_stakes(best_odds, self.stake_amount)
                expected_return = self.stake_amount * (1 + arbitrage_profit_percent / 100)
                
                # Create and return opportunity
                return ArbitrageOpportunity(
                    match_id=match_id,
                    home_team=home_team,
                    away_team=away_team,
                    market_type="BTTS",
                    bookmakers=bookmakers,
                    odds=best_odds,
                    implied_probabilities=implied_probabilities,
                    total_implied_probability=total_implied_probability,
                    arbitrage_profit_percent=arbitrage_profit_percent,
                    optimal_stakes=optimal_stakes,
                    total_stake=self.stake_amount,
                    expected_return=expected_return,
                    date=date
                )
        
        return None
    
    def check_asian_handicap_arbitrage(self,
                                    match_id: str,
                                    home_team: str,
                                    away_team: str,
                                    odds_data: pd.DataFrame,
                                    date: Optional[datetime] = None) -> List[ArbitrageOpportunity]:
        """
        Check for arbitrage opportunities in Asian Handicap markets.
        
        Args:
            match_id: Match identifier
            home_team: Name of the home team
            away_team: Name of the away team
            odds_data: DataFrame containing odds data from multiple bookmakers
            date: Optional match date
            
        Returns:
            List of ArbitrageOpportunity objects for Asian Handicap markets
        """
        # Filter for Asian Handicap markets
        market_odds = odds_data[odds_data['market'].str.startswith('ah_')].copy()
        
        if market_odds.empty:
            return []
        
        opportunities = []
        
        # Group by handicap line (e.g., ah_-1.5)
        for handicap_line, group in market_odds.groupby('market'):
            # Get best odds for home and away with this handicap
            best_home_row = group[group['outcome_type'] == 'home'].sort_values('odds', ascending=False).iloc[0] if not group[group['outcome_type'] == 'home'].empty else None
            best_away_row = group[group['outcome_type'] == 'away'].sort_values('odds', ascending=False).iloc[0] if not group[group['outcome_type'] == 'away'].empty else None
            
            # Skip if any outcome is missing
            if best_home_row is None or best_away_row is None:
                continue
            
            # Create odds dictionary
            best_odds = {
                'home': best_home_row['odds'],
                'away': best_away_row['odds']
            }
            
            # Create bookmakers dictionary
            bookmakers = {
                'home': best_home_row['bookmaker_name'],
                'away': best_away_row['bookmaker_name']
            }
            
            # Calculate implied probabilities
            implied_probabilities = self.get_implied_probabilities(best_odds)
            total_implied_probability = sum(implied_probabilities.values())
            
            # Check if this is an arbitrage opportunity
            if total_implied_probability < 1.0:
                arbitrage_profit_percent = (1 - total_implied_probability) * 100
                
                # Check if profit meets minimum threshold
                if arbitrage_profit_percent >= self.min_profit_percent:
                    # Calculate optimal stakes
                    optimal_stakes = self.calculate_optimal_stakes(best_odds, self.stake_amount)
                    expected_return = self.stake_amount * (1 + arbitrage_profit_percent / 100)
                    
                    # Extract handicap value from market name (e.g., "-1.5" from "ah_-1.5")
                    handicap_value = handicap_line.split('_')[-1]
                    
                    # Create and append opportunity
                    opportunities.append(ArbitrageOpportunity(
                        match_id=match_id,
                        home_team=home_team,
                        away_team=away_team,
                        market_type=f"Asian Handicap {handicap_value}",
                        bookmakers=bookmakers,
                        odds=best_odds,
                        implied_probabilities=implied_probabilities,
                        total_implied_probability=total_implied_probability,
                        arbitrage_profit_percent=arbitrage_profit_percent,
                        optimal_stakes=optimal_stakes,
                        total_stake=self.stake_amount,
                        expected_return=expected_return,
                        date=date
                    ))
        
        return opportunities
    
    def calculate_optimal_stakes(self,
                             odds: Dict[str, float],
                             total_stake: float = 100.0) -> Dict[str, float]:
        """
        Calculate optimal stake distribution to guarantee the same return regardless of outcome.
        
        Args:
            odds: Dictionary mapping outcomes to odds values
            total_stake: Total amount to stake
            
        Returns:
            Dictionary mapping outcomes to optimal stake amounts
        """
        # Calculate implied probabilities
        implied_probs = self.get_implied_probabilities(odds)
        
        # Calculate proportion of total stake for each outcome
        proportions = {outcome: prob for outcome, prob in implied_probs.items()}
        
        # Calculate actual stake amounts
        stakes = {}
        for outcome, proportion in proportions.items():
            stakes[outcome] = round(total_stake * proportion, 2)
        
        return stakes
    
    def get_implied_probabilities(self,
                              odds: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate implied probabilities from odds.
        
        Args:
            odds: Dictionary mapping outcomes to odds values
            
        Returns:
            Dictionary mapping outcomes to implied probabilities
        """
        return {outcome: 1 / odd for outcome, odd in odds.items()}
    
    def is_valid_arbitrage(self,
                        implied_probabilities: Dict[str, float],
                        min_profit_percent: Optional[float] = None) -> bool:
        """
        Check if a set of implied probabilities represents a valid arbitrage opportunity.
        
        Args:
            implied_probabilities: Dictionary mapping outcomes to implied probabilities
            min_profit_percent: Optional minimum profit percentage (uses instance value if None)
            
        Returns:
            True if valid arbitrage opportunity, False otherwise
        """
        # Use instance value if not specified
        if min_profit_percent is None:
            min_profit_percent = self.min_profit_percent
        
        # Calculate total implied probability
        total_implied_probability = sum(implied_probabilities.values())
        
        # Check if total is less than 1 (indicating arbitrage opportunity)
        if total_implied_probability < 1.0:
            # Calculate profit percentage
            profit_percent = (1 - total_implied_probability) * 100
            
            # Check if profit meets minimum threshold
            return profit_percent >= min_profit_percent
        
        return False
    
    def to_dataframe(self, opportunities: List[ArbitrageOpportunity]) -> pd.DataFrame:
        """
        Convert a list of arbitrage opportunities to a DataFrame for analysis.
        
        Args:
            opportunities: List of ArbitrageOpportunity objects
            
        Returns:
            DataFrame representation of the opportunities
        """
        if not opportunities:
            return pd.DataFrame()
        
        data = []
        for opp in opportunities:
            row = {
                'match_id': opp.match_id,
                'home_team': opp.home_team,
                'away_team': opp.away_team,
                'market_type': opp.market_type,
                'profit_percent': opp.arbitrage_profit_percent,
                'total_stake': opp.total_stake,
                'expected_return': opp.expected_return,
                'date': opp.date,
                'timestamp': opp.timestamp
            }
            
            # Add odds and bookmakers
            for outcome, odd in opp.odds.items():
                row[f'{outcome}_odds'] = odd
                row[f'{outcome}_bookmaker'] = opp.bookmakers.get(outcome, '')
                row[f'{outcome}_stake'] = opp.optimal_stakes.get(outcome, 0.0)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the last arbitrage detection run.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            'opportunities_found': self.opportunities_found,
            'matches_analyzed': self.matches_analyzed,
            'last_run_timestamp': self.last_run_timestamp.isoformat() if self.last_run_timestamp else None,
            'min_profit_percent': self.min_profit_percent,
            'max_bookmaker_margin': self.max_bookmaker_margin,
            'markets_analyzed': self.markets,
            'excluded_bookmakers': list(self.excluded_bookmakers),
            'included_bookmakers': list(self.included_bookmakers)
        }


class CorrelationHandler:
    """
    Handler for managing correlated bets.
    
    This class provides functionality to detect and handle correlated betting outcomes,
    which is essential for accurate arbitrage and value betting calculations.
    """
    
    # Define common correlated markets
    CORRELATED_OUTCOMES = {
        ('1X2', 'home', 'btts', 'no'): 0.4,    # Home win and BTTS No
        ('1X2', 'away', 'btts', 'no'): 0.4,    # Away win and BTTS No
        ('1X2', 'home', 'over_under_2.5', 'over'): 0.5,  # Home win and Over 2.5 goals
        ('1X2', 'away', 'over_under_2.5', 'over'): 0.5,  # Away win and Over 2.5 goals
        ('1X2', 'draw', 'over_under_2.5', 'under'): 0.5,  # Draw and Under 2.5 goals
        ('btts', 'yes', 'over_under_2.5', 'over'): 0.7,  # BTTS Yes and Over 2.5 goals
        ('btts', 'no', 'over_under_2.5', 'under'): 0.7,  # BTTS No and Under 2.5 goals
    }
    
    @staticmethod
    def get_correlation(market1: str, outcome1: str, market2: str, outcome2: str) -> float:
        """
        Get the correlation coefficient between two betting outcomes.
        
        Args:
            market1: First market (e.g., '1X2')
            outcome1: First outcome (e.g., 'home')
            market2: Second market (e.g., 'btts')
            outcome2: Second outcome (e.g., 'no')
            
        Returns:
            Correlation coefficient between -1.0 and 1.0
        """
        # Check for exact match in predefined correlations
        key = (market1, outcome1, market2, outcome2)
        if key in CorrelationHandler.CORRELATED_OUTCOMES:
            return CorrelationHandler.CORRELATED_OUTCOMES[key]
        
        # Check for the reverse combination
        reverse_key = (market2, outcome2, market1, outcome1)
        if reverse_key in CorrelationHandler.CORRELATED_OUTCOMES:
            return CorrelationHandler.CORRELATED_OUTCOMES[reverse_key]
        
        # Default to no correlation
        return 0.0
    
    @staticmethod
    def is_correlated(market1: str, outcome1: str, market2: str, outcome2: str, threshold: float = 0.3) -> bool:
        """
        Check if two betting outcomes are significantly correlated.
        
        Args:
            market1: First market (e.g., '1X2')
            outcome1: First outcome (e.g., 'home')
            market2: Second market (e.g., 'btts')
            outcome2: Second outcome (e.g., 'no')
            threshold: Correlation threshold to consider significant
            
        Returns:
            True if correlation is above threshold, False otherwise
        """
        correlation = abs(CorrelationHandler.get_correlation(market1, outcome1, market2, outcome2))
        return correlation >= threshold
    
    @staticmethod
    def calculate_joint_probability(prob1: float, prob2: float, correlation: float) -> float:
        """
        Calculate the joint probability of two correlated events.
        
        Args:
            prob1: Probability of first event
            prob2: Probability of second event
            correlation: Correlation coefficient between events
            
        Returns:
            Joint probability of both events occurring
        """
        # Independent case
        if correlation == 0:
            return prob1 * prob2
        
        # Perfect positive correlation
        if correlation == 1:
            return min(prob1, prob2)
        
        # Perfect negative correlation
        if correlation == -1:
            return max(0, prob1 + prob2 - 1)
        
        # General case (using a simplified model)
        if correlation > 0:
            # Positive correlation increases joint probability
            independent_prob = prob1 * prob2
            max_dependent_prob = min(prob1, prob2)
            return independent_prob + correlation * (max_dependent_prob - independent_prob)
        else:
            # Negative correlation decreases joint probability
            independent_prob = prob1 * prob2
            min_dependent_prob = max(0, prob1 + prob2 - 1)
            return independent_prob + abs(correlation) * (min_dependent_prob - independent_prob) 