"""
PageRank Adaptation for Soccer Team Rankings

This module implements a PageRank-based algorithm for ranking soccer teams.
The algorithm treats teams as nodes in a graph, with match outcomes as edges.

Reference: Lazova, V., Basnarkov, L. (2015): PageRank Approach to Ranking National 
           Football Teams. arXiv.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import joblib

# Import project components
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("data.pagerank_ratings")

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Define paths
RATINGS_DIR = os.path.join(DATA_DIR, "ratings")
PAGERANK_RATINGS_DIR = os.path.join(RATINGS_DIR, "pagerank")
os.makedirs(PAGERANK_RATINGS_DIR, exist_ok=True)


class PageRankTeamRater:
    """
    Implementation of a PageRank-based algorithm for ranking soccer teams.
    
    This system models teams as nodes in a directed graph, where:
    1. Matches create directed edges from loser to winner
    2. Edge weights depend on score differential, venue, recency, etc.
    3. PageRank algorithm calculates the "importance" of each team
    4. Teams with high PageRank values are considered stronger
    
    For ties, bidirectional edges with lower weights are created.
    """
    
    def __init__(self, 
                damping_factor: float = 0.85,  # Standard PageRank damping factor
                weight_score_diff: bool = True,  # Whether to weight edges by score difference
                weight_home_advantage: bool = True,  # Whether to adjust for home advantage
                recency_halflife: Optional[int] = 365,  # Days for recency weight to halve
                include_draws: bool = True,  # Whether to include draws in the graph
                draw_weight: float = 0.5,  # Weight of edges for drawn matches
                min_edge_weight: float = 0.1,  # Minimum edge weight
                max_edge_weight: float = 5.0  # Maximum edge weight
                ):
        """
        Initialize the PageRank team rater.
        
        Args:
            damping_factor: PageRank damping factor (probability of following links)
            weight_score_diff: Whether to weight edges by score difference
            weight_home_advantage: Whether to adjust for home advantage
            recency_halflife: Days for edge weight to halve due to recency (None to disable)
            include_draws: Whether to include draws in the graph
            draw_weight: Weight of edges for drawn matches
            min_edge_weight: Minimum edge weight
            max_edge_weight: Maximum edge weight
        """
        self.damping_factor = damping_factor
        self.weight_score_diff = weight_score_diff
        self.weight_home_advantage = weight_home_advantage
        self.recency_halflife = recency_halflife
        self.include_draws = include_draws
        self.draw_weight = draw_weight
        self.min_edge_weight = min_edge_weight
        self.max_edge_weight = max_edge_weight
        
        # Initialize graph and rankings
        self.graph = nx.DiGraph()
        self.rankings = {}
        self.team_info = {}
        
        # Fit metadata
        self.last_fit_date = None
        self.fit_metadata = {
            "model_type": "pagerank",
            "created_at": datetime.now().isoformat(),
            "damping_factor": damping_factor,
            "weight_score_diff": weight_score_diff,
            "weight_home_advantage": weight_home_advantage,
            "recency_halflife": recency_halflife,
            "include_draws": include_draws,
            "n_teams": 0,
            "n_matches": 0
        }
    
    def _calculate_edge_weight(self, 
                             home_goals: int, 
                             away_goals: int, 
                             match_date: datetime,
                             current_date: datetime,
                             match_importance: float = 1.0) -> float:
        """
        Calculate the edge weight for a match.
        
        Args:
            home_goals: Number of goals scored by home team
            away_goals: Number of goals scored by away team
            match_date: Date of the match
            current_date: Current date for recency calculation
            match_importance: Importance factor for the match
            
        Returns:
            float: Edge weight
        """
        # Base weight
        weight = 1.0
        
        # Score difference weighting
        if self.weight_score_diff:
            goal_diff = abs(home_goals - away_goals)
            if goal_diff > 0:
                # Increasing weight for larger goal differences, with diminishing returns
                weight *= 1.0 + np.log1p(goal_diff)
        
        # Apply match importance
        weight *= match_importance
        
        # Apply recency weighting if enabled
        if self.recency_halflife is not None:
            days_ago = (current_date - match_date).days
            if days_ago > 0:
                # Exponential decay based on half-life
                recency_factor = 2 ** (-days_ago / self.recency_halflife)
                weight *= recency_factor
        
        # Clamp weight to limits
        weight = max(self.min_edge_weight, min(self.max_edge_weight, weight))
        
        return weight
    
    def _add_match_to_graph(self, 
                          home_team: Any, 
                          away_team: Any,
                          home_goals: int,
                          away_goals: int,
                          match_date: datetime,
                          current_date: datetime,
                          match_importance: float = 1.0):
        """
        Add a match to the directed graph.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_date: Date of the match
            current_date: Current date for recency calculation
            match_importance: Importance factor for the match
        """
        # Ensure teams are in the graph
        if home_team not in self.graph:
            self.graph.add_node(home_team)
        if away_team not in self.graph:
            self.graph.add_node(away_team)
        
        # Calculate edge weight
        weight = self._calculate_edge_weight(
            home_goals, away_goals, match_date, current_date, match_importance
        )
        
        # Home advantage adjustment
        if self.weight_home_advantage:
            home_advantage_factor = 0.85  # Reduce weight for home wins, increase for away wins
            if home_goals > away_goals:  # Home win
                weight *= home_advantage_factor
            elif away_goals > home_goals:  # Away win
                weight /= home_advantage_factor
        
        # Add edges based on match outcome
        if home_goals > away_goals:  # Home win
            # Add/update edge from loser to winner
            if self.graph.has_edge(away_team, home_team):
                # Add to existing weight
                self.graph[away_team][home_team]["weight"] += weight
            else:
                # Create new edge
                self.graph.add_edge(away_team, home_team, weight=weight)
                
        elif away_goals > home_goals:  # Away win
            # Add/update edge from loser to winner
            if self.graph.has_edge(home_team, away_team):
                # Add to existing weight
                self.graph[home_team][away_team]["weight"] += weight
            else:
                # Create new edge
                self.graph.add_edge(home_team, away_team, weight=weight)
                
        elif self.include_draws:  # Draw
            draw_weight = weight * self.draw_weight
            
            # Add bidirectional edges with draw weight
            if self.graph.has_edge(home_team, away_team):
                self.graph[home_team][away_team]["weight"] += draw_weight
            else:
                self.graph.add_edge(home_team, away_team, weight=draw_weight)
                
            if self.graph.has_edge(away_team, home_team):
                self.graph[away_team][home_team]["weight"] += draw_weight
            else:
                self.graph.add_edge(away_team, home_team, weight=draw_weight)
    
    def fit(self, 
           matches_df: pd.DataFrame, 
           reference_date: Optional[datetime] = None,
           competition_importance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Build the team graph and calculate PageRank ratings.
        
        Args:
            matches_df: DataFrame containing match data
            reference_date: Date to use for recency calculations (defaults to max date in data)
            competition_importance: Optional dictionary mapping competition names to importance factors
            
        Returns:
            Dict: Fitting results
        """
        # Check required columns
        required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals', 'date']
        
        # If column names are different, try to map them
        col_mapping = {}
        
        if not all(col in matches_df.columns for col in required_cols):
            # Try to map columns automatically
            possible_mappings = {
                'home_team': ['home_club_id', 'home_team_id', 'home_id'],
                'away_team': ['away_club_id', 'away_team_id', 'away_id'],
                'home_goals': ['home_club_goals', 'home_score', 'home_team_goals'],
                'away_goals': ['away_club_goals', 'away_score', 'away_team_goals'],
                'date': ['match_date', 'game_date', 'date_time']
            }
            
            for req_col, alternates in possible_mappings.items():
                for alt_col in alternates:
                    if alt_col in matches_df.columns:
                        col_mapping[req_col] = alt_col
                        break
        
        # Create a copy of the dataframe with correct column names
        matches = matches_df.copy()
        for req_col, alt_col in col_mapping.items():
            matches[req_col] = matches_df[alt_col]
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_dtype(matches['date']):
            matches['date'] = pd.to_datetime(matches['date'])
        
        # Sort matches by date
        matches = matches.sort_values('date')
        
        # Set reference date if not provided
        if reference_date is None:
            reference_date = matches['date'].max()
        
        # Default competition importance
        if competition_importance is None:
            competition_importance = {
                'league': 1.0,
                'cup': 1.1,
                'champions_league': 1.2,
                'international': 1.3
            }
        
        # Reset graph and rankings
        self.graph = nx.DiGraph()
        self.rankings = {}
        
        # Process each match and build the graph
        for idx, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            match_date = match['date']
            
            # Determine match importance
            if 'competition' in match:
                competition = match['competition']
                importance = competition_importance.get(competition, 1.0)
            else:
                importance = 1.0
            
            # Add match to graph
            self._add_match_to_graph(
                home_team=home_team,
                away_team=away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                match_date=match_date,
                current_date=reference_date,
                match_importance=importance
            )
            
            # Store team info
            for team_id in [home_team, away_team]:
                if team_id not in self.team_info:
                    self.team_info[team_id] = {
                        'matches_played': 0,
                        'wins': 0,
                        'draws': 0,
                        'losses': 0,
                        'goals_for': 0,
                        'goals_against': 0
                    }
            
            # Update team stats
            self.team_info[home_team]['matches_played'] += 1
            self.team_info[away_team]['matches_played'] += 1
            
            self.team_info[home_team]['goals_for'] += home_goals
            self.team_info[home_team]['goals_against'] += away_goals
            self.team_info[away_team]['goals_for'] += away_goals
            self.team_info[away_team]['goals_against'] += home_goals
            
            if home_goals > away_goals:
                self.team_info[home_team]['wins'] += 1
                self.team_info[away_team]['losses'] += 1
            elif away_goals > home_goals:
                self.team_info[away_team]['wins'] += 1
                self.team_info[home_team]['losses'] += 1
            else:
                self.team_info[home_team]['draws'] += 1
                self.team_info[away_team]['draws'] += 1
        
        # Calculate PageRank
        try:
            # Use NetworkX's PageRank implementation
            self.rankings = nx.pagerank(
                self.graph,
                alpha=self.damping_factor,
                weight='weight'
            )
            
            logger.info(f"PageRank calculation completed for {len(self.rankings)} teams")
            
        except nx.PowerIterationFailedConvergence:
            # Handle convergence issues
            logger.warning("PageRank failed to converge, using personalized PageRank with uniform teleport vector")
            
            # Use personalized PageRank with uniform teleport vector
            n_teams = len(self.graph.nodes())
            personalization = {team: 1.0/n_teams for team in self.graph.nodes()}
            
            self.rankings = nx.pagerank(
                self.graph,
                alpha=self.damping_factor,
                weight='weight',
                personalization=personalization,
                max_iter=1000
            )
        
        # Update fit metadata
        self.last_fit_date = reference_date
        self.fit_metadata.update({
            "fitted_at": datetime.now().isoformat(),
            "reference_date": reference_date.isoformat(),
            "n_teams": len(self.rankings),
            "n_matches": len(matches)
        })
        
        # Return fit results
        return {
            "n_teams": len(self.rankings),
            "n_matches": len(matches),
            "damping_factor": self.damping_factor,
            "reference_date": reference_date.isoformat(),
            "top_teams": sorted(self.rankings.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_rankings(self, 
                   top_n: Optional[int] = None, 
                   team_ids: Optional[List[Any]] = None) -> Dict[Any, Dict[str, Any]]:
        """
        Get team rankings with optional filtering.
        
        Args:
            top_n: Optional limit to return only the top N teams
            team_ids: Optional list of team IDs to filter by
            
        Returns:
            Dict: Dictionary of team rankings with additional stats
        """
        if not self.rankings:
            raise ValueError("Model must be fitted before getting rankings")
        
        # Get sorted rankings
        sorted_rankings = sorted(self.rankings.items(), key=lambda x: x[1], reverse=True)
        
        # Filter by team_ids if provided
        if team_ids is not None:
            sorted_rankings = [item for item in sorted_rankings if item[0] in team_ids]
        
        # Limit to top_n if provided
        if top_n is not None:
            sorted_rankings = sorted_rankings[:top_n]
        
        # Build detailed rankings dictionary
        rankings_dict = {}
        for rank, (team_id, score) in enumerate(sorted_rankings, 1):
            # Get team info if available
            team_stats = self.team_info.get(team_id, {})
            
            # Team ranking details
            rankings_dict[team_id] = {
                "team_id": team_id,
                "rank": rank,
                "pagerank_score": score,
                "pagerank_normalized": score / max(item[1] for item in sorted_rankings),
                "in_degree": self.graph.in_degree(team_id, weight='weight') if team_id in self.graph else 0,
                "out_degree": self.graph.out_degree(team_id, weight='weight') if team_id in self.graph else 0,
                **team_stats
            }
        
        return rankings_dict
    
    def predict_match(self, home_team: Any, away_team: Any) -> Dict[str, Any]:
        """
        Predict the outcome of a match based on team PageRank scores.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            
        Returns:
            Dict: Prediction results
        """
        if not self.rankings:
            raise ValueError("Model must be fitted before making predictions")
        
        # Check if teams are in the rankings
        if home_team not in self.rankings or away_team not in self.rankings:
            raise ValueError(f"Team not found in rankings: {home_team if home_team not in self.rankings else away_team}")
        
        # Get team scores
        home_score = self.rankings[home_team]
        away_score = self.rankings[away_team]
        
        # Apply home advantage factor
        home_advantage_factor = 1.2  # Home team gets a 20% boost
        adjusted_home_score = home_score * home_advantage_factor
        
        # Calculate win probabilities
        # Using a logistic function to convert relative scores to probabilities
        total_score = adjusted_home_score + away_score
        p_home_win = adjusted_home_score / total_score
        p_away_win = away_score / total_score
        
        # Include draw probability
        # For simplicity, using a fixed model based on the scoreline being close
        score_diff = abs(adjusted_home_score - away_score) / total_score
        p_draw = 0.3 * (1 - score_diff)  # Higher probability of draw when scores are close
        
        # Normalize probabilities
        total_prob = p_home_win + p_draw + p_away_win
        p_home_win /= total_prob
        p_draw /= total_prob
        p_away_win /= total_prob
        
        # Infer expected goals
        # Using a model where expected goals relates to team strength
        avg_goals_per_team = 1.3  # Average goals per team in soccer
        expected_home_goals = avg_goals_per_team * (adjusted_home_score / (np.mean(list(self.rankings.values()))))
        expected_away_goals = avg_goals_per_team * (away_score / (np.mean(list(self.rankings.values()))))
        
        # Return prediction
        return {
            'home_team': {
                'id': home_team,
                'pagerank_score': home_score,
                'expected_goals': float(expected_home_goals)
            },
            'away_team': {
                'id': away_team,
                'pagerank_score': away_score,
                'expected_goals': float(expected_away_goals)
            },
            'prediction': {
                'p_home_win': float(p_home_win),
                'p_draw': float(p_draw),
                'p_away_win': float(p_away_win),
                'expected_goal_diff': float(expected_home_goals - expected_away_goals)
            }
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            str: Path to the saved file
        """
        # Default path if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                PAGERANK_RATINGS_DIR, 
                f"pagerank_ratings_{timestamp}.pkl"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, "wb") as f:
            joblib.dump(self, f)
        
        logger.info(f"PageRank team rater saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "PageRankTeamRater":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved file
            
        Returns:
            PageRankTeamRater: Loaded model
        """
        with open(filepath, "rb") as f:
            model = joblib.load(f)
        
        logger.info(f"PageRank team rater loaded from {filepath}")
        
        return model


def calculate_pagerank_ratings(
    matches_df: pd.DataFrame,
    damping_factor: float = 0.85,
    weight_score_diff: bool = True,
    weight_home_advantage: bool = True,
    recency_halflife: Optional[int] = 365,
    include_draws: bool = True,
    reference_date: Optional[datetime] = None,
    competition_importance: Optional[Dict[str, float]] = None
) -> Tuple[PageRankTeamRater, Dict[Any, Dict[str, Any]]]:
    """
    Calculate PageRank-based ratings for teams based on match data.
    
    Args:
        matches_df: DataFrame containing match data
        damping_factor: PageRank damping factor
        weight_score_diff: Whether to weight edges by score difference
        weight_home_advantage: Whether to adjust for home advantage
        recency_halflife: Days for recency weight to halve
        include_draws: Whether to include draws in the graph
        reference_date: Date to use for recency calculations
        competition_importance: Optional dictionary mapping competition names to importance factors
        
    Returns:
        Tuple[PageRankTeamRater, Dict]: Model and team rankings
    """
    # Initialize rater
    pagerank_rater = PageRankTeamRater(
        damping_factor=damping_factor,
        weight_score_diff=weight_score_diff,
        weight_home_advantage=weight_home_advantage,
        recency_halflife=recency_halflife,
        include_draws=include_draws
    )
    
    # Fit model
    pagerank_rater.fit(
        matches_df=matches_df,
        reference_date=reference_date,
        competition_importance=competition_importance
    )
    
    # Get rankings
    rankings = pagerank_rater.get_rankings()
    
    return pagerank_rater, rankings


def predict_with_pagerank(
    pagerank_rater: PageRankTeamRater,
    fixtures_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict match outcomes using PageRank ratings.
    
    Args:
        pagerank_rater: Trained PageRank team rater
        fixtures_df: DataFrame containing fixtures to predict
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Check required columns
    required_cols = ['home_team', 'away_team']
    missing_cols = [col for col in required_cols if col not in fixtures_df.columns]
    
    if missing_cols:
        # Try to map columns automatically
        possible_mappings = {
            'home_team': ['home_club_id', 'home_team_id', 'home_id'],
            'away_team': ['away_club_id', 'away_team_id', 'away_id']
        }
        
        for req_col in missing_cols:
            for alt_col in possible_mappings[req_col]:
                if alt_col in fixtures_df.columns:
                    fixtures_df[req_col] = fixtures_df[alt_col]
                    break
    
    # Make predictions
    predictions = []
    
    for idx, fixture in fixtures_df.iterrows():
        try:
            home_team = fixture['home_team']
            away_team = fixture['away_team']
            
            # Predict match
            prediction = pagerank_rater.predict_match(home_team, away_team)
            
            # Extract prediction information
            pred_info = {
                'fixture_id': idx,
                'home_team': home_team,
                'away_team': away_team,
                'home_pagerank': prediction['home_team']['pagerank_score'],
                'away_pagerank': prediction['away_team']['pagerank_score'],
                'expected_home_goals': prediction['home_team']['expected_goals'],
                'expected_away_goals': prediction['away_team']['expected_goals'],
                'expected_goal_diff': prediction['prediction']['expected_goal_diff'],
                'p_home_win': prediction['prediction']['p_home_win'],
                'p_draw': prediction['prediction']['p_draw'],
                'p_away_win': prediction['prediction']['p_away_win']
            }
            
            predictions.append(pred_info)
        
        except Exception as e:
            logger.warning(f"Error predicting match {idx}: {e}")
    
    # Create DataFrame with predictions
    if predictions:
        predictions_df = pd.DataFrame(predictions)
    else:
        # Create empty DataFrame with expected columns
        predictions_df = pd.DataFrame(columns=[
            'fixture_id', 'home_team', 'away_team', 'home_pagerank', 'away_pagerank',
            'expected_home_goals', 'expected_away_goals', 'expected_goal_diff',
            'p_home_win', 'p_draw', 'p_away_win'
        ])
    
    return predictions_df


def generate_pagerank_features(matches_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Generate PageRank rating features for a match dataset.
    
    Args:
        matches_df: DataFrame containing match data
        **kwargs: Additional arguments to pass to calculate_pagerank_ratings
        
    Returns:
        pd.DataFrame: DataFrame with PageRank features
    """
    # Calculate ratings
    pagerank_rater, _ = calculate_pagerank_ratings(matches_df, **kwargs)
    
    # Extract all unique teams
    all_teams = set()
    for _, match in matches_df.iterrows():
        all_teams.add(match['home_team'])
        all_teams.add(match['away_team'])
    
    # Get rankings for all teams
    rankings = pagerank_rater.get_rankings(team_ids=list(all_teams))
    
    # Create a lookup from team ID to ranking data
    rankings_lookup = {team_id: data for team_id, data in rankings.items()}
    
    # Create features DataFrame
    features = []
    
    for idx, match in matches_df.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        
        home_data = rankings_lookup.get(home_team, {})
        away_data = rankings_lookup.get(away_team, {})
        
        # Skip matches where teams aren't ranked
        if not home_data or not away_data:
            continue
        
        # Calculate prediction
        try:
            prediction = pagerank_rater.predict_match(home_team, away_team)
            
            # Create feature row
            feature = {
                'match_id': idx,
                'home_team': home_team,
                'away_team': away_team,
                'home_pagerank': home_data.get('pagerank_score', 0),
                'away_pagerank': away_data.get('pagerank_score', 0),
                'home_pagerank_rank': home_data.get('rank', 0),
                'away_pagerank_rank': away_data.get('rank', 0),
                'home_pagerank_normalized': home_data.get('pagerank_normalized', 0),
                'away_pagerank_normalized': away_data.get('pagerank_normalized', 0),
                'home_in_degree': home_data.get('in_degree', 0),
                'home_out_degree': home_data.get('out_degree', 0),
                'away_in_degree': away_data.get('in_degree', 0),
                'away_out_degree': away_data.get('out_degree', 0),
                'p_home_win': prediction['prediction']['p_home_win'],
                'p_draw': prediction['prediction']['p_draw'],
                'p_away_win': prediction['prediction']['p_away_win'],
                'expected_goal_diff': prediction['prediction']['expected_goal_diff']
            }
            
            features.append(feature)
        except Exception as e:
            logger.warning(f"Error generating features for match {idx}: {e}")
    
    # Create DataFrame
    if features:
        features_df = pd.DataFrame(features)
    else:
        # Create empty DataFrame with expected columns
        features_df = pd.DataFrame(columns=[
            'match_id', 'home_team', 'away_team', 'home_pagerank', 'away_pagerank',
            'home_pagerank_rank', 'away_pagerank_rank', 'home_pagerank_normalized',
            'away_pagerank_normalized', 'home_in_degree', 'home_out_degree',
            'away_in_degree', 'away_out_degree', 'p_home_win', 'p_draw', 'p_away_win',
            'expected_goal_diff'
        ])
    
    return features_df 