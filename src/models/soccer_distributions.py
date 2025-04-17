"""
Soccer Distribution Models Module

Implements specialized probabilistic models for soccer match prediction:
- Dixon-Coles modified Poisson model
- Bayesian hierarchical models
- Dynamic team strength models

These models were selected based on academic research showing their
effectiveness in soccer prediction tasks.
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
import statsmodels.api as sm
import joblib
from pathlib import Path

# Import project components
from src.utils.logger import get_logger

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.soccer_distributions")

# Define paths
MODELS_DIR = os.path.join(DATA_DIR, "models")
DISTRIBUTION_MODELS_DIR = os.path.join(MODELS_DIR, "distributions")
os.makedirs(DISTRIBUTION_MODELS_DIR, exist_ok=True)


class DixonColesModel:
    """
    Implementation of the Dixon-Coles model for soccer prediction.
    
    This model extends the basic Poisson model with:
    1. Team-specific attack and defense parameters
    2. Home advantage parameter
    3. Low-score correction factor for better handling of 0-0, 1-0, 0-1, 1-1 results
    
    Reference: Dixon, M.J., Coles, S.G. (1997) Modelling Association Football Scores 
               and Inefficiencies in the Football Betting Market.
    """
    
    def __init__(self, teams: Optional[List[Any]] = None):
        """
        Initialize the Dixon-Coles model.
        
        Args:
            teams: List of team identifiers
        """
        self.teams = teams or []
        self.team_params = {}
        self.home_advantage = 0.0
        self.rho = 0.0  # Low-score correction factor
        self.fitted = False
        self.model_info = {
            "model_type": "dixon_coles",
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {}
        }
    
    def _tau(self, home_goals: int, away_goals: int, 
            home_expect: float, away_expect: float) -> float:
        """
        Calculate the low-score correction factor tau.
        
        Args:
            home_goals: Number of goals scored by home team
            away_goals: Number of goals scored by away team
            home_expect: Expected number of goals by home team
            away_expect: Expected number of goals by away team
            
        Returns:
            float: The correction factor value
        """
        if home_goals == 0 and away_goals == 0:
            return 1.0 - self.rho * home_expect * away_expect
        elif home_goals == 0 and away_goals == 1:
            return 1.0 + self.rho * home_expect
        elif home_goals == 1 and away_goals == 0:
            return 1.0 + self.rho * away_expect
        elif home_goals == 1 and away_goals == 1:
            return 1.0 - self.rho
        else:
            return 1.0
    
    def _poisson_probability(self, goals: int, expected_goals: float) -> float:
        """
        Calculate Poisson probability for a given number of goals.
        
        Args:
            goals: Actual number of goals
            expected_goals: Expected number of goals
            
        Returns:
            float: Poisson probability
        """
        return np.exp(-expected_goals) * np.power(expected_goals, goals) / np.math.factorial(goals)
    
    def _match_probability(self, home_team: Any, away_team: Any, 
                          home_goals: int, away_goals: int) -> float:
        """
        Calculate the probability of a specific match outcome.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            home_goals: Number of goals scored by home team
            away_goals: Number of goals scored by away team
            
        Returns:
            float: Probability of the given outcome
        """
        if home_team not in self.team_params or away_team not in self.team_params:
            return 0.0
        
        # Get team parameters
        attack_home = self.team_params[home_team]['attack']
        defense_home = self.team_params[home_team]['defense']
        attack_away = self.team_params[away_team]['attack']
        defense_away = self.team_params[away_team]['defense']
        
        # Calculate expected goals
        home_expected = np.exp(attack_home + defense_away + self.home_advantage)
        away_expected = np.exp(attack_away + defense_home)
        
        # Apply Dixon-Coles correction
        home_prob = self._poisson_probability(home_goals, home_expected)
        away_prob = self._poisson_probability(away_goals, away_expected)
        correction = self._tau(home_goals, away_goals, home_expected, away_expected)
        
        return home_prob * away_prob * correction
    
    def _negative_log_likelihood(self, params: np.ndarray, 
                                matches: pd.DataFrame, teams: List[Any]) -> float:
        """
        Calculate negative log-likelihood for parameter optimization.
        
        Args:
            params: Model parameters (attack, defense for each team, home advantage, rho)
            matches: DataFrame containing match data
            teams: List of team identifiers
            
        Returns:
            float: Negative log-likelihood
        """
        # Number of teams
        n_teams = len(teams)
        
        # Extract parameters
        self.home_advantage = params[0]
        self.rho = params[1]
        
        # Initialize team parameters
        self.team_params = {}
        
        # Set parameters for all teams except the reference team
        for i, team in enumerate(teams[:-1]):
            self.team_params[team] = {
                'attack': params[2 + i],
                'defense': params[2 + n_teams - 1 + i]
            }
        
        # Set parameters for the reference team (constraints for identifiability)
        # The reference team's attack and defense are set to balance the sum
        reference_team = teams[-1]
        attack_sum = sum(self.team_params[team]['attack'] for team in teams[:-1])
        defense_sum = sum(self.team_params[team]['defense'] for team in teams[:-1])
        
        self.team_params[reference_team] = {
            'attack': -attack_sum,
            'defense': -defense_sum
        }
        
        # Calculate log-likelihood
        log_likelihood = 0.0
        
        for _, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            prob = self._match_probability(home_team, away_team, home_goals, away_goals)
            
            # Add to log-likelihood, handling potential numerical issues
            if prob > 0:
                log_likelihood += np.log(prob)
            else:
                log_likelihood -= 100  # Penalty for impossible outcome
        
        return -log_likelihood
    
    def fit(self, matches_df: pd.DataFrame, max_iter: int = 100, 
           tol: float = 1e-6) -> Dict[str, Any]:
        """
        Fit the Dixon-Coles model to match data.
        
        Args:
            matches_df: DataFrame containing match data with home_team, away_team, home_goals, away_goals
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dict: Fitting results
        """
        # Check required columns
        required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals']
        
        # If column names are different, try to map them
        col_mapping = {}
        
        if not all(col in matches_df.columns for col in required_cols):
            # Try to map columns automatically
            possible_mappings = {
                'home_team': ['home_club_id', 'home_team_id', 'home_id'],
                'away_team': ['away_club_id', 'away_team_id', 'away_id'],
                'home_goals': ['home_club_goals', 'home_score', 'home_team_goals'],
                'away_goals': ['away_club_goals', 'away_score', 'away_team_goals']
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
        
        # Get all unique teams
        home_teams = matches['home_team'].unique()
        away_teams = matches['away_team'].unique()
        self.teams = list(np.unique(np.concatenate([home_teams, away_teams])))
        n_teams = len(self.teams)
        
        # Initialize parameters
        # [home_advantage, rho, attack_1, attack_2, ..., defense_1, defense_2, ...]
        initial_params = np.zeros(2 + 2 * (n_teams - 1))
        initial_params[0] = 0.3  # Initial home advantage
        initial_params[1] = 0.1  # Initial rho
        
        # Bounds for parameters
        bounds = [(0.0, 1.0), (-0.2, 0.2)]  # home_advantage, rho
        bounds += [(-2.0, 2.0)] * (2 * (n_teams - 1))  # attack and defense parameters
        
        # Perform optimization
        try:
            result = optimize.minimize(
                self._negative_log_likelihood,
                initial_params,
                args=(matches, self.teams),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': tol}
            )
            
            # Check if optimization was successful
            if result.success:
                # Update parameters
                self._negative_log_likelihood(result.x, matches, self.teams)
                
                # Mark model as fitted
                self.fitted = True
                self.model_info["trained"] = True
                self.model_info["training_date"] = datetime.now().isoformat()
                self.model_info["num_teams"] = n_teams
                self.model_info["num_matches"] = len(matches)
                self.model_info["optimization_result"] = {
                    "success": result.success,
                    "negative_log_likelihood": result.fun,
                    "num_iterations": result.nit
                }
                
                logger.info(f"Dixon-Coles model successfully fitted with {n_teams} teams and {len(matches)} matches")
                
                return {
                    "success": True,
                    "message": "Model training successful",
                    "negative_log_likelihood": result.fun,
                    "home_advantage": self.home_advantage,
                    "rho": self.rho
                }
            else:
                logger.warning(f"Dixon-Coles model optimization failed: {result.message}")
                return {
                    "success": False,
                    "message": f"Optimization failed: {result.message}"
                }
        
        except Exception as e:
            logger.error(f"Error fitting Dixon-Coles model: {e}")
            return {
                "success": False,
                "message": f"Error fitting model: {str(e)}"
            }
    
    def predict_score_probabilities(self, home_team: Any, away_team: Any, 
                                   max_goals: int = 10) -> np.ndarray:
        """
        Predict the probability distribution of scores.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            max_goals: Maximum number of goals to consider
            
        Returns:
            np.ndarray: 2D array of probabilities for each score
        """
        if not self.fitted:
            logger.warning("Model not fitted yet. Call fit() first.")
            return np.zeros((max_goals + 1, max_goals + 1))
        
        # Check if teams exist in model
        if home_team not in self.team_params or away_team not in self.team_params:
            logger.warning(f"Teams {home_team} and/or {away_team} not in trained model")
            return np.zeros((max_goals + 1, max_goals + 1))
        
        # Initialize probability matrix
        score_probs = np.zeros((max_goals + 1, max_goals + 1))
        
        # Calculate all score probabilities
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                score_probs[i, j] = self._match_probability(home_team, away_team, i, j)
        
        # Normalize to ensure probabilities sum to 1
        score_probs = score_probs / score_probs.sum()
        
        return score_probs
    
    def predict_match_outcome(self, home_team: Any, away_team: Any) -> Dict[str, float]:
        """
        Predict the probability of match outcomes (home win, draw, away win).
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            
        Returns:
            Dict: Probabilities for each outcome
        """
        if not self.fitted:
            logger.warning("Model not fitted yet. Call fit() first.")
            return {"home_win": 0.33, "draw": 0.33, "away_win": 0.33}
        
        # Get score probabilities
        score_probs = self.predict_score_probabilities(home_team, away_team)
        
        # Calculate outcome probabilities
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for i in range(score_probs.shape[0]):
            for j in range(score_probs.shape[1]):
                if i > j:
                    home_win += score_probs[i, j]
                elif i == j:
                    draw += score_probs[i, j]
                else:
                    away_win += score_probs[i, j]
        
        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win
        }
    
    def get_team_ratings(self) -> Dict[Any, Dict[str, float]]:
        """
        Get attack and defense ratings for all teams.
        
        Returns:
            Dict: Team ratings
        """
        if not self.fitted:
            logger.warning("Model not fitted yet. Call fit() first.")
            return {}
        
        ratings = {}
        
        for team, params in self.team_params.items():
            attack = params['attack']
            defense = params['defense']
            
            # Higher values are better for attack, lower values are better for defense
            # We'll convert to a standardized rating format
            ratings[team] = {
                'attack': attack,
                'defense': -defense,  # Negate so higher values are better
                'overall': attack - defense  # Overall rating as attack minus defense
            }
        
        return ratings
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Optional filepath to save the model
            
        Returns:
            str: Path where the model was saved
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(DISTRIBUTION_MODELS_DIR, f"dixon_coles_{timestamp}.joblib")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model_data = {
            'teams': self.teams,
            'team_params': self.team_params,
            'home_advantage': self.home_advantage,
            'rho': self.rho,
            'fitted': self.fitted,
            'model_info': self.model_info
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "DixonColesModel":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            DixonColesModel: Loaded model
        """
        model_data = joblib.load(filepath)
        
        model = cls(teams=model_data['teams'])
        model.team_params = model_data['team_params']
        model.home_advantage = model_data['home_advantage']
        model.rho = model_data['rho']
        model.fitted = model_data['fitted']
        model.model_info = model_data['model_info']
        
        logger.info(f"Model loaded from {filepath}")
        
        return model


def train_dixon_coles_model(matches_df: pd.DataFrame, 
                            match_weight_days: Optional[int] = None) -> DixonColesModel:
    """
    Train a Dixon-Coles model with optional time-weighting of matches.
    
    Args:
        matches_df: DataFrame containing match data
        match_weight_days: Optional half-life in days for time-weighting matches
        
    Returns:
        DixonColesModel: Trained Dixon-Coles model
    """
    # Create a copy of the dataframe
    matches = matches_df.copy()
    
    # Apply time-weighting if specified
    if match_weight_days is not None:
        # Ensure date column is datetime
        if 'date' in matches.columns:
            if not pd.api.types.is_datetime64_dtype(matches['date']):
                matches['date'] = pd.to_datetime(matches['date'])
            
            # Calculate match weights based on recency
            now = datetime.now()
            matches['days_ago'] = (now - matches['date']).dt.days
            
            # Calculate decay factor
            decay_lambda = np.log(2) / match_weight_days
            matches['weight'] = np.exp(-decay_lambda * matches['days_ago'])
            
            # Replicate matches according to weights
            weighted_matches = []
            for _, match in matches.iterrows():
                weight = match['weight']
                repeats = max(1, int(np.round(weight * 10)))  # Scale factor for reasonable repetition
                
                for _ in range(repeats):
                    weighted_matches.append(match)
            
            matches = pd.DataFrame(weighted_matches)
    
    # Initialize and train the model
    model = DixonColesModel()
    model.fit(matches)
    
    return model


def calculate_proper_scoring_rule(predictions: pd.DataFrame, 
                                  outcomes: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate proper scoring rules (Ranked Probability Score, Brier Score) for predictions.
    
    Args:
        predictions: DataFrame with prediction probabilities (home_win, draw, away_win)
        outcomes: DataFrame with actual outcomes (1 for home win, 0 for draw, -1 for away win)
        
    Returns:
        Dict: Scoring metrics
    """
    # Check if predictions and outcomes have the same length
    if len(predictions) != len(outcomes):
        raise ValueError("Predictions and outcomes must have the same length")
    
    # Extract prediction probabilities
    p_home = predictions['home_win'].values
    p_draw = predictions['draw'].values
    p_away = predictions['away_win'].values
    
    # Create one-hot encoding of actual outcomes
    outcomes_values = outcomes.values.flatten()
    y_home = (outcomes_values == 1).astype(int)
    y_draw = (outcomes_values == 0).astype(int)
    y_away = (outcomes_values == -1).astype(int)
    
    # Calculate Brier score components
    brier_home = np.mean((p_home - y_home) ** 2)
    brier_draw = np.mean((p_draw - y_draw) ** 2)
    brier_away = np.mean((p_away - y_away) ** 2)
    
    # Overall Brier score (multiclass)
    brier_score = (brier_home + brier_draw + brier_away) / 3
    
    # Calculate Ranked Probability Score (RPS)
    rps_values = []
    
    for i in range(len(predictions)):
        # Cumulative predictions
        p_cum = np.cumsum([p_home[i], p_draw[i], p_away[i]])
        
        # Cumulative outcomes
        y_cum = np.cumsum([y_home[i], y_draw[i], y_away[i]])
        
        # RPS for this match
        rps = np.sum((p_cum - y_cum) ** 2) / 2
        rps_values.append(rps)
    
    rps_score = np.mean(rps_values)
    
    # Calculate logarithmic score (log loss)
    epsilon = 1e-15  # To avoid log(0)
    log_scores = []
    
    for i in range(len(predictions)):
        if y_home[i] == 1:
            log_scores.append(-np.log(p_home[i] + epsilon))
        elif y_draw[i] == 1:
            log_scores.append(-np.log(p_draw[i] + epsilon))
        else:
            log_scores.append(-np.log(p_away[i] + epsilon))
    
    log_score = np.mean(log_scores)
    
    return {
        "brier_score": brier_score,
        "brier_home": brier_home,
        "brier_draw": brier_draw,
        "brier_away": brier_away,
        "rps": rps_score,
        "log_score": log_score
    }


def predict_with_dixon_coles(model_path: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using a pre-trained Dixon-Coles model.
    
    Args:
        model_path: Path to the saved model
        matches_df: DataFrame with matches to predict
        
    Returns:
        pd.DataFrame: DataFrame with match predictions
    """
    # Load the model
    model = DixonColesModel.load(model_path)
    
    # Prepare predictions dataframe
    predictions = []
    
    # Determine column names
    home_team_col = 'home_team'
    away_team_col = 'away_team'
    
    # Try to map columns automatically if standard names not found
    if home_team_col not in matches_df.columns:
        for col in ['home_club_id', 'home_team_id', 'home_id']:
            if col in matches_df.columns:
                home_team_col = col
                break
    
    if away_team_col not in matches_df.columns:
        for col in ['away_club_id', 'away_team_id', 'away_id']:
            if col in matches_df.columns:
                away_team_col = col
                break
    
    # Copy key columns
    id_cols = []
    for col in ['match_id', 'game_id', 'id', 'date']:
        if col in matches_df.columns:
            id_cols.append(col)
    
    # Make predictions for each match
    for _, match in matches_df.iterrows():
        home_team = match[home_team_col]
        away_team = match[away_team_col]
        
        # Skip if teams not in model
        if home_team not in model.team_params or away_team not in model.team_params:
            continue
        
        # Get prediction
        outcome_probs = model.predict_match_outcome(home_team, away_team)
        
        # Calculate most likely score
        score_probs = model.predict_score_probabilities(home_team, away_team)
        max_score_idx = np.unravel_index(np.argmax(score_probs), score_probs.shape)
        most_likely_score = f"{max_score_idx[0]}-{max_score_idx[1]}"
        
        # Create prediction dict
        pred_dict = {
            home_team_col: home_team,
            away_team_col: away_team,
            'home_win_prob': outcome_probs['home_win'],
            'draw_prob': outcome_probs['draw'],
            'away_win_prob': outcome_probs['away_win'],
            'most_likely_score': most_likely_score,
            'dc_home_exp_goals': np.sum([i * np.sum(score_probs[i,:]) for i in range(score_probs.shape[0])]),
            'dc_away_exp_goals': np.sum([j * np.sum(score_probs[:,j]) for j in range(score_probs.shape[1])])
        }
        
        # Add ID columns
        for col in id_cols:
            pred_dict[col] = match[col]
        
        predictions.append(pred_dict)
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    return predictions_df


def create_value_bets(predictions_df: pd.DataFrame, 
                     odds_df: pd.DataFrame,
                     min_edge: float = 0.05,
                     stake: float = 1.0) -> pd.DataFrame:
    """
    Create value bets by comparing model predictions with bookmaker odds.
    
    Args:
        predictions_df: DataFrame with model predictions
        odds_df: DataFrame with bookmaker odds
        min_edge: Minimum edge (difference between fair odds and bookmaker odds)
        stake: Stake amount per bet
        
    Returns:
        pd.DataFrame: DataFrame with value betting opportunities
    """
    # Check if DataFrames have matching IDs
    common_id = None
    for id_col in ['match_id', 'game_id', 'id']:
        if id_col in predictions_df.columns and id_col in odds_df.columns:
            common_id = id_col
            break
    
    if common_id is None:
        logger.warning("No common ID column found between predictions and odds")
        return pd.DataFrame()
    
    # Determine odds columns
    home_odds_col = None
    draw_odds_col = None
    away_odds_col = None
    
    for col in odds_df.columns:
        if 'home' in col.lower() and 'odds' in col.lower():
            home_odds_col = col
        elif 'draw' in col.lower() and 'odds' in col.lower():
            draw_odds_col = col
        elif 'away' in col.lower() and 'odds' in col.lower():
            away_odds_col = col
    
    if home_odds_col is None or draw_odds_col is None or away_odds_col is None:
        logger.warning("Could not identify odds columns")
        return pd.DataFrame()
    
    # Create merged DataFrame
    merged = pd.merge(
        predictions_df,
        odds_df[[common_id, home_odds_col, draw_odds_col, away_odds_col]],
        on=common_id
    )
    
    # Calculate implied probabilities from odds
    merged['implied_home_prob'] = 1 / merged[home_odds_col]
    merged['implied_draw_prob'] = 1 / merged[draw_odds_col]
    merged['implied_away_prob'] = 1 / merged[away_odds_col]
    
    # Calculate edges
    merged['home_edge'] = merged['home_win_prob'] - merged['implied_home_prob']
    merged['draw_edge'] = merged['draw_prob'] - merged['implied_draw_prob']
    merged['away_edge'] = merged['away_win_prob'] - merged['implied_away_prob']
    
    # Calculate expected value
    merged['home_ev'] = merged['home_win_prob'] * (merged[home_odds_col] - 1) - (1 - merged['home_win_prob'])
    merged['draw_ev'] = merged['draw_prob'] * (merged[draw_odds_col] - 1) - (1 - merged['draw_prob'])
    merged['away_ev'] = merged['away_win_prob'] * (merged[away_odds_col] - 1) - (1 - merged['away_win_prob'])
    
    # Select bets with positive edge and EV
    value_bets = merged[
        ((merged['home_edge'] > min_edge) & (merged['home_ev'] > 0)) |
        ((merged['draw_edge'] > min_edge) & (merged['draw_ev'] > 0)) |
        ((merged['away_edge'] > min_edge) & (merged['away_ev'] > 0))
    ].copy()
    
    # Determine best bet for each match
    value_bets['best_edge'] = value_bets[['home_edge', 'draw_edge', 'away_edge']].max(axis=1)
    value_bets['best_ev'] = value_bets[['home_ev', 'draw_ev', 'away_ev']].max(axis=1)
    
    # Determine bet type and odds
    conditions = [
        value_bets['home_edge'] == value_bets['best_edge'],
        value_bets['draw_edge'] == value_bets['best_edge'],
        value_bets['away_edge'] == value_bets['best_edge']
    ]
    choices = ['Home', 'Draw', 'Away']
    value_bets['bet_type'] = np.select(conditions, choices, default='None')
    
    # Get corresponding odds
    value_bets['bet_odds'] = np.select(
        conditions,
        [value_bets[home_odds_col], value_bets[draw_odds_col], value_bets[away_odds_col]],
        default=0
    )
    
    # Calculate stake and potential profit
    value_bets['stake'] = stake
    value_bets['potential_profit'] = stake * (value_bets['bet_odds'] - 1)
    
    # Sort by expected value
    value_bets = value_bets.sort_values('best_ev', ascending=False)
    
    return value_bets


class BivariatePoissonModel:
    """
    Implementation of the Bivariate Poisson model for soccer prediction.
    
    This model extends basic Poisson models by accounting for correlation between
    home and away team scores, which is particularly important in soccer where
    team strategies are interdependent.
    
    Reference: Karlis, D., Ntzoufras, I. (2003) Analysis of sports data by using 
               bivariate Poisson models. Journal of the Royal Statistical Society: 
               Series D (The Statistician)
    """
    
    def __init__(self, teams: Optional[List[Any]] = None):
        """
        Initialize the Bivariate Poisson model.
        
        Args:
            teams: List of team identifiers
        """
        self.teams = teams or []
        self.team_params = {}
        self.home_advantage = 0.0
        self.covariance = 0.0  # Correlation parameter between scores
        self.fitted = False
        self.model_info = {
            "model_type": "bivariate_poisson",
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {}
        }
    
    def _bivariate_poisson_pmf(self, x: int, y: int, lambda1: float, lambda2: float, lambda3: float) -> float:
        """
        Calculate the probability mass function for the bivariate Poisson distribution.
        
        Args:
            x: First count variable (home goals)
            y: Second count variable (away goals)
            lambda1: Rate parameter for first Poisson process
            lambda2: Rate parameter for second Poisson process
            lambda3: Rate parameter for the common Poisson process (covariance term)
            
        Returns:
            float: Probability of observing (x, y)
        """
        probability = 0.0
        
        # Calculate sum term for the bivariate PMF
        for i in range(min(x, y) + 1):
            term1 = np.exp(-(lambda1 + lambda2 + lambda3))
            term2 = (lambda1 ** (x - i)) / np.math.factorial(x - i)
            term3 = (lambda2 ** (y - i)) / np.math.factorial(y - i)
            term4 = (lambda3 ** i) / np.math.factorial(i)
            probability += term1 * term2 * term3 * term4 * np.math.factorial(i)
        
        return probability
    
    def _match_probability(self, home_team: Any, away_team: Any, 
                          home_goals: int, away_goals: int) -> float:
        """
        Calculate the probability of a specific match outcome.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            home_goals: Number of goals scored by home team
            away_goals: Number of goals scored by away team
            
        Returns:
            float: Probability of the given outcome
        """
        if home_team not in self.team_params or away_team not in self.team_params:
            return 0.0
        
        # Get team parameters
        attack_home = self.team_params[home_team]['attack']
        defense_home = self.team_params[home_team]['defense']
        attack_away = self.team_params[away_team]['attack']
        defense_away = self.team_params[away_team]['defense']
        
        # Calculate lambda parameters for bivariate Poisson
        lambda1 = np.exp(attack_home + defense_away + self.home_advantage)  # Home team scoring rate
        lambda2 = np.exp(attack_away + defense_home)  # Away team scoring rate
        lambda3 = self.covariance  # Covariance term (common rate parameter)
        
        # Calculate bivariate Poisson probability
        return self._bivariate_poisson_pmf(home_goals, away_goals, lambda1, lambda2, lambda3)
    
    def _negative_log_likelihood(self, params: np.ndarray, 
                                matches: pd.DataFrame, teams: List[Any]) -> float:
        """
        Calculate negative log-likelihood for parameter optimization.
        
        Args:
            params: Model parameters (attack, defense for each team, home advantage, covariance)
            matches: DataFrame containing match data
            teams: List of team identifiers
            
        Returns:
            float: Negative log-likelihood
        """
        # Number of teams
        n_teams = len(teams)
        
        # Extract parameters
        self.home_advantage = params[0]
        self.covariance = max(0, params[1])  # Covariance must be non-negative
        
        # Initialize team parameters
        self.team_params = {}
        
        # Set parameters for all teams except the reference team
        for i, team in enumerate(teams[:-1]):
            self.team_params[team] = {
                'attack': params[2 + i],
                'defense': params[2 + n_teams - 1 + i]
            }
        
        # Set parameters for the reference team (constraints for identifiability)
        reference_team = teams[-1]
        attack_sum = sum(self.team_params[team]['attack'] for team in teams[:-1])
        defense_sum = sum(self.team_params[team]['defense'] for team in teams[:-1])
        
        self.team_params[reference_team] = {
            'attack': -attack_sum,
            'defense': -defense_sum
        }
        
        # Calculate log-likelihood
        log_likelihood = 0.0
        
        for _, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            prob = self._match_probability(home_team, away_team, home_goals, away_goals)
            
            # Add to log-likelihood, handling potential numerical issues
            if prob > 0:
                log_likelihood += np.log(prob)
            else:
                log_likelihood -= 100  # Penalty for impossible outcome
        
        return -log_likelihood
    
    def fit(self, matches_df: pd.DataFrame, max_iter: int = 100, 
           tol: float = 1e-6) -> Dict[str, Any]:
        """
        Fit the Bivariate Poisson model to match data.
        
        Args:
            matches_df: DataFrame containing match data with home_team, away_team, home_goals, away_goals
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dict: Fitting results
        """
        # Check required columns
        required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals']
        
        # If column names are different, try to map them
        col_mapping = {}
        
        if not all(col in matches_df.columns for col in required_cols):
            # Try to map columns automatically
            possible_mappings = {
                'home_team': ['home_club_id', 'home_team_id', 'home_id'],
                'away_team': ['away_club_id', 'away_team_id', 'away_id'],
                'home_goals': ['home_club_goals', 'home_score', 'home_team_goals'],
                'away_goals': ['away_club_goals', 'away_score', 'away_team_goals']
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
        
        # Get all unique teams
        home_teams = matches['home_team'].unique()
        away_teams = matches['away_team'].unique()
        self.teams = list(np.unique(np.concatenate([home_teams, away_teams])))
        n_teams = len(self.teams)
        
        logger.info(f"Fitting Bivariate Poisson model with {n_teams} teams and {len(matches)} matches")
        
        # Initial parameter values
        # Format: [home_advantage, covariance, attack_1, ..., attack_{n-1}, defense_1, ..., defense_{n-1}]
        initial_params = np.zeros(2 + 2 * (n_teams - 1))
        initial_params[0] = 0.3  # Initial home advantage
        initial_params[1] = 0.1  # Initial covariance
        
        # Set bounds for parameters
        param_bounds = [(0.0, 1.0),   # home_advantage
                        (0.0, 0.5)]   # covariance (non-negative)
        
        # Add bounds for team parameters
        for _ in range(2 * (n_teams - 1)):
            param_bounds.append((-3.0, 3.0))  # Reasonable bounds for attack/defense parameters
            
        # Minimize negative log-likelihood
        result = optimize.minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            args=(matches, self.teams),
            method='L-BFGS-B',
            bounds=param_bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )
        
        # Update model info
        self.fitted = True
        self.model_info.update({
            "trained": True,
            "convergence": result.success,
            "log_likelihood": -result.fun,
            "n_iterations": result.nit,
            "trained_at": datetime.now().isoformat(),
            "n_matches": len(matches),
            "n_teams": n_teams
        })
        
        logger.info(f"Bivariate Poisson model fitting completed: {result.success}")
        
        return self.model_info
    
    def predict_score_probabilities(self, home_team: Any, away_team: Any, 
                                   max_goals: int = 10) -> np.ndarray:
        """
        Predict probabilities for all possible score combinations up to max_goals.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            max_goals: Maximum number of goals to consider
            
        Returns:
            np.ndarray: 2D array of probabilities for each score combination
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if home_team not in self.team_params or away_team not in self.team_params:
            raise ValueError(f"Team not in model: {home_team if home_team not in self.team_params else away_team}")
        
        # Initialize probability matrix
        score_probs = np.zeros((max_goals + 1, max_goals + 1))
        
        # Get team parameters
        attack_home = self.team_params[home_team]['attack']
        defense_home = self.team_params[home_team]['defense']
        attack_away = self.team_params[away_team]['attack']
        defense_away = self.team_params[away_team]['defense']
        
        # Calculate lambda parameters
        lambda1 = np.exp(attack_home + defense_away + self.home_advantage)
        lambda2 = np.exp(attack_away + defense_home)
        lambda3 = self.covariance
        
        # Calculate probabilities for each score combination
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                score_probs[i, j] = self._bivariate_poisson_pmf(i, j, lambda1, lambda2, lambda3)
        
        return score_probs
    
    def predict_match_outcome(self, home_team: Any, away_team: Any) -> Dict[str, float]:
        """
        Predict the outcome of a match (home win, draw, away win).
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            
        Returns:
            Dict: Probabilities for each outcome
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get score probabilities
        score_probs = self.predict_score_probabilities(home_team, away_team)
        
        # Calculate outcome probabilities
        n_goals = score_probs.shape[0]
        
        home_win_prob = np.sum(score_probs[np.triu_indices(n_goals, 1)])  # Upper triangle excluding diagonal
        draw_prob = np.sum(np.diag(score_probs))  # Diagonal elements
        away_win_prob = np.sum(score_probs[np.tril_indices(n_goals, -1)])  # Lower triangle excluding diagonal
        
        return {
            "home_win": float(home_win_prob),
            "draw": float(draw_prob),
            "away_win": float(away_win_prob)
        }
    
    def get_team_ratings(self) -> Dict[Any, Dict[str, float]]:
        """
        Get the attack and defense ratings for all teams.
        
        Returns:
            Dict: Team ratings dictionary
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting ratings")
        
        ratings = {}
        for team in self.teams:
            attack = self.team_params[team]['attack']
            defense = self.team_params[team]['defense']
            
            # Calculate an overall rating as a combination of attack and defense
            overall = np.exp(attack) - np.exp(defense)
            
            ratings[team] = {
                "attack": float(np.exp(attack)),
                "defense": float(np.exp(defense)),
                "overall": float(overall)
            }
        
        return ratings
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            str: Path to the saved model
        """
        if not self.fitted:
            logger.warning("Saving an unfitted model")
        
        # Default path if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                DISTRIBUTION_MODELS_DIR, 
                f"bivariate_poisson_{timestamp}.pkl"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, "wb") as f:
            joblib.dump(self, f)
        
        logger.info(f"Bivariate Poisson model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "BivariatePoissonModel":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            BivariatePoissonModel: Loaded model
        """
        with open(filepath, "rb") as f:
            model = joblib.load(f)
        
        logger.info(f"Bivariate Poisson model loaded from {filepath}")
        
        return model


def train_bivariate_poisson_model(matches_df: pd.DataFrame, 
                              match_weight_days: Optional[int] = None) -> BivariatePoissonModel:
    """
    Train a Bivariate Poisson model on match data.
    
    Args:
        matches_df: DataFrame containing match data
        match_weight_days: Optional parameter for weighting recent matches more heavily
        
    Returns:
        BivariatePoissonModel: Trained model
    """
    # Initialize model
    model = BivariatePoissonModel()
    
    # Apply time weighting if requested
    if match_weight_days is not None and 'date' in matches_df.columns:
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_dtype(matches_df['date']):
            matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        # Calculate match weights based on recency
        latest_date = matches_df['date'].max()
        matches_df['days_old'] = (latest_date - matches_df['date']).dt.days
        matches_df['weight'] = np.exp(-matches_df['days_old'] / match_weight_days)
        
        # We need to implement weighted fitting for the bivariate Poisson model
        # This would require modifying the _negative_log_likelihood method
        # For now, we'll filter to more recent matches
        cutoff_date = latest_date - pd.Timedelta(days=match_weight_days * 3)
        filtered_matches = matches_df[matches_df['date'] >= cutoff_date].copy()
        
        logger.info(f"Filtered to {len(filtered_matches)} matches out of {len(matches_df)} based on recency")
        
        # Fit model
        model.fit(filtered_matches)
    else:
        # Fit model without time weighting
        model.fit(matches_df)
    
    return model


class DoubleWeibullModel:
    """
    Implementation of the Double Weibull model for soccer prediction.
    
    This model uses Weibull distributions instead of Poisson for modeling goal counts,
    which can better capture the heavy-tailed nature of goal distributions and
    has been shown to outperform Poisson-based models in some contexts.
    
    Reference: Hubáček et al. (2022) evaluated models using the Open International 
               Soccer Database v2, with Double Weibull models showing strong performance.
    """
    
    def __init__(self, teams: Optional[List[Any]] = None):
        """
        Initialize the Double Weibull model.
        
        Args:
            teams: List of team identifiers
        """
        self.teams = teams or []
        self.team_params = {}
        self.home_advantage = 0.0
        self.shape_home = 1.5  # Weibull shape parameter for home teams
        self.shape_away = 1.5  # Weibull shape parameter for away teams
        self.fitted = False
        self.model_info = {
            "model_type": "double_weibull",
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {}
        }
    
    def _weibull_pmf(self, k: int, scale: float, shape: float) -> float:
        """
        Calculate the Weibull PMF for integer values (discrete approximation).
        
        Args:
            k: Number of goals
            scale: Weibull scale parameter
            shape: Weibull shape parameter
            
        Returns:
            float: Probability of observing k goals
        """
        # For k=0, calculate probability of x < 0.5
        if k == 0:
            return 1 - np.exp(-(0.5 / scale) ** shape)
        
        # For k>0, calculate probability between k-0.5 and k+0.5
        cdf_lower = 1 - np.exp(-((k - 0.5) / scale) ** shape)
        cdf_upper = 1 - np.exp(-((k + 0.5) / scale) ** shape)
        
        return cdf_upper - cdf_lower
    
    def _match_probability(self, home_team: Any, away_team: Any, 
                          home_goals: int, away_goals: int) -> float:
        """
        Calculate the probability of a specific match outcome.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            home_goals: Number of goals scored by home team
            away_goals: Number of goals scored by away team
            
        Returns:
            float: Probability of the given outcome
        """
        if home_team not in self.team_params or away_team not in self.team_params:
            return 0.0
        
        # Get team parameters
        attack_home = self.team_params[home_team]['attack']
        defense_home = self.team_params[home_team]['defense']
        attack_away = self.team_params[away_team]['attack']
        defense_away = self.team_params[away_team]['defense']
        
        # Calculate scale parameters for Weibull distributions
        # Scale is analogous to lambda in Poisson, representing expected goals
        home_scale = np.exp(attack_home + defense_away + self.home_advantage)
        away_scale = np.exp(attack_away + defense_home)
        
        # Calculate Weibull PMFs for home and away goals
        home_prob = self._weibull_pmf(home_goals, home_scale, self.shape_home)
        away_prob = self._weibull_pmf(away_goals, away_scale, self.shape_away)
        
        # Assuming independence between home and away goals
        return home_prob * away_prob
    
    def _negative_log_likelihood(self, params: np.ndarray, 
                                matches: pd.DataFrame, teams: List[Any]) -> float:
        """
        Calculate negative log-likelihood for parameter optimization.
        
        Args:
            params: Model parameters (attack, defense for each team, home advantage, shapes)
            matches: DataFrame containing match data
            teams: List of team identifiers
            
        Returns:
            float: Negative log-likelihood
        """
        # Number of teams
        n_teams = len(teams)
        
        # Extract parameters
        self.home_advantage = params[0]
        self.shape_home = max(1.0, params[1])  # Shape must be positive, typically > 1
        self.shape_away = max(1.0, params[2])
        
        # Initialize team parameters
        self.team_params = {}
        
        # Set parameters for all teams except the reference team
        for i, team in enumerate(teams[:-1]):
            self.team_params[team] = {
                'attack': params[3 + i],
                'defense': params[3 + n_teams - 1 + i]
            }
        
        # Set parameters for the reference team (constraints for identifiability)
        reference_team = teams[-1]
        attack_sum = sum(self.team_params[team]['attack'] for team in teams[:-1])
        defense_sum = sum(self.team_params[team]['defense'] for team in teams[:-1])
        
        self.team_params[reference_team] = {
            'attack': -attack_sum,
            'defense': -defense_sum
        }
        
        # Calculate log-likelihood
        log_likelihood = 0.0
        
        for _, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            
            prob = self._match_probability(home_team, away_team, home_goals, away_goals)
            
            # Add to log-likelihood, handling potential numerical issues
            if prob > 0:
                log_likelihood += np.log(prob)
            else:
                log_likelihood -= 100  # Penalty for impossible outcome
        
        return -log_likelihood
    
    def fit(self, matches_df: pd.DataFrame, max_iter: int = 100, 
           tol: float = 1e-6) -> Dict[str, Any]:
        """
        Fit the Double Weibull model to match data.
        
        Args:
            matches_df: DataFrame containing match data with home_team, away_team, home_goals, away_goals
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Dict: Fitting results
        """
        # Check required columns
        required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals']
        
        # If column names are different, try to map them
        col_mapping = {}
        
        if not all(col in matches_df.columns for col in required_cols):
            # Try to map columns automatically
            possible_mappings = {
                'home_team': ['home_club_id', 'home_team_id', 'home_id'],
                'away_team': ['away_club_id', 'away_team_id', 'away_id'],
                'home_goals': ['home_club_goals', 'home_score', 'home_team_goals'],
                'away_goals': ['away_club_goals', 'away_score', 'away_team_goals']
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
        
        # Get all unique teams
        home_teams = matches['home_team'].unique()
        away_teams = matches['away_team'].unique()
        self.teams = list(np.unique(np.concatenate([home_teams, away_teams])))
        n_teams = len(self.teams)
        
        logger.info(f"Fitting Double Weibull model with {n_teams} teams and {len(matches)} matches")
        
        # Initial parameter values
        # Format: [home_advantage, shape_home, shape_away, attack_1, ..., attack_{n-1}, defense_1, ..., defense_{n-1}]
        initial_params = np.zeros(3 + 2 * (n_teams - 1))
        initial_params[0] = 0.3  # Initial home advantage
        initial_params[1] = 1.5  # Initial shape for home teams
        initial_params[2] = 1.5  # Initial shape for away teams
        
        # Set bounds for parameters
        param_bounds = [
            (0.0, 1.0),    # home_advantage
            (1.0, 3.0),    # shape_home
            (1.0, 3.0)     # shape_away
        ]
        
        # Add bounds for team parameters
        for _ in range(2 * (n_teams - 1)):
            param_bounds.append((-3.0, 3.0))  # Reasonable bounds for attack/defense parameters
            
        # Minimize negative log-likelihood
        result = optimize.minimize(
            fun=self._negative_log_likelihood,
            x0=initial_params,
            args=(matches, self.teams),
            method='L-BFGS-B',
            bounds=param_bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )
        
        # Update model info
        self.fitted = True
        self.model_info.update({
            "trained": True,
            "convergence": result.success,
            "log_likelihood": -result.fun,
            "n_iterations": result.nit,
            "trained_at": datetime.now().isoformat(),
            "n_matches": len(matches),
            "n_teams": n_teams,
            "shape_home": float(self.shape_home),
            "shape_away": float(self.shape_away)
        })
        
        logger.info(f"Double Weibull model fitting completed: {result.success}")
        
        return self.model_info
    
    def predict_score_probabilities(self, home_team: Any, away_team: Any, 
                                   max_goals: int = 10) -> np.ndarray:
        """
        Predict probabilities for all possible score combinations up to max_goals.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            max_goals: Maximum number of goals to consider
            
        Returns:
            np.ndarray: 2D array of probabilities for each score combination
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if home_team not in self.team_params or away_team not in self.team_params:
            raise ValueError(f"Team not in model: {home_team if home_team not in self.team_params else away_team}")
        
        # Initialize probability matrix
        score_probs = np.zeros((max_goals + 1, max_goals + 1))
        
        # Get team parameters
        attack_home = self.team_params[home_team]['attack']
        defense_home = self.team_params[home_team]['defense']
        attack_away = self.team_params[away_team]['attack']
        defense_away = self.team_params[away_team]['defense']
        
        # Calculate scale parameters for Weibull distributions
        home_scale = np.exp(attack_home + defense_away + self.home_advantage)
        away_scale = np.exp(attack_away + defense_home)
        
        # Calculate probabilities for each score combination
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                home_prob = self._weibull_pmf(home_goals, home_scale, self.shape_home)
                away_prob = self._weibull_pmf(away_goals, away_scale, self.shape_away)
                score_probs[home_goals, away_goals] = home_prob * away_prob
        
        # Normalize to ensure probabilities sum to 1
        score_probs = score_probs / np.sum(score_probs)
        
        return score_probs
    
    def predict_match_outcome(self, home_team: Any, away_team: Any) -> Dict[str, float]:
        """
        Predict the outcome of a match (home win, draw, away win).
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            
        Returns:
            Dict: Probabilities for each outcome
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get score probabilities
        score_probs = self.predict_score_probabilities(home_team, away_team)
        
        # Calculate outcome probabilities
        n_goals = score_probs.shape[0]
        
        home_win_prob = np.sum(score_probs[np.triu_indices(n_goals, 1)])  # Upper triangle excluding diagonal
        draw_prob = np.sum(np.diag(score_probs))  # Diagonal elements
        away_win_prob = np.sum(score_probs[np.tril_indices(n_goals, -1)])  # Lower triangle excluding diagonal
        
        return {
            "home_win": float(home_win_prob),
            "draw": float(draw_prob),
            "away_win": float(away_win_prob)
        }
    
    def get_team_ratings(self) -> Dict[Any, Dict[str, float]]:
        """
        Get the attack and defense ratings for all teams.
        
        Returns:
            Dict: Team ratings dictionary
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting ratings")
        
        ratings = {}
        for team in self.teams:
            attack = self.team_params[team]['attack']
            defense = self.team_params[team]['defense']
            
            # Calculate an overall rating as a combination of attack and defense
            overall = np.exp(attack) - np.exp(defense)
            
            ratings[team] = {
                "attack": float(np.exp(attack)),
                "defense": float(np.exp(defense)),
                "overall": float(overall)
            }
        
        return ratings
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            str: Path to the saved model
        """
        if not self.fitted:
            logger.warning("Saving an unfitted model")
        
        # Default path if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                DISTRIBUTION_MODELS_DIR, 
                f"double_weibull_{timestamp}.pkl"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, "wb") as f:
            joblib.dump(self, f)
        
        logger.info(f"Double Weibull model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "DoubleWeibullModel":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            DoubleWeibullModel: Loaded model
        """
        with open(filepath, "rb") as f:
            model = joblib.load(f)
        
        logger.info(f"Double Weibull model loaded from {filepath}")
        
        return model


def train_double_weibull_model(matches_df: pd.DataFrame, 
                            match_weight_days: Optional[int] = None) -> DoubleWeibullModel:
    """
    Train a Double Weibull model on match data.
    
    Args:
        matches_df: DataFrame containing match data
        match_weight_days: Optional parameter for weighting recent matches more heavily
        
    Returns:
        DoubleWeibullModel: Trained model
    """
    # Initialize model
    model = DoubleWeibullModel()
    
    # Apply time weighting if requested
    if match_weight_days is not None and 'date' in matches_df.columns:
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_dtype(matches_df['date']):
            matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        # Calculate match weights based on recency
        latest_date = matches_df['date'].max()
        matches_df['days_old'] = (latest_date - matches_df['date']).dt.days
        matches_df['weight'] = np.exp(-matches_df['days_old'] / match_weight_days)
        
        # Filter to more recent matches as a simple way to handle recency
        cutoff_date = latest_date - pd.Timedelta(days=match_weight_days * 3)
        filtered_matches = matches_df[matches_df['date'] >= cutoff_date].copy()
        
        logger.info(f"Filtered to {len(filtered_matches)} matches out of {len(matches_df)} based on recency")
        
        # Fit model
        model.fit(filtered_matches)
    else:
        # Fit model without time weighting
        model.fit(matches_df)
    
    return model


class ScoreDrivenModel:
    """
    Implementation of a Score-Driven Generalized Autoregressive Score (GAS) model for soccer prediction.
    
    This model captures the time-varying nature of team strengths using score-driven dynamics,
    where team parameters evolve over time based on prediction errors. This approach has been
    shown to be particularly effective for capturing temporal dynamics in soccer.
    
    Reference: Mattera (2023) employed score-driven models to predict binary outcomes in soccer matches,
               achieving high predictive accuracy with the GAS model.
    """
    
    def __init__(self, teams: Optional[List[Any]] = None):
        """
        Initialize the Score-Driven model.
        
        Args:
            teams: List of team identifiers
        """
        self.teams = teams or []
        self.team_params = {}
        self.home_advantage = 0.0
        self.scale_parameter = 1.0  # Scaling factor for score updates
        
        # Dynamic parameters
        self.dynamics_enabled = True
        self.team_dynamics = {}  # Stores time series of team parameters
        self.persistence = 0.97  # Persistence parameter (omega) for GAS dynamics
        self.learning_rate = 0.05  # Learning rate parameter (alpha) for GAS dynamics
        
        self.fitted = False
        self.model_info = {
            "model_type": "score_driven",
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {}
        }
    
    def _initialize_team_parameters(self):
        """Initialize team attack and defense parameters."""
        # Set all team parameters to initial values
        for team in self.teams:
            if team not in self.team_params:
                self.team_params[team] = {
                    'attack': 0.0,  # Log scale
                    'defense': 0.0,  # Log scale
                }
            
            # Initialize dynamics if enabled
            if self.dynamics_enabled and team not in self.team_dynamics:
                self.team_dynamics[team] = {
                    'attack': [],
                    'defense': [],
                    'dates': []
                }
    
    def _poisson_likelihood(self, goals: int, expected_goals: float) -> float:
        """
        Calculate Poisson likelihood for a given number of goals.
        
        Args:
            goals: Actual number of goals
            expected_goals: Expected number of goals
            
        Returns:
            float: Poisson likelihood
        """
        return np.exp(-expected_goals) * np.power(expected_goals, goals) / np.math.factorial(goals)
    
    def _score_function(self, goals: int, expected_goals: float) -> float:
        """
        Calculate the score function for the GAS update.
        
        The score function is the derivative of the log-likelihood with respect
        to the parameter being updated. For Poisson, this is (y - mu) / sigma.
        
        Args:
            goals: Actual number of goals
            expected_goals: Expected number of goals
            
        Returns:
            float: Score function value
        """
        # For Poisson, the score function is (y - mu)
        return goals - expected_goals
    
    def _update_team_parameters(self, match_date: datetime, 
                              home_team: Any, away_team: Any,
                              home_goals: int, away_goals: int):
        """
        Update team parameters based on match outcome using GAS dynamics.
        
        Args:
            match_date: Date of the match
            home_team: Home team identifier
            away_team: Away team identifier
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
        """
        # Get current parameters
        attack_home = self.team_params[home_team]['attack']
        defense_home = self.team_params[home_team]['defense']
        attack_away = self.team_params[away_team]['attack']
        defense_away = self.team_params[away_team]['defense']
        
        # Calculate expected goals
        home_expected = np.exp(attack_home + defense_away + self.home_advantage)
        away_expected = np.exp(attack_away + defense_home)
        
        # Calculate scores (derivatives of log-likelihood)
        home_score = self._score_function(home_goals, home_expected)
        away_score = self._score_function(away_goals, away_expected)
        
        # Update team parameters using GAS dynamics
        # New parameter = omega * old parameter + alpha * score
        
        # Home team attack update
        new_attack_home = (
            self.persistence * attack_home + 
            self.learning_rate * home_score * self.scale_parameter
        )
        
        # Home team defense update
        new_defense_home = (
            self.persistence * defense_home - 
            self.learning_rate * away_score * self.scale_parameter
        )
        
        # Away team attack update
        new_attack_away = (
            self.persistence * attack_away + 
            self.learning_rate * away_score * self.scale_parameter
        )
        
        # Away team defense update
        new_defense_away = (
            self.persistence * defense_away - 
            self.learning_rate * home_score * self.scale_parameter
        )
        
        # Update parameters
        self.team_params[home_team]['attack'] = new_attack_home
        self.team_params[home_team]['defense'] = new_defense_home
        self.team_params[away_team]['attack'] = new_attack_away
        self.team_params[away_team]['defense'] = new_defense_away
        
        # Store dynamics if enabled
        if self.dynamics_enabled:
            # Store current parameters
            self.team_dynamics[home_team]['attack'].append(new_attack_home)
            self.team_dynamics[home_team]['defense'].append(new_defense_home)
            self.team_dynamics[home_team]['dates'].append(match_date)
            
            self.team_dynamics[away_team]['attack'].append(new_attack_away)
            self.team_dynamics[away_team]['defense'].append(new_defense_away)
            self.team_dynamics[away_team]['dates'].append(match_date)
    
    def _match_probability(self, home_team: Any, away_team: Any, 
                        home_goals: int, away_goals: int) -> float:
        """
        Calculate the probability of a specific match outcome.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            home_goals: Number of goals scored by home team
            away_goals: Number of goals scored by away team
            
        Returns:
            float: Probability of the given outcome
        """
        if home_team not in self.team_params or away_team not in self.team_params:
            return 0.0
        
        # Get team parameters
        attack_home = self.team_params[home_team]['attack']
        defense_home = self.team_params[home_team]['defense']
        attack_away = self.team_params[away_team]['attack']
        defense_away = self.team_params[away_team]['defense']
        
        # Calculate expected goals
        home_expected = np.exp(attack_home + defense_away + self.home_advantage)
        away_expected = np.exp(attack_away + defense_home)
        
        # Calculate goal probabilities (independent Poisson)
        home_prob = self._poisson_likelihood(home_goals, home_expected)
        away_prob = self._poisson_likelihood(away_goals, away_expected)
        
        return home_prob * away_prob
    
    def fit(self, matches_df: pd.DataFrame, 
           learning_rate: float = 0.05,
           persistence: float = 0.97,
           home_advantage_init: float = 0.3,
           scale_parameter: float = 1.0,
           store_dynamics: bool = True) -> Dict[str, Any]:
        """
        Fit the Score-Driven model to match data.
        
        This model is fitted sequentially through time, updating parameters
        after each match using score dynamics.
        
        Args:
            matches_df: DataFrame containing match data with home_team, away_team, home_goals, away_goals, date
            learning_rate: Learning rate for parameter updates
            persistence: Persistence parameter for GAS dynamics
            home_advantage_init: Initial home advantage value
            scale_parameter: Scaling factor for score updates
            store_dynamics: Whether to store team parameter time series
            
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
        
        # Get all unique teams
        home_teams = matches['home_team'].unique()
        away_teams = matches['away_team'].unique()
        self.teams = list(np.unique(np.concatenate([home_teams, away_teams])))
        
        # Set model parameters
        self.learning_rate = learning_rate
        self.persistence = persistence
        self.home_advantage = home_advantage_init
        self.scale_parameter = scale_parameter
        self.dynamics_enabled = store_dynamics
        
        # Initialize team parameters
        self._initialize_team_parameters()
        
        logger.info(f"Fitting Score-Driven model with {len(self.teams)} teams and {len(matches)} matches")
        
        # Sequential fitting through time
        log_likelihood = 0.0
        
        for idx, match in matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_goals = match['home_goals']
            away_goals = match['away_goals']
            match_date = match['date']
            
            # Calculate current match probability
            prob = self._match_probability(home_team, away_team, home_goals, away_goals)
            
            # Add to log-likelihood if probability is valid
            if prob > 0:
                log_likelihood += np.log(prob)
            
            # Update team parameters based on match outcome
            self._update_team_parameters(match_date, home_team, away_team, home_goals, away_goals)
        
        # Update model info
        self.fitted = True
        self.model_info.update({
            "trained": True,
            "log_likelihood": float(log_likelihood),
            "trained_at": datetime.now().isoformat(),
            "n_matches": len(matches),
            "n_teams": len(self.teams),
            "learning_rate": self.learning_rate,
            "persistence": self.persistence,
            "home_advantage": float(self.home_advantage),
            "scale_parameter": float(self.scale_parameter)
        })
        
        logger.info(f"Score-Driven model fitting completed, log-likelihood: {log_likelihood:.2f}")
        
        return self.model_info
    
    def predict_score_probabilities(self, home_team: Any, away_team: Any, 
                                   max_goals: int = 10) -> np.ndarray:
        """
        Predict probabilities for all possible score combinations up to max_goals.
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            max_goals: Maximum number of goals to consider
            
        Returns:
            np.ndarray: 2D array of probabilities for each score combination
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if home_team not in self.team_params or away_team not in self.team_params:
            raise ValueError(f"Team not in model: {home_team if home_team not in self.team_params else away_team}")
        
        # Initialize probability matrix
        score_probs = np.zeros((max_goals + 1, max_goals + 1))
        
        # Get team parameters
        attack_home = self.team_params[home_team]['attack']
        defense_home = self.team_params[home_team]['defense']
        attack_away = self.team_params[away_team]['attack']
        defense_away = self.team_params[away_team]['defense']
        
        # Calculate expected goals
        home_expected = np.exp(attack_home + defense_away + self.home_advantage)
        away_expected = np.exp(attack_away + defense_home)
        
        # Calculate probabilities for each score combination
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                home_prob = self._poisson_likelihood(home_goals, home_expected)
                away_prob = self._poisson_likelihood(away_goals, away_expected)
                score_probs[home_goals, away_goals] = home_prob * away_prob
        
        # Normalize to ensure probabilities sum to 1
        score_probs = score_probs / np.sum(score_probs)
        
        return score_probs
    
    def predict_match_outcome(self, home_team: Any, away_team: Any) -> Dict[str, float]:
        """
        Predict the outcome of a match (home win, draw, away win).
        
        Args:
            home_team: Home team identifier
            away_team: Away team identifier
            
        Returns:
            Dict: Probabilities for each outcome
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get score probabilities
        score_probs = self.predict_score_probabilities(home_team, away_team)
        
        # Calculate outcome probabilities
        n_goals = score_probs.shape[0]
        
        home_win_prob = np.sum(score_probs[np.triu_indices(n_goals, 1)])  # Upper triangle excluding diagonal
        draw_prob = np.sum(np.diag(score_probs))  # Diagonal elements
        away_win_prob = np.sum(score_probs[np.tril_indices(n_goals, -1)])  # Lower triangle excluding diagonal
        
        return {
            "home_win": float(home_win_prob),
            "draw": float(draw_prob),
            "away_win": float(away_win_prob)
        }
    
    def get_team_ratings(self, as_of_date: Optional[datetime] = None) -> Dict[Any, Dict[str, float]]:
        """
        Get the attack and defense ratings for all teams.
        
        Args:
            as_of_date: Optional date for historical ratings (if dynamics are stored)
            
        Returns:
            Dict: Team ratings dictionary
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting ratings")
        
        # If requesting historical ratings
        if as_of_date is not None and self.dynamics_enabled:
            ratings = {}
            
            for team in self.teams:
                if team not in self.team_dynamics:
                    continue
                    
                # Find the closest date before as_of_date
                dates = np.array(self.team_dynamics[team]['dates'])
                if len(dates) == 0:
                    continue
                    
                valid_dates = dates[dates <= as_of_date]
                if len(valid_dates) == 0:
                    continue
                    
                # Get the index of the most recent date
                idx = np.argmax(valid_dates)
                
                # Get attack and defense at that time
                attack = self.team_dynamics[team]['attack'][idx]
                defense = self.team_dynamics[team]['defense'][idx]
                
                # Calculate an overall rating
                overall = np.exp(attack) - np.exp(defense)
                
                ratings[team] = {
                    "attack": float(np.exp(attack)),
                    "defense": float(np.exp(defense)),
                    "overall": float(overall),
                    "as_of_date": valid_dates[idx].strftime("%Y-%m-%d")
                }
                
            return ratings
        
        # Return current ratings
        ratings = {}
        for team in self.teams:
            attack = self.team_params[team]['attack']
            defense = self.team_params[team]['defense']
            
            # Calculate an overall rating as a combination of attack and defense
            overall = np.exp(attack) - np.exp(defense)
            
            ratings[team] = {
                "attack": float(np.exp(attack)),
                "defense": float(np.exp(defense)),
                "overall": float(overall)
            }
        
        return ratings
    
    def get_rating_time_series(self, team: Any) -> Dict[str, List]:
        """
        Get time series of ratings for a specific team.
        
        Args:
            team: Team identifier
            
        Returns:
            Dict: Dictionary of attack, defense, and date time series
        """
        if not self.dynamics_enabled:
            raise ValueError("Team dynamics not stored. Set store_dynamics=True when fitting.")
        
        if team not in self.team_dynamics:
            raise ValueError(f"Team {team} not found in stored dynamics")
        
        # Convert to regular Python lists for serialization
        return {
            "attack": [float(x) for x in self.team_dynamics[team]['attack']],
            "defense": [float(x) for x in self.team_dynamics[team]['defense']],
            "dates": [d.strftime("%Y-%m-%d") for d in self.team_dynamics[team]['dates']]
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            str: Path to the saved model
        """
        if not self.fitted:
            logger.warning("Saving an unfitted model")
        
        # Default path if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                DISTRIBUTION_MODELS_DIR, 
                f"score_driven_{timestamp}.pkl"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, "wb") as f:
            joblib.dump(self, f)
        
        logger.info(f"Score-Driven model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "ScoreDrivenModel":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            ScoreDrivenModel: Loaded model
        """
        with open(filepath, "rb") as f:
            model = joblib.load(f)
        
        logger.info(f"Score-Driven model loaded from {filepath}")
        
        return model


def train_score_driven_model(matches_df: pd.DataFrame,
                           learning_rate: float = 0.05,
                           persistence: float = 0.97,
                           store_dynamics: bool = True) -> ScoreDrivenModel:
    """
    Train a Score-Driven model on match data.
    
    Args:
        matches_df: DataFrame containing match data
        learning_rate: Learning rate for parameter updates
        persistence: Persistence parameter for GAS dynamics
        store_dynamics: Whether to store team parameter time series
        
    Returns:
        ScoreDrivenModel: Trained model
    """
    # Initialize model
    model = ScoreDrivenModel()
    
    # Fit model
    model.fit(
        matches_df, 
        learning_rate=learning_rate,
        persistence=persistence,
        store_dynamics=store_dynamics
    )
    
    return model 