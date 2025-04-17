"""
Expected Goals (xG) Modeling Module

This module implements xG models to estimate the quality of scoring chances and predict goals.
xG is a statistical measure that represents the probability of a shot resulting in a goal
based on various factors such as shot location, type, game context, etc.

Research shows that xG models provide better predictive power than simply using past goals
as a measure of team offensive and defensive capability.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import project components
from src.utils.logger import get_logger

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.expected_goals")

# Define paths
MODELS_DIR = os.path.join(DATA_DIR, "models")
XG_MODELS_DIR = os.path.join(MODELS_DIR, "xg")
os.makedirs(XG_MODELS_DIR, exist_ok=True)


class ExpectedGoalsModel:
    """
    A model for predicting the probability of shots resulting in goals (xG).
    
    This model can be used to:
    1. Calculate xG values for individual shots
    2. Aggregate xG values for teams in matches
    3. Evaluate team attacking and defensive quality
    4. Predict match outcomes based on expected goals
    """
    
    def __init__(self, model_type: str = "gradient_boosting"):
        """
        Initialize the Expected Goals model.
        
        Args:
            model_type: Type of model to use ("gradient_boosting" or "random_forest")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.fitted = False
        self.model_info = {
            "model_type": model_type,
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {}
        }
    
    def _create_model(self):
        """Create the underlying machine learning model."""
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Created {self.model_type} model for xG prediction")
    
    def fit(self, shots_df: pd.DataFrame, goal_col: str = "goal", 
           feature_cols: Optional[List[str]] = None, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Fit the xG model to shots data.
        
        Args:
            shots_df: DataFrame containing shot data
            goal_col: Name of the column indicating whether a shot resulted in a goal (1) or not (0)
            feature_cols: List of feature columns to use
            test_size: Proportion of data to use for testing
            
        Returns:
            Dict: Training results
        """
        # Use default features if not provided
        if feature_cols is None:
            # These are examples of common xG features
            possible_features = [
                # Distance and angle
                "distance", "angle", "distance_to_goal", "angle_to_goal",
                
                # Shot characteristics
                "is_header", "is_volley", "is_penalty", "is_free_kick", "is_open_play",
                "body_part", "shot_power", "shot_technique",
                
                # Game context
                "home_team", "minute", "game_state", "is_fast_break", "is_counter_attack",
                "defender_distance", "goalkeeper_distance", "num_defenders_in_range",
                
                # Player characteristics
                "striker_skill", "striker_form", "striker_xg_history",
                
                # Team characteristics
                "team_attacking_strength", "opponent_defensive_strength"
            ]
            
            # Use only features available in the dataset
            feature_cols = [col for col in possible_features if col in shots_df.columns]
            
            if len(feature_cols) < 3:
                logger.warning("Few recognized xG features found in data. Using all numeric columns.")
                # Use any numeric column except the goal column as a fallback
                numeric_cols = shots_df.select_dtypes(include=['number']).columns.tolist()
                feature_cols = [col for col in numeric_cols if col != goal_col]
        
        self.feature_names = feature_cols
        logger.info(f"Using {len(feature_cols)} features for xG modeling: {feature_cols}")
        
        # Prepare data
        X = shots_df[feature_cols].copy()
        y = shots_df[goal_col].astype(int)
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and fit model
        if self.model is None:
            self._create_model()
            
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_preds = self.model.predict_proba(X_train_scaled)[:, 1]
        test_preds = self.model.predict_proba(X_test_scaled)[:, 1]
        
        train_logloss = log_loss(y_train, train_preds)
        test_logloss = log_loss(y_test, test_preds)
        
        train_brier = brier_score_loss(y_train, train_preds)
        test_brier = brier_score_loss(y_test, test_preds)
        
        train_auc = roc_auc_score(y_train, train_preds)
        test_auc = roc_auc_score(y_test, test_preds)
        
        # Calculate feature importance
        if hasattr(self.model, "feature_importances_"):
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        else:
            feature_importance = {}
        
        # Update model info
        self.fitted = True
        performance = {
            "train_log_loss": float(train_logloss),
            "test_log_loss": float(test_logloss),
            "train_brier_score": float(train_brier),
            "test_brier_score": float(test_brier),
            "train_auc": float(train_auc),
            "test_auc": float(test_auc),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "feature_importance": feature_importance
        }
        
        self.model_info.update({
            "trained": True,
            "trained_at": datetime.now().isoformat(),
            "features": self.feature_names,
            "performance": performance
        })
        
        logger.info(f"xG model training completed. Test AUC: {test_auc:.4f}, Test Brier score: {test_brier:.4f}")
        
        return self.model_info
    
    def predict_shot_xg(self, shot_data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Predict xG values for shots.
        
        Args:
            shot_data: DataFrame or dict with shot features
            
        Returns:
            np.ndarray: Predicted xG values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert dict to DataFrame if necessary
        if isinstance(shot_data, dict):
            shot_data = pd.DataFrame([shot_data])
        
        # Ensure all required features are present
        missing_features = [f for f in self.feature_names if f not in shot_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features for xG prediction: {missing_features}")
        
        # Prepare features
        X = shot_data[self.feature_names].copy()
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Ensure all columns from training exist (add missing columns with zeros)
        for col in self.model.feature_names_in_:
            if col not in X.columns:
                X[col] = 0
                
        # Select only columns the model knows about
        X = X[self.model.feature_names_in_]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict probabilities
        xg_values = self.model.predict_proba(X_scaled)[:, 1]
        
        return xg_values
    
    def calculate_match_xg(self, 
                         home_shots: pd.DataFrame, 
                         away_shots: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate total xG for each team in a match.
        
        Args:
            home_shots: DataFrame with home team shots
            away_shots: DataFrame with away team shots
            
        Returns:
            Tuple[float, float]: (home_xg, away_xg)
        """
        home_xg = self.predict_shot_xg(home_shots).sum()
        away_xg = self.predict_shot_xg(away_shots).sum()
        
        return home_xg, away_xg
    
    def predict_match_outcome(self, 
                            home_xg: float, 
                            away_xg: float,
                            simulation_count: int = 10000) -> Dict[str, float]:
        """
        Predict match outcome probabilities based on team xG values.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            simulation_count: Number of match simulations to run
            
        Returns:
            Dict: Probabilities for home win, draw, and away win
        """
        # Simulate matches using Poisson distribution
        home_goals = np.random.poisson(home_xg, simulation_count)
        away_goals = np.random.poisson(away_xg, simulation_count)
        
        # Calculate outcome counts
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        # Calculate probabilities
        home_win_prob = home_wins / simulation_count
        draw_prob = draws / simulation_count
        away_win_prob = away_wins / simulation_count
        
        return {
            "home_win": float(home_win_prob),
            "draw": float(draw_prob),
            "away_win": float(away_win_prob)
        }
    
    def calculate_team_xg_strength(self, 
                                 matches_df: pd.DataFrame, 
                                 shots_df: pd.DataFrame,
                                 days_window: int = 180) -> pd.DataFrame:
        """
        Calculate team offensive and defensive strength based on xG.
        
        Args:
            matches_df: DataFrame with match information
            shots_df: DataFrame with shot information
            days_window: Number of days to include in the window
            
        Returns:
            pd.DataFrame: DataFrame with team xG attack and defense ratings
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating team strength")
        
        # Calculate xG for each shot
        shots_df = shots_df.copy()
        shots_df['xg'] = self.predict_shot_xg(shots_df)
        
        # Get match IDs for shots
        match_ids = shots_df['match_id'].unique()
        
        # Filter matches to those with shots
        filtered_matches = matches_df[matches_df['match_id'].isin(match_ids)].copy()
        
        # Ensure date column is datetime
        if 'date' in filtered_matches.columns and not pd.api.types.is_datetime64_dtype(filtered_matches['date']):
            filtered_matches['date'] = pd.to_datetime(filtered_matches['date'])
        
        # Calculate xG for each team in each match
        match_xg = []
        
        for match_id, match in filtered_matches.iterrows():
            match_shots = shots_df[shots_df['match_id'] == match['match_id']]
            
            # Separate home and away shots
            home_shots = match_shots[match_shots['team_id'] == match['home_team_id']]
            away_shots = match_shots[match_shots['team_id'] == match['away_team_id']]
            
            # Calculate total xG
            home_xg = home_shots['xg'].sum()
            away_xg = away_shots['xg'].sum()
            
            match_xg.append({
                'match_id': match['match_id'],
                'date': match['date'],
                'home_team_id': match['home_team_id'],
                'away_team_id': match['away_team_id'],
                'home_xg': home_xg,
                'away_xg': away_xg,
                'home_goals': match['home_goals'],
                'away_goals': match['away_goals']
            })
        
        match_xg_df = pd.DataFrame(match_xg)
        
        # Calculate team strength within the time window
        latest_date = filtered_matches['date'].max()
        cutoff_date = latest_date - pd.Timedelta(days=days_window)
        
        recent_matches = match_xg_df[match_xg_df['date'] >= cutoff_date]
        
        # Calculate team metrics
        team_stats = {}
        
        for team_id in set(filtered_matches['home_team_id']) | set(filtered_matches['away_team_id']):
            # Home matches
            home_matches = recent_matches[recent_matches['home_team_id'] == team_id]
            # Away matches
            away_matches = recent_matches[recent_matches['away_team_id'] == team_id]
            
            # Combine stats
            total_matches = len(home_matches) + len(away_matches)
            
            if total_matches > 0:
                # Attacking strength (xG created per match)
                attack_xg = (home_matches['home_xg'].sum() + away_matches['away_xg'].sum()) / total_matches
                
                # Defensive strength (xG conceded per match)
                defense_xg = (home_matches['away_xg'].sum() + away_matches['home_xg'].sum()) / total_matches
                
                # xG difference
                xg_diff = attack_xg - defense_xg
                
                # Actual goals vs xG
                actual_goals = (home_matches['home_goals'].sum() + away_matches['away_goals'].sum())
                expected_goals = (home_matches['home_xg'].sum() + away_matches['away_xg'].sum())
                
                finishing_efficiency = actual_goals / expected_goals if expected_goals > 0 else 1.0
                
                # Save team stats
                team_stats[team_id] = {
                    'team_id': team_id,
                    'matches_played': total_matches,
                    'attack_xg': attack_xg,
                    'defense_xg': defense_xg,
                    'xg_difference': xg_diff,
                    'finishing_efficiency': finishing_efficiency,
                    'as_of_date': latest_date
                }
        
        # Convert to DataFrame
        team_strength_df = pd.DataFrame.from_dict(team_stats, orient='index')
        
        # Calculate league averages
        avg_attack = team_strength_df['attack_xg'].mean()
        avg_defense = team_strength_df['defense_xg'].mean()
        
        # Calculate relative strengths
        team_strength_df['attack_strength'] = team_strength_df['attack_xg'] / avg_attack
        team_strength_df['defense_strength'] = avg_defense / team_strength_df['defense_xg']  # Inverted so higher is better
        team_strength_df['overall_strength'] = (team_strength_df['attack_strength'] + team_strength_df['defense_strength']) / 2
        
        return team_strength_df
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            str: Path to the saved model
        """
        if not self.fitted:
            logger.warning("Saving an unfitted xG model")
        
        # Default path if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                XG_MODELS_DIR, 
                f"xg_{self.model_type}_{timestamp}.pkl"
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, "wb") as f:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_info': self.model_info,
                'fitted': self.fitted
            }, f)
        
        logger.info(f"xG model saved to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "ExpectedGoalsModel":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            ExpectedGoalsModel: Loaded model
        """
        with open(filepath, "rb") as f:
            model_data = joblib.load(f)
        
        # Create model instance
        model_type = model_data['model_info']['model_type']
        xg_model = cls(model_type=model_type)
        
        # Restore model attributes
        xg_model.model = model_data['model']
        xg_model.scaler = model_data['scaler']
        xg_model.feature_names = model_data['feature_names']
        xg_model.model_info = model_data['model_info']
        xg_model.fitted = model_data['fitted']
        
        logger.info(f"xG model loaded from {filepath}")
        
        return xg_model


def preprocess_shot_data(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw shot data to create features for xG modeling.
    
    Args:
        shots_df: DataFrame with raw shot information
        
    Returns:
        pd.DataFrame: DataFrame with xG features
    """
    df = shots_df.copy()
    
    # Example preprocessing steps
    
    # 1. Convert pitch coordinates to distance and angle from goal
    if all(col in df.columns for col in ['x', 'y']):
        # Assumes coordinates in a 100x100 system, with (100, 50) being the goal position
        df['distance_to_goal'] = np.sqrt((100 - df['x'])**2 + (50 - df['y'])**2)
        
        # Calculate angle
        opposite = np.abs(50 - df['y'])
        adjacent = 100 - df['x']
        # Handle edge cases to avoid division by zero
        df['angle_to_goal'] = np.where(
            adjacent > 0,
            np.degrees(np.arctan(opposite / adjacent)),
            90.0  # shots from the goal line have 90 degree angle
        )
    
    # 2. Extract shot type information
    if 'shot_type' in df.columns:
        df['is_header'] = df['shot_type'].str.contains('header', case=False).astype(int)
        df['is_volley'] = df['shot_type'].str.contains('volley', case=False).astype(int)
    
    # 3. Extract shot situation
    if 'shot_situation' in df.columns:
        df['is_penalty'] = df['shot_situation'].str.contains('penalty', case=False).astype(int)
        df['is_free_kick'] = df['shot_situation'].str.contains('free_kick', case=False).astype(int)
        df['is_open_play'] = (~df['shot_situation'].str.contains('penalty|free_kick', case=False)).astype(int)
    
    # 4. Encode categorical variables
    # (dummies will be created during modeling)
    
    return df


def train_xg_model(shots_df: pd.DataFrame, 
                  model_type: str = "gradient_boosting",
                  test_size: float = 0.2) -> ExpectedGoalsModel:
    """
    Train an xG model on shot data.
    
    Args:
        shots_df: DataFrame with shot data
        model_type: Type of model to use
        test_size: Proportion of data to use for testing
        
    Returns:
        ExpectedGoalsModel: Trained model
    """
    # Preprocess data if needed
    if not any(col.startswith('angle') for col in shots_df.columns):
        logger.info("Preprocessing shot data")
        shots_df = preprocess_shot_data(shots_df)
    
    # Initialize model
    model = ExpectedGoalsModel(model_type=model_type)
    
    # Train model
    model.fit(shots_df, test_size=test_size)
    
    return model


def find_shots_for_match(shot_db: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """
    Find all shots for a specific match.
    
    Args:
        shot_db: DataFrame with shot data
        match_id: Match ID to filter by
        
    Returns:
        pd.DataFrame: DataFrame with shots for the match
    """
    return shot_db[shot_db['match_id'] == match_id].copy()


def calculate_team_xg_over_time(shots_df: pd.DataFrame, 
                             matches_df: pd.DataFrame,
                             xg_model: ExpectedGoalsModel,
                             window_size: int = 10) -> pd.DataFrame:
    """
    Calculate team xG metrics over time using a rolling window.
    
    Args:
        shots_df: DataFrame with shot data
        matches_df: DataFrame with match data
        xg_model: Trained xG model
        window_size: Number of matches to include in rolling window
        
    Returns:
        pd.DataFrame: DataFrame with team xG metrics over time
    """
    # Calculate xG for each shot
    shots_df = shots_df.copy()
    shots_df['xg'] = xg_model.predict_shot_xg(shots_df)
    
    # Calculate xG for each match
    match_xg = []
    
    for match_id, match in matches_df.iterrows():
        match_shots = shots_df[shots_df['match_id'] == match['match_id']]
        
        # Skip if no shots data
        if len(match_shots) == 0:
            continue
        
        # Separate home and away shots
        home_shots = match_shots[match_shots['team_id'] == match['home_team_id']]
        away_shots = match_shots[match_shots['team_id'] == match['away_team_id']]
        
        # Calculate total xG
        home_xg = home_shots['xg'].sum()
        away_xg = away_shots['xg'].sum()
        
        match_xg.append({
            'match_id': match['match_id'],
            'date': match['date'],
            'home_team_id': match['home_team_id'],
            'away_team_id': match['away_team_id'],
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_goals': match['home_goals'],
            'away_goals': match['away_goals']
        })
    
    match_xg_df = pd.DataFrame(match_xg)
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_dtype(match_xg_df['date']):
        match_xg_df['date'] = pd.to_datetime(match_xg_df['date'])
    
    # Sort by date
    match_xg_df = match_xg_df.sort_values('date')
    
    # Calculate rolling xG for each team
    team_xg_over_time = []
    
    teams = set(match_xg_df['home_team_id']) | set(match_xg_df['away_team_id'])
    
    for team_id in teams:
        # Get team matches
        team_home = match_xg_df[match_xg_df['home_team_id'] == team_id].copy()
        team_away = match_xg_df[match_xg_df['away_team_id'] == team_id].copy()
        
        # Add team perspective columns
        team_home['team_xg'] = team_home['home_xg']
        team_home['opponent_xg'] = team_home['away_xg']
        team_home['team_goals'] = team_home['home_goals']
        team_home['opponent_goals'] = team_home['away_goals']
        team_home['is_home'] = True
        
        team_away['team_xg'] = team_away['away_xg']
        team_away['opponent_xg'] = team_away['home_xg']
        team_away['team_goals'] = team_away['away_goals']
        team_away['opponent_goals'] = team_away['home_goals']
        team_away['is_home'] = False
        
        # Combine and sort
        team_matches = pd.concat([team_home, team_away]).sort_values('date')
        
        # Calculate rolling metrics
        team_matches['rolling_attack_xg'] = team_matches['team_xg'].rolling(window=window_size, min_periods=3).mean()
        team_matches['rolling_defense_xg'] = team_matches['opponent_xg'].rolling(window=window_size, min_periods=3).mean()
        team_matches['rolling_xg_diff'] = team_matches['rolling_attack_xg'] - team_matches['rolling_defense_xg']
        
        team_matches['rolling_goals_scored'] = team_matches['team_goals'].rolling(window=window_size, min_periods=3).mean()
        team_matches['rolling_goals_conceded'] = team_matches['opponent_goals'].rolling(window=window_size, min_periods=3).mean()
        team_matches['rolling_goal_diff'] = team_matches['rolling_goals_scored'] - team_matches['rolling_goals_conceded']
        
        # Calculate finishing efficiency (goals / xG)
        team_matches['rolling_finishing'] = (
            team_matches['rolling_goals_scored'] / team_matches['rolling_attack_xg']
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        # Add team ID
        team_matches['team_id'] = team_id
        
        # Append to result
        team_xg_over_time.append(team_matches)
    
    # Combine all teams
    result_df = pd.concat(team_xg_over_time)
    
    return result_df[['match_id', 'date', 'team_id', 'is_home',
                      'team_xg', 'opponent_xg', 'team_goals', 'opponent_goals',
                      'rolling_attack_xg', 'rolling_defense_xg', 'rolling_xg_diff',
                      'rolling_goals_scored', 'rolling_goals_conceded', 'rolling_goal_diff',
                      'rolling_finishing']] 