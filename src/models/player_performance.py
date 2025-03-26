"""
Player Performance Prediction Module for Soccer Prediction System.
Provides models for predicting individual player statistics and performance metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import joblib
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import xgboost as XGBRegressor
import lightgbm as LGBMRegressor

# Import project components
from src.utils.logger import get_logger
from src.data.features import load_processed_data, calculate_player_form

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback default if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.player_performance")

# Define model directories
MODELS_DIR = os.path.join(DATA_DIR, "models")
PLAYER_MODELS_DIR = os.path.join(MODELS_DIR, "player")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLAYER_MODELS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Define the performance metrics we want to predict
PERFORMANCE_METRICS = [
    'goals', 
    'assists', 
    'minutes_played',
    'shots', 
    'shots_on_target',
    'pass_completion', 
    'key_passes',
    'tackles',
    'interceptions',
    'duels_won',
    'rating'
]

class PlayerPerformanceModel:
    """
    Model for predicting player performance metrics for upcoming matches.
    """
    
    def __init__(self, metric: str, model_type: str = "gradient_boosting"):
        """
        Initialize a player performance prediction model.
        
        Args:
            metric: The performance metric to predict (goals, assists, etc.)
            model_type: Type of model to use (random_forest, gradient_boosting, xgboost, etc.)
        """
        self.metric = metric
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.preprocessor = None
        self.feature_importances = None
        
    def _create_model(self) -> Any:
        """Create the underlying machine learning model based on model_type."""
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.model_type == "xgboost":
            return XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.model_type == "lightgbm":
            return LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0, random_state=42)
        elif self.model_type == "lasso":
            return Lasso(alpha=0.1, random_state=42)
        elif self.model_type == "elastic_net":
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create a preprocessor for the input features.
        
        Args:
            X: Training data features
            
        Returns:
            ColumnTransformer: Fitted preprocessor
        """
        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols if categorical_cols else [])
            ]
        )
        
        return preprocessor
    
    def train(self, X: pd.DataFrame, y: pd.Series, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train the player performance prediction model.
        
        Args:
            X: Training data features
            y: Target variable (performance metric)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dict[str, Any]: Training results
        """
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Create and fit preprocessor
        self.preprocessor = self._create_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Create model
        if tune_hyperparameters:
            # Define hyperparameter search space based on model type
            if self.model_type == "random_forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                base_model = RandomForestRegressor(random_state=42)
            elif self.model_type == "gradient_boosting":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                base_model = GradientBoostingRegressor(random_state=42)
            elif self.model_type in ["xgboost", "lightgbm"]:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0]
                }
                base_model = XGBRegressor(random_state=42) if self.model_type == "xgboost" else LGBMRegressor(random_state=42)
            else:
                # Ridge, Lasso, ElasticNet
                param_grid = {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
                if self.model_type == "elastic_net":
                    param_grid['l1_ratio'] = [0.1, 0.3, 0.5, 0.7, 0.9]
                base_model = Ridge(random_state=42) if self.model_type == "ridge" else (
                    Lasso(random_state=42) if self.model_type == "lasso" else ElasticNet(random_state=42)
                )
            
            # Use RandomizedSearchCV for efficiency
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=20,
                cv=5,
                scoring='neg_mean_squared_error',
                random_state=42,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            logger.info(f"Best hyperparameters for {self.metric} model: {search.best_params_}")
        else:
            # Use default hyperparameters
            self.model = self._create_model()
            self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_
        
        # Log results
        logger.info(f"Trained {self.model_type} model for {self.metric}")
        logger.info(f"Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return {
            'metric': self.metric,
            'model_type': self.model_type,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importances': self.feature_importances.tolist() if self.feature_importances is not None else None
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predicted values for the performance metric
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Check if X has the expected columns
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Reorder columns to match training data
        X = X[self.feature_columns]
        
        # Preprocess features
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        # Ensure non-negative values for metrics that can't be negative
        if self.metric in ['goals', 'assists', 'minutes_played', 'shots', 'shots_on_target', 'tackles', 'interceptions', 'duels_won']:
            predictions = np.maximum(predictions, 0)
        
        # For percentages, ensure values are between 0 and 100
        if self.metric in ['pass_completion']:
            predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model to (if None, use default path)
            
        Returns:
            str: Path where the model was saved
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if path is None:
            path = os.path.join(PLAYER_MODELS_DIR, f"{self.metric}_{self.model_type}_model.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model, preprocessor, and metadata
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.feature_columns,
            'metric': self.metric,
            'model_type': self.model_type,
            'feature_importances': self.feature_importances,
            'created_at': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'PlayerPerformanceModel':
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            PlayerPerformanceModel: Loaded model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(metric=model_data['metric'], model_type=model_data['model_type'])
        
        # Load attributes
        instance.model = model_data['model']
        instance.preprocessor = model_data['preprocessor']
        instance.feature_columns = model_data['feature_columns']
        instance.feature_importances = model_data['feature_importances']
        
        logger.info(f"Loaded model from {path}")
        
        return instance


class PlayerPerformancePredictor:
    """
    Service for predicting player performance in upcoming matches.
    Manages multiple performance models for different metrics.
    """
    
    def __init__(self):
        """Initialize the player performance predictor."""
        self.models = {}
        self.scan_available_models()
    
    def scan_available_models(self):
        """Scan for available player performance models."""
        if not os.path.exists(PLAYER_MODELS_DIR):
            logger.warning(f"Player models directory not found: {PLAYER_MODELS_DIR}")
            return
        
        # Find all model files
        model_files = [f for f in os.listdir(PLAYER_MODELS_DIR) if f.endswith('.pkl')]
        
        for model_file in model_files:
            try:
                # Extract metric and model type from filename
                parts = model_file.replace('_model.pkl', '').split('_')
                metric = parts[0]
                model_type = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                
                # Register model
                model_path = os.path.join(PLAYER_MODELS_DIR, model_file)
                self.models[metric] = {
                    'path': model_path,
                    'model_type': model_type,
                    'loaded': False,
                    'model': None
                }
                logger.debug(f"Found {model_type} model for metric: {metric}")
            except Exception as e:
                logger.error(f"Error processing model file {model_file}: {e}")
    
    def load_model(self, metric: str) -> bool:
        """
        Load a specific player performance model.
        
        Args:
            metric: The performance metric to load the model for
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if metric not in self.models:
            logger.error(f"No model found for metric: {metric}")
            return False
        
        if self.models[metric]['loaded']:
            logger.debug(f"Model for {metric} already loaded")
            return True
        
        try:
            model_path = self.models[metric]['path']
            self.models[metric]['model'] = PlayerPerformanceModel.load(model_path)
            self.models[metric]['loaded'] = True
            logger.info(f"Loaded model for {metric}")
            return True
        except Exception as e:
            logger.error(f"Error loading model for {metric}: {e}")
            return False
    
    def predict_player_performance(
        self,
        player_id: int,
        match_id: int,
        team_id: int,
        opponent_id: int,
        is_home: bool,
        features: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Predict player performance for a specific match.
        
        Args:
            player_id: ID of the player
            match_id: ID of the match
            team_id: ID of the player's team
            opponent_id: ID of the opponent team
            is_home: Whether the player's team is playing at home
            features: Optional additional features for the prediction
            metrics: Optional list of metrics to predict (if None, predict all available)
            
        Returns:
            Dict[str, Any]: Prediction results for each metric
        """
        if metrics is None:
            # Use all available metrics
            metrics = list(self.models.keys())
        
        # Check which metrics we have models for
        available_metrics = [m for m in metrics if m in self.models]
        missing_metrics = [m for m in metrics if m not in self.models]
        
        if missing_metrics:
            logger.warning(f"No models available for metrics: {missing_metrics}")
            if not available_metrics:
                return {"error": "No models available for requested metrics"}
        
        # Prepare input features
        X = self._prepare_features(player_id, team_id, opponent_id, is_home, features)
        
        # Make predictions for each metric
        results = {
            "player_id": player_id,
            "match_id": match_id,
            "team_id": team_id,
            "opponent_id": opponent_id,
            "is_home": is_home,
            "predictions": {}
        }
        
        for metric in available_metrics:
            # Load model if not already loaded
            if not self.models[metric]['loaded']:
                if not self.load_model(metric):
                    results["predictions"][metric] = None
                    continue
            
            # Make prediction
            try:
                model = self.models[metric]['model']
                prediction = model.predict(X)[0]  # Assuming single-row input
                results["predictions"][metric] = float(prediction)
            except Exception as e:
                logger.error(f"Error predicting {metric} for player {player_id}: {e}")
                results["predictions"][metric] = None
        
        # Record prediction
        self._record_prediction(results)
        
        return results
    
    def _prepare_features(
        self,
        player_id: int,
        team_id: int,
        opponent_id: int,
        is_home: bool,
        features: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Args:
            player_id: ID of the player
            team_id: ID of the player's team
            opponent_id: ID of the opponent team
            is_home: Whether the player's team is playing at home
            features: Optional additional features for the prediction
            
        Returns:
            pd.DataFrame: DataFrame with features for prediction
        """
        # Start with basic features
        feature_dict = {
            'player_id': player_id,
            'team_id': team_id,
            'opponent_id': opponent_id,
            'is_home': is_home
        }
        
        # Add additional features if provided
        if features:
            feature_dict.update(features)
        
        # Convert to DataFrame
        X = pd.DataFrame([feature_dict])
        
        return X
    
    def _record_prediction(self, prediction: Dict[str, Any]):
        """
        Record prediction for tracking and evaluation.
        
        Args:
            prediction: Prediction results to record
        """
        # Create prediction record
        record = {
            "timestamp": datetime.now().isoformat(),
            "player_id": prediction["player_id"],
            "match_id": prediction["match_id"],
            "team_id": prediction["team_id"],
            "opponent_id": prediction["opponent_id"],
            "is_home": prediction["is_home"],
            "predictions": prediction["predictions"]
        }
        
        # Create prediction directory if it doesn't exist
        player_predictions_dir = os.path.join(PREDICTIONS_DIR, "player")
        os.makedirs(player_predictions_dir, exist_ok=True)
        
        # Create or append to predictions file
        predictions_file = os.path.join(player_predictions_dir, f"player_{prediction['player_id']}_predictions.jsonl")
        
        with open(predictions_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def batch_predict(
        self,
        player_matches: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple player-match combinations.
        
        Args:
            player_matches: List of dictionaries with player and match information
            metrics: Optional list of metrics to predict
            
        Returns:
            List[Dict[str, Any]]: Prediction results for each player-match combination
        """
        results = []
        
        for player_match in player_matches:
            prediction = self.predict_player_performance(
                player_id=player_match["player_id"],
                match_id=player_match["match_id"],
                team_id=player_match.get("team_id"),
                opponent_id=player_match.get("opponent_id"),
                is_home=player_match.get("is_home", True),
                features=player_match.get("features"),
                metrics=metrics
            )
            results.append(prediction)
        
        return results
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available performance metrics with trained models.
        
        Returns:
            List[str]: List of available metrics
        """
        return list(self.models.keys())
    
    def get_model_info(self, metric: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            metric: The performance metric
            
        Returns:
            Dict[str, Any]: Model information
        """
        if metric not in self.models:
            return {"error": f"No model found for metric: {metric}"}
        
        # Load model if not already loaded
        if not self.models[metric]['loaded']:
            if not self.load_model(metric):
                return {"error": f"Failed to load model for metric: {metric}"}
        
        model = self.models[metric]['model']
        
        # Get feature importances if available
        feature_importances = None
        if model.feature_importances is not None:
            feature_importances = {
                col: importance
                for col, importance in zip(model.feature_columns, model.feature_importances)
            }
            # Sort by importance
            feature_importances = {
                k: v for k, v in sorted(
                    feature_importances.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )
            }
        
        return {
            "metric": metric,
            "model_type": model.model_type,
            "feature_columns": model.feature_columns,
            "feature_importances": feature_importances,
            "file_path": self.models[metric]['path']
        }


def prepare_player_performance_dataset(
    dataset_name: str = "transfermarkt", 
    lookback_window: int = 5
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Prepare datasets for training player performance models.
    
    Args:
        dataset_name: Name of the dataset to use
        lookback_window: Number of previous matches to use for player form calculation
        
    Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: 
            (Training datasets for each metric, Test datasets for each metric)
    """
    logger.info(f"Preparing player performance dataset from {dataset_name}")
    
    # Load processed data
    data = load_processed_data(dataset_name)
    
    if 'appearances' not in data or 'matches' not in data or 'players' not in data:
        raise ValueError("Required data tables not found in processed data")
    
    # Calculate player form
    player_form = calculate_player_form(data['appearances'], window=lookback_window)
    
    # Merge with player data
    players = data['players']
    player_data = player_form.merge(players, on='player_id', how='left')
    
    # Merge with match data to get context
    matches = data['matches']
    player_match_data = player_data.merge(
        matches[['match_id', 'home_club_id', 'away_club_id', 'date']],
        on='match_id',
        how='left'
    )
    
    # Add features about whether player is on home team
    player_match_data['is_home'] = player_match_data['team_id'] == player_match_data['home_club_id']
    player_match_data['opponent_id'] = np.where(
        player_match_data['is_home'],
        player_match_data['away_club_id'],
        player_match_data['home_club_id']
    )
    
    # Prepare training datasets for each performance metric
    training_datasets = {}
    test_datasets = {}
    
    for metric in PERFORMANCE_METRICS:
        if metric not in player_match_data.columns:
            logger.warning(f"Metric {metric} not found in data, skipping")
            continue
        
        # Select features
        features = player_match_data[[
            'player_id', 'team_id', 'opponent_id', 'is_home', 'match_id',
            f'last_{lookback_window}_goals', f'last_{lookback_window}_assists', 
            f'last_{lookback_window}_minutes', f'last_{lookback_window}_goals_per_90',
            f'last_{lookback_window}_assists_per_90', 'goals_form_trend',
            'assists_form_trend', 'age', 'position'
        ]].copy()
        
        # Add the target variable
        features[metric] = player_match_data[metric]
        
        # Drop rows with missing target
        features = features.dropna(subset=[metric])
        
        # Split by date (more recent matches for testing)
        train_size = int(len(features) * 0.8)
        train_data = features.iloc[:train_size]
        test_data = features.iloc[train_size:]
        
        training_datasets[metric] = train_data
        test_datasets[metric] = test_data
        
        logger.info(f"Prepared dataset for {metric} with {len(train_data)} training and {len(test_data)} test samples")
    
    return training_datasets, test_datasets


def train_player_performance_models(
    dataset_name: str = "transfermarkt",
    lookback_window: int = 5,
    model_types: List[str] = ["gradient_boosting"],
    tune_hyperparameters: bool = False,
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train models for predicting various player performance metrics.
    
    Args:
        dataset_name: Name of the dataset to use
        lookback_window: Number of previous matches to use for player form calculation
        model_types: List of model types to train
        tune_hyperparameters: Whether to perform hyperparameter tuning
        metrics: Optional list of metrics to train models for (if None, train for all available)
        
    Returns:
        Dict[str, Dict[str, Any]]: Training results for each metric and model type
    """
    logger.info(f"Training player performance models using {dataset_name} data")
    
    # Prepare datasets
    training_datasets, test_datasets = prepare_player_performance_dataset(
        dataset_name=dataset_name,
        lookback_window=lookback_window
    )
    
    if metrics is None:
        # Use all available metrics
        metrics = list(training_datasets.keys())
    else:
        # Filter to requested metrics that are available
        metrics = [m for m in metrics if m in training_datasets]
    
    if not metrics:
        raise ValueError("No valid metrics to train models for")
    
    # Train models for each metric
    results = {}
    
    for metric in metrics:
        logger.info(f"Training models for metric: {metric}")
        
        # Get training and test data
        train_data = training_datasets[metric]
        test_data = test_datasets[metric]
        
        # Prepare features and target
        X_train = train_data.drop([metric, 'match_id'], axis=1)
        y_train = train_data[metric]
        
        X_test = test_data.drop([metric, 'match_id'], axis=1)
        y_test = test_data[metric]
        
        metric_results = {}
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model for {metric}")
            
            # Create and train model
            model = PlayerPerformanceModel(metric=metric, model_type=model_type)
            train_result = model.train(X_train, y_train, tune_hyperparameters=tune_hyperparameters)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            
            # Calculate test metrics
            test_mse = mean_squared_error(y_test, y_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Test RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
            
            # Save model
            model_path = model.save()
            
            # Store results
            metric_results[model_type] = {
                **train_result,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'model_path': model_path
            }
        
        results[metric] = metric_results
    
    return results


def get_player_predictions(
    player_id: int,
    limit: int = 10,
    metric: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get prediction history for a specific player.
    
    Args:
        player_id: ID of the player
        limit: Maximum number of predictions to return
        metric: Optional specific metric to filter by
        
    Returns:
        List[Dict[str, Any]]: List of prediction records
    """
    predictions_file = os.path.join(PREDICTIONS_DIR, "player", f"player_{player_id}_predictions.jsonl")
    
    if not os.path.exists(predictions_file):
        return []
    
    predictions = []
    
    with open(predictions_file, 'r') as f:
        for line in f:
            prediction = json.loads(line.strip())
            
            # Filter by metric if specified
            if metric is not None:
                if metric not in prediction['predictions']:
                    continue
                
                # Keep only the specified metric
                prediction['predictions'] = {metric: prediction['predictions'][metric]}
            
            predictions.append(prediction)
    
    # Sort by timestamp (newest first) and limit results
    predictions.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return predictions[:limit] 