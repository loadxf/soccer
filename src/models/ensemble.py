"""
Ensemble models for soccer match prediction.
Implements ensemble framework for combining multiple baseline and advanced models.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, confusion_matrix
import json

# Import project components
from src.utils.logger import get_logger
from src.data.features import load_feature_pipeline, apply_feature_pipeline
from src.models.baseline import BaselineMatchPredictor
from src.models.advanced import AdvancedMatchPredictor
try:
    from src.models.soccer_distributions import DixonColesModel
except ImportError:
    pass

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.ensemble")

# Define paths
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
ENSEMBLE_MODELS_DIR = os.path.join(MODELS_DIR, "ensemble")
os.makedirs(ENSEMBLE_MODELS_DIR, exist_ok=True)

# Define weighting strategies
WEIGHTING_STRATEGIES = {
    "equal": lambda models, _: {model_id: 1.0 / len(models) for model_id in models},
    "accuracy": lambda models, perf: {
        model_id: perf[model_id].get("accuracy", 0) 
        for model_id in models
    },
    "f1": lambda models, perf: {
        model_id: perf[model_id].get("f1_weighted", 0) 
        for model_id in models
    },
    "rank": lambda models, perf: {
        model_id: 1.0 / (list(sorted(models, key=lambda m: perf[m].get("accuracy", 0), reverse=True)).index(model_id) + 1)
        for model_id in models
    },
    "dynamic": None,  # Will be implemented separately
    "recency": None,  # Will be implemented separately
    "context_aware": None  # Will be implemented separately
}


class EnsemblePredictor:
    """
    Ensemble model for predicting soccer match outcomes.
    Combines multiple baseline and advanced models for improved prediction accuracy.
    
    Supports various ensemble strategies:
    - Voting: Simple majority voting or weighted average
    - Stacking: Uses a meta-model to combine predictions
    - Blending: Trains on different subsets of data
    - Boosting-based ensembles: Sequential model training
    - Dynamic weighting: Adjusts weights based on recent performance
    - Context-aware: Specializes model selection based on match context
    """
    
    ENSEMBLE_TYPES = [
        "voting", 
        "stacking", 
        "blending", 
        "calibrated_voting",
        "time_weighted",
        "performance_weighted",
        "dynamic_weighting",
        "context_aware"
    ]
    
    def __init__(self, 
                 ensemble_type: str = "voting", 
                 models: Optional[List[Union[BaselineMatchPredictor, AdvancedMatchPredictor]]] = None,
                 weights: Optional[List[float]] = None,
                 meta_model: Optional[BaseEstimator] = None,
                 dataset_name: str = "transfermarkt",
                 feature_type: str = "match_features"):
        """
        Initialize the ensemble predictor.
        
        Args:
            ensemble_type: Type of ensemble ("voting", "stacking", "blending", etc.)
            models: List of trained model instances to include in the ensemble
            weights: Weights for each model (for weighted voting)
            meta_model: Model to use for combining base model predictions (for stacking)
            dataset_name: Name of the dataset to use
            feature_type: Type of features to use
        """
        if ensemble_type not in self.ENSEMBLE_TYPES:
            raise ValueError(f"Ensemble type {ensemble_type} not supported. Must be one of {self.ENSEMBLE_TYPES}")
        
        self.ensemble_type = ensemble_type
        self.models = models or []
        self.weights = weights
        
        # Normalize weights if provided
        if self.weights:
            if len(self.weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            self.weights = np.array(self.weights) / sum(self.weights)
        
        self.meta_model = meta_model
        if ensemble_type == "stacking" and meta_model is None:
            # Default meta-model for stacking
            self.meta_model = LogisticRegression(multi_class='multinomial')
            
        self.dataset_name = dataset_name
        self.feature_type = feature_type
        self.model_info = {
            "ensemble_type": ensemble_type,
            "dataset_name": dataset_name,
            "feature_type": feature_type,
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_count": len(self.models),
            "model_types": [getattr(model, 'model_type', 'unknown') for model in self.models]
        }
        
        # For blending
        self.blend_models = None
        self.feature_pipeline = None
        
        # For calibrated ensembles
        self.calibration_data = None
        self.model_performance = None
        
        # For time-weighted ensembles
        self.time_weights = None
        self.last_update = None
        
        # Load feature pipeline
        self.load_pipeline()
    
    def add_model(self, model: Union[BaselineMatchPredictor, AdvancedMatchPredictor], weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model: Model instance to add
            weight: Weight to assign to this model
        """
        self.models.append(model)
        
        # Update weights
        if self.weights is None:
            self.weights = [1.0] * len(self.models)
        else:
            self.weights.append(weight)
            self.weights = np.array(self.weights) / sum(self.weights)
        
        # Update model info
        self.model_info["model_count"] = len(self.models)
        self.model_info["model_types"] = [getattr(model, 'model_type', 'unknown') for model in self.models]
        
        logger.info(f"Added model to ensemble. Total models: {len(self.models)}")
    
    def remove_model(self, index: int) -> Union[BaselineMatchPredictor, AdvancedMatchPredictor]:
        """
        Remove a model from the ensemble.
        
        Args:
            index: Index of the model to remove
            
        Returns:
            The removed model
        """
        if index < 0 or index >= len(self.models):
            raise ValueError(f"Invalid model index {index}. Must be between 0 and {len(self.models)-1}")
        
        removed_model = self.models.pop(index)
        
        # Update weights
        if self.weights is not None:
            self.weights = np.delete(self.weights, index)
            if len(self.weights) > 0:
                self.weights = self.weights / sum(self.weights)
        
        # Update model info
        self.model_info["model_count"] = len(self.models)
        self.model_info["model_types"] = [getattr(model, 'model_type', 'unknown') for model in self.models]
        
        logger.info(f"Removed model from ensemble. Total models: {len(self.models)}")
        return removed_model
    
    def update_weights(self, weights: List[float]) -> None:
        """
        Update the weights for each model.
        
        Args:
            weights: New weights for each model
        """
        if len(weights) != len(self.models):
            raise ValueError(f"Number of weights ({len(weights)}) does not match number of models ({len(self.models)})")
        
        self.weights = np.array(weights) / sum(weights)
        logger.info(f"Updated ensemble model weights: {self.weights}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              context_features: Optional[np.ndarray] = None) -> dict:
        """
        Train the ensemble.
        
        Args:
            X: Feature matrix
            y: Target labels
            validation_data: Optional validation data tuple (X_val, y_val)
            context_features: Optional context features for context-aware model selection
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {self.ensemble_type} ensemble with {len(self.models)} models")
        
        if len(self.models) == 0:
            logger.error("No models in ensemble, cannot train")
            return {"error": "No models in ensemble"}
        
        # For stacking and blending, we need validation data
        if self.ensemble_type in ["stacking", "blending"] and validation_data is None:
            # Split the data if validation_data is not provided
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
            validation_data = (X_val, y_val)
        else:
            X_train, y_train = X, y
            X_val, y_val = validation_data if validation_data else (None, None)
        
        # Extract context features if not provided
        if context_features is None and self.ensemble_type == "context_aware":
            context_features = self._prepare_context_features(X_train)
            
            # Also extract validation context features
            if X_val is not None:
                val_context_features = self._prepare_context_features(X_val)
        
        # Train based on ensemble type
        if self.ensemble_type == "stacking":
            # For stacking, train a meta-model using base model predictions
            self._train_stacking(X_train, y_train, X_val, y_val)
            
        elif self.ensemble_type == "blending":
            # For blending, train on different data subsets
            self._train_blending(X_train, y_train)
            
        elif self.ensemble_type == "calibrated_voting":
            # For calibrated voting, calibrate base model probabilities
            self._calibrate_weights(X_val, y_val)
            
        elif self.ensemble_type == "time_weighted":
            # For time-weighted, calculate weights based on recency
            self._calculate_time_weights(X_val, y_val)
            
        elif self.ensemble_type == "performance_weighted":
            # For performance-weighted, calculate weights based on performance metrics
            self._calculate_performance_weights(X_val, y_val)
            
        elif self.ensemble_type == "dynamic_weighting":
            # For dynamic weighting, calculate weights with recency bias
            self._calculate_dynamic_weights(X_val, y_val)
        
        elif self.ensemble_type == "context_aware":
            # For context-aware, train a model to select the best model based on context
            self._train_context_model(X_train, y_train, context_features)
            
            # Evaluate on validation data if available
            if X_val is not None and y_val is not None and val_context_features is not None:
                context_weights = self.get_context_specific_weights(val_context_features)
                logger.info(f"Context-aware validation weights shape: {context_weights.shape}")
        
        # Update model info
        self.model_info["trained"] = True
        self.model_info["trained_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.model_info["training_samples"] = len(X_train)
        
        if validation_data:
            self.model_info["validation_samples"] = len(X_val)
        
        return {
            "ensemble_type": self.ensemble_type,
            "n_models": len(self.models),
            "n_samples": len(X_train),
            "trained_at": self.model_info["trained_at"]
        }
    
    def _calibrate_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Calibrate model weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        if len(self.models) == 0:
            return
        
        # Calculate performance metrics for each model
        performances = []
        for model in self.models:
            val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            performances.append(val_accuracy)
        
        # Convert to numpy array and normalize to get weights
        performances = np.array(performances)
        self.weights = performances / sum(performances)
        
        logger.info(f"Calibrated model weights based on validation accuracy: {self.weights}")
    
    def _calculate_performance_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Calculate detailed performance-based weights using multiple metrics.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        if len(self.models) == 0:
            return
        
        # Calculate performance metrics for each model
        performances = []
        self.model_performance = []
        
        for model in self.models:
            val_pred = model.predict(X_val)
            val_pred_proba = model.predict_proba(X_val)
            
            # Calculate multiple performance metrics
            val_accuracy = accuracy_score(y_val, val_pred)
            val_log_loss = log_loss(y_val, val_pred_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, val_pred, average='macro'
            )
            
            # Combine metrics (higher is better)
            # Invert log_loss (lower is better) by using 1/(1+log_loss)
            combined_score = (val_accuracy + f1 + precision + recall + 1/(1+val_log_loss))/5
            performances.append(combined_score)
            
            # Store detailed performance for later use
            self.model_performance.append({
                "accuracy": val_accuracy,
                "log_loss": val_log_loss,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "combined_score": combined_score
            })
        
        # Convert to numpy array and normalize to get weights
        performances = np.array(performances)
        self.weights = performances / sum(performances)
        
        logger.info(f"Calculated performance-based weights: {self.weights}")

    def _get_stacking_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate features for stacking by collecting predictions from all models.
        
        Args:
            X: Input features
            
        Returns:
            Array of stacked model predictions
        """
        # Collect predictions from all models
        all_preds = []
        for model in self.models:
            pred_proba = model.predict_proba(X)
            all_preds.append(pred_proba)
        
        # Stack predictions as features
        return np.hstack(all_preds)
    
    def _blend_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using the blending ensemble approach.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (class predictions, probability predictions)
        """
        if self.blend_models is None or self.meta_model is None:
            raise ValueError("Blend models not initialized. Train the ensemble first.")
        
        # Generate features for meta-model
        blend_features = np.zeros((X.shape[0], len(self.blend_models) * 3))
        for i, model in enumerate(self.blend_models):
            pred_proba = model.predict_proba(X)
            blend_features[:, i*3:(i+1)*3] = pred_proba
        
        # Generate predictions
        pred = self.meta_model.predict(blend_features)
        pred_proba = self.meta_model.predict_proba(blend_features)
        
        return pred, pred_proba
    
    def predict(self, X: np.ndarray, context_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions with the ensemble.
        
        Args:
            X: Feature matrix
            context_features: Optional context features for context-aware model selection
            
        Returns:
            np.ndarray: Predicted classes
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Extract context features if needed
        if context_features is None and self.ensemble_type == "context_aware":
            context_features = self._prepare_context_features(X)
        
        # Get predicted probabilities
        if self.ensemble_type == "context_aware" and context_features is not None:
            # Get context-specific weights for each sample
            context_weights = self.get_context_specific_weights(context_features)
            proba = self._predict_with_context_weights(X, context_weights)
        elif self.ensemble_type == "blending":
            proba, _ = self._blend_predict(X)
        elif self.ensemble_type in ["stacking", "calibrated_voting", "time_weighted", 
                                   "performance_weighted", "dynamic_weighting"]:
            proba = self._get_weighted_probabilities(X)
        else:
            # Standard voting ensemble
            proba = self._get_weighted_probabilities(X)
        
        # Return class with highest probability
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions using the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Array of probability predictions
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models before predicting.")
        
        if self.ensemble_type == "stacking":
            meta_features = self._get_stacking_features(X)
            return self.meta_model.predict_proba(meta_features)
        
        elif self.ensemble_type == "blending":
            _, pred_proba = self._blend_predict(X)
            return pred_proba
        
        # For voting ensembles and others, use weighted voting
        return self._get_weighted_probabilities(X)
    
    def _get_weighted_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate weighted average of probability predictions from all models.
        
        Args:
            X: Input features
            
        Returns:
            Weighted average probability predictions
        """
        # Initialize array for probabilities
        all_probs = np.zeros((X.shape[0], 3))  # Assuming 3 classes
        
        # Apply time decay for time-weighted ensemble
        if self.ensemble_type == "time_weighted" and self.time_weights is not None:
            # Update weights based on time since last update
            current_time = datetime.now()
            days_since_update = (current_time - self.last_update).days
            
            # Apply time decay factor (exponential decay)
            decay_factor = 0.95  # 5% decay per day
            time_decay = np.power(decay_factor, days_since_update * self.time_weights)
            
            # Combine with performance weights
            effective_weights = self.weights * time_decay
            effective_weights = effective_weights / np.sum(effective_weights)
        else:
            effective_weights = self.weights if self.weights is not None else np.ones(len(self.models)) / len(self.models)
        
        # Get weighted predictions
        for i, model in enumerate(self.models):
            pred_proba = model.predict_proba(X)
            weight = effective_weights[i]
            all_probs += weight * pred_proba
        
        return all_probs
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the ensemble model on test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models before evaluating.")
        
        # Generate predictions
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        logloss = log_loss(y, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='macro')
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Evaluate each individual model
        model_metrics = []
        for i, model in enumerate(self.models):
            model_pred = model.predict(X)
            model_pred_proba = model.predict_proba(X)
            
            model_accuracy = accuracy_score(y, model_pred)
            model_logloss = log_loss(y, model_pred_proba)
            model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
                y, model_pred, average='macro'
            )
            
            model_metrics.append({
                "model_type": getattr(model, 'model_type', f'model_{i}'),
                "accuracy": model_accuracy,
                "log_loss": model_logloss,
                "precision": model_precision,
                "recall": model_recall,
                "f1": model_f1,
                "weight": float(self.weights[i]) if self.weights is not None else 1.0/len(self.models)
            })
        
        # Compile all metrics
        metrics = {
            "ensemble_type": self.ensemble_type,
            "accuracy": accuracy,
            "log_loss": logloss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "model_count": len(self.models),
            "individual_model_metrics": model_metrics
        }
        
        return metrics
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the ensemble model to disk.
        
        Args:
            filepath: Path to save the model to. If None, auto-generates a path.
            
        Returns:
            Path where the model was saved
        """
        if filepath is None:
            # Generate filepath
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                ENSEMBLE_MODELS_DIR, 
                f"ensemble_{self.ensemble_type}_{len(self.models)}models_{timestamp}.joblib"
            )
        
        # Prepare model data for saving
        model_data = {
            "ensemble_type": self.ensemble_type,
            "weights": self.weights,
            "model_info": self.model_info,
            "dataset_name": self.dataset_name,
            "feature_type": self.feature_type,
            "meta_model": self.meta_model,
            "blend_models": self.blend_models,
            "calibration_data": self.calibration_data,
            "model_performance": self.model_performance,
            "time_weights": self.time_weights,
            "last_update": self.last_update
        }
        
        # Save model components separately for efficiency
        models_dir = filepath + "_models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save each model
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = os.path.join(models_dir, f"model_{i}.joblib")
            if hasattr(model, 'save'):
                model.save(model_path)
            else:
                joblib.dump(model, model_path)
            model_paths.append(model_path)
        
        model_data["model_paths"] = model_paths
        
        # Save ensemble without models
        joblib.dump(model_data, filepath)
        
        logger.info(f"Saved ensemble model to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "EnsemblePredictor":
        """
        Load an ensemble model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded ensemble model
        """
        # Load model data
        model_data = joblib.load(filepath)
        
        # Check if model paths exist
        model_paths = model_data.get("model_paths", [])
        models = []
        
        for path in model_paths:
            if not os.path.exists(path):
                logger.warning(f"Model file {path} not found. Skipping.")
                continue
            
            try:
                # Try to load as a BaselineMatchPredictor or AdvancedMatchPredictor
                if path.endswith('.h5'):
                    # Special case for Keras models
                    from tensorflow.keras.models import load_model as keras_load_model
                    model = keras_load_model(path)
                else:
                    model = joblib.load(path)
                models.append(model)
            except Exception as e:
                logger.error(f"Error loading model {path}: {e}")
        
        # Create ensemble instance
        ensemble = cls(
            ensemble_type=model_data.get("ensemble_type", "voting"),
            models=models,
            weights=model_data.get("weights"),
            meta_model=model_data.get("meta_model"),
            dataset_name=model_data.get("dataset_name", "transfermarkt"),
            feature_type=model_data.get("feature_type", "match_features")
        )
        
        # Restore additional properties
        ensemble.model_info = model_data.get("model_info", {})
        ensemble.blend_models = model_data.get("blend_models")
        ensemble.calibration_data = model_data.get("calibration_data")
        ensemble.model_performance = model_data.get("model_performance")
        ensemble.time_weights = model_data.get("time_weights")
        ensemble.last_update = model_data.get("last_update")
        
        logger.info(f"Loaded ensemble model from {filepath} with {len(models)} models")
        return ensemble
    
    def load_pipeline(self) -> bool:
        """
        Load the feature transformation pipeline.
        
        Returns:
            True if pipeline was loaded successfully, False otherwise
        """
        try:
            self.feature_pipeline = load_feature_pipeline(self.dataset_name, self.feature_type)
            return True
        except Exception as e:
            logger.warning(f"Could not load feature pipeline: {e}")
            return False
    
    def process_data(self, df: pd.DataFrame, target_col: str = "result") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process data using the feature pipeline.
        
        Args:
            df: DataFrame with raw features
            target_col: Name of the target column
            
        Returns:
            Tuple of (processed features, targets)
        """
        if self.feature_pipeline is None and not self.load_pipeline():
            logger.warning("Feature pipeline not available. Using raw features.")
            
            # Extract features and target
            if target_col in df.columns:
                y = df[target_col].values
                X = df.drop(target_col, axis=1).values
                return X, y
            else:
                X = df.values
                return X, None
        
        # Apply feature pipeline
        return apply_feature_pipeline(df, self.feature_pipeline, target_col)
    
    def predict_match(self, home_team_id: int, away_team_id: int, 
                      features: Optional[Dict[str, Any]] = None, 
                      include_context: bool = True) -> Dict[str, Any]:
        """
        Predict a soccer match outcome.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            features: Optional additional features
            include_context: Whether to include context-aware model selection
            
        Returns:
            Dict[str, Any]: Prediction result
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Create feature dictionary if not provided
        if features is None:
            features = {}
        
        # Add team IDs to features
        features['home_team_id'] = home_team_id
        features['away_team_id'] = away_team_id
        
        # Convert features to DataFrame for processing
        df = pd.DataFrame([features])
        
        # Process data if pipeline is available
        if self.feature_pipeline:
            processed_X = self.pipeline_transform(df)
        else:
            # Basic processing: convert to numpy array
            processed_X = np.array([list(features.values())])
        
        # Extract context features if needed
        if self.ensemble_type == "context_aware" and include_context:
            context_features = self._prepare_context_features(processed_X)
            
            # Get context-specific weights
            context_weights = self.get_context_specific_weights(context_features)
            
            # Make prediction using context-specific weights
            proba = self._predict_with_context_weights(processed_X, context_weights)
        else:
            # Standard prediction
            proba = self.predict_proba(processed_X)
        
        # Convert probabilities to prediction result
        pred_class = np.argmax(proba[0])
        
        # Map prediction class to result
        result_map = {0: 'home_win', 1: 'draw', 2: 'away_win'}
        result = result_map.get(pred_class, str(pred_class))
        
        # Create prediction result dictionary
        prediction = {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'result': result,
            'probabilities': {
                'home_win': float(proba[0, 0]) if proba.shape[1] > 0 else None,
                'draw': float(proba[0, 1]) if proba.shape[1] > 1 else None,
                'away_win': float(proba[0, 2]) if proba.shape[1] > 2 else None
            },
            'ensemble_type': self.ensemble_type,
            'predicted_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add model contributions for interpretability
        if self.ensemble_type == "context_aware" and include_context:
            model_contributions = []
            
            for i, model in enumerate(self.models):
                model_type = getattr(model, 'model_type', f'model_{i}')
                weight = float(context_weights[0, i])
                
                model_contributions.append({
                    'model_type': model_type,
                    'weight': weight
                })
            
            prediction['model_contributions'] = model_contributions
        
        return prediction

    def _calculate_dynamic_weights(self, X_val: np.ndarray, y_val: np.ndarray, 
                                  recency_factor: float = 0.9, window_size: int = 10) -> None:
        """
        Calculate dynamic weights based on the most recent predictions.
        
        This implements a recency-weighted model evaluation, giving higher weight to
        models that performed well in the most recent timeframe.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            recency_factor: Weight decay factor for older observations (0-1)
            window_size: Number of most recent observations to use for local weight calculation
        """
        logger.info("Calculating dynamic model weights")

        if len(self.models) == 0:
            logger.warning("No models in ensemble, cannot calculate dynamic weights")
            return

        # Sort validation data by time if possible
        if hasattr(X_val, 'shape') and len(X_val.shape) == 2 and X_val.shape[1] > 0:
            # Try to find datetime column for sorting
            date_col_idx = None
            for i in range(X_val.shape[1]):
                if hasattr(X_val[:, i], 'dtype') and pd.api.types.is_datetime64_dtype(X_val[:, i].dtype):
                    date_col_idx = i
                    break
            
            if date_col_idx is not None:
                # Sort by date
                sort_idx = np.argsort(X_val[:, date_col_idx])
                X_val = X_val[sort_idx]
                y_val = y_val[sort_idx]

        # Get predictions from each model
        model_preds = []
        model_probs = []
        
        for model in self.models:
            if hasattr(model, 'predict'):
                preds = model.predict(X_val)
                model_preds.append(preds)
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_val)
                model_probs.append(probs)
            elif hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
                # For models without predict_proba, create one-hot encoded probs
                probs = np.zeros((len(y_val), len(np.unique(y_val))))
                for i, pred in enumerate(preds):
                    probs[i, pred] = 1.0
                model_probs.append(probs)
        
        # Calculate recency-weighted performance for each model
        n_samples = len(y_val)
        n_models = len(self.models)
        
        # Initialize performance metrics
        accuracy = np.zeros(n_models)
        log_loss_vals = np.zeros(n_models)
        
        # Calculate exponential decay weights for recency
        decay_weights = np.power(recency_factor, np.arange(n_samples-1, -1, -1))
        decay_weights /= decay_weights.sum()  # Normalize to sum to 1
        
        # Calculate performance metrics
        for i in range(n_models):
            # Accuracy with recency weighting
            correct = (model_preds[i] == y_val).astype(float)
            accuracy[i] = np.sum(correct * decay_weights)
            
            # Log loss with recency weighting if probability predictions are available
            if i < len(model_probs):
                # Get one-hot encoded true labels
                y_one_hot = np.zeros((n_samples, model_probs[i].shape[1]))
                for j, label in enumerate(y_val):
                    y_one_hot[j, label] = 1
                
                # Calculate log loss for each sample
                sample_losses = -np.sum(y_one_hot * np.log(np.clip(model_probs[i], 1e-15, 1.0)), axis=1)
                
                # Apply recency weighting
                log_loss_vals[i] = np.sum(sample_losses * decay_weights)
        
        # Calculate sliding window performance (for recent matches only)
        if window_size > 0 and n_samples >= window_size:
            recent_accuracy = np.zeros(n_models)
            
            for i in range(n_models):
                recent_correct = (model_preds[i][-window_size:] == y_val[-window_size:]).astype(float)
                recent_accuracy[i] = recent_correct.mean()
            
            # Combine global and recent performance
            combined_accuracy = (accuracy + recent_accuracy) / 2
        else:
            combined_accuracy = accuracy
        
        # Normalize to create weights
        weights = combined_accuracy / combined_accuracy.sum() if combined_accuracy.sum() > 0 else np.ones(n_models) / n_models
        
        # Store as instance weights
        self.weights = weights.tolist()
        
        # Store performance metrics for later reference
        self.model_performance = {
            f"model_{i}": {
                "recency_weighted_accuracy": float(accuracy[i]),
                "recency_weighted_log_loss": float(log_loss_vals[i]) if i < len(model_probs) else None,
                "recent_window_accuracy": float(recent_accuracy[i]) if window_size > 0 and n_samples >= window_size else None,
                "combined_weight": float(weights[i])
            } for i in range(n_models)
        }
        
        logger.info(f"Dynamic weights calculated: {self.weights}")

    def _prepare_context_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract context features for context-aware model selection.
        
        Args:
            X: Input features
            
        Returns:
            Context features for model selection
        """
        # Extract context features (e.g., home team strength, away team strength, etc.)
        if not hasattr(X, 'shape') or len(X.shape) < 2:
            return np.zeros((len(X) if hasattr(X, '__len__') else 1, 1))
        
        # Try to identify context columns
        context_features = []
        
        # If X is a DataFrame, use column names
        if hasattr(X, 'columns'):
            context_cols = []
            for col in X.columns:
                if any(term in str(col).lower() for term in 
                       ['form', 'strength', 'elo', 'importance', 'rank', 'position', 'home_', 'away_']):
                    context_cols.append(col)
            
            if context_cols:
                context_features = X[context_cols].values
        
        # If X is a numpy array or context features couldn't be identified, use a subset of features
        if not context_features:
            # Use first few features as context (simple heuristic)
            n_context = min(5, X.shape[1])
            context_features = X[:, :n_context]
        
        return context_features
    
    def _train_context_model(self, X: np.ndarray, y: np.ndarray, context_features: np.ndarray) -> None:
        """
        Train a meta-model to select the best model weights based on context.
        
        Args:
            X: Input features
            y: Target labels
            context_features: Context features for model selection
        """
        logger.info("Training context-aware model selection")
        
        if len(self.models) == 0:
            logger.warning("No models in ensemble, cannot train context model")
            return
        
        # Get predictions from each model
        model_probs = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                model_probs.append(probs)
        
        if not model_probs:
            logger.warning("No probability predictions available, cannot train context model")
            return
        
        # For each sample, determine which model performed best
        n_samples = len(y)
        n_models = len(model_probs)
        
        # Calculate log loss for each model for each sample
        model_losses = np.zeros((n_samples, n_models))
        
        for i in range(n_models):
            # Get one-hot encoded true labels
            y_one_hot = np.zeros((n_samples, model_probs[i].shape[1]))
            for j, label in enumerate(y):
                y_one_hot[j, label] = 1
            
            # Calculate log loss for each sample
            sample_losses = -np.sum(y_one_hot * np.log(np.clip(model_probs[i], 1e-15, 1.0)), axis=1)
            model_losses[:, i] = sample_losses
        
        # For each sample, the best model is the one with minimum loss
        best_models = np.argmin(model_losses, axis=1)
        
        # Train a classifier to predict the best model based on context
        from sklearn.ensemble import RandomForestClassifier
        
        self.context_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.context_model.fit(context_features, best_models)
        
        # Store feature importance for interpretability
        if hasattr(self.context_model, 'feature_importances_'):
            self.context_feature_importance = self.context_model.feature_importances_
            
            # Log top feature importances
            if hasattr(context_features, 'shape') and context_features.shape[1] > 0:
                top_features = np.argsort(self.context_feature_importance)[::-1]
                logger.info(f"Top context features: {top_features[:5]}")
                logger.info(f"Importance values: {self.context_feature_importance[top_features[:5]]}")
        
        logger.info("Context-aware model selection trained successfully")
    
    def get_context_specific_weights(self, context_features: np.ndarray) -> np.ndarray:
        """
        Get model weights specific to the given context.
        
        Args:
            context_features: Context features for model selection
            
        Returns:
            Model weights for the given context
        """
        if not hasattr(self, 'context_model') or self.context_model is None:
            # Fall back to default weights
            return np.array(self.weights) if self.weights is not None else np.ones(len(self.models)) / len(self.models)
        
        # Predict the best model for each context
        best_models = self.context_model.predict(context_features)
        
        # Create one-hot encoded weights (1.0 for best model, 0.0 for others)
        weights = np.zeros((len(context_features), len(self.models)))
        for i, model_idx in enumerate(best_models):
            weights[i, model_idx] = 1.0
        
        # Get model probabilities to allow for soft weighting
        if hasattr(self.context_model, 'predict_proba'):
            model_probs = self.context_model.predict_proba(context_features)
            if hasattr(model_probs, 'shape') and model_probs.shape[1] == len(self.models):
                weights = model_probs
        
        return weights
    
    def _predict_with_context_weights(self, X: np.ndarray, context_weights: np.ndarray) -> np.ndarray:
        """
        Make predictions using context-specific weights.
        
        Args:
            X: Feature matrix
            context_weights: Weights for each model for each sample
            
        Returns:
            Predicted probabilities
        """
        # Get predictions from all models
        model_probs = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                model_probs.append(probs)
            else:
                # For models without predict_proba, create simple probability matrix
                # This is a fallback and should be avoided in practice
                preds = model.predict(X)
                n_classes = len(np.unique(preds))
                probs = np.zeros((len(X), n_classes))
                for i, p in enumerate(preds):
                    probs[i, p] = 1.0
                model_probs.append(probs)
        
        if not model_probs:
            raise ValueError("No probability predictions available")
        
        # Check if all have the same shape
        n_classes = model_probs[0].shape[1]
        if not all(p.shape[1] == n_classes for p in model_probs):
            raise ValueError("Inconsistent number of classes in model predictions")
        
        # Initialize combined probabilities
        combined_proba = np.zeros((len(X), n_classes))
        
        # For each sample, compute weighted average of model predictions
        for i in range(len(X)):
            sample_weights = context_weights[i]
            
            # Normalize weights
            sample_weights = sample_weights / np.sum(sample_weights) if np.sum(sample_weights) > 0 else np.ones(len(self.models)) / len(self.models)
            
            # Compute weighted average
            for j, model_prob in enumerate(model_probs):
                combined_proba[i] += sample_weights[j] * model_prob[i]
        
        # Normalize probabilities
        combined_proba = combined_proba / np.sum(combined_proba, axis=1, keepdims=True)
        
        return combined_proba


def train_ensemble_model(
    ensemble_type: str = "voting",
    model_types: List[str] = ["logistic", "random_forest", "neural_network"],
    dataset_name: str = "transfermarkt",
    feature_type: str = "match_features",
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
    model_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> EnsemblePredictor:
    """
    Train an ensemble model with the specified configuration.
    
    Args:
        ensemble_type: Type of ensemble to create
        model_types: List of model types to include in the ensemble
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        test_size: Fraction of data to use for testing
        validation_size: Fraction of data to use for validation
        random_state: Random seed
        model_params: Optional parameters for each model type
        
    Returns:
        Trained ensemble model
    """
    from src.data.features import load_feature_dataset
    
    logger.info(f"Training {ensemble_type} ensemble with models: {model_types}")
    
    # Load dataset
    data = load_feature_dataset(dataset_name, feature_type)
    if data is None:
        raise ValueError(f"Could not load dataset {dataset_name} with features {feature_type}")
    
    # Process dataset
    X = data['X']
    y = data['y']
    feature_names = data.get('feature_names', [])
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size/(1-test_size), 
        random_state=random_state, stratify=y_train
    )
    
    logger.info(f"Data split - Train: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Initialize and train individual models
    models = []
    for model_type in model_types:
        logger.info(f"Training {model_type} model")
        
        model_param = model_params.get(model_type, {}) if model_params else {}
        
        if model_type in ["logistic", "random_forest", "xgboost"]:
            # Baseline models
            from src.models.baseline import BaselineMatchPredictor
            model = BaselineMatchPredictor(
                model_type=model_type, 
                dataset_name=dataset_name, 
                feature_type=feature_type
            )
        else:
            # Advanced models
            from src.models.advanced import AdvancedMatchPredictor
            model = AdvancedMatchPredictor(
                model_type=model_type,
                dataset_name=dataset_name,
                feature_type=feature_type,
                model_params=model_param
            )
        
        # Train the model
        model.train(X_train, y_train)
        models.append(model)
    
    # Create ensemble
    ensemble = EnsemblePredictor(
        ensemble_type=ensemble_type,
        models=models,
        dataset_name=dataset_name,
        feature_type=feature_type
    )
    
    # Train ensemble
    training_results = ensemble.train(
        X_train, y_train, validation_data=(X_val, y_val)
    )
    
    logger.info(f"Ensemble training completed with results: {training_results}")
    
    # Evaluate on test set
    test_results = ensemble.evaluate(X_test, y_test)
    logger.info(f"Ensemble test evaluation: accuracy={test_results['accuracy']:.4f}, f1={test_results['f1']:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(
        ENSEMBLE_MODELS_DIR, 
        f"{ensemble_type}_ensemble_{'-'.join(model_types)}_{timestamp}.joblib"
    )
    ensemble.save(model_path)
    
    logger.info(f"Ensemble model saved to {model_path}")
    
    return ensemble 