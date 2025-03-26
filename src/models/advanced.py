"""
Advanced models for soccer match prediction.
Implements more sophisticated models beyond baselines, including neural networks and time series models.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Conditionally import heavy ML libraries
# These imports will now only fail when actually using these models, not on import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, confusion_matrix

# Import project components
from src.utils.logger import get_logger
from src.data.features import load_feature_pipeline, apply_feature_pipeline
from src.models.baseline import BaselineMatchPredictor

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.advanced")

# Define paths
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
ADVANCED_MODELS_DIR = os.path.join(MODELS_DIR, "advanced")
os.makedirs(ADVANCED_MODELS_DIR, exist_ok=True)


class AdvancedMatchPredictor:
    """
    Advanced model for predicting soccer match outcomes.
    Implements neural networks, boosting models, and time series forecasting.
    
    Extends the functionality of BaselineMatchPredictor with more sophisticated models.
    """
    
    MODEL_TYPES = [
        "neural_network", 
        "lightgbm", 
        "catboost", 
        "deep_ensemble",
        "time_series"
    ]
    
    def __init__(self, model_type: str = "neural_network", dataset_name: str = "transfermarkt", 
                 feature_type: str = "match_features", model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced predictor.
        
        Args:
            model_type: Type of model to use (neural_network, lightgbm, catboost, deep_ensemble, time_series)
            dataset_name: Name of the dataset to use
            feature_type: Type of features to use
            model_params: Optional model parameters
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of {self.MODEL_TYPES}")
        
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.feature_type = feature_type
        self.model_params = model_params or {}
        self.model = None
        self.pipeline = None
        self.scaler = None
        self.target_encoder = None
        self.classes = None
        self.is_keras_model = model_type in ["neural_network", "deep_ensemble"]
        
        self.model_info = {
            "model_type": model_type,
            "dataset_name": dataset_name,
            "feature_type": feature_type,
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {},
        }
    
    def _create_model(self, input_dim: int = None, num_classes: int = 3):
        """
        Create the model based on model_type.
        
        Args:
            input_dim: Number of input features (required for neural networks)
            num_classes: Number of output classes
        """
        if self.model_type == "neural_network":
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for neural network models. Please install it with 'pip install tensorflow'")
            
            if input_dim is None:
                raise ValueError("input_dim is required for neural network models")
            
            # Get model parameters with defaults
            units = self.model_params.get('units', [128, 64, 32])
            dropout_rate = self.model_params.get('dropout_rate', 0.3)
            learning_rate = self.model_params.get('learning_rate', 0.001)
            activation = self.model_params.get('activation', 'relu')
            
            # Create model
            model = Sequential()
            model.add(Dense(units[0], input_dim=input_dim, activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            for unit in units[1:]:
                model.add(Dense(unit, activation=activation))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
            
            model.add(Dense(num_classes, activation='softmax'))
            
            # Compile model
            model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=Adam(learning_rate=learning_rate),
                metrics=['accuracy']
            )
            
            self.model = model
            
        elif self.model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM is required for LightGBM models. Please install it with 'pip install lightgbm'")
            
            # Get model parameters with defaults
            params = {
                'n_estimators': self.model_params.get('n_estimators', 200),
                'learning_rate': self.model_params.get('learning_rate', 0.05),
                'max_depth': self.model_params.get('max_depth', 7),
                'num_leaves': self.model_params.get('num_leaves', 31),
                'colsample_bytree': self.model_params.get('colsample_bytree', 0.8),
                'subsample': self.model_params.get('subsample', 0.8),
                'reg_alpha': self.model_params.get('reg_alpha', 0.1),
                'reg_lambda': self.model_params.get('reg_lambda', 0.1),
                'random_state': 42,
                'class_weight': 'balanced',
                'objective': 'multiclass',
                'num_class': num_classes
            }
            
            self.model = LGBMClassifier(**params)
            
        elif self.model_type == "catboost":
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost is required for CatBoost models. Please install it with 'pip install catboost'")
            
            # Get model parameters with defaults
            params = {
                'iterations': self.model_params.get('iterations', 500),
                'learning_rate': self.model_params.get('learning_rate', 0.05),
                'depth': self.model_params.get('depth', 6),
                'l2_leaf_reg': self.model_params.get('l2_leaf_reg', 3),
                'random_seed': 42,
                'loss_function': 'MultiClass',
                'classes_count': num_classes,
                'verbose': self.model_params.get('verbose', False)
            }
            
            self.model = cb.CatBoostClassifier(**params)
            
        elif self.model_type == "deep_ensemble":
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow is required for deep ensemble models. Please install it with 'pip install tensorflow'")
                
            if input_dim is None:
                raise ValueError("input_dim is required for deep ensemble models")
            
            # Create an ensemble of neural networks
            ensemble_size = self.model_params.get('ensemble_size', 5)
            units = self.model_params.get('units', [128, 64, 32])
            dropout_rate = self.model_params.get('dropout_rate', 0.3)
            learning_rate = self.model_params.get('learning_rate', 0.001)
            
            models = []
            for i in range(ensemble_size):
                model = Sequential()
                model.add(Dense(units[0], input_dim=input_dim, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
                
                for unit in units[1:]:
                    model.add(Dense(unit, activation='relu'))
                    model.add(BatchNormalization())
                    model.add(Dropout(dropout_rate))
                
                model.add(Dense(num_classes, activation='softmax'))
                
                model.compile(
                    loss='sparse_categorical_crossentropy',
                    optimizer=Adam(learning_rate=learning_rate),
                    metrics=['accuracy']
                )
                
                models.append(model)
            
            self.model = models
            
        elif self.model_type == "time_series":
            # For time series, we'll use Prophet for each outcome and combine results
            # This will be initialized when training with the actual data
            self.model = {}
        
        logger.info(f"Created {self.model_type} model")
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, 
              sample_weight: Optional[np.ndarray] = None) -> dict:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_data: Optional validation data for early stopping
            sample_weight: Optional sample weights
            
        Returns:
            dict: Training results
        """
        # Scale the features for neural network models
        if self.model_type in ["neural_network", "deep_ensemble"]:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Store classes for prediction
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        
        # Create model if not already created
        if self.model is None:
            self._create_model(input_dim=X.shape[1], num_classes=num_classes)
        
        # Train model based on type
        if self.model_type == "neural_network":
            # Prepare validation data if provided
            val_data = None
            if validation_data is not None:
                val_X, val_y = validation_data
                val_X_scaled = self.scaler.transform(val_X)
                val_data = (val_X_scaled, val_y)
            
            # Create callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss' if val_data else 'loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    os.path.join(ADVANCED_MODELS_DIR, f"nn_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"),
                    monitor='val_loss' if val_data else 'loss',
                    save_best_only=True
                )
            ]
            
            # Train the model
            epochs = self.model_params.get('epochs', 100)
            batch_size = self.model_params.get('batch_size', 32)
            
            history = self.model.fit(
                X_scaled, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                callbacks=callbacks,
                verbose=self.model_params.get('verbose', 1),
                sample_weight=sample_weight
            )
            
            training_results = {
                "model_type": self.model_type,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "epochs": len(history.history['loss']),
                "final_loss": float(history.history['loss'][-1]),
                "final_accuracy": float(history.history['accuracy'][-1])
            }
            
            if val_data:
                training_results["val_loss"] = float(history.history['val_loss'][-1])
                training_results["val_accuracy"] = float(history.history['val_accuracy'][-1])
        
        elif self.model_type == "lightgbm" or self.model_type == "catboost":
            # Train the model
            eval_set = None
            if validation_data is not None:
                val_X, val_y = validation_data
                eval_set = [(val_X, val_y)]
            
            if self.model_type == "lightgbm":
                # LightGBM training
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    early_stopping_rounds=10 if eval_set else None,
                    verbose=self.model_params.get('verbose', False),
                    sample_weight=sample_weight
                )
            else:
                # CatBoost training
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    early_stopping_rounds=10 if eval_set else None,
                    sample_weight=sample_weight
                )
            
            training_results = {
                "model_type": self.model_type,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_estimators": self.model.n_estimators_,
                "best_iteration": self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else None
            }
        
        elif self.model_type == "deep_ensemble":
            # Train each model in the ensemble
            ensemble_results = []
            
            # Prepare validation data if provided
            val_data = None
            if validation_data is not None:
                val_X, val_y = validation_data
                val_X_scaled = self.scaler.transform(val_X)
                val_data = (val_X_scaled, val_y)
            
            # Train each model
            epochs = self.model_params.get('epochs', 100)
            batch_size = self.model_params.get('batch_size', 32)
            
            for i, model in enumerate(self.model):
                callbacks = [
                    EarlyStopping(monitor='val_loss' if val_data else 'loss', patience=10, restore_best_weights=True)
                ]
                
                history = model.fit(
                    X_scaled, y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=val_data,
                    callbacks=callbacks,
                    verbose=self.model_params.get('verbose', 0),
                    sample_weight=sample_weight
                )
                
                model_result = {
                    "model_index": i,
                    "epochs": len(history.history['loss']),
                    "final_loss": float(history.history['loss'][-1]),
                    "final_accuracy": float(history.history['accuracy'][-1])
                }
                
                if val_data:
                    model_result["val_loss"] = float(history.history['val_loss'][-1])
                    model_result["val_accuracy"] = float(history.history['val_accuracy'][-1])
                
                ensemble_results.append(model_result)
            
            training_results = {
                "model_type": self.model_type,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "ensemble_size": len(self.model),
                "ensemble_results": ensemble_results
            }
        
        elif self.model_type == "time_series":
            # For time series, we train a separate model for each outcome
            # Extract temporal features if available
            date_col = self.model_params.get('date_column', 'match_date')
            if date_col not in X:
                raise ValueError(f"Date column '{date_col}' is required for time series modeling but not found in features")
            
            # Train a model for each possible outcome
            for outcome in self.classes:
                # Create Prophet model
                m = Prophet(
                    yearly_seasonality=self.model_params.get('yearly_seasonality', True),
                    weekly_seasonality=self.model_params.get('weekly_seasonality', True),
                    daily_seasonality=self.model_params.get('daily_seasonality', False),
                    seasonality_mode=self.model_params.get('seasonality_mode', 'additive')
                )
                
                # Add additional regressors (features)
                feature_names = self.model_params.get('feature_names', [])
                for feature in feature_names:
                    if feature in X:
                        m.add_regressor(feature)
                
                # Prepare dataframe for Prophet
                df = pd.DataFrame({
                    'ds': X[date_col],
                    'y': (y == outcome).astype(int)  # Binary indicator for this outcome
                })
                
                # Add regressors
                for feature in feature_names:
                    if feature in X:
                        df[feature] = X[feature]
                
                # Fit the model
                m.fit(df)
                
                # Store the model for this outcome
                self.model[str(outcome)] = m
            
            training_results = {
                "model_type": self.model_type,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "outcomes": [str(c) for c in self.classes]
            }
        
        # Update model info
        self.model_info["trained"] = True
        self.model_info["trained_at"] = datetime.now().isoformat()
        self.model_info["n_samples"] = X.shape[0]
        self.model_info["n_features"] = X.shape[1]
        self.model_info["training_results"] = training_results
        
        return training_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predicted classes
        """
        if self.model is None or not self.model_info["trained"]:
            raise ValueError("Model not trained yet")
        
        # Make probability predictions first
        probas = self.predict_proba(X)
        
        # Return the class with highest probability
        return np.array([self.classes[i] for i in np.argmax(probas, axis=1)])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions with the model.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        if self.model is None or not self.model_info["trained"]:
            raise ValueError("Model not trained yet")
        
        # Scale the features for neural network models
        if self.model_type in ["neural_network", "deep_ensemble"] and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions based on model type
        if self.model_type == "neural_network":
            return self.model.predict(X_scaled)
        
        elif self.model_type in ["lightgbm", "catboost"]:
            return self.model.predict_proba(X)
        
        elif self.model_type == "deep_ensemble":
            # Average predictions from all models in the ensemble
            ensemble_preds = np.array([model.predict(X_scaled) for model in self.model])
            return np.mean(ensemble_preds, axis=0)
        
        elif self.model_type == "time_series":
            # For time series, combine predictions from individual outcome models
            date_col = self.model_params.get('date_column', 'match_date')
            if date_col not in X:
                raise ValueError(f"Date column '{date_col}' is required for time series prediction but not found in features")
            
            # Get predictions for each outcome
            outcome_probs = np.zeros((X.shape[0], len(self.classes)))
            
            feature_names = self.model_params.get('feature_names', [])
            
            for i, outcome in enumerate(self.classes):
                m = self.model[str(outcome)]
                
                # Prepare dataframe for Prophet
                future = pd.DataFrame({'ds': X[date_col]})
                
                # Add regressors
                for feature in feature_names:
                    if feature in X:
                        future[feature] = X[feature]
                
                # Make prediction
                forecast = m.predict(future)
                
                # Store probability for this outcome
                outcome_probs[:, i] = forecast['yhat'].values
            
            # Normalize to get valid probabilities
            row_sums = outcome_probs.sum(axis=1)
            outcome_probs = outcome_probs / row_sums[:, np.newaxis]
            
            return outcome_probs
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None or not self.model_info["trained"]:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        acc = accuracy_score(y, y_pred)
        ll = log_loss(y, y_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted")
        cm = confusion_matrix(y, y_pred)
        
        # Create evaluation results
        evaluation = {
            "accuracy": float(acc),
            "log_loss": float(ll),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "evaluated_at": datetime.now().isoformat(),
            "n_samples": X.shape[0]
        }
        
        # Update model info
        self.model_info["performance"] = evaluation
        
        return evaluation
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Default path if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                ADVANCED_MODELS_DIR, 
                f"{self.dataset_name}_{self.feature_type}_{self.model_type}_{timestamp}.pkl"
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # For Keras models, save model architecture and weights separately
        if self.model_type == "neural_network":
            model_architecture = self.model.to_json()
            weights_path = filepath.replace(".pkl", "_weights.h5")
            self.model.save_weights(weights_path)
            
            # Save everything except the model
            with open(filepath, "wb") as f:
                pickle.dump({
                    "model_architecture": model_architecture,
                    "weights_path": weights_path,
                    "scaler": self.scaler,
                    "pipeline": self.pipeline,
                    "target_encoder": self.target_encoder,
                    "classes": self.classes,
                    "model_info": self.model_info,
                    "is_keras_model": True,
                    "model_type": self.model_type
                }, f)
        
        elif self.model_type == "deep_ensemble":
            # Save each model in the ensemble
            ensemble_paths = []
            for i, model in enumerate(self.model):
                model_path = filepath.replace(".pkl", f"_ensemble_{i}.h5")
                model.save(model_path)
                ensemble_paths.append(model_path)
            
            # Save everything except the models
            with open(filepath, "wb") as f:
                pickle.dump({
                    "ensemble_paths": ensemble_paths,
                    "scaler": self.scaler,
                    "pipeline": self.pipeline,
                    "target_encoder": self.target_encoder,
                    "classes": self.classes,
                    "model_info": self.model_info,
                    "is_keras_model": True,
                    "model_type": self.model_type,
                    "ensemble_size": len(self.model)
                }, f)
        
        else:
            # Save everything in one file for other model types
            with open(filepath, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "scaler": self.scaler,
                    "pipeline": self.pipeline,
                    "target_encoder": self.target_encoder,
                    "classes": self.classes,
                    "model_info": self.model_info,
                    "is_keras_model": False,
                    "model_type": self.model_type
                }, f)
        
        logger.info(f"Saved model to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "AdvancedMatchPredictor":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            AdvancedMatchPredictor: Loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the saved data
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        # Get model type and create instance
        model_type = data.get("model_type", "neural_network")
        instance = cls(model_type=model_type)
        
        # Set attributes from saved data
        instance.pipeline = data.get("pipeline")
        instance.target_encoder = data.get("target_encoder")
        instance.model_info = data.get("model_info", {})
        instance.classes = data.get("classes")
        instance.scaler = data.get("scaler")
        
        # Handle loading based on model type
        is_keras_model = data.get("is_keras_model", False)
        
        if is_keras_model:
            if model_type == "neural_network":
                # Load model architecture and weights
                model_architecture = data.get("model_architecture")
                weights_path = data.get("weights_path")
                
                if model_architecture and os.path.exists(weights_path):
                    instance.model = tf.keras.models.model_from_json(model_architecture)
                    instance.model.load_weights(weights_path)
                    logger.info(f"Loaded neural network model from {filepath} and weights from {weights_path}")
                else:
                    raise ValueError("Could not load neural network model: missing architecture or weights")
            
            elif model_type == "deep_ensemble":
                # Load each model in the ensemble
                ensemble_paths = data.get("ensemble_paths", [])
                ensemble_size = data.get("ensemble_size", 0)
                
                if ensemble_paths:
                    instance.model = []
                    for path in ensemble_paths:
                        if os.path.exists(path):
                            model = load_model(path)
                            instance.model.append(model)
                    
                    logger.info(f"Loaded deep ensemble with {len(instance.model)} models from {filepath}")
                    
                    if len(instance.model) != ensemble_size:
                        logger.warning(f"Expected {ensemble_size} models but loaded {len(instance.model)}")
                else:
                    raise ValueError("Could not load deep ensemble: missing model paths")
        else:
            # For other model types, just load the model directly
            instance.model = data.get("model")
            logger.info(f"Loaded {model_type} model from {filepath}")
        
        return instance
    
    def load_pipeline(self) -> bool:
        """
        Load feature pipeline to process raw match data.
        
        Returns:
            bool: True if pipeline loaded successfully
        """
        try:
            self.pipeline = load_feature_pipeline(self.dataset_name, self.feature_type)
            return True
        except Exception as e:
            logger.error(f"Error loading feature pipeline: {e}")
            return False
    
    def process_data(self, df: pd.DataFrame, target_col: str = "result") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process raw data with the feature pipeline.
        
        Args:
            df: Raw data as DataFrame
            target_col: Name of the target column
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Processed features and targets
        """
        # Ensure pipeline is loaded
        if self.pipeline is None and not self.load_pipeline():
            return None, None
        
        # Process data
        try:
            X, y, target_encoder = apply_feature_pipeline(
                df, self.pipeline, target_col=target_col
            )
            self.target_encoder = target_encoder
            return X, y
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None, None
    
    def predict_match(self, home_team_id: int, away_team_id: int, 
                      features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Predict the outcome of a match between two teams.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            features: Optional additional features for prediction
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if self.model is None or not self.model_info["trained"]:
            raise ValueError("Model not trained yet")
        
        if self.pipeline is None:
            self.load_pipeline()
        
        if self.pipeline is None:
            raise ValueError("Feature pipeline not available")
        
        # Create a match sample
        match_sample = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "match_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Add any additional features
        if features:
            match_sample.update(features)
        
        # Convert to DataFrame
        df = pd.DataFrame([match_sample])
        
        # Process data
        X_match, _ = self.process_data(df, target_col=None)
        
        if X_match is None:
            raise ValueError("Failed to process match data")
        
        # Make predictions
        proba = self.predict_proba(X_match)[0]
        
        # Map probabilities to outcomes
        outcome_probs = {}
        for i, cls in enumerate(self.classes):
            outcome_name = self.target_encoder.inverse_transform([cls])[0] if self.target_encoder else str(cls)
            outcome_probs[outcome_name] = float(proba[i])
        
        # Get most likely outcome
        most_likely_idx = np.argmax(proba)
        most_likely_class = self.classes[most_likely_idx]
        most_likely_outcome = self.target_encoder.inverse_transform([most_likely_class])[0] if self.target_encoder else str(most_likely_class)
        
        # Create prediction result
        result = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "prediction": most_likely_outcome,
            "probabilities": outcome_probs,
            "confidence": float(proba[most_likely_idx]),
            "model_type": self.model_type,
            "prediction_time": datetime.now().isoformat()
        }
        
        return result


def train_advanced_model(
    model_type: str = "neural_network",
    dataset_name: str = "transfermarkt",
    feature_type: str = "match_features",
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
    model_params: Optional[Dict[str, Any]] = None
) -> AdvancedMatchPredictor:
    """
    Train an advanced model with the specified configuration.
    
    Args:
        model_type: Type of model to train
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        test_size: Portion of data to use for testing
        validation_size: Portion of training data to use for validation
        random_state: Random seed for reproducibility
        model_params: Optional model parameters
        
    Returns:
        AdvancedMatchPredictor: Trained model
    """
    from src.models.training import load_feature_data
    
    # Load data
    try:
        df = load_feature_data(dataset_name, feature_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading feature data: {e}")
        raise
    
    # Create model
    model = AdvancedMatchPredictor(
        model_type=model_type, 
        dataset_name=dataset_name, 
        feature_type=feature_type,
        model_params=model_params
    )
    
    # Load pipeline
    if not model.load_pipeline():
        raise ValueError("Failed to load feature pipeline")
    
    # Process data
    X, y = model.process_data(df)
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split training data to create validation set if needed
    if model_type in ["neural_network", "deep_ensemble"] and validation_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=validation_size / (1 - test_size),
            random_state=random_state,
            stratify=y_train
        )
        validation_data = (X_val, y_val)
    else:
        validation_data = None
    
    # Train model
    logger.info(f"Training {model_type} model on {len(X_train)} samples...")
    model.train(X_train, y_train, validation_data=validation_data)
    
    # Evaluate model
    evaluation = model.evaluate(X_test, y_test)
    logger.info(f"Model evaluation: accuracy={evaluation['accuracy']:.4f}, f1={evaluation['f1']:.4f}")
    
    # Save model
    model_path = model.save()
    logger.info(f"Saved model to {model_path}")
    
    return model 