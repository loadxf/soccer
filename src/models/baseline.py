"""
Baseline models for soccer match prediction.
Implements simple but effective baseline models for comparison.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, confusion_matrix
from datetime import datetime
from pathlib import Path

# Import project components
from src.utils.logger import get_logger
from src.data.features import load_feature_pipeline, apply_feature_pipeline

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.baseline")

# Define paths
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


class BaselineMatchPredictor:
    """
    Baseline model for predicting soccer match outcomes.
    Uses logistic regression, random forest, or XGBoost.
    """
    
    def __init__(self, model_type: str = "logistic", dataset_name: str = "transfermarkt", feature_type: str = "match_features"):
        """
        Initialize the baseline predictor.
        
        Args:
            model_type: Type of model to use ("logistic", "random_forest", or "xgboost")
            dataset_name: Name of the dataset to use
            feature_type: Type of features to use
        """
        # Check if XGBoost is available when requested
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available. Falling back to random_forest model.")
            model_type = "random_forest"
            
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.feature_type = feature_type
        self.model = None
        self.pipeline = None
        self.target_encoder = None
        self.model_info = {
            "model_type": model_type,
            "dataset_name": dataset_name,
            "feature_type": feature_type,
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {},
        }
    
    def _create_model(self):
        """Create the model based on model_type."""
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs",
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42
            )
        elif self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is required for this model type. Please install it with 'pip install xgboost'")
                
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective="multi:softprob",
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Created {self.model_type} model")
    
    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> dict:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            sample_weight: Optional sample weights
            
        Returns:
            dict: Training results
        """
        if self.model is None:
            self._create_model()
        
        # Check if we need to fit with sample weights
        if sample_weight is not None and hasattr(self.model, "fit") and "sample_weight" in self.model.fit.__code__.co_varnames:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        
        # Update model info
        self.model_info["trained"] = True
        self.model_info["trained_at"] = datetime.now().isoformat()
        self.model_info["n_samples"] = X.shape[0]
        self.model_info["n_features"] = X.shape[1]
        
        # Return training info
        return {
            "model_type": self.model_type,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "trained_at": self.model_info["trained_at"]
        }
    
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
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions with the model.
        
        Args:
            X: Feature matrix
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if self.model is None or not self.model_info["trained"]:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
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
                MODELS_DIR, 
                f"{self.dataset_name}_{self.feature_type}_{self.model_type}_{timestamp}.pkl"
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model, pipeline, and info
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "pipeline": self.pipeline,
                "target_encoder": self.target_encoder,
                "model_info": self.model_info
            }, f)
        
        logger.info(f"Saved model to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "BaselineMatchPredictor":
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            BaselineMatchPredictor: Loaded model
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            model_type=data["model_info"]["model_type"],
            dataset_name=data["model_info"]["dataset_name"],
            feature_type=data["model_info"]["feature_type"]
        )
        
        # Set the loaded attributes
        instance.model = data["model"]
        instance.pipeline = data["pipeline"]
        instance.target_encoder = data["target_encoder"]
        instance.model_info = data["model_info"]
        
        logger.info(f"Loaded model from {filepath}")
        return instance
    
    def load_pipeline(self) -> bool:
        """
        Load the feature pipeline for this model.
        
        Returns:
            bool: True if pipeline was loaded successfully
        """
        self.pipeline, self.target_encoder = load_feature_pipeline(
            self.dataset_name, self.feature_type
        )
        
        if self.pipeline is None:
            logger.error(f"Failed to load pipeline for {self.dataset_name}/{self.feature_type}")
            return False
        
        logger.info(f"Loaded feature pipeline for {self.dataset_name}/{self.feature_type}")
        return True
    
    def process_data(self, df: pd.DataFrame, target_col: str = "result") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process data through the feature pipeline.
        
        Args:
            df: DataFrame with raw features
            target_col: Name of the target column
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Processed features and target
        """
        if self.pipeline is None:
            if not self.load_pipeline():
                raise ValueError("Feature pipeline not available")
        
        from src.data.features import apply_feature_pipeline
        X, y = apply_feature_pipeline(df, self.pipeline, self.target_encoder, target_col)
        
        return X, y
    
    def predict_match(self, home_team_id: int, away_team_id: int, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a prediction for a match between two teams.
        This is a simplified version that would need real feature engineering in production.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            features: Optional dictionary of additional features
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if self.model is None or not self.model_info["trained"]:
            raise ValueError("Model not trained yet")
        
        if self.pipeline is None:
            if not self.load_pipeline():
                raise ValueError("Feature pipeline not available")
        
        # Create a minimal feature set (this would be more complex in a real implementation)
        feature_dict = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            # Add dummy features for demonstration
            "home_goals_scored": 1.5,
            "home_goals_conceded": 1.0,
            "away_goals_scored": 1.2,
            "away_goals_conceded": 1.1,
            "home_win_streak": 2,
            "away_win_streak": 1,
            "h2h_home_wins": 2,
            "h2h_away_wins": 1,
            "h2h_draws": 1,
        }
        
        # Update with any provided features
        if features is not None:
            feature_dict.update(features)
        
        # Convert to DataFrame
        df = pd.DataFrame([feature_dict])
        
        # Apply feature pipeline
        X, _ = self.process_data(df, target_col=None)
        
        # Get predictions
        probabilities = self.predict_proba(X)[0]
        
        # Create result
        result = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_win_probability": float(probabilities[self.target_encoder.transform(["home_win"])[0]]),
            "draw_probability": float(probabilities[self.target_encoder.transform(["draw"])[0]]),
            "away_win_probability": float(probabilities[self.target_encoder.transform(["away_win"])[0]]),
            "prediction": self.target_encoder.inverse_transform([np.argmax(probabilities)])[0],
            "confidence": float(np.max(probabilities)),
            "predicted_at": datetime.now().isoformat(),
            "model_type": self.model_type
        }
        
        return result


def train_baseline_model(
    dataset_name: str = "transfermarkt",
    feature_type: str = "match_features",
    model_type: str = "logistic",
    test_size: float = 0.2,
    random_state: int = 42
) -> BaselineMatchPredictor:
    """
    Train a baseline model from scratch.
    
    Args:
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        model_type: Type of model to use
        test_size: Portion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        BaselineMatchPredictor: Trained model
    """
    from sklearn.model_selection import train_test_split
    
    # Load feature data
    features_path = os.path.join(FEATURES_DIR, dataset_name, f"{feature_type}.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    
    df = pd.read_csv(features_path)
    logger.info(f"Loaded feature dataset with {len(df)} samples")
    
    # Create model
    model = BaselineMatchPredictor(model_type=model_type, dataset_name=dataset_name, feature_type=feature_type)
    
    # Load pipeline
    if not model.load_pipeline():
        raise ValueError("Failed to load feature pipeline")
    
    # Process data
    X, y = model.process_data(df, target_col="result")
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Train model
    train_result = model.train(X_train, y_train)
    logger.info(f"Trained model: {train_result}")
    
    # Evaluate model
    eval_result = model.evaluate(X_test, y_test)
    logger.info(f"Model evaluation: accuracy={eval_result['accuracy']:.4f}, f1={eval_result['f1']:.4f}")
    
    # Save model
    model_path = model.save()
    logger.info(f"Saved model to {model_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a baseline model for soccer match prediction")
    parser.add_argument("--dataset", type=str, default="transfermarkt",
                        help="Dataset to use (default: transfermarkt)")
    parser.add_argument("--feature-type", type=str, default="match_features",
                        help="Type of features to use (default: match_features)")
    parser.add_argument("--model-type", type=str, choices=["logistic", "random_forest", "xgboost"], default="logistic",
                        help="Type of model to use (default: logistic)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Portion of data to use for testing (default: 0.2)")
    
    args = parser.parse_args()
    
    try:
        model = train_baseline_model(
            dataset_name=args.dataset,
            feature_type=args.feature_type,
            model_type=args.model_type,
            test_size=args.test_size
        )
        
        # Example prediction
        prediction = model.predict_match(1, 2)
        logger.info(f"Example prediction: {prediction}")
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise 