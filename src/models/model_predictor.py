"""
Model predictor interface for the soccer prediction system.

This module provides a standard interface for prediction models 
to interact with the betting system, regardless of their implementation.
"""

from typing import Dict, List, Union, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
import pickle
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelPredictor(ABC):
    """
    Abstract base class for model predictors.
    
    This class defines the interface that all prediction models must implement
    to interact with the betting system. It provides methods for predicting
    match outcomes, extracting probabilities, and calculating confidence scores.
    """
    
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the model.
        
        Args:
            features: DataFrame containing features for prediction
            
        Returns:
            DataFrame with prediction results
        """
        pass
    
    @abstractmethod
    def predict_match(self, match_id: str, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make a prediction for a specific match.
        
        Args:
            match_id: Unique identifier for the match
            features: DataFrame containing features for the match
            
        Returns:
            Dictionary containing prediction results for the match
        """
        pass
    
    @abstractmethod
    def get_confidence(self, prediction: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a prediction.
        
        Args:
            prediction: Prediction dictionary from predict_match
            
        Returns:
            Confidence score between 0 and 1
        """
        pass
    
    @property
    @abstractmethod
    def supported_markets(self) -> List[str]:
        """
        List of betting markets supported by this predictor.
        
        Returns:
            List of market names (e.g., '1X2', 'over_under', 'btts')
        """
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model predictor to a file.
        
        Args:
            path: Path to save the model
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ModelPredictor':
        """
        Load a model predictor from a file.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded model predictor instance
        """
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            
            if not isinstance(model, cls):
                raise TypeError(f"Loaded object is not a {cls.__name__}")
                
            return model
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            raise


class XGBoostPredictor(ModelPredictor):
    """
    Model predictor implementation using XGBoost models.
    
    This predictor uses trained XGBoost models to predict various
    soccer match outcomes. It supports multiple markets and provides
    calibrated probability outputs.
    """
    
    def __init__(self, 
                models: Dict[str, Any], 
                feature_columns: List[str],
                calibrators: Optional[Dict[str, Any]] = None,
                confidence_method: str = 'probability',
                feature_importance: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize the XGBoost predictor.
        
        Args:
            models: Dictionary mapping market names to XGBoost model objects
            feature_columns: List of feature column names used by the models
            calibrators: Optional dictionary of probability calibrators
            confidence_method: Method to calculate confidence ('probability', 'margin', or 'ensemble')
            feature_importance: Optional dictionary of feature importance values
        """
        self._models = models
        self._feature_columns = feature_columns
        self._calibrators = calibrators or {}
        self._confidence_method = confidence_method
        self._feature_importance = feature_importance or {}
        
        # Validate the models
        for market, model in self._models.items():
            if not hasattr(model, 'predict_proba') and not hasattr(model, 'predict'):
                raise ValueError(f"Model for market '{market}' must have predict or predict_proba method")
    
    @property
    def supported_markets(self) -> List[str]:
        """
        List markets supported by this predictor.
        
        Returns:
            List of market names
        """
        return list(self._models.keys())
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple matches.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            DataFrame with prediction results for all supported markets
        """
        # Ensure all required features are present
        missing_features = set(self._feature_columns) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare the input features
        X = features[self._feature_columns].copy()
        
        # Initialize results dictionary
        results = {}
        
        # Make predictions for each market
        for market, model in self._models.items():
            try:
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                    
                    # Apply calibration if available
                    if market in self._calibrators:
                        probas = self._calibrators[market].transform(probas)
                    
                    # For binary classification
                    if probas.shape[1] == 2:
                        results[f"{market}_prob"] = probas[:, 1]
                    # For multiclass classification
                    else:
                        for i in range(probas.shape[1]):
                            results[f"{market}_prob_{i}"] = probas[:, i]
                
                # Get predictions
                preds = model.predict(X)
                results[market] = preds
                
            except Exception as e:
                logger.error(f"Error predicting for market {market}: {str(e)}")
                # Don't fail completely, just skip this market
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results, index=features.index)
        
        # Calculate confidence scores
        if self._confidence_method == 'probability':
            # Use highest probability as confidence
            confidence = []
            for _, row in results_df.iterrows():
                max_prob = 0.0
                for col in row.index:
                    if col.endswith('_prob') or '_prob_' in col:
                        if row[col] > max_prob:
                            max_prob = row[col]
                confidence.append(max_prob)
            
            results_df['confidence'] = confidence
            
        elif self._confidence_method == 'margin':
            # Use margin between top probabilities as confidence
            confidence = []
            for _, row in results_df.iterrows():
                probs = []
                for col in row.index:
                    if col.endswith('_prob') or '_prob_' in col:
                        probs.append(row[col])
                
                if len(probs) >= 2:
                    # Sort probabilities in descending order
                    probs.sort(reverse=True)
                    # Margin between top two probabilities
                    margin = probs[0] - probs[1]
                    # Scale to [0, 1]
                    conf = min(1.0, margin * 2)
                else:
                    conf = probs[0] if probs else 0.5
                
                confidence.append(conf)
            
            results_df['confidence'] = confidence
        
        return results_df
    
    def predict_match(self, match_id: str, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions for a specific match.
        
        Args:
            match_id: Unique identifier for the match
            features: DataFrame containing features for the match
            
        Returns:
            Dictionary with prediction results for the match
        """
        if match_id not in features.index:
            raise ValueError(f"Match ID {match_id} not found in features DataFrame")
        
        # Get features for this match
        match_features = features.loc[[match_id]]
        
        # Get predictions
        predictions = self.predict(match_features)
        
        # Convert to dictionary
        result = {'match_id': match_id}
        for col in predictions.columns:
            result[col] = predictions.loc[match_id, col]
        
        # Add feature importance if available
        if self._feature_importance:
            important_features = {}
            for market, importances in self._feature_importance.items():
                if market in self._models:
                    # Sort feature importances
                    sorted_idx = np.argsort(importances)[::-1]
                    top_features = [
                        (self._feature_columns[idx], float(importances[idx]))
                        for idx in sorted_idx[:10]  # Top 10 features
                    ]
                    important_features[market] = top_features
            
            result['important_features'] = important_features
        
        return result
    
    def get_confidence(self, prediction: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a prediction.
        
        Args:
            prediction: Prediction dictionary from predict_match
            
        Returns:
            Confidence score between 0 and 1
        """
        if 'confidence' in prediction:
            return float(prediction['confidence'])
        
        # If confidence wasn't pre-calculated, use highest probability
        max_prob = 0.0
        for key, value in prediction.items():
            if key.endswith('_prob') or '_prob_' in key:
                if isinstance(value, (int, float)) and value > max_prob:
                    max_prob = value
        
        return max_prob
    
    @classmethod
    def from_saved_models(cls, 
                        base_path: Union[str, Path],
                        markets: List[str],
                        feature_list_path: Optional[Union[str, Path]] = None,
                        calibrators_path: Optional[Union[str, Path]] = None,
                        confidence_method: str = 'probability') -> 'XGBoostPredictor':
        """
        Create a predictor from saved XGBoost models.
        
        Args:
            base_path: Base directory containing saved models
            markets: List of markets to load models for
            feature_list_path: Path to the feature list file
            calibrators_path: Path to the calibrators file
            confidence_method: Method to calculate confidence
            
        Returns:
            XGBoostPredictor instance
        """
        base_path = Path(base_path)
        
        # Load models
        models = {}
        for market in markets:
            model_path = base_path / f"{market}_model.pkl"
            if model_path.exists():
                try:
                    models[market] = joblib.load(model_path)
                except Exception as e:
                    logger.error(f"Error loading model for {market}: {str(e)}")
        
        if not models:
            raise ValueError(f"No models could be loaded from {base_path}")
        
        # Load feature list
        if feature_list_path is None:
            feature_list_path = base_path / "feature_columns.pkl"
        
        try:
            with open(feature_list_path, 'rb') as f:
                feature_columns = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading feature list: {str(e)}")
            raise
        
        # Load calibrators if available
        calibrators = {}
        if calibrators_path is not None:
            try:
                with open(calibrators_path, 'rb') as f:
                    calibrators = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading calibrators: {str(e)}")
        
        # Load feature importance if available
        feature_importance = {}
        for market in models:
            importance_path = base_path / f"{market}_importance.pkl"
            if importance_path.exists():
                try:
                    with open(importance_path, 'rb') as f:
                        feature_importance[market] = pickle.load(f)
                except Exception:
                    pass
        
        return cls(
            models=models,
            feature_columns=feature_columns,
            calibrators=calibrators,
            confidence_method=confidence_method,
            feature_importance=feature_importance
        )


class EnsemblePredictor(ModelPredictor):
    """
    Model predictor that combines multiple predictors.
    
    This predictor aggregates predictions from multiple model predictors
    to provide more robust and reliable forecasts.
    """
    
    def __init__(self, 
                predictors: List[ModelPredictor],
                weights: Optional[List[float]] = None,
                aggregation_method: str = 'weighted_average'):
        """
        Initialize the ensemble predictor.
        
        Args:
            predictors: List of model predictors to combine
            weights: Optional weights for each predictor (must sum to 1)
            aggregation_method: Method to aggregate predictions ('weighted_average', 'voting', or 'stacking')
        """
        if not predictors:
            raise ValueError("At least one predictor is required")
        
        self._predictors = predictors
        
        # Validate weights or use equal weights
        if weights is None:
            self._weights = [1.0 / len(predictors)] * len(predictors)
        else:
            if len(weights) != len(predictors):
                raise ValueError("Number of weights must match number of predictors")
            
            if abs(sum(weights) - 1.0) > 1e-6:
                # Normalize weights to sum to 1
                total = sum(weights)
                self._weights = [w / total for w in weights]
            else:
                self._weights = weights
        
        self._aggregation_method = aggregation_method
        
        # Get combined list of supported markets
        self._markets = set()
        for predictor in predictors:
            self._markets.update(predictor.supported_markets)
    
    @property
    def supported_markets(self) -> List[str]:
        """
        List markets supported by at least one predictor in the ensemble.
        
        Returns:
            List of market names
        """
        return list(self._markets)
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the ensemble of models.
        
        Args:
            features: DataFrame containing features for prediction
            
        Returns:
            DataFrame with aggregated prediction results
        """
        # Get predictions from each predictor
        all_predictions = []
        for i, predictor in enumerate(self._predictors):
            try:
                predictions = predictor.predict(features)
                weight = self._weights[i]
                
                # Add weight information
                predictions['predictor_weight'] = weight
                predictions['predictor_index'] = i
                
                all_predictions.append(predictions)
            except Exception as e:
                logger.error(f"Error from predictor {i}: {str(e)}")
                # Continue with remaining predictors
        
        if not all_predictions:
            raise RuntimeError("All predictors failed to generate predictions")
        
        # Combine predictions based on the aggregation method
        if self._aggregation_method == 'weighted_average':
            return self._aggregate_weighted_average(all_predictions)
        elif self._aggregation_method == 'voting':
            return self._aggregate_voting(all_predictions)
        else:
            # Default to weighted average for unsupported methods
            return self._aggregate_weighted_average(all_predictions)
    
    def _aggregate_weighted_average(self, predictions_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate predictions using weighted average.
        
        Args:
            predictions_list: List of prediction DataFrames
            
        Returns:
            Aggregated predictions DataFrame
        """
        # Initialize result with first predictor's index
        result = pd.DataFrame(index=predictions_list[0].index)
        
        # Track which markets each predictor supports
        market_predictors = {}
        
        # Identify all probability columns and their corresponding markets
        prob_columns = {}
        for df in predictions_list:
            for col in df.columns:
                if col.endswith('_prob') or '_prob_' in col:
                    market = col.split('_prob')[0]
                    if market not in prob_columns:
                        prob_columns[market] = []
                    prob_columns[market].append(col)
        
        # Aggregate probability predictions
        for market, cols in prob_columns.items():
            weighted_probs = {}
            total_weights = {}
            
            for i, df in enumerate(predictions_list):
                weight = self._weights[i]
                for col in cols:
                    if col in df.columns:
                        if col not in weighted_probs:
                            weighted_probs[col] = df[col] * weight
                            total_weights[col] = weight
                        else:
                            weighted_probs[col] += df[col] * weight
                            total_weights[col] += weight
            
            # Normalize by total weight
            for col, weighted_prob in weighted_probs.items():
                result[col] = weighted_prob / total_weights[col]
        
        # Determine classes for each market based on probabilities
        for market in prob_columns:
            cols = [col for col in result.columns if col.startswith(f"{market}_prob")]
            
            if len(cols) == 1:
                # Binary classification
                result[market] = (result[cols[0]] > 0.5).astype(int)
            else:
                # Multiclass classification - choose class with highest probability
                class_predictions = np.zeros(len(result), dtype=int)
                
                for idx, row in result[cols].iterrows():
                    class_predictions[result.index.get_loc(idx)] = np.argmax(row.values)
                
                result[market] = class_predictions
        
        # Calculate confidence
        confidence = []
        for idx, row in result.iterrows():
            max_prob = 0.0
            for col in row.index:
                if col.endswith('_prob') or '_prob_' in col:
                    if row[col] > max_prob:
                        max_prob = row[col]
            confidence.append(max_prob)
        
        result['confidence'] = confidence
        
        return result
    
    def _aggregate_voting(self, predictions_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate predictions using weighted voting.
        
        Args:
            predictions_list: List of prediction DataFrames
            
        Returns:
            Aggregated predictions DataFrame
        """
        # Initialize result with first predictor's index
        result = pd.DataFrame(index=predictions_list[0].index)
        
        # Determine all markets across all predictors
        all_markets = set()
        for df in predictions_list:
            for col in df.columns:
                if not (col.endswith('_prob') or '_prob_' in col or 
                        col == 'confidence' or col == 'predictor_weight' or 
                        col == 'predictor_index'):
                    all_markets.add(col)
        
        # For each market, perform weighted voting
        for market in all_markets:
            votes = {}
            weights = {}
            
            for i, df in enumerate(predictions_list):
                if market in df.columns:
                    weight = self._weights[i]
                    for idx, value in df[market].items():
                        if idx not in votes:
                            votes[idx] = {}
                            weights[idx] = {}
                        
                        if value not in votes[idx]:
                            votes[idx][value] = 0
                            weights[idx][value] = 0
                        
                        votes[idx][value] += 1
                        weights[idx][value] += weight
            
            # Determine final prediction based on weighted votes
            predictions = []
            for idx in result.index:
                if idx in votes and votes[idx]:
                    # Get class with highest weight
                    best_class = max(weights[idx].items(), key=lambda x: x[1])[0]
                    predictions.append(best_class)
                else:
                    # No votes for this instance
                    predictions.append(None)
            
            result[market] = predictions
        
        # Aggregate probabilities using weighted average
        prob_result = self._aggregate_weighted_average(predictions_list)
        
        # Add probability columns and confidence
        for col in prob_result.columns:
            if col.endswith('_prob') or '_prob_' in col or col == 'confidence':
                result[col] = prob_result[col]
        
        return result
    
    def predict_match(self, match_id: str, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make a prediction for a specific match using the ensemble.
        
        Args:
            match_id: Unique identifier for the match
            features: DataFrame containing features for the match
            
        Returns:
            Dictionary containing aggregated prediction results for the match
        """
        if match_id not in features.index:
            raise ValueError(f"Match ID {match_id} not found in features DataFrame")
        
        # Get features for this match
        match_features = features.loc[[match_id]]
        
        # Get predictions
        predictions = self.predict(match_features)
        
        # Convert to dictionary
        result = {'match_id': match_id}
        for col in predictions.columns:
            result[col] = predictions.loc[match_id, col]
        
        # Add individual model predictions
        individual_predictions = []
        for i, predictor in enumerate(self._predictors):
            try:
                pred = predictor.predict_match(match_id, features)
                pred['predictor_index'] = i
                pred['predictor_weight'] = self._weights[i]
                individual_predictions.append(pred)
            except Exception as e:
                logger.error(f"Error from predictor {i} for match {match_id}: {str(e)}")
        
        result['individual_predictions'] = individual_predictions
        
        return result
    
    def get_confidence(self, prediction: Dict[str, Any]) -> float:
        """
        Calculate confidence score for an ensemble prediction.
        
        Args:
            prediction: Prediction dictionary from predict_match
            
        Returns:
            Confidence score between 0 and 1
        """
        if 'confidence' in prediction:
            return float(prediction['confidence'])
        
        # If we have individual predictions, use their weighted confidence
        if 'individual_predictions' in prediction:
            total_confidence = 0.0
            total_weight = 0.0
            
            for pred in prediction['individual_predictions']:
                if 'predictor_weight' in pred:
                    weight = pred['predictor_weight']
                    
                    # Get confidence from individual predictor
                    idx = pred.get('predictor_index', -1)
                    if 0 <= idx < len(self._predictors):
                        conf = self._predictors[idx].get_confidence(pred)
                        total_confidence += conf * weight
                        total_weight += weight
            
            if total_weight > 0:
                return total_confidence / total_weight
        
        # Fall back to highest probability
        max_prob = 0.0
        for key, value in prediction.items():
            if key.endswith('_prob') or '_prob_' in key:
                if isinstance(value, (int, float)) and value > max_prob:
                    max_prob = value
        
        return max_prob 