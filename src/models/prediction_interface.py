"""
Prediction interface for betting strategies.

This module provides standardized interfaces for prediction models to interact
with betting strategies and other components of the betting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class ModelPredictor(ABC):
    """
    Abstract interface for prediction models to standardize interaction with betting strategies.
    
    This interface ensures that different prediction models (machine learning models,
    statistical models, etc.) can be used interchangeably with betting strategies.
    """
    
    @abstractmethod
    def predict(self, 
               match_data: Dict[str, Any],
               **kwargs) -> Dict[str, float]:
        """
        Generate predictions for a single match.
        
        Args:
            match_data: Dictionary containing match information
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dict[str, float]: Dictionary mapping outcomes to probabilities
        """
        pass
    
    @abstractmethod
    def predict_batch(self,
                     matches_df: pd.DataFrame,
                     **kwargs) -> pd.DataFrame:
        """
        Generate predictions for multiple matches.
        
        Args:
            matches_df: DataFrame containing match information
            **kwargs: Additional model-specific parameters
            
        Returns:
            pd.DataFrame: DataFrame with match_id and predicted probabilities
        """
        pass
    
    @abstractmethod
    def get_confidence(self, 
                      match_data: Dict[str, Any],
                      predictions: Dict[str, float]) -> float:
        """
        Calculate confidence score for predictions.
        
        Args:
            match_data: Dictionary containing match information
            predictions: Dictionary mapping outcomes to probabilities
            
        Returns:
            float: Confidence score between 0 and 1
        """
        pass
    
    @property
    def model_name(self) -> str:
        """Get the name of the prediction model."""
        return self.__class__.__name__
    
    @property
    @abstractmethod
    def supported_markets(self) -> List[str]:
        """Get the list of markets supported by this predictor."""
        pass
    
    @abstractmethod
    def calibrate(self, 
                 historical_data: pd.DataFrame,
                 actual_results: pd.DataFrame) -> None:
        """
        Calibrate the model based on historical predictions and actual results.
        
        Args:
            historical_data: DataFrame with historical match data and predictions
            actual_results: DataFrame with actual match results
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass


class XGBoostPredictor(ModelPredictor):
    """
    Implementation of ModelPredictor using XGBoost models.
    
    This predictor uses XGBoost models to predict match outcomes for various markets.
    """
    
    def __init__(self, 
                model_paths: Dict[str, str] = None,
                feature_columns: List[str] = None,
                confidence_threshold: float = 0.7):
        """
        Initialize the XGBoost predictor.
        
        Args:
            model_paths: Dictionary mapping market names to model file paths
            feature_columns: List of feature column names required by the model
            confidence_threshold: Threshold for high confidence predictions
        """
        self._models = {}
        self._feature_columns = feature_columns or []
        self._confidence_threshold = confidence_threshold
        
        # Load models if paths provided
        if model_paths:
            for market, path in model_paths.items():
                self.load_model_for_market(market, path)
    
    def load_model_for_market(self, market: str, path: str) -> None:
        """
        Load a model for a specific market.
        
        Args:
            market: Market name (e.g. 'home_win', 'over_under_2.5')
            path: Path to the model file
        """
        try:
            import xgboost as xgb
            self._models[market] = xgb.Booster()
            self._models[market].load_model(path)
            logger.info(f"Loaded XGBoost model for {market} market from {path}")
        except Exception as e:
            logger.error(f"Failed to load model for {market}: {str(e)}")
            raise
    
    def predict(self, match_data: Dict[str, Any], **kwargs) -> Dict[str, float]:
        """
        Generate predictions for a single match using loaded XGBoost models.
        
        Args:
            match_data: Dictionary containing match information
            **kwargs: Additional parameters including 'markets' to specify which markets to predict
            
        Returns:
            Dict[str, float]: Dictionary mapping outcomes to probabilities
        """
        # Extract requested markets or use all loaded models
        markets = kwargs.get('markets', list(self._models.keys()))
        
        # Check that we have models for the requested markets
        missing_markets = [m for m in markets if m not in self._models]
        if missing_markets:
            logger.warning(f"Missing models for markets: {missing_markets}")
        
        # Prepare features
        features = self._prepare_features(match_data)
        
        # Generate predictions for each market
        predictions = {}
        for market in markets:
            if market in self._models:
                # Convert features to DMatrix
                try:
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(features)
                    
                    # Get prediction
                    pred_prob = self._models[market].predict(dmatrix)[0]
                    
                    # Store prediction
                    predictions[market] = float(pred_prob)
                except Exception as e:
                    logger.error(f"Error predicting for market {market}: {str(e)}")
                    predictions[market] = None
        
        return predictions
    
    def predict_batch(self, matches_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate predictions for multiple matches using loaded XGBoost models.
        
        Args:
            matches_df: DataFrame containing match information
            **kwargs: Additional parameters including 'markets' to specify which markets to predict
            
        Returns:
            pd.DataFrame: DataFrame with match_id and predicted probabilities
        """
        # Extract requested markets or use all loaded models
        markets = kwargs.get('markets', list(self._models.keys()))
        
        # Check that we have models for the requested markets
        missing_markets = [m for m in markets if m not in self._models]
        if missing_markets:
            logger.warning(f"Missing models for markets: {missing_markets}")
        
        # Prepare features for all matches
        features_list = []
        match_ids = []
        
        for _, row in matches_df.iterrows():
            match_data = row.to_dict()
            features = self._prepare_features(match_data)
            features_list.append(features)
            match_ids.append(match_data.get('match_id'))
        
        # Create results DataFrame
        results = pd.DataFrame({'match_id': match_ids})
        
        # Generate predictions for each market
        for market in markets:
            if market in self._models:
                try:
                    import xgboost as xgb
                    
                    # Stack features and convert to DMatrix
                    stacked_features = np.vstack(features_list)
                    dmatrix = xgb.DMatrix(stacked_features)
                    
                    # Get predictions
                    pred_probs = self._models[market].predict(dmatrix)
                    
                    # Add to results
                    results[f'{market}_prob'] = pred_probs
                except Exception as e:
                    logger.error(f"Error batch predicting for market {market}: {str(e)}")
                    results[f'{market}_prob'] = None
        
        # Add confidence scores
        results['confidence'] = results.apply(
            lambda x: self._calculate_confidence(x, markets), axis=1
        )
        
        return results
    
    def _prepare_features(self, match_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for prediction from match data.
        
        Args:
            match_data: Dictionary containing match information
            
        Returns:
            np.ndarray: Array of features
        """
        # Extract features in the correct order
        features = []
        for col in self._feature_columns:
            if col in match_data:
                features.append(match_data[col])
            else:
                logger.warning(f"Missing feature: {col}")
                features.append(0.0)  # Default value
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_confidence(self, row: pd.Series, markets: List[str]) -> float:
        """
        Calculate confidence for a set of predictions.
        
        Args:
            row: Series containing predictions
            markets: List of markets that were predicted
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Simple heuristic: higher confidence when probabilities are further from 0.5
        confidences = []
        
        for market in markets:
            prob_col = f'{market}_prob'
            if prob_col in row and row[prob_col] is not None:
                # Distance from 0.5 normalized to [0, 1]
                distance = abs(row[prob_col] - 0.5) * 2
                confidences.append(distance)
        
        if confidences:
            return np.mean(confidences)
        else:
            return 0.0
    
    def get_confidence(self, match_data: Dict[str, Any], predictions: Dict[str, float]) -> float:
        """
        Calculate confidence score for predictions.
        
        Args:
            match_data: Dictionary containing match information
            predictions: Dictionary mapping outcomes to probabilities
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Simple heuristic: higher confidence when probabilities are further from 0.5
        confidences = []
        
        for market, prob in predictions.items():
            if prob is not None:
                # Distance from 0.5 normalized to [0, 1]
                distance = abs(prob - 0.5) * 2
                confidences.append(distance)
        
        if confidences:
            return np.mean(confidences)
        else:
            return 0.0
    
    @property
    def supported_markets(self) -> List[str]:
        """Get the list of markets supported by this predictor."""
        return list(self._models.keys())
    
    def calibrate(self, historical_data: pd.DataFrame, actual_results: pd.DataFrame) -> None:
        """
        Calibrate the model based on historical predictions and actual results.
        
        Args:
            historical_data: DataFrame with historical match data and predictions
            actual_results: DataFrame with actual match results
        """
        # Implement Platt scaling or isotonic regression for calibration
        # This is a simplified version - a real implementation would be more complex
        logger.info("XGBoost calibration not implemented in this version")
    
    def save_model(self, path: str) -> None:
        """
        Save all models to disk.
        
        Args:
            path: Base path to save the models
        """
        import os
        
        for market, model in self._models.items():
            model_path = os.path.join(path, f"{market}_model.json")
            try:
                model.save_model(model_path)
                logger.info(f"Saved model for {market} to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model for {market}: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """
        Load models from disk.
        
        Args:
            path: Base path to load the models from
        """
        import os
        import glob
        
        # Find all model files
        model_files = glob.glob(os.path.join(path, "*_model.json"))
        
        for model_file in model_files:
            # Extract market name from filename
            filename = os.path.basename(model_file)
            market = filename.replace("_model.json", "")
            
            # Load the model
            self.load_model_for_market(market, model_file)


class EnsemblePredictor(ModelPredictor):
    """
    Ensemble predictor that combines multiple predictors.
    
    This predictor aggregates predictions from multiple underlying predictors
    using configurable aggregation methods.
    """
    
    def __init__(self, 
                predictors: List[ModelPredictor],
                weights: Optional[Dict[str, List[float]]] = None,
                aggregation_method: str = 'weighted_average'):
        """
        Initialize the ensemble predictor.
        
        Args:
            predictors: List of ModelPredictor instances
            weights: Optional dictionary mapping markets to weight lists for each predictor
            aggregation_method: Method to aggregate predictions ('average', 'weighted_average', 'max', 'median')
        """
        self._predictors = predictors
        self._weights = weights or {}
        self._aggregation_method = aggregation_method
        
        # Validate weights if provided
        if weights:
            for market, market_weights in weights.items():
                if len(market_weights) != len(predictors):
                    logger.warning(f"Weights for market {market} don't match number of predictors")
    
    def predict(self, match_data: Dict[str, Any], **kwargs) -> Dict[str, float]:
        """
        Generate ensemble predictions for a single match.
        
        Args:
            match_data: Dictionary containing match information
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, float]: Dictionary mapping outcomes to probabilities
        """
        # Get predictions from all models
        all_predictions = []
        
        for predictor in self._predictors:
            try:
                predictions = predictor.predict(match_data, **kwargs)
                all_predictions.append(predictions)
            except Exception as e:
                logger.error(f"Error in predictor {predictor.model_name}: {str(e)}")
        
        # Aggregate predictions
        return self._aggregate_predictions(all_predictions)
    
    def predict_batch(self, matches_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate ensemble predictions for multiple matches.
        
        Args:
            matches_df: DataFrame containing match information
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: DataFrame with match_id and predicted probabilities
        """
        # Get predictions from all models
        all_prediction_dfs = []
        
        for predictor in self._predictors:
            try:
                predictions_df = predictor.predict_batch(matches_df, **kwargs)
                all_prediction_dfs.append(predictions_df)
            except Exception as e:
                logger.error(f"Error in batch prediction with {predictor.model_name}: {str(e)}")
        
        # Combine and aggregate predictions
        return self._aggregate_prediction_dfs(all_prediction_dfs)
    
    def _aggregate_predictions(self, all_predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate predictions from multiple models.
        
        Args:
            all_predictions: List of prediction dictionaries from different models
            
        Returns:
            Dict[str, float]: Aggregated predictions
        """
        # Collect all markets
        all_markets = set()
        for predictions in all_predictions:
            all_markets.update(predictions.keys())
        
        # Aggregate predictions for each market
        aggregated = {}
        
        for market in all_markets:
            # Collect predictions for this market
            market_predictions = []
            for predictions in all_predictions:
                if market in predictions and predictions[market] is not None:
                    market_predictions.append(predictions[market])
            
            # Skip if no valid predictions
            if not market_predictions:
                aggregated[market] = None
                continue
            
            # Get weights for this market if available
            weights = self._weights.get(market, [1.0] * len(self._predictors))
            
            # Ensure weights align with available predictions
            effective_weights = weights[:len(market_predictions)]
            if sum(effective_weights) == 0:
                effective_weights = [1.0] * len(market_predictions)
            
            # Perform aggregation based on method
            if self._aggregation_method == 'average':
                aggregated[market] = np.mean(market_predictions)
            elif self._aggregation_method == 'weighted_average':
                aggregated[market] = np.average(market_predictions, weights=effective_weights)
            elif self._aggregation_method == 'max':
                aggregated[market] = np.max(market_predictions)
            elif self._aggregation_method == 'median':
                aggregated[market] = np.median(market_predictions)
            else:
                # Default to average
                aggregated[market] = np.mean(market_predictions)
        
        return aggregated
    
    def _aggregate_prediction_dfs(self, all_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate prediction DataFrames from multiple models.
        
        Args:
            all_dfs: List of prediction DataFrames from different models
            
        Returns:
            pd.DataFrame: Aggregated predictions DataFrame
        """
        if not all_dfs:
            return pd.DataFrame()
        
        # Extract match_ids from first DataFrame
        result_df = pd.DataFrame({'match_id': all_dfs[0]['match_id']})
        
        # Get all probability columns
        prob_columns = set()
        for df in all_dfs:
            prob_cols = [col for col in df.columns if col.endswith('_prob')]
            prob_columns.update(prob_cols)
        
        # Aggregate each probability column
        for prob_col in prob_columns:
            # Extract market name
            market = prob_col.replace('_prob', '')
            
            # Get weights for this market if available
            weights = self._weights.get(market, [1.0] * len(self._predictors))
            
            # Collect values from each DataFrame
            values_list = []
            for i, df in enumerate(all_dfs):
                if prob_col in df.columns:
                    values_list.append((df[prob_col], weights[i] if i < len(weights) else 1.0))
            
            # Aggregate values based on method
            if not values_list:
                result_df[prob_col] = None
            elif self._aggregation_method == 'average':
                result_df[prob_col] = np.mean([values[0] for values, _ in values_list], axis=0)
            elif self._aggregation_method == 'weighted_average':
                # Calculate weighted average for each row
                weighted_values = []
                weighted_weights = []
                
                for values, weight in values_list:
                    weighted_values.append(values * weight)
                    weighted_weights.append(np.ones_like(values) * weight)
                
                sum_weights = np.sum(weighted_weights, axis=0)
                sum_values = np.sum(weighted_values, axis=0)
                
                # Avoid division by zero
                result_df[prob_col] = np.divide(
                    sum_values, 
                    sum_weights, 
                    out=np.zeros_like(sum_values), 
                    where=sum_weights!=0
                )
            elif self._aggregation_method == 'max':
                result_df[prob_col] = np.max([values[0] for values, _ in values_list], axis=0)
            elif self._aggregation_method == 'median':
                result_df[prob_col] = np.median([values[0] for values, _ in values_list], axis=0)
            else:
                # Default to average
                result_df[prob_col] = np.mean([values[0] for values, _ in values_list], axis=0)
        
        # Calculate confidence
        result_df['confidence'] = result_df.apply(
            lambda x: self._calculate_confidence(x, [col.replace('_prob', '') for col in prob_columns]), 
            axis=1
        )
        
        return result_df
    
    def _calculate_confidence(self, row: pd.Series, markets: List[str]) -> float:
        """
        Calculate confidence for aggregated predictions.
        
        Args:
            row: Series containing predictions
            markets: List of markets that were predicted
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Get confidence from individual predictors
        individual_confidences = []
        
        for predictor in self._predictors:
            # Extract predictions for this predictor
            predictions = {}
            for market in markets:
                prob_col = f'{market}_prob'
                if prob_col in row and row[prob_col] is not None:
                    predictions[market] = row[prob_col]
            
            try:
                # Calculate confidence
                if predictions:
                    confidence = predictor.get_confidence({}, predictions)
                    individual_confidences.append(confidence)
            except Exception as e:
                logger.error(f"Error calculating confidence for {predictor.model_name}: {str(e)}")
        
        # Aggregate confidences
        if individual_confidences:
            if self._aggregation_method == 'max':
                return max(individual_confidences)
            elif self._aggregation_method == 'median':
                return np.median(individual_confidences)
            else:
                return np.mean(individual_confidences)
        else:
            return 0.0
    
    def get_confidence(self, match_data: Dict[str, Any], predictions: Dict[str, float]) -> float:
        """
        Calculate confidence score for ensemble predictions.
        
        Args:
            match_data: Dictionary containing match information
            predictions: Dictionary mapping outcomes to probabilities
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Simple combination of distance from 0.5 and agreement between models
        confidences = []
        
        for market, prob in predictions.items():
            if prob is not None:
                # Distance from 0.5 normalized to [0, 1]
                distance = abs(prob - 0.5) * 2
                confidences.append(distance)
        
        if confidences:
            return np.mean(confidences)
        else:
            return 0.0
    
    @property
    def supported_markets(self) -> List[str]:
        """Get the list of markets supported by at least one predictor."""
        all_markets = set()
        for predictor in self._predictors:
            all_markets.update(predictor.supported_markets)
        return list(all_markets)
    
    def calibrate(self, historical_data: pd.DataFrame, actual_results: pd.DataFrame) -> None:
        """
        Calibrate all predictors based on historical predictions and actual results.
        
        Args:
            historical_data: DataFrame with historical match data and predictions
            actual_results: DataFrame with actual match results
        """
        for predictor in self._predictors:
            try:
                predictor.calibrate(historical_data, actual_results)
            except Exception as e:
                logger.error(f"Error calibrating {predictor.model_name}: {str(e)}")
    
    def save_model(self, path: str) -> None:
        """
        Save all predictors to disk.
        
        Args:
            path: Base path to save the predictors
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save each predictor
        for i, predictor in enumerate(self._predictors):
            predictor_path = os.path.join(path, f"predictor_{i}")
            os.makedirs(predictor_path, exist_ok=True)
            
            try:
                predictor.save_model(predictor_path)
            except Exception as e:
                logger.error(f"Error saving predictor {i}: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """
        Load all predictors from disk.
        
        Args:
            path: Base path to load the predictors from
        """
        import os
        
        # Load each predictor
        for i, predictor in enumerate(self._predictors):
            predictor_path = os.path.join(path, f"predictor_{i}")
            
            if os.path.exists(predictor_path):
                try:
                    predictor.load_model(predictor_path)
                except Exception as e:
                    logger.error(f"Error loading predictor {i}: {str(e)}") 