"""
Models module for Soccer Prediction System.
Contains machine learning model implementations and training pipelines.
"""

from src.models.baseline import BaselineMatchPredictor, train_baseline_model
from src.models.advanced import AdvancedMatchPredictor, train_advanced_model
from src.models.ensemble import EnsemblePredictor, train_ensemble_model
from src.models.explainability import ModelExplainer, generate_model_explanations
from src.models.time_series import TimeSeriesPredictor
from src.models.player_performance import PlayerPerformanceModel, PlayerPerformancePredictor, train_player_performance_models

__all__ = [
    'BaselineMatchPredictor',
    'AdvancedMatchPredictor',
    'EnsemblePredictor',
    'TimeSeriesPredictor',
    'train_baseline_model',
    'train_advanced_model',
    'train_ensemble_model',
    'ModelExplainer',
    'generate_model_explanations',
    'PlayerPerformanceModel',
    'PlayerPerformancePredictor',
    'train_player_performance_models'
] 