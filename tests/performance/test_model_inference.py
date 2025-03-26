"""
Performance tests specifically for model inference in the Soccer Prediction System.
These tests focus on measuring the speed and efficiency of prediction functionality.
"""

import os
import sys
import time
import pytest
import random
import numpy as np
import pandas as pd
from unittest import mock

# Add the root directory to sys.path to allow importing the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import actual models (use mocks if they're not available)
try:
    from src.models.prediction import PredictionService
    REAL_MODELS_AVAILABLE = True
except ImportError:
    REAL_MODELS_AVAILABLE = False


# Mock models for testing if real ones aren't available
class MockBasicModel:
    """A simple mock model for testing."""
    
    def __init__(self):
        self.name = "basic_model"
    
    def predict(self, features):
        """Make a prediction with minimal computation."""
        time.sleep(0.001)  # Simulate minimal computation
        home_str = features.get("home_strength", 0.5)
        away_str = features.get("away_strength", 0.5)
        
        # Simple calculation based on team strengths
        home_win = 0.45 + (home_str - away_str) * 0.3
        away_win = 0.45 + (away_str - home_str) * 0.3
        draw = 1.0 - home_win - away_win
        
        return {
            "home_win": max(0.05, min(0.9, home_win)),
            "draw": max(0.05, min(0.9, draw)),
            "away_win": max(0.05, min(0.9, away_win))
        }


class MockEnsembleModel:
    """A mock ensemble model that combines multiple predictions."""
    
    def __init__(self):
        self.name = "ensemble_model"
        self.models = [MockBasicModel() for _ in range(3)]
    
    def predict(self, features):
        """Make a prediction using ensemble of models."""
        # Get predictions from all models
        predictions = [model.predict(features) for model in self.models]
        
        # Average the predictions
        result = {
            "home_win": sum(p["home_win"] for p in predictions) / len(predictions),
            "draw": sum(p["draw"] for p in predictions) / len(predictions),
            "away_win": sum(p["away_win"] for p in predictions) / len(predictions)
        }
        
        # Add small random variation to simulate ensemble behavior
        for key in result:
            result[key] += random.uniform(-0.02, 0.02)
            result[key] = max(0.05, min(0.9, result[key]))
        
        # Normalize to ensure they sum to 1
        total = sum(result.values())
        return {k: v/total for k, v in result.items()}


class MockAdvancedModel:
    """A mock advanced model with more computation."""
    
    def __init__(self):
        self.name = "advanced_model"
        self.weights = np.random.random((10, 3))  # Random weights
    
    def predict(self, features):
        """Make a prediction with more intensive computation."""
        time.sleep(0.005)  # Simulate more computation
        
        # Convert features to a vector (simplified)
        feature_vector = np.array([
            features.get("home_strength", 0.5),
            features.get("away_strength", 0.5),
            features.get("home_form", 0.5),
            features.get("away_form", 0.5),
            features.get("home_goals_avg", 1.5),
            features.get("away_goals_avg", 1.2),
            features.get("home_defense", 0.5),
            features.get("away_defense", 0.5),
            features.get("days_rest_home", 5),
            features.get("days_rest_away", 5)
        ])
        
        # Apply matrix multiplication for prediction
        raw_output = np.dot(feature_vector, self.weights)
        probs = np.exp(raw_output) / np.sum(np.exp(raw_output))  # Softmax
        
        return {
            "home_win": float(probs[0]),
            "draw": float(probs[1]),
            "away_win": float(probs[2])
        }


# Mock prediction service
class MockPredictionService:
    """Mock prediction service for testing."""
    
    def __init__(self):
        self.models = {
            "basic": MockBasicModel(),
            "advanced": MockAdvancedModel(),
            "ensemble": MockEnsembleModel()
        }
        self.default_model = "ensemble"
    
    def get_available_models(self):
        """Get list of available models."""
        return [
            {"name": name, "accuracy": random.uniform(0.65, 0.85)}
            for name in self.models.keys()
        ]
    
    def predict_match(self, home_team_id, away_team_id, model_name=None):
        """Predict a match outcome."""
        # Use specified model or default
        model = self.models.get(model_name or self.default_model, self.models[self.default_model])
        
        # Create features from team IDs
        features = self._generate_features(home_team_id, away_team_id)
        
        # Make prediction
        return model.predict(features)
    
    def _generate_features(self, home_team_id, away_team_id):
        """Generate features for a match."""
        # In a real system, these would come from a database
        # Here we simulate them based on team IDs
        return {
            "home_strength": (home_team_id % 10) / 10.0 + 0.4,
            "away_strength": (away_team_id % 10) / 10.0 + 0.4,
            "home_form": random.uniform(0.3, 0.8),
            "away_form": random.uniform(0.3, 0.8),
            "home_goals_avg": random.uniform(0.8, 2.5),
            "away_goals_avg": random.uniform(0.8, 2.0),
            "home_defense": random.uniform(0.4, 0.8),
            "away_defense": random.uniform(0.4, 0.8),
            "days_rest_home": random.randint(3, 7),
            "days_rest_away": random.randint(3, 7)
        }


# Get the appropriate prediction service
if REAL_MODELS_AVAILABLE:
    # Use real prediction service
    from src.models.prediction import prediction_service
else:
    # Use mock prediction service
    prediction_service = MockPredictionService()


class TestModelInference:
    """Performance tests for model inference."""
    
    @pytest.mark.benchmark(
        group="model_inference",
        min_time=0.1,
        max_time=0.5,
        min_rounds=50,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_basic_model_inference(self, benchmark):
        """Test basic model inference performance."""
        # Set up test data
        home_team_id = random.randint(1, 20)
        away_team_id = random.randint(1, 20)
        
        # Define the function to benchmark
        def predict_with_basic():
            return prediction_service.predict_match(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                model_name="basic"
            )
        
        # Run the benchmark
        result = benchmark(predict_with_basic)
        
        # Verify the result
        assert isinstance(result, dict)
        assert "home_win" in result
        assert "draw" in result
        assert "away_win" in result
        assert abs(sum(result.values()) - 1.0) < 1e-6
    
    @pytest.mark.benchmark(
        group="model_inference",
        min_time=0.1,
        max_time=1.0,
        min_rounds=20,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_advanced_model_inference(self, benchmark):
        """Test advanced model inference performance."""
        # Set up test data
        home_team_id = random.randint(1, 20)
        away_team_id = random.randint(1, 20)
        
        # Define the function to benchmark
        def predict_with_advanced():
            return prediction_service.predict_match(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                model_name="advanced"
            )
        
        # Run the benchmark
        result = benchmark(predict_with_advanced)
        
        # Verify the result
        assert isinstance(result, dict)
        assert "home_win" in result
        assert "draw" in result
        assert "away_win" in result
        assert abs(sum(result.values()) - 1.0) < 1e-6
    
    @pytest.mark.benchmark(
        group="model_inference",
        min_time=0.1,
        max_time=1.0,
        min_rounds=20,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_ensemble_model_inference(self, benchmark):
        """Test ensemble model inference performance."""
        # Set up test data
        home_team_id = random.randint(1, 20)
        away_team_id = random.randint(1, 20)
        
        # Define the function to benchmark
        def predict_with_ensemble():
            return prediction_service.predict_match(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                model_name="ensemble"
            )
        
        # Run the benchmark
        result = benchmark(predict_with_ensemble)
        
        # Verify the result
        assert isinstance(result, dict)
        assert "home_win" in result
        assert "draw" in result
        assert "away_win" in result
        assert abs(sum(result.values()) - 1.0) < 1e-6
    
    @pytest.mark.benchmark(
        group="model_inference",
        min_time=0.5,
        max_time=2.0,
        min_rounds=10,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_batch_inference_performance(self, benchmark):
        """Test batch inference performance for multiple matches."""
        # Set up test data - 20 matches
        batch_size = 20
        matches = [
            (random.randint(1, 20), random.randint(1, 20))
            for _ in range(batch_size)
        ]
        
        # Define batch prediction function
        def predict_batch_matches():
            return [
                prediction_service.predict_match(
                    home_team_id=home_id,
                    away_team_id=away_id,
                    model_name="ensemble"
                )
                for home_id, away_id in matches
            ]
        
        # Run the benchmark
        result = benchmark(predict_batch_matches)
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == batch_size
        
        for prediction in result:
            assert isinstance(prediction, dict)
            assert "home_win" in prediction
            assert "draw" in prediction
            assert "away_win" in prediction
            assert abs(sum(prediction.values()) - 1.0) < 1e-6


class TestModelLoadingPerformance:
    """Tests for model loading performance."""
    
    @pytest.mark.benchmark(
        group="model_loading",
        min_time=0.1,
        max_time=1.0,
        min_rounds=5,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_model_switching_overhead(self, benchmark):
        """Test the overhead of switching between different models."""
        models = ["basic", "advanced", "ensemble"]
        
        # Set up test data
        home_team_id = random.randint(1, 20)
        away_team_id = random.randint(1, 20)
        
        # Define function that switches models
        def switch_models_and_predict():
            results = {}
            for model in models:
                results[model] = prediction_service.predict_match(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    model_name=model
                )
            return results
        
        # Run the benchmark
        result = benchmark(switch_models_and_predict)
        
        # Verify the result
        assert isinstance(result, dict)
        assert len(result) == len(models)
        for model in models:
            assert model in result
            model_result = result[model]
            assert "home_win" in model_result
            assert "draw" in model_result
            assert "away_win" in model_result
            assert abs(sum(model_result.values()) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 