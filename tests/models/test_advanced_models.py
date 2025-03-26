"""
Tests for advanced model implementation.
"""

import os
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.models.advanced import AdvancedMatchPredictor, train_advanced_model

# Mock data for testing
@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    # Create sample data with 100 rows
    np.random.seed(42)
    data = {
        'home_team_id': np.random.randint(1, 20, 100),
        'away_team_id': np.random.randint(1, 20, 100),
        'home_team_rank': np.random.randint(1, 20, 100),
        'away_team_rank': np.random.randint(1, 20, 100),
        'home_team_form': np.random.uniform(0, 1, 100),
        'away_team_form': np.random.uniform(0, 1, 100),
        'match_date': [datetime.now().strftime("%Y-%m-%d")] * 100,
        'result': np.random.choice([0, 1, 2], 100)  # 0=home win, 1=draw, 2=away win
    }
    
    # Add some more features for testing
    data['home_goals_avg'] = np.random.uniform(0, 3, 100)
    data['away_goals_avg'] = np.random.uniform(0, 3, 100)
    data['home_goals_against_avg'] = np.random.uniform(0, 3, 100)
    data['away_goals_against_avg'] = np.random.uniform(0, 3, 100)
    data['home_shots_avg'] = np.random.uniform(5, 20, 100)
    data['away_shots_avg'] = np.random.uniform(5, 20, 100)
    
    return pd.DataFrame(data)


class TestAdvancedModels:
    """Tests for the AdvancedMatchPredictor class."""
    
    def test_init_model(self):
        """Test model initialization."""
        # Test all model types
        for model_type in AdvancedMatchPredictor.MODEL_TYPES:
            model = AdvancedMatchPredictor(model_type=model_type)
            assert model.model_type == model_type
            assert model.model is None
            assert model.model_info["trained"] is False
    
    def test_model_creation(self):
        """Test model creation."""
        # Test neural network model creation
        model = AdvancedMatchPredictor(model_type="neural_network")
        model._create_model(input_dim=10, num_classes=3)
        assert model.model is not None
        
        # Test lightgbm model creation
        model = AdvancedMatchPredictor(model_type="lightgbm")
        model._create_model(num_classes=3)
        assert model.model is not None
        
        # Test catboost model creation
        model = AdvancedMatchPredictor(model_type="catboost")
        model._create_model(num_classes=3)
        assert model.model is not None
        
        # Test deep ensemble model creation
        model = AdvancedMatchPredictor(model_type="deep_ensemble", model_params={"ensemble_size": 2})
        model._create_model(input_dim=10, num_classes=3)
        assert model.model is not None
        assert len(model.model) == 2
        
        # Test time series model creation
        model = AdvancedMatchPredictor(model_type="time_series")
        model._create_model()
        assert model.model is not None
        assert isinstance(model.model, dict)
    
    def test_simple_training(self, mock_data):
        """Test model training with mock data."""
        # Create a very small batch for quick testing
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, 20)
        
        # Test neural network training (with small epochs to be fast)
        model = AdvancedMatchPredictor(
            model_type="neural_network", 
            model_params={"epochs": 2, "batch_size": 10, "verbose": 0}
        )
        results = model.train(X, y)
        assert model.model is not None
        assert model.model_info["trained"] is True
        assert "epochs" in results
        
        # Test lightgbm training
        model = AdvancedMatchPredictor(
            model_type="lightgbm",
            model_params={"n_estimators": 10, "verbose": False}
        )
        results = model.train(X, y)
        assert model.model is not None
        assert model.model_info["trained"] is True
        assert "n_estimators" in results
        
        # Test prediction
        preds = model.predict(X)
        assert len(preds) == len(y)
        
        # Test probability prediction
        proba = model.predict_proba(X)
        assert proba.shape[0] == len(y)
        assert proba.shape[1] == 3  # Three classes
    
    def test_model_evaluation(self, mock_data):
        """Test model evaluation."""
        # Create a very small batch for quick testing
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, 20)
        
        # Test evaluation for lightgbm (fastest model)
        model = AdvancedMatchPredictor(
            model_type="lightgbm",
            model_params={"n_estimators": 10, "verbose": False}
        )
        model.train(X, y)
        
        evaluation = model.evaluate(X, y)
        assert "accuracy" in evaluation
        assert "f1" in evaluation
        assert "precision" in evaluation
        assert "recall" in evaluation
        assert "confusion_matrix" in evaluation
    
    def test_save_load(self, tmp_path):
        """Test model saving and loading."""
        # Create a very small batch for quick testing
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, 20)
        
        # Test saving and loading for lightgbm (fastest model)
        model = AdvancedMatchPredictor(
            model_type="lightgbm",
            model_params={"n_estimators": 10, "verbose": False}
        )
        model.train(X, y)
        
        # Save model
        save_path = os.path.join(tmp_path, "test_model.pkl")
        model_path = model.save(filepath=save_path)
        assert os.path.exists(model_path)
        
        # Load model
        loaded_model = AdvancedMatchPredictor.load(model_path)
        assert loaded_model.model_type == model.model_type
        assert loaded_model.model_info["trained"] is True
        
        # Test prediction with loaded model
        preds = loaded_model.predict(X)
        assert len(preds) == len(y)
    
    def test_predict_match(self, tmp_path, monkeypatch):
        """Test match prediction functionality."""
        # We need to mock the pipeline loading and processing
        class MockPipeline:
            def transform(self, X):
                return X
        
        class MockEncoder:
            def inverse_transform(self, y):
                return ["Home Win" if i == 0 else "Draw" if i == 1 else "Away Win" for i in y]
        
        # Create a very small batch for quick testing
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 3, 20)
        
        # Create and train model
        model = AdvancedMatchPredictor(
            model_type="lightgbm",
            model_params={"n_estimators": 10, "verbose": False}
        )
        model.train(X, y)
        model.pipeline = MockPipeline()
        model.target_encoder = MockEncoder()
        
        # Mock the process_data method
        def mock_process_data(self, df, target_col=None):
            return np.random.randn(1, 10), None
        
        monkeypatch.setattr(AdvancedMatchPredictor, "process_data", mock_process_data)
        
        # Test prediction
        prediction = model.predict_match(home_team_id=1, away_team_id=2)
        assert "prediction" in prediction
        assert "probabilities" in prediction
        assert "confidence" in prediction
        assert "home_team_id" in prediction
        assert "away_team_id" in prediction
    
    def test_train_advanced_model_function(self, monkeypatch, mock_data):
        """Test the train_advanced_model helper function."""
        # Mock the load_feature_data function
        def mock_load_feature_data(dataset_name, feature_type):
            return mock_data
        
        # Mock the apply_feature_pipeline function
        def mock_apply_feature_pipeline(df, pipeline, target_col=None):
            # Return mock processed data
            X = np.random.randn(len(df), 10)
            y = np.array(df[target_col] if target_col else np.zeros(len(df)))
            encoder = MockEncoder()
            return X, y, encoder
        
        # Mock encoder
        class MockEncoder:
            def inverse_transform(self, y):
                return ["Home Win" if i == 0 else "Draw" if i == 1 else "Away Win" for i in y]
        
        # Mock the load_pipeline function
        def mock_load_pipeline(self):
            self.pipeline = "mock_pipeline"
            return True
        
        # Apply monkeypatches
        monkeypatch.setattr("src.models.training.load_feature_data", mock_load_feature_data)
        monkeypatch.setattr("src.data.features.apply_feature_pipeline", mock_apply_feature_pipeline)
        monkeypatch.setattr(AdvancedMatchPredictor, "load_pipeline", mock_load_pipeline)
        
        # Mock the save method
        def mock_save(self, filepath=None):
            return "mock_model_path.pkl"
        
        monkeypatch.setattr(AdvancedMatchPredictor, "save", mock_save)
        
        # Test the function with a lightweight model
        model = train_advanced_model(
            model_type="lightgbm",
            dataset_name="mock_dataset",
            feature_type="mock_features",
            test_size=0.2,
            validation_size=0.0,
            model_params={"n_estimators": 10, "verbose": False}
        )
        
        assert isinstance(model, AdvancedMatchPredictor)
        assert model.model_type == "lightgbm"
        assert model.model_info["trained"] is True 