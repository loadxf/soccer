"""
Model Testing Framework for Soccer Prediction System.

This module provides a comprehensive testing framework for model evaluation,
validation, and performance assessment.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

# Assuming these imports will be available based on project structure
from src.models.baseline import LogisticRegressionModel, RandomForestModel, XGBoostModel
from src.models.advanced import NeuralNetworkModel, LightGBMModel, CatBoostModel, DeepEnsembleModel
from src.models.evaluation import evaluate_model, calculate_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Define test fixtures
@pytest.fixture
def sample_data():
    """Generate sample data for model testing."""
    # Create a synthetic dataset for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features that could represent team statistics
    home_team_strength = np.random.normal(0, 1, n_samples)
    away_team_strength = np.random.normal(0, 1, n_samples)
    home_recent_form = np.random.normal(0, 1, n_samples)
    away_recent_form = np.random.normal(0, 1, n_samples)
    home_goals_scored_avg = np.random.normal(1.5, 0.5, n_samples)
    away_goals_scored_avg = np.random.normal(1.2, 0.5, n_samples)
    
    # Generate target: home win (2), draw (1), away win (0)
    probabilities = np.column_stack([
        0.25 + 0.1 * (away_team_strength - home_team_strength + away_recent_form - home_recent_form),  # away win
        0.3 + 0.05 * np.abs(home_team_strength - away_team_strength),  # draw
        0.35 + 0.1 * (home_team_strength - away_team_strength + home_recent_form - away_recent_form)   # home win
    ])
    
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Generate outcomes based on probabilities
    target = np.array([np.random.choice([0, 1, 2], p=p) for p in probabilities])
    
    # Create feature matrix
    X = np.column_stack([
        home_team_strength, away_team_strength,
        home_recent_form, away_recent_form,
        home_goals_scored_avg, away_goals_scored_avg
    ])
    
    # Create and return train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': [
            'home_team_strength', 'away_team_strength',
            'home_recent_form', 'away_recent_form',
            'home_goals_scored_avg', 'away_goals_scored_avg'
        ]
    }

@pytest.fixture
def model_config():
    """Model configuration for testing."""
    return {
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'multi_class': 'multinomial'
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'num_class': 3,
            'random_state': 42
        }
    }

class TestModelFramework:
    """Test suite for the model framework."""
    
    def test_model_initialization(self, model_config):
        """Test that all models can be initialized."""
        # Initialize baseline models
        lr_model = LogisticRegressionModel(**model_config['logistic_regression'])
        rf_model = RandomForestModel(**model_config['random_forest'])
        xgb_model = XGBoostModel(**model_config['xgboost'])
        
        # Test model attributes
        assert hasattr(lr_model, 'model')
        assert hasattr(rf_model, 'model')
        assert hasattr(xgb_model, 'model')
        
        # Initialize advanced models with minimal configs
        nn_model = NeuralNetworkModel(input_dim=6, hidden_dims=[32, 16], output_dim=3)
        lgbm_model = LightGBMModel(n_estimators=100, objective='multiclass', num_class=3)
        
        # Test model attributes
        assert hasattr(nn_model, 'model')
        assert hasattr(lgbm_model, 'model')
    
    def test_model_training(self, sample_data, model_config):
        """Test that models can be trained."""
        X_train, y_train = sample_data['X_train'], sample_data['y_train']
        
        # Train logistic regression model
        lr_model = LogisticRegressionModel(**model_config['logistic_regression'])
        lr_model.train(X_train, y_train)
        
        # Train random forest model
        rf_model = RandomForestModel(**model_config['random_forest'])
        rf_model.train(X_train, y_train)
        
        # Test that models are fitted
        assert hasattr(lr_model.model, 'coef_')
        assert hasattr(rf_model.model, 'estimators_')
    
    def test_model_prediction(self, sample_data, model_config):
        """Test that models can generate predictions."""
        X_train, y_train = sample_data['X_train'], sample_data['y_train']
        X_test = sample_data['X_test']
        
        # Train and generate predictions using logistic regression
        lr_model = LogisticRegressionModel(**model_config['logistic_regression'])
        lr_model.train(X_train, y_train)
        
        # Test class predictions
        predictions = lr_model.predict(X_test)
        assert predictions.shape == sample_data['y_test'].shape
        assert np.all(np.isin(np.unique(predictions), [0, 1, 2]))
        
        # Test probability predictions
        probabilities = lr_model.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 3)  # 3 classes
        assert np.allclose(np.sum(probabilities, axis=1), 1.0)  # probabilities sum to 1
    
    def test_model_evaluation(self, sample_data, model_config):
        """Test model evaluation functionality."""
        X_train, y_train = sample_data['X_train'], sample_data['y_train']
        X_test, y_test = sample_data['X_test'], sample_data['y_test']
        
        # Train model
        rf_model = RandomForestModel(**model_config['random_forest'])
        rf_model.train(X_train, y_train)
        
        # Generate predictions
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)
        
        # Calculate metrics manually
        manual_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'log_loss': log_loss(y_test, y_proba)
        }
        
        # Calculate metrics using framework
        framework_metrics = calculate_metrics(y_test, y_pred, y_proba)
        
        # Compare metrics
        for metric in manual_metrics:
            assert np.isclose(manual_metrics[metric], framework_metrics[metric], rtol=1e-5)
    
    def test_model_comparison(self, sample_data, model_config):
        """Test comparison of multiple models."""
        X_train, y_train = sample_data['X_train'], sample_data['y_train']
        X_test, y_test = sample_data['X_test'], sample_data['y_test']
        
        # Initialize and train models
        models = {
            'logistic_regression': LogisticRegressionModel(**model_config['logistic_regression']),
            'random_forest': RandomForestModel(**model_config['random_forest']),
            'xgboost': XGBoostModel(**model_config['xgboost'])
        }
        
        # Train all models
        for name, model in models.items():
            model.train(X_train, y_train)
        
        # Evaluate all models
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            metrics = calculate_metrics(y_test, y_pred, y_proba)
            results[name] = metrics
        
        # Check that all models have metrics
        assert len(results) == 3
        assert all(metric in results['logistic_regression'] for metric in ['accuracy', 'f1_macro', 'log_loss'])
    
    def test_model_persistence(self, sample_data, model_config, tmp_path):
        """Test model saving and loading."""
        X_train, y_train = sample_data['X_train'], sample_data['y_train']
        X_test = sample_data['X_test']
        
        # Train model
        rf_model = RandomForestModel(**model_config['random_forest'])
        rf_model.train(X_train, y_train)
        
        # Get predictions before saving
        predictions_before = rf_model.predict(X_test)
        
        # Save model
        model_path = os.path.join(tmp_path, "rf_model.joblib")
        rf_model.save(model_path)
        
        # Check that model file exists
        assert os.path.exists(model_path)
        
        # Create new model instance and load
        new_model = RandomForestModel()
        new_model.load(model_path)
        
        # Get predictions after loading
        predictions_after = new_model.predict(X_test)
        
        # Check that predictions are the same
        assert np.array_equal(predictions_before, predictions_after)
    
    def test_cross_validation(self, sample_data, model_config):
        """Test cross-validation functionality."""
        X = np.vstack([sample_data['X_train'], sample_data['X_test']])
        y = np.concatenate([sample_data['y_train'], sample_data['y_test']])
        
        # Initialize model
        lr_model = LogisticRegressionModel(**model_config['logistic_regression'])
        
        # Perform cross-validation
        from sklearn.model_selection import cross_validate
        cv_results = cross_validate(
            lr_model.model, X, y, 
            cv=5, 
            scoring=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        )
        
        # Check that we have cross-validation results
        assert 'test_accuracy' in cv_results
        assert len(cv_results['test_accuracy']) == 5
        assert all(0 <= score <= 1 for score in cv_results['test_accuracy'])


if __name__ == "__main__":
    pytest.main() 