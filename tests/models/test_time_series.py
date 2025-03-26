"""
Tests for time series model implementation.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.models.time_series import TimeSeriesPredictor

# Mock data for testing
@pytest.fixture
def mock_time_series_data():
    """Create mock time series data for testing."""
    # Create 100 days of data
    dates = [datetime.now() - timedelta(days=i) for i in range(100)]
    dates.reverse()  # Earliest date first
    
    np.random.seed(42)
    data = {
        'date': dates,
        'home_team_id': np.random.randint(1, 20, 100),
        'away_team_id': np.random.randint(1, 20, 100),
        'home_team_rank': np.random.randint(1, 20, 100),
        'away_team_rank': np.random.randint(1, 20, 100),
        'home_team_form': np.random.uniform(0, 1, 100),
        'away_team_form': np.random.uniform(0, 1, 100),
        'home_goals_scored': np.random.randint(0, 5, 100),
        'away_goals_scored': np.random.randint(0, 5, 100),
        'result': np.random.choice([0, 1, 2], 100)  # 0=home win, 1=draw, 2=away win
    }
    
    # Add some more features for testing
    data['home_goals_avg'] = np.random.uniform(0, 3, 100)
    data['away_goals_avg'] = np.random.uniform(0, 3, 100)
    data['home_goals_against_avg'] = np.random.uniform(0, 3, 100)
    data['away_goals_against_avg'] = np.random.uniform(0, 3, 100)
    data['home_shots_avg'] = np.random.uniform(5, 20, 100)
    data['away_shots_avg'] = np.random.uniform(5, 20, 100)
    
    # Add trend component to simulate time series pattern
    trend = np.linspace(0, 2, 100)
    seasonality = 0.5 * np.sin(np.linspace(0, 10*np.pi, 100))
    data['target_regression'] = trend + seasonality + 0.2 * np.random.randn(100)
    
    return pd.DataFrame(data)


class TestTimeSeriesModels:
    """Tests for the TimeSeriesPredictor class."""
    
    def test_init_model(self):
        """Test model initialization."""
        # Test valid model types
        for model_type in TimeSeriesPredictor.MODEL_TYPES:
            model = TimeSeriesPredictor(
                model_type=model_type,
                dataset_name="test_dataset",
                feature_type="basic",
                look_back=5,
                forecast_horizon=1
            )
            assert model.model_type == model_type
            assert model.model is None
            assert model.is_fitted is False
            
        # Test invalid model type
        with pytest.raises(ValueError):
            TimeSeriesPredictor(
                model_type="invalid_model",
                dataset_name="test_dataset",
                feature_type="basic",
                look_back=5,
                forecast_horizon=1
            )
    
    def test_create_sequences(self):
        """Test sequence creation."""
        model = TimeSeriesPredictor(
            model_type="lstm",
            dataset_name="test_dataset",
            feature_type="basic",
            look_back=3,
            forecast_horizon=1
        )
        
        # Create test data
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Test with single step forecast
        X, y = model._create_sequences(data, look_back=3, forecast_horizon=1)
        assert X.shape == (7, 3)
        assert y.shape == (7,)
        assert np.array_equal(X[0], [1, 2, 3])
        assert y[0] == 4
        
        # Test with multi-step forecast
        X, y = model._create_sequences(data, look_back=3, forecast_horizon=2)
        assert X.shape == (6, 3)
        assert y.shape == (6, 2)
        assert np.array_equal(X[0], [1, 2, 3])
        assert np.array_equal(y[0], [4, 5])
    
    @pytest.mark.skipif("tensorflow" not in sys.modules, reason="TensorFlow not installed")
    def test_lstm_model_creation(self):
        """Test LSTM model creation."""
        model = TimeSeriesPredictor(
            model_type="lstm",
            dataset_name="test_dataset",
            feature_type="basic",
            look_back=5,
            forecast_horizon=1,
            model_params={"task_type": "regression", "units": 32}
        )
        
        # Create LSTM model
        lstm_model = model._create_lstm_model(input_shape=(10, 5, 3), output_shape=1)
        assert lstm_model is not None
        
        # Check if model has expected layers
        assert len(lstm_model.layers) > 0
        assert "lstm" in lstm_model.layers[0].name.lower()
    
    @pytest.mark.skipif("tensorflow" not in sys.modules, reason="TensorFlow not installed")
    def test_gru_model_creation(self):
        """Test GRU model creation."""
        model = TimeSeriesPredictor(
            model_type="gru",
            dataset_name="test_dataset",
            feature_type="basic",
            look_back=5,
            forecast_horizon=1,
            model_params={"task_type": "regression", "units": 32}
        )
        
        # Create GRU model
        gru_model = model._create_gru_model(input_shape=(10, 5, 3), output_shape=1)
        assert gru_model is not None
        
        # Check if model has expected layers
        assert len(gru_model.layers) > 0
        assert "gru" in gru_model.layers[0].name.lower()
    
    @patch('src.models.time_series.Prophet')
    def test_prophet_fit(self, mock_prophet, mock_time_series_data):
        """Test fitting a Prophet model."""
        # Set up Prophet mock
        mock_prophet_instance = MagicMock()
        mock_prophet.return_value = mock_prophet_instance
        
        # Create model with mocked Prophet
        model = TimeSeriesPredictor(
            model_type="prophet",
            dataset_name="test_dataset",
            feature_type="basic",
            look_back=5,
            forecast_horizon=1,
            model_params={"task_type": "regression", "seasonality_mode": "multiplicative"}
        )
        
        # Train the model
        results = model.fit(
            df=mock_time_series_data,
            target_col="target_regression",
            test_size=0.2
        )
        
        # Verify Prophet was used
        mock_prophet.assert_called_once()
        mock_prophet_instance.fit.assert_called_once()
        
        # Check results
        assert model.is_fitted
        assert results['model_info']['trained']
        assert results['model_info']['dataset_name'] == "test_dataset"
    
    @patch('src.models.time_series.ARIMA')
    def test_arima_fit(self, mock_arima, mock_time_series_data):
        """Test fitting an ARIMA model."""
        # Set up ARIMA mock
        mock_arima_instance = MagicMock()
        mock_arima_result = MagicMock()
        mock_arima_instance.fit.return_value = mock_arima_result
        mock_arima.return_value = mock_arima_instance
        
        # Create model with mocked ARIMA
        model = TimeSeriesPredictor(
            model_type="arima",
            dataset_name="test_dataset",
            feature_type="basic",
            look_back=5,
            forecast_horizon=1,
            model_params={"p": 1, "d": 1, "q": 1, "auto": False}
        )
        
        # Train the model
        results = model.fit(
            df=mock_time_series_data,
            target_col="target_regression",
            test_size=0.2
        )
        
        # Verify ARIMA was used
        mock_arima.assert_called_once()
        mock_arima_instance.fit.assert_called_once()
        
        # Check results
        assert model.is_fitted
        assert results['model_info']['trained']
    
    def test_save_load(self, tmp_path, mock_time_series_data):
        """Test model saving and loading with minimal mock implementation."""
        # Skip actual model fitting to make test faster
        with patch.object(TimeSeriesPredictor, '_preprocess_time_series', return_value=(np.random.rand(10, 5), np.random.rand(10), mock_time_series_data)):
            with patch.object(TimeSeriesPredictor, 'evaluate', return_value={'mse': 0.1, 'rmse': 0.3}):
                with patch.object(TimeSeriesPredictor, '_create_model'):
                    model = TimeSeriesPredictor(
                        model_type="arima",  # Simplest model for testing
                        dataset_name="test_dataset",
                        feature_type="basic",
                        look_back=5,
                        forecast_horizon=1,
                        model_params={"auto": False}
                    )
                    
                    # Mock the fitting process
                    model.is_fitted = True
                    model.model = MagicMock()
                    model.scaler = MagicMock()
                    model.target_scaler = MagicMock()
                    
                    # Test saving
                    save_path = os.path.join(tmp_path, "time_series_model.pkl")
                    saved_path = model.save(output_dir=tmp_path)
                    assert os.path.exists(saved_path)
                    
                    # Test loading - we need to mock the unpickle process
                    with patch('pickle.load', return_value=model):
                        loaded_model = TimeSeriesPredictor.load(saved_path)
                        assert loaded_model.model_type == "arima"
                        assert loaded_model.dataset_name == "test_dataset"
                        assert loaded_model.is_fitted
    
    def test_predict(self, mock_time_series_data):
        """Test prediction functionality with mocks."""
        # Create a small dataset for predictions
        X_test = np.random.rand(5, 10, 3)  # [samples, lookback, features]
        
        # Test LSTM prediction
        with patch.object(TimeSeriesPredictor, 'is_fitted', True):
            lstm_model = TimeSeriesPredictor(
                model_type="lstm",
                dataset_name="test_dataset",
                feature_type="basic",
                look_back=10,
                forecast_horizon=1,
                model_params={"task_type": "regression"}
            )
            
            # Mock the keras model
            lstm_model.model = MagicMock()
            lstm_model.model.predict.return_value = np.random.rand(5, 1)
            lstm_model.target_scaler = MagicMock()
            lstm_model.target_scaler.inverse_transform.return_value = np.random.rand(5, 1)
            
            # Test prediction
            predictions = lstm_model.predict(X_test)
            
            # Check predictions
            assert lstm_model.model.predict.called
            assert predictions is not None
            assert len(predictions) == 5
    
    def test_evaluate(self, mock_time_series_data):
        """Test evaluation functionality with mocks."""
        # Create test data
        X_test = np.random.rand(20, 10, 3)
        y_test = np.random.rand(20)
        
        # Test evaluation for regression
        with patch.object(TimeSeriesPredictor, 'is_fitted', True):
            with patch.object(TimeSeriesPredictor, 'predict', return_value=np.random.rand(20)):
                model = TimeSeriesPredictor(
                    model_type="lstm",
                    dataset_name="test_dataset",
                    feature_type="basic",
                    look_back=10,
                    forecast_horizon=1,
                    model_params={"task_type": "regression"}
                )
                model.model = MagicMock()
                
                # Evaluate the model
                metrics = model.evaluate(X_test, y_test)
                
                # Check metrics
                assert "mse" in metrics
                assert "rmse" in metrics
                assert "mae" in metrics
                assert "r2" in metrics
    
    @patch('src.data.features.load_match_features')
    def test_predict_match(self, mock_load_features):
        """Test match prediction functionality."""
        # Mock the load_match_features function
        mock_df = pd.DataFrame({
            'feature1': [0.1, 0.2],
            'feature2': [1.0, 2.0],
            'date': [datetime.now(), datetime.now() + timedelta(days=1)]
        })
        mock_load_features.return_value = mock_df
        
        # Create model with mocks
        with patch.object(TimeSeriesPredictor, 'is_fitted', True):
            with patch.object(TimeSeriesPredictor, 'process_data', return_value=(np.random.rand(1, 5, 3), None)):
                with patch.object(TimeSeriesPredictor, 'predict', return_value=np.array([[0.8, 0.1, 0.1]])):
                    model = TimeSeriesPredictor(
                        model_type="lstm",
                        dataset_name="test_dataset",
                        feature_type="basic",
                        look_back=5,
                        forecast_horizon=1,
                        model_params={"task_type": "classification"}
                    )
                    model.model = MagicMock()
                    
                    # Test prediction
                    prediction = model.predict_match(
                        home_team_id=1,
                        away_team_id=2,
                        date="2023-01-01"
                    )
                    
                    # Check prediction structure
                    assert "prediction" in prediction
                    assert "home_team_id" in prediction
                    assert "away_team_id" in prediction
                    assert "date" in prediction 