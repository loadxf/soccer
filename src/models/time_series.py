"""
Time-series specific modeling approaches for soccer match prediction.
This module implements specialized time-series models for predicting match outcomes.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Conditionally import statsmodels
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Conditionally import Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Conditionally import pmdarima
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

# Conditionally import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Import project components
from src.utils.logger import get_logger
from src.data.features import load_feature_pipeline, apply_feature_pipeline

# Conditionally import AdvancedMatchPredictor
try:
    from src.models.advanced import AdvancedMatchPredictor
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.time_series")

# Define paths
MODELS_DIR = os.path.join(DATA_DIR, "models")
TIMESERIES_MODELS_DIR = os.path.join(MODELS_DIR, "time_series")
os.makedirs(TIMESERIES_MODELS_DIR, exist_ok=True)


class TimeSeriesPredictor:
    """
    Base class for time-series models for predicting soccer match outcomes.
    """
    
    MODEL_TYPES = ["arima", "prophet", "lstm", "gru", "encoder_decoder"]
    
    def __init__(self, model_type: str, dataset_name: str, feature_type: str, 
                 look_back: int = 5, forecast_horizon: int = 1, model_params: Optional[Dict] = None):
        """
        Initialize the time-series predictor.
        
        Args:
            model_type: Type of time-series model 
                ("arima", "prophet", "lstm", "gru", "encoder_decoder")
            dataset_name: Name of the dataset to use
            feature_type: Type of features to use
            look_back: Number of past observations to consider
            forecast_horizon: Number of steps ahead to forecast
            model_params: Optional model parameters
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Invalid model type: {model_type}. Must be one of {self.MODEL_TYPES}")
        
        # Check if required libraries are available
        if model_type == "arima" and not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA models. Please install it with 'pip install statsmodels'")
        elif model_type == "prophet" and not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet models. Please install it with 'pip install prophet'")
        elif model_type in ["lstm", "gru", "encoder_decoder"] and not TENSORFLOW_AVAILABLE:
            raise ImportError("tensorflow is required for deep learning models. Please install it with 'pip install tensorflow'")
            
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.feature_type = feature_type
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        self.model_params = model_params or {}
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.is_fitted = False
        self.feature_names = None
        self.target_name = None
        
        # Model metadata
        self.model_info = {
            "model_type": model_type,
            "dataset_name": dataset_name,
            "feature_type": feature_type,
            "look_back": look_back,
            "forecast_horizon": forecast_horizon,
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {},
        }

    def _create_sequences(self, data: np.ndarray, look_back: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and targets for time-series modeling.
        
        Args:
            data: Input time-series data
            look_back: Number of past observations to consider
            forecast_horizon: Number of steps ahead to forecast
            
        Returns:
            X, y: Sequences and targets
        """
        X, y = [], []
        for i in range(len(data) - look_back - forecast_horizon + 1):
            X.append(data[i:(i + look_back)])
            # For multi-step forecasting
            if forecast_horizon > 1:
                y.append(data[(i + look_back):(i + look_back + forecast_horizon)])
            else:
                y.append(data[i + look_back])
        
        return np.array(X), np.array(y)

    def _preprocess_time_series(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Preprocess time-series data for training.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            
        Returns:
            X, y, processed_df: Time-series sequences, targets, and processed dataframe
        """
        logger.info(f"Preprocessing time-series data with look_back={self.look_back}, forecast_horizon={self.forecast_horizon}")
        
        # Make a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Sort by date if available
        if 'date' in df_copy.columns:
            df_copy = df_copy.sort_values('date')
        
        # Store feature names
        self.feature_names = [col for col in df_copy.columns if col != target_col]
        self.target_name = target_col
        
        # Prepare features and target
        X_raw = df_copy.drop(columns=[target_col]).values
        y_raw = df_copy[target_col].values
        
        # Scale the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # For regression tasks, also scale the target
        if self.model_type in ['prophet', 'arima'] or self.model_params.get('task_type') == 'regression':
            self.target_scaler = StandardScaler()
            y_scaled = self.target_scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()
        else:
            y_scaled = y_raw
        
        # Create time-series sequences
        X_seq, y_seq = self._create_sequences(X_scaled, self.look_back, self.forecast_horizon)
        
        # For LSTM/GRU models, reshape input to [samples, time_steps, features]
        if self.model_type in ['lstm', 'gru', 'encoder_decoder']:
            if len(X_seq.shape) < 3:
                X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], -1)
        
        return X_seq, y_seq, df_copy

    def _create_model(self, input_shape: Tuple, output_shape: int) -> Any:
        """
        Create a time-series model based on model_type.
        
        Args:
            input_shape: Shape of input data
            output_shape: Shape of output data
            
        Returns:
            Created model
        """
        if self.model_type == 'lstm':
            return self._create_lstm_model(input_shape, output_shape)
        elif self.model_type == 'gru':
            return self._create_gru_model(input_shape, output_shape)
        elif self.model_type == 'encoder_decoder':
            return self._create_encoder_decoder_model(input_shape, output_shape)
        elif self.model_type == 'arima':
            # ARIMA model is created during fitting
            return None
        elif self.model_type == 'prophet':
            # Prophet model is created during fitting
            return Prophet(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _create_lstm_model(self, input_shape: Tuple, output_shape: int) -> Any:
        """
        Create an LSTM model for time-series forecasting.
        
        Args:
            input_shape: Shape of input data
            output_shape: Shape of output data
            
        Returns:
            LSTM model
        """
        # Get model parameters or use defaults
        units = self.model_params.get('units', 64)
        dropout = self.model_params.get('dropout', 0.2)
        learning_rate = self.model_params.get('learning_rate', 0.001)
        
        model = Sequential()
        model.add(LSTM(units, return_sequences=True, input_shape=input_shape[1:]))
        model.add(Dropout(dropout))
        model.add(LSTM(units // 2))
        model.add(Dropout(dropout))
        model.add(Dense(output_shape))
        
        optimizer = Adam(learning_rate=learning_rate)
        
        # Compile model
        if self.model_params.get('task_type') == 'regression':
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
        
        return model

    def _create_gru_model(self, input_shape: Tuple, output_shape: int) -> Any:
        """
        Create a GRU model for time-series forecasting.
        
        Args:
            input_shape: Shape of input data
            output_shape: Shape of output data
            
        Returns:
            GRU model
        """
        # Get model parameters or use defaults
        units = self.model_params.get('units', 64)
        dropout = self.model_params.get('dropout', 0.2)
        learning_rate = self.model_params.get('learning_rate', 0.001)
        
        model = Sequential()
        model.add(GRU(units, return_sequences=True, input_shape=input_shape[1:]))
        model.add(Dropout(dropout))
        model.add(GRU(units // 2))
        model.add(Dropout(dropout))
        model.add(Dense(output_shape))
        
        optimizer = Adam(learning_rate=learning_rate)
        
        # Compile model
        if self.model_params.get('task_type') == 'regression':
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
        
        return model

    def _create_encoder_decoder_model(self, input_shape: Tuple, output_shape: int) -> Any:
        """
        Create an encoder-decoder model for time-series forecasting.
        
        Args:
            input_shape: Shape of input data
            output_shape: Shape of output data
            
        Returns:
            Encoder-decoder model
        """
        # Get model parameters or use defaults
        units = self.model_params.get('units', 64)
        dropout = self.model_params.get('dropout', 0.2)
        learning_rate = self.model_params.get('learning_rate', 0.001)
        
        model = Sequential()
        model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape[1:]))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(units // 2)))
        model.add(Dropout(dropout))
        model.add(Dense(output_shape))
        
        optimizer = Adam(learning_rate=learning_rate)
        
        # Compile model
        if self.model_params.get('task_type') == 'regression':
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', 
                         metrics=['accuracy'])
        
        return model

    def fit(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2, 
            validation_split: float = 0.1, random_state: int = 42) -> Dict:
        """
        Fit the time-series model.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            test_size: Portion of data to use for testing
            validation_split: Portion of training data to use for validation
            random_state: Random seed
            
        Returns:
            Dict with training results
        """
        logger.info(f"Fitting {self.model_type} time-series model")
        start_time = datetime.now()
        
        # Preprocess data
        X, y, processed_df = self._preprocess_time_series(df, target_col)
        
        # Split data into train and test sets
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Fit model based on model type
        if self.model_type in ['lstm', 'gru', 'encoder_decoder']:
            # Create deep learning model
            output_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1
            self.model = self._create_model(X_train.shape, output_shape)
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=os.path.join(TIMESERIES_MODELS_DIR, f"{self.model_type}_best_model.h5"),
                    save_best_only=True
                )
            ]
            
            # Train model
            batch_size = self.model_params.get('batch_size', 32)
            epochs = self.model_params.get('epochs', 100)
            
            history = self.model.fit(
                X_train, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store training history
            train_history = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
        elif self.model_type == 'arima':
            # For ARIMA, we need to reshape the data
            # We'll use the first feature for simplicity, or combine features if needed
            if isinstance(target_col, str):
                ts_data = processed_df[target_col].values
            else:
                ts_data = y_train
                
            # Auto-ARIMA to find best parameters
            if self.model_params.get('auto', True):
                logger.info("Using auto-ARIMA to find best parameters")
                auto_model = pm.auto_arima(
                    ts_data,
                    seasonal=self.model_params.get('seasonal', True),
                    m=self.model_params.get('m', 12),  # seasonal period
                    d=self.model_params.get('d', None),
                    start_p=self.model_params.get('start_p', 1),
                    start_q=self.model_params.get('start_q', 1),
                    max_p=self.model_params.get('max_p', 5),
                    max_q=self.model_params.get('max_q', 5),
                    information_criterion=self.model_params.get('information_criterion', 'aic'),
                    trace=True,
                    random_state=random_state
                )
                
                # Get best parameters
                order = auto_model.order
                seasonal_order = auto_model.seasonal_order
                
                # Create SARIMAX model with best parameters
                self.model = SARIMAX(
                    ts_data,
                    order=order,
                    seasonal_order=seasonal_order
                )
            else:
                # Use specified parameters
                p = self.model_params.get('p', 1)
                d = self.model_params.get('d', 1)
                q = self.model_params.get('q', 1)
                
                self.model = ARIMA(
                    ts_data,
                    order=(p, d, q)
                )
            
            # Fit the model
            self.model = self.model.fit()
            train_history = {'aic': self.model.aic, 'bic': self.model.bic}
            
        elif self.model_type == 'prophet':
            # For Prophet, we need to reshape the data
            # Prophet requires 'ds' (date) and 'y' (target) columns
            if 'date' in processed_df.columns:
                prophet_df = pd.DataFrame({
                    'ds': processed_df['date'],
                    'y': processed_df[target_col]
                })
            else:
                # If date is not available, create artificial dates
                prophet_df = pd.DataFrame({
                    'ds': pd.date_range(start='2020-01-01', periods=len(processed_df)),
                    'y': processed_df[target_col]
                })
            
            # Create and fit Prophet model
            self.model = Prophet(**self.model_params)
            self.model.fit(prophet_df)
            
            train_history = {'fit_time': (datetime.now() - start_time).total_seconds()}
        
        # Evaluate on test set
        test_metrics = self.evaluate(X_test, y_test)
        
        # Update model info
        self.model_info.update({
            'trained': True,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'training_duration': (datetime.now() - start_time).total_seconds(),
            'performance': test_metrics,
            'training_history': train_history
        })
        
        self.is_fitted = True
        
        logger.info(f"Model training completed in {self.model_info['training_duration']:.2f} seconds")
        logger.info(f"Test metrics: {test_metrics}")
        
        return {
            'model_info': self.model_info,
            'test_metrics': test_metrics
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the time-series model.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # For deep learning models (LSTM, GRU, etc.)
        if self.model_type in ['lstm', 'gru', 'encoder_decoder']:
            # Reshape input if needed
            if len(X.shape) < 3:
                X = X.reshape(X.shape[0], X.shape[1], -1)
                
            # Get raw predictions
            predictions = self.model.predict(X)
            
            # Inverse transform if regression
            if self.model_params.get('task_type') == 'regression' and self.target_scaler is not None:
                predictions = self.target_scaler.inverse_transform(predictions)
                
            return predictions
            
        # For ARIMA models
        elif self.model_type == 'arima':
            # ARIMA forecast
            steps = X.shape[0] if isinstance(X, np.ndarray) else 1
            forecast = self.model.forecast(steps=steps)
            
            # Inverse transform if needed
            if self.target_scaler is not None:
                forecast = self.target_scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
                
            return forecast
            
        # For Prophet models
        elif self.model_type == 'prophet':
            # Create a future dataframe
            steps = X.shape[0] if isinstance(X, np.ndarray) else 1
            future = self.model.make_future_dataframe(periods=steps)
            
            # Make predictions
            forecast = self.model.predict(future)
            predictions = forecast['yhat'].values[-steps:]
            
            # Inverse transform if needed
            if self.target_scaler is not None:
                predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                
            return predictions
        
        else:
            raise ValueError(f"Prediction not implemented for model type: {self.model_type}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates for classification tasks.
        
        Args:
            X: Input feature matrix
            
        Returns:
            Probability estimates
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Only applicable for classification tasks
        if self.model_params.get('task_type') != 'classification':
            raise ValueError("predict_proba() is only applicable for classification tasks")
        
        # For deep learning models (LSTM, GRU, etc.)
        if self.model_type in ['lstm', 'gru', 'encoder_decoder']:
            # Reshape input if needed
            if len(X.shape) < 3:
                X = X.reshape(X.shape[0], X.shape[1], -1)
                
            # Get probability estimates
            return self.model.predict(X)
        
        else:
            raise ValueError(f"Probability estimates not available for model type: {self.model_type}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the time-series model.
        
        Args:
            X: Input feature matrix
            y: Target values
            
        Returns:
            Dict with evaluation metrics
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get predictions
        y_pred = self.predict(X)
        
        # For regression tasks
        if self.model_params.get('task_type') == 'regression' or self.model_type in ['arima', 'prophet']:
            # Flatten arrays if needed
            y_true = y.flatten() if len(y.shape) > 1 else y
            y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
            
            # Calculate regression metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate R² if possible
            try:
                r2 = r2_score(y_true, y_pred)
            except:
                r2 = float('nan')
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }
        
        # For classification tasks
        else:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Flatten arrays if needed
            y_true = y.flatten() if len(y.shape) > 1 else y
            y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
            
            # Convert to integers if needed
            y_true = y_true.astype(int)
            y_pred = y_pred.astype(int)
            
            # Calculate classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

    def plot_predictions(self, X: np.ndarray, y: np.ndarray, savefig: bool = False,
                        output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model predictions against actual values.
        
        Args:
            X: Input feature matrix
            y: Target values
            savefig: Whether to save the figure
            output_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Flatten arrays if needed
        y_true = y.flatten() if len(y.shape) > 1 else y
        y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot actual values
        plt.plot(y_true, label='Actual', marker='o')
        
        # Plot predictions
        plt.plot(y_pred, label='Predicted', marker='x')
        
        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel(self.target_name or 'Value')
        plt.title(f'{self.model_type.upper()} Model Predictions vs Actual')
        plt.legend()
        plt.grid(True)
        
        # Save figure if requested
        if savefig:
            path = output_path or os.path.join(
                TIMESERIES_MODELS_DIR, 
                f"{self.model_type}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(path)
            logger.info(f"Saved prediction plot to {path}")
        
        return plt.gcf()

    def save(self, output_dir: Optional[str] = None) -> str:
        """
        Save the time-series model.
        
        Args:
            output_dir: Output directory (if None, use default)
            
        Returns:
            Path to the saved model
        """
        if not self.is_fitted:
            logger.warning("Saving an unfitted model")
        
        # Use default directory if not specified
        output_dir = output_dir or TIMESERIES_MODELS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename based on model type and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = f"{self.model_type}_model_{timestamp}.pkl"
        model_path = os.path.join(output_dir, model_file)
        
        # Special handling for Keras models
        if self.model_type in ['lstm', 'gru', 'encoder_decoder']:
            # Save the Keras model separately
            keras_model_path = model_path.replace('.pkl', '.h5')
            self.model.save(keras_model_path)
            
            # Temporarily set model to None to avoid serialization issues
            original_model = self.model
            self.model = None
            
            # Save the predictor object without the model
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)
            
            # Restore the model
            self.model = original_model
            logger.info(f"Model saved to {model_path} and {keras_model_path}")
            
            return model_path
        
        # For other model types
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)
                
            logger.info(f"Model saved to {model_path}")
            return model_path

    @classmethod
    def load(cls, model_path: str) -> 'TimeSeriesPredictor':
        """
        Load a time-series model from a file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_path}")
        
        with open(model_path, 'rb') as f:
            predictor = pickle.load(f)
            
        # Special handling for Keras models
        if predictor.model_type in ['lstm', 'gru', 'encoder_decoder']:
            # Load the Keras model
            keras_model_path = model_path.replace('.pkl', '.h5')
            if os.path.exists(keras_model_path):
                predictor.model = load_model(keras_model_path)
            else:
                logger.warning(f"Keras model file not found: {keras_model_path}")
        
        return predictor

    def process_data(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process data for prediction.
        
        Args:
            df: Input DataFrame
            target_col: Optional target column name
            
        Returns:
            Processed X and y (if target_col is provided)
        """
        # Make a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()
        
        # Use the target column from initialization if not provided
        target_col = target_col or self.target_name
        
        # Sort by date if available
        if 'date' in df_copy.columns:
            df_copy = df_copy.sort_values('date')
        
        # Prepare features
        if target_col in df_copy.columns:
            X_raw = df_copy.drop(columns=[target_col]).values
            y_raw = df_copy[target_col].values
        else:
            X_raw = df_copy.values
            y_raw = None
        
        # Scale the features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_raw)
        else:
            X_scaled = X_raw
        
        # Create time-series sequences
        if y_raw is not None:
            # If target is available, create sequences with target
            if self.target_scaler is not None:
                y_scaled = self.target_scaler.transform(y_raw.reshape(-1, 1)).flatten()
            else:
                y_scaled = y_raw
                
            X_seq, y_seq = self._create_sequences(X_scaled, self.look_back, self.forecast_horizon)
            
            # Reshape for deep learning models
            if self.model_type in ['lstm', 'gru', 'encoder_decoder'] and len(X_seq.shape) < 3:
                X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], -1)
                
            return X_seq, y_seq
        else:
            # If target is not available, create sequences without target
            # Simple approach: just use the last look_back observations
            if len(X_scaled) >= self.look_back:
                X_seq = np.array([X_scaled[-self.look_back:]])
                
                # Reshape for deep learning models
                if self.model_type in ['lstm', 'gru', 'encoder_decoder'] and len(X_seq.shape) < 3:
                    X_seq = X_seq.reshape(X_seq.shape[0], X_seq.shape[1], -1)
                    
                return X_seq, None
            else:
                raise ValueError(f"Not enough data points. Need at least {self.look_back} observations.")

    def predict_match(self, home_team_id: Union[str, int], away_team_id: Union[str, int], 
                     date: Optional[str] = None, features: Optional[Dict] = None) -> Dict:
        """
        Predict a single match outcome.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            date: Optional match date (format: YYYY-MM-DD)
            features: Optional additional features
            
        Returns:
            Dict with prediction results
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Import data loading function
        from src.data.features import load_match_features
        
        # Load match features
        try:
            match_df = load_match_features(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                date=date,
                additional_features=features
            )
        except Exception as e:
            logger.error(f"Error loading match features: {e}")
            raise
        
        # Process data
        X, _ = self.process_data(match_df)
        
        # Make prediction
        pred = self.predict(X)
        
        # Format prediction based on task type
        if self.model_params.get('task_type') == 'classification':
            # Classification result (0 = home win, 1 = draw, 2 = away win)
            pred_class = np.argmax(pred[0]) if len(pred.shape) > 1 else int(pred[0])
            
            result_mapping = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
            prediction = result_mapping.get(pred_class, f'Class {pred_class}')
            
            # Get probabilities if available
            probabilities = {}
            if hasattr(self, 'predict_proba') and callable(getattr(self, 'predict_proba')):
                probs = self.predict_proba(X)[0]
                for i, label in result_mapping.items():
                    if i < len(probs):
                        probabilities[label] = float(probs[i])
            
            return {
                'prediction': prediction,
                'probabilities': probabilities,
                'confidence': float(np.max(pred[0])) if len(pred.shape) > 1 else 1.0,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'date': date
            }
        else:
            # Regression result (could be goals, score difference, etc.)
            prediction = float(pred[0])
            return {
                'prediction': prediction,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'date': date
            } 