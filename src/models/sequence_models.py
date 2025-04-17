"""
Sequence Models for Soccer Prediction

This module implements transformer-based and recurrent neural network models
for predicting soccer matches using sequence data such as team form.
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Dropout, BatchNormalization, 
    Input, Concatenate, Embedding, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss, roc_auc_score

# Import project components
from src.utils.logger import get_logger
from src.models.baseline import BaseMatchPredictor

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.sequence_models")

# Define paths
MODELS_DIR = os.path.join(DATA_DIR, "models")
SEQUENCE_MODELS_DIR = os.path.join(MODELS_DIR, "sequence")
os.makedirs(SEQUENCE_MODELS_DIR, exist_ok=True)


class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block with multi-head attention and feed forward network.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TeamSequenceEncoder(tf.keras.layers.Layer):
    """
    Encodes a sequence of team features using a transformer architecture.
    """
    def __init__(self, 
                 sequence_length: int, 
                 feature_dim: int, 
                 embed_dim: int = 64, 
                 num_heads: int = 2, 
                 ff_dim: int = 64,
                 num_transformer_blocks: int = 2,
                 dropout_rate: float = 0.1):
        super(TeamSequenceEncoder, self).__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        # Project input features to embedding dimension
        self.projection = Dense(embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_transformer_blocks)
        ]
        
        # Global pooling to get a single vector representation
        self.global_pooling = GlobalAveragePooling1D()
        self.final_projection = Dense(embed_dim)
    
    def call(self, inputs, training=False):
        x = self.projection(inputs)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        # Global pooling to get a fixed-size representation
        x = self.global_pooling(x)
        x = self.final_projection(x)
        
        return x


class SoccerTransformerModel:
    """
    Transformer-based model for soccer match prediction using team sequence data.
    
    This model uses two transformer encoders to process sequences of features for
    home and away teams, then combines these encodings with match-specific features
    to predict match outcomes.
    """
    
    def __init__(self,
                 sequence_length: int = 5,
                 team_feature_dim: int = 10,
                 match_feature_dim: int = 20,
                 embed_dim: int = 64,
                 num_heads: int = 2,
                 ff_dim: int = 64,
                 num_transformer_blocks: int = 2,
                 dropout_rate: float = 0.2,
                 l2_reg: float = 1e-4,
                 learning_rate: float = 0.001,
                 num_classes: int = 3):
        """
        Initialize the soccer transformer model.
        
        Args:
            sequence_length: Number of past matches to consider for each team
            team_feature_dim: Dimension of features for each team match
            match_feature_dim: Dimension of match-specific features
            embed_dim: Embedding dimension for transformer
            num_heads: Number of attention heads
            ff_dim: Feed forward network dimension
            num_transformer_blocks: Number of transformer blocks
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
            learning_rate: Learning rate for Adam optimizer
            num_classes: Number of output classes (typically 3: home win, draw, away win)
        """
        self.sequence_length = sequence_length
        self.team_feature_dim = team_feature_dim
        self.match_feature_dim = match_feature_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Create model
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self) -> Model:
        """
        Build the transformer model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Inputs
        home_sequence_input = Input(shape=(self.sequence_length, self.team_feature_dim), 
                                    name="home_sequence")
        away_sequence_input = Input(shape=(self.sequence_length, self.team_feature_dim), 
                                    name="away_sequence")
        match_features_input = Input(shape=(self.match_feature_dim,), 
                                    name="match_features")
        
        # Team sequence encoders (shared weights)
        team_encoder = TeamSequenceEncoder(
            sequence_length=self.sequence_length,
            feature_dim=self.team_feature_dim,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_transformer_blocks=self.num_transformer_blocks,
            dropout_rate=self.dropout_rate
        )
        
        # Encode team sequences
        home_encoding = team_encoder(home_sequence_input)
        away_encoding = team_encoder(away_sequence_input)
        
        # Concatenate team encodings with match features
        concatenated = Concatenate()([home_encoding, away_encoding, match_features_input])
        
        # Final prediction layers
        x = Dense(64, activation="relu", kernel_regularizer=l2(self.l2_reg))(concatenated)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation="relu", kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        if self.num_classes == 2:
            # Binary classification
            outputs = Dense(1, activation="sigmoid", name="output")(x)
        else:
            # Multi-class classification
            outputs = Dense(self.num_classes, activation="softmax", name="output")(x)
        
        # Create model
        model = Model(
            inputs=[home_sequence_input, away_sequence_input, match_features_input],
            outputs=outputs
        )
        
        # Compile model
        if self.num_classes == 2:
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
        else:
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        
        return model
    
    def fit(self, 
            X_home_seq: np.ndarray, 
            X_away_seq: np.ndarray, 
            X_match: np.ndarray, 
            y: np.ndarray,
            validation_data: Optional[Tuple] = None,
            batch_size: int = 32,
            epochs: int = 100,
            patience: int = 10,
            verbose: int = 1) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_home_seq: Home team sequence features
            X_away_seq: Away team sequence features
            X_match: Match-specific features
            y: Target labels
            validation_data: Optional validation data as tuple (X_val, y_val)
            batch_size: Batch size for training
            epochs: Maximum number of epochs
            patience: Patience for early stopping
            verbose: Verbosity level
            
        Returns:
            Dictionary with training results
        """
        # Prepare model checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(SEQUENCE_MODELS_DIR, f"transformer_{timestamp}.h5")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=patience, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, save_best_only=True)
        ]
        
        # Prepare validation data
        val_data = None
        if validation_data is not None:
            (X_val_home_seq, X_val_away_seq, X_val_match), y_val = validation_data
            val_data = (
                [X_val_home_seq, X_val_away_seq, X_val_match],
                y_val
            )
        
        # Train model
        self.history = self.model.fit(
            [X_home_seq, X_away_seq, X_match],
            y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Get training summary
        train_loss = self.history.history['loss'][-1]
        train_acc = self.history.history['accuracy'][-1]
        
        val_loss = None
        val_acc = None
        if val_data is not None:
            val_loss = self.history.history['val_loss'][-1]
            val_acc = self.history.history['val_accuracy'][-1]
        
        results = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epochs": len(self.history.history['loss']),
            "model_path": checkpoint_path
        }
        
        return results
    
    def predict(self, 
                X_home_seq: np.ndarray, 
                X_away_seq: np.ndarray, 
                X_match: np.ndarray) -> np.ndarray:
        """
        Make predictions for new data.
        
        Args:
            X_home_seq: Home team sequence features
            X_away_seq: Away team sequence features
            X_match: Match-specific features
            
        Returns:
            Predicted probabilities for each class
        """
        return self.model.predict([X_home_seq, X_away_seq, X_match])
    
    def evaluate(self, 
                X_home_seq: np.ndarray, 
                X_away_seq: np.ndarray, 
                X_match: np.ndarray, 
                y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_home_seq: Home team sequence features
            X_away_seq: Away team sequence features
            X_match: Match-specific features
            y: Target labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Evaluate model with built-in metrics
        eval_results = self.model.evaluate([X_home_seq, X_away_seq, X_match], y, verbose=0)
        
        # Format basic results
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = float(eval_results[i])
        
        # Get predictions for additional metrics
        y_pred_probs = self.predict(X_home_seq, X_away_seq, X_match)
        
        # Format depends on the target type
        if self.num_classes > 2:
            # Multi-class classification
            # Convert one-hot encoded y to class indices if needed
            if len(y.shape) > 1 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y
                
            # Get predicted classes
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Calculate additional metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted'
            )
            
            # Add to metrics dictionary
            metrics.update({
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            })
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Soccer-specific metrics
            try:
                # Assuming 0=home win, 1=draw, 2=away win
                # Calculate home advantage bias
                home_win_rate = np.mean(y_true == 0)
                home_win_pred_rate = np.mean(y_pred == 0)
                home_advantage_bias = home_win_pred_rate - home_win_rate
                
                # Calculate draw bias
                draw_rate = np.mean(y_true == 1)
                draw_pred_rate = np.mean(y_pred == 1)
                draw_bias = draw_pred_rate - draw_rate
                
                # Calculate upset detection (when lower probability outcome happens)
                # An upset is when the model gave <0.3 probability to the actual outcome
                upsets = []
                detected_upsets = []
                
                for i, true_class in enumerate(y_true):
                    true_prob = y_pred_probs[i, true_class]
                    if true_prob < 0.3:  # This is an upset
                        upsets.append(i)
                        if y_pred[i] == true_class:  # Model correctly predicted despite low probability
                            detected_upsets.append(i)
                
                upset_detection_rate = len(detected_upsets) / max(1, len(upsets))
                
                # Add soccer-specific metrics
                metrics.update({
                    "home_advantage_bias": float(home_advantage_bias),
                    "draw_bias": float(draw_bias),
                    "upset_detection_rate": float(upset_detection_rate)
                })
            except Exception as e:
                # If soccer-specific metrics calculation fails, log but don't crash
                pass
                
        elif self.num_classes == 2:
            # Binary classification
            # Get predicted classes (threshold 0.5)
            y_pred = (y_pred_probs > 0.5).astype(int)
            if y_pred.shape[1] == 1:
                y_pred = y_pred.flatten()
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, y_pred, average='binary'
            )
            
            # Try to calculate AUC
            try:
                if y_pred_probs.shape[1] == 1:
                    auc = roc_auc_score(y, y_pred_probs.flatten())
                else:
                    auc = roc_auc_score(y, y_pred_probs[:, 1])
            except:
                auc = 0.0
            
            # Update metrics
            metrics.update({
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc)
            })
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
        
        # Add evaluation timestamp
        metrics["evaluated_at"] = datetime.now().isoformat()
        
        return metrics
    
    def evaluate_time_series(self,
                            cv_splits: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]
                            ) -> Dict[str, Any]:
        """
        Evaluate the model on time-series cross-validation splits.
        
        Args:
            cv_splits: List of (train_data, test_data, train_y, test_y) for each fold
            
        Returns:
            Dictionary with evaluation metrics for each fold and aggregated
        """
        fold_results = []
        
        for i, (_, test_data, _, test_y) in enumerate(cv_splits):
            # Prepare test data
            X_home_seq = test_data["X_home_seq"]
            X_away_seq = test_data["X_away_seq"]
            X_match = test_data["X_match"]
            
            # Format test labels based on model output format
            if self.num_classes > 2:
                from tensorflow.keras.utils import to_categorical
                test_y_final = to_categorical(test_y, num_classes=self.num_classes)
            else:
                test_y_final = test_y
            
            # Evaluate on this fold
            fold_metrics = self.evaluate(X_home_seq, X_away_seq, X_match, test_y_final)
            fold_metrics["fold"] = i
            fold_results.append(fold_metrics)
        
        # Calculate aggregated metrics
        aggregated = {}
        metric_keys = set()
        for result in fold_results:
            metric_keys.update(result.keys())
        
        # Remove non-numeric and fold-specific metrics
        metric_keys = [k for k in metric_keys if k not in 
                     ["fold", "evaluated_at", "confusion_matrix"]]
        
        # Calculate mean and std for each metric
        for key in metric_keys:
            values = [r.get(key, 0) for r in fold_results]
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        
        # Return all results
        return {
            "fold_results": fold_results,
            "aggregated": aggregated,
            "num_folds": len(cv_splits)
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model (if None, use a timestamp-based name)
            
        Returns:
            Path where the model was saved
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(SEQUENCE_MODELS_DIR, f"transformer_{timestamp}")
        
        # Save model architecture and weights
        model_path = f"{filepath}.h5"
        self.model.save(model_path)
        
        # Save model parameters
        params = {
            "sequence_length": self.sequence_length,
            "team_feature_dim": self.team_feature_dim,
            "match_feature_dim": self.match_feature_dim,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg,
            "learning_rate": self.learning_rate,
            "num_classes": self.num_classes
        }
        
        params_path = f"{filepath}_params.pkl"
        joblib.dump(params, params_path)
        
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    @classmethod
    def load(cls, filepath: str) -> "SoccerTransformerModel":
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model (without extension)
            
        Returns:
            Loaded SoccerTransformerModel instance
        """
        # Load model parameters
        params_path = f"{filepath}_params.pkl"
        params = joblib.load(params_path)
        
        # Create model instance with loaded parameters
        model_instance = cls(**params)
        
        # Load model weights
        model_path = f"{filepath}.h5"
        model_instance.model = tf.keras.models.load_model(model_path)
        
        logger.info(f"Model loaded from {model_path}")
        
        return model_instance


class SequenceDataProcessor:
    """
    Preprocessor for preparing sequence data for teams and matches.
    """
    
    def __init__(self, sequence_length: int = 5):
        """
        Initialize the sequence data processor.
        
        Args:
            sequence_length: Number of past matches to include in each sequence
        """
        self.sequence_length = sequence_length
    
    def prepare_team_sequences(self, matches_df: pd.DataFrame, team_features: List[str], 
                              cutoff_date: Optional[pd.Timestamp] = None) -> Dict[int, np.ndarray]:
        """
        Prepare sequence data for each team from a dataframe of matches.
        
        Args:
            matches_df: DataFrame containing match data
            team_features: List of team feature column names to include
            cutoff_date: Optional cutoff date to prevent data leakage (only include matches before this date)
            
        Returns:
            Dictionary mapping team IDs to sequences of feature vectors
        """
        # Ensure matches are sorted by date
        matches_df = matches_df.copy()
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        matches_df = matches_df.sort_values('date')
        
        # Apply cutoff date if provided to prevent data leakage
        if cutoff_date is not None:
            matches_df = matches_df[matches_df['date'] < cutoff_date]
        
        # Get unique teams
        home_teams = matches_df['home_club_id'].unique()
        away_teams = matches_df['away_club_id'].unique()
        all_teams = np.unique(np.concatenate([home_teams, away_teams]))
        
        # Initialize sequence dictionary
        team_sequences = {}
        
        # Process each team
        for team_id in all_teams:
            # Get home matches
            home_matches = matches_df[matches_df['home_club_id'] == team_id].copy()
            
            # Rename columns to a consistent format
            home_features = {}
            for feature in team_features:
                if feature.startswith('home_'):
                    home_features[feature] = feature.replace('home_', '')
                else:
                    home_features[f'home_{feature}'] = feature
            
            home_df = home_matches.rename(columns=home_features)
            
            # Get away matches
            away_matches = matches_df[matches_df['away_club_id'] == team_id].copy()
            
            # Rename columns to a consistent format
            away_features = {}
            for feature in team_features:
                if feature.startswith('away_'):
                    away_features[feature] = feature.replace('away_', '')
                else:
                    away_features[f'away_{feature}'] = feature
            
            away_df = away_matches.rename(columns=away_features)
            
            # Combine home and away matches
            team_df = pd.concat([home_df, away_df])
            team_df = team_df.sort_values('date')
            
            # Get feature vectors
            features = [col for col in team_features if not col.startswith('home_') and not col.startswith('away_')]
            feature_vectors = team_df[features].values
            
            # Pad sequences if needed
            if len(feature_vectors) < self.sequence_length:
                # Pad with zeros
                padding = np.zeros((self.sequence_length - len(feature_vectors), len(features)))
                feature_vectors = np.vstack([padding, feature_vectors])
            
            team_sequences[team_id] = feature_vectors
        
        return team_sequences
    
    def prepare_match_data(self, match_row: pd.Series, team_sequences: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for a single match.
        
        Args:
            match_row: Series containing match data
            team_sequences: Dictionary mapping team IDs to sequences of feature vectors
            
        Returns:
            Tuple of (home_team_sequence, away_team_sequence, match_features)
        """
        home_team_id = match_row['home_club_id']
        away_team_id = match_row['away_club_id']
        
        # Get team sequences
        home_team_seq = team_sequences.get(home_team_id, None)
        away_team_seq = team_sequences.get(away_team_id, None)
        
        # Handle case where team doesn't have sequence data
        if home_team_seq is None or away_team_seq is None:
            feature_dim = next(iter(team_sequences.values())).shape[1] if team_sequences else 1
            if home_team_seq is None:
                home_team_seq = np.zeros((self.sequence_length, feature_dim))
            if away_team_seq is None:
                away_team_seq = np.zeros((self.sequence_length, feature_dim))
        
        # Extract most recent sequences
        home_seq = home_team_seq[-self.sequence_length:]
        away_seq = away_team_seq[-self.sequence_length:]
        
        # Ensure sequences have the correct shape
        if home_seq.shape[0] < self.sequence_length:
            padding = np.zeros((self.sequence_length - home_seq.shape[0], home_seq.shape[1]))
            home_seq = np.vstack([padding, home_seq])
        
        if away_seq.shape[0] < self.sequence_length:
            padding = np.zeros((self.sequence_length - away_seq.shape[0], away_seq.shape[1]))
            away_seq = np.vstack([padding, away_seq])
        
        # Extract match-specific features
        match_features = match_row.values
        
        return home_seq, away_seq, match_features
    
    def prepare_dataset(self, matches_df: pd.DataFrame, team_features: List[str], match_features: List[str], 
                       train_indices: Optional[np.ndarray] = None, 
                       test_indices: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare the full dataset for model training with temporal awareness.
        
        Args:
            matches_df: DataFrame containing match data
            team_features: List of team feature column names
            match_features: List of match-specific feature columns
            train_indices: Optional indices for training samples
            test_indices: Optional indices for test samples
            
        Returns:
            Tuple of (train_data, test_data, labels) where each data dict contains arrays for model inputs
        """
        # Ensure dates are in datetime format
        matches_df = matches_df.copy()
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        # Sort by date
        sorted_df = matches_df.sort_values('date')
        
        # If indices not provided, use all data
        if train_indices is None and test_indices is None:
            train_indices = np.arange(len(sorted_df))
            test_indices = np.array([], dtype=int)
        
        # Get training and test dataframes
        train_df = sorted_df.iloc[train_indices].copy() if len(train_indices) > 0 else pd.DataFrame()
        test_df = sorted_df.iloc[test_indices].copy() if len(test_indices) > 0 else pd.DataFrame()
        
        # Initialize data containers
        train_data = {
            'X_home_seq': np.zeros((len(train_df), self.sequence_length, len([f for f in team_features if not f.startswith('home_') and not f.startswith('away_')])))
            if len(train_df) > 0 else np.array([]),
            'X_away_seq': np.zeros((len(train_df), self.sequence_length, len([f for f in team_features if not f.startswith('home_') and not f.startswith('away_')])))
            if len(train_df) > 0 else np.array([]),
            'X_match': np.zeros((len(train_df), len(match_features)))
            if len(train_df) > 0 else np.array([])
        }
        
        test_data = {
            'X_home_seq': np.zeros((len(test_df), self.sequence_length, len([f for f in team_features if not f.startswith('home_') and not f.startswith('away_')])))
            if len(test_df) > 0 else np.array([]),
            'X_away_seq': np.zeros((len(test_df), self.sequence_length, len([f for f in team_features if not f.startswith('home_') and not f.startswith('away_')])))
            if len(test_df) > 0 else np.array([]),
            'X_match': np.zeros((len(test_df), len(match_features)))
            if len(test_df) > 0 else np.array([])
        }
        
        # Prepare labels
        if len(train_df) > 0:
            if 'result' in train_df.columns:
                # Classification target
                train_y = train_df['result'].values
            elif 'home_win' in train_df.columns:
                # Binary classification (home win or not)
                train_y = train_df['home_win'].values
            else:
                # Default to score difference
                train_y = train_df['home_club_goals'] - train_df['away_club_goals']
        else:
            train_y = np.array([])
        
        if len(test_df) > 0:
            if 'result' in test_df.columns:
                test_y = test_df['result'].values
            elif 'home_win' in test_df.columns:
                test_y = test_df['home_win'].values
            else:
                test_y = test_df['home_club_goals'] - test_df['away_club_goals']
        else:
            test_y = np.array([])
        
        # Process training data
        if len(train_df) > 0:
            # For training data, we can use all prior matches
            train_team_sequences = self.prepare_team_sequences(sorted_df, team_features)
            
            for i, (_, match) in enumerate(train_df.iterrows()):
                # Get match date
                match_date = match['date']
                
                # For each match, prepare team sequences using only data prior to the match
                prior_df = sorted_df[sorted_df['date'] < match_date]
                match_team_sequences = self.prepare_team_sequences(prior_df, team_features)
                
                # Get home and away team IDs
                home_team_id = match['home_club_id']
                away_team_id = match['away_club_id']
                
                # Get team sequences
                if home_team_id in match_team_sequences and away_team_id in match_team_sequences:
                    home_seq = match_team_sequences[home_team_id][-self.sequence_length:]
                    away_seq = match_team_sequences[away_team_id][-self.sequence_length:]
                    
                    # Pad if needed
                    if home_seq.shape[0] < self.sequence_length:
                        padding = np.zeros((self.sequence_length - home_seq.shape[0], home_seq.shape[1]))
                        home_seq = np.vstack([padding, home_seq])
                    
                    if away_seq.shape[0] < self.sequence_length:
                        padding = np.zeros((self.sequence_length - away_seq.shape[0], away_seq.shape[1]))
                        away_seq = np.vstack([padding, away_seq])
                    
                    # Store sequences
                    train_data['X_home_seq'][i] = home_seq
                    train_data['X_away_seq'][i] = away_seq
                
                # Store match features
                train_data['X_match'][i] = match[match_features].values
        
        # Process test data
        if len(test_df) > 0:
            for i, (_, match) in enumerate(test_df.iterrows()):
                # Get match date
                match_date = match['date']
                
                # For test data, only use data before the match date to prevent leakage
                prior_df = sorted_df[sorted_df['date'] < match_date]
                match_team_sequences = self.prepare_team_sequences(prior_df, team_features)
                
                # Get home and away team IDs
                home_team_id = match['home_club_id']
                away_team_id = match['away_club_id']
                
                # Get team sequences
                if home_team_id in match_team_sequences and away_team_id in match_team_sequences:
                    home_seq = match_team_sequences[home_team_id][-self.sequence_length:]
                    away_seq = match_team_sequences[away_team_id][-self.sequence_length:]
                    
                    # Pad if needed
                    if home_seq.shape[0] < self.sequence_length:
                        padding = np.zeros((self.sequence_length - home_seq.shape[0], home_seq.shape[1]))
                        home_seq = np.vstack([padding, home_seq])
                    
                    if away_seq.shape[0] < self.sequence_length:
                        padding = np.zeros((self.sequence_length - away_seq.shape[0], away_seq.shape[1]))
                        away_seq = np.vstack([padding, away_seq])
                    
                    # Store sequences
                    test_data['X_home_seq'][i] = home_seq
                    test_data['X_away_seq'][i] = away_seq
                
                # Store match features
                test_data['X_match'][i] = match[match_features].values
        
        # Combine labels
        all_y = np.concatenate([train_y, test_y]) if len(train_y) > 0 or len(test_y) > 0 else np.array([])
        
        return train_data, test_data, all_y
    
    def prepare_time_aware_cv_splits(self, matches_df: pd.DataFrame, team_features: List[str], 
                                    match_features: List[str], cv_splitter: Any, 
                                    groups: Optional[np.ndarray] = None) -> List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """
        Prepare data for time-aware cross-validation.
        
        Args:
            matches_df: DataFrame containing match data
            team_features: List of team feature column names
            match_features: List of match-specific feature columns
            cv_splitter: Cross-validation splitter object
            groups: Optional grouping for time-based CV
            
        Returns:
            List of (train_data, test_data, train_y, test_y) for each fold
        """
        # Ensure dates are in datetime format
        matches_df = matches_df.copy()
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        # Generate CV splits
        X = np.arange(len(matches_df))  # Just for splitting
        y = matches_df['result'].values if 'result' in matches_df.columns else np.zeros(len(matches_df))
        
        cv_splits = []
        for train_indices, test_indices in cv_splitter.split(X, y, groups=groups):
            # Prepare data for this split
            train_data, test_data, all_y = self.prepare_dataset(
                matches_df, team_features, match_features, train_indices, test_indices
            )
            
            # Get labels for train and test sets
            train_y = all_y[:len(train_indices)] if len(train_indices) > 0 else np.array([])
            test_y = all_y[len(train_indices):] if len(test_indices) > 0 else np.array([])
            
            cv_splits.append((train_data, test_data, train_y, test_y))
        
        return cv_splits 