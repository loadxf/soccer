"""
Model Calibration and Uncertainty Estimation

This module implements techniques for calibrating prediction probabilities
and estimating uncertainty for soccer match predictions. These methods improve
the reliability of predicted probabilities, making them more suitable for
decision-making and betting applications.

Techniques implemented:
- Platt scaling
- Isotonic regression
- Beta calibration
- Temperature scaling
- Conformal prediction intervals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import logging

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss

# Import project components
try:
    from src.utils.logger import get_logger
    logger = get_logger("models.calibration")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("models.calibration")

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Define paths
CALIBRATION_DIR = os.path.join(DATA_DIR, "calibration")
os.makedirs(CALIBRATION_DIR, exist_ok=True)


class ProbabilityCalibrator:
    """
    Calibrates prediction probabilities to make them more reliable.
    
    Soccer prediction models often produce overconfident or underconfident
    probability estimates. This class implements methods to calibrate these
    probabilities, making them more accurate representations of true probability.
    """
    
    CALIBRATION_METHODS = ["platt", "isotonic", "beta", "temperature", "ensemble"]
    
    def __init__(self, method: str = "platt", class_labels: Optional[List[Any]] = None):
        """
        Initialize the probability calibrator.
        
        Args:
            method: Calibration method to use
            class_labels: List of class labels (for multiclass calibration)
        """
        if method not in self.CALIBRATION_METHODS:
            raise ValueError(f"Calibration method '{method}' not supported. Choose from: {self.CALIBRATION_METHODS}")
        
        self.method = method
        self.class_labels = class_labels or [0, 1, 2]  # Default: home win, draw, away win
        self.calibrators = {}
        self.is_fitted = False
        self.calibration_info = {
            "method": method,
            "created_at": datetime.now().isoformat(),
            "trained": False,
            "performance": {}
        }
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit calibration models using predicted probabilities and true labels.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities (shape: n_samples, n_classes)
            
        Returns:
            Self for method chaining
        """
        if len(y_prob.shape) == 1:
            # Binary case, reshape to (n_samples, 2)
            y_prob = np.vstack([1 - y_prob, y_prob]).T
        
        n_classes = y_prob.shape[1]
        
        if n_classes != len(self.class_labels):
            self.class_labels = list(range(n_classes))
        
        if self.method == "platt":
            self._fit_platt(y_true, y_prob)
        elif self.method == "isotonic":
            self._fit_isotonic(y_true, y_prob)
        elif self.method == "beta":
            self._fit_beta(y_true, y_prob)
        elif self.method == "temperature":
            self._fit_temperature(y_true, y_prob)
        elif self.method == "ensemble":
            self._fit_ensemble(y_true, y_prob)
        
        self.is_fitted = True
        self.calibration_info["trained"] = True
        self.calibration_info["trained_at"] = datetime.now().isoformat()
        self.calibration_info["n_samples"] = len(y_true)
        
        # Evaluate calibration metrics
        self.evaluate(y_true, y_prob)
        
        return self
    
    def _fit_platt(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Fit Platt scaling calibration (logistic regression).
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        """
        logger.info("Fitting Platt scaling calibration")
        n_classes = y_prob.shape[1]
        
        for i in range(n_classes):
            # One-vs-rest approach for multiclass
            binary_y = (y_true == i).astype(int)
            
            # Use the predicted probability for this class as feature
            lr = LogisticRegression(C=1.0, solver='lbfgs')
            lr.fit(y_prob[:, i].reshape(-1, 1), binary_y)
            
            self.calibrators[i] = lr
    
    def _fit_isotonic(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Fit isotonic regression calibration.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        """
        logger.info("Fitting isotonic regression calibration")
        n_classes = y_prob.shape[1]
        
        for i in range(n_classes):
            # One-vs-rest approach for multiclass
            binary_y = (y_true == i).astype(int)
            
            # Fit isotonic regression
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(y_prob[:, i], binary_y)
            
            self.calibrators[i] = ir
    
    def _fit_beta(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Fit beta calibration (generalization of logistic regression).
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        """
        logger.info("Fitting beta calibration")
        n_classes = y_prob.shape[1]
        
        for i in range(n_classes):
            # One-vs-rest approach for multiclass
            binary_y = (y_true == i).astype(int)
            
            # Beta calibration uses transformed inputs: log(p/(1-p))
            # Handle edge cases to avoid log(0) or log(inf)
            p = np.clip(y_prob[:, i], 1e-15, 1-1e-15)
            x = np.log(p / (1 - p))
            x = np.column_stack([np.ones_like(x), x, -np.log(1 - p)])
            
            # Fit logistic regression with the transformed inputs
            lr = LogisticRegression(C=1.0, solver='lbfgs')
            lr.fit(x, binary_y)
            
            self.calibrators[i] = {
                'model': lr,
                'type': 'beta'
            }
    
    def _fit_temperature(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Fit temperature scaling calibration.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        """
        logger.info("Fitting temperature scaling calibration")
        
        # Temperature scaling applies a single parameter T to all logits
        # First, convert probabilities to logits
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob / (1 - y_prob))
        
        # For multiclass, we need to handle the logits differently
        if y_prob.shape[1] > 2:
            logits = np.log(y_prob + eps) - np.mean(np.log(y_prob + eps), axis=1, keepdims=True)
        
        # Optimize the temperature parameter using NLL loss
        from scipy.optimize import minimize

        def temperature_nll(T):
            # Scale logits by temperature
            scaled_logits = logits / T
            
            # Convert to probabilities
            if y_prob.shape[1] <= 2:
                scaled_probs = 1 / (1 + np.exp(-scaled_logits))
                if scaled_probs.shape[1] == 1:
                    scaled_probs = np.column_stack([1 - scaled_probs, scaled_probs])
            else:
                # For multiclass, use softmax
                exp_logits = np.exp(scaled_logits)
                scaled_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Calculate NLL loss
            loss = log_loss(y_true, scaled_probs)
            return loss
        
        # Find optimal temperature
        res = minimize(temperature_nll, x0=np.array([1.0]), method='nelder-mead')
        T = res.x[0]
        
        self.calibrators['temperature'] = T
    
    def _fit_ensemble(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """
        Fit ensemble of calibration methods.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
        """
        logger.info("Fitting ensemble calibration")
        
        # Create separate calibrators using different methods
        platt_calibrator = ProbabilityCalibrator(method="platt", class_labels=self.class_labels)
        isotonic_calibrator = ProbabilityCalibrator(method="isotonic", class_labels=self.class_labels)
        
        # Split data for training individual calibrators and the ensemble
        indices = np.arange(len(y_true))
        train_idx, ensemble_idx = train_test_split(indices, test_size=0.3, random_state=42)
        
        # Train individual calibrators
        platt_calibrator.fit(y_true[train_idx], y_prob[train_idx])
        isotonic_calibrator.fit(y_true[train_idx], y_prob[train_idx])
        
        # Get calibrated probabilities from each method
        platt_probs = platt_calibrator.calibrate(y_prob[ensemble_idx])
        isotonic_probs = isotonic_calibrator.calibrate(y_prob[ensemble_idx])
        
        # Create features for the ensemble model: original probs + calibrated probs
        ensemble_features = np.concatenate([
            y_prob[ensemble_idx],
            platt_probs,
            isotonic_probs
        ], axis=1)
        
        # Train a model to combine the calibrations
        n_classes = y_prob.shape[1]
        for i in range(n_classes):
            # One-vs-rest approach for multiclass
            binary_y = (y_true[ensemble_idx] == i).astype(int)
            
            # Train a logistic regression to combine calibrations
            ensemble_model = LogisticRegression(C=1.0, solver='lbfgs')
            ensemble_model.fit(ensemble_features, binary_y)
            
            self.calibrators[i] = {
                'model': ensemble_model,
                'type': 'ensemble',
                'components': {
                    'platt': platt_calibrator.calibrators[i],
                    'isotonic': isotonic_calibrator.calibrators[i]
                }
            }
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate predicted probabilities.
        
        Args:
            y_prob: Original predicted probabilities
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet. Call fit() first.")
        
        if len(y_prob.shape) == 1:
            # Binary case, reshape to (n_samples, 2)
            y_prob = np.vstack([1 - y_prob, y_prob]).T
        
        n_classes = y_prob.shape[1]
        n_samples = y_prob.shape[0]
        
        # Initialize calibrated probabilities
        calibrated = np.zeros((n_samples, n_classes))
        
        if self.method == "temperature":
            # Temperature scaling is applied to all classes simultaneously
            T = self.calibrators['temperature']
            
            # Convert to logits
            eps = 1e-15
            y_prob_clip = np.clip(y_prob, eps, 1 - eps)
            
            if n_classes <= 2:
                logits = np.log(y_prob_clip / (1 - y_prob_clip))
                # Scale logits by temperature
                scaled_logits = logits / T
                # Convert back to probabilities
                calibrated = 1 / (1 + np.exp(-scaled_logits))
                if n_classes == 1:
                    calibrated = np.column_stack([1 - calibrated, calibrated])
            else:
                logits = np.log(y_prob_clip + eps) - np.mean(np.log(y_prob_clip + eps), axis=1, keepdims=True)
                # Scale logits by temperature
                scaled_logits = logits / T
                # Convert back to probabilities using softmax
                exp_logits = np.exp(scaled_logits)
                calibrated = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        elif self.method == "ensemble":
            # For ensemble, we need to get calibrated probabilities from components first
            platt_probs = np.zeros((n_samples, n_classes))
            isotonic_probs = np.zeros((n_samples, n_classes))
            
            for i in range(n_classes):
                # Get component calibrators
                platt_calibrator = self.calibrators[i]['components']['platt']
                isotonic_calibrator = self.calibrators[i]['components']['isotonic']
                
                # Apply each calibration
                platt_probs[:, i] = platt_calibrator.predict_proba(y_prob[:, i].reshape(-1, 1))[:, 1]
                isotonic_probs[:, i] = isotonic_calibrator.predict(y_prob[:, i])
            
            # Combine original and calibrated probabilities
            ensemble_features = np.concatenate([y_prob, platt_probs, isotonic_probs], axis=1)
            
            # Apply ensemble model for each class
            for i in range(n_classes):
                ensemble_model = self.calibrators[i]['model']
                calibrated[:, i] = ensemble_model.predict_proba(ensemble_features)[:, 1]
            
            # Normalize to ensure probabilities sum to 1
            calibrated = calibrated / np.sum(calibrated, axis=1, keepdims=True)
            
        elif self.method == "beta":
            # Beta calibration requires special handling
            for i in range(n_classes):
                # Get the beta calibration model
                beta_calibrator = self.calibrators[i]['model']
                
                # Transform inputs as in fit method
                p = np.clip(y_prob[:, i], 1e-15, 1-1e-15)
                x = np.log(p / (1 - p))
                x = np.column_stack([np.ones_like(x), x, -np.log(1 - p)])
                
                # Apply calibration
                calibrated[:, i] = beta_calibrator.predict_proba(x)[:, 1]
            
            # Normalize to ensure probabilities sum to 1
            calibrated = calibrated / np.sum(calibrated, axis=1, keepdims=True)
            
        else:
            # Platt and isotonic methods
            for i in range(n_classes):
                calibrator = self.calibrators[i]
                
                if self.method == "platt":
                    # Platt scaling uses logistic regression
                    calibrated[:, i] = calibrator.predict_proba(y_prob[:, i].reshape(-1, 1))[:, 1]
                else:
                    # Isotonic regression directly predicts calibrated probabilities
                    calibrated[:, i] = calibrator.predict(y_prob[:, i])
            
            # Normalize to ensure probabilities sum to 1
            calibrated = calibrated / np.sum(calibrated, axis=1, keepdims=True)
        
        return calibrated
    
    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Evaluate calibration performance.
        
        Args:
            y_true: True labels
            y_prob: Original predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet. Call fit() first.")
        
        # Get calibrated probabilities
        calibrated_prob = self.calibrate(y_prob)
        
        # Calculate metrics for both original and calibrated probabilities
        orig_loss = log_loss(y_true, y_prob)
        cal_loss = log_loss(y_true, calibrated_prob)
        
        orig_brier = brier_score_loss(y_true, y_prob, sample_weight=None, pos_label=None)
        cal_brier = brier_score_loss(y_true, calibrated_prob, sample_weight=None, pos_label=None)
        
        # Calculate reliability diagrams data (for plotting)
        n_classes = y_prob.shape[1]
        reliability_data = []
        
        for i in range(n_classes):
            # One-vs-rest approach for multiclass
            binary_y = (y_true == i).astype(int)
            
            # Calculate calibration curve for original probabilities
            prob_true_orig, prob_pred_orig = calibration_curve(
                binary_y, y_prob[:, i], n_bins=10, strategy='uniform'
            )
            
            # Calculate calibration curve for calibrated probabilities
            prob_true_cal, prob_pred_cal = calibration_curve(
                binary_y, calibrated_prob[:, i], n_bins=10, strategy='uniform'
            )
            
            reliability_data.append({
                'class': i,
                'original': {
                    'true_prob': prob_true_orig.tolist(),
                    'pred_prob': prob_pred_orig.tolist()
                },
                'calibrated': {
                    'true_prob': prob_true_cal.tolist(),
                    'pred_prob': prob_pred_cal.tolist()
                }
            })
        
        # Store evaluation results
        evaluation = {
            'original_log_loss': float(orig_loss),
            'calibrated_log_loss': float(cal_loss),
            'original_brier_score': float(orig_brier),
            'calibrated_brier_score': float(cal_brier),
            'log_loss_improvement': float(orig_loss - cal_loss),
            'brier_score_improvement': float(orig_brier - cal_brier),
            'reliability_data': reliability_data,
            'evaluated_at': datetime.now().isoformat()
        }
        
        self.calibration_info['performance'] = evaluation
        
        return evaluation
    
    def plot_reliability_diagram(self, reliability_data: Optional[List[Dict]] = None, 
                                output_path: Optional[str] = None) -> None:
        """
        Plot reliability diagram showing calibration performance.
        
        Args:
            reliability_data: Optional reliability data (if not provided, use stored data)
            output_path: Optional path to save the plot
        """
        if reliability_data is None:
            reliability_data = self.calibration_info.get('performance', {}).get('reliability_data')
        
        if reliability_data is None:
            raise ValueError("No reliability data available. Run evaluate() first.")
        
        n_classes = len(reliability_data)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the perfectly calibrated line
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        
        # Color map for classes
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']
        
        # Plot each class
        for i, data in enumerate(reliability_data):
            color = colors[i % len(colors)]
            class_label = self.class_labels[i] if i < len(self.class_labels) else i
            
            # Original probabilities (dotted line)
            ax.plot(
                data['original']['pred_prob'], 
                data['original']['true_prob'],
                linestyle=':', marker='o', color=color,
                label=f'Original (Class {class_label})'
            )
            
            # Calibrated probabilities (solid line)
            ax.plot(
                data['calibrated']['pred_prob'], 
                data['calibrated']['true_prob'],
                linestyle='-', marker='x', color=color,
                label=f'Calibrated (Class {class_label})'
            )
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability (Fraction of Positives)')
        ax.set_title(f'Reliability Diagram - {self.method.capitalize()} Calibration')
        ax.legend(loc='upper left')
        ax.grid(True)
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the calibrator to a file.
        
        Args:
            filepath: Optional filepath to save to
            
        Returns:
            Filepath where the calibrator was saved
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")
        
        if filepath is None:
            # Generate default filepath
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(CALIBRATION_DIR, f"calibrator_{self.method}_{timestamp}.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'class_labels': self.class_labels,
                'calibrators': self.calibrators,
                'is_fitted': self.is_fitted,
                'calibration_info': self.calibration_info
            }, f)
        
        logger.info(f"Saved calibrator to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "ProbabilityCalibrator":
        """
        Load a calibrator from a file.
        
        Args:
            filepath: Path to the saved calibrator
            
        Returns:
            Loaded ProbabilityCalibrator instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        calibrator = cls(method=data['method'], class_labels=data['class_labels'])
        calibrator.calibrators = data['calibrators']
        calibrator.is_fitted = data['is_fitted']
        calibrator.calibration_info = data['calibration_info']
        
        logger.info(f"Loaded calibrator from {filepath}")
        return calibrator


class ConformalPredictionIntervals:
    """
    Implements conformal prediction for uncertainty estimation.
    
    Conformal prediction provides rigorous uncertainty estimates with 
    statistical guarantees, regardless of the underlying model.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal prediction.
        
        Args:
            alpha: Significance level (e.g., 0.1 for 90% prediction intervals)
        """
        self.alpha = alpha
        self.threshold = None
        self.is_fitted = False
        self.info = {
            "method": "conformal_prediction",
            "alpha": alpha,
            "created_at": datetime.now().isoformat(),
            "trained": False
        }
    
    def fit(self, y_probs: np.ndarray, y_true: np.ndarray) -> "ConformalPredictionIntervals":
        """
        Fit conformal prediction intervals.
        
        Args:
            y_probs: Predicted probabilities (n_samples, n_classes)
            y_true: True class labels
            
        Returns:
            Self for method chaining
        """
        # Extract non-conformity scores: 1 - P(true class)
        scores = []
        
        for i, y in enumerate(y_true):
            # Get probability of the true class
            prob_true_class = y_probs[i, y]
            # Non-conformity score: 1 - P(true class)
            scores.append(1 - prob_true_class)
        
        scores = np.array(scores)
        
        # Find the threshold for the desired significance level
        self.threshold = np.quantile(scores, 1 - self.alpha)
        
        self.is_fitted = True
        self.info["trained"] = True
        self.info["trained_at"] = datetime.now().isoformat()
        self.info["n_samples"] = len(y_true)
        self.info["threshold"] = float(self.threshold)
        
        return self
    
    def predict_sets(self, y_probs: np.ndarray) -> List[List[int]]:
        """
        Predict conformal prediction sets for each sample.
        
        Args:
            y_probs: Predicted probabilities (n_samples, n_classes)
            
        Returns:
            List of prediction sets (one set per sample)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        n_samples = y_probs.shape[0]
        n_classes = y_probs.shape[1]
        
        prediction_sets = []
        
        for i in range(n_samples):
            # Find classes where 1 - P(class) <= threshold
            # This is equivalent to P(class) >= 1 - threshold
            pred_set = [c for c in range(n_classes) if y_probs[i, c] >= 1 - self.threshold]
            prediction_sets.append(pred_set)
        
        return prediction_sets
    
    def evaluate(self, y_probs: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate conformal prediction performance.
        
        Args:
            y_probs: Predicted probabilities
            y_true: True class labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        prediction_sets = self.predict_sets(y_probs)
        
        # Calculate coverage (true class in prediction set)
        coverage = 0
        set_sizes = []
        
        for i, true_class in enumerate(y_true):
            pred_set = prediction_sets[i]
            set_sizes.append(len(pred_set))
            if true_class in pred_set:
                coverage += 1
        
        coverage = coverage / len(y_true)
        avg_set_size = np.mean(set_sizes)
        
        return {
            "coverage": float(coverage),
            "target_coverage": 1 - self.alpha,
            "average_set_size": float(avg_set_size),
            "threshold": float(self.threshold),
            "evaluated_at": datetime.now().isoformat()
        }
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the conformal prediction model.
        
        Args:
            filepath: Optional filepath to save to
            
        Returns:
            Filepath where the model was saved
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        if filepath is None:
            # Generate default filepath
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(CALIBRATION_DIR, f"conformal_{timestamp}.pkl")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'threshold': self.threshold,
                'is_fitted': self.is_fitted,
                'info': self.info
            }, f)
        
        logger.info(f"Saved conformal prediction model to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> "ConformalPredictionIntervals":
        """
        Load a conformal prediction model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ConformalPredictionIntervals instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(alpha=data['alpha'])
        model.threshold = data['threshold']
        model.is_fitted = data['is_fitted']
        model.info = data['info']
        
        logger.info(f"Loaded conformal prediction model from {filepath}")
        return model


def calibrate_model_predictions(model_path: str, X: np.ndarray, y: np.ndarray,
                              method: str = "platt", output_dir: Optional[str] = None) -> str:
    """
    Calibrate a model's predictions and save the calibrator.
    
    Args:
        model_path: Path to the model file
        X: Feature matrix for calibration
        y: True labels for calibration
        method: Calibration method
        output_dir: Optional directory to save calibrator
        
    Returns:
        Path to the saved calibrator
    """
    logger.info(f"Calibrating model predictions using {method} method")
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get model type from filename or model object
        model_type = os.path.basename(model_path).split('_')[0]
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
    # Get predictions from the model
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)
    elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
        y_prob = model.model.predict_proba(X)
    else:
        raise ValueError("Model does not have predict_proba method")
    
    # Initialize calibrator
    calibrator = ProbabilityCalibrator(method=method)
    
    # Fit calibrator
    calibrator.fit(y, y_prob)
    
    # Create output path
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(CALIBRATION_DIR, f"{model_type}_{method}_{timestamp}.pkl")
    else:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{model_type}_{method}_{timestamp}.pkl")
    
    # Save calibrator
    calibrator.save(output_path)
    
    # Plot and save reliability diagram
    diagram_path = output_path.replace('.pkl', '_reliability.png')
    calibrator.plot_reliability_diagram(output_path=diagram_path)
    
    logger.info(f"Saved calibrated model to {output_path}")
    return output_path 