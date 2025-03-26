"""
Model Explainability module for Soccer Prediction System.
Provides tools and utilities for explaining model predictions and interpreting model behavior.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
import joblib

# Conditionally import explainability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from pdpbox import pdp
    PDPBOX_AVAILABLE = True
except ImportError:
    PDPBOX_AVAILABLE = False

# Conditionally import alibi explainers but don't attempt to import specifics
# Just mark as available so other code can check before trying to use it
try:
    import alibi
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False

try:
    import eli5
    from eli5.sklearn import explain_weights, explain_prediction
    ELI5_AVAILABLE = True
except ImportError:
    ELI5_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from sklearn.inspection import permutation_importance

# Import project components
from src.utils.logger import get_logger
from src.models.baseline import BaselineMatchPredictor
try:
    from src.models.advanced import AdvancedMatchPredictor
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False

try:
    from src.models.ensemble import EnsemblePredictor
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback default if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.explainability")

# Define paths
EXPLAINABILITY_DIR = os.path.join(DATA_DIR, "explainability")
PLOTS_DIR = os.path.join(EXPLAINABILITY_DIR, "plots")
os.makedirs(EXPLAINABILITY_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# Function to get alibi explainers if available
def get_alibi_explainer(explainer_type):
    """Get alibi explainer if available, returns None if not"""
    if not ALIBI_AVAILABLE:
        logger.warning(f"Alibi library not available for {explainer_type} explainer")
        return None
    
    try:
        if explainer_type == "anchor":
            from alibi.explainers import AnchorTabular
            return AnchorTabular
        elif explainer_type == "counterfactual":
            from alibi.explainers import CounterfactualProto
            return CounterfactualProto
        else:
            logger.warning(f"Unknown alibi explainer type: {explainer_type}")
            return None
    except ImportError as e:
        logger.warning(f"Error importing alibi explainer {explainer_type}: {e}")
        return None


class ModelExplainer:
    """
    Model explainability class that provides various methods to interpret and explain 
    model predictions for soccer match outcomes.
    """
    
    SUPPORTED_EXPLAINERS = [
        "shap", 
        "lime", 
        "pdp", 
        "permutation", 
        "anchor",
        "counterfactual",
        "eli5"
    ]
    
    def __init__(self, model, feature_names: List[str], class_names: List[str] = None):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained model instance (BaselineMatchPredictor, AdvancedMatchPredictor, or EnsemblePredictor)
            feature_names: List of feature names used by the model
            class_names: List of class names for classification models (default: ['Home Win', 'Draw', 'Away Win'])
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['Home Win', 'Draw', 'Away Win']
        
        # Determine model type and prediction function
        self.model_type = self._determine_model_type()
        self.predict_fn = self._get_prediction_function()
        
        # Initialize explainers
        self.explainers = {}
        
        # Metadata
        self.metadata = {
            "model_type": self.model_type,
            "model_name": getattr(model, "name", "Unknown"),
            "num_features": len(feature_names),
            "classes": self.class_names,
            "timestamp": datetime.now().isoformat()
        }
    
    def _determine_model_type(self) -> str:
        """Determine the type of model we're explaining."""
        if isinstance(self.model, BaselineMatchPredictor):
            return "baseline"
        elif ADVANCED_MODELS_AVAILABLE and isinstance(self.model, AdvancedMatchPredictor):
            return "advanced"
        elif ENSEMBLE_AVAILABLE and isinstance(self.model, EnsemblePredictor):
            return "ensemble"
        elif hasattr(self.model, "predict_proba"):
            return "sklearn"
        elif TENSORFLOW_AVAILABLE and isinstance(self.model, tf.keras.Model):
            return "keras"
        elif hasattr(self.model, "booster_"):
            return "xgboost"
        elif hasattr(self.model, "model"):
            return "lightgbm"
        else:
            logger.warning("Unknown model type, some explainers may not work correctly")
            return "unknown"
    
    def _get_prediction_function(self) -> Callable:
        """Get the appropriate prediction function based on model type."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        elif TENSORFLOW_AVAILABLE and isinstance(self.model, tf.keras.Model):
            return self.model.predict
        elif self.model_type in ["baseline", "advanced", "ensemble"]:
            return self.model.predict_proba
        else:
            # Fallback to predict if predict_proba is not available
            return self.model.predict
    
    def explain_with_shap(self, X: np.ndarray, sample_idx: Optional[int] = None, 
                         background_samples: int = 100, max_display: int = 20) -> Dict[str, Any]:
        """
        Generate SHAP explanations for the model.
        
        Args:
            X: Feature matrix to explain
            sample_idx: Optional index of single sample to explain (if None, explain all samples)
            background_samples: Number of background samples for SHAP explainer
            max_display: Maximum number of features to display in plots
            
        Returns:
            Dict containing SHAP values and explanations
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is required for SHAP explanations. Please install it with 'pip install shap'")
            
        # Create SHAP explainer if not already created
        if "shap" not in self.explainers:
            if self.model_type == "keras":
                # For neural networks
                background = X[:min(background_samples, len(X))]
                self.explainers["shap"] = shap.DeepExplainer(self.model, background)
            elif self.model_type in ["xgboost", "lightgbm"]:
                # For tree-based models
                self.explainers["shap"] = shap.TreeExplainer(self.model)
            else:
                # For other model types
                background = X[:min(background_samples, len(X))]
                self.explainers["shap"] = shap.KernelExplainer(self.predict_fn, background)
        
        # Generate SHAP values
        explainer = self.explainers["shap"]
        if sample_idx is not None:
            # Explain a single sample
            X_explain = X[sample_idx].reshape(1, -1)
            shap_values = explainer.shap_values(X_explain)
            expected_value = explainer.expected_value
        else:
            # Explain all samples
            shap_values = explainer.shap_values(X)
            expected_value = explainer.expected_value
        
        # Generate plots
        if sample_idx is not None:
            # Force plot for single sample
            plt.figure(figsize=(20, 3))
            force_plot = shap.force_plot(
                expected_value[0] if isinstance(expected_value, list) else expected_value,
                shap_values[0] if isinstance(shap_values, list) else shap_values,
                X_explain, 
                feature_names=self.feature_names,
                show=False
            )
            
            # Save plots
            force_plot_path = os.path.join(PLOTS_DIR, f"shap_force_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            shap.save_html(force_plot_path.replace('.png', '.html'), force_plot)
            
            # Summary plot for all classes
            plt.figure(figsize=(10, 10))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values, X_explain, feature_names=self.feature_names, 
                                class_names=self.class_names, show=False)
            else:
                shap.summary_plot(shap_values, X_explain, feature_names=self.feature_names, show=False)
            summary_plot_path = os.path.join(PLOTS_DIR, f"shap_summary_plot_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(summary_plot_path)
            plt.close()
        else:
            # Summary plot for all samples
            plt.figure(figsize=(10, 10))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values, X, feature_names=self.feature_names, 
                                class_names=self.class_names, show=False)
            else:
                shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
            summary_plot_path = os.path.join(PLOTS_DIR, f"shap_summary_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(summary_plot_path)
            plt.close()
        
        # Return SHAP values and paths to plots
        result = {
            "method": "shap",
            "timestamp": datetime.now().isoformat(),
            "plots": {
                "summary_plot": summary_plot_path
            },
            "metadata": {
                "num_samples": 1 if sample_idx is not None else len(X),
                "sample_idx": sample_idx,
                "background_samples": background_samples
            }
        }
        
        if sample_idx is not None:
            result["plots"]["force_plot"] = force_plot_path.replace('.png', '.html')
        
        return result
    
    def explain_with_lime(self, X: np.ndarray, sample_idx: int, 
                          num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanations for a specific sample.
        
        Args:
            X: Feature matrix
            sample_idx: Index of the sample to explain
            num_features: Number of top features to include in the explanation
            
        Returns:
            Dict containing LIME explanation
        """
        # Create LIME explainer if not already created
        if "lime" not in self.explainers:
            self.explainers["lime"] = lime.lime_tabular.LimeTabularExplainer(
                X, 
                feature_names=self.feature_names,
                class_names=self.class_names,
                discretize_continuous=True,
                mode="classification"
            )
        
        # Get sample to explain
        sample = X[sample_idx]
        
        # Get explanation
        explainer = self.explainers["lime"]
        explanation = explainer.explain_instance(
            sample, 
            self.predict_fn,
            num_features=num_features
        )
        
        # Generate and save plot
        plt.figure(figsize=(10, 6))
        lime_plot_path = os.path.join(PLOTS_DIR, f"lime_explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        explanation.save_to_file(lime_plot_path.replace('.png', '.html'))
        
        # Extract explanation data
        explanation_data = {}
        for i, class_idx in enumerate(range(len(self.class_names))):
            class_name = self.class_names[i]
            exp_list = explanation.as_list(label=class_idx)
            explanation_data[class_name] = {
                "features": [item[0] for item in exp_list],
                "weights": [item[1] for item in exp_list]
            }
        
        return {
            "method": "lime",
            "timestamp": datetime.now().isoformat(),
            "explanation": explanation_data,
            "plots": {
                "lime_plot": lime_plot_path.replace('.png', '.html')
            },
            "metadata": {
                "sample_idx": sample_idx,
                "num_features": num_features,
                "prediction": self.predict_fn(sample.reshape(1, -1))[0].tolist()
            }
        }
    
    def explain_with_pdp(self, X: np.ndarray, feature_idx: int, 
                         num_ice_lines: int = 50) -> Dict[str, Any]:
        """
        Generate Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots.
        
        Args:
            X: Feature matrix
            feature_idx: Index of the feature to explain
            num_ice_lines: Number of ICE lines to display
            
        Returns:
            Dict containing PDP and ICE explanations
        """
        feature_name = self.feature_names[feature_idx]
        
        # Create PDPBox plot
        pdp_isolate = pdp.pdp_isolate(
            model=self.model,
            dataset=X,
            model_features=self.feature_names,
            feature=feature_name
        )
        
        # Generate plot
        plt.figure(figsize=(10, 6))
        pdp_plot = pdp.pdp_plot(
            pdp_isolate, 
            feature_name,
            plot_lines=True,
            frac_to_plot=min(num_ice_lines / len(X), 1.0),
            plot_pts_dist=True,
            center=True
        )
        
        # Save plot
        pdp_plot_path = os.path.join(PLOTS_DIR, f"pdp_plot_{feature_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(pdp_plot_path)
        plt.close()
        
        return {
            "method": "pdp",
            "timestamp": datetime.now().isoformat(),
            "feature": feature_name,
            "plots": {
                "pdp_plot": pdp_plot_path
            },
            "metadata": {
                "feature_idx": feature_idx,
                "num_ice_lines": num_ice_lines,
                "mean_impact": pdp_isolate.pdp.mean()
            }
        }
    
    def explain_with_permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                           n_repeats: int = 10, random_state: int = 42) -> Dict[str, Any]:
        """
        Calculate permutation feature importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            
        Returns:
            Dict containing permutation importance results
        """
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model, X, y, 
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Generate plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(PLOTS_DIR, f"permutation_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Save CSV
        csv_path = os.path.join(EXPLAINABILITY_DIR, f"permutation_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        importance_df.to_csv(csv_path, index=False)
        
        return {
            "method": "permutation_importance",
            "timestamp": datetime.now().isoformat(),
            "importance": importance_df.to_dict(orient='records'),
            "plots": {
                "importance_plot": plot_path
            },
            "metadata": {
                "n_repeats": n_repeats,
                "top_features": importance_df.head(10)['Feature'].tolist()
            }
        }
    
    def explain_with_anchor(self, X: np.ndarray, sample_idx: int, 
                           threshold: float = 0.95) -> Dict[str, Any]:
        """
        Generate Anchor explanations for a specific sample.
        
        Args:
            X: Feature matrix
            sample_idx: Index of the sample to explain
            threshold: Minimum precision threshold for the anchor
            
        Returns:
            Dict containing Anchor explanation
        """
        # Create Anchor explainer if not already created
        if "anchor" not in self.explainers:
            # Create a predict function that returns class labels instead of probabilities
            predict_fn = lambda x: np.argmax(self.predict_fn(x), axis=1)
            
            self.explainers["anchor"] = get_alibi_explainer("anchor")(
                predict_fn,
                feature_names=self.feature_names
            )
            # Fit explainer on data
            self.explainers["anchor"].fit(X)
        
        # Get sample to explain
        sample = X[sample_idx].reshape(1, -1)
        
        # Get explanation
        explainer = self.explainers["anchor"]
        explanation = explainer.explain(sample, threshold=threshold)
        
        # Predicted class
        predicted_class = np.argmax(self.predict_fn(sample), axis=1)[0]
        predicted_class_name = self.class_names[predicted_class]
        
        return {
            "method": "anchor",
            "timestamp": datetime.now().isoformat(),
            "sample_idx": sample_idx,
            "predicted_class": predicted_class_name,
            "anchor": explanation.anchor,
            "precision": explanation.precision,
            "coverage": explanation.coverage,
            "metadata": {
                "threshold": threshold
            }
        }
    
    def explain_with_counterfactual(self, X: np.ndarray, sample_idx: int, 
                                   target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate counterfactual explanations for a specific sample.
        
        Args:
            X: Feature matrix
            sample_idx: Index of the sample to explain
            target_class: Target class for counterfactual (if None, use next most likely class)
            
        Returns:
            Dict containing counterfactual explanation
        """
        # Create counterfactual explainer if not already created
        if "counterfactual" not in self.explainers:
            # Adapt model for counterfactual explainer
            shape = (1,) + X[0].shape
            cf_explainer = get_alibi_explainer("counterfactual")(
                self.predict_fn,
                shape,
                kappa=0.0,
                theta=0.0,
                use_kdtree=True,
                feature_names=self.feature_names
            )
            # Fit explainer on data
            cf_explainer.fit(X)
            self.explainers["counterfactual"] = cf_explainer
        
        # Get sample to explain
        sample = X[sample_idx].reshape(1, -1)
        
        # Determine target class if not provided
        pred_probs = self.predict_fn(sample)[0]
        pred_class = np.argmax(pred_probs)
        
        if target_class is None:
            # Use next most likely class as target
            sorted_indices = np.argsort(pred_probs)[::-1]
            target_class = sorted_indices[1] if sorted_indices[0] == pred_class else sorted_indices[0]
        
        # Get explanation
        explainer = self.explainers["counterfactual"]
        explanation = explainer.explain(sample, target_class=target_class)
        
        # Extract counterfactual
        cf = explanation.cf["X"]
        cf_class = np.argmax(self.predict_fn(cf), axis=1)[0]
        
        # Create DataFrame with original and counterfactual
        cf_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Original": sample[0],
            "Counterfactual": cf[0],
            "Difference": cf[0] - sample[0],
            "Pct_Change": (cf[0] - sample[0]) / (np.abs(sample[0]) + 1e-10) * 100
        })
        
        # Sort by absolute difference
        cf_df = cf_df.sort_values("Difference", key=lambda x: np.abs(x), ascending=False)
        
        # Save DataFrame
        csv_path = os.path.join(EXPLAINABILITY_DIR, f"counterfactual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        cf_df.to_csv(csv_path, index=False)
        
        return {
            "method": "counterfactual",
            "timestamp": datetime.now().isoformat(),
            "sample_idx": sample_idx,
            "original_class": self.class_names[pred_class],
            "counterfactual_class": self.class_names[cf_class],
            "counterfactual_data": cf_df.to_dict(orient='records'),
            "metadata": {
                "target_class": self.class_names[target_class],
                "csv_path": csv_path
            }
        }
    
    def explain_with_eli5(self, X: np.ndarray, sample_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate ELI5 explanations for the model and/or a specific prediction.
        
        Args:
            X: Feature matrix
            sample_idx: Optional index of a sample to explain (if None, explain the model)
            
        Returns:
            Dict containing ELI5 explanation
        """
        # ELI5 only works with certain sklearn-compatible models
        if self.model_type not in ["sklearn", "xgboost", "lightgbm"]:
            logger.warning(f"ELI5 explanation not supported for model type: {self.model_type}")
            return {
                "method": "eli5",
                "timestamp": datetime.now().isoformat(),
                "error": f"ELI5 explanation not supported for model type: {self.model_type}"
            }
        
        result = {
            "method": "eli5",
            "timestamp": datetime.now().isoformat()
        }
        
        # Get model weights explanation
        try:
            weights_explanation = explain_weights(
                self.model,
                feature_names=self.feature_names
            )
            weights_html = eli5.formatters.html.format_as_html(weights_explanation)
            
            # Save HTML
            weights_path = os.path.join(EXPLAINABILITY_DIR, f"eli5_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            with open(weights_path, 'w') as f:
                f.write(weights_html)
            
            result["model_weights"] = {
                "html_path": weights_path
            }
        except Exception as e:
            logger.warning(f"Failed to generate ELI5 weights explanation: {e}")
            result["model_weights"] = {"error": str(e)}
        
        # Get prediction explanation if sample_idx is provided
        if sample_idx is not None:
            try:
                sample = X[sample_idx].reshape(1, -1)
                pred_explanation = explain_prediction(
                    self.model,
                    sample[0],
                    feature_names=self.feature_names
                )
                pred_html = eli5.formatters.html.format_as_html(pred_explanation)
                
                # Save HTML
                pred_path = os.path.join(EXPLAINABILITY_DIR, f"eli5_prediction_{sample_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                with open(pred_path, 'w') as f:
                    f.write(pred_html)
                
                result["prediction"] = {
                    "sample_idx": sample_idx,
                    "html_path": pred_path,
                    "predicted_class": self.class_names[np.argmax(self.predict_fn(sample)[0])]
                }
            except Exception as e:
                logger.warning(f"Failed to generate ELI5 prediction explanation: {e}")
                result["prediction"] = {"error": str(e)}
        
        return result
    
    def explain_prediction(self, sample: np.ndarray, methods: List[str] = None) -> Dict[str, Any]:
        """
        Generate explanations for a single prediction using multiple methods.
        
        Args:
            sample: Feature vector to explain (1D array)
            methods: List of explainer methods to use (default: all supported methods)
            
        Returns:
            Dict containing explanations from all methods
        """
        if methods is None:
            methods = ["shap", "lime", "anchor", "counterfactual"]
        
        # Reshape sample if needed
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        # Initialize results dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "prediction": {
                "probabilities": self.predict_fn(sample)[0].tolist(),
                "class": self.class_names[np.argmax(self.predict_fn(sample)[0])]
            },
            "explanations": {}
        }
        
        # Generate explanations for each method
        if "shap" in methods:
            try:
                # Create temporary matrix with the sample as the only element
                X_temp = np.repeat(sample, 100, axis=0)
                shap_results = self.explain_with_shap(X_temp, sample_idx=0)
                results["explanations"]["shap"] = shap_results
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {e}")
                results["explanations"]["shap"] = {"error": str(e)}
        
        if "lime" in methods:
            try:
                # Create temporary matrix with the sample as the only element
                X_temp = np.repeat(sample, 100, axis=0)
                lime_results = self.explain_with_lime(X_temp, sample_idx=0)
                results["explanations"]["lime"] = lime_results
            except Exception as e:
                logger.error(f"Error generating LIME explanation: {e}")
                results["explanations"]["lime"] = {"error": str(e)}
        
        if "anchor" in methods:
            try:
                # Create temporary matrix with the sample as the only element
                X_temp = np.repeat(sample, 100, axis=0)
                anchor_results = self.explain_with_anchor(X_temp, sample_idx=0)
                results["explanations"]["anchor"] = anchor_results
            except Exception as e:
                logger.error(f"Error generating Anchor explanation: {e}")
                results["explanations"]["anchor"] = {"error": str(e)}
        
        if "counterfactual" in methods:
            try:
                # Create temporary matrix with the sample as the only element
                X_temp = np.repeat(sample, 100, axis=0)
                cf_results = self.explain_with_counterfactual(X_temp, sample_idx=0)
                results["explanations"]["counterfactual"] = cf_results
            except Exception as e:
                logger.error(f"Error generating Counterfactual explanation: {e}")
                results["explanations"]["counterfactual"] = {"error": str(e)}
        
        # Save results
        result_path = os.path.join(EXPLAINABILITY_DIR, f"prediction_explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, default=lambda o: str(o) if isinstance(o, (np.ndarray, np.generic)) else o)
        
        return results
    
    def generate_global_explanations(self, X: np.ndarray, y: np.ndarray = None, 
                                   methods: List[str] = None) -> Dict[str, Any]:
        """
        Generate global explanations for the model using multiple methods.
        
        Args:
            X: Feature matrix
            y: Optional target variable (required for some methods)
            methods: List of explainer methods to use
            
        Returns:
            Dict containing global explanations
        """
        if methods is None:
            methods = ["shap", "permutation"]
        
        # Initialize results dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "explanations": {}
        }
        
        # Generate explanations for each method
        if "shap" in methods:
            try:
                # Subsample if X is large
                if len(X) > 1000:
                    indices = np.random.choice(len(X), 1000, replace=False)
                    X_sample = X[indices]
                else:
                    X_sample = X
                
                shap_results = self.explain_with_shap(X_sample)
                results["explanations"]["shap"] = shap_results
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {e}")
                results["explanations"]["shap"] = {"error": str(e)}
        
        if "permutation" in methods and y is not None:
            try:
                # Subsample if X is large
                if len(X) > 1000:
                    indices = np.random.choice(len(X), 1000, replace=False)
                    X_sample = X[indices]
                    y_sample = y[indices]
                else:
                    X_sample = X
                    y_sample = y
                
                perm_results = self.explain_with_permutation_importance(X_sample, y_sample)
                results["explanations"]["permutation"] = perm_results
            except Exception as e:
                logger.error(f"Error generating permutation importance: {e}")
                results["explanations"]["permutation"] = {"error": str(e)}
        
        # Save results
        result_path = os.path.join(EXPLAINABILITY_DIR, f"global_explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(result_path, 'w') as f:
            json.dump(results, f, default=lambda o: str(o) if isinstance(o, (np.ndarray, np.generic)) else o)
        
        return results


def generate_model_explanations(
    model_path: str,
    dataset_name: str = None,
    feature_type: str = None,
    target_col: str = "result",
    methods: List[str] = None,
    sample_indices: List[int] = None,
    num_samples: int = 5,
    save_dir: str = None
) -> Dict[str, Any]:
    """
    Comprehensive function to generate explanations for a model.
    
    Args:
        model_path: Path to the saved model
        dataset_name: Name of dataset to use (if None, use the one from the model)
        feature_type: Type of features to use (if None, use the one from the model)
        target_col: Name of the target column
        methods: List of explanation methods to use
        sample_indices: Specific sample indices to explain
        num_samples: Number of random samples to explain if sample_indices is None
        save_dir: Directory to save explanations (if None, use default)
        
    Returns:
        Dict containing all explanations
    """
    from src.models.training import load_feature_data
    
    logger.info(f"Generating explanations for model: {model_path}")
    
    # Load the model
    try:
        # Try to load as BaselineMatchPredictor first
        model = BaselineMatchPredictor.load(model_path)
    except Exception:
        try:
            # Try to load as AdvancedMatchPredictor
            from src.models.advanced import AdvancedMatchPredictor
            model = AdvancedMatchPredictor.load(model_path)
        except Exception:
            try:
                # Try to load as EnsemblePredictor
                from src.models.ensemble import EnsemblePredictor
                model = EnsemblePredictor.load(model_path)
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                raise
    
    logger.info(f"Successfully loaded model: {model.__class__.__name__}")
    
    # Use dataset and feature type from model if not provided
    dataset_name = dataset_name or getattr(model, "dataset_name", None)
    feature_type = feature_type or getattr(model, "feature_type", None)
    
    if dataset_name is None or feature_type is None:
        raise ValueError("Dataset name and feature type must be provided if the model doesn't have them")
    
    # Load data
    try:
        df = load_feature_data(dataset_name, feature_type)
        logger.info(f"Loaded dataset: {dataset_name}, feature type: {feature_type}, shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
    # Process data
    X, y = model.process_data(df, target_col=target_col)
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Get feature names
    feature_names = df.drop(columns=[target_col]).columns.tolist() if target_col in df.columns else df.columns.tolist()
    
    # Set up class names based on target type
    if target_col in df.columns:
        class_names = sorted(df[target_col].unique().astype(str).tolist())
    else:
        class_names = ["Class_0", "Class_1", "Class_2"]  # Default
    
    # Initialize explainer
    explainer = ModelExplainer(model, feature_names, class_names)
    
    # Set default methods if not provided
    if methods is None:
        methods = ["shap", "permutation", "lime", "pdp"]
    
    # Generate global explanations
    global_explanations = explainer.generate_global_explanations(X, y, methods=["shap", "permutation"])
    
    # Generate sample-specific explanations
    sample_explanations = []
    
    # If sample indices are provided, use them; otherwise, select random samples
    if sample_indices is None:
        if num_samples > len(X):
            num_samples = len(X)
        sample_indices = np.random.choice(len(X), num_samples, replace=False)
    
    for idx in sample_indices:
        sample_explanation = explainer.explain_prediction(X[idx], methods=methods)
        sample_explanations.append(sample_explanation)
    
    # Compile all explanations
    all_explanations = {
        "model_path": model_path,
        "dataset": dataset_name,
        "feature_type": feature_type,
        "timestamp": datetime.now().isoformat(),
        "global_explanations": global_explanations,
        "sample_explanations": sample_explanations,
        "metadata": {
            "num_features": len(feature_names),
            "num_samples": len(sample_indices),
            "methods": methods,
            "class_names": class_names
        }
    }
    
    # Save all explanations
    save_dir = save_dir or EXPLAINABILITY_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"all_explanations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(save_path, 'w') as f:
        json.dump(all_explanations, f, default=lambda o: str(o) if isinstance(o, (np.ndarray, np.generic)) else o)
    
    logger.info(f"All explanations saved to: {save_path}")
    
    return all_explanations 