"""
Hyperparameter optimization module for soccer prediction models.
Implements advanced optimization strategies for both baseline and advanced models.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

# Import project components
from src.utils.logger import get_logger
from src.models.baseline import BaselineMatchPredictor
from src.models.advanced import AdvancedMatchPredictor
from src.models.training import load_feature_data

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.hyperopt")

# Define paths
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
TUNING_DIR = os.path.join(DATA_DIR, "tuning")
os.makedirs(TUNING_DIR, exist_ok=True)

# Define search spaces for different model types
SEARCH_SPACES = {
    # Baseline models
    "logistic": {
        "C": hp.loguniform("C", np.log(0.001), np.log(100)),
        "class_weight": hp.choice("class_weight", [None, "balanced"]),
        "solver": hp.choice("solver", ["lbfgs", "newton-cg", "sag"]),
        "max_iter": scope.int(hp.quniform("max_iter", 100, 2000, 100))
    },
    "random_forest": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 50)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 30, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 10, 1)),
        "class_weight": hp.choice("class_weight", [None, "balanced"]),
        "max_features": hp.choice("max_features", ["sqrt", "log2", None])
    },
    "xgboost": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 50)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "gamma": hp.uniform("gamma", 0, 5),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.0001), np.log(1)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.0001), np.log(1)),
        "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 10, 1))
    },
    
    # Advanced models
    "neural_network": {
        "units": hp.choice("units", [
            [64, 32],
            [128, 64],
            [256, 128],
            [128, 64, 32],
            [256, 128, 64]
        ]),
        "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.5),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.01)),
        "batch_size": scope.int(hp.quniform("batch_size", 16, 128, 16)),
        "activation": hp.choice("activation", ["relu", "elu", "selu"]),
        "optimizer": hp.choice("optimizer", ["adam", "rmsprop"]),
        "l2_reg": hp.loguniform("l2_reg", np.log(0.0001), np.log(0.1))
    },
    "lightgbm": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 50)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "num_leaves": scope.int(hp.quniform("num_leaves", 10, 150, 1)),
        "min_child_samples": scope.int(hp.quniform("min_child_samples", 5, 100, 5)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(0.0001), np.log(1)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.0001), np.log(1))
    },
    "catboost": {
        "iterations": scope.int(hp.quniform("iterations", 50, 1000, 50)),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "depth": scope.int(hp.quniform("depth", 3, 10, 1)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(0.1), np.log(10)),
        "border_count": scope.int(hp.quniform("border_count", 32, 255, 32)),
        "bagging_temperature": hp.uniform("bagging_temperature", 0, 1)
    }
}

def objective_baseline(params: Dict[str, Any], model_type: str, X_train: np.ndarray, 
                       y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                       random_state: int = 42) -> Dict[str, Any]:
    """
    Objective function for optimizing baseline models with hyperopt.
    
    Args:
        params: Hyperparameters to evaluate
        model_type: Type of baseline model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed for reproducibility
        
    Returns:
        Dict containing loss value and status
    """
    try:
        # Create and configure the model
        model = BaselineMatchPredictor(model_type=model_type)
        
        # Need to set parameters
        model._create_model()
        
        # Set parameters
        for param, value in params.items():
            setattr(model.model, param, value)
        
        # Train the model
        model.train(X_train, y_train)
        
        # Get predictions on validation set
        y_pred = model.predict(X_val)
        
        # Calculate loss (negative F1 score, since we want to maximize F1)
        f1 = f1_score(y_val, y_pred, average='weighted')
        val_loss = -f1  # Negative because hyperopt minimizes
        
        # Log progress
        logger.info(f"Baseline {model_type} - Params: {params} - Validation F1: {f1:.4f}")
        
        return {
            'loss': val_loss,
            'status': STATUS_OK,
            'model_type': model_type,
            'params': params,
            'metrics': {
                'f1_score': f1,
                'accuracy': accuracy_score(y_val, y_pred)
            }
        }
    
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        # Return a large loss value when an error occurs
        return {
            'loss': 999,
            'status': STATUS_OK,
            'error': str(e)
        }

def objective_advanced(params: Dict[str, Any], model_type: str, X_train: np.ndarray, 
                       y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                       random_state: int = 42) -> Dict[str, Any]:
    """
    Objective function for optimizing advanced models with hyperopt.
    
    Args:
        params: Hyperparameters to evaluate
        model_type: Type of advanced model to train
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        random_state: Random seed for reproducibility
        
    Returns:
        Dict containing loss value and status
    """
    try:
        # Handle special parameters based on model type
        model_params = params.copy()
        
        # For neural network, prepare special parameters
        if model_type == "neural_network":
            # Extract optimizer parameter if present, and remove from model_params
            optimizer_name = model_params.pop("optimizer", "adam")
            if optimizer_name == "adam":
                learning_rate = model_params.get("learning_rate", 0.001)
                model_params["optimizer"] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer_name == "rmsprop":
                learning_rate = model_params.get("learning_rate", 0.001)
                model_params["optimizer"] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            
            # Add regularizer if l2_reg is specified
            if "l2_reg" in model_params:
                l2_reg = model_params.pop("l2_reg")
                model_params["kernel_regularizer"] = tf.keras.regularizers.l2(l2_reg)
        
        # Create and train the model
        model = AdvancedMatchPredictor(model_type=model_type, model_params=model_params)
        
        # Determine input dimensions for model creation
        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        
        # Initialize model architecture
        model._create_model(input_dim=input_dim, num_classes=num_classes)
        
        # Train the model
        validation_data = (X_val, y_val)
        model.train(X_train, y_train, validation_data=validation_data)
        
        # Get predictions on validation set
        y_pred = model.predict(X_val)
        
        # Calculate loss (negative F1 score, since we want to maximize F1)
        f1 = f1_score(y_val, y_pred, average='weighted')
        val_loss = -f1  # Negative because hyperopt minimizes
        
        # Calculate additional metrics
        accuracy = accuracy_score(y_val, y_pred)
        
        # For neural networks and other probabilistic models, also calculate log loss
        probas = model.predict_proba(X_val)
        log_loss_value = log_loss(y_val, probas) if probas is not None else None
        
        # Log progress
        logger.info(f"Advanced {model_type} - Validation F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'loss': val_loss,
            'status': STATUS_OK,
            'model_type': model_type,
            'params': params,
            'metrics': {
                'f1_score': f1,
                'accuracy': accuracy,
                'log_loss': log_loss_value
            }
        }
        
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        # Return a large loss value when an error occurs
        return {
            'loss': 999,
            'status': STATUS_OK,
            'error': str(e)
        }

def optimize_hyperparameters(
    model_type: str,
    dataset_name: str = "transfermarkt",
    feature_type: str = "match_features",
    target_col: str = "result",
    test_size: float = 0.2,
    val_size: float = 0.25,  # Percentage of training data to use for validation
    max_evals: int = 50,
    random_state: int = 42,
    search_space: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a given model using Bayesian optimization with hyperopt.
    
    Args:
        model_type: Type of model to optimize
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        target_col: Name of the target column
        test_size: Portion of data to use for testing
        val_size: Portion of training data to use for validation
        max_evals: Maximum number of evaluations to perform
        random_state: Random seed for reproducibility
        search_space: Custom search space to use (optional)
        
    Returns:
        Dict containing optimization results
    """
    # Record start time
    start_time = datetime.now()
    
    # Check if model type is supported
    baseline_models = ["logistic", "random_forest", "xgboost"]
    advanced_models = ["neural_network", "lightgbm", "catboost"]
    
    is_baseline = model_type in baseline_models
    is_advanced = model_type in advanced_models
    
    if not (is_baseline or is_advanced):
        raise ValueError(f"Unsupported model type for hyperparameter optimization: {model_type}")
    
    # Load data
    try:
        df = load_feature_data(dataset_name, feature_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading feature data: {e}")
        raise
    
    # Prepare model for data processing
    if is_baseline:
        model = BaselineMatchPredictor(model_type=model_type, dataset_name=dataset_name, feature_type=feature_type)
    else:
        model = AdvancedMatchPredictor(model_type=model_type, dataset_name=dataset_name, feature_type=feature_type)
    
    # Load pipeline for data processing
    if not model.load_pipeline():
        raise ValueError("Failed to load feature pipeline")
    
    # Process data
    X, y = model.process_data(df, target_col=target_col)
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Split data into train and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Further split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state, stratify=y_train_full
    )
    
    # Get search space
    if search_space is None:
        if model_type not in SEARCH_SPACES:
            raise ValueError(f"No default search space defined for model type: {model_type}")
        search_space = SEARCH_SPACES[model_type]
    
    # Initialize trials object to store optimization results
    trials = Trials()
    
    # Initialize best params and scores
    best_params = None
    best_score = -float('inf')  # Start with worst possible score (we'll maximize)
    
    # Define objective function based on model type
    if is_baseline:
        objective = lambda params: objective_baseline(
            params, model_type, X_train, y_train, X_val, y_val, random_state
        )
    else:
        objective = lambda params: objective_advanced(
            params, model_type, X_train, y_train, X_val, y_val, random_state
        )
    
    # Run hyperparameter optimization
    logger.info(f"Starting hyperparameter optimization for {model_type} model")
    logger.info(f"Search space: {search_space}")
    logger.info(f"Maximum evaluations: {max_evals}")
    
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.RandomState(random_state)
    )
    
    # Get results
    results = [t['result'] for t in trials.trials if 'result' in t]
    valid_results = [r for r in results if 'metrics' in r]
    
    if valid_results:
        # Find the best parameters
        best_result = min(valid_results, key=lambda x: x['loss'])
        best_params = best_result['params']
        best_metrics = best_result['metrics']
        best_score = best_metrics['f1_score']
        
        # Train final model with best parameters
        if is_baseline:
            final_model = BaselineMatchPredictor(model_type=model_type, dataset_name=dataset_name, feature_type=feature_type)
            final_model._create_model()
            
            # Set parameters
            for param, value in best_params.items():
                setattr(final_model.model, param, value)
            
            # Train on full training set
            final_model.train(X_train_full, y_train_full)
        else:
            model_params = best_params.copy()
            
            # Handle special parameters for neural networks
            if model_type == "neural_network":
                if "optimizer" in model_params:
                    optimizer_name = model_params.pop("optimizer")
                    learning_rate = model_params.get("learning_rate", 0.001)
                    
                    if optimizer_name == "adam":
                        model_params["optimizer"] = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    elif optimizer_name == "rmsprop":
                        model_params["optimizer"] = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
                
                if "l2_reg" in model_params:
                    l2_reg = model_params.pop("l2_reg")
                    model_params["kernel_regularizer"] = tf.keras.regularizers.l2(l2_reg)
            
            final_model = AdvancedMatchPredictor(
                model_type=model_type, 
                dataset_name=dataset_name, 
                feature_type=feature_type,
                model_params=model_params
            )
            
            # Initialize model architecture
            input_dim = X_train_full.shape[1]
            num_classes = len(np.unique(y_train_full))
            final_model._create_model(input_dim=input_dim, num_classes=num_classes)
            
            # Train on full training set
            final_model.train(X_train_full, y_train_full)
        
        # Evaluate on test set
        test_eval = final_model.evaluate(X_test, y_test)
        
        # Save model
        model_path = final_model.save()
        logger.info(f"Saved optimized model to {model_path}")
        
        # Calculate optimization time
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Create trial results
        all_trials = []
        for t in trials.trials:
            if 'result' in t and 'metrics' in t['result']:
                trial_data = {
                    'iteration': t['tid'],
                    'params': t['result']['params'],
                    'f1_score': t['result']['metrics']['f1_score'],
                    'accuracy': t['result']['metrics']['accuracy']
                }
                if 'log_loss' in t['result']['metrics'] and t['result']['metrics']['log_loss'] is not None:
                    trial_data['log_loss'] = t['result']['metrics']['log_loss']
                all_trials.append(trial_data)
        
        # Save hyperparameter optimization results
        results_data = {
            'model_type': model_type,
            'dataset_name': dataset_name,
            'feature_type': feature_type,
            'max_evals': max_evals,
            'optimization_time': optimization_time,
            'best_params': best_params,
            'best_validation_metrics': best_metrics,
            'test_metrics': test_eval,
            'model_path': model_path,
            'trials': all_trials,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results to file
        results_path = os.path.join(
            TUNING_DIR,
            f"{dataset_name}_{feature_type}_{model_type}_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            json_results = {}
            for k, v in results_data.items():
                if isinstance(v, dict):
                    # Handle nested dictionaries
                    json_results[k] = {
                        inner_k: (inner_v.item() if hasattr(inner_v, "item") else inner_v)
                        for inner_k, inner_v in v.items()
                    }
                else:
                    json_results[k] = v.item() if hasattr(v, "item") else v
            
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Saved hyperparameter optimization results to {results_path}")
        
        # Return results
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'test_metrics': test_eval,
            'model_path': model_path,
            'results_path': results_path,
            'optimization_time': optimization_time
        }
    
    else:
        logger.error("No valid trials completed during hyperparameter optimization")
        return {
            'error': "No valid trials completed",
            'optimization_time': (datetime.now() - start_time).total_seconds()
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for soccer prediction models")
    parser.add_argument("model_type", type=str, help="Type of model to optimize")
    parser.add_argument("--dataset", type=str, default="transfermarkt", help="Dataset to use")
    parser.add_argument("--features", type=str, default="match_features", help="Feature type to use")
    parser.add_argument("--max-evals", type=int, default=50, help="Maximum evaluations to perform")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Run hyperparameter optimization
    results = optimize_hyperparameters(
        model_type=args.model_type,
        dataset_name=args.dataset,
        feature_type=args.features,
        max_evals=args.max_evals,
        random_state=args.random_state
    )
    
    # Print summary of results
    if 'best_params' in results:
        print("\nBest parameters found:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        print("\nValidation metrics:")
        for metric, value in results['best_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        print("\nTest metrics:")
        for metric, value in results['test_metrics'].items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        print(f"\nOptimized model saved to: {results['model_path']}")
        print(f"Detailed results saved to: {results['results_path']}")
        print(f"Optimization time: {results['optimization_time']:.2f} seconds")
    else:
        print(f"Error during optimization: {results.get('error', 'Unknown error')}") 