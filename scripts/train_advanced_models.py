#!/usr/bin/env python
"""
Script to train and evaluate advanced models.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root directory to the Python path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)

# Import project components
from src.utils.logger import setup_logger, get_logger
from src.models.advanced import AdvancedMatchPredictor, train_advanced_model
from src.models.training import load_feature_data
from src.models.evaluation import evaluate_model_performance, compare_models

# Setup logging
logger = get_logger("scripts.train_advanced_models")

# Default parameters for each model type
DEFAULT_PARAMS = {
    "neural_network": {
        "units": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "activation": "relu",
        "epochs": 100,
        "batch_size": 32,
        "verbose": 1
    },
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 31,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1
    },
    "catboost": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "verbose": False
    },
    "deep_ensemble": {
        "ensemble_size": 5,
        "units": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "verbose": 0
    },
    "time_series": {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
        "seasonality_mode": "additive"
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate advanced models")
    
    # Model selection
    parser.add_argument("--model-type", choices=["neural_network", "lightgbm", "catboost", 
                                               "deep_ensemble", "time_series", "all"],
                       default="lightgbm", help="Type of advanced model to train")
    
    # Data options
    parser.add_argument("--dataset", default="transfermarkt",
                       help="Dataset to use for training")
    parser.add_argument("--feature-type", default="match_features",
                       help="Feature set to use")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Portion of data to use for testing")
    parser.add_argument("--validation-size", type=float, default=0.1,
                       help="Portion of training data to use for validation")
    
    # Training options
    parser.add_argument("--params", type=str, default=None,
                       help="JSON string or file path with model parameters")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation after training")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with baseline models")
    
    # Output options
    parser.add_argument("--output-dir", default=None,
                       help="Directory to save models and evaluation results")
    
    return parser.parse_args()


def load_model_params(params_input, model_type):
    """
    Load model parameters from a JSON string or file, or use defaults.
    
    Args:
        params_input: JSON string or file path
        model_type: Type of model to get parameters for
        
    Returns:
        dict: Model parameters
    """
    # Start with default parameters
    params = DEFAULT_PARAMS.get(model_type, {}).copy()
    
    if params_input:
        # Check if input is a file path
        if os.path.exists(params_input):
            with open(params_input, 'r') as f:
                input_params = json.load(f)
        else:
            # Try to parse as JSON string
            try:
                input_params = json.loads(params_input)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse params: {params_input}")
                return params
        
        # Get model specific parameters if present
        if model_type in input_params:
            model_params = input_params[model_type]
        else:
            model_params = input_params
        
        # Update default parameters with input
        params.update(model_params)
    
    return params


def train_single_model(model_type, dataset_name, feature_type, test_size, validation_size, model_params):
    """
    Train a single advanced model.
    
    Args:
        model_type: Type of model to train
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        test_size: Portion of data to use for testing
        validation_size: Portion of training data to use for validation
        model_params: Model parameters
        
    Returns:
        AdvancedMatchPredictor: Trained model
    """
    logger.info(f"Training {model_type} model on {dataset_name} dataset with {feature_type} features")
    
    # Train the model
    try:
        model = train_advanced_model(
            model_type=model_type,
            dataset_name=dataset_name,
            feature_type=feature_type,
            test_size=test_size,
            validation_size=validation_size,
            model_params=model_params
        )
        
        logger.info(f"Model trained successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


def evaluate_model(model_path, dataset_name=None, feature_type=None):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model
        dataset_name: Name of the dataset to use (if None, use model's dataset)
        feature_type: Type of features to use (if None, use model's features)
        
    Returns:
        dict: Evaluation results
    """
    logger.info(f"Evaluating model: {model_path}")
    
    try:
        evaluation = evaluate_model_performance(
            model_path=model_path,
            dataset_name=dataset_name,
            feature_type=feature_type,
            generate_plots=True
        )
        
        # Print results
        logger.info("Evaluation results:")
        for metric, value in evaluation.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
        
        return evaluation
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None


def main():
    """Main function to train and evaluate advanced models."""
    args = parse_args()
    
    # Determine which models to train
    models_to_train = []
    if args.model_type == "all":
        models_to_train = list(DEFAULT_PARAMS.keys())
    else:
        models_to_train = [args.model_type]
    
    # Train and evaluate each model
    model_paths = []
    for model_type in models_to_train:
        try:
            # Load model parameters
            model_params = load_model_params(args.params, model_type)
            
            # Train the model
            model = train_single_model(
                model_type=model_type,
                dataset_name=args.dataset,
                feature_type=args.feature_type,
                test_size=args.test_size,
                validation_size=args.validation_size,
                model_params=model_params
            )
            
            # Save the model
            model_path = model.save()
            model_paths.append(model_path)
            
            # Print model performance
            if model.model_info and "performance" in model.model_info:
                performance = model.model_info["performance"]
                logger.info(f"Model performance: accuracy={performance.get('accuracy'):.4f}, f1={performance.get('f1'):.4f}")
            
            # Evaluate if requested
            if not args.skip_evaluation:
                evaluate_model(model_path)
        
        except Exception as e:
            logger.error(f"Error processing {model_type} model: {e}")
    
    # Compare with baseline models if requested
    if args.compare and model_paths:
        try:
            # Find baseline models
            baseline_dir = os.path.join(PROJECT_ROOT, "data", "models")
            baseline_models = []
            
            if os.path.exists(baseline_dir):
                for file in os.listdir(baseline_dir):
                    if file.endswith(".pkl") and os.path.isfile(os.path.join(baseline_dir, file)):
                        # Skip advanced models
                        if "advanced" not in file:
                            baseline_models.append(os.path.join(baseline_dir, file))
            
            # If baseline models found, compare
            if baseline_models:
                logger.info(f"Comparing advanced models with baseline models")
                compare_paths = baseline_models + model_paths
                
                comparison = compare_models(
                    model_paths=compare_paths,
                    dataset_name=args.dataset,
                    feature_type=args.feature_type,
                    generate_plots=True
                )
                
                # Print results
                logger.info("\nComparison results:")
                for model_name, metrics in comparison.get("model_metrics", {}).items():
                    logger.info(f"\n{model_name}:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"  {metric}: {value:.4f}")
                        else:
                            logger.info(f"  {metric}: {value}")
                
                # Print best model
                if "best_model" in comparison:
                    logger.info(f"\nBest model: {comparison['best_model']}")
                    logger.info(f"Best accuracy: {comparison['best_accuracy']:.4f}")
            
            else:
                logger.warning("No baseline models found for comparison")
        
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
    
    logger.info("Script completed")


if __name__ == "__main__":
    main() 