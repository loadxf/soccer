"""
Command-line interface for soccer prediction models.
Provides a user-friendly interface to train, evaluate, and use models.
"""

import argparse
import os
import json
import sys
import textwrap
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import click

# Import project components
from src.utils.logger import get_logger
from src.models.baseline import BaselineMatchPredictor, train_baseline_model
from src.models.advanced import AdvancedMatchPredictor, train_advanced_model
from src.models.ensemble import EnsemblePredictor, train_ensemble_model
from src.models.training import train_model, train_multiple_models
from src.models.evaluation import evaluate_model_performance, compare_models, analyze_feature_importance
from src.models.hyperopt import optimize_hyperparameters
from src.models.explainability import ModelExplainer, generate_model_explanations
from src.models.time_series import TimeSeriesPredictor
from src.models.player_performance import (
    PlayerPerformanceModel, 
    PlayerPerformancePredictor,
    train_player_performance_models,
    get_player_predictions,
    PERFORMANCE_METRICS
)

# Setup logger
logger = get_logger("models.cli")

# Define model types
BASELINE_MODELS = ["logistic", "random_forest", "xgboost"]
ADVANCED_MODELS = ["neural_network", "lightgbm", "catboost", "deep_ensemble", "time_series"]
ENSEMBLE_TYPES = ["voting", "stacking", "blending", "calibrated_voting", "time_weighted", "performance_weighted"]
TIMESERIES_MODELS = ["arima", "prophet", "lstm", "gru", "encoder_decoder"]
ALL_MODEL_TYPES = BASELINE_MODELS + ADVANCED_MODELS


def parse_args():
    """Parse command line arguments for model CLI."""
    parser = argparse.ArgumentParser(description="Soccer Prediction Model CLI")
    
    # General options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model-type", choices=ALL_MODEL_TYPES + ["all"],
                              default="logistic", help="Type of model to train")
    train_parser.add_argument("--dataset", default="transfermarkt",
                              help="Dataset to use for training")
    train_parser.add_argument("--features", default="match_features",
                              help="Feature set to use")
    train_parser.add_argument("--test-size", type=float, default=0.2,
                              help="Portion of data to use for testing")
    train_parser.add_argument("--tuning", action="store_true",
                              help="Perform hyperparameter tuning")
    train_parser.add_argument("--cv-folds", type=int, default=5,
                              help="Number of cross-validation folds")
    train_parser.add_argument("--validation-size", type=float, default=0.1,
                              help="Portion of training data to use for validation (for advanced models)")
    train_parser.add_argument("--output-dir", default=None,
                              help="Directory to save models")
    train_parser.add_argument("--params", type=str, default=None,
                              help="JSON string or file path with model parameters")
    
    # Time series command (new)
    timeseries_parser = subparsers.add_parser("timeseries", help="Train a time series model")
    timeseries_parser.add_argument("--model-type", choices=TIMESERIES_MODELS,
                              default="lstm", help="Type of time series model to train")
    timeseries_parser.add_argument("--dataset", default="transfermarkt",
                              help="Dataset to use for training")
    timeseries_parser.add_argument("--features", default="match_features",
                              help="Feature set to use")
    timeseries_parser.add_argument("--target-col", default="result",
                              help="Target column to predict (default: result)")
    timeseries_parser.add_argument("--test-size", type=float, default=0.2,
                              help="Portion of data to use for testing")
    timeseries_parser.add_argument("--validation-split", type=float, default=0.1,
                              help="Portion of training data to use for validation")
    timeseries_parser.add_argument("--look-back", type=int, default=5,
                              help="Number of previous time steps to consider")
    timeseries_parser.add_argument("--forecast-horizon", type=int, default=1,
                              help="Number of steps ahead to forecast")
    timeseries_parser.add_argument("--task-type", choices=["regression", "classification"],
                              default="classification", help="Type of prediction task")
    timeseries_parser.add_argument("--output-dir", default=None,
                              help="Directory to save models")
    timeseries_parser.add_argument("--params", type=str, default=None,
                              help="JSON string or file path with model parameters")
    timeseries_parser.add_argument("--plot", action="store_true",
                              help="Generate plots after training")
    
    # Ensemble command
    ensemble_parser = subparsers.add_parser("ensemble", help="Train an ensemble model")
    ensemble_parser.add_argument("--base-models", nargs="+", 
                                 choices=BASELINE_MODELS + ADVANCED_MODELS,
                                 default=["logistic", "random_forest", "xgboost"],
                                 help="Base models to use in ensemble")
    ensemble_parser.add_argument("--ensemble-type", choices=["voting", "stacking", "blending"],
                                 default="voting", help="Type of ensemble to create")
    ensemble_parser.add_argument("--dataset", default="transfermarkt",
                                 help="Dataset to use for training")
    ensemble_parser.add_argument("--features", default="match_features",
                                 help="Feature set to use")
    ensemble_parser.add_argument("--test-size", type=float, default=0.2,
                                 help="Portion of data to use for testing")
    ensemble_parser.add_argument("--validation-size", type=float, default=0.1,
                                 help="Portion of training data to use for validation")
    ensemble_parser.add_argument("--output-dir", default=None,
                                 help="Directory to save ensemble")
    ensemble_parser.add_argument("--existing-models", nargs="+", default=None,
                                 help="Paths to existing model files to include in ensemble")
    
    # Hyperopt command (new)
    hyperopt_parser = subparsers.add_parser("hyperopt", help="Run hyperparameter optimization")
    hyperopt_parser.add_argument("--model-type", required=True, choices=ALL_MODEL_TYPES,
                               help="Type of model to optimize")
    hyperopt_parser.add_argument("--dataset", default="transfermarkt",
                               help="Dataset to use for optimization")
    hyperopt_parser.add_argument("--features", default="match_features",
                               help="Feature set to use")
    hyperopt_parser.add_argument("--max-evals", type=int, default=50,
                               help="Maximum evaluations to perform")
    hyperopt_parser.add_argument("--test-size", type=float, default=0.2,
                               help="Portion of data to use for testing")
    hyperopt_parser.add_argument("--val-size", type=float, default=0.25,
                               help="Portion of training data to use for validation")
    hyperopt_parser.add_argument("--output-dir", default=None,
                               help="Directory to save results")
    hyperopt_parser.add_argument("--custom-space", type=str, default=None,
                               help="JSON string or file path with custom search space")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    evaluate_parser.add_argument("--model-path", required=True,
                                help="Path to the trained model file")
    evaluate_parser.add_argument("--dataset", default="transfermarkt",
                                help="Dataset to use for evaluation")
    evaluate_parser.add_argument("--features", default="match_features",
                                help="Feature set to use")
    evaluate_parser.add_argument("--plots", action="store_true",
                                help="Generate evaluation plots")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--model-paths", nargs="+", required=True,
                               help="Paths to the trained model files to compare")
    compare_parser.add_argument("--dataset", default="transfermarkt",
                               help="Dataset to use for comparison")
    compare_parser.add_argument("--features", default="match_features",
                               help="Feature set to use")
    compare_parser.add_argument("--plots", action="store_true",
                               help="Generate comparison plots")
    
    # Importance command
    importance_parser = subparsers.add_parser("importance", help="Analyze feature importance")
    importance_parser.add_argument("--model-path", required=True,
                                  help="Path to the trained model file")
    importance_parser.add_argument("--n-top", type=int, default=20,
                                  help="Number of top features to show")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions with a trained model")
    predict_parser.add_argument("--model-path", required=True,
                              help="Path to the trained model file")
    predict_parser.add_argument("--home-team", required=True,
                              help="ID or name of the home team")
    predict_parser.add_argument("--away-team", required=True,
                              help="ID or name of the away team")
    predict_parser.add_argument("--date", default=None,
                              help="Date of the match (YYYY-MM-DD)")
    predict_parser.add_argument("--features", default=None,
                              help="JSON string or file path with additional features")
    
    # Time series predict command (new)
    ts_predict_parser = subparsers.add_parser("ts-predict", help="Make time series predictions")
    ts_predict_parser.add_argument("--model-path", required=True,
                                help="Path to the time series model file")
    ts_predict_parser.add_argument("--home-team", required=True,
                                help="ID or name of the home team")
    ts_predict_parser.add_argument("--away-team", required=True,
                                help="ID or name of the away team")
    ts_predict_parser.add_argument("--date", default=None,
                                help="Date of the match (YYYY-MM-DD)")
    ts_predict_parser.add_argument("--features", default=None,
                                help="JSON string or file path with additional features")
    ts_predict_parser.add_argument("--horizon", type=int, default=1,
                                help="Number of steps ahead to forecast")
    ts_predict_parser.add_argument("--plot", action="store_true",
                                help="Generate prediction plots")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--model-type", choices=["baseline", "advanced", "ensemble", "timeseries", "all"],
                            default="all", help="Type of models to list")
    list_parser.add_argument("--dataset", default=None,
                            help="Filter by dataset")
    list_parser.add_argument("--features", default=None,
                            help="Filter by feature type")
    
    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Generate model explanations")
    explain_parser.add_argument("--model", "-m", required=True,
                        help="Path to the model file")
    explain_parser.add_argument("--dataset", "-d", default=None,
                        help="Name of the dataset to use (if not specified, uses the one from the model)")
    explain_parser.add_argument("--feature-type", "-f", default=None,
                        help="Type of features to use (if not specified, uses the one from the model)")
    explain_parser.add_argument("--target-col", "-t", default="result",
                        help="Name of the target column (default: result)")
    explain_parser.add_argument("--methods", "-mt", nargs="+", 
                        default=["shap", "lime", "permutation", "pdp"],
                        help="Explanation methods to use (default: shap, lime, permutation, pdp)")
    explain_parser.add_argument("--samples", "-s", type=int, default=5,
                        help="Number of samples to explain (default: 5)")
    explain_parser.add_argument("--sample-indices", "-i", nargs="+", type=int,
                        help="Specific sample indices to explain (if provided, --samples is ignored)")
    explain_parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to save explanations (if not specified, uses default)")
    explain_parser.add_argument("--pdp-features", "-p", nargs="+", type=int,
                        help="Feature indices to generate PDP plots for (default: top 5 important features)")
    
    return parser.parse_args()


def load_model_params(params_input: str) -> Dict[str, Any]:
    """
    Load model parameters from a JSON string or file.
    
    Args:
        params_input: JSON string or file path
        
    Returns:
        Dict[str, Any]: Model parameters
    """
    if not params_input:
        return {}
    
    # Check if input is a file path
    if os.path.exists(params_input):
        with open(params_input, 'r') as f:
            return json.load(f)
    
    # Otherwise, try to parse as JSON string
    try:
        return json.loads(params_input)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse params: {params_input}")
        return {}


def train_cmd(args):
    """Handle the train command."""
    # Load model parameters
    model_params = load_model_params(args.params)
    
    # Determine output directory
    output_dir = args.output_dir
    
    if args.model_type == "all":
        # Train all model types
        logger.info("Training all model types")
        results = {}
        
        # Train baseline models
        for model_type in BASELINE_MODELS:
            try:
                logger.info(f"Training baseline model: {model_type}")
                model = train_baseline_model(
                    model_type=model_type,
                    dataset_name=args.dataset,
                    feature_type=args.features,
                    test_size=args.test_size,
                    random_state=42
                )
                
                # Save results
                results[model_type] = {
                    "model_path": model.save(output_dir),
                    "performance": model.model_info.get("performance", {})
                }
                
                logger.info(f"Trained and saved {model_type} model")
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
        
        # Train advanced models
        for model_type in ADVANCED_MODELS:
            try:
                logger.info(f"Training advanced model: {model_type}")
                model_specific_params = model_params.get(model_type, {})
                
                model = train_advanced_model(
                    model_type=model_type,
                    dataset_name=args.dataset,
                    feature_type=args.features,
                    test_size=args.test_size,
                    validation_size=args.validation_size,
                    random_state=42,
                    model_params=model_specific_params
                )
                
                # Save results
                results[model_type] = {
                    "model_path": model.save(output_dir),
                    "performance": model.model_info.get("performance", {})
                }
                
                logger.info(f"Trained and saved {model_type} model")
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
        
        # Print summary
        logger.info("\nTraining summary:")
        for model_type, result in results.items():
            if "performance" in result and "accuracy" in result["performance"]:
                accuracy = result["performance"]["accuracy"]
                f1 = result["performance"].get("f1", "N/A")
                logger.info(f"  {model_type}: accuracy={accuracy:.4f}, f1={f1}")
            else:
                logger.info(f"  {model_type}: training completed, but no performance metrics available")
    
    elif args.model_type in BASELINE_MODELS:
        # Train a single baseline model
        logger.info(f"Training baseline model: {args.model_type}")
        
        # Handle hyperparameter tuning
        hyperparameter_tuning = args.tuning
        cv_folds = args.cv_folds if args.tuning else None
        
        if hyperparameter_tuning:
            # Use the training module with hyperparameter tuning
            from src.models.training import train_model
            
            result = train_model(
                model_type=args.model_type,
                dataset_name=args.dataset,
                feature_type=args.features,
                test_size=args.test_size,
                cv_folds=cv_folds,
                hyperparameter_tuning=True,
                random_state=42
            )
            
            logger.info(f"Model saved to: {result.get('model_path')}")
            if "best_params" in result:
                logger.info(f"Best parameters: {result['best_params']}")
            if "performance" in result:
                logger.info(f"Performance: {result['performance']}")
        else:
            # Use the simpler baseline training function
            model = train_baseline_model(
                model_type=args.model_type,
                dataset_name=args.dataset,
                feature_type=args.features,
                test_size=args.test_size,
                random_state=42
            )
            
            model_path = model.save(output_dir)
            logger.info(f"Model saved to: {model_path}")
            
            if model.model_info and "performance" in model.model_info:
                performance = model.model_info["performance"]
                logger.info(f"Performance: accuracy={performance.get('accuracy'):.4f}, f1={performance.get('f1'):.4f}")
    
    elif args.model_type in ADVANCED_MODELS:
        # Train a single advanced model
        logger.info(f"Training advanced model: {args.model_type}")
        
        # Get model specific parameters
        model_specific_params = model_params.get(args.model_type, {})
        
        # Train the model
        model = train_advanced_model(
            model_type=args.model_type,
            dataset_name=args.dataset,
            feature_type=args.features,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=42,
            model_params=model_specific_params
        )
        
        # Save the model
        model_path = model.save(output_dir)
        logger.info(f"Model saved to: {model_path}")
        
        # Print performance
        if model.model_info and "performance" in model.model_info:
            performance = model.model_info["performance"]
            logger.info(f"Performance: accuracy={performance.get('accuracy'):.4f}, f1={performance.get('f1'):.4f}")
    
    else:
        logger.error(f"Unknown model type: {args.model_type}")


def hyperopt_cmd(args):
    """Handle the hyperopt command."""
    logger.info(f"Running hyperparameter optimization for {args.model_type} model")
    
    # Load custom search space if provided
    custom_space = None
    if args.custom_space:
        custom_space = load_model_params(args.custom_space)
        if custom_space:
            logger.info(f"Using custom search space: {json.dumps(custom_space, indent=2)}")
    
    try:
        # Run hyperparameter optimization
        results = optimize_hyperparameters(
            model_type=args.model_type,
            dataset_name=args.dataset,
            feature_type=args.features,
            test_size=args.test_size,
            val_size=args.val_size,
            max_evals=args.max_evals,
            random_state=42,
            search_space=custom_space
        )
        
        # Check for error
        if 'error' in results:
            logger.error(f"Hyperparameter optimization failed: {results['error']}")
            return
        
        # Print results
        logger.info("\nHyperparameter Optimization Results:")
        
        # Print best parameters
        logger.info("\nBest parameters:")
        for param, value in results['best_params'].items():
            logger.info(f"  {param}: {value}")
        
        # Print metrics
        logger.info("\nValidation metrics:")
        for metric, value in results['best_metrics'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
        
        logger.info("\nTest metrics:")
        for metric, value in results['test_metrics'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
        
        # Print model path
        logger.info(f"\nOptimized model saved to: {results['model_path']}")
        logger.info(f"Detailed results saved to: {results['results_path']}")
        logger.info(f"Optimization time: {results['optimization_time']:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {e}")


def evaluate_cmd(args):
    """Handle the evaluate command."""
    logger.info(f"Evaluating model: {args.model_path}")
    
    try:
        # Determine if it's a baseline or advanced model
        is_advanced = "advanced" in args.model_path
        
        # Load the model
        if is_advanced:
            model = AdvancedMatchPredictor.load(args.model_path)
        else:
            model = BaselineMatchPredictor.load(args.model_path)
        
        # Run evaluation
        evaluation = evaluate_model_performance(
            model_path=args.model_path,
            dataset_name=args.dataset,
            feature_type=args.features,
            generate_plots=args.plots
        )
        
        # Print results
        logger.info("Evaluation results:")
        for metric, value in evaluation.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
        
        # Print path to any generated plots
        if args.plots and "plot_paths" in evaluation:
            logger.info("\nGenerated plots:")
            for plot_name, path in evaluation.get("plot_paths", {}).items():
                logger.info(f"  {plot_name}: {path}")
        
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")


def compare_cmd(args):
    """Handle the compare command."""
    logger.info(f"Comparing models: {args.model_paths}")
    
    try:
        # Run comparison
        comparison = compare_models(
            model_paths=args.model_paths,
            dataset_name=args.dataset,
            feature_type=args.features,
            generate_plots=args.plots
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
        
        # Print path to any generated plots
        if args.plots and "plot_paths" in comparison:
            logger.info("\nGenerated plots:")
            for plot_name, path in comparison.get("plot_paths", {}).items():
                logger.info(f"  {plot_name}: {path}")
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")


def importance_cmd(args):
    """Handle the feature importance command."""
    logger.info(f"Analyzing feature importance for model: {args.model_path}")
    
    try:
        # Check if this is a model type that supports feature importance
        is_advanced = "advanced" in args.model_path
        model_type = "advanced"
        
        if is_advanced:
            # Load model to check type
            model = AdvancedMatchPredictor.load(args.model_path)
            model_type = model.model_type
        else:
            # Load model to check type
            model = BaselineMatchPredictor.load(args.model_path)
            model_type = model.model_type
        
        # Check if model supports feature importance
        supported_models = ["random_forest", "xgboost", "lightgbm", "catboost"]
        if model_type not in supported_models:
            logger.warning(f"Feature importance not supported for model type: {model_type}")
            logger.warning(f"Supported model types: {supported_models}")
            return
        
        # Run feature importance analysis
        importance = analyze_feature_importance(
            model_path=args.model_path,
            n_top_features=args.n_top
        )
        
        # Print results
        logger.info("\nFeature importance:")
        for feature, value in importance.get("feature_importance", {}).items():
            logger.info(f"  {feature}: {value:.4f}")
        
        # Print path to any generated plots
        if "plot_path" in importance:
            logger.info(f"\nFeature importance plot: {importance['plot_path']}")
        
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")


def predict_cmd(args):
    """Handle the predict command."""
    logger.info(f"Making prediction with model: {args.model_path}")
    
    try:
        # Load additional features if provided
        features = None
        if args.features:
            features = load_model_params(args.features)
        
        # Check if it's a baseline or advanced model
        is_advanced = "advanced" in args.model_path
        
        # Load the model and make prediction
        if is_advanced:
            model = AdvancedMatchPredictor.load(args.model_path)
        else:
            model = BaselineMatchPredictor.load(args.model_path)
        
        # Make prediction
        prediction = model.predict_match(
            home_team_id=args.home_team,
            away_team_id=args.away_team,
            features=features
        )
        
        # Print results
        logger.info("\nPrediction:")
        logger.info(f"  Home Team: {args.home_team}")
        logger.info(f"  Away Team: {args.away_team}")
        logger.info(f"  Predicted Outcome: {prediction.get('prediction')}")
        logger.info(f"  Confidence: {prediction.get('confidence', 0.0):.4f}")
        
        if "probabilities" in prediction:
            logger.info("\nProbabilities:")
            for outcome, prob in prediction["probabilities"].items():
                logger.info(f"  {outcome}: {prob:.4f}")
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")


def list_cmd(args):
    """Handle the list command."""
    logger.info("Listing available models")
    
    from config.default_config import DATA_DIR
    
    # Define model directories
    models_dir = os.path.join(DATA_DIR, "models")
    baseline_dir = os.path.join(models_dir, "baseline")
    advanced_dir = os.path.join(models_dir, "advanced")
    ensemble_dir = os.path.join(models_dir, "ensemble")
    
    model_type = args.model_type if hasattr(args, 'model_type') else args.type
    
    # Get list of models
    baseline_models = []
    advanced_models = []
    ensemble_models = []
    
    if model_type in ["baseline", "all"] and os.path.exists(baseline_dir):
        baseline_models = [f for f in os.listdir(baseline_dir) if f.endswith((".joblib", ".pkl"))]
    
    if model_type in ["advanced", "all"] and os.path.exists(advanced_dir):
        advanced_models = [f for f in os.listdir(advanced_dir) if f.endswith((".joblib", ".pkl", ".h5"))]
    
    if model_type in ["ensemble", "all"] and os.path.exists(ensemble_dir):
        ensemble_models = [f for f in os.listdir(ensemble_dir) if f.endswith((".joblib", ".pkl"))]
    
    # Print results
    print("\nAvailable Models:")
    
    if model_type in ["baseline", "all"]:
        print("\nBaseline Models:")
        if baseline_models:
            for model in baseline_models:
                print(f"  - {model}")
        else:
            print("  No baseline models found")
    
    if model_type in ["advanced", "all"]:
        print("\nAdvanced Models:")
        if advanced_models:
            for model in advanced_models:
                print(f"  - {model}")
        else:
            print("  No advanced models found")
    
    if model_type in ["ensemble", "all"]:
        print("\nEnsemble Models:")
        if ensemble_models:
            for model in ensemble_models:
                print(f"  - {model}")
        else:
            print("  No ensemble models found")
    
    # Save to file if output path provided
    if args.output:
        model_list = {
            "baseline": baseline_models,
            "advanced": advanced_models,
            "ensemble": ensemble_models
        }
        
        with open(args.output, 'w') as f:
            json.dump(model_list, f, indent=2)
        
        logger.info(f"Model list saved to {args.output}")


def ensemble_cmd(args):
    """Handle the ensemble command."""
    logger.info(f"Training {args.ensemble_type} ensemble with models: {args.base_models}")
    
    # Parse model parameters if provided
    model_params = None
    if args.params:
        model_params = load_model_params(args.params)
    
    # Train ensemble model
    try:
        ensemble = train_ensemble_model(
            ensemble_type=args.ensemble_type,
            model_types=args.base_models,
            dataset_name=args.dataset,
            feature_type=args.features,
            test_size=args.test_size,
            validation_size=args.validation_size,
            random_state=42,
            model_params=model_params
        )
        
        # If save path provided, save the model
        if args.output_dir:
            ensemble.save(args.output_dir)
            logger.info(f"Ensemble model saved to {args.output_dir}")
        
        # Print ensemble performance
        from src.data.features import load_feature_dataset
        
        # Load dataset
        data = load_feature_dataset(args.dataset, args.features)
        if data is None:
            logger.error(f"Could not load dataset {args.dataset} with features {args.features}")
            return
        
        # Evaluate ensemble
        from sklearn.model_selection import train_test_split
        X, y = data['X'], data['y']
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        results = ensemble.evaluate(X_test, y_test)
        
        # Display results
        print("\nEnsemble Model Evaluation:")
        print(f"  - Ensemble Type: {args.ensemble_type}")
        print(f"  - Models: {', '.join(args.base_models)}")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - F1 Score: {results['f1']:.4f}")
        print(f"  - Precision: {results['precision']:.4f}")
        print(f"  - Recall: {results['recall']:.4f}")
        print(f"  - Log Loss: {results['log_loss']:.4f}")
        
        # Show individual model performances
        print("\nIndividual Model Performance:")
        for model_result in results['individual_model_metrics']:
            print(f"  - {model_result['model_type']} (weight: {model_result['weight']:.3f}):")
            print(f"      Accuracy: {model_result['accuracy']:.4f}")
            print(f"      F1 Score: {model_result['f1']:.4f}")
            
    except Exception as e:
        logger.error(f"Error training ensemble model: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


def explain_cmd(args):
    """Handle the explain command."""
    logger.info(f"Generating explanations for model: {args.model}")
    
    try:
        # Generate explanations
        results = generate_model_explanations(
            model_path=args.model,
            dataset_name=args.dataset,
            feature_type=args.feature_type,
            target_col=args.target_col,
            methods=args.methods,
            sample_indices=args.sample_indices,
            num_samples=args.samples,
            save_dir=args.output_dir
        )
        
        # Print summary of explanations
        print(f"\nModel Explanation Summary:")
        print(f"==========================")
        print(f"Model: {args.model}")
        print(f"Dataset: {results.get('dataset', 'Not specified')}")
        print(f"Methods used: {', '.join(args.methods)}")
        print(f"Number of samples explained: {len(results.get('sample_explanations', []))}")
        
        # Print global feature importance summary if available
        if "global_explanations" in results and "explanations" in results["global_explanations"]:
            if "permutation" in results["global_explanations"]["explanations"]:
                perm_results = results["global_explanations"]["explanations"]["permutation"]
                if "importance" in perm_results and isinstance(perm_results["importance"], list) and len(perm_results["importance"]) > 0:
                    top_features = perm_results["importance"][:5]
                    print("\nTop 5 Features (Permutation Importance):")
                    for i, feat in enumerate(top_features):
                        print(f"{i+1}. {feat['Feature']}: {feat['Importance']:.6f}")
        
        # Print final results location
        if "metadata" in results and "save_path" in results["metadata"]:
            print(f"\nAll explanations saved to: {results['metadata']['save_path']}")
        
    except Exception as e:
        logger.error(f"Error generating explanations: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


def timeseries_cmd(args):
    """Handle the time series command."""
    logger.info(f"Training {args.model_type} time series model")
    
    # Load model parameters
    model_params = load_model_params(args.params) or {}
    
    # Add task type to parameters
    model_params['task_type'] = args.task_type
    
    try:
        # Load training data
        from src.data.features import load_feature_data
        
        df = load_feature_data(
            dataset_name=args.dataset,
            feature_type=args.features
        )
        
        if df is None or df.empty:
            logger.error(f"Failed to load dataset {args.dataset} with features {args.features}")
            return 1
        
        # Create and train the time series model
        model = TimeSeriesPredictor(
            model_type=args.model_type,
            dataset_name=args.dataset,
            feature_type=args.features,
            look_back=args.look_back,
            forecast_horizon=args.forecast_horizon,
            model_params=model_params
        )
        
        # Train the model
        results = model.fit(
            df=df,
            target_col=args.target_col,
            test_size=args.test_size,
            validation_split=args.validation_split
        )
        
        # Save the model
        save_path = model.save(args.output_dir)
        logger.info(f"Model saved to: {save_path}")
        
        # Print performance metrics
        logger.info("\nTime Series Model Training Results:")
        logger.info(f"Model Type: {args.model_type}")
        logger.info(f"Look Back: {args.look_back}")
        logger.info(f"Forecast Horizon: {args.forecast_horizon}")
        
        test_metrics = results.get('test_metrics', {})
        for metric, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{metric}: {value:.4f}")
            else:
                logger.info(f"{metric}: {value}")
        
        # Generate plot if requested
        if args.plot:
            from sklearn.model_selection import train_test_split
            
            # Split data for evaluation
            X = df.drop(columns=[args.target_col])
            y = df[args.target_col]
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=42
            )
            
            # Process test data
            X_processed, y_processed = model.process_data(X_test, args.target_col)
            
            # Generate plot
            plot = model.plot_predictions(X_processed, y_processed, savefig=True)
            logger.info(f"Prediction plot saved")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error training time series model: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def ts_predict_cmd(args):
    """Handle the time series prediction command."""
    logger.info(f"Making time series prediction with model: {args.model_path}")
    
    try:
        # Load the model
        model = TimeSeriesPredictor.load(args.model_path)
        
        # Load additional features if provided
        features = None
        if args.features:
            features = load_model_params(args.features)
        
        # Make prediction
        prediction = model.predict_match(
            home_team_id=args.home_team,
            away_team_id=args.away_team,
            date=args.date,
            features=features
        )
        
        # Print results
        logger.info("\nTime Series Prediction:")
        logger.info(f"Model Type: {model.model_type}")
        logger.info(f"Look Back: {model.look_back}")
        logger.info(f"Forecast Horizon: {model.forecast_horizon}")
        logger.info(f"Home Team: {args.home_team}")
        logger.info(f"Away Team: {args.away_team}")
        logger.info(f"Date: {args.date or 'Not specified'}")
        
        if "prediction" in prediction:
            if isinstance(prediction["prediction"], str):
                logger.info(f"Predicted Outcome: {prediction['prediction']}")
            else:
                logger.info(f"Predicted Value: {prediction['prediction']:.4f}")
        
        if "probabilities" in prediction:
            logger.info("\nProbabilities:")
            for outcome, prob in prediction["probabilities"].items():
                logger.info(f"  {outcome}: {prob:.4f}")
                
        if "confidence" in prediction:
            logger.info(f"Confidence: {prediction['confidence']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error making time series prediction: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


@click.group()
def player():
    """Commands for player performance prediction models."""
    pass


@player.command(name="train")
@click.option("--dataset", default="transfermarkt", help="Dataset to use for training")
@click.option("--lookback", default=5, help="Number of previous matches to use for form calculation")
@click.option("--model-type", default="gradient_boosting", 
              type=click.Choice(["random_forest", "gradient_boosting", "xgboost", "lightgbm", "ridge", "lasso", "elastic_net"]), 
              help="Type of model to train")
@click.option("--tune/--no-tune", default=False, help="Whether to perform hyperparameter tuning")
@click.option("--metric", multiple=True, help="Performance metrics to train models for")
def train_player_models(dataset, lookback, model_type, tune, metric):
    """Train player performance prediction models."""
    try:
        # If metrics provided as options, use them; otherwise train all available metrics
        metrics = list(metric) if metric else None
        
        click.echo(f"Training {model_type} models for player performance prediction")
        
        results = train_player_performance_models(
            dataset_name=dataset,
            lookback_window=lookback,
            model_types=[model_type],
            tune_hyperparameters=tune,
            metrics=metrics
        )
        
        # Print results
        click.echo("\n=== Training Results ===")
        for metric, metric_results in results.items():
            model_result = metric_results[model_type]
            click.echo(f"\nMetric: {metric}")
            click.echo(f"Training RMSE: {model_result['rmse']:.4f}")
            click.echo(f"Test RMSE: {model_result['test_rmse']:.4f}")
            click.echo(f"Test R²: {model_result['test_r2']:.4f}")
            click.echo(f"Model saved to: {model_result['model_path']}")
        
    except Exception as e:
        click.echo(f"Error training player models: {e}")
        logger.error(f"Error training player models: {e}", exc_info=True)


@player.command(name="predict")
@click.option("--player-id", required=True, type=int, help="ID of the player")
@click.option("--match-id", required=True, type=int, help="ID of the match")
@click.option("--team-id", required=True, type=int, help="ID of the player's team")
@click.option("--opponent-id", required=True, type=int, help="ID of the opponent team")
@click.option("--is-home/--is-away", default=True, help="Whether the player's team is playing at home")
@click.option("--metric", multiple=True, help="Performance metrics to predict")
def predict_player_performance(player_id, match_id, team_id, opponent_id, is_home, metric):
    """Predict player performance for an upcoming match."""
    try:
        predictor = PlayerPerformancePredictor()
        
        # Get available metrics if none specified
        if not metric:
            available_metrics = predictor.get_available_metrics()
            if not available_metrics:
                click.echo("No player performance models available. Train models first.")
                return
            metrics = available_metrics
        else:
            metrics = list(metric)
        
        # Make prediction
        prediction = predictor.predict_player_performance(
            player_id=player_id,
            match_id=match_id,
            team_id=team_id,
            opponent_id=opponent_id,
            is_home=is_home,
            metrics=metrics
        )
        
        # Print prediction results
        click.echo("\n=== Player Performance Prediction ===")
        click.echo(f"Player ID: {player_id}")
        click.echo(f"Match ID: {match_id}")
        click.echo(f"Team: {team_id} vs Opponent: {opponent_id} ({'Home' if is_home else 'Away'})")
        
        for metric, value in prediction["predictions"].items():
            if value is not None:
                click.echo(f"{metric.capitalize()}: {value:.2f}")
            else:
                click.echo(f"{metric.capitalize()}: N/A (Prediction failed)")
                
    except Exception as e:
        click.echo(f"Error predicting player performance: {e}")
        logger.error(f"Error predicting player performance: {e}", exc_info=True)


@player.command(name="list-models")
def list_player_models():
    """List available player performance models."""
    try:
        predictor = PlayerPerformancePredictor()
        metrics = predictor.get_available_metrics()
        
        if not metrics:
            click.echo("No player performance models available. Train models first.")
            return
        
        click.echo("\n=== Available Player Performance Models ===")
        
        for metric in metrics:
            model_info = predictor.get_model_info(metric)
            
            if "error" in model_info:
                click.echo(f"{metric}: {model_info['error']}")
            else:
                click.echo(f"\nMetric: {metric}")
                click.echo(f"Model Type: {model_info['model_type']}")
                click.echo(f"File: {os.path.basename(model_info['file_path'])}")
                
                # Print top feature importances if available
                if model_info['feature_importances']:
                    click.echo("\nTop Feature Importances:")
                    for feat, imp in list(model_info['feature_importances'].items())[:5]:
                        click.echo(f"  {feat}: {imp:.4f}")
                
    except Exception as e:
        click.echo(f"Error listing player models: {e}")
        logger.error(f"Error listing player models: {e}", exc_info=True)


@player.command(name="history")
@click.option("--player-id", required=True, type=int, help="ID of the player")
@click.option("--limit", default=10, help="Maximum number of predictions to show")
@click.option("--metric", help="Filter predictions by specific metric")
def player_prediction_history(player_id, limit, metric):
    """Show prediction history for a specific player."""
    try:
        predictions = get_player_predictions(
            player_id=player_id,
            limit=limit,
            metric=metric
        )
        
        if not predictions:
            click.echo(f"No prediction history found for player {player_id}")
            return
        
        click.echo(f"\n=== Prediction History for Player {player_id} ===")
        
        for i, pred in enumerate(predictions):
            click.echo(f"\n[{i+1}] Match ID: {pred['match_id']} - {pred['timestamp']}")
            click.echo(f"Team: {pred['team_id']} vs Opponent: {pred['opponent_id']} ({'Home' if pred['is_home'] else 'Away'})")
            
            for metric, value in pred['predictions'].items():
                if value is not None:
                    click.echo(f"{metric.capitalize()}: {value:.2f}")
                else:
                    click.echo(f"{metric.capitalize()}: N/A")
                    
    except Exception as e:
        click.echo(f"Error retrieving player prediction history: {e}")
        logger.error(f"Error retrieving player prediction history: {e}", exc_info=True)


def main():
    """Main entry point for the models CLI."""
    args = parse_args()
    
    if args.debug:
        logger.setLevel("DEBUG")
    
    try:
        if args.command == "train":
            return train_cmd(args)
        elif args.command == "timeseries":
            return timeseries_cmd(args)
        elif args.command == "hyperopt":
            return hyperopt_cmd(args)
        elif args.command == "evaluate":
            return evaluate_cmd(args)
        elif args.command == "compare":
            return compare_cmd(args)
        elif args.command == "importance":
            return importance_cmd(args)
        elif args.command == "predict":
            return predict_cmd(args)
        elif args.command == "ts-predict":
            return ts_predict_cmd(args)
        elif args.command == "list":
            return list_cmd(args)
        elif args.command == "ensemble":
            return ensemble_cmd(args)
        elif args.command == "explain":
            return explain_cmd(args)
        elif args.command == "player":
            return train_player_models(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    main() 