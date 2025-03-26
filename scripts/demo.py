#!/usr/bin/env python
"""
Demo script for the Soccer Prediction System.
Demonstrates the model training and prediction capabilities.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from src.utils.logger import get_logger
from src.models.training import train_model, train_multiple_models, ensemble_models
from src.models.evaluation import evaluate_model_performance, compare_models, analyze_feature_importance
from src.models.prediction import PredictionService

# Setup logger
logger = get_logger("demo")

# Define paths
DATA_DIR = os.path.join(project_root, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
DEMO_DIR = os.path.join(DATA_DIR, "demo")
os.makedirs(DEMO_DIR, exist_ok=True)


def train_demo_models(
    dataset_name="transfermarkt",
    feature_type="match_features",
    with_tuning=False
):
    """Train a set of demo models."""
    logger.info("Training demo models...")
    
    # Create synthetic/mock data if real data doesn't exist
    features_path = os.path.join(FEATURES_DIR, dataset_name, f"{feature_type}.csv")
    if not os.path.exists(features_path):
        logger.info(f"Feature file not found: {features_path}")
        logger.info("Creating synthetic data for demo purposes...")
        create_synthetic_data(dataset_name, feature_type)
    
    # Train multiple models
    results = train_multiple_models(
        model_types=["logistic", "random_forest", "xgboost"],
        dataset_name=dataset_name,
        feature_type=feature_type,
        target_col="result",
        hyperparameter_tuning=with_tuning,
        create_ensemble=True,
        test_size=0.2,
        cv_folds=3
    )
    
    # Print results summary
    print("\nTraining Results:")
    print("-" * 80)
    for model_type, result in results.items():
        if model_type == "ensemble":
            print(f"Ensemble created with {len(result.get('models', []))} models")
        else:
            performance = result.get("performance", {})
            acc = performance.get("accuracy", 0)
            f1 = performance.get("f1", 0)
            print(f"{model_type.capitalize()} Model - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print("-" * 80)
    
    # Return model paths
    model_paths = [result.get("model_path") for model_type, result in results.items() 
                  if model_type != "ensemble" and "model_path" in result]
    
    return model_paths


def create_synthetic_data(dataset_name, feature_type):
    """Create synthetic data for demo purposes."""
    # Create directory if it doesn't exist
    feature_dir = os.path.join(FEATURES_DIR, dataset_name)
    os.makedirs(feature_dir, exist_ok=True)
    
    # Create synthetic dataset
    n_samples = 1000
    n_teams = 20
    
    # Generate team IDs
    home_teams = np.random.randint(1, n_teams + 1, n_samples)
    away_teams = np.random.randint(1, n_teams + 1, n_samples)
    
    # Ensure home and away teams are different
    for i in range(n_samples):
        while home_teams[i] == away_teams[i]:
            away_teams[i] = np.random.randint(1, n_teams + 1)
    
    # Generate features
    home_form = np.random.uniform(0, 1, n_samples)
    away_form = np.random.uniform(0, 1, n_samples)
    home_goals_scored_avg = np.random.uniform(0.5, 3, n_samples)
    away_goals_scored_avg = np.random.uniform(0.5, 3, n_samples)
    home_goals_conceded_avg = np.random.uniform(0.5, 2.5, n_samples)
    away_goals_conceded_avg = np.random.uniform(0.5, 2.5, n_samples)
    
    # Historical head-to-head
    h2h_home_wins = np.random.randint(0, 10, n_samples)
    h2h_away_wins = np.random.randint(0, 10, n_samples)
    h2h_draws = np.random.randint(0, 5, n_samples)
    
    # Team rankings
    home_ranking = np.random.randint(1, 101, n_samples)
    away_ranking = np.random.randint(1, 101, n_samples)
    
    # Home advantage and match importance
    home_advantage = np.random.uniform(0.5, 1.5, n_samples)
    match_importance = np.random.uniform(0.5, 2, n_samples)
    
    # Generate target variable - match result
    # 0: home win, 1: draw, 2: away win
    probabilities = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        # Calculate probabilities based on features
        home_strength = home_form[i] + 0.8 * home_goals_scored_avg[i] - 0.6 * home_goals_conceded_avg[i] + 0.5 * (h2h_home_wins[i] / (h2h_home_wins[i] + h2h_draws[i] + h2h_away_wins[i] + 1e-10)) - 0.01 * home_ranking[i] + 0.3 * home_advantage[i]
        away_strength = away_form[i] + 0.8 * away_goals_scored_avg[i] - 0.6 * away_goals_conceded_avg[i] + 0.5 * (h2h_away_wins[i] / (h2h_home_wins[i] + h2h_draws[i] + h2h_away_wins[i] + 1e-10)) - 0.01 * away_ranking[i]
        
        # Add some randomness
        home_strength += np.random.normal(0, 0.5)
        away_strength += np.random.normal(0, 0.5)
        
        # Convert to probabilities
        probabilities[i, 0] = max(0.1, min(0.8, home_strength / (home_strength + away_strength + 0.5)))  # Home win
        probabilities[i, 2] = max(0.1, min(0.8, away_strength / (home_strength + away_strength + 0.5)))  # Away win
        probabilities[i, 1] = 1.0 - probabilities[i, 0] - probabilities[i, 2]  # Draw
    
    # Sample results based on probabilities
    results = np.array(['home_win', 'draw', 'away_win'])
    match_results = [np.random.choice(results, p=probabilities[i]) for i in range(n_samples)]
    
    # Create DataFrame
    data = {
        'home_team_id': home_teams,
        'away_team_id': away_teams,
        'home_form': home_form,
        'away_form': away_form,
        'home_goals_scored_avg': home_goals_scored_avg,
        'away_goals_scored_avg': away_goals_scored_avg,
        'home_goals_conceded_avg': home_goals_conceded_avg,
        'away_goals_conceded_avg': away_goals_conceded_avg,
        'h2h_home_wins': h2h_home_wins,
        'h2h_away_wins': h2h_away_wins,
        'h2h_draws': h2h_draws,
        'home_ranking': home_ranking,
        'away_ranking': away_ranking,
        'home_advantage': home_advantage,
        'match_importance': match_importance,
        'result': match_results
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = os.path.join(feature_dir, f"{feature_type}.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Created synthetic dataset with {n_samples} samples at {output_path}")


def evaluate_demo_models(model_paths):
    """Evaluate the trained demo models."""
    if not model_paths:
        logger.error("No model paths provided for evaluation")
        return
    
    logger.info(f"Evaluating {len(model_paths)} models...")
    
    # Evaluate individual models
    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace(".pkl", "")
        logger.info(f"Evaluating model: {model_name}")
        
        try:
            # Perform evaluation
            eval_result = evaluate_model_performance(
                model_path=model_path,
                generate_plots=True,
                verbose=True
            )
            
            # Analyze feature importance
            if model_name != "ensemble":
                try:
                    importance_result = analyze_feature_importance(
                        model_path=model_path,
                        n_top_features=10
                    )
                except ValueError as e:
                    logger.warning(f"Could not analyze feature importance: {e}")
        
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
    
    # Compare models
    if len(model_paths) > 1:
        try:
            logger.info("Comparing models...")
            comparison_result = compare_models(
                model_paths=model_paths,
                generate_plots=True
            )
        except Exception as e:
            logger.error(f"Error comparing models: {e}")


def run_demo_predictions():
    """Run demo predictions using trained models."""
    logger.info("Running demo predictions...")
    
    # Initialize prediction service
    prediction_service = PredictionService()
    
    # Get available models
    models = prediction_service.get_available_models()
    
    if not models:
        logger.error("No models available for prediction")
        return
    
    # Print available models
    print("\nAvailable Models:")
    print("-" * 80)
    for name, info in models.items():
        if info["type"] == "single":
            model_type = info.get("model_type", "unknown")
            print(f"{name:<30} (Type: {model_type:<15} Single model)")
        else:
            ensemble_type = info.get("ensemble_type", "unknown")
            n_models = info.get("n_models", "?")
            print(f"{name:<30} (Type: {ensemble_type:<15} Ensemble with {n_models} models)")
    print("-" * 80)
    
    # Create some test match scenarios
    test_matches = [
        {"home_team_id": 1, "away_team_id": 2, "match_id": 1001},
        {"home_team_id": 3, "away_team_id": 4, "match_id": 1002},
        {"home_team_id": 5, "away_team_id": 1, "match_id": 1003},
        {"home_team_id": 2, "away_team_id": 3, "match_id": 1004},
        {"home_team_id": 4, "away_team_id": 5, "match_id": 1005}
    ]
    
    # Make predictions using different models
    print("\nPredictions using different models:")
    print("-" * 80)
    
    # Try to use each model for the first match
    first_match = test_matches[0]
    print(f"Match: Home Team {first_match['home_team_id']} vs Away Team {first_match['away_team_id']}")
    
    for model_name in models:
        try:
            result = prediction_service.predict_match(
                home_team_id=first_match["home_team_id"],
                away_team_id=first_match["away_team_id"],
                model_name=model_name
            )
            
            if "error" in result:
                print(f"  {model_name}: Error - {result['error']}")
            else:
                print(f"  {model_name}: {result['prediction']} " 
                      f"(Home: {result['home_win_probability']:.2f}, "
                      f"Draw: {result['draw_probability']:.2f}, "
                      f"Away: {result['away_win_probability']:.2f}, "
                      f"Confidence: {result['confidence']:.2f})")
        except Exception as e:
            print(f"  {model_name}: Exception - {str(e)}")
    
    print("-" * 80)
    
    # Try batch predictions with the ensemble model
    ensemble_models = [name for name, info in models.items() if info["type"] == "ensemble"]
    if ensemble_models:
        ensemble_model = ensemble_models[0]
        try:
            print(f"\nBatch predictions using {ensemble_model}:")
            print("-" * 80)
            
            batch_results = prediction_service.batch_predict(
                matches=test_matches,
                model_name=ensemble_model
            )
            
            for result in batch_results:
                if "error" in result:
                    print(f"Match {result.get('match_id', 'unknown')}: Error - {result['error']}")
                else:
                    print(f"Match {result.get('match_id', 'unknown')}: {result['prediction']} "
                          f"(Confidence: {result['confidence']:.2f})")
            
            print("-" * 80)
        except Exception as e:
            logger.error(f"Error making batch predictions: {e}")
    else:
        logger.warning("No ensemble models available for batch prediction")
    
    # Save demo results
    results_path = os.path.join(DEMO_DIR, f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(results_path, "w") as f:
            json.dump({
                "predictions": batch_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, default=str)
        logger.info(f"Demo prediction results saved to {results_path}")
    except Exception as e:
        logger.error(f"Error saving demo results: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Soccer Prediction System Demo")
    
    parser.add_argument("--train", action="store_true",
                      help="Train demo models")
    parser.add_argument("--evaluate", action="store_true",
                      help="Evaluate demo models")
    parser.add_argument("--predict", action="store_true",
                      help="Run demo predictions")
    parser.add_argument("--tuning", action="store_true",
                      help="Perform hyperparameter tuning during training")
    parser.add_argument("--dataset", type=str, default="transfermarkt",
                      help="Dataset name for demo (default: transfermarkt)")
    parser.add_argument("--feature-type", type=str, default="match_features",
                      help="Feature type for demo (default: match_features)")
    parser.add_argument("--all", action="store_true",
                      help="Run full demo (train, evaluate, predict)")
    
    return parser.parse_args()


def main():
    """Run the demo script."""
    args = parse_args()
    
    # Show info
    logger.info("Soccer Prediction System Demo")
    logger.info(f"Project root: {project_root}")
    
    # Run all steps if --all is specified
    if args.all:
        args.train = args.evaluate = args.predict = True
    
    # If no arguments, run the full demo
    if not (args.train or args.evaluate or args.predict):
        logger.info("No specific actions specified. Running full demo.")
        args.train = args.evaluate = args.predict = True
    
    # Run the requested steps
    model_paths = []
    
    if args.train:
        model_paths = train_demo_models(
            dataset_name=args.dataset,
            feature_type=args.feature_type,
            with_tuning=args.tuning
        )
    
    if args.evaluate:
        # If we didn't train models in this run, try to find existing ones
        if not model_paths and os.path.exists(MODELS_DIR):
            model_paths = [os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR) 
                          if f.endswith(".pkl") and os.path.isfile(os.path.join(MODELS_DIR, f))]
        
        if model_paths:
            evaluate_demo_models(model_paths)
        else:
            logger.error("No models found to evaluate. Please run with --train first.")
    
    if args.predict:
        run_demo_predictions()
    
    logger.info("Demo completed.")


if __name__ == "__main__":
    main() 