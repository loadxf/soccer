#!/usr/bin/env python
"""
Ensemble Model Training Script

This script demonstrates how to train ensemble models using the Soccer Prediction System.
It shows how to create various types of ensembles, including voting, stacking, and blending.
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project components
from src.utils.logger import get_logger
from src.models.ensemble import EnsemblePredictor, train_ensemble_model
from src.models.baseline import BaselineMatchPredictor
from src.models.advanced import AdvancedMatchPredictor

# Setup logger
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments for ensemble training script."""
    parser = argparse.ArgumentParser(description="Ensemble Model Training Script")
    
    # General options
    parser.add_argument("--ensemble-type", choices=["voting", "stacking", "blending", 
                                               "calibrated_voting", "time_weighted", "performance_weighted"],
                      default="voting", help="Type of ensemble to train")
    parser.add_argument("--models", nargs="+",
                      default=["logistic", "random_forest", "neural_network"],
                      help="List of models to include in the ensemble")
    parser.add_argument("--dataset", default="transfermarkt",
                      help="Dataset to use for training")
    parser.add_argument("--features", default="match_features",
                      help="Feature set to use")
    parser.add_argument("--output-dir", default="data/models/ensemble",
                      help="Directory to save the models")
    parser.add_argument("--name", help="Name prefix for the saved model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()


def main():
    """Main function for the ensemble training script."""
    args = parse_args()
    
    # Configure logger
    if args.verbose:
        logger.setLevel("DEBUG")
    
    # Print script configuration
    logger.info(f"Training {args.ensemble_type} ensemble")
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Features: {args.features}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train ensemble model
    try:
        ensemble = train_ensemble_model(
            ensemble_type=args.ensemble_type,
            model_types=args.models,
            dataset_name=args.dataset,
            feature_type=args.features
        )
        
        # Save the trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.name or f"{args.ensemble_type}_ensemble"
        model_path = os.path.join(
            args.output_dir, 
            f"{model_name}_{'-'.join(args.models)}_{timestamp}.joblib"
        )
        
        ensemble.save(model_path)
        logger.info(f"Ensemble model saved to {model_path}")
        
        # Print ensemble summary
        print("\nEnsemble Model Summary:")
        print(f"  - Type: {args.ensemble_type}")
        print(f"  - Models: {', '.join(args.models)}")
        print(f"  - Weights: {ensemble.weights}")
        print(f"  - Model Count: {len(ensemble.models)}")
        
        # Demonstrate loading and using the ensemble
        print("\nDemonstrating ensemble loading and prediction:")
        loaded_ensemble = EnsemblePredictor.load(model_path)
        print(f"  - Loaded ensemble with {len(loaded_ensemble.models)} models")
        
        # Make a sample prediction
        sample_prediction = loaded_ensemble.predict_match(
            home_team_id=1, away_team_id=2
        )
        
        # Print prediction result
        print("\nSample Prediction:")
        print(f"  - Match: Home Team #{sample_prediction['match']['home_team_id']} vs Away Team #{sample_prediction['match']['away_team_id']}")
        print(f"  - Predicted Outcome: {sample_prediction['prediction']['outcome']}")
        print(f"  - Probabilities:")
        print(f"    * Home Win: {sample_prediction['prediction']['probabilities']['home_win']:.4f}")
        print(f"    * Draw: {sample_prediction['prediction']['probabilities']['draw']:.4f}")
        print(f"    * Away Win: {sample_prediction['prediction']['probabilities']['away_win']:.4f}")
        
    except Exception as e:
        logger.error(f"Error training ensemble model: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    logger.info("Ensemble training completed successfully")


if __name__ == "__main__":
    main() 