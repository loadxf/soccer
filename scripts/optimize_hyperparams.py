#!/usr/bin/env python
"""
Script to run hyperparameter optimization for soccer prediction models.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# Add the project root directory to the Python path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)

# Import project components
from src.utils.logger import setup_logger, get_logger
from src.models.hyperopt import optimize_hyperparameters

# Setup logging
logger = get_logger("scripts.optimize_hyperparams")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for Soccer Prediction Models")
    
    parser.add_argument("--model-type", required=True,
                        help="Type of model to optimize (e.g., logistic, random_forest, neural_network)")
    parser.add_argument("--dataset", default="transfermarkt",
                        help="Dataset to use (default: transfermarkt)")
    parser.add_argument("--features", default="match_features",
                        help="Type of features to use (default: match_features)")
    parser.add_argument("--target", default="result",
                        help="Target column (default: result)")
    parser.add_argument("--max-evals", type=int, default=50,
                        help="Maximum number of evaluations to perform (default: 50)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Portion of data to use for testing (default: 0.2)")
    parser.add_argument("--val-size", type=float, default=0.25,
                        help="Portion of training data to use for validation (default: 0.25)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--custom-space", type=str, default=None,
                        help="JSON file with custom search space")
    parser.add_argument("--run-all", action="store_true",
                        help="Run optimization for all model types")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save optimization results")
    parser.add_argument("--verbose", action="store_true",
                        help="Increase output verbosity")
    
    return parser.parse_args()

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def run_optimization(model_type, args, custom_space=None):
    """Run hyperparameter optimization for a single model type."""
    logger.info(f"Starting hyperparameter optimization for {model_type}")
    
    start_time = datetime.now()
    
    try:
        results = optimize_hyperparameters(
            model_type=model_type,
            dataset_name=args.dataset,
            feature_type=args.features,
            target_col=args.target,
            test_size=args.test_size,
            val_size=args.val_size,
            max_evals=args.max_evals,
            random_state=args.random_state,
            search_space=custom_space
        )
        
        # Calculate total time
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Check for error
        if 'error' in results:
            logger.error(f"Optimization for {model_type} failed: {results['error']}")
            return {
                'model_type': model_type,
                'status': 'failed',
                'error': results['error'],
                'time_taken': total_time
            }
        
        # Print results
        logger.info(f"\nOptimization for {model_type} completed in {format_time(total_time)}")
        logger.info(f"Best validation F1 score: {results['best_metrics']['f1_score']:.4f}")
        logger.info(f"Best test F1 score: {results['test_metrics']['f1']:.4f}")
        logger.info(f"Optimized model saved to: {results['model_path']}")
        logger.info(f"Detailed results saved to: {results['results_path']}")
        
        # Return summary
        return {
            'model_type': model_type,
            'status': 'success',
            'best_params': results['best_params'],
            'best_val_f1': results['best_metrics']['f1_score'],
            'test_f1': results['test_metrics']['f1'],
            'test_accuracy': results['test_metrics']['accuracy'],
            'model_path': results['model_path'],
            'results_path': results['results_path'],
            'time_taken': total_time
        }
    
    except Exception as e:
        # Calculate total time even if there was an error
        total_time = (datetime.now() - start_time).total_seconds()
        
        logger.error(f"Error during optimization for {model_type}: {e}")
        return {
            'model_type': model_type,
            'status': 'error',
            'error': str(e),
            'time_taken': total_time
        }

def main():
    """Main function."""
    args = parse_args()
    
    # Setup verbose logging if requested
    if args.verbose:
        setup_logger(level="DEBUG")
    
    # Load custom search space if provided
    custom_space = None
    if args.custom_space:
        try:
            with open(args.custom_space, 'r') as f:
                custom_space = json.load(f)
            logger.info(f"Loaded custom search space from {args.custom_space}")
        except Exception as e:
            logger.error(f"Error loading custom search space: {e}")
            return
    
    # Set up model types to optimize
    if args.run_all:
        # Both baseline and advanced models
        model_types = [
            "logistic", 
            "random_forest", 
            "xgboost", 
            "neural_network", 
            "lightgbm", 
            "catboost"
        ]
    else:
        model_types = [args.model_type]
    
    # Track overall results
    overall_results = []
    overall_start_time = datetime.now()
    
    # Run optimization for each model type
    for model_type in model_types:
        # For the custom search space, extract just the relevant part for this model
        model_search_space = None
        if custom_space and model_type in custom_space:
            model_search_space = custom_space[model_type]
        
        # Run optimization
        result = run_optimization(model_type, args, model_search_space)
        overall_results.append(result)
    
    # Calculate total time
    total_time = (datetime.now() - overall_start_time).total_seconds()
    
    # Print summary of all optimizations
    logger.info("\n" + "="*50)
    logger.info("HYPERPARAMETER OPTIMIZATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total time: {format_time(total_time)}")
    logger.info(f"Models optimized: {len(overall_results)}")
    
    # Count successful optimizations
    successful = [r for r in overall_results if r['status'] == 'success']
    logger.info(f"Successful optimizations: {len(successful)}/{len(overall_results)}")
    
    # Table header for successful optimizations
    if successful:
        logger.info("\nSuccessful models:")
        logger.info(f"{'Model Type':<15} {'Val F1':<10} {'Test F1':<10} {'Test Acc':<10} {'Time':<15}")
        logger.info("-"*60)
        
        # Sort by test F1 score
        successful.sort(key=lambda x: x.get('test_f1', 0), reverse=True)
        
        for result in successful:
            logger.info(f"{result['model_type']:<15} "
                       f"{result.get('best_val_f1', 0):<10.4f} "
                       f"{result.get('test_f1', 0):<10.4f} "
                       f"{result.get('test_accuracy', 0):<10.4f} "
                       f"{format_time(result['time_taken']):<15}")
    
    # Print failed optimizations
    failed = [r for r in overall_results if r['status'] != 'success']
    if failed:
        logger.info("\nFailed optimizations:")
        for result in failed:
            logger.info(f"  {result['model_type']}: {result.get('error', 'Unknown error')}")
    
    # Save overall results to JSON
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(
            args.output_dir,
            f"hyperopt_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump({
                'overall': {
                    'total_time': total_time,
                    'models_optimized': len(overall_results),
                    'successful': len(successful),
                    'failed': len(failed),
                    'timestamp': datetime.now().isoformat()
                },
                'results': overall_results
            }, f, indent=2, default=str)
        
        logger.info(f"\nSummary saved to {results_file}")


if __name__ == "__main__":
    main() 