#!/usr/bin/env python3
"""
Script to run the model benchmarking tool on existing models.
"""

import os
import sys
import glob
import argparse
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_benchmarking import run_benchmarks
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("run_benchmarks")

# Try to get DATA_DIR from config or set default
try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent, "data")

# Define models directory 
MODELS_DIR = os.path.join(DATA_DIR, "models")


def find_model_files(base_dir=MODELS_DIR, n_models=3, model_type=None):
    """
    Find model files in the models directory.
    
    Args:
        base_dir: Base directory to search in
        n_models: Maximum number of models to return
        model_type: Type of models to look for ('baseline', 'advanced', 'ensemble', or None for all)
        
    Returns:
        List[str]: Paths to model files
    """
    if model_type:
        search_pattern = os.path.join(base_dir, model_type, "*.pkl")
    else:
        search_pattern = os.path.join(base_dir, "**", "*.pkl")
        
    model_files = glob.glob(search_pattern, recursive=True)
    
    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    
    return model_files[:n_models]


def main():
    """Main function to run model benchmarks."""
    parser = argparse.ArgumentParser(description="Run Soccer Prediction Model Benchmarks")
    parser.add_argument("--type", choices=["baseline", "advanced", "ensemble", "all"], 
                        default="all", help="Type of models to benchmark")
    parser.add_argument("--count", type=int, default=3, help="Number of models to benchmark")
    parser.add_argument("--dataset", default="transfermarkt", help="Dataset name")
    parser.add_argument("--features", default="match_features", help="Feature type")
    parser.add_argument("--target", default="result", help="Target column name")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for timing measurements")
    parser.add_argument("--no-plots", action="store_false", dest="generate_plots", help="Disable plot generation")
    
    args = parser.parse_args()
    
    try:
        # Find model files
        if args.type == "all":
            # Get models of each type
            baseline_models = find_model_files(os.path.join(MODELS_DIR, "baseline"), 
                                              n_models=args.count)
            advanced_models = find_model_files(os.path.join(MODELS_DIR, "advanced"), 
                                              n_models=args.count)
            ensemble_models = find_model_files(os.path.join(MODELS_DIR, "ensemble"), 
                                              n_models=args.count)
            model_paths = baseline_models + advanced_models + ensemble_models
        else:
            model_paths = find_model_files(os.path.join(MODELS_DIR, args.type), 
                                          n_models=args.count)
        
        if not model_paths:
            logger.error(f"No models found of type '{args.type}'")
            sys.exit(1)
        
        logger.info(f"Found {len(model_paths)} models to benchmark")
        for i, path in enumerate(model_paths, 1):
            logger.info(f"  {i}. {os.path.basename(path)}")
        
        # Run benchmarks
        run_benchmarks(
            model_paths=model_paths,
            dataset_name=args.dataset,
            feature_type=args.features,
            target_col=args.target,
            n_runs=args.runs,
            generate_plots=args.generate_plots,
            output_format="both"
        )
        
    except Exception as e:
        logger.error(f"Error running benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 