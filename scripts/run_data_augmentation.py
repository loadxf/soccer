#!/usr/bin/env python
"""
Script to run data augmentation for the Soccer Prediction System.
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components
from src.data.augmentation import run_augmentation
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("scripts.run_data_augmentation")


def main():
    """Run data augmentation based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run data augmentation")
    
    # Required arguments
    parser.add_argument('dataset', type=str, help='Name of the dataset to augment')
    parser.add_argument('augmentation_type', type=str, 
                       choices=['oversample', 'undersample', 'synthetic', 'noise', 'time_series'],
                       help='Type of augmentation to apply')
    
    # Optional arguments for all augmentation types
    parser.add_argument('--no-save', action='store_true', help='Do not save the augmented data')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
    
    # Arguments for specific augmentation types
    parser.add_argument('--target-col', type=str, help='Target column for classification augmentation')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of synthetic samples to generate')
    parser.add_argument('--noise-level', type=float, default=0.05, help='Noise level for adding Gaussian noise')
    parser.add_argument('--columns', type=str, help='Comma-separated list of columns to add noise to or use in time series')
    parser.add_argument('--time-col', type=str, help='Time column for time series augmentation')
    parser.add_argument('--shift-values', type=str, help='Comma-separated list of shift values for time series augmentation')
    parser.add_argument('--window-sizes', type=str, help='Comma-separated list of window sizes for time series augmentation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Build kwargs for the augmentation function
    kwargs = {
        'save': not args.no_save,
        'random_state': args.random_state
    }
    
    # Add type-specific arguments
    if args.augmentation_type in ['oversample', 'undersample']:
        if not args.target_col:
            logger.error(f"Target column (--target-col) is required for {args.augmentation_type}")
            sys.exit(1)
        kwargs['target_col'] = args.target_col
    
    elif args.augmentation_type == 'synthetic':
        if args.n_samples:
            kwargs['n_samples'] = args.n_samples
    
    elif args.augmentation_type == 'noise':
        if not args.columns:
            logger.error("Columns (--columns) is required for noise augmentation")
            sys.exit(1)
        kwargs['columns'] = args.columns.split(',')
        if args.noise_level:
            kwargs['noise_level'] = args.noise_level
    
    elif args.augmentation_type == 'time_series':
        if not args.time_col or not args.columns:
            logger.error("Time column (--time-col) and columns (--columns) are required for time series augmentation")
            sys.exit(1)
        kwargs['time_col'] = args.time_col
        kwargs['value_cols'] = args.columns.split(',')
        
        if args.shift_values:
            kwargs['shift_values'] = [int(val) for val in args.shift_values.split(',')]
        if args.window_sizes:
            kwargs['window_sizes'] = [int(val) for val in args.window_sizes.split(',')]
    
    # Log the configuration
    logger.info(f"Running {args.augmentation_type} augmentation on {args.dataset} dataset")
    logger.info(f"Configuration: {json.dumps(kwargs, default=str)}")
    
    # Run the augmentation
    start_time = datetime.now()
    
    try:
        result_df = run_augmentation(args.dataset, args.augmentation_type, **kwargs)
        
        # Log results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result_df.empty:
            logger.error(f"Augmentation failed or returned empty result")
            sys.exit(1)
        
        logger.info(f"Augmentation completed in {duration:.2f} seconds")
        logger.info(f"Result shape: {result_df.shape}")
        
    except Exception as e:
        logger.error(f"Error during augmentation: {e}")
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 