"""
Model benchmarking tool for soccer prediction system.
Compares models on various metrics including performance, execution time and resource usage.
"""

import os
import sys
import json
import time
import psutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from memory_profiler import memory_usage

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components
from src.utils.logger import get_logger
from src.models.baseline import BaselineMatchPredictor
from src.models.advanced import AdvancedMatchPredictor
from src.models.ensemble import EnsemblePredictor
from src.models.evaluation import evaluate_model_performance, compare_models
from src.models.training import load_feature_data

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent, "data")

# Setup logger
logger = get_logger("benchmarking")

# Define paths
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmarks")
BENCHMARK_RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")
BENCHMARK_PLOTS_DIR = os.path.join(BENCHMARK_DIR, "plots")
os.makedirs(BENCHMARK_DIR, exist_ok=True)
os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)
os.makedirs(BENCHMARK_PLOTS_DIR, exist_ok=True)


def load_model(model_path: str) -> Union[BaselineMatchPredictor, AdvancedMatchPredictor, EnsemblePredictor]:
    """
    Load a model from file path, detecting model type automatically.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Union[BaselineMatchPredictor, AdvancedMatchPredictor, EnsemblePredictor]: The loaded model
    """
    logger.info(f"Loading model from {model_path}")
    try:
        # Try loading as different model types
        if "baseline" in model_path:
            return BaselineMatchPredictor.load(model_path)
        elif "advanced" in model_path:
            return AdvancedMatchPredictor.load(model_path)
        elif "ensemble" in model_path:
            return EnsemblePredictor.load(model_path)
        else:
            # If path doesn't indicate model type, try each type
            try:
                return BaselineMatchPredictor.load(model_path)
            except Exception:
                try:
                    return AdvancedMatchPredictor.load(model_path)
                except Exception:
                    try:
                        return EnsemblePredictor.load(model_path)
                    except Exception as e:
                        logger.error(f"Failed to load model from {model_path}: {e}")
                        raise ValueError(f"Could not load model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def measure_prediction_time(model: Union[BaselineMatchPredictor, AdvancedMatchPredictor, EnsemblePredictor], 
                            X: np.ndarray, 
                            n_runs: int = 5) -> Dict[str, float]:
    """
    Measure prediction time for a model.
    
    Args:
        model: The model to benchmark
        X: Input features
        n_runs: Number of times to run prediction for averaging
        
    Returns:
        Dict[str, float]: Dictionary with timing metrics
    """
    prediction_times = []
    proba_times = []
    
    # Warmup
    _ = model.predict(X[:10])
    _ = model.predict_proba(X[:10])
    
    # Measure prediction time
    for _ in range(n_runs):
        start_time = time.time()
        _ = model.predict(X)
        end_time = time.time()
        prediction_times.append(end_time - start_time)
    
    # Measure probability prediction time
    for _ in range(n_runs):
        start_time = time.time()
        _ = model.predict_proba(X)
        end_time = time.time()
        proba_times.append(end_time - start_time)
    
    return {
        "avg_prediction_time": np.mean(prediction_times),
        "min_prediction_time": np.min(prediction_times),
        "max_prediction_time": np.max(prediction_times),
        "std_prediction_time": np.std(prediction_times),
        "avg_proba_time": np.mean(proba_times),
        "min_proba_time": np.min(proba_times),
        "max_proba_time": np.max(proba_times),
        "std_proba_time": np.std(proba_times),
    }


def measure_memory_usage(model: Union[BaselineMatchPredictor, AdvancedMatchPredictor, EnsemblePredictor],
                         X: np.ndarray) -> Dict[str, float]:
    """
    Measure memory usage of a model during prediction.
    
    Args:
        model: The model to benchmark
        X: Input features
        
    Returns:
        Dict[str, float]: Dictionary with memory usage metrics in MiB
    """
    def predict_func():
        model.predict(X)
    
    def predict_proba_func():
        model.predict_proba(X)
    
    # Measure memory usage for prediction
    mem_usage_predict = memory_usage((predict_func, (), {}), interval=0.1, timeout=30)
    
    # Measure memory usage for probability prediction
    mem_usage_proba = memory_usage((predict_proba_func, (), {}), interval=0.1, timeout=30)
    
    # Calculate metrics
    baseline = memory_usage(-1, interval=0.1, timeout=1)[0]
    
    return {
        "baseline_memory": baseline,
        "peak_memory_predict": max(mem_usage_predict) - baseline,
        "avg_memory_predict": np.mean(mem_usage_predict) - baseline,
        "peak_memory_proba": max(mem_usage_proba) - baseline,
        "avg_memory_proba": np.mean(mem_usage_proba) - baseline,
    }


def measure_inference_scaling(model: Union[BaselineMatchPredictor, AdvancedMatchPredictor, EnsemblePredictor],
                             X: np.ndarray,
                             batch_sizes: List[int] = [1, 10, 100, 1000, 10000]) -> Dict[str, List[float]]:
    """
    Measure how prediction time scales with batch size.
    
    Args:
        model: The model to benchmark
        X: Input features
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dict[str, List[float]]: Dictionary with scaling metrics
    """
    predict_times = []
    proba_times = []
    
    for batch_size in batch_sizes:
        if batch_size > len(X):
            logger.warning(f"Batch size {batch_size} is larger than dataset size {len(X)}. Using full dataset.")
            batch_size = len(X)
        
        # Measure prediction time
        start_time = time.time()
        _ = model.predict(X[:batch_size])
        end_time = time.time()
        predict_times.append(end_time - start_time)
        
        # Measure probability prediction time
        start_time = time.time()
        _ = model.predict_proba(X[:batch_size])
        end_time = time.time()
        proba_times.append(end_time - start_time)
    
    return {
        "batch_sizes": batch_sizes,
        "predict_times": predict_times,
        "proba_times": proba_times,
    }


def run_benchmarks(model_paths: List[str],
                  dataset_name: str = "transfermarkt",
                  feature_type: str = "match_features",
                  target_col: str = "result",
                  n_runs: int = 5,
                  generate_plots: bool = True,
                  output_format: str = "both") -> Dict[str, Any]:
    """
    Run comprehensive benchmarks on multiple models.
    
    Args:
        model_paths: List of paths to the models to benchmark
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        target_col: Name of the target column
        n_runs: Number of runs for averaging timing measurements
        generate_plots: Whether to generate benchmark plots
        output_format: Output format ("json", "console", or "both")
        
    Returns:
        Dict[str, Any]: Comprehensive benchmark results
    """
    benchmark_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load dataset
    try:
        logger.info(f"Loading dataset {dataset_name} with feature type {feature_type}")
        df = load_feature_data(dataset_name, feature_type)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Load all models
    models = {}
    for model_path in model_paths:
        try:
            model = load_model(model_path)
            model_name = os.path.basename(model_path).split('.')[0]
            models[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {e}")
            continue
    
    if not models:
        logger.error("No models could be loaded. Exiting.")
        return {}
    
    # Prepare data for benchmarking
    X_dict = {}
    y_dict = {}
    
    for model_name, model in models.items():
        logger.info(f"Processing data for model: {model_name}")
        X, y = model.process_data(df, target_col=target_col)
        if X is None or y is None:
            logger.error(f"Failed to process data for model {model_name}")
            continue
        X_dict[model_name] = X
        y_dict[model_name] = y
    
    # Get performance metrics using existing evaluation functions
    logger.info("Evaluating model performance")
    performance_metrics = compare_models(
        model_paths=model_paths,
        dataset_name=dataset_name,
        feature_type=feature_type,
        target_col=target_col,
        generate_plots=generate_plots
    )
    
    # Benchmark each model
    for model_name, model in models.items():
        if model_name not in X_dict or model_name not in y_dict:
            continue
        
        X = X_dict[model_name]
        y = y_dict[model_name]
        
        logger.info(f"Benchmarking model: {model_name}")
        model_benchmark = {}
        
        # Measure prediction time
        model_benchmark["timing"] = measure_prediction_time(model, X, n_runs=n_runs)
        
        # Measure memory usage
        model_benchmark["memory"] = measure_memory_usage(model, X)
        
        # Measure scaling with batch size
        model_benchmark["scaling"] = measure_inference_scaling(model, X)
        
        # Model metadata
        model_info = getattr(model, "model_info", {})
        model_benchmark["metadata"] = {
            "model_type": model_info.get("model_type", "unknown"),
            "feature_count": X.shape[1],
            "dataset_size": len(X),
            "created_at": model_info.get("created_at", "unknown"),
        }
        
        # Get model file size
        model_path = [path for path in model_paths if model_name in path][0]
        model_benchmark["metadata"]["file_size_mb"] = os.path.getsize(model_path) / (1024 * 1024)
        
        benchmark_results[model_name] = model_benchmark
    
    # Combine performance and benchmark metrics
    benchmark_results["performance"] = performance_metrics
    benchmark_results["metadata"] = {
        "benchmark_timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "feature_type": feature_type,
        "n_runs": n_runs,
    }
    
    # Generate comparative plots if requested
    if generate_plots and len(models) > 1:
        plot_path = os.path.join(BENCHMARK_PLOTS_DIR, f"benchmark_{timestamp}")
        os.makedirs(plot_path, exist_ok=True)
        
        # Plot prediction times
        plt.figure(figsize=(12, 8))
        model_names = list(models.keys())
        predict_times = [benchmark_results[name]["timing"]["avg_prediction_time"] for name in model_names]
        proba_times = [benchmark_results[name]["timing"]["avg_proba_time"] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, predict_times, width, label="predict()")
        plt.bar(x + width/2, proba_times, width, label="predict_proba()")
        
        plt.xlabel("Model")
        plt.ylabel("Average Time (seconds)")
        plt.title("Prediction Time Comparison")
        plt.xticks(x, model_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        
        predict_time_path = os.path.join(plot_path, "prediction_time_comparison.png")
        plt.savefig(predict_time_path)
        plt.close()
        
        # Plot memory usage
        plt.figure(figsize=(12, 8))
        peak_memory = [benchmark_results[name]["memory"]["peak_memory_predict"] for name in model_names]
        avg_memory = [benchmark_results[name]["memory"]["avg_memory_predict"] for name in model_names]
        
        plt.bar(x - width/2, peak_memory, width, label="Peak Memory")
        plt.bar(x + width/2, avg_memory, width, label="Average Memory")
        
        plt.xlabel("Model")
        plt.ylabel("Memory Usage (MiB)")
        plt.title("Memory Usage Comparison")
        plt.xticks(x, model_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        
        memory_usage_path = os.path.join(plot_path, "memory_usage_comparison.png")
        plt.savefig(memory_usage_path)
        plt.close()
        
        # Plot scaling behavior for each model
        for model_name in model_names:
            plt.figure(figsize=(12, 8))
            scaling = benchmark_results[model_name]["scaling"]
            
            plt.plot(scaling["batch_sizes"], scaling["predict_times"], 'o-', label="predict()")
            plt.plot(scaling["batch_sizes"], scaling["proba_times"], 's-', label="predict_proba()")
            
            plt.xlabel("Batch Size")
            plt.ylabel("Time (seconds)")
            plt.title(f"Scaling Behavior: {model_name}")
            plt.xscale("log")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            scaling_path = os.path.join(plot_path, f"scaling_{model_name}.png")
            plt.savefig(scaling_path)
            plt.close()
        
        # Add plot paths to results
        benchmark_results["plots"] = {
            "prediction_time": predict_time_path,
            "memory_usage": memory_usage_path,
            "scaling": {model_name: os.path.join(plot_path, f"scaling_{model_name}.png") for model_name in model_names}
        }
    
    # Save results to file
    results_path = os.path.join(
        BENCHMARK_RESULTS_DIR,
        f"benchmark_results_{dataset_name}_{feature_type}_{timestamp}.json"
    )
    
    # Serialize to JSON
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    logger.info(f"Benchmark results saved to {results_path}")
    
    # Print results to console if requested
    if output_format in ["console", "both"]:
        print("\n===== MODEL BENCHMARKING RESULTS =====\n")
        
        # Performance comparison table
        print("\n----- PERFORMANCE METRICS -----\n")
        metrics_table = []
        for model_name in models.keys():
            model_metrics = performance_metrics.get("model_metrics", {}).get(model_name, {})
            if model_metrics:
                metrics_table.append([
                    model_name,
                    model_metrics.get("accuracy", "N/A"),
                    model_metrics.get("precision_weighted", "N/A"),
                    model_metrics.get("recall_weighted", "N/A"),
                    model_metrics.get("f1_weighted", "N/A"),
                    model_metrics.get("log_loss", "N/A")
                ])
        
        print(tabulate(
            metrics_table,
            headers=["Model", "Accuracy", "Precision", "Recall", "F1", "Log Loss"],
            floatfmt=".4f"
        ))
        
        # Timing comparison table
        print("\n----- TIMING METRICS (seconds) -----\n")
        timing_table = []
        for model_name in models.keys():
            timing = benchmark_results.get(model_name, {}).get("timing", {})
            if timing:
                timing_table.append([
                    model_name,
                    timing.get("avg_prediction_time", "N/A"),
                    timing.get("avg_proba_time", "N/A")
                ])
        
        print(tabulate(
            timing_table,
            headers=["Model", "Avg Prediction Time", "Avg Probability Time"],
            floatfmt=".6f"
        ))
        
        # Memory usage comparison table
        print("\n----- MEMORY USAGE (MiB) -----\n")
        memory_table = []
        for model_name in models.keys():
            memory = benchmark_results.get(model_name, {}).get("memory", {})
            if memory:
                memory_table.append([
                    model_name,
                    memory.get("peak_memory_predict", "N/A"),
                    memory.get("avg_memory_predict", "N/A"),
                    benchmark_results.get(model_name, {}).get("metadata", {}).get("file_size_mb", "N/A")
                ])
        
        print(tabulate(
            memory_table,
            headers=["Model", "Peak Memory", "Avg Memory", "Model Size (MB)"],
            floatfmt=".2f"
        ))
    
    return benchmark_results


def main():
    """Main function to run the benchmarking tool."""
    parser = argparse.ArgumentParser(description="Soccer Prediction Model Benchmarking Tool")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to model files to benchmark")
    parser.add_argument("--dataset", default="transfermarkt", help="Dataset name")
    parser.add_argument("--features", default="match_features", help="Feature type")
    parser.add_argument("--target", default="result", help="Target column name")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs for timing measurements")
    parser.add_argument("--no-plots", action="store_false", dest="generate_plots", help="Disable plot generation")
    parser.add_argument("--output", choices=["json", "console", "both"], default="both", help="Output format")
    
    args = parser.parse_args()
    
    try:
        run_benchmarks(
            model_paths=args.models,
            dataset_name=args.dataset,
            feature_type=args.features,
            target_col=args.target,
            n_runs=args.runs,
            generate_plots=args.generate_plots,
            output_format=args.output
        )
    except Exception as e:
        logger.error(f"Error running benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 