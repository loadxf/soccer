"""
Evaluation module for soccer prediction models.
Handles performance evaluation and visualization of model results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    log_loss
)

# Import project components
from src.utils.logger import get_logger
from src.models.baseline import BaselineMatchPredictor
from src.models.training import load_feature_data, TRAINING_DIR

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback default if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.evaluation")

# Define paths
EVALUATION_DIR = os.path.join(DATA_DIR, "evaluation")
PLOTS_DIR = os.path.join(EVALUATION_DIR, "plots")
os.makedirs(EVALUATION_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate_model_performance(
    model_path: str,
    dataset_name: Optional[str] = None,
    feature_type: Optional[str] = None,
    target_col: str = "result",
    generate_plots: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a trained model.
    
    Args:
        model_path: Path to the saved model
        dataset_name: Name of the dataset to use (if None, use the one from the model)
        feature_type: Type of features to use (if None, use the one from the model)
        target_col: Name of the target column
        generate_plots: Whether to generate performance visualization plots
        verbose: Whether to print evaluation results
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation results
    """
    # Load the model
    try:
        model = BaselineMatchPredictor.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise
    
    # Use dataset and feature type from model if not provided
    dataset_name = dataset_name or model.dataset_name
    feature_type = feature_type or model.feature_type
    
    # Load data
    try:
        df = load_feature_data(dataset_name, feature_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading feature data: {e}")
        raise
    
    # Process data
    X, y = model.process_data(df, target_col=target_col)
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Convert class labels to indices if needed
    classes = np.unique(y)
    class_indices = {label: i for i, label in enumerate(classes)}
    y_indices = np.array([class_indices[label] for label in y])
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro"),
        "recall_macro": recall_score(y, y_pred, average="macro"),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "precision_weighted": precision_score(y, y_pred, average="weighted"),
        "recall_weighted": recall_score(y, y_pred, average="weighted"),
        "f1_weighted": f1_score(y, y_pred, average="weighted"),
        "log_loss": log_loss(y, y_proba)
    }
    
    # Per-class metrics
    class_metrics = {}
    for i, class_label in enumerate(classes):
        mask = (y == class_label)
        class_metrics[str(class_label)] = {
            "precision": precision_score(y == class_label, y_pred == class_label),
            "recall": recall_score(y == class_label, y_pred == class_label),
            "f1": f1_score(y == class_label, y_pred == class_label),
            "support": np.sum(mask)
        }
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate plots if requested
    if generate_plots:
        # Create directory for this evaluation
        plot_dir = os.path.join(PLOTS_DIR, f"{model.model_type}_{dataset_name}_{feature_type}_{timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Normalized Confusion Matrix - {model.model_type}")
        plt.tight_layout()
        confusion_matrix_path = os.path.join(plot_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        plt.close()
        
        # ROC curve for multiclass (one-vs-rest)
        # Only works for 3 or more classes
        if len(classes) >= 3:
            plt.figure(figsize=(10, 8))
            for i, class_label in enumerate(classes):
                mask = (y == class_label)
                if np.any(mask):
                    fpr, tpr, _ = roc_curve(mask, y_proba[:, i])
                    auc = roc_auc_score(mask, y_proba[:, i])
                    plt.plot(fpr, tpr, label=f'{class_label} (AUC = {auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model.model_type}')
            plt.legend(loc='best')
            roc_curve_path = os.path.join(plot_dir, "roc_curve.png")
            plt.savefig(roc_curve_path)
            plt.close()
        
        # Precision-Recall curve
        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(classes):
            mask = (y == class_label)
            if np.any(mask):
                precision, recall, _ = precision_recall_curve(mask, y_proba[:, i])
                ap = average_precision_score(mask, y_proba[:, i])
                plt.plot(recall, precision, label=f'{class_label} (AP = {ap:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model.model_type}')
        plt.legend(loc='best')
        pr_curve_path = os.path.join(plot_dir, "precision_recall_curve.png")
        plt.savefig(pr_curve_path)
        plt.close()
        
        # Class distribution plot
        plt.figure(figsize=(10, 6))
        sns.countplot(x=y)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution in Dataset')
        plt.xticks(range(len(classes)), classes)
        class_dist_path = os.path.join(plot_dir, "class_distribution.png")
        plt.savefig(class_dist_path)
        plt.close()
        
        plot_paths = {
            "confusion_matrix": confusion_matrix_path,
            "roc_curve": roc_curve_path if len(classes) >= 3 else None,
            "precision_recall_curve": pr_curve_path,
            "class_distribution": class_dist_path
        }
    else:
        plot_paths = None
    
    # Create comprehensive evaluation results
    eval_results = {
        "model_info": model.model_info,
        "dataset_name": dataset_name,
        "feature_type": feature_type,
        "target_col": target_col,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "class_distribution": {str(c): int(np.sum(y == c)) for c in classes},
        "metrics": metrics,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "plot_paths": plot_paths,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save evaluation results
    results_path = os.path.join(
        EVALUATION_DIR,
        f"{model.model_type}_{dataset_name}_{feature_type}_evaluation_{timestamp}.json"
    )
    
    # Convert numpy values to native Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj
    
    json_results = {k: convert_for_json(v) if not isinstance(v, dict) else 
                   {sk: convert_for_json(sv) for sk, sv in v.items()} 
                   for k, v in eval_results.items()}
    
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # Print results if verbose
    if verbose:
        print(f"\nEvaluation Results for {model.model_type} model:")
        print(f"Dataset: {dataset_name}, Feature type: {feature_type}")
        print(f"Number of samples: {len(y)}, Number of features: {X.shape[1]}")
        print("\nOverall Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nClass-wise Metrics:")
        for class_label, metrics_dict in class_metrics.items():
            print(f"  Class {class_label}:")
            for metric, value in metrics_dict.items():
                print(f"    {metric}: {value:.4f}" if metric != "support" else f"    {metric}: {value}")
    
    logger.info(f"Saved comprehensive evaluation results to {results_path}")
    
    return eval_results


def compare_models(
    model_paths: List[str],
    dataset_name: Optional[str] = None,
    feature_type: Optional[str] = None,
    target_col: str = "result",
    generate_plots: bool = True
) -> Dict[str, Any]:
    """
    Compare multiple models on the same dataset.
    
    Args:
        model_paths: List of paths to trained models
        dataset_name: Name of the dataset to use (if None, use the one from the first model)
        feature_type: Type of features to use (if None, use the one from the first model)
        target_col: Name of the target column
        generate_plots: Whether to generate comparison plots
        
    Returns:
        Dict[str, Any]: Comparison results
    """
    if not model_paths:
        raise ValueError("No models provided for comparison")
    
    # Load all models
    models = []
    for path in model_paths:
        try:
            model = BaselineMatchPredictor.load(path)
            models.append(model)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
    
    if not models:
        raise ValueError("No models could be loaded for comparison")
    
    # Use dataset and feature type from the first model if not provided
    dataset_name = dataset_name or models[0].dataset_name
    feature_type = feature_type or models[0].feature_type
    
    # Load data
    try:
        df = load_feature_data(dataset_name, feature_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading feature data: {e}")
        raise
    
    # Process data with the first model's pipeline
    X, y = models[0].process_data(df, target_col=target_col)
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Collect predictions and evaluation metrics for each model
    model_results = []
    for model in models:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision_macro": precision_score(y, y_pred, average="macro"),
            "recall_macro": recall_score(y, y_pred, average="macro"),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "precision_weighted": precision_score(y, y_pred, average="weighted"),
            "recall_weighted": recall_score(y, y_pred, average="weighted"),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "log_loss": log_loss(y, y_proba)
        }
        
        model_results.append({
            "model_type": model.model_type,
            "model_info": model.model_info,
            "metrics": metrics,
            "predictions": y_pred,
            "probabilities": y_proba
        })
    
    # Timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate comparison plots if requested
    if generate_plots:
        # Create directory for this comparison
        plot_dir = os.path.join(PLOTS_DIR, f"comparison_{dataset_name}_{feature_type}_{timestamp}")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Performance metrics comparison bar chart
        metrics_to_compare = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
        metrics_data = []
        
        for model_result in model_results:
            for metric in metrics_to_compare:
                metrics_data.append({
                    "Model": model_result["model_type"],
                    "Metric": metric,
                    "Value": model_result["metrics"][metric]
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(x="Metric", y="Value", hue="Model", data=metrics_df)
        plt.title("Performance Metrics Comparison")
        plt.ylim(0, 1)
        plt.tight_layout()
        metrics_comparison_path = os.path.join(plot_dir, "metrics_comparison.png")
        plt.savefig(metrics_comparison_path)
        plt.close()
        
        # ROC curve comparison (if applicable)
        classes = np.unique(y)
        class_indices = {label: i for i, label in enumerate(classes)}
        
        if len(classes) >= 3:
            # For each class, compare ROC curves
            for i, class_label in enumerate(classes):
                plt.figure(figsize=(10, 8))
                for model_result in model_results:
                    model_name = model_result["model_type"]
                    y_proba = model_result["probabilities"]
                    
                    mask = (y == class_label)
                    fpr, tpr, _ = roc_curve(mask, y_proba[:, i])
                    auc = roc_auc_score(mask, y_proba[:, i])
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve Comparison - Class {class_label}')
                plt.legend(loc='best')
                class_roc_path = os.path.join(plot_dir, f"roc_curve_class_{class_label}.png")
                plt.savefig(class_roc_path)
                plt.close()
        
        # Precision-Recall curve comparison
        for i, class_label in enumerate(classes):
            plt.figure(figsize=(10, 8))
            for model_result in model_results:
                model_name = model_result["model_type"]
                y_proba = model_result["probabilities"]
                
                mask = (y == class_label)
                precision, recall, _ = precision_recall_curve(mask, y_proba[:, i])
                ap = average_precision_score(mask, y_proba[:, i])
                plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.2f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve Comparison - Class {class_label}')
            plt.legend(loc='best')
            class_pr_path = os.path.join(plot_dir, f"pr_curve_class_{class_label}.png")
            plt.savefig(class_pr_path)
            plt.close()
        
        plot_paths = {
            "metrics_comparison": metrics_comparison_path,
            "roc_curves": [os.path.join(plot_dir, f"roc_curve_class_{class_label}.png") for class_label in classes] if len(classes) >= 3 else None,
            "pr_curves": [os.path.join(plot_dir, f"pr_curve_class_{class_label}.png") for class_label in classes]
        }
    else:
        plot_paths = None
    
    # Create comparison results
    comparison_results = {
        "dataset_name": dataset_name,
        "feature_type": feature_type,
        "target_col": target_col,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "model_results": [
            {
                "model_type": result["model_type"],
                "metrics": result["metrics"]
            }
            for result in model_results
        ],
        "plot_paths": plot_paths,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save comparison results
    results_path = os.path.join(
        EVALUATION_DIR,
        f"model_comparison_{dataset_name}_{feature_type}_{timestamp}.json"
    )
    
    # Convert numpy values to native Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj
    
    # Recursively convert all values in the dictionary
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(item) for item in d]
        else:
            return convert_for_json(d)
    
    json_results = convert_dict(comparison_results)
    
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # Print summary of comparison
    print("\nModel Comparison Results:")
    print(f"Dataset: {dataset_name}, Feature type: {feature_type}")
    print(f"Number of models compared: {len(models)}")
    print("\nPerformance Metrics:")
    
    # Create a formatted table for display
    metric_names = ["Accuracy", "Precision (w)", "Recall (w)", "F1 (w)", "Log Loss"]
    metric_keys = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "log_loss"]
    
    # Print header
    print(f"{'Model':<15}", end="")
    for name in metric_names:
        print(f"{name:<15}", end="")
    print()
    
    # Print separator
    print("-" * (15 * (len(metric_names) + 1)))
    
    # Print metrics for each model
    for result in model_results:
        model_name = result["model_type"]
        print(f"{model_name:<15}", end="")
        
        for key in metric_keys:
            value = result["metrics"][key]
            # Format log_loss differently since it's not bounded between 0 and 1
            if key == "log_loss":
                print(f"{value:<15.4f}", end="")
            else:
                print(f"{value:<15.4f}", end="")
        print()
    
    logger.info(f"Saved model comparison results to {results_path}")
    
    return comparison_results


def analyze_feature_importance(
    model_path: str,
    dataset_name: Optional[str] = None,
    feature_type: Optional[str] = None,
    target_col: str = "result",
    n_top_features: int = 20
) -> Dict[str, Any]:
    """
    Analyze feature importance for a trained model.
    
    Args:
        model_path: Path to the saved model
        dataset_name: Name of the dataset to use (if None, use the one from the model)
        feature_type: Type of features to use (if None, use the one from the model)
        target_col: Name of the target column
        n_top_features: Number of top features to analyze
        
    Returns:
        Dict[str, Any]: Feature importance analysis results
    """
    # Load the model
    try:
        model = BaselineMatchPredictor.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise
    
    # Check if model supports feature importance
    if not hasattr(model.model, "feature_importances_") and not hasattr(model.model, "coef_"):
        raise ValueError(f"Model type {model.model_type} does not support feature importance analysis")
    
    # Use dataset and feature type from model if not provided
    dataset_name = dataset_name or model.dataset_name
    feature_type = feature_type or model.feature_type
    
    # Load data to get feature names
    try:
        df = load_feature_data(dataset_name, feature_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading feature data: {e}")
        raise
    
    # Process data
    X, y = model.process_data(df, target_col=target_col)
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Get feature names from pipeline
    try:
        # Try to get feature names from column transformer
        feature_names = model.pipeline.named_steps['preprocessor'].get_feature_names_out()
    except:
        # Fallback to using column indices
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Get feature importances
    if hasattr(model.model, "feature_importances_"):
        # For tree-based models
        importances = model.model.feature_importances_
    elif hasattr(model.model, "coef_"):
        # For linear models
        if len(model.model.coef_.shape) > 1:
            # Multiclass case - use norm of coefficients
            importances = np.linalg.norm(model.model.coef_, axis=0)
        else:
            # Binary case
            importances = np.abs(model.model.coef_)
    else:
        raise ValueError("Could not extract feature importances from the model")
    
    # Combine feature names and importances
    feature_importance = list(zip(feature_names, importances))
    
    # Sort by importance (descending)
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Take top N features
    top_features = feature_importance[:n_top_features]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=[imp for _, imp in top_features], y=[name for name, _ in top_features])
    plt.title(f"Top {n_top_features} Feature Importances - {model.model_type}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join(PLOTS_DIR, f"feature_importance_{model.model_type}_{timestamp}")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "feature_importance.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Create results
    results = {
        "model_type": model.model_type,
        "dataset_name": dataset_name,
        "feature_type": feature_type,
        "n_features": X.shape[1],
        "top_features": [
            {"name": name, "importance": float(imp)} 
            for name, imp in top_features
        ],
        "all_features": [
            {"name": name, "importance": float(imp)} 
            for name, imp in feature_importance
        ],
        "plot_path": plot_path,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    results_path = os.path.join(
        EVALUATION_DIR,
        f"feature_importance_{model.model_type}_{dataset_name}_{feature_type}_{timestamp}.json"
    )
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print top features
    print(f"\nTop {n_top_features} Features for {model.model_type} model:")
    for i, (name, importance) in enumerate(top_features, 1):
        print(f"{i}. {name}: {importance:.4f}")
    
    logger.info(f"Saved feature importance analysis to {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and visualize soccer prediction models")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained model file")
    parser.add_argument("--compare", type=str, nargs="+", default=None,
                       help="Paths to multiple models for comparison")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset to use (default: use dataset from model)")
    parser.add_argument("--feature-type", type=str, default=None,
                       help="Type of features to use (default: use feature type from model)")
    parser.add_argument("--target-col", type=str, default="result",
                       help="Name of the target column (default: result)")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable generation of plots")
    parser.add_argument("--feature-importance", action="store_true",
                       help="Analyze feature importance")
    parser.add_argument("--top-features", type=int, default=20,
                       help="Number of top features to show (default: 20)")
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Compare multiple models
            model_paths = [args.model_path] + args.compare
            compare_models(
                model_paths=model_paths,
                dataset_name=args.dataset,
                feature_type=args.feature_type,
                target_col=args.target_col,
                generate_plots=not args.no_plots
            )
        elif args.feature_importance:
            # Analyze feature importance
            analyze_feature_importance(
                model_path=args.model_path,
                dataset_name=args.dataset,
                feature_type=args.feature_type,
                target_col=args.target_col,
                n_top_features=args.top_features
            )
        else:
            # Evaluate a single model
            evaluate_model_performance(
                model_path=args.model_path,
                dataset_name=args.dataset,
                feature_type=args.feature_type,
                target_col=args.target_col,
                generate_plots=not args.no_plots,
                verbose=True
            )
    except Exception as e:
        logger.error(f"Error in evaluation script: {e}")
        raise 