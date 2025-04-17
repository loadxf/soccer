"""
Enhanced Soccer Prediction Pipeline

This script provides a unified pipeline to run the enhanced soccer prediction
improvements, including:
- Advanced feature engineering
- Feature selection
- Model hyperparameter optimization
- Model calibration
- Ensemble modeling with dynamic weighting and context awareness

Usage:
    python -m src.models.enhance_soccer_predictions --dataset <dataset_name> --mode <train|evaluate|predict>
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhance_soccer_predictions.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("enhance_soccer_predictions")

# Import project components
try:
    from src.data.advanced_features import create_all_advanced_features, calculate_team_differential_features
    from src.models.hyperopt import (
        optimize_hyperparameters_with_feature_selection,
        optimize_calibration_method,
        optimize_ensemble_weights
    )
    from src.models.calibration import ProbabilityCalibrator
    from src.models.ensemble import EnsemblePredictor
    from src.models.baseline import BaselineMatchPredictor
    from src.models.advanced import AdvancedMatchPredictor
    from src.models.time_series import TimeSeriesPredictor
    from src.models.soccer_distributions import DixonColesModel
    from src.data.features import load_feature_pipeline
    from src.models.training import load_feature_data, train_multiple_models
    from src.models.evaluation import evaluate_model_performance
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you've implemented all the required modules first.")
    sys.exit(1)

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Define paths
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
ENHANCED_DIR = os.path.join(MODELS_DIR, "enhanced")
os.makedirs(ENHANCED_DIR, exist_ok=True)


def load_and_prepare_data(
    dataset_name: str,
    feature_type: str = "match_features", 
    target_col: str = "result",
    use_advanced_features: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Load and prepare data for enhanced soccer prediction.
    
    Args:
        dataset_name: Name of the dataset
        feature_type: Type of features to use
        target_col: Name of the target column
        use_advanced_features: Whether to use advanced features
        test_size: Size of the test set
        random_state: Random seed
        
    Returns:
        Dictionary with prepared data
    """
    logger.info(f"Loading data from {dataset_name} with feature type {feature_type}")
    
    # Load feature data
    df = load_feature_data(dataset_name, feature_type)
    
    # If using advanced features, apply advanced feature engineering
    if use_advanced_features:
        logger.info("Enhancing features with advanced feature engineering")
        
        # Apply team differential features
        df = calculate_team_differential_features(df)
        
        # Try to load and apply more advanced features if available
        try:
            # Look for player data
            player_data_path = os.path.join(DATA_DIR, "processed", dataset_name, "players.csv")
            player_data = pd.read_csv(player_data_path) if os.path.exists(player_data_path) else None
            
            # Look for league standings data
            standings_path = os.path.join(DATA_DIR, "processed", dataset_name, "standings.csv")
            standings = pd.read_csv(standings_path) if os.path.exists(standings_path) else None
            
            # Apply all advanced features
            df = create_all_advanced_features(df, player_data, standings)
            
            logger.info(f"Advanced features added. Total features: {len(df.columns)}")
        except Exception as e:
            logger.warning(f"Couldn't apply all advanced features: {e}")
            logger.warning("Proceeding with basic differential features only")
    
    # Prepare data for modeling
    from sklearn.model_selection import train_test_split
    
    # Save column names for feature selection
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Convert target to numeric if needed
    if target_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            # For classification, convert target to numeric
            target_map = {val: i for i, val in enumerate(df[target_col].unique())}
            df[target_col] = df[target_col].map(target_map)
            logger.info(f"Converted target to numeric. Target mapping: {target_map}")
    
    # Split into train and test
    if test_size > 0:
        # Sort by date if available for temporal validation
        if 'match_date' in df.columns and pd.api.types.is_datetime64_dtype(df['match_date']):
            df = df.sort_values('match_date')
            # Use temporal split (last n% as test)
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Convert to numpy arrays
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values
    else:
        # Use all data
        X_train = df[feature_cols].values
        y_train = df[target_col].values
        X_test = None
        y_test = None
        train_df = df
        test_df = None
    
    # Return prepared data
    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df
    }


def train_enhanced_soccer_models(
    dataset_name: str,
    feature_type: str = "match_features",
    target_col: str = "result",
    n_trials: int = 50,
    model_types: List[str] = ["lightgbm", "neural_network", "dixon_coles"],
    calibration: bool = True,
    create_ensemble: bool = True,
    ensemble_type: str = "context_aware",
    save_models: bool = True,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train enhanced soccer prediction models with optimized features and hyperparameters.
    
    Args:
        dataset_name: Name of the dataset
        feature_type: Type of features to use
        target_col: Name of the target column
        n_trials: Number of optimization trials
        model_types: Types of models to train
        calibration: Whether to calibrate model probabilities
        create_ensemble: Whether to create an ensemble
        ensemble_type: Type of ensemble
        save_models: Whether to save models
        random_state: Random seed
        
    Returns:
        Dictionary with trained models and results
    """
    logger.info(f"Training enhanced soccer models for {dataset_name}")
    
    # Load and prepare data
    data = load_and_prepare_data(
        dataset_name=dataset_name,
        feature_type=feature_type,
        target_col=target_col,
        use_advanced_features=True,
        test_size=0.2,
        random_state=random_state
    )
    
    # Initialize results dictionary
    results = {"models": {}, "feature_importance": {}, "calibrators": {}}
    model_paths = []
    
    # Train models with feature selection and hyperparameter optimization
    for model_type in model_types:
        logger.info(f"Training {model_type} model with optimized features and hyperparameters")
        
        # Get model class
        if model_type in ["lightgbm", "neural_network", "catboost"]:
            ModelClass = AdvancedMatchPredictor
        elif model_type in ["logistic", "random_forest", "xgboost"]:
            ModelClass = BaselineMatchPredictor
        elif model_type == "dixon_coles":
            ModelClass = DixonColesModel
        else:
            logger.warning(f"Unsupported model type: {model_type}. Skipping.")
            continue
        
        try:
            # Optimize hyperparameters with feature selection
            optimization_results = optimize_hyperparameters_with_feature_selection(
                model_type=model_type,
                model_class=ModelClass,
                X=data["X_train"],
                y=data["y_train"],
                feature_names=data["feature_cols"],
                cv_type="time",
                n_trials=n_trials,
                scoring="f1_weighted",
                cv_folds=5,
                random_state=random_state
            )
            
            # Get best model and selected features
            best_model = optimization_results["best_model"]
            best_params = optimization_results["best_params"]
            selected_features = optimization_results["selected_features"]
            
            # Store results
            results["models"][model_type] = {
                "best_model": best_model,
                "best_params": best_params,
                "best_score": optimization_results["best_score"],
                "selected_features": selected_features,
                "n_features": len(selected_features)
            }
            
            # Save model if requested
            if save_models:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(ENHANCED_DIR, f"{model_type}_enhanced_{timestamp}.pkl")
                
                # Save model
                if hasattr(best_model, 'save'):
                    model_path = best_model.save(model_path)
                else:
                    import pickle
                    with open(model_path, 'wb') as f:
                        pickle.dump(best_model, f)
                
                # Store model path
                results["models"][model_type]["model_path"] = model_path
                model_paths.append(model_path)
                
                # Save feature importance separately
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = best_model.feature_importances_
                    
                    # Map importance to feature names
                    importance_dict = {selected_features[i]: float(feature_importance[i]) 
                                     for i in range(len(selected_features))}
                    
                    # Sort by importance
                    importance_dict = {k: v for k, v in sorted(importance_dict.items(), 
                                                            key=lambda item: item[1], reverse=True)}
                    
                    # Store feature importance
                    results["feature_importance"][model_type] = importance_dict
            
            # Evaluate model on test data if available
            if data["X_test"] is not None and data["y_test"] is not None:
                # Select features for test data
                feature_indices = [i for i, name in enumerate(data["feature_cols"]) if name in selected_features]
                X_test_selected = data["X_test"][:, feature_indices]
                
                # Make predictions
                y_pred = best_model.predict(X_test_selected)
                y_prob = best_model.predict_proba(X_test_selected)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, f1_score, log_loss
                accuracy = accuracy_score(data["y_test"], y_pred)
                f1 = f1_score(data["y_test"], y_pred, average='weighted')
                loss = log_loss(data["y_test"], y_prob)
                
                # Store metrics
                results["models"][model_type]["test_metrics"] = {
                    "accuracy": float(accuracy),
                    "f1_score": float(f1),
                    "log_loss": float(loss)
                }
                
                # Calibrate probabilities if requested
                if calibration:
                    try:
                        logger.info(f"Calibrating {model_type} model probabilities")
                        
                        # Split test data for calibration
                        from sklearn.model_selection import train_test_split
                        X_cal, X_eval, y_cal, y_eval = train_test_split(
                            X_test_selected, data["y_test"], test_size=0.5, random_state=random_state
                        )
                        
                        # Optimize calibration method
                        calibration_results = optimize_calibration_method(
                            model_path=model_path,
                            X_cal=X_cal,
                            y_cal=y_cal,
                            n_trials=10,
                            random_state=random_state
                        )
                        
                        # Store calibration results
                        results["calibrators"][model_type] = {
                            "best_method": calibration_results["best_method"],
                            "log_loss_improvement": calibration_results["results"]["improvement"],
                            "calibrator_path": calibration_results["calibrator_path"]
                        }
                        
                        # Evaluate calibrated probabilities on evaluation data
                        best_calibrator = calibration_results["best_calibrator"]
                        
                        # Get raw probabilities
                        eval_probs = best_model.predict_proba(X_eval)
                        
                        # Calibrate probabilities
                        calibrated_probs = best_calibrator.calibrate(eval_probs)
                        
                        # Calculate log loss
                        orig_loss = log_loss(y_eval, eval_probs)
                        cal_loss = log_loss(y_eval, calibrated_probs)
                        
                        # Store metrics
                        results["calibrators"][model_type]["eval_metrics"] = {
                            "original_log_loss": float(orig_loss),
                            "calibrated_log_loss": float(cal_loss),
                            "improvement": float(orig_loss - cal_loss)
                        }
                    except Exception as e:
                        logger.error(f"Failed to calibrate {model_type} model: {e}")
        
        except Exception as e:
            logger.error(f"Failed to train {model_type} model: {e}")
    
    # Create ensemble if requested
    if create_ensemble and len(model_paths) >= 2:
        try:
            logger.info(f"Creating {ensemble_type} ensemble with {len(model_paths)} models")
            
            # Optimize ensemble weights
            ensemble_results = optimize_ensemble_weights(
                model_paths=model_paths,
                X=data["X_train"],
                y=data["y_train"],
                ensemble_type=ensemble_type,
                n_trials=20,
                cv_type="time",
                cv_folds=5,
                random_state=random_state
            )
            
            # Store ensemble results
            results["ensemble"] = {
                "ensemble_type": ensemble_type,
                "best_params": ensemble_results["best_params"],
                "best_score": ensemble_results["best_score"],
                "model_count": len(model_paths),
                "ensemble_path": ensemble_results["ensemble_path"]
            }
            
            # Evaluate ensemble on test data if available
            if data["X_test"] is not None and data["y_test"] is not None:
                ensemble = ensemble_results["ensemble"]
                
                # Make predictions
                y_pred = ensemble.predict(data["X_test"])
                y_prob = ensemble.predict_proba(data["X_test"])
                
                # Calculate metrics
                accuracy = accuracy_score(data["y_test"], y_pred)
                f1 = f1_score(data["y_test"], y_pred, average='weighted')
                loss = log_loss(data["y_test"], y_prob)
                
                # Store metrics
                results["ensemble"]["test_metrics"] = {
                    "accuracy": float(accuracy),
                    "f1_score": float(f1),
                    "log_loss": float(loss)
                }
        except Exception as e:
            logger.error(f"Failed to create ensemble: {e}")
    
    return results


def evaluate_enhanced_models(
    model_paths: List[str],
    dataset_name: str,
    feature_type: str = "match_features",
    target_col: str = "result",
    include_calibration: bool = True
) -> Dict[str, Any]:
    """
    Evaluate enhanced soccer prediction models.
    
    Args:
        model_paths: Paths to trained models
        dataset_name: Name of the dataset
        feature_type: Type of features to use
        target_col: Name of the target column
        include_calibration: Whether to include calibration in evaluation
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating enhanced models on {dataset_name}")
    
    # Load and prepare data
    data = load_and_prepare_data(
        dataset_name=dataset_name,
        feature_type=feature_type,
        target_col=target_col,
        use_advanced_features=True,
        test_size=0.0  # Use all data for evaluation
    )
    
    # Initialize results
    results = {"models": {}, "ensemble": None}
    
    # Check if model_paths is a list or a single string
    if isinstance(model_paths, str):
        model_paths = [model_paths]
    
    # Evaluate each model
    for model_path in model_paths:
        try:
            # Get model type from filename
            model_type = os.path.basename(model_path).split('_')[0]
            
            # Evaluate model
            evaluation = evaluate_model_performance(
                model_path=model_path,
                dataset_name=dataset_name,
                feature_type=feature_type,
                target_col=target_col,
                generate_plots=True
            )
            
            # Store evaluation results
            results["models"][model_type] = evaluation
            
            # Apply calibration if requested
            if include_calibration:
                # Look for calibrator with same model type
                calibrator_dir = os.path.join(DATA_DIR, "calibration")
                if os.path.exists(calibrator_dir):
                    calibrator_files = [f for f in os.listdir(calibrator_dir) 
                                     if f.startswith(f"calibrator_{model_type}") and f.endswith(".pkl")]
                    
                    if calibrator_files:
                        # Use the most recent calibrator
                        calibrator_files.sort(reverse=True)
                        calibrator_path = os.path.join(calibrator_dir, calibrator_files[0])
                        
                        try:
                            # Load calibrator
                            from src.models.calibration import ProbabilityCalibrator
                            calibrator = ProbabilityCalibrator.load(calibrator_path)
                            
                            # Load model
                            import pickle
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            
                            # Make predictions
                            y_prob = model.predict_proba(data["X_train"])
                            
                            # Calibrate probabilities
                            calibrated_prob = calibrator.calibrate(y_prob)
                            
                            # Calculate metrics
                            from sklearn.metrics import log_loss
                            orig_loss = log_loss(data["y_train"], y_prob)
                            cal_loss = log_loss(data["y_train"], calibrated_prob)
                            
                            # Store calibration results
                            results["models"][model_type]["calibration"] = {
                                "method": calibrator.method,
                                "original_log_loss": float(orig_loss),
                                "calibrated_log_loss": float(cal_loss),
                                "improvement": float(orig_loss - cal_loss)
                            }
                        except Exception as e:
                            logger.error(f"Failed to apply calibration to {model_type} model: {e}")
        except Exception as e:
            logger.error(f"Failed to evaluate model {model_path}: {e}")
    
    # Check for ensemble models
    ensemble_dir = os.path.join(ENHANCED_DIR, "ensemble")
    if os.path.exists(ensemble_dir):
        ensemble_files = [f for f in os.listdir(ensemble_dir) if f.endswith(".pkl")]
        if ensemble_files:
            # Use the most recent ensemble
            ensemble_files.sort(reverse=True)
            ensemble_path = os.path.join(ensemble_dir, ensemble_files[0])
            
            try:
                # Load ensemble
                import pickle
                with open(ensemble_path, 'rb') as f:
                    ensemble = pickle.load(f)
                
                # Make predictions
                y_pred = ensemble.predict(data["X_train"])
                y_prob = ensemble.predict_proba(data["X_train"])
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, f1_score, log_loss
                accuracy = accuracy_score(data["y_train"], y_pred)
                f1 = f1_score(data["y_train"], y_pred, average='weighted')
                loss = log_loss(data["y_train"], y_prob)
                
                # Store ensemble evaluation
                results["ensemble"] = {
                    "ensemble_type": ensemble.ensemble_type,
                    "model_count": len(ensemble.models),
                    "metrics": {
                        "accuracy": float(accuracy),
                        "f1_score": float(f1),
                        "log_loss": float(loss)
                    }
                }
            except Exception as e:
                logger.error(f"Failed to evaluate ensemble: {e}")
    
    return results


def predict_match(
    home_team_id: int,
    away_team_id: int,
    model_path: str,
    calibrator_path: Optional[str] = None,
    features: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predict a soccer match outcome using an enhanced model.
    
    Args:
        home_team_id: ID of the home team
        away_team_id: ID of the away team
        model_path: Path to the trained model
        calibrator_path: Optional path to a probability calibrator
        features: Optional additional features
        
    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Predicting match: {home_team_id} vs {away_team_id}")
    
    # Load model
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Create feature dictionary if not provided
    if features is None:
        features = {}
    
    # Add team IDs
    features["home_team_id"] = home_team_id
    features["away_team_id"] = away_team_id
    
    # Make prediction
    if hasattr(model, 'predict_match'):
        prediction = model.predict_match(home_team_id, away_team_id, features)
    else:
        # Create feature vector
        feature_vector = np.array([[home_team_id, away_team_id] + list(features.values())])
        
        # Make prediction
        pred_class = model.predict(feature_vector)[0]
        proba = model.predict_proba(feature_vector)[0]
        
        # Format prediction
        prediction = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "result": ["home_win", "draw", "away_win"][pred_class],
            "probabilities": {
                "home_win": float(proba[0]),
                "draw": float(proba[1]),
                "away_win": float(proba[2])
            }
        }
    
    # Apply calibration if requested
    if calibrator_path and os.path.exists(calibrator_path):
        try:
            # Load calibrator
            from src.models.calibration import ProbabilityCalibrator
            calibrator = ProbabilityCalibrator.load(calibrator_path)
            
            # Get raw probabilities
            raw_proba = np.array([
                prediction["probabilities"]["home_win"],
                prediction["probabilities"]["draw"],
                prediction["probabilities"]["away_win"]
            ]).reshape(1, 3)
            
            # Calibrate probabilities
            calibrated_proba = calibrator.calibrate(raw_proba)[0]
            
            # Update prediction with calibrated probabilities
            prediction["calibrated_probabilities"] = {
                "home_win": float(calibrated_proba[0]),
                "draw": float(calibrated_proba[1]),
                "away_win": float(calibrated_proba[2])
            }
            
            # Update result based on calibrated probabilities
            max_idx = np.argmax(calibrated_proba)
            prediction["calibrated_result"] = ["home_win", "draw", "away_win"][max_idx]
            
        except Exception as e:
            logger.error(f"Failed to apply calibration: {e}")
    
    return prediction


def run_enhanced_pipeline(args):
    """Main function to run the enhanced soccer prediction pipeline."""
    if args.mode == "train":
        # Train enhanced models
        results = train_enhanced_soccer_models(
            dataset_name=args.dataset,
            feature_type=args.feature_type,
            target_col=args.target,
            n_trials=args.trials,
            model_types=args.models.split(","),
            calibration=args.calibration,
            create_ensemble=args.ensemble,
            ensemble_type=args.ensemble_type,
            save_models=True,
            random_state=args.seed
        )
        
        # Save results
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(ENHANCED_DIR, f"training_results_{timestamp}.json")
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for section, section_data in results.items():
            if section == "models":
                serializable_results[section] = {}
                for model_type, model_data in section_data.items():
                    serializable_results[section][model_type] = {
                        k: v for k, v in model_data.items() 
                        if k not in ["best_model"]
                    }
            elif section == "ensemble":
                if section_data:
                    serializable_results[section] = {
                        k: v for k, v in section_data.items()
                        if k != "ensemble"
                    }
            else:
                serializable_results[section] = section_data
        
        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Training results saved to {results_path}")
        
    elif args.mode == "evaluate":
        # Evaluate models
        if args.models:
            model_paths = args.models.split(",")
        else:
            # Find all enhanced models
            enhanced_files = [os.path.join(ENHANCED_DIR, f) for f in os.listdir(ENHANCED_DIR) 
                            if f.endswith(".pkl") and not f.startswith("ensemble")]
            model_paths = enhanced_files
        
        results = evaluate_enhanced_models(
            model_paths=model_paths,
            dataset_name=args.dataset,
            feature_type=args.feature_type,
            target_col=args.target,
            include_calibration=args.calibration
        )
        
        # Save results
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(ENHANCED_DIR, f"evaluation_results_{timestamp}.json")
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
    elif args.mode == "predict":
        # Predict a match
        if not args.home_team or not args.away_team:
            logger.error("Home team ID and away team ID are required for prediction")
            return
        
        if not args.model:
            # Find the most recent enhanced model
            enhanced_files = [f for f in os.listdir(ENHANCED_DIR) 
                            if f.endswith(".pkl") and not f.startswith("ensemble")]
            
            if not enhanced_files:
                logger.error("No enhanced models found")
                return
            
            # Use the most recent model
            enhanced_files.sort(reverse=True)
            model_path = os.path.join(ENHANCED_DIR, enhanced_files[0])
        else:
            model_path = args.model
        
        # Find a calibrator if requested
        calibrator_path = None
        if args.calibration:
            calibrator_dir = os.path.join(DATA_DIR, "calibration")
            if os.path.exists(calibrator_dir):
                calibrator_files = [f for f in os.listdir(calibrator_dir) if f.endswith(".pkl")]
                if calibrator_files:
                    # Use the most recent calibrator
                    calibrator_files.sort(reverse=True)
                    calibrator_path = os.path.join(calibrator_dir, calibrator_files[0])
        
        # Convert string features to dictionary
        features = {}
        if args.features:
            for feature_str in args.features.split(","):
                key, value = feature_str.split("=")
                # Try to convert value to numeric
                try:
                    value = float(value)
                except ValueError:
                    pass
                features[key] = value
        
        # Make prediction
        prediction = predict_match(
            home_team_id=int(args.home_team),
            away_team_id=int(args.away_team),
            model_path=model_path,
            calibrator_path=calibrator_path,
            features=features
        )
        
        # Print prediction
        import json
        print(json.dumps(prediction, indent=2))
    
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Soccer Prediction Pipeline")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "predict"],
                       help="Operation mode: train, evaluate, or predict")
    
    parser.add_argument("--dataset", type=str, default="transfermarkt",
                       help="Dataset name")
    
    parser.add_argument("--feature-type", type=str, default="match_features",
                       help="Feature type")
    
    parser.add_argument("--target", type=str, default="result",
                       help="Target column name")
    
    parser.add_argument("--models", type=str, default="lightgbm,neural_network,dixon_coles",
                       help="Comma-separated list of model types or paths")
    
    parser.add_argument("--trials", type=int, default=50,
                       help="Number of optimization trials")
    
    parser.add_argument("--calibration", action="store_true", default=True,
                       help="Whether to calibrate probabilities")
    
    parser.add_argument("--ensemble", action="store_true", default=True,
                       help="Whether to create an ensemble")
    
    parser.add_argument("--ensemble-type", type=str, default="context_aware",
                       choices=["voting", "stacking", "dynamic_weighting", "context_aware"],
                       help="Type of ensemble to create")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    parser.add_argument("--model", type=str, default=None,
                       help="Path to the model for prediction")
    
    parser.add_argument("--home-team", type=str, default=None,
                       help="Home team ID for prediction")
    
    parser.add_argument("--away-team", type=str, default=None,
                       help="Away team ID for prediction")
    
    parser.add_argument("--features", type=str, default=None,
                       help="Additional features for prediction (key1=value1,key2=value2)")
    
    args = parser.parse_args()
    
    run_enhanced_pipeline(args) 