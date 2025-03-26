"""
Training module for soccer prediction models.
Handles the training and evaluation of various models.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import joblib

# Import project components
from src.utils.logger import get_logger
from src.models.baseline import BaselineMatchPredictor
try:
    # Import soccer-specific models
    from src.models.soccer_distributions import DixonColesModel, train_dixon_coles_model
except ImportError:
    pass

try:
    # Import advanced soccer features
    from src.data.soccer_features import load_or_create_advanced_features
except ImportError:
    pass

try:
    from config.default_config import DATA_DIR, DEFAULT_MODEL_PARAMS
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")
    DEFAULT_MODEL_PARAMS = {
        "logistic": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "multi_class": "multinomial",
            "solver": "lbfgs"
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced"
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss"
        },
        "dixon_coles": {
            "match_weight_days": 90  # Half-life in days for time-weighting matches
        }
    }

# Setup logger
logger = get_logger("models.training")

# Define paths
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODELS_DIR = os.path.join(DATA_DIR, "models")
TRAINING_DIR = os.path.join(DATA_DIR, "training")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)


def load_feature_data(dataset_name: str, feature_type: str) -> pd.DataFrame:
    """
    Load feature data from a CSV file.
    
    Args:
        dataset_name: Name of the dataset
        feature_type: Type of features to load
        
    Returns:
        pd.DataFrame: Loaded feature data
    """
    features_path = os.path.join(FEATURES_DIR, dataset_name, f"{feature_type}.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    
    df = pd.read_csv(features_path)
    logger.info(f"Loaded feature dataset with {len(df)} samples")
    return df


def load_advanced_soccer_features(dataset_name: str) -> pd.DataFrame:
    """
    Load or create advanced soccer features.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        pd.DataFrame: DataFrame with advanced soccer features
    """
    try:
        # Try to import soccer features module
        from src.data.soccer_features import load_or_create_advanced_features
        
        # Check if processed data directory exists
        processed_dir = os.path.join(DATA_DIR, "processed", dataset_name)
        
        if not os.path.exists(processed_dir):
            logger.warning(f"Processed data directory not found: {processed_dir}")
            return pd.DataFrame()
        
        # Find match data file
        match_files = [f for f in os.listdir(processed_dir) if 'match' in f.lower() or 'game' in f.lower()]
        
        if not match_files:
            logger.warning(f"No match data files found in {processed_dir}")
            return pd.DataFrame()
        
        # Load match data
        match_file = match_files[0]
        match_path = os.path.join(processed_dir, match_file)
        matches_df = pd.read_csv(match_path)
        
        # Find shot data file if available
        shot_files = [f for f in os.listdir(processed_dir) if 'shot' in f.lower()]
        shots_df = None
        
        if shot_files:
            shot_file = shot_files[0]
            shot_path = os.path.join(processed_dir, shot_file)
            shots_df = pd.read_csv(shot_path)
        
        # Create advanced features
        features_df = load_or_create_advanced_features(matches_df, shots_df)
        
        # Save features
        advanced_features_dir = os.path.join(FEATURES_DIR, dataset_name)
        os.makedirs(advanced_features_dir, exist_ok=True)
        
        features_path = os.path.join(advanced_features_dir, "advanced_soccer_features.csv")
        features_df.to_csv(features_path, index=False)
        
        logger.info(f"Created and saved advanced soccer features for {dataset_name}")
        
        return features_df
    
    except ImportError:
        logger.warning("Soccer features module not available")
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error creating advanced soccer features: {e}")
        return pd.DataFrame()


def train_model(
    model_type: str = "logistic",
    dataset_name: str = "transfermarkt",
    feature_type: str = "match_features",
    target_col: str = "result",
    test_size: float = 0.2,
    cv_folds: int = 5,
    hyperparameter_tuning: bool = False,
    random_state: int = 42,
    model_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train a model with optional hyperparameter tuning.
    
    Args:
        model_type: Type of model to train
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        target_col: Name of the target column
        test_size: Portion of data to use for testing
        cv_folds: Number of cross-validation folds
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        random_state: Random seed for reproducibility
        model_params: Optional model parameters to override defaults
        
    Returns:
        Dict[str, Any]: Training results
    """
    # Record start time
    start_time = datetime.now()
    
    # Check if it's a distribution model
    if model_type == "dixon_coles":
        return train_dixon_coles(
            dataset_name=dataset_name,
            match_weight_days=model_params.get('match_weight_days', 90) if model_params else 90
        )
    
    # For standard ML models, proceed with the original implementation
    # Load data
    try:
        df = load_feature_data(dataset_name, feature_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading feature data: {e}")
        raise
    
    # Create model
    model = BaselineMatchPredictor(model_type=model_type, dataset_name=dataset_name, feature_type=feature_type)
    
    # Load pipeline
    if not model.load_pipeline():
        raise ValueError("Failed to load feature pipeline")
    
    # Process data
    X, y = model.process_data(df, target_col=target_col)
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Hyperparameter tuning if requested
    if hyperparameter_tuning:
        logger.info(f"Performing hyperparameter tuning with {cv_folds}-fold cross-validation")
        
        # Define parameter grid based on model type
        if model_type == "logistic":
            param_grid = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "class_weight": ["balanced", None],
                "solver": ["lbfgs", "newton-cg", "sag"]
            }
        elif model_type == "random_forest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        elif model_type == "xgboost":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        else:
            raise ValueError(f"Unsupported model type for tuning: {model_type}")
        
        # Create the base model
        model._create_model()
        
        # Create cross-validator
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Create grid search
        grid_search = GridSearchCV(
            model.model,
            param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        logger.info(f"Best parameters: {best_params}")
        
        # Use best estimator
        model.model = grid_search.best_estimator_
        
        # Record hyperparameter tuning results
        tuning_results = {
            "best_params": best_params,
            "best_score": grid_search.best_score_,
            "cv_results": {
                "mean_test_score": grid_search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": grid_search.cv_results_["std_test_score"].tolist(),
                "params": [str(p) for p in grid_search.cv_results_["params"]]
            }
        }
        
        # Save tuning results
        tuning_path = os.path.join(
            TRAINING_DIR, 
            f"{dataset_name}_{feature_type}_{model_type}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(tuning_path, 'w') as f:
            json.dump(tuning_results, f, indent=4)
        
        logger.info(f"Hyperparameter tuning results saved to {tuning_path}")
    else:
        # Use default parameters
        model._create_model()
        
        # Apply custom parameters if provided
        if model_params:
            for param, value in model_params.items():
                if hasattr(model.model, param):
                    setattr(model.model, param, value)
    
    # Train the model
    model.train(X_train, y_train)
    
    # Evaluate on test set
    evaluation = model.evaluate(X_test, y_test)
    
    # Save the model
    model_path = model.save()
    
    # Calculate training duration
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    # Prepare results
    results = {
        "model_type": model_type,
        "dataset_name": dataset_name,
        "feature_type": feature_type,
        "target_col": target_col,
        "test_size": test_size,
        "hyperparameter_tuning": hyperparameter_tuning,
        "training_duration": training_duration,
        "model_path": model_path,
        "evaluation": evaluation
    }
    
    if hyperparameter_tuning:
        results["hyperparameter_tuning_results"] = tuning_results
    
    logger.info(f"Model training completed in {training_duration:.2f} seconds")
    logger.info(f"Test accuracy: {evaluation['accuracy']:.4f}")
    
    return results


def train_dixon_coles(dataset_name: str, match_weight_days: int = 90) -> Dict[str, Any]:
    """
    Train a Dixon-Coles soccer prediction model.
    
    Args:
        dataset_name: Name of the dataset to use
        match_weight_days: Half-life in days for time-weighting matches
        
    Returns:
        Dict[str, Any]: Training results
    """
    # Record start time
    start_time = datetime.now()
    
    # Check if required packages are available
    try:
        from src.models.soccer_distributions import DixonColesModel, train_dixon_coles_model
    except ImportError:
        logger.error("Soccer distributions module not available")
        return {
            "success": False,
            "message": "Soccer distributions module not available. Make sure src/models/soccer_distributions.py exists."
        }
    
    # Find match data
    processed_dir = os.path.join(DATA_DIR, "processed", dataset_name)
    
    if not os.path.exists(processed_dir):
        logger.error(f"Processed data directory not found: {processed_dir}")
        return {
            "success": False,
            "message": f"Processed data directory not found: {processed_dir}"
        }
    
    # Find match data file
    match_files = [f for f in os.listdir(processed_dir) if 'match' in f.lower() or 'game' in f.lower()]
    
    if not match_files:
        logger.error(f"No match data files found in {processed_dir}")
        return {
            "success": False,
            "message": f"No match data files found in {processed_dir}"
        }
    
    # Load match data
    match_file = match_files[0]
    match_path = os.path.join(processed_dir, match_file)
    matches_df = pd.read_csv(match_path)
    
    # Check required columns
    required_columns = [
        ('home_team', ['home_club_id', 'home_team_id', 'home_id']),
        ('away_team', ['away_club_id', 'away_team_id', 'away_id']),
        ('home_goals', ['home_club_goals', 'home_score', 'home_team_goals']),
        ('away_goals', ['away_club_goals', 'away_score', 'away_team_goals']),
        ('date', ['match_date', 'game_date'])
    ]
    
    column_mapping = {}
    
    for req_col, alternatives in required_columns:
        if req_col in matches_df.columns:
            column_mapping[req_col] = req_col
        else:
            for alt_col in alternatives:
                if alt_col in matches_df.columns:
                    column_mapping[req_col] = alt_col
                    break
            
            if req_col not in column_mapping:
                logger.error(f"Could not find required column {req_col} or alternatives {alternatives}")
                return {
                    "success": False,
                    "message": f"Could not find required column {req_col} or alternatives {alternatives}"
                }
    
    # Create a copy of the dataframe with correct column names
    matches = matches_df.copy()
    for req_col, actual_col in column_mapping.items():
        if req_col != actual_col:
            matches[req_col] = matches_df[actual_col]
    
    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_dtype(matches['date']):
        matches['date'] = pd.to_datetime(matches['date'])
    
    # Train the model
    logger.info(f"Training Dixon-Coles model on {len(matches)} matches with match_weight_days={match_weight_days}")
    model = DixonColesModel()
    result = model.fit(matches)
    
    if not result.get('success', False):
        logger.error(f"Failed to train Dixon-Coles model: {result.get('message', 'Unknown error')}")
        return result
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, "distributions", f"dixon_coles_{dataset_name}_{timestamp}.joblib")
    model.save(model_path)
    
    # Calculate training duration
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    
    # Create team ratings table
    team_ratings = model.get_team_ratings()
    ratings_df = pd.DataFrame([
        {"team": team, "attack": params["attack"], "defense": params["defense"], "overall": params["overall"]}
        for team, params in team_ratings.items()
    ])
    
    # Save ratings
    ratings_path = os.path.join(MODELS_DIR, "distributions", f"dixon_coles_ratings_{dataset_name}_{timestamp}.csv")
    ratings_df.to_csv(ratings_path, index=False)
    
    # Prepare results
    results = {
        "success": True,
        "model_type": "dixon_coles",
        "dataset_name": dataset_name,
        "match_weight_days": match_weight_days,
        "num_matches": len(matches),
        "num_teams": len(team_ratings),
        "training_duration": training_duration,
        "model_path": model_path,
        "ratings_path": ratings_path,
        "home_advantage": model.home_advantage,
        "rho": model.rho  # Low-score correction factor
    }
    
    logger.info(f"Dixon-Coles model training completed in {training_duration:.2f} seconds")
    logger.info(f"Model saved to {model_path}")
    
    return results


def ensemble_models(
    model_paths: List[str],
    ensemble_type: str = "voting",
    weights: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Create an ensemble of models.
    
    Args:
        model_paths: List of paths to trained models
        ensemble_type: Type of ensemble ("voting" or "stacking")
        weights: Optional weights for each model (for voting)
        
    Returns:
        Dict[str, Any]: Ensemble information
    """
    if not model_paths:
        raise ValueError("No models provided for ensemble")
    
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
        raise ValueError("No models could be loaded for ensemble")
    
    # Validate weights if provided
    if weights is not None:
        if len(weights) != len(models):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(models)})")
        # Normalize weights to sum to 1
        weights = [w / sum(weights) for w in weights]
    else:
        # Equal weights if not provided
        weights = [1.0 / len(models)] * len(models)
    
    # Create ensemble info
    ensemble_info = {
        "ensemble_type": ensemble_type,
        "models": [model.model_info for model in models],
        "weights": weights,
        "model_paths": model_paths,
        "created_at": datetime.now().isoformat()
    }
    
    # Save ensemble info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_path = os.path.join(
        MODELS_DIR,
        f"ensemble_{ensemble_type}_{timestamp}.json"
    )
    with open(ensemble_path, "w") as f:
        json.dump(ensemble_info, f, indent=2, default=str)
    
    logger.info(f"Created {ensemble_type} ensemble with {len(models)} models")
    logger.info(f"Saved ensemble info to {ensemble_path}")
    
    return ensemble_info


def predict_with_ensemble(
    ensemble_path: str,
    home_team_id: int,
    away_team_id: int,
    features: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make a prediction using an ensemble of models.
    
    Args:
        ensemble_path: Path to the ensemble info file
        home_team_id: ID of the home team
        away_team_id: ID of the away team
        features: Optional dictionary of additional features
        
    Returns:
        Dict[str, Any]: Prediction results
    """
    # Load ensemble info
    with open(ensemble_path, "r") as f:
        ensemble_info = json.load(f)
    
    ensemble_type = ensemble_info["ensemble_type"]
    model_paths = ensemble_info["model_paths"]
    weights = ensemble_info["weights"]
    
    # Load all models
    models = []
    for path in model_paths:
        try:
            model = BaselineMatchPredictor.load(path)
            models.append(model)
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
    
    if not models:
        raise ValueError("No models could be loaded for ensemble prediction")
    
    # Get predictions from all models
    predictions = []
    for model in models:
        try:
            pred = model.predict_match(home_team_id, away_team_id, features)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error predicting with model: {e}")
    
    if not predictions:
        raise ValueError("No predictions could be made with the ensemble")
    
    # Combine predictions based on ensemble type
    if ensemble_type == "voting":
        # Weighted average of probabilities
        home_win_prob = sum(pred["home_win_probability"] * weight for pred, weight in zip(predictions, weights))
        draw_prob = sum(pred["draw_probability"] * weight for pred, weight in zip(predictions, weights))
        away_win_prob = sum(pred["away_win_probability"] * weight for pred, weight in zip(predictions, weights))
        
        # Normalize probabilities
        total_prob = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total_prob
        draw_prob /= total_prob
        away_win_prob /= total_prob
        
        # Determine prediction
        if home_win_prob >= draw_prob and home_win_prob >= away_win_prob:
            prediction = "home_win"
            confidence = home_win_prob
        elif away_win_prob >= home_win_prob and away_win_prob >= draw_prob:
            prediction = "away_win"
            confidence = away_win_prob
        else:
            prediction = "draw"
            confidence = draw_prob
        
        # Create result
        result = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_win_probability": float(home_win_prob),
            "draw_probability": float(draw_prob),
            "away_win_probability": float(away_win_prob),
            "prediction": prediction,
            "confidence": float(confidence),
            "predicted_at": datetime.now().isoformat(),
            "ensemble_type": ensemble_type,
            "n_models": len(models),
            "individual_predictions": [
                {
                    "model_type": pred["model_type"],
                    "prediction": pred["prediction"],
                    "confidence": pred["confidence"]
                }
                for pred in predictions
            ]
        }
    else:
        raise ValueError(f"Unsupported ensemble type: {ensemble_type}")
    
    return result


def evaluate_model_file(
    model_path: str,
    dataset_name: Optional[str] = None,
    feature_type: Optional[str] = None,
    target_col: str = "result",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluate a saved model on a dataset.
    
    Args:
        model_path: Path to the saved model
        dataset_name: Name of the dataset to use (if None, use the one from the model)
        feature_type: Type of features to use (if None, use the one from the model)
        target_col: Name of the target column
        test_size: Portion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dict[str, Any]: Evaluation results
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
    
    # Split data
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Evaluate model
    eval_result = model.evaluate(X_test, y_test)
    logger.info(f"Model evaluation: accuracy={eval_result['accuracy']:.4f}, f1={eval_result['f1']:.4f}")
    
    # Create and return evaluation results
    evaluation_results = {
        "model_path": model_path,
        "model_type": model.model_type,
        "dataset_name": dataset_name,
        "feature_type": feature_type,
        "target_col": target_col,
        "test_size": test_size,
        "n_test_samples": X_test.shape[0],
        "performance": eval_result,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save evaluation results
    results_path = os.path.join(
        TRAINING_DIR,
        f"evaluation_{dataset_name}_{feature_type}_{model.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_path, "w") as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    logger.info(f"Saved evaluation results to {results_path}")
    
    return evaluation_results


def train_multiple_models(
    model_types: List[str] = ["logistic", "random_forest", "xgboost"],
    dataset_name: str = "transfermarkt",
    feature_type: str = "match_features",
    create_ensemble: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Train multiple models and optionally create an ensemble.
    
    Args:
        model_types: List of model types to train
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        create_ensemble: Whether to create an ensemble of the trained models
        **kwargs: Additional arguments for train_model
        
    Returns:
        Dict[str, Any]: Training results
    """
    results = {}
    model_paths = []
    
    # Train each model type
    for model_type in model_types:
        try:
            logger.info(f"Training {model_type} model")
            result = train_model(model_type=model_type, dataset_name=dataset_name, feature_type=feature_type, **kwargs)
            results[model_type] = result
            model_paths.append(result["model_path"])
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
    
    # Create ensemble if requested and more than one model was trained
    if create_ensemble and len(model_paths) > 1:
        try:
            ensemble_info = ensemble_models(model_paths)
            results["ensemble"] = ensemble_info
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate soccer prediction models")
    parser.add_argument("--model-type", type=str, nargs="+", default=["logistic", "random_forest", "xgboost"],
                        help="Type(s) of model to train")
    parser.add_argument("--dataset", type=str, default="transfermarkt",
                        help="Dataset to use (default: transfermarkt)")
    parser.add_argument("--feature-type", type=str, default="match_features",
                        help="Type of features to use (default: match_features)")
    parser.add_argument("--target-col", type=str, default="result",
                        help="Name of the target column (default: result)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Portion of data to use for testing (default: 0.2)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--tune", action="store_true",
                        help="Perform hyperparameter tuning")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Do not create an ensemble of the trained models")
    parser.add_argument("--evaluate", type=str, default=None,
                        help="Evaluate a saved model instead of training")
    
    args = parser.parse_args()
    
    try:
        if args.evaluate:
            # Evaluate a saved model
            evaluate_model_file(
                model_path=args.evaluate,
                dataset_name=args.dataset,
                feature_type=args.feature_type,
                target_col=args.target_col,
                test_size=args.test_size
            )
        else:
            # Train models
            train_multiple_models(
                model_types=args.model_type,
                dataset_name=args.dataset,
                feature_type=args.feature_type,
                target_col=args.target_col,
                test_size=args.test_size,
                cv_folds=args.cv_folds,
                hyperparameter_tuning=args.tune,
                create_ensemble=not args.no_ensemble
            )
    except Exception as e:
        logger.error(f"Error in training script: {e}")
        raise 