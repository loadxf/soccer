"""
Hyperparameter optimization module for soccer prediction models.

This module implements advanced hyperparameter optimization techniques
using Bayesian optimization with libraries like Optuna.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime
from pathlib import Path
import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, log_loss

# Import project components
from src.utils.logger import get_logger
from src.models.time_validation import TimeSeriesSplit, SeasonBasedSplit, MatchDayBasedSplit

try:
    # Import sequence models
    from src.models.sequence_models import SoccerTransformerModel, SequenceDataProcessor
except ImportError:
    pass

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.hyperopt")

# Define paths
MODELS_DIR = os.path.join(DATA_DIR, "models")
HYPEROPT_DIR = os.path.join(DATA_DIR, "hyperopt")
os.makedirs(HYPEROPT_DIR, exist_ok=True)


def create_parameter_space(model_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Create search space for hyperparameter optimization.
    
    Args:
        model_type: Type of model to optimize
        
    Returns:
        Dictionary defining parameter space for the model
    """
    param_spaces = {
        "logistic": {
            "C": {"type": "loguniform", "low": 1e-3, "high": 1e3},
            "class_weight": {"type": "categorical", "choices": ["balanced", None]},
            "max_iter": {"type": "fixed", "value": 1000},
            "solver": {"type": "categorical", "choices": ["lbfgs", "newton-cg", "sag"]}
        },
        "random_forest": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 30},
            "min_samples_split": {"type": "int", "low": 2, "high": 20},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
            "bootstrap": {"type": "categorical", "choices": [True, False]},
            "class_weight": {"type": "categorical", "choices": ["balanced", "balanced_subsample", None]}
        },
        "xgboost": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "loguniform", "low": 0.005, "high": 0.3},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
            "gamma": {"type": "loguniform", "low": 1e-5, "high": 1.0}
        },
        "lightgbm": {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 15},
            "learning_rate": {"type": "loguniform", "low": 0.005, "high": 0.3},
            "num_leaves": {"type": "int", "low": 20, "high": 150},
            "min_child_samples": {"type": "int", "low": 5, "high": 100},
            "subsample": {"type": "uniform", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "uniform", "low": 0.6, "high": 1.0}
        },
        "neural_network": {
            "hidden_layer_sizes": {"type": "categorical", "choices": [(50,), (100,), (50, 50), (100, 50), (100, 100)]},
            "activation": {"type": "categorical", "choices": ["relu", "tanh"]},
            "alpha": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            "learning_rate": {"type": "categorical", "choices": ["constant", "adaptive", "invscaling"]},
            "learning_rate_init": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "max_iter": {"type": "fixed", "value": 500}
        },
        "transformer": {
            "embed_dim": {"type": "int", "low": 16, "high": 128},
            "num_heads": {"type": "int", "low": 1, "high": 4},
            "ff_dim": {"type": "int", "low": 32, "high": 256},
            "num_transformer_blocks": {"type": "int", "low": 1, "high": 4},
            "dropout_rate": {"type": "uniform", "low": 0.1, "high": 0.5},
            "l2_reg": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
            "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
            "sequence_length": {"type": "int", "low": 3, "high": 10}
        },
        "dixon_coles": {
            "match_weight_days": {"type": "int", "low": 30, "high": 180},
            "home_advantage_prior": {"type": "uniform", "low": 0.1, "high": 0.5},
            "rho_prior": {"type": "uniform", "low": -0.2, "high": 0.0}
        },
        "dynamic_ensemble": {
            "recency_factor": {"type": "uniform", "low": 0.7, "high": 0.99},
            "window_size": {"type": "int", "low": 5, "high": 20},
            "weight_smoothing": {"type": "uniform", "low": 0.0, "high": 0.3},
            "calibration_method": {"type": "categorical", "choices": ["platt", "isotonic", "beta"]}
        },
        "context_aware": {
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
            "feature_selection": {"type": "categorical", "choices": ["auto", "manual", "importance"]},
            "context_weight": {"type": "uniform", "low": 0.5, "high": 1.0}
        }
    }
    
    return param_spaces.get(model_type, {})


def configure_trial_params(trial: optuna.Trial, param_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Configure trial parameters based on the parameter space.
    
    Args:
        trial: Optuna trial object
        param_space: Parameter space definition
        
    Returns:
        Dictionary of parameters for this trial
    """
    params = {}
    
    for param_name, param_config in param_space.items():
        param_type = param_config["type"]
        
        if param_type == "fixed":
            params[param_name] = param_config["value"]
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])
        elif param_type == "int":
            params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])
        elif param_type == "uniform":
            params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"])
        elif param_type == "loguniform":
            params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=True)
    
    return params


def objective_function(
    trial: optuna.Trial,
    model_class: Any,
    param_space: Dict[str, Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    cv_splitter: Any,
    cv_groups: Optional[np.ndarray] = None,
    scoring: str = "f1_weighted",
    n_jobs: int = -1
) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        model_class: Scikit-learn compatible model class
        param_space: Parameter space definition
        X: Feature matrix
        y: Target vector
        cv_splitter: Cross-validation splitter
        cv_groups: Optional grouping for time-based CV
        scoring: Scoring method
        n_jobs: Number of parallel jobs
        
    Returns:
        Score value (higher is better)
    """
    # Configure parameters for this trial
    params = configure_trial_params(trial, param_space)
    
    # Initialize model with trial parameters
    model = model_class(**params)
    
    # Perform cross-validation
    if cv_groups is not None:
        scores = cross_val_score(model, X, y, cv=cv_splitter, groups=cv_groups, scoring=scoring, n_jobs=n_jobs)
    else:
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs)
    
    # Return mean score
    return scores.mean()


def get_cv_splitter(
    cv_type: str = "random",
    n_splits: int = 5,
    test_size: float = 0.2,
    gap: int = 0,
    date_column: str = "date",
    season_column: str = "season",
    **kwargs
) -> Union[Any, TimeSeriesSplit, SeasonBasedSplit, MatchDayBasedSplit]:
    """
    Get a cross-validation splitter based on the specified type.
    
    Args:
        cv_type: Type of cross-validation ('random', 'time', 'season', or 'matchday')
        n_splits: Number of splits (folds)
        test_size: Size of test set
        gap: Gap between train and test sets (for time-based CV)
        date_column: Name of date column (for time-based CV)
        season_column: Name of season column (for season-based CV)
        **kwargs: Additional arguments for specific CV types
        
    Returns:
        Cross-validation splitter object
    """
    from sklearn.model_selection import StratifiedKFold
    
    if cv_type == "random":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=kwargs.get('random_state', 42))
    
    elif cv_type == "time":
        return TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=kwargs.get('max_train_size', None)
        )
    
    elif cv_type == "season":
        return SeasonBasedSplit(
            test_seasons=kwargs.get('test_seasons', 1),
            max_train_seasons=kwargs.get('max_train_seasons', None),
            rolling=kwargs.get('rolling', True),
            season_column=season_column
        )
    
    elif cv_type == "matchday":
        return MatchDayBasedSplit(
            n_future_match_days=kwargs.get('n_future_match_days', 1),
            n_test_periods=kwargs.get('n_test_periods', 10),
            min_train_match_days=kwargs.get('min_train_match_days', 3),
            date_column=date_column,
            season_column=season_column
        )
    
    else:
        logger.warning(f"Unknown CV type '{cv_type}', falling back to random CV")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=kwargs.get('random_state', 42))


def optimize_hyperparameters(
    model_type: str,
    model_class: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv_type: str = "time",
    n_trials: int = 50,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    cv_folds: int = 5,
    random_state: int = 42,
    temporal_params: Optional[Dict[str, Any]] = None,
    cv_groups: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization using Bayesian optimization.
    
    Args:
        model_type: Type of model to optimize
        model_class: Scikit-learn compatible model class
        X: Feature matrix
        y: Target vector
        cv_type: Type of cross-validation
        n_trials: Number of optimization trials
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        temporal_params: Parameters for temporal validation
        cv_groups: Optional grouping for time-based CV
        
    Returns:
        Dictionary with optimization results
    """
    start_time = time.time()
    
    # Check if it's a sequence model
    if model_type == "transformer":
        return optimize_sequence_model_hyperparameters(
            X, y, cv_type, n_trials, cv_folds, random_state, temporal_params, cv_groups
        )
    
    # For standard ML models, proceed with the original implementation
    # Create parameter space
    param_space = create_parameter_space(model_type)
    
    if not param_space:
        logger.warning(f"No parameter space defined for model type: {model_type}")
        return {"error": f"No parameter space defined for model type: {model_type}"}
    
    # Get CV splitter
    if temporal_params is None:
        temporal_params = {}
    
    cv_params = {
        'cv_type': cv_type,
        'n_splits': cv_folds,
        'test_size': temporal_params.get('test_size', 0.2),
        'gap': temporal_params.get('gap', 0),
        'date_column': temporal_params.get('date_column', 'date'),
        'season_column': temporal_params.get('season_column', 'season'),
        'random_state': random_state
    }
    
    # Add specific parameters for different CV types
    if cv_type == "season":
        cv_params.update({
            'test_seasons': temporal_params.get('test_seasons', 1),
            'max_train_seasons': temporal_params.get('max_train_seasons', None),
            'rolling': temporal_params.get('rolling', True)
        })
    elif cv_type == "matchday":
        cv_params.update({
            'n_future_match_days': temporal_params.get('n_future_match_days', 1),
            'n_test_periods': temporal_params.get('n_test_periods', 10),
            'min_train_match_days': temporal_params.get('min_train_match_days', 3)
        })
    
    cv_splitter = get_cv_splitter(**cv_params)
    
    # Create Optuna sampler and pruner
    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    # Create Optuna study
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name=f"{model_type}_optimization"
    )
    
    # Create objective function
    def objective(trial):
        return objective_function(
            trial, model_class, param_space, X, y, cv_splitter, cv_groups, scoring, n_jobs
        )
    
    # Run optimization
    logger.info(f"Starting hyperparameter optimization for {model_type} model with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    # Get best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    # Get trial information
    trials_df = study.trials_dataframe()
    
    # Calculate optimization time
    optimization_time = time.time() - start_time
    
    # Prepare results
    results = {
        "model_type": model_type,
        "best_params": best_params,
        "best_score": best_score,
        "n_trials": n_trials,
        "cv_type": cv_type,
        "temporal_params": temporal_params,
        "trials": trials_df.to_dict('records') if not trials_df.empty else [],
        "optimization_time": optimization_time,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(HYPEROPT_DIR, f"{model_type}_hyperopt_{timestamp}.json")
    
    # Filter trials to save to JSON (some objects might not be serializable)
    filtered_results = results.copy()
    
    if 'trials' in filtered_results:
        filtered_trials = []
        for trial in filtered_results['trials']:
            filtered_trial = {k: v for k, v in trial.items() if isinstance(v, (int, float, str, bool, list, dict)) or v is None}
            filtered_trials.append(filtered_trial)
        filtered_results['trials'] = filtered_trials
    
    with open(results_path, 'w') as f:
        json.dump(filtered_results, f, indent=4)
    
    # Save the study for future reference
    study_path = os.path.join(HYPEROPT_DIR, f"{model_type}_study_{timestamp}.pkl")
    joblib.dump(study, study_path)
    
    logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
    logger.info(f"Best score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Results saved to {results_path}")
    
    return results


def plot_optimization_results(
    study: optuna.Study,
    output_path: Optional[str] = None
) -> None:
    """
    Plot optimization results.
    
    Args:
        study: Optuna study object
        output_path: Directory to save plots (if None, show plots)
    """
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances
        from optuna.visualization import plot_slice, plot_contour
        
        # Create output directory if specified
        if output_path:
            os.makedirs(output_path, exist_ok=True)
        
        # Plot optimization history
        fig = plot_optimization_history(study)
        if output_path:
            fig.write_image(os.path.join(output_path, "optimization_history.png"))
        else:
            fig.show()
        
        # Plot parameter importances
        fig = plot_param_importances(study)
        if output_path:
            fig.write_image(os.path.join(output_path, "param_importances.png"))
        else:
            fig.show()
        
        # Plot slice plots for important parameters
        param_importances = optuna.importance.get_param_importances(study)
        top_params = list(param_importances.keys())[:3]  # Top 3 parameters
        
        for param in top_params:
            fig = plot_slice(study, params=[param])
            if output_path:
                fig.write_image(os.path.join(output_path, f"slice_{param}.png"))
            else:
                fig.show()
        
        # Plot contour plots for important parameters
        if len(top_params) >= 2:
            fig = plot_contour(study, params=top_params[:2])
            if output_path:
                fig.write_image(os.path.join(output_path, "contour_plot.png"))
            else:
                fig.show()
        
    except ImportError:
        logger.warning("Could not import required visualization packages")
    except Exception as e:
        logger.error(f"Error plotting optimization results: {e}")


def load_optimization_results(model_type: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Load optimization results.
    
    Args:
        model_type: Type of model
        timestamp: Specific timestamp to load (if None, load the latest)
        
    Returns:
        Dictionary with optimization results
    """
    # Find optimization results files
    hyperopt_files = [f for f in os.listdir(HYPEROPT_DIR) if f.startswith(f"{model_type}_hyperopt_") and f.endswith(".json")]
    
    if not hyperopt_files:
        logger.warning(f"No optimization results found for {model_type}")
        return {}
    
    # If timestamp is specified, find that specific file
    if timestamp:
        target_file = f"{model_type}_hyperopt_{timestamp}.json"
        if target_file not in hyperopt_files:
            logger.warning(f"No optimization results found for {model_type} with timestamp {timestamp}")
            return {}
        
        results_path = os.path.join(HYPEROPT_DIR, target_file)
    else:
        # Sort by timestamp and get the latest
        hyperopt_files.sort(reverse=True)
        results_path = os.path.join(HYPEROPT_DIR, hyperopt_files[0])
    
    # Load results
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        return results
    except Exception as e:
        logger.error(f"Error loading optimization results: {e}")
        return {}


def load_study(model_type: str, timestamp: Optional[str] = None) -> Optional[optuna.Study]:
    """
    Load Optuna study.
    
    Args:
        model_type: Type of model
        timestamp: Specific timestamp to load (if None, load the latest)
        
    Returns:
        Optuna study object
    """
    # Find study files
    study_files = [f for f in os.listdir(HYPEROPT_DIR) if f.startswith(f"{model_type}_study_") and f.endswith(".pkl")]
    
    if not study_files:
        logger.warning(f"No study found for {model_type}")
        return None
    
    # If timestamp is specified, find that specific file
    if timestamp:
        target_file = f"{model_type}_study_{timestamp}.pkl"
        if target_file not in study_files:
            logger.warning(f"No study found for {model_type} with timestamp {timestamp}")
            return None
        
        study_path = os.path.join(HYPEROPT_DIR, target_file)
    else:
        # Sort by timestamp and get the latest
        study_files.sort(reverse=True)
        study_path = os.path.join(HYPEROPT_DIR, study_files[0])
    
    # Load study
    try:
        study = joblib.load(study_path)
        return study
    except Exception as e:
        logger.error(f"Error loading study: {e}")
        return None


def objective_function_sequence_model(
    trial: optuna.Trial,
    param_space: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    team_features: List[str],
    match_features: List[str],
    target_col: str,
    cv_splits: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]
) -> float:
    """
    Objective function for Optuna optimization of sequence models.
    
    Args:
        trial: Optuna trial object
        param_space: Parameter space definition
        df: DataFrame with match data
        team_features: List of team features
        match_features: List of match features
        target_col: Target column name
        cv_splits: Pre-generated CV splits
        
    Returns:
        Score value (higher is better)
    """
    # Configure parameters for this trial
    params = configure_trial_params(trial, param_space)
    
    # Extract sequence length from parameters (this affects data preparation)
    sequence_length = params.pop("sequence_length", 5)
    batch_size = params.pop("batch_size", 32)
    
    # Get number of classes
    if target_col == 'result':
        if 'result' in df.columns:
            num_classes = len(np.unique(df['result']))
        else:
            num_classes = 3  # Default for soccer results (home win, draw, away win)
    elif target_col == 'home_win':
        num_classes = 2
    else:
        num_classes = 1
    
    # Get dimensions for transformer model
    team_feature_dim = len([f for f in team_features if not f.startswith('home_') and not f.startswith('away_')])
    match_feature_dim = len(match_features)
    
    # Cross-validation loop
    fold_scores = []
    
    for train_data, test_data, train_y, test_y in cv_splits:
        # Create and configure the model
        model = SoccerTransformerModel(
            sequence_length=sequence_length,
            team_feature_dim=team_feature_dim,
            match_feature_dim=match_feature_dim,
            num_classes=num_classes,
            **params
        )
        
        # Prepare final target variable format
        if num_classes > 2:
            # Convert to one-hot encoding for multi-class
            from tensorflow.keras.utils import to_categorical
            train_y_final = to_categorical(train_y, num_classes=num_classes)
            test_y_final = to_categorical(test_y, num_classes=num_classes)
        else:
            train_y_final = train_y
            test_y_final = test_y
        
        # Prepare validation data for early stopping
        validation_data = (
            [test_data['X_home_seq'], test_data['X_away_seq'], test_data['X_match']],
            test_y_final
        )
        
        # Train the model with early stopping
        model.fit(
            train_data['X_home_seq'],
            train_data['X_away_seq'],
            train_data['X_match'],
            train_y_final,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=20,  # Limit epochs for optimization
            patience=5,
            verbose=0
        )
        
        # Evaluate on test set
        metrics = model.evaluate(
            test_data['X_home_seq'],
            test_data['X_away_seq'],
            test_data['X_match'],
            test_y_final
        )
        
        # Use accuracy as the score
        score = metrics.get('accuracy', 0)
        fold_scores.append(score)
        
        # Report intermediate values
        trial.report(score, len(fold_scores) - 1)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Return mean score across folds
    return np.mean(fold_scores)


def optimize_sequence_model_hyperparameters(
    df: pd.DataFrame,
    team_features: List[str] = None,
    match_features: List[str] = None,
    target_col: str = "result",
    cv_type: str = "time",
    n_trials: int = 30,
    cv_folds: int = 3,
    random_state: int = 42,
    temporal_params: Optional[Dict[str, Any]] = None,
    cv_groups: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization for sequence models.
    
    Args:
        df: DataFrame with match data
        team_features: List of team features
        match_features: List of match features
        target_col: Target column name
        cv_type: Type of cross-validation
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
        temporal_params: Parameters for temporal validation
        cv_groups: Optional grouping for time-based CV
        
    Returns:
        Dictionary with optimization results
    """
    start_time = time.time()
    logger.info("Starting hyperparameter optimization for sequence model")
    
    # Ensure SoccerTransformerModel is available
    if 'SoccerTransformerModel' not in globals():
        logger.error("SoccerTransformerModel not available. Make sure src/models/sequence_models.py exists.")
        return {"error": "SoccerTransformerModel not available"}
    
    # Set default features if not provided
    if team_features is None:
        # Use basic team features
        team_features = [
            'home_club_goals', 'away_club_goals',
            'home_club_possession', 'away_club_possession',
            'home_club_shots_on_target', 'away_club_shots_on_target',
            'home_club_shots_total', 'away_club_shots_total'
        ]
        
        # Add advanced features if available
        advanced_features = [
            'home_xg', 'away_xg',
            'home_elo_pre', 'away_elo_pre',
            'home_form', 'away_form'
        ]
        
        for feature in advanced_features:
            if feature in df.columns:
                team_features.append(feature)
    
    if match_features is None:
        # Use basic match features
        match_features = ['league_id', 'venue', 'season']
        
        # Add advanced features if available
        advanced_features = [
            'match_importance', 'rest_advantage',
            'home_rest_days', 'away_rest_days',
            'elo_diff', 'form_diff'
        ]
        
        for feature in advanced_features:
            if feature in df.columns:
                match_features.append(feature)
    
    # Convert categorical features to numeric
    for feature in match_features:
        if feature in df.columns and df[feature].dtype == 'object':
            df[feature] = pd.factorize(df[feature])[0]
    
    # Set up temporal validation
    if temporal_params is None:
        temporal_params = {}
    
    cv_params = {
        'cv_type': cv_type,
        'n_splits': cv_folds,
        'test_size': temporal_params.get('test_size', 0.2),
        'gap': temporal_params.get('gap', 0),
        'date_column': temporal_params.get('date_column', 'date'),
        'season_column': temporal_params.get('season_column', 'season'),
        'random_state': random_state
    }
    
    # Add specific parameters for different CV types
    if cv_type == "season":
        cv_params.update({
            'test_seasons': temporal_params.get('test_seasons', 1),
            'max_train_seasons': temporal_params.get('max_train_seasons', None),
            'rolling': temporal_params.get('rolling', True)
        })
    elif cv_type == "matchday":
        cv_params.update({
            'n_future_match_days': temporal_params.get('n_future_match_days', 1),
            'n_test_periods': temporal_params.get('n_test_periods', 10),
            'min_train_match_days': temporal_params.get('min_train_match_days', 3)
        })
    
    # Get the appropriate CV splitter
    cv_splitter = get_cv_splitter(**cv_params)
    
    # Generate CV splits (use default sequence length for now)
    processor = SequenceDataProcessor(sequence_length=5)
    
    # Generate time-aware splits once
    cv_splits = processor.prepare_time_aware_cv_splits(
        df, team_features, match_features, cv_splitter, groups=cv_groups
    )
    
    # Create parameter space
    param_space = create_parameter_space("transformer")
    
    # Create Optuna sampler and pruner
    sampler = TPESampler(seed=random_state)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    # Create Optuna study
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name="transformer_optimization"
    )
    
    # Create objective function
    def objective(trial):
        return objective_function_sequence_model(
            trial, param_space, df, team_features, match_features, target_col, cv_splits
        )
    
    # Run optimization
    logger.info(f"Starting hyperparameter optimization for transformer model with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    # Get best parameters and score
    best_params = study.best_params
    best_score = study.best_value
    
    # Get trial information
    trials_df = study.trials_dataframe()
    
    # Calculate optimization time
    optimization_time = time.time() - start_time
    
    # Prepare results
    results = {
        "model_type": "transformer",
        "best_params": best_params,
        "best_score": best_score,
        "n_trials": n_trials,
        "cv_type": cv_type,
        "temporal_params": temporal_params,
        "team_features": team_features,
        "match_features": match_features,
        "trials": trials_df.to_dict('records') if not trials_df.empty else [],
        "optimization_time": optimization_time,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(HYPEROPT_DIR, f"transformer_hyperopt_{timestamp}.json")
    
    # Filter trials to save to JSON (some objects might not be serializable)
    filtered_results = results.copy()
    
    if 'trials' in filtered_results:
        filtered_trials = []
        for trial in filtered_results['trials']:
            filtered_trial = {k: v for k, v in trial.items() if isinstance(v, (int, float, str, bool, list, dict)) or v is None}
            filtered_trials.append(filtered_trial)
        filtered_results['trials'] = filtered_trials
    
    with open(results_path, 'w') as f:
        json.dump(filtered_results, f, indent=4)
    
    # Save the study for future reference
    study_path = os.path.join(HYPEROPT_DIR, f"transformer_study_{timestamp}.pkl")
    joblib.dump(study, study_path)
    
    logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
    logger.info(f"Best score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Results saved to {results_path}")
    
    return results 

def optimize_hyperparameters_with_feature_selection(
    model_type: str,
    model_class: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cv_type: str = "time",
    n_trials: int = 50,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    cv_folds: int = 5,
    random_state: int = 42,
    temporal_params: Optional[Dict[str, Any]] = None,
    cv_groups: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters with feature selection for soccer prediction models.
    
    This implementation adds feature selection as part of the optimization
    process, using Optuna to identify the best feature set along with
    the best hyperparameters.
    
    Args:
        model_type: Type of model to optimize
        model_class: Model class to instantiate
        X: Feature matrix
        y: Target vector
        feature_names: Names of the features
        cv_type: Type of cross-validation
        n_trials: Number of optimization trials
        scoring: Scoring metric
        n_jobs: Number of parallel jobs
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        temporal_params: Parameters for temporal cross-validation
        cv_groups: Optional groups for cross-validation
        
    Returns:
        Dictionary with optimization results
    """
    logger.info(f"Optimizing hyperparameters with feature selection for {model_type} model")
    
    # Get parameter space for model
    param_space = create_parameter_space(model_type)
    
    if not param_space:
        logger.warning(f"No parameter space defined for model type: {model_type}")
        param_space = {}
    
    # Get CV splitter
    if temporal_params is None:
        temporal_params = {}
    
    cv_splitter = get_cv_splitter(
        cv_type=cv_type,
        n_splits=cv_folds,
        **temporal_params
    )
    
    # Create study
    study = optuna.create_study(
        direction="maximize", 
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner()
    )
    
    # Categorize features for soccer prediction
    feature_categories = categorize_soccer_features(feature_names)
    
    # Define objective function with feature selection
    def objective(trial):
        # Configure model parameters
        params = configure_trial_params(trial, param_space)
        
        # Feature selection
        selected_features = []
        
        # For each feature category, determine if it should be included
        for category, features in feature_categories.items():
            # Skip if no features in this category
            if not features:
                continue
            
            # For key categories (like team strength, form), always include
            if category in ['team_strength', 'team_form', 'match_importance']:
                selected_features.extend(features)
            else:
                # For other categories, let Optuna decide
                include_category = trial.suggest_categorical(f"include_{category}", [True, False])
                
                if include_category:
                    # If including, decide how many features to include
                    if len(features) > 3:
                        n_features = trial.suggest_int(f"n_{category}_features", 1, len(features))
                        # Let Optuna suggest which specific features to include
                        indices = trial.suggest_categorical(
                            f"{category}_indices",
                            list(range(len(features)))[:n_features]
                        )
                        if isinstance(indices, int):
                            indices = [indices]
                        selected_features.extend([features[i] for i in indices])
                    else:
                        # If few features, include all
                        selected_features.extend(features)
        
        # Make sure we have at least some features
        if not selected_features:
            # Include at least basic features
            for category in ['team_strength', 'basic_stats']:
                if category in feature_categories and feature_categories[category]:
                    selected_features.extend(feature_categories[category])
        
        # Get indices of selected features
        feature_indices = [i for i, name in enumerate(feature_names) if name in selected_features]
        
        # Create model with the parameters
        model = model_class(**params)
        
        # Select features and evaluate
        X_selected = X[:, feature_indices]
        
        try:
            # Perform cross-validation
            if cv_groups is not None:
                scores = cross_val_score(model, X_selected, y, cv=cv_splitter, groups=cv_groups, scoring=scoring)
            else:
                scores = cross_val_score(model, X_selected, y, cv=cv_splitter, scoring=scoring)
            
            # Calculate score
            score = scores.mean()
            
            # Store feature selection information
            trial.set_user_attr("selected_features", selected_features)
            trial.set_user_attr("n_features", len(selected_features))
            trial.set_user_attr("feature_categories", {k: len([f for f in selected_features if f in v]) 
                                                   for k, v in feature_categories.items()})
            
            return score
        except Exception as e:
            logger.error(f"Error in trial: {e}")
            return float('-inf')
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
    # Get best parameters and features
    best_params = study.best_params
    best_trial = study.best_trial
    best_features = best_trial.user_attrs.get("selected_features", [])
    
    # Log results
    logger.info(f"Best score: {best_trial.value:.4f}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Number of selected features: {len(best_features)}")
    
    # Create final model with best parameters
    best_model = model_class(**best_params)
    
    # Get indices of best features
    best_feature_indices = [i for i, name in enumerate(feature_names) if name in best_features]
    
    # Fit on entire dataset
    X_best = X[:, best_feature_indices]
    best_model.fit(X_best, y)
    
    # Create timestamp for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the optimization results
    results = {
        "model_type": model_type,
        "best_score": float(best_trial.value),
        "best_params": best_params,
        "n_trials": n_trials,
        "timestamp": timestamp,
        "scoring": scoring,
        "cv_type": cv_type,
        "cv_folds": cv_folds,
        "selected_features": best_features,
        "n_features": len(best_features),
        "feature_counts_by_category": best_trial.user_attrs.get("feature_categories", {})
    }
    
    # Save study for later analysis
    study_path = os.path.join(HYPEROPT_DIR, f"{model_type}_study_{timestamp}.pkl")
    joblib.dump(study, study_path)
    
    # Save results
    results_path = os.path.join(HYPEROPT_DIR, f"{model_type}_results_{timestamp}.json")
    with open(results_path, "w") as f:
        # Convert numpy types to Python types for JSON serialization
        results_json = {}
        for k, v in results.items():
            if isinstance(v, dict):
                results_json[k] = {str(kk): float(vv) if isinstance(vv, (np.float32, np.float64)) else vv 
                                 for kk, vv in v.items()}
            elif isinstance(v, (np.float32, np.float64)):
                results_json[k] = float(v)
            else:
                results_json[k] = v
        
        json.dump(results_json, f, indent=2)
    
    # Return results
    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": float(best_trial.value),
        "selected_features": best_features,
        "feature_indices": best_feature_indices,
        "study": study,
        "study_path": study_path,
        "results_path": results_path
    }

def categorize_soccer_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize soccer features into meaningful groups for feature selection.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary with categories as keys and lists of feature names as values
    """
    categories = {
        "team_strength": [],
        "team_form": [],
        "player_stats": [],
        "match_context": [],
        "match_importance": [],
        "betting_odds": [],
        "historical_h2h": [],
        "basic_stats": [],
        "advanced_stats": [],
        "time_features": [],
        "other": []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        # Team strength features
        if any(term in feature_lower for term in ['elo', 'strength', 'rating', 'rank', 'position']):
            categories["team_strength"].append(feature)
        
        # Team form features
        elif any(term in feature_lower for term in ['form', 'streak', 'last_', 'recent_', 'exp_form']):
            categories["team_form"].append(feature)
        
        # Player stats features
        elif any(term in feature_lower for term in ['player', 'squad', 'lineup', 'injury', 'suspension']):
            categories["player_stats"].append(feature)
        
        # Match context features
        elif any(term in feature_lower for term in ['home_', 'away_', 'venue', 'distance', 'travel']):
            categories["match_context"].append(feature)
        
        # Match importance features
        elif any(term in feature_lower for term in ['importance', 'crucial', 'must_win', 'pressure', 'season_progress']):
            categories["match_importance"].append(feature)
        
        # Betting odds features
        elif any(term in feature_lower for term in ['odds', 'probability', 'implied', 'bet', 'market']):
            categories["betting_odds"].append(feature)
        
        # Historical head-to-head features
        elif any(term in feature_lower for term in ['h2h', 'head_to_head', 'against_', 'vs_']):
            categories["historical_h2h"].append(feature)
        
        # Basic stats features
        elif any(term in feature_lower for term in ['goals', 'score', 'win', 'loss', 'draw', 'points']):
            categories["basic_stats"].append(feature)
        
        # Advanced stats features
        elif any(term in feature_lower for term in ['xg', 'expected', 'possession', 'shots', 'passes', 'tackles']):
            categories["advanced_stats"].append(feature)
        
        # Time features
        elif any(term in feature_lower for term in ['month', 'day', 'year', 'week', 'season', 'date']):
            categories["time_features"].append(feature)
        
        # Other features
        else:
            categories["other"].append(feature)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}

def optimize_calibration_method(
    model_path: str,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    n_trials: int = 20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Optimize calibration method for a soccer prediction model.
    
    Args:
        model_path: Path to the trained model
        X_cal: Features for calibration
        y_cal: Labels for calibration
        n_trials: Number of optimization trials
        random_state: Random seed
        
    Returns:
        Dictionary with calibration optimization results
    """
    logger.info(f"Optimizing calibration for model: {model_path}")
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")
    
    # Get model predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_cal)
    elif hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
        y_proba = model.model.predict_proba(X_cal)
    else:
        raise ValueError("Model does not have predict_proba method")
    
    # Create study
    study = optuna.create_study(
        direction="minimize",  # Minimize log loss
        sampler=TPESampler(seed=random_state)
    )
    
    # Load calibration module
    try:
        from src.models.calibration import ProbabilityCalibrator
    except ImportError:
        raise ImportError("Calibration module not found. Please implement src/models/calibration.py first.")
    
    # Define objective function
    def objective(trial):
        # Select calibration method
        method = trial.suggest_categorical("method", ["platt", "isotonic", "beta", "temperature", "ensemble"])
        
        # Train-validation split for calibration
        X_train, X_val, y_train, y_val, proba_train, proba_val = train_test_split(
            X_cal, y_cal, y_proba, test_size=0.3, random_state=trial.number
        )
        
        # Create and fit calibrator
        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(y_train, proba_train)
        
        # Calibrate validation predictions
        calibrated_proba = calibrator.calibrate(proba_val)
        
        # Calculate log loss with calibrated probabilities
        loss = log_loss(y_val, calibrated_proba)
        
        # Store calibrator for best trial
        trial.set_user_attr("calibrator", calibrator)
        
        return loss
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Get best calibration method and calibrator
    best_method = study.best_params["method"]
    best_calibrator = study.best_trial.user_attrs["calibrator"]
    
    # Evaluate on full calibration dataset
    calibrated_proba = best_calibrator.calibrate(y_proba)
    calibrated_loss = log_loss(y_cal, calibrated_proba)
    original_loss = log_loss(y_cal, y_proba)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save calibrator
    calibrator_path = os.path.join(HYPEROPT_DIR, f"calibrator_{best_method}_{timestamp}.pkl")
    best_calibrator.save(calibrator_path)
    
    # Compile results
    results = {
        "best_method": best_method,
        "original_log_loss": float(original_loss),
        "calibrated_log_loss": float(calibrated_loss),
        "improvement": float(original_loss - calibrated_loss),
        "improvement_pct": float((original_loss - calibrated_loss) / original_loss * 100),
        "timestamp": timestamp,
        "n_trials": n_trials,
        "calibrator_path": calibrator_path
    }
    
    # Save results
    results_path = os.path.join(HYPEROPT_DIR, f"calibration_results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Best calibration method: {best_method}")
    logger.info(f"Log loss improvement: {results['improvement']:.4f} ({results['improvement_pct']:.2f}%)")
    
    return {
        "best_calibrator": best_calibrator,
        "best_method": best_method,
        "results": results,
        "study": study,
        "calibrator_path": calibrator_path,
        "results_path": results_path
    }

def optimize_ensemble_weights(
    model_paths: List[str],
    X: np.ndarray,
    y: np.ndarray,
    ensemble_type: str = "dynamic_weighting",
    n_trials: int = 30,
    cv_type: str = "time", 
    cv_folds: int = 5,
    temporal_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Optimize ensemble weights and configuration for soccer prediction.
    
    Args:
        model_paths: Paths to trained models
        X: Feature matrix
        y: Target vector
        ensemble_type: Type of ensemble
        n_trials: Number of optimization trials
        cv_type: Type of cross-validation
        cv_folds: Number of cross-validation folds
        temporal_params: Parameters for temporal cross-validation
        random_state: Random seed
        
    Returns:
        Dictionary with optimization results
    """
    logger.info(f"Optimizing {ensemble_type} ensemble with {len(model_paths)} models")
    
    # Get parameter space for ensemble type
    param_space = create_parameter_space(ensemble_type)
    
    # Load models
    models = []
    for path in model_paths:
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            models.append(model)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
    
    if not models:
        raise ValueError("No models could be loaded")
    
    # Get CV splitter
    if temporal_params is None:
        temporal_params = {}
    
    cv_splitter = get_cv_splitter(
        cv_type=cv_type,
        n_splits=cv_folds,
        **temporal_params
    )
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state)
    )
    
    # Import ensemble module
    try:
        from src.models.ensemble import EnsemblePredictor
    except ImportError:
        raise ImportError("Ensemble module not found")
    
    # Define objective function
    def objective(trial):
        # Get ensemble parameters
        params = configure_trial_params(trial, param_space)
        
        # Create ensemble
        ensemble = EnsemblePredictor(
            ensemble_type=ensemble_type,
            models=models
        )
        
        # Set up cross-validation
        cv_scores = []
        
        for train_idx, val_idx in cv_splitter.split(X, y):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Apply parameters to ensemble
            if ensemble_type == "dynamic_weighting":
                recency_factor = params.get("recency_factor", 0.9)
                window_size = params.get("window_size", 10)
                # Train ensemble
                ensemble.train(X_train, y_train)
                # Apply dynamic weighting
                ensemble._calculate_dynamic_weights(X_val, y_val, 
                                                   recency_factor=recency_factor,
                                                   window_size=window_size)
            elif ensemble_type == "context_aware":
                # Extract context features
                context_features = ensemble._prepare_context_features(X_train)
                val_context_features = ensemble._prepare_context_features(X_val)
                # Train context-aware ensemble
                ensemble.train(X_train, y_train, context_features=context_features)
                # Make predictions
                y_pred = ensemble.predict(X_val, context_features=val_context_features)
            else:
                # Standard ensemble training
                ensemble.train(X_train, y_train)
                # Make predictions
                y_pred = ensemble.predict(X_val)
            
            # Calculate score
            score = accuracy_score(y_val, y_pred)
            cv_scores.append(score)
        
        # Return mean score
        return np.mean(cv_scores)
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    
    # Create final ensemble with best parameters
    final_ensemble = EnsemblePredictor(
        ensemble_type=ensemble_type,
        models=models
    )
    
    # Train final ensemble on all data
    if ensemble_type == "context_aware":
        # Extract context features
        context_features = final_ensemble._prepare_context_features(X)
        # Train context-aware ensemble
        final_ensemble.train(X, y, context_features=context_features)
    else:
        # Standard ensemble training
        final_ensemble.train(X, y)
    
    # Apply best parameters
    if ensemble_type == "dynamic_weighting":
        recency_factor = best_params.get("recency_factor", 0.9)
        window_size = best_params.get("window_size", 10)
        final_ensemble._calculate_dynamic_weights(X, y, 
                                               recency_factor=recency_factor,
                                               window_size=window_size)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save ensemble
    ensemble_path = os.path.join(HYPEROPT_DIR, f"ensemble_{ensemble_type}_{timestamp}.pkl")
    with open(ensemble_path, "wb") as f:
        pickle.dump(final_ensemble, f)
    
    # Create results
    results = {
        "ensemble_type": ensemble_type,
        "best_params": best_params,
        "best_score": float(study.best_value),
        "n_models": len(models),
        "model_types": [getattr(model, "model_type", "unknown") for model in models],
        "timestamp": timestamp,
        "n_trials": n_trials,
        "ensemble_path": ensemble_path
    }
    
    # Save results
    results_path = os.path.join(HYPEROPT_DIR, f"ensemble_results_{timestamp}.json")
    with open(results_path, "w") as f:
        # Convert numpy types for JSON serialization
        results_json = {}
        for k, v in results.items():
            if isinstance(v, dict):
                results_json[k] = {str(kk): float(vv) if isinstance(vv, (np.float32, np.float64)) else vv 
                                 for kk, vv in v.items()}
            elif isinstance(v, (np.float32, np.float64)):
                results_json[k] = float(v)
            else:
                results_json[k] = v
        
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Best ensemble parameters: {best_params}")
    logger.info(f"Best score: {study.best_value:.4f}")
    
    return {
        "ensemble": final_ensemble,
        "best_params": best_params,
        "best_score": float(study.best_value),
        "study": study,
        "ensemble_path": ensemble_path,
        "results_path": results_path
    } 