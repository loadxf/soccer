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
from src.models.time_validation import TimeSeriesSplit, SeasonBasedSplit, MatchDayBasedSplit
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
    # Import Elo ratings
    from src.data.elo_ratings import generate_elo_features
except ImportError:
    pass

try:
    # Import match context features
    from src.data.match_context import generate_match_context_features
except ImportError:
    pass

try:
    # Import sequence models
    from src.models.sequence_models import SoccerTransformerModel, SequenceDataProcessor
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
SEQUENCE_MODELS_DIR = os.path.join(DATA_DIR, "sequence_models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(SEQUENCE_MODELS_DIR, exist_ok=True)


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


def load_advanced_soccer_features(dataset_name: str, include_elo: bool = True, 
                                include_match_context: bool = True) -> pd.DataFrame:
    """
    Load or create advanced soccer features.
    
    Args:
        dataset_name: Name of the dataset
        include_elo: Whether to include Elo rating features
        include_match_context: Whether to include match context features
        
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
        
        # Add Elo rating features if requested
        if include_elo:
            try:
                from src.data.elo_ratings import generate_elo_features
                elo_features = generate_elo_features(matches_df)
                
                # Merge Elo features with other features
                if not features_df.empty and not elo_features.empty:
                    # Ensure we have common columns for merging
                    merge_cols = list(set(features_df.columns) & set(elo_features.columns))
                    if merge_cols:
                        # Keep only new columns from elo_features
                        elo_cols_to_add = [col for col in elo_features.columns if col not in features_df.columns 
                                         or col in ['home_elo_pre', 'away_elo_pre', 'elo_diff']]
                        features_df = pd.merge(features_df, elo_features[merge_cols + elo_cols_to_add], on=merge_cols, how='left')
                        logger.info("Added Elo rating features")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to add Elo rating features: {e}")
        
        # Add match context features if requested
        if include_match_context:
            try:
                from src.data.match_context import generate_match_context_features
                
                # Find clubs data file if available
                club_files = [f for f in os.listdir(processed_dir) if 'club' in f.lower() or 'team' in f.lower()]
                clubs_df = None
                
                if club_files:
                    club_file = club_files[0]
                    club_path = os.path.join(processed_dir, club_file)
                    clubs_df = pd.read_csv(club_path)
                
                # Generate context features
                context_features = generate_match_context_features(matches_df, clubs_df)
                
                # Merge context features with other features
                if not features_df.empty and not context_features.empty:
                    # Ensure we have common columns for merging
                    merge_cols = list(set(features_df.columns) & set(context_features.columns))
                    if merge_cols:
                        # Keep only new columns from context_features
                        context_cols_to_add = [col for col in context_features.columns 
                                           if col not in features_df.columns 
                                           or col in ['home_rest_days', 'away_rest_days', 'rest_advantage', 'match_importance']]
                        features_df = pd.merge(features_df, context_features[merge_cols + context_cols_to_add], on=merge_cols, how='left')
                        logger.info("Added match context features")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to add match context features: {e}")
        
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


def get_cv_splitter(
    cv_type: str = "random",
    n_splits: int = 5,
    test_size: float = 0.2,
    gap: int = 0,
    date_column: str = "date",
    season_column: str = "season",
    **kwargs
) -> Union[StratifiedKFold, TimeSeriesSplit, SeasonBasedSplit, MatchDayBasedSplit]:
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


def train_model(
    model_type: str = "logistic",
    dataset_name: str = "transfermarkt",
    feature_type: str = "match_features",
    target_col: str = "result",
    test_size: float = 0.2,
    cv_type: str = "time",
    cv_folds: int = 5,
    hyperparameter_tuning: bool = False,
    random_state: int = 42,
    model_params: Optional[Dict[str, Any]] = None,
    temporal_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train a model with optional hyperparameter tuning.
    
    Args:
        model_type: Type of model to train
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        target_col: Name of the target column
        test_size: Portion of data to use for testing
        cv_type: Type of cross-validation ('random', 'time', 'season', or 'matchday')
        cv_folds: Number of cross-validation folds
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        random_state: Random seed for reproducibility
        model_params: Optional model parameters to override defaults
        temporal_params: Optional parameters for temporal validation
        
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
        if feature_type == "advanced_soccer_features":
            df = load_advanced_soccer_features(dataset_name)
        else:
            df = load_feature_data(dataset_name, feature_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading feature data: {e}")
        raise
    
    # Set model parameters
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.get(model_type, {}).copy()
    
    # Create model
    model = BaselineMatchPredictor(
        model_type=model_type,
        dataset_name=dataset_name,
        feature_type=feature_type,
        model_params=model_params
    )
    
    # Process data
    X, y = model.process_data(df, target_col=target_col)
    if X is None or y is None:
        raise ValueError("Failed to process data")
    
    # Set parameters for temporal validation
    if temporal_params is None:
        temporal_params = {}
    
    cv_params = {
        'cv_type': cv_type,
        'n_splits': cv_folds,
        'test_size': test_size,
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
    
    # Add date/season information for temporal validation
    cv_groups = None
    if cv_type in ["time", "season", "matchday"] and isinstance(df, pd.DataFrame):
        date_col = cv_params['date_column']
        season_col = cv_params['season_column']
        
        # Use date column for time-based CV if available
        if cv_type == "time" and date_col in df.columns:
            cv_groups = df[date_col]
        
        # Use season column for season-based CV if available
        elif cv_type == "season" and season_col in df.columns:
            cv_groups = df[season_col]
    
    # Get the appropriate CV splitter
    cv_splitter = get_cv_splitter(**cv_params)
    
    # Hyperparameter tuning with cross-validation
    if hyperparameter_tuning:
        # Define parameter grid based on model type
        if model_type == "logistic":
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'class_weight': ['balanced', None],
                'max_iter': [1000]
            }
        elif model_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        elif model_type == "xgboost":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            # Default minimal grid for other models
            param_grid = {}
        
        if param_grid:
            # Create GridSearchCV
            grid_search = GridSearchCV(
                model.model,
                param_grid,
                cv=cv_splitter,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # Perform grid search
            if cv_groups is not None and cv_type in ["time", "season", "matchday"]:
                grid_search.fit(X, y, groups=cv_groups)
            else:
                grid_search.fit(X, y)
            
            # Update model with best parameters
            model.model = grid_search.best_estimator_
            model.model_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Split data for final evaluation
    if cv_type == "random":
        # Standard random train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        # Use temporal validation for final split
        splits = list(cv_splitter.split(X, y, groups=cv_groups))
        
        if not splits:
            logger.warning("No valid splits found with the specified temporal validation. Falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            # Use the most recent split for evaluation
            train_indices, test_indices = splits[-1]
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
    
    # Train the model
    logger.info(f"Training {model_type} model on {len(X_train)} samples")
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    logger.info(f"Model trained in {datetime.now() - start_time}")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_{dataset_name}_{feature_type}_{timestamp}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    model.save(model_path)
    
    # Save training details
    training_details = {
        "model_type": model_type,
        "dataset_name": dataset_name,
        "feature_type": feature_type,
        "model_params": model.model_params,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(accuracy),
        "training_time": str(datetime.now() - start_time),
        "timestamp": timestamp,
        "model_path": model_path,
        "cv_type": cv_type,
        "temporal_params": temporal_params
    }
    
    # Save training details
    training_details_path = os.path.join(TRAINING_DIR, f"{model_type}_{dataset_name}_{feature_type}_{timestamp}.json")
    with open(training_details_path, 'w') as f:
        json.dump(training_details, f, indent=4)
    
    return training_details


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
    model_types = []
    
    for path in model_paths:
        try:
            # Check if it's a sequence model
            if "transformer" in path:
                try:
                    model = SoccerTransformerModel.load(path.replace(".h5", ""))
                    models.append(model)
                    model_types.append("transformer")
                    logger.info(f"Loaded transformer model from {path}")
                except Exception as e:
                    logger.error(f"Error loading transformer model from {path}: {e}")
            else:
                # Standard ML models
                model = BaselineMatchPredictor.load(path)
                models.append(model)
                model_types.append(model.model_type)
                logger.info(f"Loaded {model.model_type} model from {path}")
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
        "model_types": model_types,
        "weights": weights,
        "model_paths": model_paths,
        "created_at": datetime.now().isoformat()
    }
    
    # Create model info for each model
    model_infos = []
    for i, model in enumerate(models):
        if model_types[i] == "transformer":
            # Get transformer model info
            model_info = {
                "model_type": "transformer",
                "sequence_length": model.sequence_length,
                "team_feature_dim": model.team_feature_dim,
                "match_feature_dim": model.match_feature_dim,
                "num_classes": model.num_classes
            }
        else:
            # Standard model info
            model_info = model.model_info
        
        model_infos.append(model_info)
    
    ensemble_info["models"] = model_infos
    
    # Save ensemble info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_path = os.path.join(
        MODELS_DIR,
        f"ensemble_{ensemble_type}_{timestamp}.json"
    )
    with open(ensemble_path, "w") as f:
        json.dump(ensemble_info, f, indent=2, default=str)
    
    logger.info(f"Created {ensemble_type} ensemble with {len(models)} models of types: {', '.join(model_types)}")
    logger.info(f"Saved ensemble info to {ensemble_path}")
    
    return ensemble_info


def predict_with_ensemble(
    ensemble_path: str,
    home_team_id: int,
    away_team_id: int,
    features: Optional[Dict[str, Any]] = None,
    sequence_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make a prediction using an ensemble of models.
    
    Args:
        ensemble_path: Path to the ensemble info file
        home_team_id: ID of the home team
        away_team_id: ID of the away team
        features: Optional dictionary of additional features
        sequence_data: Optional dictionary with sequence data for transformer models
            (home_sequence, away_sequence, match_features)
        
    Returns:
        Dict[str, Any]: Prediction results
    """
    # Load ensemble info
    with open(ensemble_path, "r") as f:
        ensemble_info = json.load(f)
    
    ensemble_type = ensemble_info["ensemble_type"]
    model_paths = ensemble_info["model_paths"]
    model_types = ensemble_info.get("model_types", ["unknown"] * len(model_paths))
    weights = ensemble_info["weights"]
    
    # Load all models
    models = []
    for i, path in enumerate(model_paths):
        model_type = model_types[i] if i < len(model_types) else "unknown"
        try:
            if model_type == "transformer":
                model = SoccerTransformerModel.load(path.replace(".h5", ""))
            else:
                model = BaselineMatchPredictor.load(path)
            models.append(model)
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
    
    if not models:
        raise ValueError("No models could be loaded for ensemble prediction")
    
    # Get predictions from all models
    predictions = []
    for i, model in enumerate(models):
        try:
            if model_types[i] == "transformer":
                if sequence_data is None:
                    logger.warning("Sequence data not provided for transformer model prediction")
                    continue
                    
                # Make transformer model prediction
                probs = model.predict(
                    sequence_data["home_sequence"], 
                    sequence_data["away_sequence"], 
                    sequence_data["match_features"]
                )
                
                # Format as consistent prediction
                if model.num_classes == 3:
                    # Home win (0), draw (1), away win (2)
                    home_win_prob = float(probs[0][0])
                    draw_prob = float(probs[0][1])
                    away_win_prob = float(probs[0][2])
                    
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
                    
                elif model.num_classes == 2:
                    # Binary - home win (1) or not (0)
                    home_win_prob = float(probs[0][0])
                    draw_prob = 0.0  # Not directly predicted
                    away_win_prob = 1.0 - home_win_prob
                    
                    # Determine prediction
                    if home_win_prob >= 0.5:
                        prediction = "home_win"
                        confidence = home_win_prob
                    else:
                        prediction = "away_win"  # Assuming not home_win means away_win
                        confidence = away_win_prob
                else:
                    # Regression or other format - not directly compatible
                    logger.warning(f"Unsupported transformer model output format with {model.num_classes} classes")
                    continue
                
                pred = {
                    "model_type": "transformer",
                    "home_win_probability": home_win_prob,
                    "draw_probability": draw_prob,
                    "away_win_probability": away_win_prob,
                    "prediction": prediction,
                    "confidence": confidence
                }
            else:
                # Standard model prediction
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
            if model_type == "transformer":
                result = train_sequence_model(dataset_name=dataset_name, feature_type=feature_type, **kwargs)
            else:
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


def train_sequence_model(
    dataset_name: str = "transfermarkt",
    feature_type: str = "match_features",
    sequence_length: int = 5,
    team_features: Optional[List[str]] = None,
    match_features: Optional[List[str]] = None,
    target_col: str = "result",
    test_size: float = 0.2,
    cv_type: str = "time",
    cv_folds: int = 5,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    random_state: int = 42,
    model_params: Optional[Dict[str, Any]] = None,
    temporal_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train a sequence-based transformer model for soccer prediction.
    
    Args:
        dataset_name: Name of the dataset to use
        feature_type: Type of features to use
        sequence_length: Number of past matches to include for each team
        team_features: List of team-specific features to use (if None, use defaults)
        match_features: List of match-specific features to use (if None, use defaults)
        target_col: Name of the target column
        test_size: Portion of data to use for testing
        cv_type: Type of cross-validation ('random', 'time', 'season', or 'matchday')
        cv_folds: Number of cross-validation folds
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Patience for early stopping
        random_state: Random seed for reproducibility
        model_params: Optional model parameters
        temporal_params: Optional parameters for temporal validation
        
    Returns:
        Dict[str, Any]: Training results
    """
    # Record start time
    start_time = datetime.now()
    
    # Ensure SoccerTransformerModel is available
    if 'SoccerTransformerModel' not in globals():
        logger.error("SoccerTransformerModel not available. Make sure src/models/sequence_models.py exists.")
        raise ImportError("SoccerTransformerModel not available")
    
    # Load data
    try:
        if feature_type == "advanced_soccer_features":
            df = load_advanced_soccer_features(dataset_name)
        else:
            df = load_feature_data(dataset_name, feature_type)
    except FileNotFoundError as e:
        logger.error(f"Error loading feature data: {e}")
        raise
    
    # Ensure required columns exist
    if 'date' not in df.columns:
        logger.error("Date column not found in dataset")
        raise ValueError("Date column not found in dataset")
    
    if 'home_club_id' not in df.columns or 'away_club_id' not in df.columns:
        logger.error("Team ID columns not found in dataset")
        raise ValueError("Team ID columns not found in dataset")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Determine target column values
    if target_col not in df.columns:
        if target_col == "result" and "home_club_goals" in df.columns and "away_club_goals" in df.columns:
            # Create result column (0: home win, 1: draw, 2: away win)
            df['result'] = np.select(
                [df['home_club_goals'] > df['away_club_goals'], 
                 df['home_club_goals'] == df['away_club_goals']],
                [0, 1], 2
            )
        else:
            logger.error(f"Target column {target_col} not found")
            raise ValueError(f"Target column {target_col} not found")
    
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
        if df[feature].dtype == 'object':
            df[feature] = pd.factorize(df[feature])[0]
    
    # Set up temporal validation
    if temporal_params is None:
        temporal_params = {}
    
    cv_params = {
        'cv_type': cv_type,
        'n_splits': cv_folds,
        'test_size': test_size,
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
    
    # Add date/season information for temporal validation
    cv_groups = None
    if cv_type in ["time", "season", "matchday"]:
        date_col = cv_params['date_column']
        season_col = cv_params['season_column']
        
        # Use date column for time-based CV if available
        if cv_type == "time" and date_col in df.columns:
            cv_groups = df[date_col]
        
        # Use season column for season-based CV if available
        elif cv_type == "season" and season_col in df.columns:
            cv_groups = df[season_col]
    
    # Get the appropriate CV splitter
    cv_splitter = get_cv_splitter(**cv_params)
    
    # Initialize sequence processor
    processor = SequenceDataProcessor(sequence_length=sequence_length)
    
    # Generate CV splits
    X = np.arange(len(df))  # Just for splitting
    y = df[target_col].values
    
    # Get parameters for transformer model
    team_feature_dim = len([f for f in team_features if not f.startswith('home_') and not f.startswith('away_')])
    match_feature_dim = len(match_features)
    
    # Set default model parameters if not provided
    if model_params is None:
        model_params = {
            'embed_dim': 64,
            'num_heads': 2,
            'ff_dim': 64,
            'num_transformer_blocks': 2,
            'dropout_rate': 0.2,
            'l2_reg': 1e-4,
            'learning_rate': 0.001
        }
    
    # Determine number of output classes
    if target_col == 'result':
        num_classes = len(np.unique(y))
    elif target_col == 'home_win':
        num_classes = 2
    else:
        # Regression task or custom target
        num_classes = 1
    
    # Create the model
    model = SoccerTransformerModel(
        sequence_length=sequence_length,
        team_feature_dim=team_feature_dim,
        match_feature_dim=match_feature_dim,
        num_classes=num_classes,
        **model_params
    )
    
    # Prepare for time-based CV
    if cv_type == "random":
        # Use simple train-test split
        train_indices, test_indices = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=y
        )
        train_data, test_data, all_y = processor.prepare_dataset(
            df, team_features, match_features, train_indices, test_indices
        )
        train_y = all_y[:len(train_indices)]
        test_y = all_y[len(train_indices):]
    else:
        # Use temporal validation for final split
        splits = list(cv_splitter.split(X, y, groups=cv_groups))
        
        if not splits:
            logger.warning("No valid splits found with the specified temporal validation. Falling back to random split.")
            train_indices, test_indices = train_test_split(
                np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=y
            )
            train_data, test_data, all_y = processor.prepare_dataset(
                df, team_features, match_features, train_indices, test_indices
            )
            train_y = all_y[:len(train_indices)]
            test_y = all_y[len(train_indices):]
        else:
            # Use the most recent split for evaluation
            train_indices, test_indices = splits[-1]
            train_data, test_data, all_y = processor.prepare_dataset(
                df, team_features, match_features, train_indices, test_indices
            )
            train_y = all_y[:len(train_indices)]
            test_y = all_y[len(test_indices):]
    
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
    
    # Train the model
    logger.info(f"Training transformer model on {len(train_indices)} samples")
    training_results = model.fit(
        train_data['X_home_seq'],
        train_data['X_away_seq'],
        train_data['X_match'],
        train_y_final,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        verbose=1
    )
    
    # Evaluate on test set
    test_metrics = model.evaluate(
        test_data['X_home_seq'],
        test_data['X_away_seq'],
        test_data['X_match'],
        test_y_final
    )
    
    logger.info(f"Model trained in {datetime.now() - start_time}")
    logger.info(f"Test accuracy: {test_metrics.get('accuracy', 0):.4f}")
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"transformer_{dataset_name}_{feature_type}_{timestamp}"
    model_path = os.path.join(SEQUENCE_MODELS_DIR, model_filename)
    os.makedirs(SEQUENCE_MODELS_DIR, exist_ok=True)
    saved_path = model.save(model_path)
    
    # Save training details
    training_details = {
        "model_type": "transformer",
        "dataset_name": dataset_name,
        "feature_type": feature_type,
        "sequence_length": sequence_length,
        "team_features": team_features,
        "match_features": match_features,
        "model_params": model_params,
        "train_size": len(train_indices),
        "test_size": len(test_indices),
        "accuracy": float(test_metrics.get('accuracy', 0)),
        "training_time": str(datetime.now() - start_time),
        "timestamp": timestamp,
        "model_path": saved_path,
        "cv_type": cv_type,
        "temporal_params": temporal_params,
        "training_history": {
            "train_loss": training_results.get("train_loss"),
            "train_acc": training_results.get("train_acc"),
            "val_loss": training_results.get("val_loss"),
            "val_acc": training_results.get("val_acc"),
            "epochs": training_results.get("epochs")
        }
    }
    
    # Save training details
    training_details_path = os.path.join(TRAINING_DIR, f"transformer_{dataset_name}_{feature_type}_{timestamp}.json")
    with open(training_details_path, 'w') as f:
        json.dump(training_details, f, indent=4, default=str)
    
    return training_details


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