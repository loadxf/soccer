"""
Prediction module for soccer prediction system.
Provides unified interfaces for making predictions with various models.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from pathlib import Path

# Import project components
from src.utils.logger import get_logger
from src.models.baseline import BaselineMatchPredictor
from src.models.training import predict_with_ensemble

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback default if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("models.prediction")

# Define model directories
MODELS_DIR = os.path.join(DATA_DIR, "models")
ENSEMBLE_DIR = os.path.join(MODELS_DIR, "ensembles")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


class PredictionService:
    """
    Service for making predictions using trained models.
    Provides a unified interface for prediction with single models or ensembles.
    """
    
    def __init__(self, default_model_path: Optional[str] = None):
        """
        Initialize the prediction service.
        
        Args:
            default_model_path: Path to the default model or ensemble to use
        """
        self.models = {}
        self.ensembles = {}
        self.default_model_path = default_model_path
        self.default_model = None
        
        # Load default model if provided
        if default_model_path:
            self.load_default_model()
        
        # Scan for available models and ensembles
        self.scan_available_models()
    
    def load_default_model(self):
        """Load the default model."""
        if self.default_model_path:
            try:
                # Check if it's an ensemble
                if "ensemble" in self.default_model_path:
                    logger.info(f"Loading default ensemble: {self.default_model_path}")
                    # We don't load the ensemble until prediction time
                    self.default_model = {"type": "ensemble", "path": self.default_model_path}
                else:
                    logger.info(f"Loading default model: {self.default_model_path}")
                    self.default_model = BaselineMatchPredictor.load(self.default_model_path)
            except Exception as e:
                logger.error(f"Error loading default model: {e}")
    
    def scan_available_models(self):
        """Scan for available models and ensembles."""
        # Scan for individual models
        for item in os.listdir(MODELS_DIR):
            if item.endswith(".pkl") and os.path.isfile(os.path.join(MODELS_DIR, item)):
                model_path = os.path.join(MODELS_DIR, item)
                model_name = item.replace(".pkl", "")
                self.models[model_name] = {"path": model_path, "loaded": False, "model": None}
                logger.debug(f"Found model: {model_name}")
        
        # Scan for ensembles
        if os.path.exists(ENSEMBLE_DIR):
            for item in os.listdir(ENSEMBLE_DIR):
                if item.endswith(".json") and os.path.isfile(os.path.join(ENSEMBLE_DIR, item)):
                    ensemble_path = os.path.join(ENSEMBLE_DIR, item)
                    ensemble_name = item.replace(".json", "")
                    self.ensembles[ensemble_name] = {"path": ensemble_path}
                    logger.debug(f"Found ensemble: {ensemble_name}")
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return False
        
        if self.models[model_name]["loaded"]:
            logger.debug(f"Model {model_name} already loaded")
            return True
        
        try:
            model_path = self.models[model_name]["path"]
            self.models[model_name]["model"] = BaselineMatchPredictor.load(model_path)
            self.models[model_name]["loaded"] = True
            logger.info(f"Loaded model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def predict_match(
        self,
        home_team_id: int,
        away_team_id: int,
        model_name: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Predict the outcome of a match.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            model_name: Name of the model to use (if None, use default)
            features: Optional additional features for the prediction
            return_probabilities: Whether to return probabilities in addition to the prediction
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Determine which model to use
        if model_name:
            # Check if it's an ensemble
            if model_name in self.ensembles:
                return self.predict_with_ensemble(
                    home_team_id, 
                    away_team_id, 
                    ensemble_name=model_name, 
                    features=features
                )
            
            # Otherwise it's a single model
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return {"error": f"Model {model_name} not found"}
            
            # Load model if not already loaded
            if not self.models[model_name]["loaded"]:
                if not self.load_model(model_name):
                    return {"error": f"Failed to load model {model_name}"}
            
            model = self.models[model_name]["model"]
        else:
            # Use default model
            if not self.default_model:
                logger.error("No default model specified")
                return {"error": "No default model specified"}
            
            # Check if default is an ensemble
            if isinstance(self.default_model, dict) and self.default_model["type"] == "ensemble":
                return self.predict_with_ensemble(
                    home_team_id, 
                    away_team_id, 
                    ensemble_path=self.default_model["path"], 
                    features=features
                )
            
            model = self.default_model
        
        # Make prediction
        try:
            result = model.predict_match(home_team_id, away_team_id, features)
            
            # Record prediction
            self._record_prediction(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                model_name=model_name or "default",
                prediction=result,
                is_ensemble=False
            )
            
            return result
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"error": f"Prediction error: {str(e)}"}
    
    def predict_with_ensemble(
        self,
        home_team_id: int,
        away_team_id: int,
        ensemble_name: Optional[str] = None,
        ensemble_path: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict the outcome of a match using an ensemble.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            ensemble_name: Name of the ensemble to use
            ensemble_path: Path to the ensemble to use (alternative to ensemble_name)
            features: Optional additional features for the prediction
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        # Get ensemble path
        if ensemble_name:
            if ensemble_name not in self.ensembles:
                logger.error(f"Ensemble {ensemble_name} not found")
                return {"error": f"Ensemble {ensemble_name} not found"}
            ensemble_path = self.ensembles[ensemble_name]["path"]
        elif not ensemble_path:
            logger.error("No ensemble specified")
            return {"error": "No ensemble specified"}
        
        # Make prediction
        try:
            result = predict_with_ensemble(
                ensemble_path=ensemble_path,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                features=features
            )
            
            # Record prediction
            self._record_prediction(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                model_name=ensemble_name or os.path.basename(ensemble_path),
                prediction=result,
                is_ensemble=True
            )
            
            return result
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return {"error": f"Ensemble prediction error: {str(e)}"}
    
    def _record_prediction(
        self,
        home_team_id: int,
        away_team_id: int,
        model_name: str,
        prediction: Dict[str, Any],
        is_ensemble: bool
    ):
        """Record a prediction for future analysis."""
        record = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "model_name": model_name,
            "is_ensemble": is_ensemble,
            "prediction": prediction,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_filename = f"prediction_{home_team_id}_{away_team_id}_{model_name}_{timestamp}.json"
        record_path = os.path.join(PREDICTIONS_DIR, record_filename)
        
        # Save record
        with open(record_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
    
    def batch_predict(
        self,
        matches: List[Dict[str, Any]],
        model_name: Optional[str] = None,
        extra_features: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple matches.
        
        Args:
            matches: List of match dictionaries with 'home_team_id' and 'away_team_id'
            model_name: Name of the model to use (if None, use default)
            extra_features: Dictionary of extra features keyed by match_id
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        results = []
        
        for match in matches:
            home_team_id = match.get("home_team_id")
            away_team_id = match.get("away_team_id")
            match_id = match.get("match_id", None)
            
            if not home_team_id or not away_team_id:
                logger.error(f"Missing team IDs in match: {match}")
                results.append({"error": "Missing team IDs", "match": match})
                continue
            
            # Get extra features for this match if available
            features = None
            if extra_features and match_id in extra_features:
                features = extra_features[match_id]
            
            # Make prediction
            prediction = self.predict_match(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                model_name=model_name,
                features=features
            )
            
            # Add match info to prediction
            prediction["match_id"] = match_id
            prediction["home_team_id"] = home_team_id
            prediction["away_team_id"] = away_team_id
            
            results.append(prediction)
        
        return results
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available models.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of model information
        """
        model_info = {}
        
        # Get info for individual models
        for name, info in self.models.items():
            # Load model info if not already loaded
            if not info["loaded"]:
                try:
                    # Try to extract basic info without fully loading the model
                    model_path = info["path"]
                    model_info[name] = {
                        "type": "single",
                        "path": model_path,
                        "filename": os.path.basename(model_path),
                        "last_modified": datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
                    }
                except Exception as e:
                    logger.error(f"Error getting info for model {name}: {e}")
                    model_info[name] = {
                        "type": "single",
                        "path": info["path"],
                        "error": str(e)
                    }
            else:
                # Model is loaded, get full info
                model = info["model"]
                model_info[name] = {
                    "type": "single",
                    "path": info["path"],
                    "model_type": model.model_type,
                    "dataset_name": model.dataset_name,
                    "feature_type": model.feature_type,
                    "info": model.model_info
                }
        
        # Get info for ensembles
        for name, info in self.ensembles.items():
            try:
                with open(info["path"], "r") as f:
                    ensemble_data = json.load(f)
                
                model_info[name] = {
                    "type": "ensemble",
                    "path": info["path"],
                    "ensemble_type": ensemble_data.get("ensemble_type", "unknown"),
                    "n_models": len(ensemble_data.get("models", [])),
                    "created_at": ensemble_data.get("created_at", None)
                }
            except Exception as e:
                logger.error(f"Error getting info for ensemble {name}: {e}")
                model_info[name] = {
                    "type": "ensemble",
                    "path": info["path"],
                    "error": str(e)
                }
        
        return model_info
    
    def get_prediction_history(
        self,
        limit: int = 100,
        home_team_id: Optional[int] = None,
        away_team_id: Optional[int] = None,
        model_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical predictions.
        
        Args:
            limit: Maximum number of predictions to return
            home_team_id: Filter by home team ID
            away_team_id: Filter by away team ID
            model_name: Filter by model name
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            List[Dict[str, Any]]: List of historical predictions
        """
        predictions = []
        
        # Convert dates to datetime objects for comparison
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        # List all prediction files
        prediction_files = [f for f in os.listdir(PREDICTIONS_DIR) if f.endswith(".json")]
        
        # Sort by modification time (newest first)
        prediction_files.sort(key=lambda x: os.path.getmtime(os.path.join(PREDICTIONS_DIR, x)), reverse=True)
        
        count = 0
        for filename in prediction_files:
            if count >= limit:
                break
                
            try:
                file_path = os.path.join(PREDICTIONS_DIR, filename)
                with open(file_path, "r") as f:
                    prediction = json.load(f)
                
                # Apply filters
                if home_team_id is not None and prediction.get("home_team_id") != home_team_id:
                    continue
                    
                if away_team_id is not None and prediction.get("away_team_id") != away_team_id:
                    continue
                    
                if model_name is not None and prediction.get("model_name") != model_name:
                    continue
                
                # Apply date filters
                timestamp = prediction.get("timestamp")
                if timestamp:
                    pred_dt = datetime.fromisoformat(timestamp)
                    if start_dt and pred_dt < start_dt:
                        continue
                    if end_dt and pred_dt > end_dt:
                        continue
                
                predictions.append(prediction)
                count += 1
                
            except Exception as e:
                logger.error(f"Error reading prediction file {filename}: {e}")
        
        return predictions
    
    def predict_upcoming_matches(
        self,
        upcoming_fixtures: pd.DataFrame,
        model_name: Optional[str] = None,
        include_features: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for upcoming fixtures.
        
        Args:
            upcoming_fixtures: DataFrame of upcoming matches
            model_name: Name of the model to use (if None, use default)
            include_features: Whether to include feature data in the results
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        if upcoming_fixtures.empty:
            logger.warning("No upcoming fixtures provided for prediction")
            return []
        
        logger.info(f"Predicting {len(upcoming_fixtures)} upcoming matches")
        
        # Prepare fixtures for prediction
        matches = []
        for _, fixture in upcoming_fixtures.iterrows():
            # Extract home and away teams
            if "HomeTeam" not in fixture or "AwayTeam" not in fixture:
                logger.warning("Missing team information in fixture")
                continue
            
            home_team = fixture["HomeTeam"]
            away_team = fixture["AwayTeam"]
            
            # Convert team names to IDs if needed
            # In this implementation, we use team names directly
            home_team_id = home_team
            away_team_id = away_team
            
            # Create match entry
            match = {
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team": home_team,
                "away_team": away_team,
                "date": fixture.get("Date", datetime.now()).isoformat() if not pd.isna(fixture.get("Date")) else datetime.now().isoformat(),
                "league": fixture.get("League", ""),
                "league_name": fixture.get("LeagueName", "")
            }
            
            # Prepare features if available
            try:
                from src.data.fixtures import prepare_match_features
                features = prepare_match_features(home_team, away_team)
                if features and "features" in features:
                    match["features"] = features["features"]
            except ImportError:
                logger.warning("Fixtures module not available, making prediction without additional features")
                match["features"] = {}
            
            matches.append(match)
        
        # Make batch prediction
        predictions = self.batch_predict(matches, model_name=model_name)
        
        # Format results
        results = []
        for prediction, match in zip(predictions, matches):
            result = {
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "date": match["date"],
                "league": match.get("league", ""),
                "league_name": match.get("league_name", ""),
                "prediction": prediction["prediction"],
                "probabilities": prediction.get("probabilities", {})
            }
            
            # Include features if requested
            if include_features and "features" in match:
                result["features"] = match["features"]
            
            results.append(result)
        
        return results
    
    def predict_specific_match(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[str] = None,
        model_name: Optional[str] = None,
        include_features: bool = True
    ) -> Dict[str, Any]:
        """
        Make a prediction for a specific match with detailed features.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date in ISO format (YYYY-MM-DD)
            model_name: Name of the model to use (if None, use default)
            include_features: Whether to include feature data in the result
            
        Returns:
            Dict[str, Any]: Prediction result with details
        """
        logger.info(f"Predicting specific match: {home_team} vs {away_team}")
        
        # Prepare features
        features = {}
        try:
            from src.data.fixtures import prepare_match_features
            feature_data = prepare_match_features(home_team, away_team, match_date)
            if feature_data and "features" in feature_data:
                features = feature_data["features"]
        except ImportError:
            logger.warning("Fixtures module not available, making prediction without additional features")
        
        # Format match for prediction
        match = {
            "home_team_id": home_team,  # Using team name as ID
            "away_team_id": away_team,
            "features": features
        }
        
        # Make prediction
        if model_name:
            if model_name in self.ensembles:
                prediction = self.predict_with_ensemble(
                    home_team_id=home_team,
                    away_team_id=away_team,
                    ensemble_name=model_name,
                    features=features
                )
            else:
                # Load the model if needed
                if model_name in self.models and not self.models[model_name]["loaded"]:
                    self.load_model(model_name)
                
                if model_name in self.models and self.models[model_name]["loaded"]:
                    model = self.models[model_name]["model"]
                    prediction = model.predict_match(home_team, away_team, features)
                else:
                    logger.error(f"Model {model_name} not found or could not be loaded")
                    prediction = {"error": f"Model {model_name} not found or could not be loaded"}
        else:
            # Use default model
            prediction = self.predict_match(home_team, away_team, features=features)
        
        # Format result
        result = {
            "home_team": home_team,
            "away_team": away_team,
            "date": match_date or datetime.now().isoformat(),
            "prediction": prediction.get("prediction", "unknown"),
            "probabilities": prediction.get("probabilities", {})
        }
        
        # Include features if requested
        if include_features and features:
            result["features"] = features
        
        # Include feature importance if available
        if "feature_importance" in prediction:
            result["feature_importance"] = prediction["feature_importance"]
        
        return result
    
    def get_upcoming_fixtures(self, days_ahead: int = 30, team: Optional[str] = None, 
                             league: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get upcoming fixtures with optional filtering.
        
        Args:
            days_ahead: Number of days ahead to include
            team: Filter by team name
            league: Filter by league code
            
        Returns:
            List[Dict[str, Any]]: List of upcoming fixtures
        """
        try:
            from src.data.fixtures import get_upcoming_fixtures
            
            # Get fixtures
            fixtures_df = get_upcoming_fixtures(days_ahead, team, league)
            
            if fixtures_df.empty:
                return []
            
            # Convert to list of dicts
            fixtures_list = []
            for _, row in fixtures_df.iterrows():
                fixture = row.to_dict()
                
                # Convert any datetime objects to ISO format strings
                for key, value in fixture.items():
                    if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                        fixture[key] = value.isoformat()
                
                fixtures_list.append(fixture)
            
            return fixtures_list
        
        except ImportError:
            logger.error("Fixtures module not available")
            return []
    
    def predict_upcoming_fixture(
        self,
        fixture: Dict[str, Any],
        model_name: Optional[str] = None,
        include_features: bool = True
    ) -> Dict[str, Any]:
        """
        Make a prediction for a specific upcoming fixture.
        
        Args:
            fixture: Fixture data dictionary
            model_name: Name of the model to use (if None, use default)
            include_features: Whether to include feature data in the result
            
        Returns:
            Dict[str, Any]: Prediction result
        """
        # Extract team information
        home_team = fixture.get("HomeTeam")
        away_team = fixture.get("AwayTeam")
        
        if not home_team or not away_team:
            logger.error("Missing team information in fixture")
            return {"error": "Missing team information"}
        
        # Extract date if available
        match_date = None
        if "Date" in fixture:
            date_value = fixture["Date"]
            if isinstance(date_value, str):
                match_date = date_value
            elif isinstance(date_value, (pd.Timestamp, datetime)):
                match_date = date_value.isoformat()
        
        # Make prediction
        return self.predict_specific_match(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            model_name=model_name,
            include_features=include_features
        )


# Singleton instance for easy import elsewhere
prediction_service = PredictionService()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Soccer match prediction")
    parser.add_argument("--home", type=int, required=True,
                        help="Home team ID")
    parser.add_argument("--away", type=int, required=True,
                        help="Away team ID")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to use for prediction")
    parser.add_argument("--list-models", action="store_true", 
                        help="List all available models")
    parser.add_argument("--set-default", type=str, default=None,
                        help="Set default model")
    
    args = parser.parse_args()
    
    service = PredictionService()
    
    if args.list_models:
        models = service.get_available_models()
        print("\nAvailable Models:")
        print("-" * 80)
        for name, info in models.items():
            if info["type"] == "single":
                model_type = info.get("model_type", "unknown")
                print(f"{name:<30} (Type: {model_type:<15} Single model)")
            else:
                ensemble_type = info.get("ensemble_type", "unknown")
                n_models = info.get("n_models", "?")
                print(f"{name:<30} (Type: {ensemble_type:<15} Ensemble with {n_models} models)")
        print("-" * 80)
    
    elif args.set_default:
        service.default_model_path = args.set_default
        service.load_default_model()
        print(f"Default model set to: {args.set_default}")
    
    elif args.home and args.away:
        result = service.predict_match(
            home_team_id=args.home,
            away_team_id=args.away,
            model_name=args.model
        )
        
        print("\nMatch Prediction:")
        print("-" * 80)
        print(f"Home Team ID: {args.home}")
        print(f"Away Team ID: {args.away}")
        print(f"Model: {args.model or 'default'}")
        print("-" * 80)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Prediction: {result['prediction']}")
            print(f"Home Win: {result['home_win_probability']:.4f}")
            print(f"Draw:     {result['draw_probability']:.4f}")
            print(f"Away Win: {result['away_win_probability']:.4f}")
            print(f"Confidence: {result['confidence']:.4f}")
        print("-" * 80) 