"""
Pydantic models for API requests and responses.

This module defines the data models used in the Soccer Prediction System API.
These models validate incoming requests and provide structure for API responses.
"""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, conint, confloat, validator


# Enums
class MatchStatus(str, Enum):
    """Possible statuses for a soccer match."""
    SCHEDULED = "SCHEDULED"
    LIVE = "LIVE"
    FINISHED = "FINISHED"
    POSTPONED = "POSTPONED"
    CANCELLED = "CANCELLED"


class PredictionResult(str, Enum):
    """Possible predicted results for a match."""
    HOME_WIN = "HOME_WIN"
    DRAW = "DRAW"
    AWAY_WIN = "AWAY_WIN"


class ModelType(str, Enum):
    """Types of prediction models available."""
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"
    LOGISTIC_REGRESSION = "logistic_regression"


# Base models
class TeamBase(BaseModel):
    """Base model for team data."""
    id: int = Field(..., description="Unique identifier for the team")
    name: str = Field(..., description="Team name")
    country: str = Field(..., description="Country the team is from")


class MatchBase(BaseModel):
    """Base model for match data."""
    id: int = Field(..., description="Unique identifier for the match")
    date: str = Field(..., description="Date of the match in ISO format (YYYY-MM-DD)")
    home_team_id: int = Field(..., description="ID of the home team")
    home_team: str = Field(..., description="Name of the home team")
    away_team_id: int = Field(..., description="ID of the away team")
    away_team: str = Field(..., description="Name of the away team")
    competition_id: int = Field(..., description="ID of the competition")
    competition: str = Field(..., description="Name of the competition")


class PredictionModelBase(BaseModel):
    """Base model for prediction model data."""
    name: str = Field(..., description="Name of the prediction model")
    description: str = Field(..., description="Description of the model")
    accuracy: float = Field(..., description="Accuracy score of the model", ge=0.0, le=1.0)
    last_trained: str = Field(..., description="Date when model was last trained in ISO format (YYYY-MM-DD)")


# Request models
class LoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(..., description="Username or email address")
    password: str = Field(..., description="User password", min_length=8)
    
    class Config:
        schema_extra = {
            "example": {
                "username": "user@example.com",
                "password": "password123"
            }
        }


class TokenRefreshRequest(BaseModel):
    """Request model for refreshing an access token."""
    refresh_token: str = Field(..., description="Refresh token obtained from login")
    
    class Config:
        schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class CustomPredictionRequest(BaseModel):
    """Request model for custom match prediction."""
    home_team_id: int = Field(..., description="ID of the home team")
    away_team_id: int = Field(..., description="ID of the away team")
    match_date: Optional[str] = Field(None, description="Date of the match in ISO format (YYYY-MM-DD)")
    competition_id: Optional[int] = Field(None, description="ID of the competition")
    features: Optional[Dict[str, Any]] = Field(None, description="Custom features for the prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "home_team_id": 1,
                "away_team_id": 2,
                "match_date": "2023-01-15",
                "competition_id": 1,
                "features": {
                    "home_team_form": 0.75,
                    "away_team_form": 0.65,
                    "home_team_ranking": 3,
                    "away_team_ranking": 7
                }
            }
        }


class BatchMatchItem(BaseModel):
    """Model for a single match item in a batch prediction request."""
    home_team_id: int = Field(..., description="ID of the home team")
    away_team_id: int = Field(..., description="ID of the away team")
    match_date: Optional[str] = Field(None, description="Date of the match in ISO format (YYYY-MM-DD)")
    competition_id: Optional[int] = Field(None, description="ID of the competition")
    features: Optional[Dict[str, Any]] = Field(None, description="Custom features for this specific match prediction")


class BatchPredictionRequest(BaseModel):
    """Request model for batch match predictions."""
    matches: List[BatchMatchItem] = Field(..., description="List of matches to predict", min_items=1, max_items=100)
    model_name: Optional[str] = Field(None, description="Name of the prediction model to use")
    
    class Config:
        schema_extra = {
            "example": {
                "matches": [
                    {
                        "home_team_id": 1,
                        "away_team_id": 2,
                        "match_date": "2023-01-15"
                    },
                    {
                        "home_team_id": 3,
                        "away_team_id": 4,
                        "match_date": "2023-01-16"
                    }
                ],
                "model_name": "ensemble"
            }
        }


class ResetCacheRequest(BaseModel):
    """Request model for resetting the cache."""
    scope: str = Field(..., description="Scope of cache to reset (all, teams, matches, predictions)")
    
    @validator('scope')
    def validate_scope(cls, v):
        """Validate the scope field."""
        allowed_scopes = ["all", "teams", "matches", "predictions"]
        if v not in allowed_scopes:
            raise ValueError(f"scope must be one of: {', '.join(allowed_scopes)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "scope": "teams"
            }
        }


# Response models
class ErrorResponse(BaseModel):
    """Model for API error responses."""
    detail: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: Optional[str] = Field(None, description="Time of error in ISO format")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "An error occurred while processing your request",
                "error_code": "INVALID_REQUEST",
                "timestamp": "2023-02-01T12:34:56"
            }
        }


class TokenResponse(BaseModel):
    """Response model for authentication tokens."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token validity in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token (if requested)")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class TeamDetail(TeamBase):
    """Detailed team model with additional information."""
    founded: Optional[int] = Field(None, description="Year the team was founded")
    stadium: Optional[str] = Field(None, description="Home stadium of the team")
    league: Optional[str] = Field(None, description="Current league the team plays in")
    logo_url: Optional[str] = Field(None, description="URL to the team's logo image")
    website: Optional[str] = Field(None, description="Official team website")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Manchester United",
                "country": "England",
                "founded": 1878,
                "stadium": "Old Trafford",
                "league": "Premier League",
                "logo_url": "https://example.com/logos/manchester_united.png",
                "website": "https://www.manutd.com"
            }
        }


class TeamList(BaseModel):
    """Response model for a list of teams."""
    teams: List[TeamBase] = Field(..., description="List of teams")
    total: int = Field(..., description="Total number of teams available")
    
    class Config:
        schema_extra = {
            "example": {
                "teams": [
                    {
                        "id": 1,
                        "name": "Manchester United",
                        "country": "England"
                    },
                    {
                        "id": 2,
                        "name": "Barcelona",
                        "country": "Spain"
                    },
                    {
                        "id": 3,
                        "name": "Bayern Munich",
                        "country": "Germany"
                    }
                ],
                "total": 3
            }
        }


class MatchDetail(MatchBase):
    """Detailed match model with additional information."""
    home_goals: Optional[int] = Field(None, description="Goals scored by the home team")
    away_goals: Optional[int] = Field(None, description="Goals scored by the away team")
    status: MatchStatus = Field(MatchStatus.SCHEDULED, description="Current status of the match")
    venue: Optional[str] = Field(None, description="Venue where the match is played")
    referee: Optional[str] = Field(None, description="Referee officiating the match")
    attendance: Optional[int] = Field(None, description="Number of spectators")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "date": "2023-01-01",
                "home_team_id": 1,
                "home_team": "Manchester United",
                "away_team_id": 2,
                "away_team": "Barcelona",
                "home_goals": 2,
                "away_goals": 1,
                "status": "FINISHED",
                "competition_id": 1,
                "competition": "Champions League",
                "venue": "Old Trafford",
                "referee": "Mike Dean",
                "attendance": 73000
            }
        }


class MatchList(BaseModel):
    """Response model for a list of matches."""
    matches: List[MatchBase] = Field(..., description="List of matches")
    total: int = Field(..., description="Total number of matches available")
    
    class Config:
        schema_extra = {
            "example": {
                "matches": [
                    {
                        "id": 1,
                        "date": "2023-01-01",
                        "home_team_id": 1,
                        "home_team": "Manchester United",
                        "away_team_id": 2,
                        "away_team": "Barcelona",
                        "competition_id": 1,
                        "competition": "Champions League"
                    },
                    {
                        "id": 2,
                        "date": "2023-01-08",
                        "home_team_id": 3,
                        "home_team": "Bayern Munich",
                        "away_team_id": 4,
                        "away_team": "Juventus",
                        "competition_id": 1,
                        "competition": "Champions League"
                    }
                ],
                "total": 2
            }
        }


class MatchBasicInfo(BaseModel):
    """Basic match information model."""
    home_team_id: int = Field(..., description="ID of the home team")
    home_team: str = Field(..., description="Name of the home team")
    away_team_id: int = Field(..., description="ID of the away team")
    away_team: str = Field(..., description="Name of the away team")
    date: str = Field(..., description="Date of the match in ISO format (YYYY-MM-DD)")


class PredictionDetail(BaseModel):
    """Detailed prediction model with probabilities and results."""
    home_win_prob: float = Field(..., description="Probability of home team winning", ge=0.0, le=1.0)
    draw_prob: float = Field(..., description="Probability of a draw", ge=0.0, le=1.0)
    away_win_prob: float = Field(..., description="Probability of away team winning", ge=0.0, le=1.0)
    expected_home_goals: Optional[float] = Field(None, description="Expected goals for home team")
    expected_away_goals: Optional[float] = Field(None, description="Expected goals for away team")
    predicted_result: PredictionResult = Field(..., description="Predicted match result")
    confidence: float = Field(..., description="Confidence level of prediction", ge=0.0, le=1.0)
    model_name: str = Field(..., description="Name of the model used for prediction")


class PredictionResponse(BaseModel):
    """Response model for a single match prediction."""
    match: MatchBasicInfo = Field(..., description="Basic information about the match")
    prediction: PredictionDetail = Field(..., description="Prediction details")
    timestamp: str = Field(..., description="Timestamp of when the prediction was made in ISO format")
    
    class Config:
        schema_extra = {
            "example": {
                "match": {
                    "home_team_id": 1,
                    "home_team": "Manchester United",
                    "away_team_id": 2,
                    "away_team": "Barcelona",
                    "date": "2023-01-15"
                },
                "prediction": {
                    "home_win_prob": 0.45,
                    "draw_prob": 0.25,
                    "away_win_prob": 0.30,
                    "expected_home_goals": 1.5,
                    "expected_away_goals": 1.2,
                    "predicted_result": "HOME_WIN",
                    "confidence": 0.75,
                    "model_name": "ensemble"
                },
                "timestamp": "2023-01-14T12:34:56"
            }
        }


class BatchPredictionItem(BaseModel):
    """Model for a single prediction in a batch response."""
    match: MatchBasicInfo = Field(..., description="Basic information about the match")
    prediction: PredictionDetail = Field(..., description="Prediction details")


class BatchPredictionResponse(BaseModel):
    """Response model for batch match predictions."""
    predictions: List[BatchPredictionItem] = Field(..., description="List of match predictions")
    timestamp: str = Field(..., description="Timestamp of when the predictions were made in ISO format")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "match": {
                            "home_team_id": 1,
                            "home_team": "Manchester United",
                            "away_team_id": 2,
                            "away_team": "Barcelona",
                            "date": "2023-01-15"
                        },
                        "prediction": {
                            "home_win_prob": 0.45,
                            "draw_prob": 0.25,
                            "away_win_prob": 0.30,
                            "predicted_result": "HOME_WIN",
                            "confidence": 0.75,
                            "model_name": "ensemble"
                        }
                    },
                    {
                        "match": {
                            "home_team_id": 3,
                            "home_team": "Bayern Munich",
                            "away_team_id": 4,
                            "away_team": "Juventus",
                            "date": "2023-01-16"
                        },
                        "prediction": {
                            "home_win_prob": 0.35,
                            "draw_prob": 0.40,
                            "away_win_prob": 0.25,
                            "predicted_result": "DRAW",
                            "confidence": 0.65,
                            "model_name": "ensemble"
                        }
                    }
                ],
                "timestamp": "2023-01-14T12:34:56"
            }
        }


class ModelsListResponse(BaseModel):
    """Response model for available prediction models."""
    models: List[PredictionModelBase] = Field(..., description="List of available prediction models")
    
    class Config:
        schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "xgboost",
                        "description": "XGBoost classification model",
                        "accuracy": 0.72,
                        "last_trained": "2023-01-01"
                    },
                    {
                        "name": "neural_network",
                        "description": "Deep neural network model",
                        "accuracy": 0.68,
                        "last_trained": "2023-01-05"
                    },
                    {
                        "name": "ensemble",
                        "description": "Ensemble model combining multiple predictors",
                        "accuracy": 0.75,
                        "last_trained": "2023-01-10"
                    }
                ]
            }
        }


class PredictionHistoryItem(BaseModel):
    """Model for a single prediction history item."""
    id: int = Field(..., description="Unique identifier for the prediction record")
    timestamp: str = Field(..., description="When the prediction was made in ISO format")
    match: MatchBasicInfo = Field(..., description="Basic information about the match")
    model: str = Field(..., description="Name of the model used")
    prediction: PredictionDetail = Field(..., description="Prediction details")


class PredictionHistoryResponse(BaseModel):
    """Response model for prediction history."""
    history: List[PredictionHistoryItem] = Field(..., description="List of historical predictions")
    total: int = Field(..., description="Total number of predictions available")
    
    class Config:
        schema_extra = {
            "example": {
                "history": [
                    {
                        "id": 1,
                        "timestamp": "2023-01-01T12:00:00",
                        "match": {
                            "home_team_id": 1,
                            "home_team": "Manchester United",
                            "away_team_id": 2,
                            "away_team": "Barcelona",
                            "date": "2023-01-15"
                        },
                        "model": "xgboost",
                        "prediction": {
                            "home_win_prob": 0.6,
                            "draw_prob": 0.2,
                            "away_win_prob": 0.2,
                            "predicted_result": "HOME_WIN",
                            "confidence": 0.7,
                            "model_name": "xgboost"
                        }
                    }
                ],
                "total": 1
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Overall health status (ok, degraded, error)")
    version: str = Field(..., description="API version")
    time: str = Field(..., description="Current server time in ISO format")
    services: Dict[str, str] = Field(..., description="Status of individual services")
    uptime: Optional[str] = Field(None, description="System uptime in human-readable format")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "time": "2023-02-01T12:34:56",
                "services": {
                    "database": "up",
                    "redis": "up",
                    "models": "up"
                },
                "uptime": "2d 3h 45m 12s"
            }
        }


class ResetCacheResponse(BaseModel):
    """Response model for reset cache operation."""
    message: str = Field(..., description="Result message")
    deleted_count: int = Field(..., description="Number of cache entries deleted")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Cache teams reset successfully",
                "deleted_count": 15
            }
        }


# Player performance models
class PlayerBase(BaseModel):
    """Base model for player data."""
    id: int = Field(..., description="Unique identifier for the player")
    name: str = Field(..., description="Player name")
    position: Optional[str] = Field(None, description="Player position")
    team_id: int = Field(..., description="ID of the player's team")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1234,
                "name": "John Smith",
                "position": "Forward",
                "team_id": 42
            }
        }


class PlayerPerformanceRequest(BaseModel):
    """Request model for player performance prediction."""
    player_id: int = Field(..., description="ID of the player")
    match_id: int = Field(..., description="ID of the match")
    team_id: int = Field(..., description="ID of the player's team")
    opponent_id: int = Field(..., description="ID of the opponent team")
    is_home: bool = Field(..., description="Whether the player's team is playing at home")
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features for the prediction")
    metrics: Optional[List[str]] = Field(None, description="Performance metrics to predict (if None, predict all available)")
    
    class Config:
        schema_extra = {
            "example": {
                "player_id": 1234,
                "match_id": 5678,
                "team_id": 42,
                "opponent_id": 43,
                "is_home": True,
                "features": {
                    "recent_form": 0.75,
                    "minutes_last_match": 90,
                    "goals_last_5": 3
                },
                "metrics": ["goals", "assists", "minutes_played"]
            }
        }


class BatchPlayerItem(BaseModel):
    """Model for a single player item in a batch prediction request."""
    player_id: int = Field(..., description="ID of the player")
    match_id: int = Field(..., description="ID of the match")
    team_id: int = Field(..., description="ID of the player's team")
    opponent_id: int = Field(..., description="ID of the opponent team")
    is_home: bool = Field(..., description="Whether the player's team is playing at home")
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features for this specific player prediction")


class BatchPlayerPredictionRequest(BaseModel):
    """Request model for batch player performance predictions."""
    player_matches: List[BatchPlayerItem] = Field(..., description="List of player-match combinations to predict", min_items=1, max_items=100)
    metrics: Optional[List[str]] = Field(None, description="Performance metrics to predict")
    
    class Config:
        schema_extra = {
            "example": {
                "player_matches": [
                    {
                        "player_id": 1234,
                        "match_id": 5678,
                        "team_id": 42,
                        "opponent_id": 43,
                        "is_home": True
                    },
                    {
                        "player_id": 5678,
                        "match_id": 5678,
                        "team_id": 42,
                        "opponent_id": 43,
                        "is_home": True
                    }
                ],
                "metrics": ["goals", "assists", "minutes_played"]
            }
        }


class PlayerPerformanceMetric(BaseModel):
    """Model for a single performance metric prediction."""
    metric: str = Field(..., description="Name of the performance metric")
    value: Optional[float] = Field(None, description="Predicted value for the metric")


class PlayerPerformanceDetail(BaseModel):
    """Detailed player performance prediction model."""
    player_id: int = Field(..., description="ID of the player")
    match_id: int = Field(..., description="ID of the match")
    team_id: int = Field(..., description="ID of the player's team")
    opponent_id: int = Field(..., description="ID of the opponent team")
    is_home: bool = Field(..., description="Whether the player's team is playing at home")
    predictions: Dict[str, Optional[float]] = Field(..., description="Predictions for each requested metric")


class PlayerPerformanceResponse(BaseModel):
    """Response model for a player performance prediction."""
    prediction: PlayerPerformanceDetail = Field(..., description="Prediction details")
    timestamp: str = Field(..., description="Timestamp of when the prediction was made in ISO format")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": {
                    "player_id": 1234,
                    "match_id": 5678,
                    "team_id": 42,
                    "opponent_id": 43,
                    "is_home": True,
                    "predictions": {
                        "goals": 0.75,
                        "assists": 0.25,
                        "minutes_played": 85.5
                    }
                },
                "timestamp": "2023-03-15T14:30:00.000Z"
            }
        }


class BatchPlayerPredictionItem(BaseModel):
    """Model for a single player performance prediction in a batch response."""
    player_id: int = Field(..., description="ID of the player")
    match_id: int = Field(..., description="ID of the match")
    predictions: Dict[str, Optional[float]] = Field(..., description="Predictions for each requested metric")


class BatchPlayerPredictionResponse(BaseModel):
    """Response model for batch player performance predictions."""
    predictions: List[BatchPlayerPredictionItem] = Field(..., description="List of player performance predictions")
    timestamp: str = Field(..., description="Timestamp of when the predictions were made in ISO format")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "player_id": 1234,
                        "match_id": 5678,
                        "predictions": {
                            "goals": 0.75,
                            "assists": 0.25,
                            "minutes_played": 85.5
                        }
                    },
                    {
                        "player_id": 5678,
                        "match_id": 5678,
                        "predictions": {
                            "goals": 0.5,
                            "assists": 0.75,
                            "minutes_played": 90.0
                        }
                    }
                ],
                "timestamp": "2023-03-15T14:30:00.000Z"
            }
        } 