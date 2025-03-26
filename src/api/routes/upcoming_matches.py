"""
API Routes for Upcoming Matches

This module provides API endpoints for retrieving and predicting upcoming football matches.
"""

from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

# Import project components
from src.models.prediction import PredictionService
from src.data.fixtures import get_upcoming_fixtures, refresh_fixture_data

# Create router
router = APIRouter(
    prefix="/api/matches/upcoming",
    tags=["Upcoming Matches"],
    responses={404: {"description": "Not found"}},
)

# Get prediction service
def get_prediction_service():
    """Get the prediction service singleton."""
    from src.models.prediction import PredictionService
    return PredictionService()


@router.get("/", response_model=Dict[str, Any])
async def get_fixtures(
    days: int = Query(30, description="Number of days ahead to include"),
    team: Optional[str] = Query(None, description="Filter by team name"),
    league: Optional[str] = Query(None, description="Filter by league code"),
    page: int = Query(1, description="Page number"),
    page_size: int = Query(20, description="Items per page"),
):
    """
    Get upcoming fixtures with optional filtering.
    """
    # Get fixtures
    fixtures_df = get_upcoming_fixtures(days, team, league)
    
    if fixtures_df.empty:
        return {
            "total": 0,
            "page": page,
            "page_size": page_size,
            "items": []
        }
    
    # Calculate total
    total = len(fixtures_df)
    
    # Apply pagination
    start = (page - 1) * page_size
    end = start + page_size
    
    # Get paginated subset
    paginated_df = fixtures_df.iloc[start:end]
    
    # Convert to list of dicts
    fixtures_list = []
    for _, row in paginated_df.iterrows():
        fixture = row.to_dict()
        
        # Convert any datetime objects to ISO format strings
        for key, value in fixture.items():
            if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
                fixture[key] = value.isoformat()
        
        fixtures_list.append(fixture)
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": fixtures_list
    }


@router.get("/refresh", response_model=Dict[str, Any])
async def refresh_fixtures(force: bool = Query(False, description="Force refresh even if cache is recent")):
    """
    Refresh upcoming fixtures data from the source.
    """
    count = refresh_fixture_data(force)
    
    return {
        "status": "success",
        "message": f"Refreshed {count} fixtures",
        "count": count
    }


@router.get("/predictions", response_model=Dict[str, Any])
async def predict_upcoming_fixtures(
    days: int = Query(30, description="Number of days ahead to include"),
    team: Optional[str] = Query(None, description="Filter by team name"),
    league: Optional[str] = Query(None, description="Filter by league code"),
    model: Optional[str] = Query(None, description="Model to use for prediction"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Get predictions for upcoming fixtures with optional filtering.
    """
    # Get fixtures
    fixtures_df = get_upcoming_fixtures(days, team, league)
    
    if fixtures_df.empty:
        return {
            "status": "error",
            "message": "No upcoming fixtures found",
            "predictions": []
        }
    
    # Make predictions
    predictions = prediction_service.predict_upcoming_matches(
        upcoming_fixtures=fixtures_df,
        model_name=model
    )
    
    return {
        "status": "success",
        "count": len(predictions),
        "predictions": predictions
    }


@router.get("/predict/{home_team}/{away_team}", response_model=Dict[str, Any])
async def predict_specific_match(
    home_team: str,
    away_team: str,
    date: Optional[str] = Query(None, description="Match date (YYYY-MM-DD)"),
    model: Optional[str] = Query(None, description="Model to use for prediction"),
    include_features: bool = Query(True, description="Include feature data in response"),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Get a prediction for a specific match.
    """
    try:
        prediction = prediction_service.predict_specific_match(
            home_team=home_team,
            away_team=away_team,
            match_date=date,
            model_name=model,
            include_features=include_features
        )
        
        return {
            "status": "success",
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teams", response_model=Dict[str, Any])
async def get_available_teams():
    """
    Get a list of teams with upcoming fixtures.
    """
    # Get all fixtures for the next 60 days
    fixtures_df = get_upcoming_fixtures(days=60)
    
    if fixtures_df.empty:
        return {
            "status": "success",
            "teams": []
        }
    
    # Extract unique teams
    home_teams = fixtures_df["HomeTeam"].unique().tolist() if "HomeTeam" in fixtures_df.columns else []
    away_teams = fixtures_df["AwayTeam"].unique().tolist() if "AwayTeam" in fixtures_df.columns else []
    
    # Combine and deduplicate
    all_teams = list(set(home_teams + away_teams))
    all_teams.sort()
    
    return {
        "status": "success",
        "count": len(all_teams),
        "teams": all_teams
    }


@router.get("/leagues", response_model=Dict[str, Any])
async def get_available_leagues():
    """
    Get a list of leagues with upcoming fixtures.
    """
    # Get all fixtures for the next 60 days
    fixtures_df = get_upcoming_fixtures(days=60)
    
    if fixtures_df.empty:
        return {
            "status": "success",
            "leagues": []
        }
    
    # Extract unique leagues
    leagues = []
    if "League" in fixtures_df.columns and "LeagueName" in fixtures_df.columns:
        for _, row in fixtures_df[["League", "LeagueName"]].drop_duplicates().iterrows():
            leagues.append({
                "code": row["League"],
                "name": row["LeagueName"]
            })
    
    return {
        "status": "success",
        "count": len(leagues),
        "leagues": leagues
    } 