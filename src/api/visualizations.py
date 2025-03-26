"""
API endpoints for data visualizations.

This module provides API endpoints for generating various visualizations
of soccer data, model performance, and predictions.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body, status
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel

from src.api.auth import get_current_user
from src.utils.visualization import (
    create_team_performance_chart,
    create_prediction_confidence_chart,
    create_model_comparison_chart,
    create_feature_importance_chart,
    create_confusion_matrix_chart,
    create_roc_curve_chart,
    create_prediction_history_chart,
    create_player_performance_chart,
    create_league_standings_chart,
    VisualizationError
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/visualizations",
    tags=["visualizations"],
    dependencies=[Depends(get_current_user)]
)

class MatchIdRequest(BaseModel):
    match_ids: List[str]

class ModelIdRequest(BaseModel):
    model_ids: List[str]

@router.get("/team-performance")
async def get_team_performance_chart(
    team_name: str = Query(..., description="Name of the team"),
    last_n_matches: int = Query(10, description="Number of last matches to include")
):
    """
    Generate a visualization of a team's recent performance.
    """
    try:
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        data = pd.DataFrame({
            'team_name': [team_name] * last_n_matches,
            'opponent': [f'Opponent {i}' for i in range(last_n_matches)],
            'match_date': pd.date_range(end=pd.Timestamp.now(), periods=last_n_matches),
            'goals_scored': np.random.randint(0, 5, size=last_n_matches),
            'goals_conceded': np.random.randint(0, 4, size=last_n_matches)
        })
        
        image = create_team_performance_chart(data, team_name, last_n_matches)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for team {team_name}"
            )
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating team performance chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate team performance chart: {str(e)}"
        )

@router.post("/prediction-confidence")
async def get_prediction_confidence_chart(
    request: MatchIdRequest
):
    """
    Generate a visualization of prediction confidence for selected matches.
    """
    try:
        match_ids = request.match_ids
        
        if len(match_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No match IDs provided"
            )
        
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        predictions = []
        actual_results = []
        
        for i, match_id in enumerate(match_ids):
            home_team = f"Team {i*2 + 1}"
            away_team = f"Team {i*2 + 2}"
            
            # Generate random probabilities that sum to 1
            probs = np.random.dirichlet(np.ones(3))
            
            predictions.append({
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "probabilities": {
                    "home_win": probs[0],
                    "draw": probs[1],
                    "away_win": probs[2]
                }
            })
            
            # Generate a random actual result
            home_score = np.random.randint(0, 4)
            away_score = np.random.randint(0, 4)
            actual_results.append({
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score
            })
        
        image = create_prediction_confidence_chart(predictions, actual_results)
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating prediction confidence chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate prediction confidence chart: {str(e)}"
        )

@router.post("/model-comparison")
async def get_model_comparison_chart(
    request: ModelIdRequest
):
    """
    Generate a visualization comparing different model performances.
    """
    try:
        model_ids = request.model_ids
        
        if len(model_ids) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least two model IDs are required for comparison"
            )
        
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
        model_metrics = {}
        
        for model_id in model_ids:
            # Generate random metrics between 0.6 and 0.9
            model_metrics[model_id] = {
                metric: np.random.uniform(0.6, 0.95) for metric in metrics
            }
        
        image = create_model_comparison_chart(model_metrics)
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating model comparison chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate model comparison chart: {str(e)}"
        )

@router.get("/feature-importance")
async def get_feature_importance_chart(
    model_id: str = Query(..., description="ID of the model"),
    top_n: int = Query(20, description="Number of top features to display")
):
    """
    Generate a visualization of feature importances for a model.
    """
    try:
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        features = [
            "home_team_ranking", "away_team_ranking", "home_team_form", "away_team_form",
            "home_goals_last_5", "away_goals_last_5", "home_conceded_last_5", "away_conceded_last_5",
            "head_to_head_wins", "head_to_head_draws", "head_to_head_losses",
            "home_team_injuries", "away_team_injuries", "home_team_possession_avg",
            "away_team_possession_avg", "home_team_shots_avg", "away_team_shots_avg",
            "home_team_goals_avg", "away_team_goals_avg", "match_importance",
            "days_since_last_match_home", "days_since_last_match_away",
            "home_team_goal_diff", "away_team_goal_diff", "home_team_clean_sheets",
            "away_team_clean_sheets", "home_team_failed_to_score", "away_team_failed_to_score",
            "home_team_key_player_rating", "away_team_key_player_rating"
        ]
        
        # Generate random importances
        importances = np.random.dirichlet(np.ones(len(features))) * 10
        
        feature_importances = {
            features[i]: importances[i] for i in range(len(features))
        }
        
        image = create_feature_importance_chart(feature_importances, top_n)
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating feature importance chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate feature importance chart: {str(e)}"
        )

@router.get("/confusion-matrix")
async def get_confusion_matrix_chart(
    model_id: str = Query(..., description="ID of the model"),
    dataset_id: str = Query(..., description="ID of the dataset")
):
    """
    Generate a confusion matrix visualization for a model on a dataset.
    """
    try:
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        n_samples = 200
        
        # Generate random true labels
        y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.2, 0.4])
        
        # Generate predicted labels with some error
        y_pred = []
        for label in y_true:
            if np.random.random() < 0.8:  # 80% chance of correct prediction
                y_pred.append(label)
            else:
                # Choose a wrong label
                wrong_labels = [0, 1, 2]
                wrong_labels.remove(label)
                y_pred.append(np.random.choice(wrong_labels))
        
        image = create_confusion_matrix_chart(y_true, y_pred)
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating confusion matrix chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate confusion matrix chart: {str(e)}"
        )

@router.get("/roc-curve")
async def get_roc_curve_chart(
    model_id: str = Query(..., description="ID of the model"),
    dataset_id: str = Query(..., description="ID of the dataset")
):
    """
    Generate a ROC curve visualization for a model on a dataset.
    """
    try:
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        n_samples = 100
        
        # Generate random true labels (one-hot encoded)
        y_true = np.zeros((n_samples, 3))
        true_classes = np.random.choice([0, 1, 2], size=n_samples)
        for i, label in enumerate(true_classes):
            y_true[i, label] = 1
        
        # Generate random prediction scores
        y_score = np.zeros((n_samples, 3))
        for i in range(n_samples):
            # High probability for the true class, lower for others
            probs = np.random.dirichlet(np.ones(3) * 2)
            if np.random.random() < 0.8:  # 80% chance of the true class having highest probability
                max_idx = np.argmax(y_true[i])
                other_idx = [j for j in range(3) if j != max_idx]
                probs[max_idx] = np.random.uniform(0.6, 0.9)
                remaining = 1 - probs[max_idx]
                probs[other_idx[0]] = remaining * np.random.random()
                probs[other_idx[1]] = remaining - probs[other_idx[0]]
            y_score[i] = probs
        
        image = create_roc_curve_chart(y_true, y_score)
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating ROC curve chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate ROC curve chart: {str(e)}"
        )

@router.get("/prediction-history")
async def get_prediction_history_chart(
    metric_name: str = Query("accuracy", description="Name of the metric to plot"),
    start_date: Optional[str] = Query(None, description="Start date in ISO format"),
    end_date: Optional[str] = Query(None, description="End date in ISO format")
):
    """
    Generate a visualization of prediction accuracy over time.
    """
    try:
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = pd.Timestamp.now() - pd.Timedelta(days=90)
            
        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = pd.Timestamp.now()
        
        # Generate dates
        dates = pd.date_range(start=start, end=end, freq='W')
        
        # Generate metrics with a slight upward trend and some randomness
        base_value = 0.65
        trend_slope = 0.005
        random_factor = 0.05
        
        metrics = []
        for i, date in enumerate(dates):
            value = base_value + trend_slope * i + np.random.uniform(-random_factor, random_factor)
            value = min(max(value, 0), 1)  # Clamp between 0 and 1
            metrics.append(value)
        
        prediction_history = [
            {
                "date": date,
                "metrics": {
                    metric_name: metric,
                    "precision": np.random.uniform(0.6, 0.9),
                    "recall": np.random.uniform(0.6, 0.9)
                }
            }
            for date, metric in zip(dates, metrics)
        ]
        
        image = create_prediction_history_chart(prediction_history, metric_name)
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating prediction history chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate prediction history chart: {str(e)}"
        )

@router.get("/player-performance")
async def get_player_performance_chart(
    player_name: str = Query(..., description="Name of the player"),
    metrics: str = Query("goals,assists,minutes_played", description="Comma-separated list of metrics")
):
    """
    Generate a visualization of a player's performance over time.
    """
    try:
        metrics_list = metrics.split(',')
        
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        n_matches = 15
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_matches, freq='W')
        
        data = {
            'player_name': [player_name] * n_matches,
            'match_date': dates,
            'team': ['Player Team'] * n_matches,
            'opponent': [f'Opponent {i}' for i in range(n_matches)]
        }
        
        # Add requested metrics
        valid_metrics = ['goals', 'assists', 'minutes_played', 'shots', 'passes', 'tackles']
        
        for metric in valid_metrics:
            if metric in metrics_list:
                if metric == 'goals':
                    data[metric] = np.random.randint(0, 3, size=n_matches)
                elif metric == 'assists':
                    data[metric] = np.random.randint(0, 2, size=n_matches)
                elif metric == 'minutes_played':
                    # Most games played full 90 minutes, some substituted
                    data[metric] = [90 if np.random.random() < 0.7 else np.random.choice([0, 45, 60, 75]) 
                                  for _ in range(n_matches)]
                elif metric == 'shots':
                    data[metric] = np.random.randint(0, 6, size=n_matches)
                elif metric == 'passes':
                    data[metric] = np.random.randint(20, 60, size=n_matches)
                elif metric == 'tackles':
                    data[metric] = np.random.randint(0, 8, size=n_matches)
        
        player_data = pd.DataFrame(data)
        
        image = create_player_performance_chart(player_data, player_name, metrics_list)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for player {player_name}"
            )
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating player performance chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate player performance chart: {str(e)}"
        )

@router.get("/league-standings")
async def get_league_standings_chart(
    season: str = Query(..., description="Season to visualize"),
    top_n: int = Query(10, description="Number of top teams to display")
):
    """
    Generate a visualization of league standings.
    """
    try:
        # In a real application, you would retrieve this data from a database
        # This is mock data for demonstration purposes
        teams = [
            "Arsenal", "Aston Villa", "Brighton", "Burnley", "Chelsea",
            "Crystal Palace", "Everton", "Fulham", "Leeds United", "Leicester City",
            "Liverpool", "Manchester City", "Manchester United", "Newcastle United",
            "Sheffield United", "Southampton", "Tottenham Hotspur", "West Brom",
            "West Ham United", "Wolverhampton Wanderers"
        ]
        
        # Generate random points (higher for top teams)
        base_points = np.linspace(85, 25, len(teams))
        noise = np.random.randint(-5, 6, size=len(teams))
        points = base_points + noise
        
        data = {
            'team_name': teams,
            'season': [season] * len(teams),
            'points': points.astype(int),
            'matches_played': [38] * len(teams),
            'wins': np.random.randint(5, 25, size=len(teams)),
            'draws': np.random.randint(5, 15, size=len(teams)),
            'losses': np.random.randint(5, 20, size=len(teams))
        }
        
        standings_data = pd.DataFrame(data)
        
        image = create_league_standings_chart(standings_data, season, top_n)
        
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No standings data found for season {season}"
            )
        
        return {"image": image}
    except VisualizationError as e:
        logger.error(f"Error generating league standings chart: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate league standings chart: {str(e)}"
        ) 