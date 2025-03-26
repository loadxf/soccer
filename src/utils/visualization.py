"""
Visualization utilities for the soccer prediction system.

This module provides functions for creating various visualizations of soccer data,
model predictions, and performance metrics.
"""

import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

from src.utils.logger import get_logger

logger = get_logger(__name__)

class VisualizationError(Exception):
    """Exception raised for errors in the visualization module."""
    pass

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def create_team_performance_chart(
    team_data: pd.DataFrame,
    team_name: str,
    last_n_matches: int = 10
) -> str:
    """
    Create a visualization of a team's recent performance.
    
    Args:
        team_data: DataFrame containing team match data
        team_name: Name of the team to visualize
        last_n_matches: Number of recent matches to include
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        # Filter for the specific team and take last N matches
        df = team_data[team_data['team_name'] == team_name].sort_values('match_date').tail(last_n_matches)
        
        if df.empty:
            logger.warning(f"No data found for team {team_name}")
            return None
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot goals scored and conceded
        x = range(len(df))
        ax.bar(x, df['goals_scored'], label='Goals Scored', alpha=0.7, color='green')
        ax.bar(x, df['goals_conceded'], label='Goals Conceded', alpha=0.7, color='red')
        
        # Add match results as labels
        results = []
        for i, row in df.iterrows():
            if row['goals_scored'] > row['goals_conceded']:
                results.append('W')
            elif row['goals_scored'] < row['goals_conceded']:
                results.append('L')
            else:
                results.append('D')
        
        # Add labels and styling
        ax.set_title(f"{team_name} - Last {len(df)} Matches Performance")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['opponent']}\n{row['match_date'].strftime('%Y-%m-%d')}\n{result}" 
                         for row, result in zip(df.iterrows(), results)], rotation=45, ha='right')
        ax.set_ylabel('Goals')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating team performance chart: {str(e)}")
        raise VisualizationError(f"Failed to create team performance chart: {str(e)}")

def create_prediction_confidence_chart(
    predictions: List[Dict[str, Any]],
    actual_results: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Create a visualization of prediction confidence.
    
    Args:
        predictions: List of prediction dictionaries
        actual_results: Optional list of actual match results
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        # Extract prediction data
        teams = [f"{p['home_team']} vs {p['away_team']}" for p in predictions]
        home_win_probs = [p['probabilities']['home_win'] for p in predictions]
        draw_probs = [p['probabilities']['draw'] for p in predictions]
        away_win_probs = [p['probabilities']['away_win'] for p in predictions]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set width and positions for bars
        width = 0.25
        x = np.arange(len(teams))
        
        # Create bars
        ax.bar(x - width, home_win_probs, width, label='Home Win', color='skyblue')
        ax.bar(x, draw_probs, width, label='Draw', color='gray')
        ax.bar(x + width, away_win_probs, width, label='Away Win', color='salmon')
        
        # If actual results are provided, annotate the correct outcome
        if actual_results:
            for i, result in enumerate(actual_results):
                if result['home_score'] > result['away_score']:
                    ax.plot(x[i] - width, home_win_probs[i] + 0.03, 'o', color='green', ms=10)
                elif result['home_score'] < result['away_score']:
                    ax.plot(x[i] + width, away_win_probs[i] + 0.03, 'o', color='green', ms=10)
                else:
                    ax.plot(x[i], draw_probs[i] + 0.03, 'o', color='green', ms=10)
        
        # Add labels and styling
        ax.set_xlabel('Matches')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Confidence by Match')
        ax.set_xticks(x)
        ax.set_xticklabels(teams, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating prediction confidence chart: {str(e)}")
        raise VisualizationError(f"Failed to create prediction confidence chart: {str(e)}")

def create_model_comparison_chart(
    model_metrics: Dict[str, Dict[str, float]]
) -> str:
    """
    Create a radar chart comparing different models.
    
    Args:
        model_metrics: Dictionary of model names to metrics dictionaries
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        metrics = list(next(iter(model_metrics.values())).keys())
        models = list(model_metrics.keys())
        
        # Calculate angle for each metric
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        # Make the plot circular
        angles += angles[:1]
        
        # Create figure and polar axis
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot each model
        for model_name, model_data in model_metrics.items():
            values = [model_data[metric] for metric in metrics]
            # Close the loop
            values += values[:1]
            ax.plot(angles, values, linewidth=1, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels and styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title('Model Comparison')
        ax.grid(True)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating model comparison chart: {str(e)}")
        raise VisualizationError(f"Failed to create model comparison chart: {str(e)}")

def create_feature_importance_chart(
    feature_importances: Dict[str, float],
    top_n: int = 20
) -> str:
    """
    Create a horizontal bar chart of feature importances.
    
    Args:
        feature_importances: Dictionary mapping feature names to importance scores
        top_n: Number of top features to display
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        # Sort features by importance and take top N
        sorted_features = dict(sorted(feature_importances.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:top_n])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create horizontal bar chart
        features = list(sorted_features.keys())
        importances = list(sorted_features.values())
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {str(e)}")
        raise VisualizationError(f"Failed to create feature importance chart: {str(e)}")

def create_confusion_matrix_chart(
    y_true: List[int], 
    y_pred: List[int],
    class_names: List[str] = ['Home Win', 'Draw', 'Away Win']
) -> str:
    """
    Create a confusion matrix visualization.
    
    Args:
        y_true: List of true labels (0 for home win, 1 for draw, 2 for away win)
        y_pred: List of predicted labels
        class_names: Names of the classes
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        # Add labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating confusion matrix chart: {str(e)}")
        raise VisualizationError(f"Failed to create confusion matrix chart: {str(e)}")

def create_roc_curve_chart(
    y_true: List[int],
    y_score: List[List[float]],
    class_names: List[str] = ['Home Win', 'Draw', 'Away Win']
) -> str:
    """
    Create a ROC curve visualization for multi-class prediction.
    
    Args:
        y_true: List of true labels (one-hot encoded)
        y_score: List of predicted probabilities for each class
        class_names: Names of the classes
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        # Convert to one-hot encoding if not already
        y_true_onehot = pd.get_dummies(y_true).values if len(np.array(y_true).shape) == 1 else np.array(y_true)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot ROC for each class
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], np.array(y_score)[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        # Add reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Add labels and styling
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating ROC curve chart: {str(e)}")
        raise VisualizationError(f"Failed to create ROC curve chart: {str(e)}")

def create_prediction_history_chart(
    prediction_history: List[Dict[str, Any]],
    metric_name: str = 'accuracy'
) -> str:
    """
    Create a line chart showing prediction accuracy over time.
    
    Args:
        prediction_history: List of prediction history records with dates and metrics
        metric_name: Name of the metric to plot
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        # Extract data
        dates = [p['date'] for p in prediction_history]
        metrics = [p['metrics'][metric_name] for p in prediction_history]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot line chart
        ax.plot(dates, metrics, marker='o', linestyle='-', color='royalblue')
        
        # Add rolling average
        window = min(5, len(metrics))
        if window > 1:
            rolling_avg = pd.Series(metrics).rolling(window=window).mean()
            ax.plot(dates, rolling_avg, linestyle='--', color='crimson', 
                  label=f'{window}-period Moving Average')
        
        # Add labels and styling
        ax.set_xlabel('Date')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'{metric_name.capitalize()} Over Time')
        ax.grid(linestyle='--', alpha=0.7)
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        
        if window > 1:
            ax.legend()
            
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating prediction history chart: {str(e)}")
        raise VisualizationError(f"Failed to create prediction history chart: {str(e)}")

def create_player_performance_chart(
    player_data: pd.DataFrame,
    player_name: str,
    metrics: List[str] = ['goals', 'assists', 'minutes_played']
) -> str:
    """
    Create a visualization of a player's performance over time.
    
    Args:
        player_data: DataFrame containing player performance data
        player_name: Name of the player to visualize
        metrics: List of metrics to include in the visualization
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        # Filter for the specific player
        df = player_data[player_data['player_name'] == player_name].sort_values('match_date')
        
        if df.empty:
            logger.warning(f"No data found for player {player_name}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each metric as a line
        for metric in metrics:
            if metric in df.columns:
                ax.plot(df['match_date'], df[metric], marker='o', linestyle='-', label=metric)
        
        # Add labels and styling
        ax.set_xlabel('Match Date')
        ax.set_ylabel('Value')
        ax.set_title(f"{player_name}'s Performance Over Time")
        ax.grid(linestyle='--', alpha=0.7)
        ax.legend()
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating player performance chart: {str(e)}")
        raise VisualizationError(f"Failed to create player performance chart: {str(e)}")

def create_league_standings_chart(
    standings_data: pd.DataFrame,
    season: str,
    top_n: int = 10
) -> str:
    """
    Create a horizontal bar chart of league standings.
    
    Args:
        standings_data: DataFrame containing team standings data
        season: Season to visualize
        top_n: Number of top teams to display
        
    Returns:
        Base64 encoded string of the visualization
    """
    try:
        # Filter for the specific season and take top N teams
        df = standings_data[standings_data['season'] == season].sort_values('points', ascending=False).head(top_n)
        
        if df.empty:
            logger.warning(f"No standings data found for season {season}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot horizontal bar chart
        teams = df['team_name']
        points = df['points']
        
        y_pos = np.arange(len(teams))
        bars = ax.barh(y_pos, points, align='center', color='skyblue')
        
        # Add point values at the end of each bar
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                  f"{points.iloc[i]} pts", va='center')
        
        # Add labels and styling
        ax.set_yticks(y_pos)
        ax.set_yticklabels(teams)
        ax.invert_yaxis()  # Teams read top-to-bottom
        ax.set_xlabel('Points')
        ax.set_title(f'League Standings - {season} Season')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except Exception as e:
        logger.error(f"Error creating league standings chart: {str(e)}")
        raise VisualizationError(f"Failed to create league standings chart: {str(e)}") 