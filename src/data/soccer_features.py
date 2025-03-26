"""
Advanced Soccer Feature Engineering Module

This module implements specialized soccer feature engineering techniques
based on academic research, including:
- Time-weighted form metrics
- Expected goals (xG) modeling
- Bayesian strength indicators
- Team style features
- Market-derived features
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import scipy.stats as stats

# Import project components
from src.utils.logger import get_logger
from src.data.features import calculate_team_form

try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Setup logger
logger = get_logger("data.soccer_features")


def calculate_exponential_team_form(games_df: pd.DataFrame, half_life_days: int = 30) -> pd.DataFrame:
    """
    Calculate team form with exponential time decay, giving higher weight to recent matches.
    Based on research showing recency is critical in predicting match outcomes.
    
    Args:
        games_df: DataFrame containing match data
        half_life_days: Half-life for exponential decay (in days)
        
    Returns:
        pd.DataFrame: DataFrame with time-weighted team form features
    """
    # Ensure games are sorted by date
    games_df = games_df.copy()
    if 'date' in games_df.columns and not pd.api.types.is_datetime64_dtype(games_df['date']):
        games_df['date'] = pd.to_datetime(games_df['date'])
    
    games_df = games_df.sort_values('date')
    
    # Create separate dataframes for home and away performance
    home_team_results = games_df[['date', 'home_club_id', 'home_club_goals', 'away_club_goals']].copy()
    away_team_results = games_df[['date', 'away_club_id', 'home_club_goals', 'away_club_goals']].copy()
    
    # Rename columns for consistency
    home_team_results.columns = ['date', 'team_id', 'goals_scored', 'goals_conceded']
    away_team_results.columns = ['date', 'team_id', 'goals_conceded', 'goals_scored']
    
    # Combine home and away results
    team_results = pd.concat([home_team_results, away_team_results])
    team_results = team_results.sort_values(['team_id', 'date'])
    
    # Calculate match outcome (win=3, draw=1, loss=0)
    team_results['points'] = 1  # Default for draw
    team_results.loc[team_results['goals_scored'] > team_results['goals_conceded'], 'points'] = 3
    team_results.loc[team_results['goals_scored'] < team_results['goals_conceded'], 'points'] = 0
    
    # Calculate additional stats
    team_results['goal_diff'] = team_results['goals_scored'] - team_results['goals_conceded']
    team_results['win'] = (team_results['points'] == 3).astype(int)
    team_results['draw'] = (team_results['points'] == 1).astype(int)
    team_results['loss'] = (team_results['points'] == 0).astype(int)
    
    # Lambda for exponential decay (half-life)
    decay_lambda = np.log(2) / half_life_days
    
    form_features = []
    
    for team_id in team_results['team_id'].unique():
        team_data = team_results[team_results['team_id'] == team_id].copy()
        
        if len(team_data) <= 1:
            continue
        
        # For each match, calculate exponentially weighted metrics from previous matches
        for i in range(1, len(team_data)):
            current_date = team_data.iloc[i]['date']
            
            # Get all previous matches
            prev_matches = team_data.iloc[:i]
            
            # Calculate time difference in days
            prev_matches['days_before'] = (current_date - prev_matches['date']).dt.days
            
            # Calculate weights based on exponential decay
            prev_matches['weight'] = np.exp(-decay_lambda * prev_matches['days_before'])
            
            # Normalize weights to sum to 1
            prev_matches['weight'] = prev_matches['weight'] / prev_matches['weight'].sum()
            
            # Calculate weighted stats
            exp_points = (prev_matches['points'] * prev_matches['weight']).sum()
            exp_goals_scored = (prev_matches['goals_scored'] * prev_matches['weight']).sum()
            exp_goals_conceded = (prev_matches['goals_conceded'] * prev_matches['weight']).sum()
            exp_goal_diff = (prev_matches['goal_diff'] * prev_matches['weight']).sum()
            exp_win_rate = (prev_matches['win'] * prev_matches['weight']).sum()
            
            # Update the current row with these metrics
            team_data.at[team_data.index[i], 'exp_form_points'] = exp_points
            team_data.at[team_data.index[i], 'exp_form_goals_scored'] = exp_goals_scored
            team_data.at[team_data.index[i], 'exp_form_goals_conceded'] = exp_goals_conceded
            team_data.at[team_data.index[i], 'exp_form_goal_diff'] = exp_goal_diff
            team_data.at[team_data.index[i], 'exp_form_win_rate'] = exp_win_rate
        
        # Add to form features
        form_features.append(team_data)
    
    # Combine all team form features
    form_df = pd.concat(form_features)
    
    # Fill missing values for the first match of each team
    form_cols = ['exp_form_points', 'exp_form_goals_scored', 'exp_form_goals_conceded', 
                 'exp_form_goal_diff', 'exp_form_win_rate']
    form_df[form_cols] = form_df[form_cols].fillna(0)
    
    # Clean up temporary columns
    form_df = form_df.drop(['days_before', 'weight', 'win', 'draw', 'loss'], axis=1, errors='ignore')
    
    return form_df


def estimate_team_strengths_dixon_coles(games_df: pd.DataFrame, window_days: int = 365) -> pd.DataFrame:
    """
    Implements the Dixon-Coles model to estimate team attack and defense strengths.
    This statistical model is widely used in academic literature on soccer prediction.
    
    Args:
        games_df: DataFrame containing match data
        window_days: Only include matches within this many days of each match
        
    Returns:
        pd.DataFrame: DataFrame with team strength parameters for each match date
    """
    # Ensure games are sorted by date
    games_df = games_df.copy()
    if 'date' in games_df.columns and not pd.api.types.is_datetime64_dtype(games_df['date']):
        games_df['date'] = pd.to_datetime(games_df['date'])
    
    games_df = games_df.sort_values('date')
    
    # Get all unique teams
    home_teams = games_df['home_club_id'].unique()
    away_teams = games_df['away_club_id'].unique()
    all_teams = np.unique(np.concatenate([home_teams, away_teams]))
    
    # Track strength parameters over time
    results = []
    
    # Process each match date to get team strengths up to that point
    unique_dates = games_df['date'].unique()
    
    for current_date in unique_dates:
        # Get matches before current date (within window)
        window_start = current_date - pd.Timedelta(days=window_days)
        past_matches = games_df[(games_df['date'] < current_date) & 
                               (games_df['date'] >= window_start)].copy()
        
        if len(past_matches) < 10:  # Need sufficient matches for estimation
            continue
        
        # Simple Poisson model using Generalized Linear Model
        # This is a simplified version of Dixon-Coles
        
        # Create team indicator variables (home attack, home defense, away attack, away defense)
        # We'll use one team as reference to avoid perfect multicollinearity
        reference_team = all_teams[0]
        
        # Prepare data for modeling
        model_data = []
        
        for _, match in past_matches.iterrows():
            home_team = match['home_club_id']
            away_team = match['away_club_id']
            
            # Home goal model features
            home_row = {'goals': match['home_club_goals'], 'home': 1}
            for team in all_teams:
                if team != reference_team:
                    home_row[f'attack_{team}'] = 1 if home_team == team else 0
                    home_row[f'defense_{team}'] = 0  # Home team's defense doesn't affect home goals
            
            for team in all_teams:
                if team != reference_team:
                    home_row[f'defense_{team}'] = 1 if away_team == team else 0
                    home_row[f'attack_{team}'] = 0  # Away team's attack doesn't affect home goals
            
            # Away goal model features
            away_row = {'goals': match['away_club_goals'], 'home': 0}
            for team in all_teams:
                if team != reference_team:
                    away_row[f'attack_{team}'] = 1 if away_team == team else 0
                    away_row[f'defense_{team}'] = 0  # Away team's defense doesn't affect away goals
            
            for team in all_teams:
                if team != reference_team:
                    away_row[f'defense_{team}'] = 1 if home_team == team else 0
                    away_row[f'attack_{team}'] = 0  # Home team's attack doesn't affect away goals
            
            model_data.append(home_row)
            model_data.append(away_row)
        
        # Convert to DataFrame
        model_df = pd.DataFrame(model_data)
        
        try:
            # Fit Poisson GLM
            formula = "goals ~ home + " + " + ".join([f"attack_{team}" for team in all_teams if team != reference_team]) + " + " + \
                      " + ".join([f"defense_{team}" for team in all_teams if team != reference_team])
            
            model = sm.GLM.from_formula(formula, data=model_df, family=sm.families.Poisson()).fit()
            
            # Extract team strengths from model coefficients
            team_strengths = {'date': current_date}
            
            # Home advantage
            home_advantage = model.params['home']
            team_strengths['home_advantage'] = home_advantage
            
            # Reference team parameters
            team_strengths[f'attack_{reference_team}'] = 0  # By definition
            team_strengths[f'defense_{reference_team}'] = 0  # By definition
            
            # Other team parameters
            for team in all_teams:
                if team != reference_team:
                    if f'attack_{team}' in model.params:
                        team_strengths[f'attack_{team}'] = model.params[f'attack_{team}']
                    else:
                        team_strengths[f'attack_{team}'] = 0
                    
                    if f'defense_{team}' in model.params:
                        team_strengths[f'defense_{team}'] = model.params[f'defense_{team}']
                    else:
                        team_strengths[f'defense_{team}'] = 0
            
            # Append to results
            results.append(team_strengths)
        
        except Exception as e:
            logger.warning(f"Could not estimate team strengths for date {current_date}: {e}")
            continue
    
    if not results:
        logger.warning("Could not estimate team strengths - insufficient data")
        return pd.DataFrame()
    
    # Convert to DataFrame
    strengths_df = pd.DataFrame(results)
    
    return strengths_df


def calculate_expected_goals(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an expected goals (xG) model based on shot characteristics.
    This is based on research showing xG is a superior predictor to actual goals.
    
    Args:
        shots_df: DataFrame containing shot data with features like distance, angle, etc.
        
    Returns:
        pd.DataFrame: DataFrame with xG values for each shot
    """
    # If we don't have detailed shot data, return empty DataFrame
    if shots_df is None or len(shots_df) == 0:
        logger.warning("No shot data available for xG calculation")
        return pd.DataFrame()
    
    # Ensure required columns exist
    required_cols = ['shot_id', 'match_id', 'player_id', 'team_id', 'distance', 'angle']
    
    if not all(col in shots_df.columns for col in required_cols):
        logger.warning("Shot data missing required columns for xG calculation")
        return pd.DataFrame()
    
    # Copy the DataFrame to avoid modifying the original
    shots = shots_df.copy()
    
    # Preprocess to create additional features
    shots['distance_squared'] = shots['distance'] ** 2
    shots['angle_radians'] = np.radians(shots['angle'])
    shots['goal_line_distance'] = shots['distance'] * np.cos(shots['angle_radians'])
    
    # Create categorical features from any available columns
    if 'body_part' in shots.columns:
        shots['is_header'] = (shots['body_part'] == 'head').astype(int)
    else:
        shots['is_header'] = 0
    
    if 'is_free_kick' in shots.columns:
        shots['is_free_kick'] = shots['is_free_kick'].astype(int)
    else:
        shots['is_free_kick'] = 0
    
    if 'is_penalty' in shots.columns:
        shots['is_penalty'] = shots['is_penalty'].astype(int)
    else:
        shots['is_penalty'] = 0
    
    # Prepare training data
    X = shots[['distance', 'distance_squared', 'angle_radians', 'goal_line_distance', 
              'is_header', 'is_free_kick', 'is_penalty']].values
    
    if 'is_goal' in shots.columns:
        y = shots['is_goal'].values
    else:
        # If we don't have goal outcomes, we can't train a model
        logger.warning("Shot data missing goal outcomes for xG calculation")
        return pd.DataFrame()
    
    # Train a logistic regression model
    model = sm.Logit(y, sm.add_constant(X))
    try:
        result = model.fit(disp=0)
        
        # Calculate xG for each shot
        shots['xG'] = result.predict(sm.add_constant(X))
        
        # Limit xG to reasonable range [0.01, 0.99]
        shots['xG'] = shots['xG'].clip(0.01, 0.99)
        
        return shots[['shot_id', 'match_id', 'player_id', 'team_id', 'xG']]
    
    except Exception as e:
        logger.warning(f"Could not train xG model: {e}")
        # Fallback: use a simple distance-based model
        shots['xG'] = 1.0 / (1.0 + 0.5 * shots['distance'])
        return shots[['shot_id', 'match_id', 'player_id', 'team_id', 'xG']]


def aggregate_team_xg(shots_xg_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player xG to team level and calculate team xG metrics.
    
    Args:
        shots_xg_df: DataFrame containing shot xG data
        matches_df: DataFrame containing match data
        
    Returns:
        pd.DataFrame: DataFrame with team xG metrics for each match
    """
    if shots_xg_df is None or len(shots_xg_df) == 0:
        logger.warning("No xG data available for team aggregation")
        return pd.DataFrame()
    
    # Group by match_id and team_id to sum xG
    team_xg = shots_xg_df.groupby(['match_id', 'team_id'])['xG'].sum().reset_index()
    
    # Merge with matches data
    match_xg = []
    
    for _, match in matches_df.iterrows():
        match_id = match['match_id'] if 'match_id' in match else None
        if match_id is None:
            continue
        
        home_team_id = match['home_club_id']
        away_team_id = match['away_club_id']
        
        # Get home team xG
        home_xg_row = team_xg[(team_xg['match_id'] == match_id) & 
                              (team_xg['team_id'] == home_team_id)]
        home_xg = home_xg_row['xG'].values[0] if len(home_xg_row) > 0 else 0
        
        # Get away team xG
        away_xg_row = team_xg[(team_xg['match_id'] == match_id) & 
                              (team_xg['team_id'] == away_team_id)]
        away_xg = away_xg_row['xG'].values[0] if len(away_xg_row) > 0 else 0
        
        # Store match xG data
        match_xg.append({
            'match_id': match_id,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_xg_diff': home_xg - away_xg,
            'date': match['date']
        })
    
    return pd.DataFrame(match_xg)


def extract_betting_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from betting odds data, shown by research to be strong predictors.
    
    Args:
        matches_df: DataFrame containing match data with betting odds
        
    Returns:
        pd.DataFrame: DataFrame with betting-derived features
    """
    # Check if betting odds columns exist
    odds_columns = [col for col in matches_df.columns 
                   if any(x in col.lower() for x in ['odds', 'probability', 'bet'])]
    
    if not odds_columns:
        logger.warning("No betting odds data found in matches DataFrame")
        return pd.DataFrame()
    
    # Copy only needed columns
    required_cols = ['match_id', 'date', 'home_club_id', 'away_club_id']
    
    if not all(col in matches_df.columns for col in required_cols):
        logger.warning("Matches data missing required columns for betting features")
        return pd.DataFrame()
    
    betting_data = matches_df[required_cols + odds_columns].copy()
    
    # Convert odds to implied probabilities where needed
    implied_prob_cols = []
    
    for col in odds_columns:
        if 'odds' in col.lower() and not any(x in col.lower() for x in ['prob', 'probability']):
            # Convert decimal odds to implied probability
            new_col = col.replace('odds', 'implied_prob')
            betting_data[new_col] = 1 / betting_data[col]
            implied_prob_cols.append(new_col)
    
    # Calculate market efficiency metrics
    # If we have home, away, and draw probabilities from bookmakers
    home_prob_cols = [col for col in betting_data.columns if 'home' in col.lower() and 'prob' in col.lower()]
    away_prob_cols = [col for col in betting_data.columns if 'away' in col.lower() and 'prob' in col.lower()]
    draw_prob_cols = [col for col in betting_data.columns if 'draw' in col.lower() and 'prob' in col.lower()]
    
    # If we have multiple bookmakers, calculate variance in implied probabilities
    if len(home_prob_cols) > 1:
        betting_data['home_prob_variance'] = betting_data[home_prob_cols].var(axis=1)
    if len(away_prob_cols) > 1:
        betting_data['away_prob_variance'] = betting_data[away_prob_cols].var(axis=1)
    if len(draw_prob_cols) > 1:
        betting_data['draw_prob_variance'] = betting_data[draw_prob_cols].var(axis=1)
    
    # Calculate average implied probabilities
    if home_prob_cols:
        betting_data['avg_home_prob'] = betting_data[home_prob_cols].mean(axis=1)
    if away_prob_cols:
        betting_data['avg_away_prob'] = betting_data[away_prob_cols].mean(axis=1)
    if draw_prob_cols:
        betting_data['avg_draw_prob'] = betting_data[draw_prob_cols].mean(axis=1)
    
    # Calculate probability sum (measure of overround)
    prob_sets = []
    for i in range(min(len(home_prob_cols), len(away_prob_cols), len(draw_prob_cols))):
        prob_sets.append([home_prob_cols[i], away_prob_cols[i], draw_prob_cols[i]])
    
    if prob_sets:
        for i, prob_set in enumerate(prob_sets):
            betting_data[f'overround_{i+1}'] = betting_data[prob_set].sum(axis=1)
        
        betting_data['avg_overround'] = betting_data[[f'overround_{i+1}' for i in range(len(prob_sets))]].mean(axis=1)
    
    return betting_data


def create_advanced_match_features(matches_df: pd.DataFrame, 
                                 team_form_df: Optional[pd.DataFrame] = None,
                                 team_strength_df: Optional[pd.DataFrame] = None,
                                 match_xg_df: Optional[pd.DataFrame] = None,
                                 betting_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Combine all advanced features into a comprehensive match feature dataset.
    
    Args:
        matches_df: DataFrame containing match data
        team_form_df: DataFrame containing team form metrics
        team_strength_df: DataFrame containing team strength parameters
        match_xg_df: DataFrame containing match xG data
        betting_df: DataFrame containing betting-derived features
        
    Returns:
        pd.DataFrame: DataFrame with comprehensive match features
    """
    # Ensure matches are sorted by date
    matches_df = matches_df.copy()
    if 'date' in matches_df.columns and not pd.api.types.is_datetime64_dtype(matches_df['date']):
        matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    matches_df = matches_df.sort_values('date')
    
    # Start with basic match info
    if 'match_id' not in matches_df.columns and 'game_id' in matches_df.columns:
        matches_df['match_id'] = matches_df['game_id']
    
    match_features = matches_df[['match_id', 'date', 'home_club_id', 'away_club_id']].copy()
    
    # Add result if available
    if 'home_club_goals' in matches_df.columns and 'away_club_goals' in matches_df.columns:
        match_features['home_goals'] = matches_df['home_club_goals']
        match_features['away_goals'] = matches_df['away_club_goals']
        match_features['goal_diff'] = match_features['home_goals'] - match_features['away_goals']
        
        # Create target variable: 1 = home win, 0 = draw, -1 = away win
        match_features['result'] = np.select(
            [match_features['goal_diff'] > 0, match_features['goal_diff'] == 0, match_features['goal_diff'] < 0],
            [1, 0, -1]
        )
    
    # Add team form features if available
    if team_form_df is not None and len(team_form_df) > 0:
        # Home team form
        home_form = team_form_df.copy()
        home_form = home_form.rename(columns={
            'team_id': 'home_club_id',
            'exp_form_points': 'home_exp_form_points',
            'exp_form_goals_scored': 'home_exp_form_goals_scored',
            'exp_form_goals_conceded': 'home_exp_form_goals_conceded',
            'exp_form_goal_diff': 'home_exp_form_goal_diff',
            'exp_form_win_rate': 'home_exp_form_win_rate'
        })
        
        # Merge home team form
        match_features = pd.merge(
            match_features,
            home_form[['date', 'home_club_id', 'home_exp_form_points', 'home_exp_form_goals_scored',
                      'home_exp_form_goals_conceded', 'home_exp_form_goal_diff', 'home_exp_form_win_rate']],
            on=['date', 'home_club_id'],
            how='left'
        )
        
        # Away team form
        away_form = team_form_df.copy()
        away_form = away_form.rename(columns={
            'team_id': 'away_club_id',
            'exp_form_points': 'away_exp_form_points',
            'exp_form_goals_scored': 'away_exp_form_goals_scored',
            'exp_form_goals_conceded': 'away_exp_form_goals_conceded',
            'exp_form_goal_diff': 'away_exp_form_goal_diff',
            'exp_form_win_rate': 'away_exp_form_win_rate'
        })
        
        # Merge away team form
        match_features = pd.merge(
            match_features,
            away_form[['date', 'away_club_id', 'away_exp_form_points', 'away_exp_form_goals_scored',
                      'away_exp_form_goals_conceded', 'away_exp_form_goal_diff', 'away_exp_form_win_rate']],
            on=['date', 'away_club_id'],
            how='left'
        )
        
        # Calculate relative form metrics
        match_features['form_points_diff'] = match_features['home_exp_form_points'] - match_features['away_exp_form_points']
        match_features['form_goals_scored_diff'] = match_features['home_exp_form_goals_scored'] - match_features['away_exp_form_goals_scored']
        match_features['form_goals_conceded_diff'] = match_features['home_exp_form_goals_conceded'] - match_features['away_exp_form_goals_conceded']
        match_features['form_goal_diff_diff'] = match_features['home_exp_form_goal_diff'] - match_features['away_exp_form_goal_diff']
        match_features['form_win_rate_diff'] = match_features['home_exp_form_win_rate'] - match_features['away_exp_form_win_rate']
    
    # Add team strength parameters if available
    if team_strength_df is not None and len(team_strength_df) > 0:
        # For each match, get the most recent team strength parameters
        for i, match in match_features.iterrows():
            match_date = match['date']
            home_team = match['home_club_id']
            away_team = match['away_club_id']
            
            # Find most recent team strength parameters before this match
            prev_strengths = team_strength_df[team_strength_df['date'] < match_date]
            
            if len(prev_strengths) > 0:
                # Get the most recent record
                most_recent = prev_strengths.iloc[-1]
                
                # Home team attack and defense parameters
                if f'attack_{home_team}' in most_recent:
                    match_features.at[i, 'home_attack'] = most_recent[f'attack_{home_team}']
                if f'defense_{home_team}' in most_recent:
                    match_features.at[i, 'home_defense'] = most_recent[f'defense_{home_team}']
                
                # Away team attack and defense parameters
                if f'attack_{away_team}' in most_recent:
                    match_features.at[i, 'away_attack'] = most_recent[f'attack_{away_team}']
                if f'defense_{away_team}' in most_recent:
                    match_features.at[i, 'away_defense'] = most_recent[f'defense_{away_team}']
                
                # Home advantage
                match_features.at[i, 'home_advantage'] = most_recent['home_advantage']
        
        # Calculate expected goals based on team strengths (Dixon-Coles model)
        match_features['home_exp_goals'] = np.exp(
            match_features['home_attack'] + match_features['away_defense'] + match_features['home_advantage']
        )
        match_features['away_exp_goals'] = np.exp(
            match_features['away_attack'] + match_features['home_defense']
        )
        
        # Fill missing values with average
        strength_cols = ['home_attack', 'home_defense', 'away_attack', 'away_defense', 
                         'home_advantage', 'home_exp_goals', 'away_exp_goals']
        match_features[strength_cols] = match_features[strength_cols].fillna(match_features[strength_cols].mean())
    
    # Add xG features if available
    if match_xg_df is not None and len(match_xg_df) > 0:
        # Merge xG data
        match_features = pd.merge(
            match_features,
            match_xg_df[['match_id', 'home_xg', 'away_xg', 'home_xg_diff']],
            on='match_id',
            how='left'
        )
    
    # Add betting features if available
    if betting_df is not None and len(betting_df) > 0:
        # Merge betting data
        betting_cols = betting_df.columns.tolist()
        merge_cols = [col for col in betting_cols if col not in match_features.columns or col == 'match_id']
        
        match_features = pd.merge(
            match_features,
            betting_df[merge_cols],
            on='match_id',
            how='left'
        )
    
    # Fill missing values
    match_features = match_features.fillna(0)
    
    return match_features


def load_or_create_advanced_features(matches_df: pd.DataFrame,
                                   shots_df: Optional[pd.DataFrame] = None,
                                   half_life_days: int = 30,
                                   window_days: int = 365) -> pd.DataFrame:
    """
    One-stop function to load or create all advanced soccer features.
    
    Args:
        matches_df: DataFrame containing match data
        shots_df: Optional DataFrame containing shot data for xG modeling
        half_life_days: Half-life for exponential decay (in days)
        window_days: Only include matches within this many days for team strength estimation
        
    Returns:
        pd.DataFrame: DataFrame with comprehensive advanced match features
    """
    # Calculate exponential team form
    logger.info("Calculating exponential team form...")
    team_form_df = calculate_exponential_team_form(matches_df, half_life_days)
    
    # Estimate team strengths using Dixon-Coles model
    logger.info("Estimating team strengths using Dixon-Coles model...")
    team_strength_df = estimate_team_strengths_dixon_coles(matches_df, window_days)
    
    # Calculate expected goals if shot data available
    match_xg_df = None
    if shots_df is not None and len(shots_df) > 0:
        logger.info("Calculating expected goals (xG)...")
        shots_xg_df = calculate_expected_goals(shots_df)
        if len(shots_xg_df) > 0:
            match_xg_df = aggregate_team_xg(shots_xg_df, matches_df)
    
    # Extract betting features
    logger.info("Extracting betting features...")
    betting_df = extract_betting_features(matches_df)
    
    # Combine all features
    logger.info("Creating comprehensive match features...")
    match_features = create_advanced_match_features(
        matches_df,
        team_form_df,
        team_strength_df,
        match_xg_df,
        betting_df
    )
    
    logger.info(f"Created advanced match features dataset with {len(match_features)} matches and {len(match_features.columns)} features")
    
    return match_features 