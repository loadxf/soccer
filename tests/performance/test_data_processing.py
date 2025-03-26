"""
Performance tests for data processing operations in the Soccer Prediction System.
These tests measure the performance of data loading, transformation, and feature engineering.
"""

import os
import sys
import time
import pytest
import random
import numpy as np
import pandas as pd
from unittest import mock
from datetime import datetime, timedelta
import json

# Add the root directory to sys.path to allow importing the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import actual data processing components
try:
    from src.data.processing import FeatureEngineering
    from src.data.loader import DataLoader
    REAL_DATA_COMPONENTS_AVAILABLE = True
except ImportError:
    REAL_DATA_COMPONENTS_AVAILABLE = False


# Mock data processing classes for testing if real ones aren't available
class MockDataLoader:
    """Mock data loader for testing."""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or "data"
    
    def load_matches(self, limit=None):
        """Load match data."""
        # Simulate loading delay
        time.sleep(0.01)
        
        # Generate sample match data
        num_matches = limit or 1000
        
        matches = []
        for i in range(num_matches):
            match_date = datetime(2023, 1, 1) + timedelta(days=i % 365)
            matches.append({
                'id': i,
                'date': match_date.strftime('%Y-%m-%d'),
                'home_team_id': random.randint(1, 20),
                'away_team_id': random.randint(1, 20),
                'home_goals': random.randint(0, 5),
                'away_goals': random.randint(0, 5),
                'competition_id': random.randint(1, 5)
            })
        
        return pd.DataFrame(matches)
    
    def load_teams(self):
        """Load team data."""
        # Simulate loading delay
        time.sleep(0.005)
        
        # Generate sample team data
        teams = []
        for i in range(1, 21):
            teams.append({
                'id': i,
                'name': f'Team {i}',
                'rating': random.uniform(70, 95),
                'country': random.choice(['England', 'Spain', 'Germany', 'Italy', 'France'])
            })
        
        return pd.DataFrame(teams)
    
    def load_player_stats(self, limit=None):
        """Load player statistics."""
        # Simulate loading delay
        time.sleep(0.02)
        
        # Generate sample player data
        num_players = limit or 500
        
        players = []
        for i in range(num_players):
            team_id = random.randint(1, 20)
            players.append({
                'id': i,
                'name': f'Player {i}',
                'team_id': team_id,
                'position': random.choice(['GK', 'DF', 'MF', 'FW']),
                'age': random.randint(18, 35),
                'goals': random.randint(0, 30),
                'assists': random.randint(0, 20),
                'minutes_played': random.randint(0, 3000)
            })
        
        return pd.DataFrame(players)
    
    def load_competition_data(self):
        """Load competition data."""
        # Simulate loading delay
        time.sleep(0.003)
        
        # Generate sample competition data
        competitions = []
        for i in range(1, 6):
            competitions.append({
                'id': i,
                'name': f'Competition {i}',
                'country': random.choice(['International', 'England', 'Spain', 'Germany', 'Italy', 'France']),
                'type': random.choice(['League', 'Cup', 'Tournament'])
            })
        
        return pd.DataFrame(competitions)


class MockFeatureEngineering:
    """Mock feature engineering for testing."""
    
    def __init__(self, data_loader=None):
        self.data_loader = data_loader or MockDataLoader()
    
    def create_match_features(self, matches_df, teams_df=None):
        """Create features for match prediction."""
        # Simulate computation delay
        time.sleep(0.05)
        
        if teams_df is None:
            teams_df = self.data_loader.load_teams()
        
        # Create result column
        matches_df['result'] = np.where(
            matches_df['home_goals'] > matches_df['away_goals'], 'home_win',
            np.where(matches_df['home_goals'] < matches_df['away_goals'], 'away_win', 'draw')
        )
        
        # Merge team data
        result = pd.merge(
            matches_df,
            teams_df.rename(columns={'id': 'home_team_id', 'name': 'home_team_name', 'rating': 'home_team_rating'}),
            on='home_team_id'
        )
        
        result = pd.merge(
            result,
            teams_df.rename(columns={'id': 'away_team_id', 'name': 'away_team_name', 'rating': 'away_team_rating'}),
            on='away_team_id'
        )
        
        # Calculate some features
        result['rating_diff'] = result['home_team_rating'] - result['away_team_rating']
        result['total_goals'] = result['home_goals'] + result['away_goals']
        result['goal_diff'] = result['home_goals'] - result['away_goals']
        
        return result
    
    def create_team_form_features(self, matches_df, window=5):
        """Create team form features based on recent match history."""
        # Simulate computation delay
        time.sleep(0.07)
        
        # Create a copy to avoid modifying the original
        match_history = matches_df.copy()
        
        # Calculate points
        match_history['home_points'] = np.where(
            match_history['home_goals'] > match_history['away_goals'], 3,
            np.where(match_history['home_goals'] == match_history['away_goals'], 1, 0)
        )
        
        match_history['away_points'] = np.where(
            match_history['away_goals'] > match_history['home_goals'], 3,
            np.where(match_history['away_goals'] == match_history['home_goals'], 1, 0)
        )
        
        # Sort by date
        match_history['date'] = pd.to_datetime(match_history['date'])
        match_history = match_history.sort_values('date')
        
        # Calculate rolling form (simplified)
        # In a real implementation, this would be more complex
        team_stats = {}
        
        for team_id in range(1, 21):
            # Home form
            home_matches = match_history[match_history['home_team_id'] == team_id]
            home_form = home_matches['home_points'].rolling(window=window, min_periods=1).mean().fillna(0)
            
            # Away form
            away_matches = match_history[match_history['away_team_id'] == team_id]
            away_form = away_matches['away_points'].rolling(window=window, min_periods=1).mean().fillna(0)
            
            # Store team stats
            team_stats[team_id] = {
                'home_form': home_form.tolist() if len(home_form) > 0 else [0],
                'away_form': away_form.tolist() if len(away_form) > 0 else [0]
            }
        
        return team_stats
    
    def create_advanced_features(self, matches_df, teams_df=None, players_df=None, return_df=True):
        """Create advanced features including team and player statistics."""
        # Simulate computation delay
        time.sleep(0.1)
        
        if teams_df is None:
            teams_df = self.data_loader.load_teams()
            
        if players_df is None:
            players_df = self.data_loader.load_player_stats()
        
        # Start with basic match features
        features_df = self.create_match_features(matches_df, teams_df)
        
        # Calculate team-level aggregated stats (goals scored/conceded per game)
        team_stats = {}
        
        for team_id in range(1, 21):
            home_matches = matches_df[matches_df['home_team_id'] == team_id]
            away_matches = matches_df[matches_df['away_team_id'] == team_id]
            
            avg_home_goals = home_matches['home_goals'].mean() if len(home_matches) > 0 else 0
            avg_away_goals = away_matches['away_goals'].mean() if len(away_matches) > 0 else 0
            avg_home_conceded = home_matches['away_goals'].mean() if len(home_matches) > 0 else 0
            avg_away_conceded = away_matches['home_goals'].mean() if len(away_matches) > 0 else 0
            
            # Add to stats dictionary
            team_stats[team_id] = {
                'avg_home_goals': avg_home_goals,
                'avg_away_goals': avg_away_goals,
                'avg_home_conceded': avg_home_conceded,
                'avg_away_conceded': avg_away_conceded,
                'attack_strength': (avg_home_goals + avg_away_goals) / 2,
                'defense_weakness': (avg_home_conceded + avg_away_conceded) / 2,
            }
        
        # Add these stats to the features dataframe
        if return_df:
            for stat in ['avg_home_goals', 'avg_home_conceded', 'attack_strength', 'defense_weakness']:
                features_df[f'home_{stat}'] = features_df['home_team_id'].map(
                    lambda x: team_stats.get(x, {}).get(stat, 0)
                )
                features_df[f'away_{stat}'] = features_df['away_team_id'].map(
                    lambda x: team_stats.get(x, {}).get(stat, 0)
                )
            
            # Calculate expected goals based on team strengths
            features_df['xg_home'] = features_df['home_attack_strength'] * features_df['away_defense_weakness']
            features_df['xg_away'] = features_df['away_attack_strength'] * features_df['home_defense_weakness']
            
            return features_df
        else:
            return team_stats


# Use appropriate implementation
if REAL_DATA_COMPONENTS_AVAILABLE:
    data_loader = DataLoader()
    feature_engineering = FeatureEngineering(data_loader)
else:
    data_loader = MockDataLoader()
    feature_engineering = MockFeatureEngineering(data_loader)


class TestDataLoadingPerformance:
    """Tests for data loading performance."""
    
    @pytest.mark.benchmark(
        group="data_loading",
        min_time=0.1,
        max_time=1.0,
        min_rounds=5,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_load_matches(self, benchmark):
        """Test performance of loading match data."""
        result = benchmark(data_loader.load_matches)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'home_team_id' in result.columns
        assert 'away_team_id' in result.columns
    
    @pytest.mark.benchmark(
        group="data_loading",
        min_time=0.1,
        max_time=0.5,
        min_rounds=10,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_load_teams(self, benchmark):
        """Test performance of loading team data."""
        result = benchmark(data_loader.load_teams)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'id' in result.columns
        assert 'name' in result.columns
    
    @pytest.mark.benchmark(
        group="data_loading",
        min_time=0.1,
        max_time=1.0,
        min_rounds=5,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_load_player_stats(self, benchmark):
        """Test performance of loading player statistics."""
        result = benchmark(data_loader.load_player_stats)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'id' in result.columns
        assert 'team_id' in result.columns


class TestFeatureEngineeringPerformance:
    """Tests for feature engineering performance."""
    
    @pytest.fixture(scope="class")
    def sample_data(self):
        """Load sample data for testing."""
        matches = data_loader.load_matches(limit=500)
        teams = data_loader.load_teams()
        players = data_loader.load_player_stats(limit=300)
        
        return {
            'matches': matches,
            'teams': teams,
            'players': players
        }
    
    @pytest.mark.benchmark(
        group="feature_engineering",
        min_time=0.1,
        max_time=1.0,
        min_rounds=5,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_create_match_features(self, benchmark, sample_data):
        """Test performance of creating basic match features."""
        matches = sample_data['matches']
        teams = sample_data['teams']
        
        result = benchmark(feature_engineering.create_match_features, matches, teams)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(matches)
        assert 'rating_diff' in result.columns
        assert 'result' in result.columns
    
    @pytest.mark.benchmark(
        group="feature_engineering",
        min_time=0.1,
        max_time=2.0,
        min_rounds=3,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_create_team_form_features(self, benchmark, sample_data):
        """Test performance of creating team form features."""
        matches = sample_data['matches']
        
        result = benchmark(feature_engineering.create_team_form_features, matches)
        
        assert isinstance(result, dict)
        assert len(result) > 0
        # Check first team has form data
        first_team = next(iter(result))
        assert 'home_form' in result[first_team]
        assert 'away_form' in result[first_team]
    
    @pytest.mark.benchmark(
        group="feature_engineering",
        min_time=0.5,
        max_time=3.0,
        min_rounds=3,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_create_advanced_features(self, benchmark, sample_data):
        """Test performance of creating advanced features."""
        matches = sample_data['matches']
        teams = sample_data['teams']
        players = sample_data['players']
        
        result = benchmark(
            feature_engineering.create_advanced_features,
            matches,
            teams,
            players
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(matches)
        assert 'xg_home' in result.columns
        assert 'xg_away' in result.columns


class TestEndToEndProcessingPerformance:
    """Tests for end-to-end data processing performance."""
    
    @pytest.mark.benchmark(
        group="end_to_end",
        min_time=1.0,
        max_time=5.0,
        min_rounds=2,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_end_to_end_data_processing(self, benchmark):
        """Test performance of complete data processing pipeline."""
        def process_pipeline():
            # Load data
            matches = data_loader.load_matches(limit=200)
            teams = data_loader.load_teams()
            players = data_loader.load_player_stats(limit=200)
            
            # Process data
            features = feature_engineering.create_advanced_features(
                matches,
                teams,
                players
            )
            
            # Create team form features
            team_form = feature_engineering.create_team_form_features(matches)
            
            # Add team form to features
            for team_id, form_data in team_form.items():
                # In a real implementation, this would be more complex
                # We just calculate the average form for demonstration
                avg_home_form = sum(form_data['home_form']) / len(form_data['home_form']) if form_data['home_form'] else 0
                avg_away_form = sum(form_data['away_form']) / len(form_data['away_form']) if form_data['away_form'] else 0
                
                # Update home team form
                features.loc[features['home_team_id'] == team_id, 'home_team_form'] = avg_home_form
                
                # Update away team form
                features.loc[features['away_team_id'] == team_id, 'away_team_form'] = avg_away_form
            
            return features
        
        result = benchmark(process_pipeline)
        
        assert isinstance(result, pd.DataFrame)
        assert 'home_team_form' in result.columns
        assert 'away_team_form' in result.columns


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 