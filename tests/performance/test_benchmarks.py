"""
Benchmark tests for critical components of the Soccer Prediction System.
These tests measure the performance of key functions and components.
"""

import os
import sys
import time
import pytest
import random
import numpy as np
import pandas as pd
from unittest import mock

# Add the root directory to sys.path to allow importing the src module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the components we want to benchmark
from src.utils.auth import create_access_token
from src.utils.cache import cache_manager
from src.api.middleware import CacheMiddleware
from src.utils.metrics import timed, HTTP_REQUEST_DURATION

# Mock request for testing
class MockRequest:
    """Mock FastAPI request object for testing."""
    def __init__(self, method="GET", url="/api/v1/teams"):
        self.method = method
        self.url = url
        self.headers = {}


class TestAuthBenchmarks:
    """Benchmark tests for authentication components."""
    
    @pytest.mark.benchmark(
        group="auth",
        min_time=0.1,
        max_time=0.5,
        min_rounds=10,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_token_creation(self, benchmark):
        """Benchmark token creation performance."""
        # Create sample user data
        user_data = {"sub": "test_user", "roles": ["user"]}
        
        # Run the benchmark
        result = benchmark(create_access_token, data=user_data)
        
        # Verify the result is a string (token)
        assert isinstance(result, str)
        assert len(result) > 0


class TestCacheBenchmarks:
    """Benchmark tests for caching components."""
    
    @pytest.fixture
    def setup_cache(self):
        """Set up cache for testing."""
        # Clear cache before testing
        cache_manager.clear_all()
        yield
        # Clean up after test
        cache_manager.clear_all()
    
    @pytest.mark.benchmark(
        group="cache",
        min_time=0.1,
        max_time=0.5,
        min_rounds=100,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_cache_get(self, benchmark, setup_cache):
        """Benchmark cache retrieval performance."""
        # Set up test data
        key = "test_key"
        value = {"data": list(range(1000))}
        cache_manager.set(key, value)
        
        # Run the benchmark
        result = benchmark(cache_manager.get, key)
        
        # Verify the result
        assert result == value
    
    @pytest.mark.benchmark(
        group="cache",
        min_time=0.1,
        max_time=0.5,
        min_rounds=100,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_cache_set(self, benchmark, setup_cache):
        """Benchmark cache storage performance."""
        # Set up test data
        key = "test_key"
        value = {"data": list(range(1000))}
        
        # Run the benchmark
        benchmark(cache_manager.set, key, value)
        
        # Verify the data was stored
        result = cache_manager.get(key)
        assert result == value


class TestMetricsBenchmarks:
    """Benchmark tests for metrics components."""
    
    @pytest.mark.benchmark(
        group="metrics",
        min_time=0.1,
        max_time=0.5,
        min_rounds=100,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_metrics_timing_decorator(self, benchmark):
        """Benchmark performance of timed decorator."""
        # Create a sample function to time
        @timed(HTTP_REQUEST_DURATION, labels={"method": "GET", "endpoint": "/test"})
        def sample_function():
            # Simulate some work
            total = 0
            for i in range(1000):
                total += i
            return total
        
        # Run the benchmark
        result = benchmark(sample_function)
        
        # Verify the function executed correctly
        assert result == sum(range(1000))


# Mocked model prediction for benchmarking
def _mock_predict(home_team_id, away_team_id, features=None):
    """Mock prediction function for benchmarking."""
    # Simulate computation with NumPy
    if features is None:
        # Create some random features
        features = np.random.random(20)
    
    # Simple model simulation
    home_strength = (home_team_id % 10) / 10.0
    away_strength = (away_team_id % 10) / 10.0
    
    # Calculate probabilities (home win, draw, away win)
    raw_probs = np.array([
        0.4 + home_strength - away_strength,  # Home win
        0.3,                                  # Draw
        0.3 + away_strength - home_strength   # Away win
    ])
    
    # Ensure probabilities are in valid range and sum to 1
    probs = np.clip(raw_probs, 0.05, 0.9)
    probs = probs / np.sum(probs)
    
    # Add random noise for realistic prediction variation
    noise = np.random.normal(0, 0.05, 3)
    probs = probs + noise
    probs = np.clip(probs, 0.05, 0.9)
    probs = probs / np.sum(probs)
    
    # Sleep to simulate actual computation time
    time.sleep(0.001)
    
    return {
        "home_win": float(probs[0]),
        "draw": float(probs[1]),
        "away_win": float(probs[2])
    }


class TestModelBenchmarks:
    """Benchmark tests for prediction models."""
    
    @pytest.mark.benchmark(
        group="models",
        min_time=0.1,
        max_time=0.5,
        min_rounds=50,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_model_prediction(self, benchmark):
        """Benchmark model prediction performance."""
        # Set up test data
        home_team_id = random.randint(1, 10)
        away_team_id = random.randint(1, 10)
        
        # Run the benchmark
        result = benchmark(_mock_predict, home_team_id, away_team_id)
        
        # Verify the result
        assert isinstance(result, dict)
        assert "home_win" in result
        assert "draw" in result
        assert "away_win" in result
        assert abs(sum(result.values()) - 1.0) < 1e-6  # Probabilities should sum to approximately 1
    
    @pytest.mark.benchmark(
        group="models",
        min_time=0.1,
        max_time=1.0,
        min_rounds=20,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_batch_prediction(self, benchmark):
        """Benchmark batch prediction performance."""
        # Set up test data - 10 matches
        batch_size = 10
        matches = [
            (random.randint(1, 10), random.randint(1, 10))
            for _ in range(batch_size)
        ]
        
        # Define batch prediction function
        def predict_batch(matches):
            return [_mock_predict(home_id, away_id) for home_id, away_id in matches]
        
        # Run the benchmark
        result = benchmark(predict_batch, matches)
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == batch_size
        
        for prediction in result:
            assert isinstance(prediction, dict)
            assert "home_win" in prediction
            assert "draw" in prediction
            assert "away_win" in prediction
            assert abs(sum(prediction.values()) - 1.0) < 1e-6


# Data processing functions to benchmark
def _process_match_data(matches, teams):
    """Process match data for model training (simplified for benchmark)."""
    # Create a dataframe for demonstration
    match_df = pd.DataFrame(matches)
    team_df = pd.DataFrame(teams)
    
    # Create features (simplified)
    result_df = pd.merge(
        match_df,
        team_df.rename(columns={'id': 'home_team_id', 'name': 'home_team_name', 'rating': 'home_team_rating'}),
        on='home_team_id'
    )
    
    result_df = pd.merge(
        result_df,
        team_df.rename(columns={'id': 'away_team_id', 'name': 'away_team_name', 'rating': 'away_team_rating'}),
        on='away_team_id'
    )
    
    # Calculate some features
    result_df['rating_diff'] = result_df['home_team_rating'] - result_df['away_team_rating']
    
    # Sleep to simulate actual computation time
    time.sleep(0.005)
    
    return result_df


class TestDataProcessingBenchmarks:
    """Benchmark tests for data processing components."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        # Generate sample match data
        matches = [
            {
                'id': i,
                'date': f'2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
                'home_team_id': random.randint(1, 20),
                'away_team_id': random.randint(1, 20),
                'home_goals': random.randint(0, 5),
                'away_goals': random.randint(0, 5),
                'competition_id': random.randint(1, 5)
            }
            for i in range(100)
        ]
        
        # Generate sample team data
        teams = [
            {
                'id': i,
                'name': f'Team {i}',
                'rating': random.uniform(70, 95)
            }
            for i in range(1, 21)
        ]
        
        return {
            'matches': matches,
            'teams': teams
        }
    
    @pytest.mark.benchmark(
        group="data_processing",
        min_time=0.1,
        max_time=1.0,
        min_rounds=20,
        timer=time.time,
        disable_gc=True,
        warmup=False
    )
    def test_match_data_processing(self, benchmark, sample_data):
        """Benchmark match data processing performance."""
        # Run the benchmark
        result = benchmark(_process_match_data, sample_data['matches'], sample_data['teams'])
        
        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data['matches'])
        assert 'rating_diff' in result.columns


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 