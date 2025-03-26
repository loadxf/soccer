"""
Locust file for performance testing the Soccer Prediction System API.
This script defines different user behavior scenarios to test the performance of the API endpoints.
"""

import json
import random
import time
from typing import Dict, List, Optional
import os

from locust import HttpUser, task, tag, between, events
from locust.exception import StopUser
from locust.env import Environment

# Define API endpoints and paths
API_PREFIX = "/api/v1"
AUTH_ENDPOINT = f"{API_PREFIX}/auth/token"
TEAMS_ENDPOINT = f"{API_PREFIX}/teams"
MATCHES_ENDPOINT = f"{API_PREFIX}/matches"
PREDICTIONS_MODELS_ENDPOINT = f"{API_PREFIX}/predictions/models"
PREDICTIONS_MATCH_ENDPOINT = f"{API_PREFIX}/predictions/match"
PREDICTIONS_CUSTOM_ENDPOINT = f"{API_PREFIX}/predictions/custom"
PREDICTIONS_BATCH_ENDPOINT = f"{API_PREFIX}/predictions/batch"
HEALTH_ENDPOINT = f"{API_PREFIX}/health"

# Test user credentials (should match test users in the system)
REGULAR_USER = {"username": "test_user", "password": "testpassword"}
ADMIN_USER = {"username": "admin_user", "password": "adminpassword"}

# Test data for API calls
TEAM_IDS = [1, 2, 3, 4, 5]
MATCH_IDS = [1, 2, 3]
COMPETITION_IDS = [1, 2]


class SoccerPredictionUser(HttpUser):
    """Base user class for Soccer Prediction System API testing."""
    
    wait_time = between(1, 5)  # Wait between 1-5 seconds between tasks
    
    # Store the auth token for authenticated requests
    token: Optional[str] = None
    
    def on_start(self):
        """Log in before starting tasks."""
        self.login()
    
    def login(self):
        """Authenticate with the API and get a token."""
        credentials = REGULAR_USER
        
        # For admin users, override credentials
        if isinstance(self, AdminUser):
            credentials = ADMIN_USER
        
        with self.client.post(
            AUTH_ENDPOINT,
            data=credentials,
            catch_response=True,
            name="Auth: Login"
        ) as response:
            if response.status_code == 200:
                # Store the token for future requests
                response_data = response.json()
                self.token = response_data.get("access_token")
                if not self.token:
                    response.failure("No token in response")
                    raise StopUser()
            else:
                response.failure(f"Login failed with status code: {response.status_code}")
                raise StopUser()
    
    def authenticated_request(self, method, endpoint, name=None, data=None, params=None):
        """Make an authenticated request to the API."""
        if not self.token:
            self.login()
            
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Convert data to JSON if it's a dictionary
        if isinstance(data, dict):
            headers["Content-Type"] = "application/json"
            data = json.dumps(data)
        
        # Use the specified method (GET, POST, etc.)
        request_method = getattr(self.client, method.lower())
        
        # Make the request
        with request_method(
            endpoint,
            headers=headers,
            data=data,
            params=params,
            catch_response=True,
            name=name
        ) as response:
            # Automatically mark 4xx/5xx as failures
            if response.status_code >= 400:
                response.failure(f"Request failed with status code: {response.status_code}")
            
            return response


class RegularUser(SoccerPredictionUser):
    """Simulates a regular user accessing team and match data and making predictions."""
    
    @tag("health")
    @task(1)
    def check_health(self):
        """Check the health endpoint (no auth required)."""
        with self.client.get(HEALTH_ENDPOINT, name="Health: Check") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed with status code: {response.status_code}")
    
    @tag("teams")
    @task(5)
    def get_teams(self):
        """Get the list of teams."""
        # Sometimes include pagination parameters
        params = {}
        if random.random() > 0.5:
            params = {"skip": random.randint(0, 10), "limit": random.randint(5, 20)}
            
        self.authenticated_request("GET", TEAMS_ENDPOINT, name="Teams: List", params=params)
    
    @tag("teams")
    @task(3)
    def get_single_team(self):
        """Get details for a single team."""
        team_id = random.choice(TEAM_IDS)
        self.authenticated_request("GET", f"{TEAMS_ENDPOINT}/{team_id}", name="Teams: Get Single")
    
    @tag("matches")
    @task(5)
    def get_matches(self):
        """Get the list of matches."""
        # Sometimes include filter parameters
        params = {}
        if random.random() > 0.5:
            params = {"limit": random.randint(5, 20)}
            
            # Add additional filters randomly
            if random.random() > 0.7:
                params["team_id"] = random.choice(TEAM_IDS)
            if random.random() > 0.7:
                params["competition_id"] = random.choice(COMPETITION_IDS)
                
        self.authenticated_request("GET", MATCHES_ENDPOINT, name="Matches: List", params=params)
    
    @tag("matches")
    @task(3)
    def get_single_match(self):
        """Get details for a single match."""
        match_id = random.choice(MATCH_IDS)
        self.authenticated_request("GET", f"{MATCHES_ENDPOINT}/{match_id}", name="Matches: Get Single")
    
    @tag("predictions")
    @task(2)
    def get_prediction_models(self):
        """Get available prediction models."""
        self.authenticated_request("GET", PREDICTIONS_MODELS_ENDPOINT, name="Predictions: Get Models")
    
    @tag("predictions")
    @task(5)
    def predict_match(self):
        """Get prediction for a match."""
        match_id = random.choice(MATCH_IDS)
        
        # Sometimes include a specific model name
        params = {}
        if random.random() > 0.7:
            # Placeholder for model name - replace with actual model names if known
            params["model_name"] = random.choice(["ensemble", "xgboost", "neural"])
            
        self.authenticated_request(
            "GET", 
            f"{PREDICTIONS_MATCH_ENDPOINT}/{match_id}", 
            name="Predictions: Predict Match",
            params=params
        )
    
    @tag("predictions")
    @task(2)
    def predict_custom_match(self):
        """Get prediction for a custom match."""
        # Create a custom match configuration
        data = {
            "home_team_id": random.choice(TEAM_IDS),
            "away_team_id": random.choice(TEAM_IDS),
            "match_date": "2023-06-01",  # Example date
            "competition_id": random.choice(COMPETITION_IDS)
        }
        
        self.authenticated_request(
            "POST", 
            PREDICTIONS_CUSTOM_ENDPOINT, 
            name="Predictions: Custom Match",
            data=data
        )


class AdminUser(SoccerPredictionUser):
    """Simulates an admin user with additional access to batch predictions and admin endpoints."""
    
    # Inherit tasks from RegularUser
    @tag("predictions", "admin")
    @task(2)
    def predict_batch_matches(self):
        """Get predictions for multiple matches in batch."""
        # Create batch prediction request with 2-5 matches
        num_matches = random.randint(2, 5)
        matches = []
        
        for _ in range(num_matches):
            matches.append({
                "home_team_id": random.choice(TEAM_IDS),
                "away_team_id": random.choice(TEAM_IDS)
            })
        
        data = {"matches": matches}
        
        self.authenticated_request(
            "POST", 
            PREDICTIONS_BATCH_ENDPOINT, 
            name="Predictions: Batch",
            data=data
        )
    
    @tag("admin")
    @task(1)
    def reset_model_cache(self):
        """Reset the model cache (admin only)."""
        self.authenticated_request(
            "POST", 
            f"{API_PREFIX}/admin/reset-model-cache", 
            name="Admin: Reset Model Cache"
        )
    
    @tag("admin")
    @task(1)
    def reset_cache(self):
        """Reset the cache (admin only)."""
        scope = random.choice(["all", "teams", "matches", "predictions"])
        data = {"scope": scope}
        
        self.authenticated_request(
            "POST", 
            f"{API_PREFIX}/admin/reset-cache", 
            name="Admin: Reset Cache",
            data=data
        )


class AnonymousUser(HttpUser):
    """Simulates an unauthenticated user browsing public endpoints."""
    
    wait_time = between(1, 3)
    
    @tag("health")
    @task(5)
    def check_health(self):
        """Check the health endpoint."""
        with self.client.get(HEALTH_ENDPOINT, name="Health: Check (Anonymous)") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed with status code: {response.status_code}")
    
    @tag("auth")
    @task(2)
    def try_login_with_invalid_credentials(self):
        """Try to log in with invalid credentials."""
        credentials = {
            "username": f"invalid_user_{random.randint(1, 1000)}",
            "password": "wrongpassword"
        }
        
        with self.client.post(
            AUTH_ENDPOINT,
            data=credentials,
            catch_response=True,
            name="Auth: Invalid Login"
        ) as response:
            # This should fail with 401
            if response.status_code == 401:
                response.success()
            else:
                response.failure(f"Expected 401, got {response.status_code}")
    
    @tag("unauthorized")
    @task(2)
    def try_access_protected_endpoint(self):
        """Try to access a protected endpoint without authentication."""
        with self.client.get(
            TEAMS_ENDPOINT,
            catch_response=True,
            name="Unauthorized: Access Protected Endpoint"
        ) as response:
            # This should fail with 401
            if response.status_code == 401:
                response.success()
            else:
                response.failure(f"Expected 401, got {response.status_code}")


# Custom load shape for more realistic testing scenarios
# Uncomment and customize if needed
# from locust import LoadTestShape
# class StagesLoadShape(LoadTestShape):
#     """Custom load shape with stages for gradual ramp-up and sustained load."""
#     
#     stages = [
#         {"duration": 60, "users": 10, "spawn_rate": 1},
#         {"duration": 120, "users": 50, "spawn_rate": 5},
#         {"duration": 180, "users": 100, "spawn_rate": 10},
#         {"duration": 240, "users": 100, "spawn_rate": 10},
#         {"duration": 300, "users": 50, "spawn_rate": 5},
#         {"duration": 360, "users": 10, "spawn_rate": 5},
#         {"duration": 420, "users": 1, "spawn_rate": 1},
#     ]
#     
#     def tick(self):
#         run_time = self.get_run_time()
#         
#         for stage in self.stages:
#             if run_time < stage["duration"]:
#                 return stage["users"], stage["spawn_rate"]
#             
#         return None 