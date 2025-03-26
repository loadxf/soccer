"""
Integration tests for the Soccer Prediction System API.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add the project root to the path so that imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the FastAPI app
from src.api.server import app
from src.utils.auth import create_access_token, UserInDB, users_db

# Create test client
client = TestClient(app)


class TestAPIIntegration(unittest.TestCase):
    """Integration test cases for the API endpoints."""
    
    def setUp(self):
        """Set up test environment."""
        # Store original environment variables
        self.original_secret_key = os.environ.get("SECRET_KEY")
        self.original_expire_minutes = os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")
        
        # Set environment variables for testing
        os.environ["SECRET_KEY"] = "test_secret_key"
        os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "15"
        
        # Test user data
        self.test_username = "testuser"
        self.test_password = "testpassword"
        self.test_email = "test@example.com"
        
        # Add test user to users_db with admin role
        from src.utils.auth import get_password_hash
        users_db[self.test_username] = {
            "username": self.test_username,
            "email": self.test_email,
            "hashed_password": get_password_hash(self.test_password),
            "disabled": False,
            "roles": ["user", "admin"]
        }
        
        # Create access token for tests
        token_data = {"sub": self.test_username}
        self.access_token = create_access_token(token_data)
        self.headers = {"Authorization": f"Bearer {self.access_token}"}
        
    def tearDown(self):
        """Clean up test environment."""
        # Reset environment variables
        if self.original_secret_key:
            os.environ["SECRET_KEY"] = self.original_secret_key
        else:
            os.environ.pop("SECRET_KEY", None)
            
        if self.original_expire_minutes:
            os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = self.original_expire_minutes
        else:
            os.environ.pop("ACCESS_TOKEN_EXPIRE_MINUTES", None)
        
        # Clean up test user if added
        if self.test_username in users_db:
            del users_db[self.test_username]
    
    def test_root_endpoint(self):
        """Test the root API endpoint."""
        response = client.get("/api/v1/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "Welcome to the Soccer Prediction System API")
        self.assertEqual(data["docs"], "/api/v1/docs")
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)
        self.assertIn("uptime", data)
    
    @patch("src.api.server.db_session")
    def test_teams_endpoint(self, mock_db_session):
        """Test the teams endpoint."""
        # Mock database response
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Manchester United", "country": "England"},
            {"id": 2, "name": "Barcelona", "country": "Spain"}
        ]
        mock_db_session.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Test without authentication
        response = client.get("/api/v1/teams")
        self.assertEqual(response.status_code, 401)  # Unauthorized
        
        # Test with authentication
        response = client.get("/api/v1/teams", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["name"], "Manchester United")
        self.assertEqual(data[1]["name"], "Barcelona")
    
    def test_match_endpoint(self):
        """Test the match endpoint."""
        # Test without authentication
        response = client.get("/api/v1/matches/1")
        self.assertEqual(response.status_code, 401)  # Unauthorized
        
        # Test with authentication - existing match
        response = client.get("/api/v1/matches/1", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], 1)
        self.assertEqual(data["home_team"], "Manchester United")
        self.assertEqual(data["away_team"], "Barcelona")
        
        # Test non-existent match
        response = client.get("/api/v1/matches/999", headers=self.headers)
        self.assertEqual(response.status_code, 404)
    
    @patch("src.api.server.prediction_service.predict_match")
    def test_predict_match_endpoint(self, mock_predict_match):
        """Test the match prediction endpoint."""
        # Mock prediction service response
        mock_predict_match.return_value = {
            "home_win_prob": 0.6,
            "draw_prob": 0.2,
            "away_win_prob": 0.2,
            "prediction": "HOME_WIN"
        }
        
        # Test without authentication
        response = client.get("/api/v1/predictions/match/1")
        self.assertEqual(response.status_code, 401)  # Unauthorized
        
        # Test with authentication - existing match
        response = client.get("/api/v1/predictions/match/1", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["match"]["id"], 1)
        self.assertEqual(data["match"]["home_team"], "Manchester United")
        self.assertEqual(data["match"]["away_team"], "Barcelona")
        self.assertEqual(data["prediction"]["prediction"], "HOME_WIN")
        self.assertEqual(data["prediction"]["home_win_prob"], 0.6)
        
        # Test with specific model
        response = client.get("/api/v1/predictions/match/1?model_name=xgboost", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        # Verify the model_name was passed to the service
        mock_predict_match.assert_called_with(home_team_id=1, away_team_id=2, model_name="xgboost")
        
        # Test non-existent match
        response = client.get("/api/v1/predictions/match/999", headers=self.headers)
        self.assertEqual(response.status_code, 404)
    
    @patch("src.api.server.prediction_service.predict_match")
    def test_custom_prediction_endpoint(self, mock_predict_match):
        """Test the custom match prediction endpoint."""
        # Mock prediction service response
        mock_predict_match.return_value = {
            "home_win_prob": 0.4,
            "draw_prob": 0.3,
            "away_win_prob": 0.3,
            "prediction": "HOME_WIN"
        }
        
        # Test data
        request_data = {
            "home_team_id": 3,
            "away_team_id": 4
        }
        
        # Test without authentication
        response = client.post("/api/v1/predictions/custom", json=request_data)
        self.assertEqual(response.status_code, 401)  # Unauthorized
        
        # Test with authentication
        response = client.post("/api/v1/predictions/custom", json=request_data, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["match"]["home_team_id"], 3)
        self.assertEqual(data["match"]["away_team_id"], 4)
        self.assertEqual(data["prediction"]["prediction"], "HOME_WIN")
        self.assertEqual(data["prediction"]["home_win_prob"], 0.4)
        
        # Test without required fields
        incomplete_data = {"home_team_id": 3}
        response = client.post("/api/v1/predictions/custom", json=incomplete_data, headers=self.headers)
        self.assertEqual(response.status_code, 400)
    
    @patch("src.api.server.prediction_service.predict_match")
    def test_batch_predictions_endpoint(self, mock_predict_match):
        """Test the batch predictions endpoint."""
        # Mock prediction service response
        mock_predict_match.return_value = {
            "home_win_prob": 0.5,
            "draw_prob": 0.3,
            "away_win_prob": 0.2,
            "prediction": "HOME_WIN"
        }
        
        # Test data
        request_data = {
            "matches": [
                {"home_team_id": 1, "away_team_id": 2},
                {"home_team_id": 3, "away_team_id": 4}
            ]
        }
        
        # Test without authentication
        response = client.post("/api/v1/predictions/batch", json=request_data)
        self.assertEqual(response.status_code, 401)  # Unauthorized
        
        # Test without admin role (need to modify the token)
        non_admin_user = "regular_user"
        users_db[non_admin_user] = {
            "username": non_admin_user,
            "email": "regular@example.com",
            "hashed_password": "hashed_password",
            "disabled": False,
            "roles": ["user"]  # No admin role
        }
        token_data = {"sub": non_admin_user}
        regular_token = create_access_token(token_data)
        regular_headers = {"Authorization": f"Bearer {regular_token}"}
        
        response = client.post("/api/v1/predictions/batch", json=request_data, headers=regular_headers)
        self.assertEqual(response.status_code, 403)  # Forbidden
        
        # Clean up non-admin user
        del users_db[non_admin_user]
        
        # Test with admin authentication
        response = client.post("/api/v1/predictions/batch", json=request_data, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["predictions"][0]["match"]["home_team_id"], 1)
        self.assertEqual(data["predictions"][1]["match"]["home_team_id"], 3)
        
        # Test with invalid request format
        invalid_data = {"teams": [{"home": 1, "away": 2}]}  # Wrong format
        response = client.post("/api/v1/predictions/batch", json=invalid_data, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        
        # Test with empty matches list
        empty_data = {"matches": []}
        response = client.post("/api/v1/predictions/batch", json=empty_data, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["predictions"]), 0)
    
    def test_prediction_history_endpoint(self):
        """Test the prediction history endpoint."""
        # Test without authentication
        response = client.get("/api/v1/predictions/history")
        self.assertEqual(response.status_code, 401)  # Unauthorized
        
        # Test with authentication
        response = client.get("/api/v1/predictions/history", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("history", data)
        self.assertIn("total", data)
        
        # Test with filter parameters
        response = client.get(
            "/api/v1/predictions/history?limit=10&home_team_id=1&model_name=xgboost",
            headers=self.headers
        )
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main() 