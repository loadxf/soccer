"""
API documentation configuration module.

This module provides enhanced documentation settings for the Soccer Prediction System API.
It improves the auto-generated Swagger/OpenAPI documentation by adding detailed descriptions,
examples, and proper organization for the API endpoints.
"""

from typing import Dict, List, Any

# Tags for grouping endpoints
tags_metadata = [
    {
        "name": "authentication",
        "description": "Operations related to user authentication and authorization",
    },
    {
        "name": "teams",
        "description": "Access to soccer team information and statistics",
    },
    {
        "name": "players",
        "description": "Operations for retrieving player information and statistics",
    },
    {
        "name": "matches",
        "description": "Operations for retrieving soccer match data and results",
    },
    {
        "name": "predictions",
        "description": "Endpoints to get match and player performance predictions",
    },
    {
        "name": "admin",
        "description": "Administrative operations for system management",
    },
    {
        "name": "health",
        "description": "API health and status checking endpoints",
    },
]

# Security scheme definitions
security_schemes = {
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Enter the JWT token obtained from the login endpoint",
    },
    "ApiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key for application authentication",
    },
}

# Request examples
request_examples = {
    "login": {
        "summary": "Login request",
        "value": {
            "username": "user@example.com",
            "password": "password123"
        }
    },
    "custom_prediction": {
        "summary": "Custom prediction request",
        "value": {
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
    },
    "batch_prediction": {
        "summary": "Batch prediction request",
        "value": {
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
    },
    "reset_cache": {
        "summary": "Reset cache request",
        "value": {
            "scope": "teams"
        }
    }
}

# Response examples
response_examples = {
    "token": {
        "summary": "Authentication token",
        "value": {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer",
            "expires_in": 3600,
            "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        }
    },
    "team_detail": {
        "summary": "Team details",
        "value": {
            "id": 1,
            "name": "Manchester United",
            "country": "England",
            "founded": 1878,
            "stadium": "Old Trafford",
            "league": "Premier League",
            "logo_url": "https://example.com/logos/manchester_united.png",
            "website": "https://www.manutd.com"
        }
    },
    "teams_list": {
        "summary": "List of teams",
        "value": {
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
    },
    "match_detail": {
        "summary": "Match details",
        "value": {
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
    },
    "matches_list": {
        "summary": "List of matches",
        "value": {
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
    },
    "prediction": {
        "summary": "Match prediction",
        "value": {
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
    },
    "models_list": {
        "summary": "Available prediction models",
        "value": {
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
    },
    "health": {
        "summary": "API health status",
        "value": {
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
    },
    "error": {
        "summary": "Error response",
        "value": {
            "detail": "An error occurred while processing your request",
            "error_code": "INVALID_REQUEST",
            "timestamp": "2023-02-01T12:34:56"
        }
    }
}

# Error codes description
error_codes = {
    "AUTHENTICATION_FAILED": "Authentication credentials are invalid",
    "TOKEN_EXPIRED": "The access token has expired",
    "PERMISSION_DENIED": "User does not have permission to access this resource",
    "RATE_LIMIT_EXCEEDED": "API rate limit has been exceeded",
    "RESOURCE_NOT_FOUND": "The requested resource was not found",
    "INVALID_REQUEST": "The request data is invalid or malformed",
    "SERVER_ERROR": "An unexpected server error occurred",
    "DATABASE_ERROR": "Error connecting to or querying the database",
    "VALIDATION_ERROR": "Request data failed validation checks",
    "MODEL_ERROR": "Error in prediction model processing"
}

# Common parameters description
common_parameters = {
    "skip": {
        "description": "Number of items to skip for pagination",
        "example": 0
    },
    "limit": {
        "description": "Maximum number of items to return",
        "example": 20
    },
    "from_date": {
        "description": "Filter results from this date (YYYY-MM-DD)",
        "example": "2023-01-01"
    },
    "to_date": {
        "description": "Filter results up to this date (YYYY-MM-DD)",
        "example": "2023-01-31"
    },
    "team_id": {
        "description": "Filter by team identifier",
        "example": 1
    },
    "competition_id": {
        "description": "Filter by competition identifier",
        "example": 1
    }
}


def get_openapi_schema() -> Dict[str, Any]:
    """
    Return custom OpenAPI schema enhancements for the Soccer Prediction System API.
    
    This function generates additional schema details that are merged with the 
    auto-generated OpenAPI schema by FastAPI to produce a more comprehensive and 
    user-friendly API documentation.
    """
    openapi_schema = {
        "info": {
            "title": "Soccer Prediction System API",
            "description": """
## Soccer Prediction System API

This API provides soccer match predictions and related data including teams, 
matches, competitions, and prediction models.

### Features

* **Team Information**: Access detailed information about soccer teams
* **Match Data**: Retrieve historical match data and upcoming fixtures
* **Predictions**: Get win/draw/loss probabilities for upcoming matches
* **Models**: Choose from various prediction models or use the ensemble approach
* **Batch Processing**: Submit multiple matches for prediction in a single request
* **Customization**: Provide additional features to influence prediction results

### Rate Limiting

To ensure fair usage, this API implements rate limiting:
* Authenticated users: 100 requests per minute
* Anonymous users: 20 requests per minute

### Caching

The API uses Redis-based caching to improve performance for frequently accessed data:
* Team data: cached for 1 hour
* Match data: cached for 30 minutes
* Prediction results: cached for 10 minutes
* Health status: cached for 10 seconds

### Status Codes

* `200`: Success
* `201`: Created
* `400`: Bad request
* `401`: Unauthorized
* `403`: Forbidden
* `404`: Not found
* `429`: Too many requests
* `500`: Server error

### Authentication

* Bearer Token: Send a JWT token in the `Authorization` header
* API Key: Send your API key in the `X-API-Key` header
            """,
            "version": "1.0.0",
            "contact": {
                "name": "Soccer Prediction System Support",
                "url": "https://www.example.com/support",
                "email": "api-support@example.com",
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT",
            },
        }
    }
    
    return openapi_schema 