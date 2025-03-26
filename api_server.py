"""
Standalone API server for the Soccer Prediction System.
This version includes enhanced CORS handling for improved browser compatibility.
"""

import uvicorn
import os
import sys
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("api_server")

# Import configuration from shared config
try:
    from api_config import (
        API_HOST, API_PORT, API_VERSION, 
        CORS_ORIGINS, API_TITLE, API_DESCRIPTION,
        ENABLE_DEBUG
    )
    logger.info(f"Using configuration from api_config: host={API_HOST}, port={API_PORT}")
except ImportError:
    # Fallback configuration if api_config.py doesn't exist
    logger.warning("Could not import api_config, using default configuration")
    API_HOST = "0.0.0.0"  # Listen on all interfaces
    API_PORT = 8080
    API_VERSION = "v1"
    CORS_ORIGINS = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "*"  # Allow all origins
    ]
    API_TITLE = "Soccer Prediction System API"
    API_DESCRIPTION = "REST API for soccer match predictions"
    ENABLE_DEBUG = True

# API prefix is always constructed the same way
API_PREFIX = f"/api/{API_VERSION}"

# Create app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "X-API-Version"],
    max_age=86400,  # 24 hours
)

# Exception handler for consistent error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all exceptions to ensure proper CORS headers are set."""
    logger.error(f"Global exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": request.url.path,
            "version": API_VERSION
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Add middleware to ensure CORS headers are set correctly
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    """Add CORS headers to every response, including errors."""
    try:
        response = await call_next(request)
        
        # Ensure CORS headers are set
        if "Access-Control-Allow-Origin" not in response.headers:
            origin = request.headers.get("origin", "*")
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
        
        return response
    except Exception as e:
        logger.error(f"Middleware exception: {str(e)}")
        
        # If there's an exception, return a JSON response with CORS headers
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e),
                "path": request.url.path,
                "version": API_VERSION
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )

# Health check endpoint
@app.get(f"{API_PREFIX}/health", tags=["Health"])
async def health_check():
    """Check API health."""
    return {
        "status": "ok",
        "api": "up",
        "database": "unknown",
        "models": "unknown",
        "version": API_VERSION
    }

# Root endpoint
@app.get(f"{API_PREFIX}/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the Soccer Prediction System API",
        "version": API_VERSION,
        "docs_url": f"{API_PREFIX}/docs"
    }

# Teams endpoint
@app.get(f"{API_PREFIX}/teams", tags=["Teams"])
async def get_teams():
    """Get a list of teams."""
    return [
        {"id": 1, "name": "Team A", "country": "Country A"},
        {"id": 2, "name": "Team B", "country": "Country B"},
        {"id": 3, "name": "Team C", "country": "Country C"},
    ]

# Matches endpoint
@app.get(f"{API_PREFIX}/matches", tags=["Matches"])
async def get_matches():
    """Get a list of matches."""
    return [
        {"id": 1, "home_team": "Team A", "away_team": "Team B", "date": "2025-01-01"},
        {"id": 2, "home_team": "Team C", "away_team": "Team A", "date": "2025-01-08"},
    ]

# Prediction models endpoint
@app.get(f"{API_PREFIX}/predictions/models", tags=["Predictions"])
async def get_prediction_models():
    """Get available prediction models."""
    return [
        {"id": 1, "name": "Ensemble", "type": "ensemble", "version": "1.0"},
        {"id": 2, "name": "Logistic Regression", "type": "baseline", "version": "1.0"},
    ]

if __name__ == "__main__":
    logger.info(f"Starting standalone API server at http://{API_HOST}:{API_PORT}")
    logger.info(f"API documentation: http://{API_HOST}:{API_PORT}{API_PREFIX}/docs")
    try:
        uvicorn.run(app, host=API_HOST, port=API_PORT)
    except Exception as e:
        logger.error(f"Error starting server: {e}") 