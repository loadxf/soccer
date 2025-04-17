"""
Simple standalone API server for the Soccer Prediction System.
This is a minimal version that should work reliably on all systems.
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("simple_api_server")

# Configuration
API_HOST = "0.0.0.0"  # Listen on all interfaces for both localhost and 127.0.0.1
API_PORT = int(os.environ.get('API_PORT', 8000))
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# Define allowed CORS origins
CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://app:8000",
    "http://ui:8501",
    "*"  # Allow all origins for now to ensure remote connections work
]

# Create the FastAPI app
app = FastAPI(
    title="Soccer Prediction System API",
    description="REST API for soccer match predictions",
    version=API_VERSION,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Use specific origins instead of wildcard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for CORS headers on errors
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    try:
        response = await call_next(request)
        
        # Ensure CORS headers are set
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    except Exception as e:
        logger.error(f"Error in middleware: {str(e)}")
        
        # Return error with CORS headers
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Soccer Prediction System API - Use the /api/v1/ endpoint for API access",
        "api_url": f"/api/{API_VERSION}/",
        "docs_url": f"/api/{API_VERSION}/docs",
    }

# Health check endpoint - make this a simple root level endpoint for easier checks
@app.get("/health")
async def root_health_check():
    """Root health check endpoint"""
    return {
        "status": "ok",
        "api": "up",
        "database": "unknown",
        "models": "unknown",
        "version": API_VERSION
    }

# Health check endpoint with API prefix
@app.get(f"{API_PREFIX}/health")
async def health_check():
    """Check API health"""
    return {
        "status": "ok",
        "api": "up",
        "database": "unknown",
        "models": "unknown",
        "version": API_VERSION
    }

# API root endpoint
@app.get(f"{API_PREFIX}/")
async def api_root():
    """API v1 root endpoint"""
    return {
        "message": "Welcome to the Soccer Prediction System API",
        "version": API_VERSION,
        "docs_url": f"{API_PREFIX}/docs"
    }

# Teams endpoint
@app.get(f"{API_PREFIX}/teams")
async def get_teams():
    """Get a list of teams"""
    return [
        {"id": 1, "name": "Team A", "country": "Country A"},
        {"id": 2, "name": "Team B", "country": "Country B"},
        {"id": 3, "name": "Team C", "country": "Country C"},
    ]

# Matches endpoint
@app.get(f"{API_PREFIX}/matches")
async def get_matches():
    """Get a list of matches"""
    return [
        {"id": 1, "home_team": "Team A", "away_team": "Team B", "date": "2025-01-01"},
        {"id": 2, "home_team": "Team C", "away_team": "Team A", "date": "2025-01-08"},
    ]

# Prediction models endpoint
@app.get(f"{API_PREFIX}/predictions/models")
async def get_prediction_models():
    """Get available prediction models"""
    return [
        {"id": 1, "name": "Ensemble", "type": "ensemble", "version": "1.0"},
        {"id": 2, "name": "Logistic Regression", "type": "baseline", "version": "1.0"},
    ]

if __name__ == "__main__":
    logger.info(f"Starting simple API server on {API_HOST}:{API_PORT}")
    logger.info(f"API documentation: http://localhost:{API_PORT}{API_PREFIX}/docs")
    
    try:
        uvicorn.run(app, host=API_HOST, port=API_PORT)
    except Exception as e:
        logger.error(f"Error starting server: {e}") 