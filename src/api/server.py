"""
API Server for soccer prediction system.
Provides RESTful endpoints for data access and predictions.
"""

import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any

# Import project components
from src.utils.logger import get_logger
from src.api.routes import api_router

# Import API configuration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from api_config import CORS_ORIGINS
except ImportError:
    # Fallback if api_config.py can't be imported
    CORS_ORIGINS = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://app:8000",
        "http://ui:8501"
    ]

# Setup logger
logger = get_logger("api.server")

# Create FastAPI app
app = FastAPI(
    title="Soccer Prediction API",
    description="API for soccer prediction system with historical data and upcoming matches",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Use specific origins from config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes
app.include_router(api_router)

# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Soccer Prediction API",
        "version": "1.0.0",
        "description": "API for soccer prediction system with historical data and upcoming matches"
    }

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "API server is running"
    }

# Error handling for 404 errors
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "message": "The requested resource was not found"
        }
    )

# Error handling for 500 errors
@app.exception_handler(500)
async def server_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error"
        }
    )

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port) 