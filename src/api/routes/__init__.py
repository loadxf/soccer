"""
API routes for the soccer prediction system.
"""

from fastapi import APIRouter
# Import upcoming_matches router
from src.api.routes.upcoming_matches import router as upcoming_matches_router

# Create main router
api_router = APIRouter()

# Include all API routes
api_router.include_router(upcoming_matches_router)

# Export routers for API server
__all__ = ["api_router"] 