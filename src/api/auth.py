"""
Authentication endpoints for the Soccer Prediction System API.
Implements login, token generation, and user management.
"""

from datetime import timedelta
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse

from src.utils.auth import (
    authenticate_user, create_access_token, get_current_active_user,
    User, has_role, ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.utils.logger import get_logger
from config.default_config import API_PREFIX

# Setup logger
logger = get_logger("api.auth")

# Create router
router = APIRouter(prefix=f"{API_PREFIX}/auth", tags=["authentication"])


class Token(dict):
    """Token response model."""
    def __init__(self, access_token: str, token_type: str):
        super().__init__(access_token=access_token, token_type=token_type)


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate and get an access token.
    
    This endpoint follows the OAuth2 password flow and returns a JWT token
    that can be used to authenticate subsequent requests.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if user is None:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token with user information
    token_data = {"sub": user.get("username", "")}
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    logger.info(f"Successful login for user: {form_data.username}")
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get information about the currently authenticated user."""
    return current_user


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout the current user.
    
    Note: JWT tokens cannot be invalidated on the server side without additional
    infrastructure like a token blacklist. This endpoint is provided for API
    completeness but clients should dispose of the token themselves.
    """
    # In a more complex implementation, we would add the token to a blacklist
    # For now, we just return a success message
    logger.info(f"Logout for user: {current_user.get('username', '')}")
    return {"detail": "Successfully logged out"}


@router.get("/check-admin")
async def check_admin_access(current_user: User = Depends(has_role("admin"))):
    """Check if the current user has admin access."""
    return {"detail": "User has admin access", "username": current_user.get("username", "")}


@router.get("/refresh-token", response_model=Token)
async def refresh_access_token(current_user: User = Depends(get_current_active_user)):
    """Refresh the access token for the current user."""
    # Create a new access token
    token_data = {"sub": current_user.get("username", "")}
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    logger.info(f"Token refreshed for user: {current_user.get('username', '')}")
    return Token(access_token=access_token, token_type="bearer") 