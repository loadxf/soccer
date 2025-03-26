"""
Authentication utilities for the Soccer Prediction System.
Handles user authentication, JWT token generation and validation.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, Any

import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.utils.logger import get_logger
from src.utils.db import get_db
from config.default_config import SECRET_KEY

# Setup logger
logger = get_logger("auth")

# Password handling
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 with password flow for token generation
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")

# Token settings
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))  # 1 hour by default
ALGORITHM = "HS256"

# User data - replace with proper database storage in production
# This is just a placeholder for development
users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "disabled": False,
        "hashed_password": pwd_context.hash("adminpassword"),
        "role": "admin"
    },
    "user": {
        "username": "user",
        "email": "user@example.com",
        "full_name": "Regular User",
        "disabled": False,
        "hashed_password": pwd_context.hash("userpassword"),
        "role": "user"
    }
}


class UserInDB(dict):
    """User model with hashed password for database operations."""
    def __init__(self, **data):
        super().__init__(**data)
        for field in ["username", "email", "hashed_password", "disabled"]:
            if field not in data:
                raise ValueError(f"Missing required user field: {field}")


class User(dict):
    """User model for API responses (without password)."""
    def __init__(self, **data):
        filtered_data = {k: v for k, v in data.items() if k != "hashed_password"}
        super().__init__(**filtered_data)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database by username."""
    # This should be replaced with a proper database lookup
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.get("hashed_password", "")):
        return None
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current authenticated user from the token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    
    return User(**user)


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Check if the current user is active."""
    if current_user.get("disabled", False):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def has_role(required_role: str):
    """Dependency for role-based access control."""
    async def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        user_role = current_user.get("role", "")
        if required_role == "admin" and user_role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return current_user
    return role_checker 