"""
Migration: Add user profile fields
Created: 2023-10-15 15:30:00
"""

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def upgrade(engine):
    """
    Upgrade database schema by adding profile fields to users table.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    connection = engine.connect()
    
    # Add new columns to users table
    connection.execute(text("""
        ALTER TABLE users
        ADD COLUMN bio TEXT,
        ADD COLUMN location VARCHAR(100),
        ADD COLUMN profile_image_url VARCHAR(255),
        ADD COLUMN last_login_at TIMESTAMP,
        ADD COLUMN timezone VARCHAR(50) DEFAULT 'UTC'
    """))
    
    # Create new index for location
    connection.execute(text("""
        CREATE INDEX idx_users_location ON users (location)
    """))
    
    connection.close()


def downgrade(engine):
    """
    Downgrade database schema by removing profile fields from users table.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    connection = engine.connect()
    
    # Drop index first
    connection.execute(text("""
        DROP INDEX IF EXISTS idx_users_location
    """))
    
    # Remove columns
    connection.execute(text("""
        ALTER TABLE users
        DROP COLUMN IF EXISTS bio,
        DROP COLUMN IF EXISTS location,
        DROP COLUMN IF EXISTS profile_image_url,
        DROP COLUMN IF EXISTS last_login_at,
        DROP COLUMN IF EXISTS timezone
    """))
    
    connection.close() 