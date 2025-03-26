"""
Migration: Create users table
Created: 2023-10-10 12:00:00
"""

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def upgrade(engine):
    """
    Upgrade database schema.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    connection = engine.connect()
    metadata = MetaData()
    
    # Create users table
    users = Table(
        'users',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('username', String(50), nullable=False, unique=True),
        Column('email', String(255), nullable=False, unique=True),
        Column('password_hash', String(255), nullable=False),
        Column('full_name', String(100), nullable=True),
        Column('is_active', Boolean, nullable=False, default=True),
        Column('is_admin', Boolean, nullable=False, default=False),
        Column('created_at', DateTime, nullable=False, default=func.now()),
        Column('updated_at', DateTime, nullable=False, default=func.now(), onupdate=func.now())
    )
    
    # Create user_preferences table
    user_preferences = Table(
        'user_preferences',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        Column('preference_key', String(50), nullable=False),
        Column('preference_value', String(255), nullable=True),
        Column('created_at', DateTime, nullable=False, default=func.now()),
        Column('updated_at', DateTime, nullable=False, default=func.now(), onupdate=func.now()),
        UniqueConstraint('user_id', 'preference_key', name='uix_user_preference')
    )
    
    # Create user_activity_log table
    user_activity_log = Table(
        'user_activity_log',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('user_id', Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        Column('action', String(50), nullable=False),
        Column('entity_type', String(50), nullable=True),
        Column('entity_id', Integer, nullable=True),
        Column('details', Text, nullable=True),
        Column('ip_address', String(45), nullable=True),  # IPv6 support
        Column('user_agent', String(255), nullable=True),
        Column('created_at', DateTime, nullable=False, default=func.now())
    )
    
    # Create indices
    Index('idx_users_username', users.c.username)
    Index('idx_users_email', users.c.email)
    Index('idx_user_activity_log_user_id', user_activity_log.c.user_id)
    Index('idx_user_activity_log_created_at', user_activity_log.c.created_at)
    
    # Create tables
    metadata.create_all(connection)
    
    connection.close()


def downgrade(engine):
    """
    Downgrade database schema.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    connection = engine.connect()
    
    # Drop tables in reverse order
    connection.execute(text("DROP TABLE IF EXISTS user_activity_log"))
    connection.execute(text("DROP TABLE IF EXISTS user_preferences"))
    connection.execute(text("DROP TABLE IF EXISTS users"))
    
    connection.close() 