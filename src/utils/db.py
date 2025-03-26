"""
Database utilities for the Soccer Prediction System.
Handles database connections and common operations.
"""

import os
from contextlib import contextmanager
from typing import Optional, Dict, Any, Iterator, Union

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pymongo import MongoClient
import redis

# Import logger and config
from src.utils.logger import get_logger
try:
    from config.default_config import (
        DB_TYPE, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
        MONGO_URI, REDIS_HOST, REDIS_PORT, REDIS_DB
    )
except ImportError:
    # Fallback defaults if config is not available
    DB_TYPE = os.getenv("DB_TYPE", "postgres")
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "soccer_prediction")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/soccer_prediction")
    REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Setup logger
logger = get_logger("db")

# SQLAlchemy base class
Base = declarative_base()
metadata = MetaData()

# Connection strings
if DB_TYPE == "postgres":
    DB_CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
elif DB_TYPE == "mysql":
    DB_CONNECTION_STRING = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
elif DB_TYPE == "sqlite":
    DB_CONNECTION_STRING = f"sqlite:///{DB_NAME}.db"
else:
    raise ValueError(f"Unsupported database type: {DB_TYPE}")

# SQLAlchemy engine and session
engine = create_engine(DB_CONNECTION_STRING, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# MongoDB client
_mongo_client: Optional[MongoClient] = None

# Redis client
_redis_client: Optional[redis.Redis] = None


def get_sql_engine():
    """Get SQLAlchemy engine instance."""
    return engine


def get_db() -> Session:
    """Get a SQLAlchemy database session."""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        logger.error(f"Error getting database session: {e}")
        raise


@contextmanager
def db_session() -> Iterator[Session]:
    """Context manager for database sessions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_mongo_client() -> MongoClient:
    """Get a MongoDB client instance (singleton)."""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(MONGO_URI)
            # Ping the server to verify connection
            _mongo_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"MongoDB connection error: {e}")
            raise
    return _mongo_client


def get_mongo_db(db_name: Optional[str] = None) -> Any:
    """Get a MongoDB database instance."""
    client = get_mongo_client()
    if db_name is None:
        # Extract database name from URI if not provided
        db_name = MONGO_URI.split("/")[-1]
    return client[db_name]


def get_redis_client() -> redis.Redis:
    """Get a Redis client instance (singleton)."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            # Check connection
            _redis_client.ping()
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Redis connection error: {e}")
            raise
    return _redis_client


def create_tables():
    """Create all tables defined in SQLAlchemy models."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def drop_tables():
    """Drop all tables. Use with caution!"""
    if os.getenv("APP_ENV") == "production":
        logger.error("Refusing to drop tables in production environment")
        return
    
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise 