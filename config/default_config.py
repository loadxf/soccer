"""
Default configuration for the Soccer Prediction System.
This file contains default settings that can be overridden by environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Application settings
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() in ["true", "1", "t", "yes", "y"]
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key_for_development_only")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "127.0.0.1")

# Database configuration
DB_TYPE = os.getenv("DB_TYPE", "postgres")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "soccer_prediction")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/soccer_prediction")

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Directory paths
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "uploads"))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", BASE_DIR / "model_cache"))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# API keys
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")
ESPN_API_KEY = os.getenv("ESPN_API_KEY", "")
STATSBOMB_API_KEY = os.getenv("STATSBOMB_API_KEY", "")

# Model settings
DEFAULT_MODEL_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "latest")

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", os.path.join(BASE_DIR, "app.log"))

# Frontend URL
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://127.0.0.1:3000")

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.example.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER", "your_email@example.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your_email_password")
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "True").lower() in ["true", "1", "t", "yes", "y"]

# API settings
API_PREFIX = "/api/v1"
ALLOW_CORS_ORIGINS = [
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3000",
]

# Feature flags
ENABLE_REAL_TIME_PREDICTIONS = os.getenv("ENABLE_REAL_TIME_PREDICTIONS", "True").lower() in ["true", "1", "t", "yes", "y"]
ENABLE_BETTING_STRATEGIES = os.getenv("ENABLE_BETTING_STRATEGIES", "True").lower() in ["true", "1", "t", "yes", "y"]

# ML model parameters
DEFAULT_MODEL_PARAMS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "neural_network": {
        "hidden_layers": [64, 32],
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
    },
    "bayesian": {
        "mcmc_samples": 1000,
        "tune": 500,
        "chains": 4,
    }
}

# Metrics and Monitoring Configuration
METRICS_ENABLED = True
COLLECT_DEFAULT_METRICS = True
PUSH_GATEWAY_URL = os.getenv("PUSH_GATEWAY_URL", "")
PUSH_GATEWAY_JOB = os.getenv("PUSH_GATEWAY_JOB", "soccer_prediction_api")
METRICS_EXPORT_INTERVAL = int(os.getenv("METRICS_EXPORT_INTERVAL", "15"))  # seconds
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))  # seconds
ENABLE_PROMETHEUS_ENDPOINT = os.getenv("ENABLE_PROMETHEUS_ENDPOINT", "true").lower() == "true"
SLOW_REQUEST_THRESHOLD = float(os.getenv("SLOW_REQUEST_THRESHOLD", "1.0"))  # seconds
LOG_SLOW_REQUESTS = os.getenv("LOG_SLOW_REQUESTS", "true").lower() == "true"

# Monitoring external services
MONITOR_EXTERNAL_SERVICES = os.getenv("MONITOR_EXTERNAL_SERVICES", "true").lower() == "true"
EXTERNAL_SERVICE_TIMEOUT = float(os.getenv("EXTERNAL_SERVICE_TIMEOUT", "5.0"))  # seconds

# Email alerts
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "alerts@example.com")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "admin@example.com")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.example.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true" 