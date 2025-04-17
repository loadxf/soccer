#!/bin/bash
# Script to fix path type error in fix_permissions.py
# Error: TypeError: unsupported operand type(s) for /: 'str' and 'str'

set -e  # Exit on error

echo "Soccer Prediction System - Path Error Fix"
echo "========================================"

# Create a backup of the original file
echo "Creating backup of fix_permissions.py..."
cp scripts/fix_permissions.py scripts/fix_permissions.py.bak

# Update the fix_permissions.py file to handle string paths correctly
echo "Updating fix_permissions.py to fix path type error..."
cat > scripts/fix_permissions.py << 'EOF'
#!/usr/bin/env python3
"""
Fix data directories and permissions for the Soccer Prediction System.
This script ensures all required data directories exist and are writable.
"""

import os
import sys
import shutil
from pathlib import Path
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix-permissions")
logger.info("Starting permission and directory fix")

# Get the project root directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Import DATA_DIR from config
try:
    from config.default_config import DATA_DIR
    # Convert DATA_DIR to Path object if it's a string
    if isinstance(DATA_DIR, str):
        DATA_DIR = Path(DATA_DIR)
        logger.info(f"Converted DATA_DIR string to Path: {DATA_DIR}")
except ImportError:
    # Fallback definition if import fails
    DATA_DIR = project_root / "data"
    logger.info(f"Using fallback DATA_DIR: {DATA_DIR}")

# Define data directories that need to exist
data_dirs = [
    "raw",
    "raw/football_data",
    "processed",
    "interim",
    "external",
    "kaggle_imports",
    "uploads",
    "fixtures",
    "models",
    "training",
    "predictions",
    "explainability",
    "features",
    "augmented"
]

def ensure_dirs_exist():
    """Create all required directories."""
    for dir_path in data_dirs:
        # Handle path joining safely whether DATA_DIR is a string or Path
        if isinstance(DATA_DIR, Path):
            full_path = DATA_DIR / dir_path
        else:
            full_path = os.path.join(DATA_DIR, dir_path)
            
        try:
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created/verified directory: {full_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {full_path}: {str(e)}")

def fix_permissions():
    """Ensure data directories have correct permissions."""
    logger.info("Fixing directory permissions...")
    try:
        # Give full permissions to all directories (only for development containers)
        if os.path.exists(DATA_DIR):
            os.chmod(DATA_DIR, 0o777)
            logger.info(f"Set permissions for {DATA_DIR}")
            
        # Walk through all directories and set permissions
        for root, dirs, files in os.walk(DATA_DIR):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o777)
                except Exception as e:
                    logger.warning(f"Failed to set permissions for {os.path.join(root, d)}: {e}")
    except Exception as e:
        logger.error(f"Error fixing permissions: {e}")
        return False
    return True

def main():
    """Main function to run all maintenance tasks."""
    logger.info("Ensuring data directories exist...")
    ensure_dirs_exist()
    
    logger.info("Fixing permissions...")
    fix_permissions()

    logger.info("Directory and permission fixes complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF

# Update docker-compose.override.yml to use the fixed script
echo "Updating docker-compose.override.yml to use the fixed script..."
cat > docker-compose.override.yml << 'EOF'
version: '3'

services:
  # Override the database configuration
  db:
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=soccer_prediction
    command: 
      - "postgres"
      - "-c"
      - "max_connections=100"
      - "-c"
      - "shared_buffers=256MB"
    restart: always

  # Override the app configuration to fix permission issues
  app:
    # Use root user to ensure directory creation works
    user: "0:0"
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=soccer_prediction
      - DB_USER=postgres
      - DB_PASSWORD=postgres
    volumes:
      - ./src:/app/src
      - ./config:/app/config
      - ./scripts:/app/scripts
      - ./data:/app/data
      - app_logs:/app/logs
      - uploads_data:/app/uploads
      - ./model_cache:/app/model_cache
    command: >
      bash -c "mkdir -p /app/model_cache && chmod 777 /app/model_cache &&
               python scripts/fix_permissions.py &&
               uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload"
    restart: on-failure
    depends_on:
      db:
        condition: service_healthy
        
  # Fix frontend port conflict by using port 3001
  frontend:
    ports:
      - "3001:80"  # Change from 3000 to 3001
    restart: on-failure
    depends_on:
      app:
        condition: service_started

volumes:
  app_logs:
  uploads_data:
EOF

echo "Stopping containers to apply fix..."
docker compose down

echo "Starting containers with fixed configuration..."
docker compose up -d

echo -e "\nFixes have been applied!"
echo "The Python path error in fix_permissions.py has been fixed"
echo "The containers should now start correctly"

echo -e "\nTo check the app logs:"
echo "docker compose logs app"
echo -e "\nTo check the frontend logs:"
echo "docker compose logs frontend" 