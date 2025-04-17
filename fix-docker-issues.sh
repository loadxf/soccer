#!/bin/bash
# Script to fix Docker deployment issues
# 1. Port conflict - change frontend port from 3000 to 3001
# 2. Permission issues with model_cache directory

set -e  # Exit on error

echo "Soccer Prediction System - Docker Issues Fix"
echo "==========================================="

# Stop all containers
echo "Stopping all containers..."
docker compose down

# Create model_cache directory with proper permissions if it doesn't exist
echo "Creating model_cache directory with proper permissions..."
mkdir -p ./model_cache
chmod 777 ./model_cache

# Modify main docker-compose.yml to change the frontend port
echo "Updating port in docker-compose.yml..."
if grep -q "3000:80" docker-compose.yml; then
  # Replace port 3000 with 3001 in the main docker-compose file
  sed -i 's/3000:80/3001:80/g' docker-compose.yml
  echo "Changed port from 3000 to 3001 in docker-compose.yml"
fi

# Update docker-compose.override.yml
echo "Updating docker-compose.override.yml..."
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

# Also update .env file to ensure consistent model_cache_dir setting
echo "Updating .env file with model_cache configuration..."
if [ -f .env ]; then
  # Check if MODEL_CACHE_DIR is already set
  if grep -q "MODEL_CACHE_DIR" .env; then
    # Replace existing MODEL_CACHE_DIR line
    sed -i 's|MODEL_CACHE_DIR=.*|MODEL_CACHE_DIR=/app/model_cache|g' .env
  else
    # Add MODEL_CACHE_DIR line if it doesn't exist
    echo "MODEL_CACHE_DIR=/app/model_cache" >> .env
  fi
else
  # Create .env file if it doesn't exist
  cat > .env << EOF
# Soccer Prediction System - Environment Variables
APP_ENV=development
DEBUG=True
SECRET_KEY=dev_secret_key_for_local_testing
PORT=8000
HOST=0.0.0.0

# Database Configuration
DB_TYPE=postgres
DB_HOST=db
DB_PORT=5432
DB_NAME=soccer_prediction
DB_USER=postgres
DB_PASSWORD=postgres

# Directory Configuration
MODEL_CACHE_DIR=/app/model_cache
DATA_DIR=/app/data
UPLOAD_DIR=/app/uploads
LOG_LEVEL=INFO
EOF
fi

# Update frontend URL in other configuration files if needed
echo "Checking for other references to port 3000..."
if [ -f ./config/default_config.py ]; then
  # Update FRONTEND_URL in default_config.py if present
  if grep -q "FRONTEND_URL.*3000" ./config/default_config.py; then
    sed -i 's|http://127.0.0.1:3000|http://127.0.0.1:3001|g' ./config/default_config.py
    echo "Updated FRONTEND_URL in default_config.py"
  fi
fi

# Check for any CORS configurations that need updating
if [ -f ./config/default_config.py ]; then
  # Update CORS origins if they reference port 3000
  if grep -q "http://127.0.0.1:3000" ./config/default_config.py; then
    sed -i 's|"http://127.0.0.1:3000"|"http://127.0.0.1:3001"|g' ./config/default_config.py
    echo "Updated CORS origins in default_config.py"
  fi
fi

echo "Cleanup any existing containers that may prevent port allocation..."
docker compose rm -f frontend

echo "Starting containers with fixed configuration..."
docker compose up -d

echo -e "\nFixes have been applied!"
echo "The frontend is now available at http://localhost:3001"
echo "The model_cache directory permissions have been fixed"

echo -e "\nTo check the app logs:"
echo "docker compose logs app"
echo -e "\nTo check the frontend logs:"
echo "docker compose logs frontend" 