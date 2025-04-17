#!/bin/bash
# Script to fix PostgreSQL authentication issues

set -e

echo "Soccer Prediction System - Database Authentication Fix"
echo "===================================================="

# Stop all containers
echo "Stopping all containers..."
docker compose down

# Remove the database volume to clean state
echo "Removing database volume for clean start..."
docker volume rm soccer_postgres_data || docker volume rm $(docker volume ls -q | grep postgres_data) || echo "Volume does not exist or cannot be removed"

# Create a fresh .env file with correct database credentials
echo "Creating consistent .env file with proper DB credentials..."
cat > .env << EOF
# Soccer Prediction System - Environment Variables
APP_ENV=development
DEBUG=True
SECRET_KEY=dev_secret_key_for_local_testing
PORT=8000
HOST=0.0.0.0

# Database Configuration - MUST match docker-compose.yml
DB_TYPE=postgres
DB_HOST=db
DB_PORT=5432
DB_NAME=soccer_prediction
DB_USER=postgres
DB_PASSWORD=postgres
REDIS_HOST=redis
REDIS_PORT=6379

# Other settings
MODEL_CACHE_DIR=/app/model_cache
DATA_DIR=/app/data
UPLOAD_DIR=/app/uploads
LOG_LEVEL=INFO
EOF

echo "Starting database ONLY first to ensure proper initialization..."
docker compose up -d db

echo "Waiting 10 seconds for database to initialize..."
sleep 10

echo "Checking database connection..."
docker compose exec db pg_isready -U postgres || echo "Database not ready yet"

echo "Setting up database user permissions..."
docker compose exec db psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'postgres';" || echo "Could not alter user password"

echo "Starting remaining services..."
docker compose up -d

echo "Fix complete!"
echo ""
echo "If you still have issues, try:"
echo "1. docker compose down -v (this removes ALL data)"
echo "2. ./fix-db-auth.sh"

echo ""
echo "To view database logs:"
echo "docker compose logs db" 