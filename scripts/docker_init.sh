#!/bin/bash
# Docker container initialization script
# This script runs when the container starts and ensures proper permissions

set -e  # Exit on error

echo "Initializing Soccer Prediction System container..."

# Create necessary directories if they don't exist
mkdir -p /app/data
mkdir -p /app/uploads
mkdir -p /app/logs
mkdir -p /app/model_cache

# Ensure correct permissions for directories
chmod -R 777 /app/data
chmod -R 777 /app/uploads
chmod -R 777 /app/logs
chmod -R 777 /app/model_cache

# Check if database initialization is needed and postgres client is available
if [ ! -z "$DB_HOST" ] && command -v psql &> /dev/null; then
    # Initialize database if init-db.sh exists
    if [ -f /app/scripts/init-db.sh ]; then
        echo "Running database initialization script..."
        chmod +x /app/scripts/init-db.sh
        bash /app/scripts/init-db.sh
    else
        echo "Database initialization script not found, skipping..."
    fi
fi

echo "Container initialization complete."

# Execute whatever command was passed to the container
exec "$@" 