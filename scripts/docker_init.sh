#!/bin/bash
# Docker container initialization script
# This script runs when the container starts and ensures proper permissions

set -e  # Exit on error

echo "Initializing Soccer Prediction System container..."

# Run the permission fix script
python scripts/fix_permissions.py

# Create necessary directories if they don't exist
mkdir -p /app/data
mkdir -p /app/uploads
mkdir -p /app/logs
mkdir -p /app/model_cache

# Ensure correct permissions for directories
chmod 777 /app/data
chmod 777 /app/uploads
chmod 777 /app/logs
chmod 777 /app/model_cache

echo "Container initialization complete."

# Execute whatever command was passed to the container
exec "$@" 