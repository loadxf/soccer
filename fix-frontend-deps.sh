#!/bin/bash
# Script to fix all frontend dependencies

echo "Fixing all frontend dependencies..."

# Stop containers
echo "Stopping containers..."
docker compose down

# Install all required dependencies
echo "Installing missing dependencies..."
cd src/frontend
npm install --save date-fns react-helmet
cd ../..

# Rebuild and restart all services
echo "Rebuilding and restarting all services..."
docker compose up -d --build

echo -e "\nAll services should now be running!"
echo "Check logs with:"
echo "docker compose logs" 