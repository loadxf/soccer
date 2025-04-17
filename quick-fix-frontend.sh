#!/bin/bash
# Script to fix frontend build missing dependency

echo "Adding missing frontend dependencies..."

# Stop containers
echo "Stopping containers..."
docker compose down

# Install missing dependencies in frontend directory
echo "Installing date-fns dependency..."
cd src/frontend
npm install --save date-fns
cd ../..

# Rebuild and restart
echo "Rebuilding and restarting containers..."
docker compose up -d --build frontend

echo "Frontend fix applied! Check frontend logs with:"
echo "docker compose logs frontend" 