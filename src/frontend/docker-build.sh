#!/bin/bash
# Docker-specific build script for the frontend

# Print commands and exit on errors
set -ex

echo "Building frontend for Docker deployment..."

# Ensure we have the .env file with Docker configuration
if [ ! -f .env ]; then
  echo "Creating Docker environment configuration..."
  cat > .env << EOF
# Soccer Prediction System Frontend Environment Variables
REACT_APP_API_URL=http://app:8000
REACT_APP_ENVIRONMENT=docker
REACT_APP_ENABLE_SERVICE_WORKER=true
EOF
else
  echo "Using existing .env file..."
fi

# Create package-lock.json if missing (allows npm ci to work)
if [ ! -f package-lock.json ]; then
  echo "package-lock.json not found, creating empty one for Docker build..."
  echo "{}" > package-lock.json
fi

# Install dependencies
echo "Installing dependencies..."
npm install

# Build the app with Docker-specific configuration
echo "Building React app for production..."
npm run build

# Inform success
echo "Frontend build completed successfully!"
echo "To run in Docker: docker-compose up -d" 