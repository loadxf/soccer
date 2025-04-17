#!/bin/bash
# Script to fix frontend config directory issue

echo "Fixing frontend config directory issue..."

# Stop containers
echo "Stopping containers..."
docker compose down

# Create frontend config directory if it doesn't exist
echo "Ensuring config directory exists..."
mkdir -p src/frontend/src/config

# Create config file
echo "Creating config file..."
cat > src/frontend/src/config/index.js << 'EOF'
// Frontend configuration

// API configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const ENVIRONMENT = process.env.REACT_APP_ENVIRONMENT || 'development';
const ENABLE_SERVICE_WORKER = process.env.REACT_APP_ENABLE_SERVICE_WORKER === 'true';

// App settings
const APP_CONFIG = {
  apiUrl: API_URL,
  environment: ENVIRONMENT,
  enableServiceWorker: ENABLE_SERVICE_WORKER,
  appName: 'Soccer Prediction System',
  version: '1.0.0',
};

export default APP_CONFIG;
EOF

# Rebuild and restart frontend
echo "Rebuilding and restarting frontend..."
docker compose up -d --build frontend

echo -e "\nFrontend config has been fixed!"
echo "Check logs with:"
echo "docker compose logs frontend" 