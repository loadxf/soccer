#!/bin/bash
# Quick fix script for Docker deployment issues

set -e

echo "Soccer Prediction System - Docker Quick Fix"
echo "=========================================="

# Fix for package-lock.json issue
echo "Ensuring frontend package-lock.json exists..."
if [ ! -f ./src/frontend/package-lock.json ]; then
  echo "{}" > ./src/frontend/package-lock.json
  echo "Created empty package-lock.json"
fi

# Ensure frontend has a build directory
echo "Ensuring frontend build directory exists..."
mkdir -p ./src/frontend/build

# Ensure Nginx config file exists
echo "Checking for Nginx configuration..."
if [ ! -f ./src/frontend/nginx.conf ]; then
  echo "Creating Nginx configuration file..."
  cat > ./src/frontend/nginx.conf << 'EOF'
server {
    listen 80;
    
    # Document root where nginx serves files from
    root /usr/share/nginx/html;
    
    # Compression for better performance
    gzip on;
    gzip_comp_level 6;
    gzip_min_length 256;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript application/vnd.ms-fontobject application/x-font-ttf font/opentype image/svg+xml image/x-icon;
    
    location / {
        # First attempt to serve request as file, then as directory, 
        # then fall back to redirecting to index.html
        try_files $uri $uri/ /index.html;
        
        # Cache control for static assets
        add_header Cache-Control "public, max-age=3600";
    }

    # Don't cache index.html to ensure the latest version is served
    location = /index.html {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        expires 0;
    }
    
    # Service worker requires special cache control
    location = /serviceWorker.js {
        add_header Cache-Control "no-cache, no-store, must-revalidate";
        expires 0;
    }

    # Static file caching for improved performance
    location ~* \.(?:jpg|jpeg|gif|png|ico|svg|woff|woff2|ttf|css|js|json)$ {
        expires 30d;
        add_header Cache-Control "public, max-age=2592000";
        access_log off;
    }
    
    # Proxy API requests to the API container
    location /api/ {
        proxy_pass http://app:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Proxy health check requests to the API container
    location /health {
        proxy_pass http://app:8000/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF
fi

# Ensure .env file exists
echo "Checking .env file..."
if [ ! -f ./.env ]; then
  echo "Creating .env file..."
  cp .env.example .env 2>/dev/null || echo "No .env.example found, creating minimal .env"
  
  if [ ! -f ./.env ]; then
    cat > ./.env << EOF
# Soccer Prediction System - Environment Variables
APP_ENV=development
DEBUG=True
SECRET_KEY=dev_secret_key_for_local_testing
PORT=8000
HOST=0.0.0.0
DB_TYPE=postgres
DB_HOST=db
DB_PORT=5432
DB_NAME=soccer_prediction
DB_USER=postgres
DB_PASSWORD=postgres
REDIS_HOST=redis
REDIS_PORT=6379
EOF
  fi
fi

# Ensure frontend .env file exists
echo "Checking frontend .env file..."
if [ ! -f ./src/frontend/.env ]; then
  cat > ./src/frontend/.env << EOF
# Soccer Prediction System Frontend Environment Variables
REACT_APP_API_URL=http://app:8000
REACT_APP_ENVIRONMENT=docker
REACT_APP_ENABLE_SERVICE_WORKER=true
EOF
fi

echo "Fix complete! Now you can run:"
echo "docker-compose up -d --build"
echo ""
echo "To check logs:"
echo "docker-compose logs -f frontend" 