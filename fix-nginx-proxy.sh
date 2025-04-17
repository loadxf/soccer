#!/bin/bash
# Script to fix Nginx proxy configuration for API requests

echo "Fixing Nginx proxy configuration..."

# Create an updated nginx.conf file
cat > src/frontend/nginx.conf << 'EOF'
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
    
    # Proxy API requests to the API container - improved configuration
    location /api/ {
        # Use DNS name instead of container name to work around DNS resolution issues
        proxy_pass http://app:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;
    }
    
    # Proxy health check requests to the API container
    location /health {
        proxy_pass http://app:8000/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
        proxy_send_timeout 60s;
    }
}
EOF

echo "Updating Docker Compose configuration to ensure services can communicate..."
cat > docker-compose.override.yml << 'EOF'
version: '3'

services:
  # Override the database configuration
  db:
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=soccer_prediction
    restart: always

  # Override the app configuration to ensure it uses the correct DB credentials
  app:
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=soccer_prediction
      - DB_USER=postgres
      - DB_PASSWORD=postgres
    restart: on-failure
    # Expose internal port for health checks
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
        
  # Add dependency to ensure the frontend waits for the API
  frontend:
    restart: on-failure
    depends_on:
      app:
        condition: service_started
EOF

echo "Rebuilding containers..."
docker compose down
docker compose up -d

echo -e "\nNginx proxy configuration has been fixed!"
echo "Please refresh your browser and try logging in again."
echo "The app should now be able to properly communicate with the API through the Nginx proxy."

echo -e "\nTo check the frontend logs:"
echo "docker compose logs frontend"
echo -e "\nTo check the API logs:"
echo "docker compose logs app" 