#!/bin/bash
# Quick fix for PostgreSQL authentication error

# Stop containers
echo "Stopping containers..."
docker compose down

# Create override file if it doesn't exist
if [ ! -f docker-compose.override.yml ]; then
  echo "Creating docker-compose.override.yml..."
  cat > docker-compose.override.yml << 'EOF'
version: '3'

services:
  db:
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=soccer_prediction
    restart: always

  app:
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=soccer_prediction
      - DB_USER=postgres
      - DB_PASSWORD=postgres
    restart: on-failure
EOF
fi

# Start the database container first
echo "Starting database container..."
docker compose up -d db

# Wait for database to initialize
echo "Waiting for database to initialize (15 seconds)..."
sleep 15

# Start the rest of the services
echo "Starting remaining services..."
docker compose up -d

echo -e "\nFix applied! Check logs with:"
echo "docker compose logs app" 