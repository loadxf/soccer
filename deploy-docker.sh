#!/bin/bash
# Comprehensive Docker deployment script for Soccer Prediction System

# Print commands and exit on errors
set -ex

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Soccer Prediction System - Docker Deployment${NC}"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Build the frontend
echo -e "${YELLOW}Building frontend...${NC}"
cd src/frontend
./docker-build.sh
cd ../..

# Make sure we have an .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env.example .env || echo "Failed to copy .env.example. Creating a minimal .env file."
    
    # If copy fails, create a minimal .env file
    if [ ! -f .env ]; then
        cat > .env << EOF
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

# Build and start all services
echo -e "${YELLOW}Starting all services with Docker Compose...${NC}"
docker-compose up -d --build

# Wait for services to start
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Check services health
echo -e "${YELLOW}Checking service health...${NC}"
docker-compose ps

# Check API health
echo -e "${YELLOW}Checking API health...${NC}"
API_HEALTH=$(docker-compose exec -T app wget -q -O - http://localhost:8000/health 2>/dev/null || echo "Failed")
if [[ $API_HEALTH == *"status"*"healthy"* || $API_HEALTH == *"status"*"ok"* ]]; then
    echo -e "${GREEN}API is healthy!${NC}"
else
    echo -e "${RED}API may not be healthy. Please check logs:${NC}"
    echo "docker-compose logs app"
fi

# Provide URLs
echo -e "\n${GREEN}Deployment complete!${NC}"
echo -e "${YELLOW}Services are available at:${NC}"
echo -e "Frontend: ${GREEN}http://localhost:3000${NC}"
echo -e "API:      ${GREEN}http://localhost:8000${NC}"
echo -e "UI:       ${GREEN}http://localhost:8501${NC}"
echo -e "Database: ${GREEN}localhost:5432${NC}"
echo -e "\nTo view logs:        ${GREEN}docker-compose logs -f${NC}"
echo -e "To stop services:    ${GREEN}docker-compose down${NC}"
echo -e "To rebuild a service: ${GREEN}docker-compose up -d --build <service>${NC}"

exit 0 