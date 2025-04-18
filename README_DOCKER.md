# Soccer Prediction System - Docker Deployment Guide

This guide explains how to deploy the entire Soccer Prediction System using Docker.

## Prerequisites

- Docker installed on your server
- Docker Compose installed
- Git repository cloned

## Quick Start

The easiest way to deploy the entire stack is:

```bash
# If you're on Linux/Mac
chmod +x deploy-docker.sh
./deploy-docker.sh

# If you're on Windows
docker-compose up -d
```

This will start all services:
- API backend on port 8000
- React frontend on port 3000
- Streamlit UI on port 8501
- PostgreSQL database on port 5432
- Monitoring services (if configured)

## Manual Deployment Steps

If you prefer to deploy manually or need more control:

1. Set up environment:
   ```bash
   # Create .env file if not exists
   cp .env.example .env
   ```

2. Build the frontend:
   ```bash
   cd src/frontend
   
   # If you're on Linux/Mac
   chmod +x docker-build.sh
   ./docker-build.sh
   
   # If you're on Windows, manually ensure the environment is set
   echo "REACT_APP_API_URL=http://app:8000" > .env
   echo "REACT_APP_ENVIRONMENT=docker" >> .env
   echo "REACT_APP_ENABLE_SERVICE_WORKER=true" >> .env
   
   cd ../..
   ```

3. Start the services:
   ```bash
   docker-compose up -d
   ```

## Accessing the Services

After deployment, services are available at:
- Frontend: http://YOUR_SERVER_IP:3000
- API: http://YOUR_SERVER_IP:8000
- UI: http://YOUR_SERVER_IP:8501

## Troubleshooting

### API Connection Issues

If the frontend shows "API Offline":

1. Check if API container is running:
   ```bash
   docker-compose ps app
   ```

2. Check API logs:
   ```bash
   docker-compose logs app
   ```

3. Test API directly from your server:
   ```bash
   curl http://localhost:8000/health
   ```

4. Test network connectivity between containers:
   ```bash
   docker-compose exec frontend wget -q --spider http://app:8000/health || echo "Connection failed"
   ```

### Container Management

```bash
# View all logs
docker-compose logs

# View logs for a specific service
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f

# Restart a service
docker-compose restart frontend

# Rebuild a service after changes
docker-compose up -d --build frontend

# Stop all services
docker-compose down

# Stop all services and remove volumes
docker-compose down -v
```

## Docker Troubleshooting

### Fix Networking Issues

If containers can't communicate with each other, run the network fix script:

```bash
sudo ./fix-docker-network.sh
```

Then restart the Docker containers:

```bash
docker compose down
docker compose up -d
```

### Fix Random Device Errors

If you encounter errors related to "random_device could not be read" or entropy-related issues, run:

```bash
sudo ./fix-docker-random-device.sh
```

### Connection Retry Logic

The UI container has been updated to wait for the API service to become available. If the API service isn't ready within 60 seconds, the UI will start in offline/fallback mode.

## Production Considerations

For production deployment:

1. Set proper environment variables in `.env`
2. Set `DEBUG=False` for production
3. Use a proper secret key
4. Consider using a reverse proxy like Nginx for SSL termination
5. Set up a proper database backup strategy
6. Configure monitoring alerts

## Updating the Application

To update the application:

1. Pull the latest code:
   ```bash
   git pull
   ```

2. Rebuild services:
   ```bash
   docker-compose up -d --build
   ```

## Additional Resources

For more detailed information about the frontend Docker configuration, see [src/frontend/DOCKER_README.md](src/frontend/DOCKER_README.md). 