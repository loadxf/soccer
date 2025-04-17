# Soccer Prediction System - Docker Deployment Guide

This guide explains how to deploy the Soccer Prediction System frontend in a Docker environment.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- Git repository cloned

## Quick Deployment

The simplest way to deploy the entire system is using Docker Compose:

```bash
# Deploy all services including the frontend
docker-compose up -d
```

This will start all services including:
- API Backend (`app`)
- React Frontend (`frontend`)
- Database (`db`)
- Streamlit UI (`ui`)
- Monitoring tools (if enabled)

## Frontend Container Details

The frontend is served using Nginx and runs on port 80 in the container, which is mapped to port 3000 on the host by default. You can access it at:

```
http://localhost:3000
```

### Environment Variables

The frontend Docker container uses these environment variables:

- `REACT_APP_API_URL`: URL of the API service (default: `http://app:8000`)
- `REACT_APP_ENVIRONMENT`: Set to `docker` for Docker-specific configuration
- `REACT_APP_ENABLE_SERVICE_WORKER`: Enable/disable service worker (default: `true`)

## Troubleshooting

### API Connection Issues

If you see "API Offline" in the frontend, check:

1. Make sure the API service is running:
   ```bash
   docker-compose ps app
   ```

2. Check API logs for errors:
   ```bash
   docker-compose logs app
   ```

3. Verify networking between containers:
   ```bash
   # Enter the frontend container
   docker-compose exec frontend sh
   
   # Try to reach the API
   wget -q --spider http://app:8000/health || echo "Connection failed"
   ```

4. Check if the API is healthy:
   ```bash
   docker-compose exec app wget -q --spider http://localhost:8000/health || echo "API unhealthy"
   ```

### Frontend Container Rebuild

If you need to rebuild the frontend container:

```bash
# Rebuild and restart only the frontend
docker-compose up -d --build frontend
```

### Nginx Configuration

The frontend uses a custom Nginx configuration. If you need to modify the configuration:

1. Edit `src/frontend/nginx.conf`
2. Rebuild the frontend container:
   ```bash
   docker-compose up -d --build frontend
   ```

## Advanced Configuration

### API Proxying

To proxy API requests through the frontend container, uncomment the appropriate sections in `nginx.conf` and rebuild the container.

### Custom Build Process

To customize the build process:

1. Modify the `src/frontend/Dockerfile`
2. Update environment variables in `docker-compose.yml`
3. Rebuild with `docker-compose up -d --build frontend` 