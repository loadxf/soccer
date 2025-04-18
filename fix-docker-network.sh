#!/bin/bash
#
# Fix Docker Networking Issues Between Containers
#

set -e
echo "=== Docker Network Fix Script ==="
echo

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run with sudo or as root"
  exit 1
fi

# Stop all running containers
echo "Stopping all running containers..."
docker stop $(docker ps -q) || true

# Remove the network
echo "Removing the existing soccer-net network..."
docker network rm soccer-net || true

# Recreate network with specified subnet
echo "Creating new soccer-net network..."
docker network create --driver bridge --subnet=172.28.0.0/16 --gateway=172.28.0.1 soccer-net

# Pruning unused networks
echo "Pruning unused networks..."
docker network prune -f

# Check network configuration
echo "Network configuration:"
docker network inspect soccer-net

echo "Done! Now try restarting your Docker Compose stack with:"
echo "docker compose up -d"
echo
echo "If issues persist, try running the following command:"
echo "docker compose down && docker compose up --build" 