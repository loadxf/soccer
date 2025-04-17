#!/bin/bash
# Script to fix Docker networking issues

echo "=== Docker Network Fix Script ==="
echo ""

# Stop Docker
echo "1. Stopping Docker service..."
sudo systemctl stop docker

# Clean up network interfaces
echo "2. Cleaning up network interfaces..."
sudo ip link set dev docker0 down 2>/dev/null
sudo ip link delete docker0 2>/dev/null
sudo ip link set dev bridge0 down 2>/dev/null
sudo ip link delete bridge0 2>/dev/null

# Clean up iptables rules
echo "3. Cleaning up iptables rules..."
sudo iptables -t nat -F
sudo iptables -t filter -F FORWARD

# Apply custom Docker daemon configuration
if [ -f "docker-daemon-config.json" ]; then
  echo "4. Applying custom Docker daemon configuration..."
  sudo mkdir -p /etc/docker
  sudo cp docker-daemon-config.json /etc/docker/daemon.json
  echo "✅ Docker daemon configuration applied"
else
  echo "❌ docker-daemon-config.json not found in current directory"
fi

# Restart Docker
echo "5. Restarting Docker service..."
sudo systemctl start docker
sleep 5
sudo systemctl status docker | grep Active

# Verify Docker network
echo "6. Verifying Docker networks..."
docker network ls

echo ""
echo "=== Clean up Docker containers and networks ==="
echo "1. Stopping all containers..."
docker compose down

echo "2. Pruning networks..."
docker network prune -f

echo "3. Removing any leftover containers..."
docker rm -f $(docker ps -aq) 2>/dev/null || echo "No containers to remove"

echo ""
echo "Now try starting your containers again with:"
echo "docker compose up -d"

exit 0 