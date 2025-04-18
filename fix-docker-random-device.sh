#!/bin/bash
#
# Fix Docker Random Device Issues
# This script helps resolve the "random_device could not be read" error in Docker containers
#

set -e
echo "=== Docker Random Device Fix Script ==="
echo

# Check if the script is being run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

echo "1. Installing entropy services..."
apt-get update
apt-get install -y rng-tools haveged

echo "2. Configuring and starting services..."
systemctl enable haveged
systemctl start haveged
systemctl enable rng-tools
systemctl start rng-tools

echo "3. Checking entropy pool..."
cat /proc/sys/kernel/random/entropy_avail
echo "Available entropy should be at least 1000 for good operation"

echo "4. Setting secure permissions for /dev/urandom and /dev/random..."
chmod 666 /dev/urandom
chmod 666 /dev/random

echo "5. Adding Docker volumes in docker-compose.yml..."
echo "Make sure your docker-compose.yml contains this volume for the UI container:"
echo "- /dev/urandom:/dev/random:ro"

echo "6. Setting random-related environment variables..."
echo "Add these environment variables to your Docker containers:"
echo "- PYTHONHASHSEED=0"
echo "- RANDOM_SEED=42"
echo "- TF_DETERMINISTIC_OPS=1"

echo -e "\nSetup complete! Try rebuilding and restarting your containers:"
echo "docker compose build ui"
echo "docker compose up ui"

echo -e "\nIf issues persist, try commenting out the TensorFlow verification in docker-compose.yml command." 