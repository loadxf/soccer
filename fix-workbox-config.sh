#!/bin/bash
# Script to fix workbox configuration issues

set -e

echo "Fixing workbox configuration..."

# Backup the original file
cp src/frontend/workbox-config.js src/frontend/workbox-config.js.bak

# Fix the property name in workbox-config.js
sed -i 's/ignoreURLParametersMatching:/dontCacheBustURLsMatching:/g' src/frontend/workbox-config.js

echo "Workbox configuration fixed!"
echo 
echo "To rebuild the frontend container, run:"
echo "docker compose up -d --build frontend" 