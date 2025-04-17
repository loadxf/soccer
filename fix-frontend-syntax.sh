#!/bin/bash
# Script to fix syntax errors in frontend files

echo "Fixing frontend syntax errors..."

# Fix duplicate SportsIcon in Layout.js
echo "Fixing Layout.js - removing duplicate SportsIcon import..."
sed -i '/import SportsIcon from.*$/d' src/frontend/src/components/Layout.js

# Fix missing initializer in serviceWorkerRegistration.js
echo "Fixing serviceWorkerRegistration.js - fixing const declaration..."
sed -i 's/const is127\.0\.0\.1 = Boolean/const isLocalhost = Boolean/' src/frontend/src/serviceWorkerRegistration.js
sed -i 's/if (is127\.0\.0\.1)/if (isLocalhost)/' src/frontend/src/serviceWorkerRegistration.js

echo "Applying changes..."
docker compose down
docker compose up -d --build frontend

echo -e "\nSyntax errors have been fixed!"
echo "Check logs with:"
echo "docker compose logs frontend" 