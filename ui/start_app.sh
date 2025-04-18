#!/bin/bash
#
# Simple UI container startup script that avoids TensorFlow verification
#

echo "=== Starting Soccer Prediction UI ==="

# Initialize environment
export PYTHONPATH=/app:/app/src
export PYTHONHASHSEED=0
export RANDOM_SEED=42
export TF_DETERMINISTIC_OPS=1
export TF_CPP_MIN_LOG_LEVEL=2

# Create random devices if needed (for Docker)
if [ ! -c /dev/random ] && [ -c /dev/urandom ]; then
  echo "Creating /dev/random symlink to /dev/urandom"
  ln -sf /dev/urandom /dev/random
fi

# Start entropy services if available
if command -v haveged &> /dev/null; then
  haveged -F &
  echo "Started haveged entropy daemon"
fi

if command -v rngd &> /dev/null; then
  rngd -r /dev/urandom &
  echo "Started rngd entropy daemon"
fi

# Wait for API to become available (with timeout)
echo "Waiting for API service to become available..."
timeout=60
counter=0
api_ready=false

while [ $counter -lt $timeout ] && [ "$api_ready" = false ]; do
  if curl -s "http://app:8000/health" > /dev/null 2>&1; then
    echo "✅ API connection successful! Starting Streamlit..."
    api_ready=true
  elif curl -s "http://app:8000/" > /dev/null 2>&1; then
    echo "✅ API connection successful (base URL)! Starting Streamlit..."
    api_ready=true
  else
    counter=$((counter + 2))
    echo "⏳ Waiting for API service... ($counter/$timeout seconds)"
    sleep 2
  fi
done

if [ "$api_ready" = false ]; then
  echo "⚠️ WARNING: Could not connect to API after $timeout seconds. Continuing anyway in offline mode."
  # Setting fallback environment variable to signal app.py to use offline mode
  export USE_FALLBACK_DATA=true
fi

# Skip the TensorFlow verification and dependency check
echo "Skipping TensorFlow verification to avoid random_device errors"

# Start Streamlit directly
echo "Starting Streamlit application..."
exec streamlit run /app/ui/app.py --server.port=8501 --server.address=0.0.0.0 