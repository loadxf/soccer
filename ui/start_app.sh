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

# Skip the TensorFlow verification and dependency check
echo "Skipping TensorFlow verification to avoid random_device errors"

# Start Streamlit directly
echo "Starting Streamlit application..."
exec streamlit run /app/ui/app.py --server.port=8501 --server.address=0.0.0.0 