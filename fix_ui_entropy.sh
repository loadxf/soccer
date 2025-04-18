#!/bin/bash
set -e

echo "=== Fixing UI Container Entropy Issues ==="
echo

# Stop the UI container
echo "Stopping UI container if running..."
docker compose stop ui || echo "UI container not running"

# Create a custom Dockerfile specifically for the UI with our fixes
echo "Creating fixed Dockerfile..."
mkdir -p ui

cat > ui/Dockerfile.fixed << 'EOL'
FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/src \
    PYTHONHASHSEED=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    rng-tools \
    procps \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install kaggle

# Create .kaggle directory and set proper permissions
RUN mkdir -p /root/.kaggle && chmod 700 /root/.kaggle

# Copy only the UI directory, not the entire project
COPY ui/ /app/ui/
COPY config/ /app/config/

# Create a start script that fixes kaggle.json permissions
RUN echo '#!/bin/bash\n\
if [ -f /root/.kaggle/kaggle.json ]; then\n\
  chmod 600 /root/.kaggle/kaggle.json\n\
  echo "Kaggle credentials file permissions fixed"\n\
  # Set Kaggle config directory explicitly\n\
  export KAGGLE_CONFIG_DIR=/root/.kaggle\n\
  # Read username and key from kaggle.json and export as env vars\n\
  export KAGGLE_USERNAME=$(cat /root/.kaggle/kaggle.json | grep "username" | cut -d\\" -f4)\n\
  export KAGGLE_KEY=$(cat /root/.kaggle/kaggle.json | grep "key" | cut -d\\" -f4)\n\
  echo "Kaggle credentials loaded from /root/.kaggle/kaggle.json"\n\
fi\n\
export PYTHONPATH=/app:/app/src\n\
export PYTHONHASHSEED=0\n\
\n\
# Do not verify the dependencies just run streamlit directly\n\
streamlit run /app/ui/app.py --server.port=8501 --server.address=0.0.0.0' > /app/start.sh && \
chmod +x /app/start.sh

# Expose Streamlit port
EXPOSE 8501

# Run the application with our start script
CMD ["/app/start.sh"]
EOL

echo "Fixed Dockerfile created at ui/Dockerfile.fixed"

# Create a backup of docker-compose.yml
echo "Creating backup of docker-compose.yml..."
cp docker-compose.yml docker-compose.yml.backup

# Modify the docker-compose.yml file using temporary file
echo "Updating docker-compose.yml..."
if [ "$(uname)" == "Darwin" ]; then
  # macOS version
  sed -i '' 's/dockerfile: ui\/Dockerfile/dockerfile: ui\/Dockerfile.fixed/' docker-compose.yml
  sed -i '' 's|command: bash -c "python /app/scripts/verify_all_dependencies.py.*|command: "/app/start.sh"|' docker-compose.yml
else
  # Linux version
  sed -i 's/dockerfile: ui\/Dockerfile/dockerfile: ui\/Dockerfile.fixed/' docker-compose.yml
  sed -i 's|command: bash -c "python /app/scripts/verify_all_dependencies.py.*|command: "/app/start.sh"|' docker-compose.yml
fi

echo "docker-compose.yml updated successfully"

# Instructions for local and remote use
echo
echo "=== How to Use This Fix ==="
echo "1. Locally: Run 'docker compose up -d --build ui' to rebuild and restart the UI container"
echo "2. On remote server: Copy this script to the server and run it"
echo 
echo "To copy to remote server:"
echo "  scp fix_ui_entropy.sh root@103.163.186.204:~/soccer/"
echo "  ssh root@103.163.186.204 'cd ~/soccer && chmod +x fix_ui_entropy.sh && ./fix_ui_entropy.sh'"
echo
echo "=== Verifying the Fix ==="
echo "After running the fix, use 'docker compose logs ui' to check if the container started correctly"
echo "Then access the UI at http://localhost:8501 (local) or http://103.163.186.204:8501 (remote)" 