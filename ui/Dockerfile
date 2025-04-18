FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app:/app/src \
    PYTHONHASHSEED=0 \
    RANDOM_SEED=42 \
    TF_DETERMINISTIC_OPS=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    rng-tools \
    haveged \
    && rm -rf /var/lib/apt/lists/*

# Configure haveged to provide additional entropy
RUN echo "HAVEGED_OPTS=\"-w 1024 -v 1\"" > /etc/default/haveged

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
# Start entropy services\n\
if command -v haveged &> /dev/null; then\n\
  haveged -F || echo "Unable to start haveged, continuing anyway"\n\
fi\n\
if command -v rngd &> /dev/null; then\n\
  rngd -r /dev/urandom || echo "Unable to start rngd, continuing anyway"\n\
fi\n\
# Create device file if not exists in container\n\
if [ ! -c /dev/random ]; then\n\
  echo "Creating /dev/random device"\n\
  mknod -m 666 /dev/random c 1 8 || echo "Unable to create /dev/random, continuing anyway"\n\
fi\n\
# Start the streamlit application\n\
streamlit run /app/ui/app.py --server.port=8501 --server.address=0.0.0.0' > /app/start.sh && \
chmod +x /app/start.sh

# Expose Streamlit port
EXPOSE 8501

# Run the application with our start script
CMD ["/app/start.sh"] 