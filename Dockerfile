FROM python:3.9-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only the necessary project files
COPY config/ /app/config/
COPY src/ /app/src/
COPY main.py /app/
COPY scripts/ /app/scripts/

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features /app/data/models /app/data/evaluation /app/data/predictions /app/logs

# Copy the init script and make it executable
COPY scripts/docker_init.sh /app/docker_init.sh
RUN chmod +x /app/docker_init.sh

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# Run the application
EXPOSE 8000
EXPOSE 9091
ENTRYPOINT ["/app/docker_init.sh"]
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"] 