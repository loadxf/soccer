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

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies (only core dependencies)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only the UI directory, not the entire project
COPY ui/ /app/ui/
COPY config/ /app/config/

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "/app/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 