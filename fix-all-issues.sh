#!/bin/bash
# Script to fix all identified issues in the Soccer Prediction System

set -e  # Exit on error

echo "====== Soccer Prediction System - Fix All Issues ======"
echo

# Function to show section header
section() {
  echo
  echo "===== $1 ====="
  echo
}

# Create backup directory
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

section "Creating backups of important files"
cp -f Dockerfile "$BACKUP_DIR/" 2>/dev/null || echo "No Dockerfile to backup"
cp -f docker-compose.yml "$BACKUP_DIR/" 2>/dev/null || echo "No docker-compose.yml to backup"
cp -f src/data/features.py "$BACKUP_DIR/" 2>/dev/null || echo "No features.py to backup"
echo "Backups saved to $BACKUP_DIR"

section "1. Adding missing functions to features.py"
# Check if features.py exists and fixed version is needed
if ! grep -q "load_feature_pipeline" src/data/features.py 2>/dev/null; then
  echo "Missing functions detected in features.py, fixing..."
  
  # Use the fix-missing-functions.sh if available
  if [ -f fix-missing-functions.sh ]; then
    chmod +x fix-missing-functions.sh
    ./fix-missing-functions.sh
  else
    echo "Manually adding missing functions to features.py..."
    
    # Ensure directory exists
    mkdir -p src/data
    
    # Only append missing functions if file exists
    if [ -f src/data/features.py ]; then
      echo -e "\n# Missing functions added by fix script\n" >> src/data/features.py
      
      cat >> src/data/features.py << 'EOF'

# Get data directories
try:
    from config.default_config import DATA_DIR
    # Convert DATA_DIR to Path object if it's a string
    if isinstance(DATA_DIR, str):
        DATA_DIR = Path(DATA_DIR)
except ImportError:
    # Fallback default if config is not available
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Define paths
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_processed_data(dataset_name: str = "matches", version: str = "latest") -> pd.DataFrame:
    """Load processed data from the processed directory."""
    logger.info(f"Loading processed data: {dataset_name}, version: {version}")
    
    if version == "latest":
        # Find the latest version
        files = list(PROCESSED_DIR.glob(f"{dataset_name}_*.csv"))
        if not files:
            logger.error(f"No processed data found for {dataset_name}")
            return pd.DataFrame()
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        file_path = files[0]
    else:
        file_path = PROCESSED_DIR / f"{dataset_name}_{version}.csv"
    
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return pd.DataFrame()
    
    return pd.read_csv(file_path)

def create_feature_datasets(processed_data: pd.DataFrame, features_config: Dict = None) -> Dict[str, pd.DataFrame]:
    """Create feature datasets from processed data according to configuration."""
    logger.info("Creating feature datasets")
    
    if features_config is None:
        features_config = {
            "basic": ["team_home", "team_away", "league", "season"],
            "match_stats": ["home_goals", "away_goals", "home_shots", "away_shots"],
            "form": ["home_form", "away_form", "home_wins_last5", "away_wins_last5"],
        }
    
    feature_datasets = {}
    
    # Create datasets for each feature group
    for group_name, columns in features_config.items():
        # Filter only columns that exist in the data
        valid_columns = [col for col in columns if col in processed_data.columns]
        
        if not valid_columns:
            logger.warning(f"No valid columns found for group {group_name}")
            continue
            
        # Create the dataset
        feature_datasets[group_name] = processed_data[valid_columns].copy()
        logger.info(f"Created feature group {group_name} with {len(valid_columns)} features")
        
    return feature_datasets

def load_feature_pipeline(pipeline_name: str, version: str = "latest") -> Any:
    """Load a feature transformation pipeline from disk."""
    logger.info(f"Loading feature pipeline: {pipeline_name}, version: {version}")
    
    if version == "latest":
        # Find the latest version
        files = list(FEATURES_DIR.glob(f"pipeline_{pipeline_name}_*.joblib"))
        if not files:
            logger.warning(f"No pipeline found for {pipeline_name}, returning None")
            return None
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        file_path = files[0]
    else:
        file_path = FEATURES_DIR / f"pipeline_{pipeline_name}_{version}.joblib"
    
    logger.info(f"Loading pipeline from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Pipeline file not found: {file_path}")
        return None
    
    try:
        import joblib
        return joblib.load(file_path)
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        return None

def apply_feature_pipeline(data: pd.DataFrame, pipeline_name: str, version: str = "latest") -> pd.DataFrame:
    """Apply a feature transformation pipeline to data."""
    logger.info(f"Applying feature pipeline {pipeline_name} to data of shape {data.shape}")
    
    pipeline = load_feature_pipeline(pipeline_name, version)
    if pipeline is None:
        logger.warning("Pipeline not found, returning original data")
        return data
    
    try:
        transformed_data = pipeline.transform(data)
        logger.info(f"Data transformed successfully, new shape: {transformed_data.shape}")
        return transformed_data
    except Exception as e:
        logger.error(f"Error applying pipeline: {e}")
        return data
EOF
    fi
  fi
  
  echo "Fixed features.py with missing functions"
else
  echo "features.py already contains required functions"
fi

section "2. Creating DB initialization script"
mkdir -p scripts
cat > scripts/init-db.sh << 'EOF'
#!/bin/bash
# Database initialization script for Soccer Prediction System

set -e

echo "=== Initializing PostgreSQL Database ==="

# Wait for PostgreSQL to start
echo "Waiting for PostgreSQL to start..."
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "$DB_HOST" -U "$POSTGRES_USER" -c '\q'; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 1
done

echo "PostgreSQL is up - checking for soccer_prediction database"

# Check if database exists
db_exists=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h "$DB_HOST" -U "$POSTGRES_USER" -tAc "SELECT 1 FROM pg_database WHERE datname='$POSTGRES_DB'")

if [ "$db_exists" != "1" ]; then
  echo "Creating database $POSTGRES_DB..."
  PGPASSWORD=$POSTGRES_PASSWORD psql -h "$DB_HOST" -U "$POSTGRES_USER" -c "CREATE DATABASE $POSTGRES_DB;"
else
  echo "Database $POSTGRES_DB already exists."
fi

echo "Database setup completed successfully!"
exit 0
EOF
chmod +x scripts/init-db.sh
echo "Created scripts/init-db.sh"

section "3. Updating docker_init.sh"
if [ -f scripts/docker_init.sh ]; then
  # Backup docker_init.sh
  cp scripts/docker_init.sh "$BACKUP_DIR/"
  
  # Update docker_init.sh to fix directories and run DB init
  cat > scripts/docker_init.sh << 'EOF'
#!/bin/bash
# Docker container initialization script
# This script runs when the container starts and ensures proper permissions

set -e  # Exit on error

echo "Initializing Soccer Prediction System container..."

# Create necessary directories if they don't exist
mkdir -p /app/data
mkdir -p /app/uploads
mkdir -p /app/logs
mkdir -p /app/model_cache

# Ensure correct permissions for directories
chmod -R 777 /app/data
chmod -R 777 /app/uploads
chmod -R 777 /app/logs
chmod -R 777 /app/model_cache

# Check if database initialization is needed and postgres client is available
if [ ! -z "$DB_HOST" ] && command -v psql &> /dev/null; then
    # Initialize database if init-db.sh exists
    if [ -f /app/scripts/init-db.sh ]; then
        echo "Running database initialization script..."
        chmod +x /app/scripts/init-db.sh
        bash /app/scripts/init-db.sh
    else
        echo "Database initialization script not found, skipping..."
    fi
fi

echo "Container initialization complete."

# Execute whatever command was passed to the container
exec "$@"
EOF
  chmod +x scripts/docker_init.sh
  echo "Updated scripts/docker_init.sh"
else
  echo "No scripts/docker_init.sh found, creating new file..."
  mkdir -p scripts
  cat > scripts/docker_init.sh << 'EOF'
#!/bin/bash
# Docker container initialization script
# This script runs when the container starts and ensures proper permissions

set -e  # Exit on error

echo "Initializing Soccer Prediction System container..."

# Create necessary directories if they don't exist
mkdir -p /app/data
mkdir -p /app/uploads
mkdir -p /app/logs
mkdir -p /app/model_cache

# Ensure correct permissions for directories
chmod -R 777 /app/data
chmod -R 777 /app/uploads
chmod -R 777 /app/logs
chmod -R 777 /app/model_cache

# Check if database initialization is needed and postgres client is available
if [ ! -z "$DB_HOST" ] && command -v psql &> /dev/null; then
    # Initialize database if init-db.sh exists
    if [ -f /app/scripts/init-db.sh ]; then
        echo "Running database initialization script..."
        chmod +x /app/scripts/init-db.sh
        bash /app/scripts/init-db.sh
    else
        echo "Database initialization script not found, skipping..."
    fi
fi

echo "Container initialization complete."

# Execute whatever command was passed to the container
exec "$@"
EOF
  chmod +x scripts/docker_init.sh
  echo "Created scripts/docker_init.sh"
fi

section "4. Fixing Dockerfile permissions"
if [ -f Dockerfile ]; then
  # Create new Dockerfile with fixed permissions
  cat > Dockerfile << 'EOF'
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
    postgresql-client \
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

# Create necessary directories with correct permissions
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features \
    /app/data/models /app/data/evaluation /app/data/predictions \
    /app/logs /app/model_cache /app/uploads \
    && chmod -R 777 /app/data /app/logs /app/model_cache /app/uploads

# Copy the init script and make it executable
COPY scripts/docker_init.sh /app/docker_init.sh
RUN chmod +x /app/docker_init.sh

# Skip creating non-root user to avoid permission issues
# This is a compromise for development environments
# For production, revisit this approach with proper volume mounts

# Run the application
EXPOSE 8000
EXPOSE 9091
ENTRYPOINT ["/app/docker_init.sh"]
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
  echo "Updated Dockerfile with fixed permissions"
else
  echo "No Dockerfile found, skipping..."
fi

section "5. Rebuilding and restarting containers"
echo "Stopping any running containers..."
docker compose down

echo "Rebuilding and starting containers..."
docker compose up -d --build

echo
echo "=========================================================="
echo "All fixes have been applied!"
echo
echo "If you still experience issues, check the container logs:"
echo "docker compose logs app"
echo "docker compose logs db"
echo
echo "For detailed database initialization logs:"
echo "docker compose logs db | grep -i 'database'"
echo "==========================================================" 