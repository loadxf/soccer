#!/bin/bash
# Script to fix missing function imports in the Soccer Prediction System
# Error: ImportError: cannot import name 'load_feature_pipeline' from 'src.data.features'

set -e  # Exit on error

echo "Soccer Prediction System - Missing Function Fix"
echo "=============================================="

# Create a backup of the features file if it exists
if [ -f src/data/features.py ]; then
  echo "Creating backup of features.py..."
  cp src/data/features.py src/data/features.py.bak
fi

echo "Creating or updating src/data/features.py with missing functions..."
mkdir -p src/data

# Create/update the features.py file with the required functions
cat > src/data/features.py << 'EOF'
"""
Features module for the Soccer Prediction System.
Handles feature creation, extraction, and transformation for model training.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import joblib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data.features")

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

def save_feature_dataset(data: pd.DataFrame, name: str, version: str = None) -> str:
    """Save a feature dataset to disk."""
    if version is None:
        version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    file_path = FEATURES_DIR / f"{name}_{version}.csv"
    logger.info(f"Saving feature dataset to {file_path}")
    
    try:
        data.to_csv(file_path, index=False)
        logger.info(f"Dataset saved successfully: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return ""

# Additional functions that might be used by other modules
def get_available_feature_datasets() -> List[Dict]:
    """Get information about available feature datasets."""
    files = list(FEATURES_DIR.glob("*.csv"))
    datasets = []
    
    for file in files:
        file_name = file.name
        # Extract name and version from filename
        parts = file_name.split('_')
        if len(parts) >= 2:
            name = parts[0]
            version = '_'.join(parts[1:]).replace('.csv', '')
            
            # Get file stats
            stats = os.stat(file)
            datasets.append({
                "name": name,
                "version": version,
                "path": str(file),
                "size_bytes": stats.st_size,
                "created": pd.Timestamp(stats.st_ctime, unit='s').isoformat(),
                "modified": pd.Timestamp(stats.st_mtime, unit='s').isoformat()
            })
    
    return datasets
EOF

echo "Stopping containers to apply fix..."
docker compose down

echo "Starting containers with fixed functions..."
docker compose up -d

echo -e "\nFixes have been applied!"
echo "The missing functions in features.py have been added"
echo "The containers should now start correctly"

echo -e "\nTo check the app logs:"
echo "docker compose logs app"
echo -e "\nTo check the frontend logs:"
echo "docker compose logs frontend" 