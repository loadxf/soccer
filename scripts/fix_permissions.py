#!/usr/bin/env python3
"""
Fix data directories and permissions for the Soccer Prediction System.
This script ensures all required data directories exist and are writable.
"""

import os
import sys
import shutil
from pathlib import Path
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix-permissions")

# Get the project root directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Define data directories that need to exist
data_dirs = [
    "data/raw",
    "data/raw/football_data",
    "data/processed",
    "data/interim",
    "data/external",
    "data/kaggle_imports",
    "data/uploads",
    "data/fixtures",
    "data/models",
    "data/training",
    "data/predictions",
    "data/explainability",
    "data/features",
    "data/augmented"
]

def ensure_dirs_exist():
    """Create all required directories."""
    for dir_path in data_dirs:
        full_path = project_root / dir_path
        try:
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created/verified directory: {full_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {full_path}: {str(e)}")

def ensure_permissions():
    """Ensure directories have write permissions."""
    data_dir = project_root / "data"
    try:
        # Try to make all data directories writable
        for root, dirs, files in os.walk(data_dir):
            for d in dirs:
                try:
                    dir_path = os.path.join(root, d)
                    os.chmod(dir_path, 0o777)  # Full permissions
                    logger.info(f"Set permissions for: {dir_path}")
                except Exception as e:
                    logger.warning(f"Could not set permissions for {dir_path}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed setting permissions: {str(e)}")

def test_write_access():
    """Test write access to each directory."""
    results = {}
    for dir_path in data_dirs:
        full_path = project_root / dir_path
        test_file = full_path / "test_write_access.txt"
        try:
            with open(test_file, 'w') as f:
                f.write('Test write access')
            logger.info(f"Successfully wrote to: {test_file}")
            results[dir_path] = True
            # Clean up
            os.remove(test_file)
        except Exception as e:
            logger.error(f"Could not write to {test_file}: {str(e)}")
            results[dir_path] = False
    return results

def ensure_registry_files():
    """Ensure dataset registry files exist."""
    registry_file = project_root / "data" / "dataset_registry.json"
    if not registry_file.exists():
        try:
            with open(registry_file, 'w') as f:
                f.write('{"datasets": []}')
            logger.info(f"Created empty dataset registry: {registry_file}")
        except Exception as e:
            logger.error(f"Failed to create registry file: {str(e)}")

def fix_registry_permissions():
    """Fix permissions for registry files."""
    files_to_fix = [
        "data/dataset_registry.json",
        "data/datasets.json"
    ]
    
    for file_path in files_to_fix:
        full_path = project_root / file_path
        if full_path.exists():
            try:
                os.chmod(full_path, 0o666)  # Make file writable
                logger.info(f"Fixed permissions for: {full_path}")
            except Exception as e:
                logger.error(f"Could not fix permissions for {full_path}: {str(e)}")

def main():
    """Main function to fix permissions."""
    logger.info("Starting permission and directory fix")
    
    # Create directories
    ensure_dirs_exist()
    
    # Try to fix permissions
    ensure_permissions()
    
    # Ensure registry files exist
    ensure_registry_files()
    
    # Fix registry file permissions
    fix_registry_permissions()
    
    # Test write access
    write_results = test_write_access()
    
    # Report results
    success_count = sum(1 for v in write_results.values() if v)
    total_count = len(write_results)
    
    logger.info(f"Permission check complete: {success_count}/{total_count} directories are writable")
    
    if success_count == total_count:
        logger.info("All data directories are properly configured and writable")
        return 0
    else:
        logger.error("Some directories have permission issues. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 