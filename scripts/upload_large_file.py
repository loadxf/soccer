#!/usr/bin/env python
"""
Large File Upload Script for Soccer Prediction System

This script helps users upload files larger than Streamlit's 1000MB limit.
It copies the file to the appropriate data directory and registers it in
the dataset registry.

Usage:
    python upload_large_file.py input_file.csv [destination_dir]

Arguments:
    input_file      - Path to the large file to upload
    destination_dir - Optional destination directory (default: data/uploads)
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
import json
import pandas as pd
from datetime import datetime
import uuid

# Add the project root to the path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

# Set up data directories
DATA_DIR = project_root / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
REGISTRY_FILE = DATA_DIR / "datasets.json"

# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)

def load_registry():
    """Load the dataset registry from the JSON file."""
    if not os.path.exists(REGISTRY_FILE):
        # Create a new registry file if it doesn't exist
        registry = {
            "datasets": [],
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
        save_registry(registry)
        return registry
    
    try:
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading registry: {e}")
        return {"datasets": [], "last_updated": datetime.now().isoformat(), "version": "1.0"}

def save_registry(registry):
    """Save the dataset registry to the JSON file."""
    # Update the last_updated timestamp
    registry["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=4)
    except IOError as e:
        print(f"Error saving registry: {e}")

def register_dataset(file_path, name=None):
    """Register a dataset in the registry."""
    # Load the registry
    registry = load_registry()
    
    # Generate a unique ID
    dataset_id = str(uuid.uuid4())
    
    # Get file info
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    if name is None:
        name = file_name
    
    # Try to determine rows and columns (for CSV and Excel)
    rows = 0
    columns = []
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            # Read just the header for columns
            df_header = pd.read_csv(file_path, nrows=0)
            columns = df_header.columns.tolist()
            
            # Count lines in the file for a rough row estimate
            with open(file_path, 'r') as f:
                rows = sum(1 for _ in f) - 1  # Subtract header
        
        elif file_ext in ['.xlsx', '.xls']:
            # For Excel, we can get shape without reading the whole file
            df = pd.read_excel(file_path, sheet_name=0, nrows=5)
            columns = df.columns.tolist()
            
            # This is more approximate for Excel
            df_info = pd.read_excel(file_path, sheet_name=0, nrows=1)
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            sheet = wb.active
            rows = sheet.max_row - 1  # Subtract header
    except Exception as e:
        print(f"Warning: Could not determine file structure: {e}")
    
    # Create the dataset entry
    dataset_info = {
        "id": dataset_id,
        "name": name,
        "file_name": file_name,
        "path": str(file_path),
        "size": file_size,
        "rows": rows,
        "columns": columns,
        "format": os.path.splitext(file_path)[1].lower().replace('.', ''),
        "upload_date": datetime.now().isoformat(),
        "status": "raw",
        "source": "manual_upload",
        "description": f"Manually uploaded via upload_large_file.py script"
    }
    
    # Add the dataset to the registry
    registry["datasets"].append(dataset_info)
    
    # Save the updated registry
    save_registry(registry)
    
    return dataset_id

def upload_file(file_path, target_dir=None, custom_name=None):
    """Upload a large file to the system."""
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False
    
    # Determine target directory
    if target_dir:
        target_dir_path = Path(target_dir)
    else:
        target_dir_path = UPLOADS_DIR
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir_path, exist_ok=True)
    
    # Determine target file path
    file_name = os.path.basename(file_path)
    target_file_path = target_dir_path / file_name
    
    # Copy the file
    try:
        print(f"Copying file to {target_file_path}...")
        shutil.copy2(file_path, target_file_path)
        print(f"File copied successfully.")
    except Exception as e:
        print(f"Error copying file: {e}")
        return False
    
    # Register the dataset
    try:
        dataset_id = register_dataset(target_file_path, custom_name)
        print(f"File registered in dataset registry with ID: {dataset_id}")
    except Exception as e:
        print(f"Error registering dataset: {e}")
        return False
    
    return True

def main():
    """Main entry point for the script."""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Upload large files to the Soccer Prediction System")
    parser.add_argument("file_path", help="Path to the file to upload")
    parser.add_argument("target_dir", nargs="?", help="Optional target directory (default: data/uploads)")
    parser.add_argument("--name", help="Custom name for the dataset")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Upload the file
    success = upload_file(args.file_path, args.target_dir, args.name)
    
    if success:
        print("\n✅ File uploaded and registered successfully!")
        print("You can now access this dataset through the UI.")
    else:
        print("\n❌ File upload failed.")

if __name__ == "__main__":
    main() 