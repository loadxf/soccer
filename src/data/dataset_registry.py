"""
Dataset Registry Module

This module provides functions for managing the dataset registry.
The registry keeps track of datasets imported or uploaded to the system.
"""

import json
import os
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime

# Define data directory paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
USER_UPLOADS_DIR = DATA_DIR / "uploads"
KAGGLE_IMPORTS_DIR = DATA_DIR / "kaggle_imports"

# Define registry file path
REGISTRY_FILE = DATA_DIR / "datasets.json"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, USER_UPLOADS_DIR, KAGGLE_IMPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_registry():
    """
    Load the dataset registry from the JSON file.
    
    Returns:
        dict: The dataset registry
    """
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
    """
    Save the dataset registry to the JSON file.
    
    Args:
        registry (dict): The dataset registry to save
    """
    # Update the last_updated timestamp
    registry["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=4)
    except IOError as e:
        print(f"Error saving registry: {e}")

def get_all_datasets():
    """
    Get all datasets from the registry.
    
    Returns:
        list: List of dataset dictionaries
    """
    registry = load_registry()
    return registry.get("datasets", [])

def get_dataset(dataset_id):
    """
    Get a dataset by ID.
    
    Args:
        dataset_id (str): The dataset ID
        
    Returns:
        dict: The dataset or None if not found
    """
    datasets = get_all_datasets()
    for dataset in datasets:
        if dataset.get("id") == dataset_id:
            return dataset
    return None

def register_dataset(dataset_info):
    """
    Register a new dataset in the registry.
    
    Args:
        dataset_info (dict): Information about the dataset
        
    Returns:
        str: The dataset ID
    """
    registry = load_registry()
    
    # Generate a unique ID if not provided
    if "id" not in dataset_info:
        dataset_info["id"] = str(uuid.uuid4())
    
    # Add defaults if not provided
    if "upload_date" not in dataset_info:
        dataset_info["upload_date"] = datetime.now().isoformat()
    
    if "status" not in dataset_info:
        dataset_info["status"] = "raw"
    
    # Add the dataset to the registry
    registry["datasets"].append(dataset_info)
    
    # Save the updated registry
    save_registry(registry)
    
    return dataset_info["id"]

def update_dataset_status(dataset_id, status):
    """
    Update the status of a dataset.
    
    Args:
        dataset_id (str): The dataset ID
        status (str): The new status
        
    Returns:
        bool: True if successful, False otherwise
    """
    registry = load_registry()
    
    # Find and update the dataset
    for dataset in registry["datasets"]:
        if dataset.get("id") == dataset_id:
            dataset["status"] = status
            dataset["last_updated"] = datetime.now().isoformat()
            
            # Save the updated registry
            save_registry(registry)
            return True
    
    return False

def delete_dataset(dataset_id):
    """
    Delete a dataset from the registry.
    
    Args:
        dataset_id (str): The dataset ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    registry = load_registry()
    
    # Find the dataset
    for i, dataset in enumerate(registry["datasets"]):
        if dataset.get("id") == dataset_id:
            # Get the file path if available
            file_path = dataset.get("path")
            
            # Remove from registry
            registry["datasets"].pop(i)
            
            # Save the updated registry
            save_registry(registry)
            
            # Delete the file if available
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting file: {e}")
            
            return True
    
    return False

def get_dataset_preview(dataset_id, rows=5):
    """
    Get a preview of a dataset.
    
    Args:
        dataset_id (str): The dataset ID
        rows (int): Number of rows to preview
        
    Returns:
        DataFrame: Preview of the dataset or None if error
    """
    dataset = get_dataset(dataset_id)
    
    if not dataset:
        return None
    
    file_path = dataset.get("path")
    
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            return pd.read_csv(file_path, nrows=rows)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, nrows=rows)
        elif file_ext == '.json':
            return pd.read_json(file_path).head(rows)
        else:
            return pd.DataFrame({"Error": ["Unsupported file format"]})
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame({"Error": [str(e)]})

def save_uploaded_dataset(uploaded_file):
    """
    Save an uploaded file and register it in the dataset registry.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: The dataset ID or None if error
    """
    try:
        # Create a path in the uploads directory
        file_path = os.path.join(USER_UPLOADS_DIR, uploaded_file.name)
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Read file metadata
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        else:
            return None
        
        # Register the dataset
        dataset_info = {
            "name": uploaded_file.name,
            "filename": uploaded_file.name,
            "path": file_path,
            "upload_date": datetime.now().isoformat(),
            "rows": len(df),
            "columns": df.columns.tolist(),
            "status": "raw",
            "file_size": os.path.getsize(file_path)
        }
        
        return register_dataset(dataset_info)
    
    except Exception as e:
        print(f"Error saving uploaded dataset: {e}")
        return None

def batch_process_datasets(dataset_ids, process_function, **kwargs):
    """
    Process multiple datasets using the provided function.
    
    Args:
        dataset_ids (list): List of dataset IDs to process
        process_function (callable): Function to process each dataset
        **kwargs: Additional arguments to pass to process_function
        
    Returns:
        dict: Results for each dataset ID
    """
    results = {}
    
    for dataset_id in dataset_ids:
        try:
            # Process the dataset
            success = process_function(dataset_id, **kwargs)
            results[dataset_id] = success
        except Exception as e:
            print(f"Error processing dataset {dataset_id}: {e}")
            results[dataset_id] = False
    
    return results

# Main function for CLI usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # List all datasets
        if sys.argv[1] == "list":
            datasets = get_all_datasets()
            print(f"Found {len(datasets)} datasets:")
            for i, dataset in enumerate(datasets):
                print(f"{i+1}. {dataset.get('name')} (ID: {dataset.get('id')})")
        
        # Get dataset details
        elif sys.argv[1] == "get" and len(sys.argv) > 2:
            dataset_id = sys.argv[2]
            dataset = get_dataset(dataset_id)
            
            if dataset:
                print(f"Dataset: {dataset.get('name')}")
                print(f"ID: {dataset.get('id')}")
                print(f"Status: {dataset.get('status')}")
                print(f"Rows: {dataset.get('rows')}")
                print(f"Columns: {len(dataset.get('columns', []))}")
                print(f"Upload Date: {dataset.get('upload_date')}")
                print(f"File: {dataset.get('path')}")
            else:
                print(f"Dataset {dataset_id} not found")
        
        # Delete a dataset
        elif sys.argv[1] == "delete" and len(sys.argv) > 2:
            dataset_id = sys.argv[2]
            
            if delete_dataset(dataset_id):
                print(f"Dataset {dataset_id} deleted")
            else:
                print(f"Dataset {dataset_id} not found")
        
        # Update dataset status
        elif sys.argv[1] == "status" and len(sys.argv) > 3:
            dataset_id = sys.argv[2]
            status = sys.argv[3]
            
            if update_dataset_status(dataset_id, status):
                print(f"Dataset {dataset_id} status updated to {status}")
            else:
                print(f"Dataset {dataset_id} not found")
        
        # Print help
        else:
            print("Usage:")
            print("  python dataset_registry.py list")
            print("  python dataset_registry.py get <dataset_id>")
            print("  python dataset_registry.py delete <dataset_id>")
            print("  python dataset_registry.py status <dataset_id> <status>")
    else:
        print("Available commands: list, get, delete, status") 