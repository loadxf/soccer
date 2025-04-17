import os
import json
import uuid
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import mimetypes
import humanize
import magic
from kaggle.api.kaggle_api_extended import KaggleApi
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import zipfile
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Import DATA_DIR from config
try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback if import fails
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

# Constants
REGISTRY_FILE = os.path.join(DATA_DIR, "dataset_registry.json")
DATASETS_FILE = os.path.join(DATA_DIR, "datasets.json")
LEGACY_REGISTRY_FILE = os.path.join(DATA_DIR, "datasets.json")

# Registry cache in memory
registry = {}

def get_registry():
    """
    Get the dataset registry
    
    Returns:
        dict: Dataset registry
    """
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    return {}  # Return empty dictionary, not list

def get_human_readable_size(size_bytes):
    """Convert bytes to human-readable format."""
    return humanize.naturalsize(size_bytes)

def get_file_type(file_path):
    """Determine the file type based on its content and extension."""
    if not os.path.exists(file_path):
        return "unknown"
    
    try:
        # First try to use python-magic to get the file type
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        
        # If it's just a generic binary or text type, try to be more specific
        if file_type in ['application/octet-stream', 'text/plain']:
            # Use the file extension to be more specific
            ext = os.path.splitext(file_path)[1].lower()
            mime_type = mimetypes.guess_type(file_path)[0]
            
            if mime_type:
                return mime_type
            
            # Map common extensions to types
            extension_map = {
                '.csv': 'text/csv',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel',
                '.json': 'application/json',
                '.parquet': 'application/parquet',
                '.py': 'text/x-python',
                '.md': 'text/markdown',
                '.txt': 'text/plain',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.pdf': 'application/pdf'
            }
            
            if ext in extension_map:
                return extension_map[ext]
        
        return file_type
    except Exception as e:
        logger.warning(f"Failed to determine file type for {file_path}: {str(e)}")
        # Fallback to using extension
        return mimetypes.guess_type(file_path)[0] or "unknown"

def analyze_data_file(file_path):
    """
    Analyze a data file to extract row count and column names.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
        
    Returns:
    --------
    tuple
        (row_count, column_names)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.csv':
            # Try to read the first few rows to get columns
            df_sample = pd.read_csv(file_path, nrows=5)
            
            # Get the actual row count more efficiently
            row_count = 0
            for _ in pd.read_csv(file_path, chunksize=10000):
                row_count += len(_)
                
            return row_count, df_sample.columns.tolist()
            
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            return len(df), df.columns.tolist()
            
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
            return len(df), df.columns.tolist()
            
        elif ext == '.json':
            # This assumes the JSON is a list of objects or a single object with a list value
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                if data and isinstance(data[0], dict):
                    return len(data), list(data[0].keys())
                return len(data), []
            elif isinstance(data, dict):
                # Try to find any list in the JSON
                for key, value in data.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        return len(value), list(value[0].keys())
                
                # If no list is found, just return the keys
                return 1, list(data.keys())
            
            return 0, []
            
        # Add support for more file types as needed
        
        return 0, []
    except Exception as e:
        logger.warning(f"Failed to analyze data file {file_path}: {str(e)}")
        return 0, []

def register_kaggle_dataset(dataset_ref, target_dir=None, force_download=False):
    """
    Register a Kaggle dataset.
    
    Parameters
    ----------
    dataset_ref : str
        Kaggle dataset reference in the format "username/dataset-name".
    target_dir : str, optional
        Target directory to download the dataset to.
    force_download : bool, optional
        Whether to force download the dataset even if it already exists.
        
    Returns
    -------
    dict
        Dataset entry.
    """
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Generate a unique dataset ID
    dataset_id = str(uuid.uuid4())
    logger.info(f"Registering dataset {dataset_ref} with ID {dataset_id}")
    
    # Retrieve dataset metadata
    try:
        logger.info(f"Retrieving metadata for {dataset_ref}")
        metadata = api.dataset_view(dataset_ref)
        title = metadata.title
        description = metadata.description[:200] + ('...' if len(metadata.description) > 200 else '')
        url = f"https://www.kaggle.com/datasets/{dataset_ref}"
        logger.info(f"Retrieved metadata: {title}")
    except Exception as e:
        logger.error(f"Failed to retrieve metadata for {dataset_ref}: {e}")
        title = dataset_ref.split('/')[-1]
        description = f"Dataset from {dataset_ref}"
        url = f"https://www.kaggle.com/datasets/{dataset_ref}"
    
    # Create a unique target directory if not specified
    if target_dir is None:
        target_dir = os.path.join(DATA_DIR, f"{dataset_id}_{title.replace(' ', '_')}")
    
    # Download the dataset
    logger.info(f"Downloading dataset to {target_dir}")
    api.dataset_download_files(dataset_ref, path=target_dir, unzip=True, force=force_download)
    
    # Collect dataset statistics
    logger.info("Collecting dataset statistics")
    file_stats = []
    total_size = 0
    largest_file = {'name': '', 'size': 0}
    file_types = {}
    
    for root, _, files in os.walk(target_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            # Track largest file
            if file_size > largest_file['size']:
                largest_file = {'name': file, 'size': file_size}
            
            # Track file types
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in file_types:
                file_types[file_ext] += 1
            else:
                file_types[file_ext] = 1
            
            # Add file stats
            rel_path = os.path.relpath(file_path, target_dir)
            file_stats.append({
                'name': rel_path,
                'size': file_size,
                'human_size': humanize.naturalsize(file_size)
            })
    
    # Sort files by size (largest first)
    file_stats.sort(key=lambda x: x['size'], reverse=True)
    
    # Create dataset entry
    dataset_entry = {
        'id': dataset_id,
        'ref': dataset_ref,
        'title': title,
        'description': description,
        'url': url,
        'directory': target_dir,
        'stats': {
            'file_count': len(file_stats),
            'total_size': total_size,
            'human_size': humanize.naturalsize(total_size),
            'largest_file': {
                'name': largest_file['name'],
                'size': largest_file['size'],
                'human_size': humanize.naturalsize(largest_file['size'])
            },
            'file_types': file_types
        },
        'files': file_stats,
        'created_at': datetime.now().isoformat()
    }
    
    # Save to registry
    registry = get_registry()
    registry[dataset_id] = dataset_entry
    
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)
    
    # For compatibility, also update the datasets.json file
    datasets = []
    if os.path.exists(DATASETS_FILE):
        try:
            with open(DATASETS_FILE, 'r') as f:
                datasets = json.load(f)
        except:
            pass
    
    datasets.append({
        'id': dataset_id,
        'name': title,
        'description': description,
        'directory': target_dir
    })
    
    with open(DATASETS_FILE, 'w') as f:
        json.dump(datasets, f, indent=2)
    
    logger.info(f"Dataset {dataset_ref} registered successfully with ID {dataset_id}")
    return dataset_entry

def get_dataset_by_id(dataset_id):
    """
    Get a dataset by its ID
    
    Args:
        dataset_id (str): Dataset ID
        
    Returns:
        dict: Dataset information or None if not found
    """
    registry = get_registry()
    return registry.get(dataset_id)

def get_dataset_preview(dataset_id, max_files=5, max_lines=10, max_size=10000):
    """
    Get a preview of the dataset files.
    
    Args:
        dataset_id: The ID of the dataset
        max_files: Maximum number of files to preview
        max_lines: Maximum number of lines to preview per text file
        max_size: Maximum size (in bytes) for files to be considered for preview
    
    Returns:
        dict: A dictionary containing dataset info and file previews.
        Structure:
        {
            "id": str,
            "title": str,
            "description": str,
            "stats": {
                "total_files": int,
                "total_size": int,
                "previewed_files": int
            },
            "files": [
                {
                    "name": str,
                    "size": int,
                    "file_type": str,
                    "content": str or None,
                    "preview_message": str or None
                }
            ]
        }
    """
    # Get dataset from registry
    registry = get_registry()
    if dataset_id not in registry:
        logger.error(f"Dataset {dataset_id} not found in registry")
        return {
            "id": dataset_id,
            "title": "Unknown",
            "description": "Dataset not found in registry",
            "stats": {
                "total_files": 0,
                "total_size": 0,
                "previewed_files": 0
            },
            "files": []
        }

    dataset = registry[dataset_id]
    dataset_dir = dataset["directory"]
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory {dataset_dir} does not exist")
        return {
            "id": dataset_id,
            "title": dataset.get("title", "Unknown"),
            "description": "Dataset directory not found",
            "stats": {
                "total_files": 0,
                "total_size": 0,
                "previewed_files": 0
            },
            "files": []
        }
    
    # Get list of files in the dataset directory (recursively)
    all_files = []
    total_size = 0
    
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            # Get relative path from dataset directory
            rel_path = os.path.relpath(file_path, dataset_dir)
            all_files.append({
                "name": rel_path,
                "path": file_path,
                "size": file_size
            })
    
    # Sort files by size (smallest first for preview)
    all_files.sort(key=lambda x: x["size"])
    
    # Limit number of files to preview
    preview_files = all_files[:max_files]
    
    file_previews = []
    for file_info in preview_files:
        file_path = file_info["path"]
        file_name = file_info["name"]
        file_size = file_info["size"]
        
        # Determine file type using magic
        try:
            import magic
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
        except Exception:
            # Fallback to extension-based type detection
            _, ext = os.path.splitext(file_path)
            if ext.lower() in ['.txt', '.csv', '.md', '.json', '.py', '.js', '.html', '.css']:
                file_type = f"text/{ext[1:]}"
            elif ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                file_type = f"image/{ext[1:]}"
            else:
                file_type = "application/octet-stream"
        
        content = None
        preview_message = None
        
        # Handle based on file type
        if file_size > max_size:
            preview_message = f"File too large to preview ({humanize.naturalsize(file_size)})"
        elif file_type.startswith('text/') or file_type in ['application/json', 'application/csv', 'application/x-python']:
            try:
                # Count total lines in the file
                total_lines = count_lines(file_path)
                
                # Read the file with limited lines
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = ''.join([next(f) for _ in range(min(max_lines, total_lines))])
                
                # Add message if truncated
                if total_lines > max_lines:
                    preview_message = f"Showing first {max_lines} of {total_lines} lines"
            except UnicodeDecodeError:
                preview_message = "File appears to be binary despite text MIME type"
            except Exception as e:
                preview_message = f"Error reading file: {str(e)}"
        elif file_type.startswith('image/'):
            preview_message = f"Image file: {file_type}"
        else:
            preview_message = f"Binary file: {file_type}"
        
        file_previews.append({
            "name": file_name,
            "size": file_size,
            "file_type": file_type,
            "content": content,
            "preview_message": preview_message
        })
    
    return {
        "id": dataset_id,
        "title": dataset.get("title", "Unknown"),
        "description": dataset.get("description", ""),
        "stats": {
            "total_files": len(all_files),
            "total_size": total_size,
            "previewed_files": len(file_previews)
        },
        "files": file_previews
    }

def count_lines(file_path):
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except UnicodeDecodeError:
        return 0
    except Exception as e:
        logger.error(f"Error counting lines in {file_path}: {e}")
        return 0 