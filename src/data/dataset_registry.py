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

# Import DATA_DIR from config
try:
    from config.default_config import DATA_DIR
except ImportError:
    # Define data directory paths if import fails
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

# Define subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
USER_UPLOADS_DIR = DATA_DIR / "uploads"
KAGGLE_IMPORTS_DIR = DATA_DIR / "kaggle_imports"

# Define registry file path
REGISTRY_FILE = DATA_DIR / "datasets.json"

# List of system files and directories to ignore
SYSTEM_FILES = ['.DS_Store', '.git', '.gitignore', '__pycache__', 'Thumbs.db', 'desktop.ini']

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
        print(f"Dataset {dataset_id} not found")
        return None
    
    print(f"Getting preview for dataset: {dataset.get('name')} (ID: {dataset_id})")
    
    # Try in this order:
    # 1. preview_file (added in newer versions)
    # 2. path (could be a file or directory)
    # 3. Individual files from files list if path is a directory
    
    # Check for preview_file first (newer datasets)
    preview_file = dataset.get("preview_file")
    if preview_file and os.path.exists(preview_file) and os.path.isfile(preview_file):
        print(f"Using preview_file: {preview_file}")
        file_path = preview_file
    else:
        # Fall back to path
        file_path = dataset.get("path")
        print(f"Using path: {file_path}")
    
    if not file_path:
        print("No path found in dataset")
        return None
    
    # Check if path exists
    if not os.path.exists(file_path):
        print(f"Path does not exist: {file_path}")
        
        # If it's a directory, try to find a CSV or Excel file in it
        if dataset.get("files"):
            print(f"Trying to find a suitable file among {len(dataset.get('files'))} files listed in dataset")
            
            # Get parent directory if the path was a file
            if not os.path.isdir(file_path):
                parent_dir = os.path.dirname(file_path)
            else:
                parent_dir = file_path
                
            # Try to find a CSV or Excel file
            for file in dataset.get("files"):
                potential_file = os.path.join(parent_dir, file)
                if os.path.exists(potential_file) and os.path.isfile(potential_file):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in ['.csv', '.xlsx', '.xls', '.json']:
                        print(f"Found data file: {potential_file}")
                        file_path = potential_file
                        break
            else:
                # If no CSV/Excel found, try the first file
                if dataset.get("files") and len(dataset.get("files")) > 0:
                    potential_file = os.path.join(parent_dir, dataset.get("files")[0])
                    if os.path.exists(potential_file) and os.path.isfile(potential_file):
                        print(f"Using first available file: {potential_file}")
                        file_path = potential_file
        else:
            print("No files listed in dataset")
    elif os.path.isdir(file_path):
        print(f"Path is a directory: {file_path}")
        
        # Look in the directory for suitable files
        files_in_dir = os.listdir(file_path)
        print(f"Directory contains {len(files_in_dir)} files")
        
        # First try CSV files
        for file in files_in_dir:
            if file.lower().endswith('.csv'):
                potential_file = os.path.join(file_path, file)
                if os.path.isfile(potential_file):
                    print(f"Found CSV file in directory: {file}")
                    file_path = potential_file
                    break
        else:
            # Then try Excel files
            for file in files_in_dir:
                if file.lower().endswith(('.xlsx', '.xls')):
                    potential_file = os.path.join(file_path, file)
                    if os.path.isfile(potential_file):
                        print(f"Found Excel file in directory: {file}")
                        file_path = potential_file
                        break
            else:
                # Then try JSON files
                for file in files_in_dir:
                    if file.lower().endswith('.json'):
                        potential_file = os.path.join(file_path, file)
                        if os.path.isfile(potential_file):
                            print(f"Found JSON file in directory: {file}")
                            file_path = potential_file
                            break
                else:
                    # If none of the expected formats found, try any file
                    if files_in_dir:
                        potential_file = os.path.join(file_path, files_in_dir[0])
                        if os.path.isfile(potential_file):
                            print(f"Using first file in directory: {files_in_dir[0]}")
                            file_path = potential_file
    
    # Check again if file exists after our corrections
    if not os.path.exists(file_path) or os.path.isdir(file_path):
        print(f"File not found or is a directory: {file_path}")
        return pd.DataFrame({"Error": ["File not found or is a directory"]})
    
    # At this point, we should have a valid file path
    print(f"Final file path for preview: {file_path}")
    
    try:
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            print(f"Reading CSV file: {file_path}")
            try:
                # First try with default settings
                return pd.read_csv(file_path, nrows=rows)
            except Exception as e1:
                print(f"Standard CSV read failed, trying with different encodings and separators: {str(e1)}")
                # Try with different encodings
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                separators = [',', ';', '\t', '|']
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            df = pd.read_csv(file_path, nrows=rows, encoding=encoding, sep=sep)
                            print(f"Successfully read with encoding={encoding}, separator={sep}")
                            return df
                        except:
                            pass
                
                # If all attempts fail, raise the original error
                raise e1
                
        elif file_ext in ['.xlsx', '.xls']:
            print(f"Reading Excel file: {file_path}")
            return pd.read_excel(file_path, nrows=rows)
            
        elif file_ext == '.json':
            print(f"Reading JSON file: {file_path}")
            # For JSON files, try different approaches
            try:
                # Standard method
                return pd.read_json(file_path).head(rows)
            except:
                # Try as JSON Lines format
                try:
                    return pd.read_json(file_path, lines=True).head(rows)
                except:
                    # Try to read manually and convert
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Try to convert to DataFrame
                    if isinstance(data, list):
                        return pd.DataFrame(data).head(rows)
                    elif isinstance(data, dict):
                        return pd.DataFrame([data]).head(rows)
                    else:
                        return pd.DataFrame({"Error": ["JSON format not supported"]})
        else:
            print(f"Unsupported file format: {file_ext}")
            # Try to read as text file for preview
            try:
                with open(file_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines()[:rows]]
                return pd.DataFrame({"Content": lines})
            except:
                return pd.DataFrame({"Error": ["Unsupported file format"]})
    
    except Exception as e:
        print(f"Error reading file: {str(e)}")
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

def register_kaggle_dataset(dataset_ref, path, files):
    """
    Register a dataset imported from Kaggle.
    
    Args:
        dataset_ref (str): The Kaggle dataset reference
        path (str): Path where the dataset is stored
        files (list): List of imported files
        
    Returns:
        str: The dataset ID
    """
    # Extract dataset name from reference
    dataset_name = dataset_ref.split('/')[-1] if '/' in dataset_ref else dataset_ref
    
    # Create dataset info
    dataset_info = {
        "id": str(uuid.uuid4()),
        "name": dataset_name,
        "source": "kaggle",
        "kaggle_ref": dataset_ref,
        "path": path,
        "files": files,
        "upload_date": datetime.now().isoformat(),
        "status": "raw",
        "description": f"Imported from Kaggle: {dataset_ref}"
    }
    
    # Debug output to help diagnose issues
    print(f"Registering Kaggle dataset '{dataset_name}' with {len(files)} files in {path}")
    for i, file in enumerate(files[:5]):  # Show only the first 5 files to avoid log spamming
        print(f"  File {i+1}: {file}")
    if len(files) > 5:
        print(f"  ... and {len(files) - 5} more files")
    
    try:
        # Try to detect file types and analyze files
        file_types = []
        file_sizes = []
        row_counts = []
        columns_list = []
        data_files = []  # Track files that contain actual data (CSV, Excel, etc.)
        
        # First pass - recursively find all data files in the directory and subdirectories
        for root, dirs, all_files in os.walk(path):
            # Skip system directories
            dirs[:] = [d for d in dirs if d not in SYSTEM_FILES and not d.startswith('.')]
            
            for file in all_files:
                # Skip system files and hidden files
                if file in SYSTEM_FILES or file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                try:
                    # Get file size
                    size = os.path.getsize(file_path)
                    file_sizes.append(size)
                    
                    # Check file type and collect detailed info for data files
                    if file_ext == '.csv':
                        file_types.append("csv")
                        data_files.append((file, file_path, "csv", size))
                    elif file_ext in ['.xlsx', '.xls']:
                        file_types.append("excel") 
                        data_files.append((file, file_path, "excel", size))
                    elif file_ext == '.json':
                        file_types.append("json")
                        data_files.append((file, file_path, "json", size))
                    elif file_ext in ['.txt', '.md']:
                        file_types.append("text")
                        # Only consider text files as data if they're not documentation
                        if not any(doc_name in file.lower() for doc_name in ['readme', 'license', 'documentation']):
                            data_files.append((file, file_path, "text", size))
                    elif file_ext == '.parquet':
                        file_types.append("parquet")
                        data_files.append((file, file_path, "parquet", size))
                    else:
                        file_types.append(file_ext[1:] if file_ext else "unknown")
                except Exception as e:
                    print(f"Error analyzing file {file}: {str(e)}")
                    file_sizes.append(0)
                    file_types.append("unknown")
        
        dataset_info["file_types"] = file_types
        dataset_info["file_sizes"] = file_sizes
        
        print(f"Found {len(data_files)} potential data files among {len(files)} total files")
        
        # If we didn't find any data files, look for any readable files
        if not data_files:
            print("No standard data files found, searching for any readable files...")
            for root, dirs, all_files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in SYSTEM_FILES and not d.startswith('.')]
                for file in all_files:
                    if file in SYSTEM_FILES or file.startswith('.'):
                        continue
                    file_path = os.path.join(root, file)
                    try:
                        # Try to determine if file is readable text
                        with open(file_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline()
                            if len(first_line) > 0:
                                size = os.path.getsize(file_path)
                                data_files.append((file, file_path, "text", size))
                                print(f"  Added text file: {file}")
                    except:
                        pass  # Not a readable text file
        
        # Sort data files by size (usually larger files have more data)
        data_files.sort(key=lambda x: x[3], reverse=True)
        
        # Second pass - analyze data files for rows and columns
        primary_file_path = None
        
        # Try each data file in priority order (CSV, Excel, JSON, Parquet, Text)
        for file_format in ["csv", "excel", "json", "parquet", "text"]:
            format_files = [f for f in data_files if f[2] == file_format]
            
            if format_files:
                print(f"Trying {len(format_files)} {file_format} files...")
                
                # Try each file of this format until we find one we can read
                for file_name, file_path, file_type, file_size in format_files:
                    try:
                        if file_type == "csv":
                            # Try to read with pandas, handling potential errors
                            df = pd.read_csv(file_path, nrows=5)
                            count = len(pd.read_csv(file_path, usecols=[0]))
                            
                            row_counts.append(count)
                            columns_list.append(df.columns.tolist())
                            
                            # If we found a valid CSV, use it as primary
                            if primary_file_path is None:
                                primary_file_path = file_path
                                dataset_info["preview_file"] = file_path
                                dataset_info["columns"] = df.columns.tolist()
                                dataset_info["rows"] = count
                                print(f"Selected primary file: {file_name} with {count} rows and {len(df.columns)} columns")
                                
                        elif file_type == "excel":
                            # Try to read Excel, handling potential errors
                            df = pd.read_excel(file_path, nrows=5)
                            count = len(pd.read_excel(file_path, usecols=[0]))
                            
                            row_counts.append(count)
                            columns_list.append(df.columns.tolist())
                            
                            # If we found a valid Excel file and have no primary file yet, use it
                            if primary_file_path is None:
                                primary_file_path = file_path
                                dataset_info["preview_file"] = file_path
                                dataset_info["columns"] = df.columns.tolist()
                                dataset_info["rows"] = count
                                print(f"Selected primary file: {file_name} with {count} rows and {len(df.columns)} columns")
                                
                        elif file_type == "json":
                            # Try different JSON reading approaches
                            try:
                                # Standard dataframe approach
                                df = pd.read_json(file_path)
                                count = len(df)
                                
                                row_counts.append(count)
                                columns_list.append(df.columns.tolist())
                                
                                # If we found a valid JSON file and have no primary file yet, use it
                                if primary_file_path is None:
                                    primary_file_path = file_path
                                    dataset_info["preview_file"] = file_path
                                    dataset_info["columns"] = df.columns.tolist()
                                    dataset_info["rows"] = count
                                    print(f"Selected primary file: {file_name} with {count} rows and {len(df.columns)} columns")
                            except:
                                # Try JSON lines format
                                try:
                                    df = pd.read_json(file_path, lines=True)
                                    count = len(df)
                                    
                                    row_counts.append(count)
                                    columns_list.append(df.columns.tolist())
                                    
                                    if primary_file_path is None:
                                        primary_file_path = file_path
                                        dataset_info["preview_file"] = file_path
                                        dataset_info["columns"] = df.columns.tolist()
                                        dataset_info["rows"] = count
                                        print(f"Selected primary file (JSON Lines): {file_name}")
                                except:
                                    # Try manual JSON parsing
                                    with open(file_path, 'r') as f:
                                        json_data = json.load(f)
                                    
                                    if isinstance(json_data, list) and len(json_data) > 0:
                                        count = len(json_data)
                                        cols = list(json_data[0].keys()) if isinstance(json_data[0], dict) else []
                                        
                                        row_counts.append(count)
                                        columns_list.append(cols)
                                        
                                        if primary_file_path is None:
                                            primary_file_path = file_path
                                            dataset_info["preview_file"] = file_path
                                            dataset_info["columns"] = cols
                                            dataset_info["rows"] = count
                                            print(f"Selected primary file (manual JSON): {file_name}")
                                    elif isinstance(json_data, dict):
                                        # Some special handling for complex JSON structures
                                        cols = list(json_data.keys())
                                        
                                        if primary_file_path is None:
                                            primary_file_path = file_path
                                            dataset_info["preview_file"] = file_path
                                            dataset_info["columns"] = cols
                                            dataset_info["rows"] = 1
                                            print(f"Selected primary file (JSON object): {file_name}")
                                
                        elif file_type == "parquet":
                            # Try to read Parquet
                            df = pd.read_parquet(file_path)
                            count = len(df)
                            
                            row_counts.append(count)
                            columns_list.append(df.columns.tolist())
                            
                            if primary_file_path is None:
                                primary_file_path = file_path
                                dataset_info["preview_file"] = file_path
                                dataset_info["columns"] = df.columns.tolist()
                                dataset_info["rows"] = count
                                print(f"Selected primary file (Parquet): {file_name}")
                                
                        elif file_type == "text":
                            # For text files, try to detect delimiter
                            with open(file_path, 'r') as f:
                                sample = f.read(1024)
                            
                            if ',' in sample:
                                try:
                                    df = pd.read_csv(file_path, nrows=5)
                                    count = sum(1 for _ in open(file_path)) - 1  # Subtract header
                                    
                                    row_counts.append(count)
                                    columns_list.append(df.columns.tolist())
                                    
                                    if primary_file_path is None:
                                        primary_file_path = file_path
                                        dataset_info["preview_file"] = file_path
                                        dataset_info["columns"] = df.columns.tolist()
                                        dataset_info["rows"] = count
                                        print(f"Selected primary file (CSV-like text): {file_name}")
                                except:
                                    pass
                            elif '\t' in sample:
                                try:
                                    df = pd.read_csv(file_path, sep='\t', nrows=5)
                                    count = sum(1 for _ in open(file_path)) - 1
                                    
                                    row_counts.append(count)
                                    columns_list.append(df.columns.tolist())
                                    
                                    if primary_file_path is None:
                                        primary_file_path = file_path
                                        dataset_info["preview_file"] = file_path
                                        dataset_info["columns"] = df.columns.tolist()
                                        dataset_info["rows"] = count
                                        print(f"Selected primary file (TSV-like text): {file_name}")
                                except:
                                    pass
                            
                    except Exception as e:
                        # Log the error but continue with other files
                        print(f"Error analyzing data file {file_name}: {str(e)}")
                        row_counts.append(0)
                        columns_list.append([])
                
                # If we found a primary file of this format, stop looking at other formats
                if primary_file_path is not None:
                    break
        
        # Store file metadata
        dataset_info["row_counts"] = row_counts
        dataset_info["columns_by_file"] = columns_list
        
        # If we found a primary file, use it for path
        if primary_file_path:
            dataset_info["path"] = primary_file_path
            print(f"Using primary file for path: {primary_file_path}")
        elif len(files) > 0:
            # If we couldn't find a suitable data file but there are files, 
            # use the directory as the path (avoid .DS_Store and system files)
            real_files = [f for f in files if f not in SYSTEM_FILES and not f.startswith('.')]
            if real_files:
                dataset_info["path"] = path  # Use the directory
                print(f"No suitable data files found, using directory: {path}")
            else:
                print("Warning: No usable files found in dataset")
                dataset_info["path"] = path
            
    except Exception as e:
        print(f"Error analyzing Kaggle files: {str(e)}")
    
    # Register the dataset
    dataset_id = register_dataset(dataset_info)
    print(f"Registered dataset with ID: {dataset_id}")
    return dataset_id

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