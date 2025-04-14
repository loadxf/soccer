"""
Kaggle Manager Module

This module provides functions for managing Kaggle datasets and API credentials.
It safely handles Kaggle imports and authentication to avoid errors.
"""

import os
import sys
from pathlib import Path

# Function to check if Kaggle is properly configured
def is_kaggle_configured():
    """
    Check if Kaggle credentials are properly configured.
    
    Returns:
        bool: True if Kaggle credentials are found, False otherwise
    """
    # Check if kaggle.json exists
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    json_exists = os.path.exists(kaggle_json_path)
    
    # Check if env vars are set
    env_vars_set = os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')
    
    return json_exists or env_vars_set

# Function to safely import Kaggle without triggering authentication errors
def safe_import_kaggle():
    """
    Safely import Kaggle API without triggering authentication errors.
    
    Returns:
        tuple: (success, module or error message)
    """
    try:
        # Temporarily modify environment to prevent authentication
        old_config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
        os.environ["KAGGLE_CONFIG_DIR"] = "DISABLED_TEMP_DONT_AUTHENTICATE"
        
        # Try importing
        import kaggle
        
        # Restore original environment
        if old_config_dir:
            os.environ["KAGGLE_CONFIG_DIR"] = old_config_dir
        else:
            os.environ.pop("KAGGLE_CONFIG_DIR", None)
            
        return True, kaggle
    except ImportError:
        return False, "Kaggle package is not installed. Run: pip install kaggle"
    except Exception as e:
        return False, f"Error importing Kaggle: {str(e)}"

# Function to test Kaggle authentication
def test_kaggle_auth():
    """
    Test Kaggle authentication and return status.
    
    Returns:
        dict: Result with status and message
    """
    success, result = safe_import_kaggle()
    
    if not success:
        return {
            "status": "error",
            "message": str(result)
        }
    
    kaggle = result
    
    try:
        # Try authentication
        kaggle.api.authenticate()
        
        # If we get here, authentication worked
        return {
            "status": "success",
            "message": "Kaggle authentication successful"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Kaggle authentication failed: {str(e)}"
        }

# Function to search for Kaggle datasets
def search_kaggle_datasets(search_term="soccer", max_results=10):
    """
    Search for Kaggle datasets matching the search term.
    
    Args:
        search_term (str): Search term
        max_results (int): Maximum number of results
        
    Returns:
        list: Dataset information or None if error
    """
    success, result = safe_import_kaggle()
    
    if not success:
        return None
    
    kaggle = result
    
    try:
        # Try authentication
        kaggle.api.authenticate()
        
        # Search for datasets
        datasets = kaggle.api.dataset_list(search=search_term, max_size=max_results)
        
        # Format results
        result_list = []
        for dataset in datasets:
            result_list.append({
                "ref": dataset.ref,
                "title": dataset.title,
                "size": dataset.size,
                "lastUpdated": dataset.lastUpdated,
                "downloadCount": dataset.downloadCount
            })
        
        return result_list
    except Exception as e:
        print(f"Error searching Kaggle datasets: {e}")
        return None

# Function to import a Kaggle dataset
def import_dataset(dataset_ref, target_dir=None):
    """
    Import a dataset from Kaggle.
    
    Args:
        dataset_ref (str): Dataset reference (username/dataset-name)
        target_dir (str): Optional target directory
        
    Returns:
        dict: Result with status and path information
    """
    success, result = safe_import_kaggle()
    
    if not success:
        return {
            "status": "error",
            "message": str(result)
        }
    
    kaggle = result
    
    try:
        # Try authentication
        kaggle.api.authenticate()
        
        # Determine target directory
        if not target_dir:
            # Default to project data directory
            project_root = Path(__file__).resolve().parent.parent.parent
            target_dir = project_root / "data" / "kaggle_imports"
            
            # Also create a temp directory for download in case of permission issues
            temp_dir = project_root / "data" / "temp"
            os.makedirs(temp_dir, exist_ok=True)
        else:
            temp_dir = Path(target_dir) / "temp"
            os.makedirs(temp_dir, exist_ok=True)
        
        # Make sure target directory exists
        try:
            os.makedirs(target_dir, exist_ok=True)
            print(f"Target directory created: {target_dir}")
        except Exception as dir_error:
            print(f"Warning - could not create target directory: {str(dir_error)}")
            print(f"Will attempt to use temporary directory: {temp_dir}")
            target_dir = temp_dir
        
        # Test write permissions
        try:
            test_file = Path(target_dir) / ".write_test"
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"Target directory is writable: {target_dir}")
        except Exception as perm_error:
            print(f"Warning - target directory is not writable: {str(perm_error)}")
            print(f"Will attempt to use temporary directory: {temp_dir}")
            target_dir = temp_dir
            
            # Test the temp directory
            try:
                test_file = Path(temp_dir) / ".write_test"
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                print(f"Temp directory is writable: {temp_dir}")
            except Exception as temp_error:
                return {
                    "status": "error",
                    "message": f"Cannot write to any directory: {str(temp_error)}"
                }
        
        # Change to target directory for download
        original_dir = os.getcwd()
        os.chdir(target_dir)
        
        try:
            # Download the dataset
            print(f"Downloading dataset {dataset_ref} to {target_dir}")
            kaggle.api.dataset_download_files(dataset_ref, unzip=True)
            
            # List the downloaded files
            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            print(f"Downloaded files: {files}")
            
            # Change back to original directory
            os.chdir(original_dir)
            
            # If we used temp_dir, try to move files to the proper location
            if target_dir == temp_dir and len(files) > 0:
                proper_target = project_root / "data" / "kaggle_imports"
                try:
                    os.makedirs(proper_target, exist_ok=True)
                    
                    # Try to move files
                    moved_files = []
                    for file in files:
                        src = temp_dir / file
                        dst = proper_target / file
                        try:
                            import shutil
                            shutil.copy2(src, dst)
                            moved_files.append(file)
                            print(f"Copied {src} to {dst}")
                        except Exception as move_error:
                            print(f"Could not move file {file}: {str(move_error)}")
                    
                    # If we moved any files, update the target_dir
                    if moved_files:
                        target_dir = proper_target
                        files = moved_files
                except Exception as move_dir_error:
                    print(f"Could not move files to proper location: {str(move_dir_error)}")
            
            # Register the dataset
            try:
                from src.data.dataset_registry import register_kaggle_dataset
                registry_result = register_kaggle_dataset(dataset_ref, str(target_dir), files)
                print(f"Dataset registration result: {registry_result}")
            except Exception as reg_error:
                print(f"Warning - Could not register dataset: {str(reg_error)}")
            
            return {
                "status": "success",
                "message": f"Successfully imported {dataset_ref}",
                "path": str(target_dir),
                "files": files
            }
        finally:
            # Ensure we change back to original directory even if an error occurs
            if os.getcwd() != original_dir:
                os.chdir(original_dir)
    except Exception as e:
        print(f"Error importing dataset: {str(e)}")
        return {
            "status": "error",
            "message": f"Error importing dataset: {str(e)}"
        }

# Entry point for CLI usage
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Test Kaggle authentication
            result = test_kaggle_auth()
            print(f"Kaggle Authentication Test: {result['status']}")
            print(f"Message: {result['message']}")
        
        elif sys.argv[1] == "search":
            # Search for datasets
            search_term = sys.argv[2] if len(sys.argv) > 2 else "soccer"
            print(f"Searching for Kaggle datasets matching '{search_term}'...")
            datasets = search_kaggle_datasets(search_term)
            
            if datasets:
                print(f"Found {len(datasets)} datasets:")
                for i, dataset in enumerate(datasets):
                    print(f"{i+1}. {dataset['title']} ({dataset['ref']})")
            else:
                print("No datasets found or error occurred.")
        
        elif sys.argv[1] == "import":
            # Import a dataset
            if len(sys.argv) > 2:
                dataset_ref = sys.argv[2]
                print(f"Importing dataset {dataset_ref}...")
                result = import_dataset(dataset_ref)
                print(f"Status: {result['status']}")
                print(f"Message: {result['message']}")
                
                if result['status'] == "success":
                    print(f"Path: {result['path']}")
                    print(f"Files: {', '.join(result['files'])}")
            else:
                print("Please specify a dataset reference (username/dataset-name)")
        
        else:
            print("Unknown command. Available commands: test, search, import")
    else:
        print("Available commands: test, search, import") 