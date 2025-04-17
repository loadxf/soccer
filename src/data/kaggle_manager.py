"""
Kaggle Manager Module

This module provides functions for managing Kaggle datasets and API credentials.
It safely handles Kaggle imports and authentication to avoid errors.
"""

import os
import sys
from pathlib import Path
import logging
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to check if Kaggle is properly configured
def is_kaggle_configured():
    """
    Check if Kaggle credentials are properly configured.
    
    Returns:
        bool: True if Kaggle credentials are found, False otherwise
    """
    # Check if kaggle.json exists - Unix style path
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    json_exists = os.path.exists(kaggle_json_path)
    
    if json_exists:
        # On Unix/Linux, check file permissions (should be 600)
        if os.name != 'nt':  # Unix/Linux systems
            try:
                import stat
                file_stat = os.stat(kaggle_json_path)
                mode = file_stat.st_mode
                
                # Check if file is only readable/writable by owner
                if mode & (stat.S_IRWXG | stat.S_IRWXO):
                    logger.warning(f"Kaggle credentials file has incorrect permissions: {oct(mode)}")
                    logger.warning("For security, please run: chmod 600 ~/.kaggle/kaggle.json")
                    # Still return True, but warn user about insecure permissions
                
                logger.info(f"Kaggle credentials found at: {kaggle_json_path}")
            except Exception as e:
                logger.warning(f"Could not check file permissions: {str(e)}")
        else:
            logger.info(f"Kaggle credentials found at: {kaggle_json_path}")
        return True
    
    # Check Windows-specific path
    if os.name == 'nt':  # Windows systems
        # Try Windows-specific path formats
        win_username = os.environ.get('USERNAME')
        win_paths = [
            f"C:\\Users\\{win_username}\\.kaggle\\kaggle.json",
            os.path.join(os.environ.get('USERPROFILE', ''), '.kaggle', 'kaggle.json')
        ]
        
        for win_path in win_paths:
            if os.path.exists(win_path):
                logger.info(f"Kaggle credentials found at Windows path: {win_path}")
                return True
            else:
                logger.debug(f"Checked Windows path (not found): {win_path}")
    else:  # Additional Unix/Linux paths to check
        # Try common Linux username environment variables
        linux_username = os.environ.get('USER') or os.environ.get('LOGNAME')
        if linux_username:
            linux_paths = [
                f"/home/{linux_username}/.kaggle/kaggle.json",
                f"/var/lib/{linux_username}/.kaggle/kaggle.json"  # For service accounts
            ]
            
            for linux_path in linux_paths:
                if os.path.exists(linux_path):
                    logger.info(f"Kaggle credentials found at Linux path: {linux_path}")
                    return True
                else:
                    logger.debug(f"Checked Linux path (not found): {linux_path}")
    
    # Check if env vars are set
    env_vars_set = os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')
    
    if env_vars_set:
        logger.info("Kaggle credentials found in environment variables")
        return True
    
    # Log that no credentials were found
    logger.warning("No Kaggle credentials found in any location")
    return False

# Function to safely import Kaggle without triggering authentication errors
def safe_import_kaggle():
    """
    Safely import Kaggle API without triggering authentication errors.
    
    Returns:
        tuple: (success, module or error message)
    """
    try:
        # First check if credentials exist
        has_credentials = False
        
        # Check for credentials file
        if os.path.exists("/root/.kaggle/kaggle.json"):
            # Docker environment path
            os.environ["KAGGLE_CONFIG_DIR"] = "/root/.kaggle"
            has_credentials = True
            logger.info("Using Kaggle credentials from /root/.kaggle/kaggle.json")
        elif os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            # Standard path
            os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")
            has_credentials = True
            logger.info(f"Using Kaggle credentials from {os.path.expanduser('~/.kaggle/kaggle.json')}")
            
        # Check for environment variables as fallback
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            has_credentials = True
            logger.info("Using Kaggle credentials from environment variables")
            
        # Now import Kaggle
        import kaggle
        
        # Return success only if we have credentials
        if has_credentials:
            return True, kaggle
        else:
            return False, "No Kaggle credentials found. Please set up your kaggle.json file or environment variables."
            
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
    try:
        # Direct Kaggle import with explicit credential setup
        import kaggle
        
        # Ensure credentials are properly set
        credentials_found = False
        credentials_errors = []
        
        # Check Docker path first
        if os.path.exists("/root/.kaggle/kaggle.json"):
            try:
                # Load credentials from file
                with open("/root/.kaggle/kaggle.json", "r") as f:
                    creds = json.load(f)
                    os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
                    os.environ["KAGGLE_KEY"] = creds.get("key", "")
                os.environ["KAGGLE_CONFIG_DIR"] = "/root/.kaggle"
                credentials_found = True
                logger.info("Using Kaggle credentials from /root/.kaggle/kaggle.json")
            except Exception as e:
                credentials_errors.append(f"Error loading Docker credentials: {str(e)}")
        
        # Try standard path if Docker path failed
        if not credentials_found and os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            try:
                # Load credentials from file
                with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
                    creds = json.load(f)
                    os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
                    os.environ["KAGGLE_KEY"] = creds.get("key", "")
                os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")
                credentials_found = True
                logger.info(f"Using Kaggle credentials from {os.path.expanduser('~/.kaggle/kaggle.json')}")
            except Exception as e:
                credentials_errors.append(f"Error loading standard credentials: {str(e)}")
        
        # Check if environment variables are set
        if not credentials_found:
            if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
                credentials_found = True
                logger.info("Using Kaggle credentials from environment variables")
            else:
                error_message = "No Kaggle credentials found. Please set up your kaggle.json file or environment variables."
                if credentials_errors:
                    error_message += f" Errors: {'; '.join(credentials_errors)}"
                logger.error(error_message)
                return {
                    "status": "error",
                    "message": error_message
                }
        
        # Create a unique directory for this dataset to prevent overlap
        # Extract dataset name from reference (username/dataset-name)
        dataset_name = dataset_ref.split('/')[-1] if '/' in dataset_ref else dataset_ref
        unique_id = str(uuid.uuid4())[:8]  # Use a UUID to ensure uniqueness
        
        # Determine project root and import DATA_DIR
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Import DATA_DIR from config
        try:
            sys.path.insert(0, str(project_root))
            from config.default_config import DATA_DIR
        except ImportError:
            # Fallback definition if import fails
            DATA_DIR = project_root / "data"
        
        if not target_dir:
            # Create a dataset-specific directory inside kaggle_imports
            target_dir = DATA_DIR / "kaggle_imports" / f"{dataset_name}_{unique_id}"
            os.makedirs(target_dir, exist_ok=True)
            logger.info(f"Created unique download directory: {target_dir}")
        
        # Always create temp directory as fallback
        temp_dir = DATA_DIR / "temp" / f"{dataset_name}_{unique_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Test write permissions on target directory
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
            # Download the dataset - force fresh download
            print(f"Downloading dataset {dataset_ref} to {target_dir}")
            
            # Force fresh download by setting force=True
            kaggle.api.dataset_download_files(dataset_ref, unzip=True, force=True)
            
            # List the downloaded files
            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            print(f"Downloaded files: {files}")
            
            # Change back to original directory
            os.chdir(original_dir)
            
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