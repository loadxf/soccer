"""
Kaggle UI Helper Module

This module provides Streamlit UI components for Kaggle integration.
"""

import streamlit as st
import sys
import os
import json
from pathlib import Path

# Set explicit Kaggle credentials path for Docker environment
if os.path.exists("/root/.kaggle/kaggle.json"):
    os.environ["KAGGLE_CONFIG_DIR"] = "/root/.kaggle"
    # Also load credentials as environment variables
    try:
        with open("/root/.kaggle/kaggle.json", "r") as f:
            creds = json.load(f)
            os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
            os.environ["KAGGLE_KEY"] = creds.get("key", "")
        print("Loaded Kaggle credentials from /root/.kaggle/kaggle.json")
    except Exception as e:
        print(f"Error loading Kaggle credentials: {e}")
elif os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
    os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")
    # Also load credentials as environment variables
    try:
        with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
            creds = json.load(f)
            os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
            os.environ["KAGGLE_KEY"] = creds.get("key", "")
        print(f"Loaded Kaggle credentials from {os.path.expanduser('~/.kaggle/kaggle.json')}")
    except Exception as e:
        print(f"Error loading Kaggle credentials: {e}")

# Add the project root to sys.path if needed
script_dir = Path(__file__).resolve().parent  # ui directory
project_root = script_dir.parent  # project root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the backend kaggle_manager
from src.data.kaggle_manager import (
    is_kaggle_configured,
    safe_import_kaggle,
    test_kaggle_auth,
    search_kaggle_datasets,
    import_dataset
)

def show_kaggle_setup_instructions():
    """Display instructions for setting up Kaggle API access."""
    st.warning("Kaggle API credentials not found. Follow these steps to use Kaggle datasets:")
    
    # Get current user information for paths
    windows_username = os.environ.get('USERNAME')
    linux_username = os.environ.get('USER') or os.environ.get('LOGNAME')
    windows_path = f"C:\\Users\\{windows_username}\\.kaggle\\kaggle.json"
    linux_path = f"/home/{linux_username}/.kaggle/kaggle.json"
    
    # Display instructions directly without using an expander
    st.markdown("""
    ### Setting up Kaggle API access
    
    1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com) if you don't have one
    2. Go to your Kaggle account settings (click on your profile picture → Account)
    3. Scroll down to the API section and click "Create New API Token"
    4. This will download a `kaggle.json` file with your credentials
    5. Place this file in one of these locations:
    """)
    
    # Windows-specific instructions
    if os.name == 'nt':
        st.code(f"Windows path: {windows_path}")
        st.markdown("""
        To set up on Windows:
        1. Create the `.kaggle` folder in your user directory if it doesn't exist
        2. Copy the downloaded `kaggle.json` file into this folder
        3. Make sure the file permissions are secure (only you can read the file)
        """)
    else:
        st.code(f"Linux/Mac path: {linux_path}")
        st.markdown("""
        To set up on Ubuntu/Linux:
        1. Create the `.kaggle` directory:
           ```bash
           mkdir -p ~/.kaggle
           ```
        2. Move the downloaded kaggle.json file:
           ```bash
           mv ~/Downloads/kaggle.json ~/.kaggle/
           ```
        3. Set proper permissions (required on Linux):
           ```bash
           chmod 600 ~/.kaggle/kaggle.json
           ```
        This command makes the file only readable/writable by you, which is required for security.
        """)
        
    st.markdown("""
    6. Alternatively, set these environment variables:
       - `KAGGLE_USERNAME`: Your Kaggle username
       - `KAGGLE_KEY`: Your Kaggle API key (found in the kaggle.json file)
    
    7. Install the Kaggle API package:
       ```
       pip install kaggle
       ```
    
    For more information, see the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api).
    """)
    
    st.markdown("### Handling Large Datasets")
    st.markdown("""
    For datasets larger than Streamlit's 1000MB upload limit, you have two options:
    
    #### Option 1: Direct Import via Kaggle API
    
    Once you've set up Kaggle credentials, you can import datasets directly using the Kaggle API:
    ```python
    # Example: Download dataset from Kaggle
    !kaggle datasets download -d dataset_owner/dataset_name
    ```
    
    #### Option 2: Manual Download and Upload
    
    For very large files, we provide a command-line utility:
    
    1. Download the dataset from Kaggle manually
    2. Use our large file upload script:
       ```
       python scripts/upload_large_file.py path/to/downloaded/file.csv
       ```
    3. The file will be copied to the app's data directory and registered automatically
    
    This approach bypasses Streamlit's file size limitations.
    """)

def show_kaggle_search_interface():
    """Display a search interface for Kaggle datasets."""
    st.subheader("Search Kaggle Datasets")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input("Search term", "soccer")
    
    with col2:
        max_results = st.number_input("Max results", 5, 20, 10)
    
    if st.button("Search Kaggle"):
        with st.spinner("Searching Kaggle datasets..."):
            try:
                # First check if Kaggle is configured
                if not is_kaggle_configured():
                    st.error("Kaggle credentials not found")
                    show_kaggle_setup_instructions()
                    return
                
                # Search for datasets
                datasets = search_kaggle_datasets(search_term, max_results)
                
                if datasets:
                    st.success(f"Found {len(datasets)} datasets")
                    
                    # Display datasets in a table
                    for i, dataset in enumerate(datasets):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**{i+1}. {dataset['title']}**")
                                st.markdown(f"Reference: `{dataset['ref']}`")
                                st.markdown(f"Size: {dataset['size']}")
                            
                            with col2:
                                if st.button("Import", key=f"import_{i}"):
                                    import_kaggle_dataset(dataset['ref'])
                    
                else:
                    st.warning("No datasets found or error occurred")
            
            except Exception as e:
                st.error(f"Error searching Kaggle: {str(e)}")

def import_kaggle_dataset(dataset_ref):
    """Import a Kaggle dataset and show progress."""
    with st.spinner(f"Importing dataset {dataset_ref}..."):
        try:
            # First check if Kaggle is configured
            if not is_kaggle_configured():
                st.error("Kaggle credentials not found")
                
                # Show OS-specific paths
                if os.name == 'nt':  # Windows
                    windows_username = os.environ.get('USERNAME')
                    st.warning(f"On Windows, place kaggle.json at: C:\\Users\\{windows_username}\\.kaggle\\kaggle.json")
                else:  # Linux/Mac
                    linux_username = os.environ.get('USER') or os.environ.get('LOGNAME')
                    st.warning(f"""
                        On Linux/Ubuntu, place kaggle.json at: /home/{linux_username}/.kaggle/kaggle.json
                        And set permissions: chmod 600 ~/.kaggle/kaggle.json
                    """)
                
                show_kaggle_setup_instructions()
                return False
            
            # Import the dataset
            result = import_dataset(dataset_ref)
            
            if result['status'] == "success":
                st.success(f"Successfully imported {dataset_ref}")
                st.json({
                    "path": result['path'],
                    "files": result['files']
                })
                return {
                    "status": "success",
                    "registry_result": result.get('path')
                }
            else:
                st.error(f"Error importing dataset: {result['message']}")
                return {"status": "error", "message": result['message']}
        
        except Exception as e:
            st.error(f"Error importing dataset: {str(e)}")
            return {"status": "error", "message": str(e)}

def verify_kaggle_setup():
    """Test Kaggle authentication and display the result."""
    with st.spinner("Verifying Kaggle setup..."):
        try:
            # First check if Kaggle is configured
            if not is_kaggle_configured():
                st.error("❌ Kaggle credentials not found")
                
                # Show OS-specific credential locations
                if os.name == 'nt':  # Windows
                    # Show possible credential locations for Windows
                    st.warning(f"""
                        On Windows, your credentials should be in one of these locations:
                        - `C:\\Users\\{os.environ.get('USERNAME')}\\.kaggle\\kaggle.json`
                        - `{os.path.join(os.environ.get('USERPROFILE', ''), '.kaggle', 'kaggle.json')}`
                    """)
                else:  # Linux/Mac
                    linux_username = os.environ.get('USER') or os.environ.get('LOGNAME')
                    st.warning(f"""
                        On Linux/Ubuntu, your credentials should be in:
                        - `/home/{linux_username}/.kaggle/kaggle.json`
                        
                        Make sure permissions are set correctly:
                        ```bash
                        chmod 600 ~/.kaggle/kaggle.json
                        ```
                    """)
                
                show_kaggle_setup_instructions()
                return False
                
            # Test authentication
            result = test_kaggle_auth()
            
            if result['status'] == "success":
                st.success("✅ Kaggle authentication successful")
                return True
            else:
                st.error(f"❌ Kaggle authentication failed: {result['message']}")
                show_kaggle_setup_instructions()
                return False
        
        except Exception as e:
            st.error(f"Error verifying Kaggle setup: {str(e)}")
            return False

# Demo UI if run directly
if __name__ == "__main__":
    st.set_page_config(page_title="Kaggle Integration Demo", page_icon="📊")
    
    st.title("Kaggle Integration Demo")
    
    # Check if Kaggle is configured
    if is_kaggle_configured():
        st.success("Kaggle credentials found")
        
        # Verify button
        if st.button("Verify Kaggle Setup"):
            verify_kaggle_setup()
        
        # Show search interface
        show_kaggle_search_interface()
    
    else:
        st.error("Kaggle credentials not found")
        show_kaggle_setup_instructions()
        
        # Manual dataset reference import
        st.subheader("Import by Reference")
        dataset_ref = st.text_input("Dataset reference (username/dataset-name)")
        
        if st.button("Import Dataset") and dataset_ref:
            import_kaggle_dataset(dataset_ref) 