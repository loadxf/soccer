"""
Kaggle UI Helper Module

This module provides Streamlit UI components for Kaggle integration.
"""

import streamlit as st
import sys
from pathlib import Path

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
    
    # Display instructions directly without using an expander
    st.markdown("""
    ### Setting up Kaggle API access
    
    1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com) if you don't have one
    2. Go to your Kaggle account settings (click on your profile picture → Account)
    3. Scroll down to the API section and click "Create New API Token"
    4. This will download a `kaggle.json` file with your credentials
    5. Place this file in the `~/.kaggle/` directory:
       - Windows: `C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json`
       - Linux/Mac: `~/.kaggle/kaggle.json`
    6. Install the Kaggle API package:
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
                show_kaggle_setup_instructions()
                return
            
            # Import the dataset
            result = import_dataset(dataset_ref)
            
            if result['status'] == "success":
                st.success(f"Successfully imported {dataset_ref}")
                st.json({
                    "path": result['path'],
                    "files": result['files']
                })
                return True
            else:
                st.error(f"Error importing dataset: {result['message']}")
                return False
        
        except Exception as e:
            st.error(f"Error importing dataset: {str(e)}")
            return False

def verify_kaggle_setup():
    """Test Kaggle authentication and display the result."""
    with st.spinner("Verifying Kaggle setup..."):
        try:
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