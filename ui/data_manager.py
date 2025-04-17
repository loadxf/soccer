"""
Data Manager Module

This module provides a clean interface for managing datasets in the UI.
"""

import streamlit as st
import pandas as pd
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path if needed
script_dir = Path(__file__).resolve().parent  # ui directory
project_root = script_dir.parent  # project root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import DATA_DIR from config
try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback definition if import fails
    DATA_DIR = os.path.join(project_root, "data")

# Try to import kaggle_ui module
try:
    from ui.kaggle_ui import show_kaggle_setup_instructions, import_kaggle_dataset, verify_kaggle_setup
    import src.data.kaggle_manager as kaggle_manager
    KAGGLE_AVAILABLE = kaggle_manager.is_kaggle_configured()
except ImportError:
    KAGGLE_AVAILABLE = False

# Try to import dataset_registry and pipeline
try:
    from src.data.dataset_registry import (
        get_all_datasets,
        get_dataset,
        get_dataset_preview,
        save_uploaded_dataset,
        register_dataset,
        delete_dataset,
        update_dataset_status,
        batch_process_datasets
    )
    from src.data.pipeline import run_pipeline, download_football_data
    REGISTRY_AVAILABLE = True
except ImportError as e:
    REGISTRY_AVAILABLE = False
    print(f"Error importing dataset registry: {e}")

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_manager_ui')

class DataManager:
    """Class for managing datasets in the UI."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.registry_available = REGISTRY_AVAILABLE
        self.kaggle_available = KAGGLE_AVAILABLE
        
        # Define feature engineering configuration
        self.feature_config = {
            "window_sizes": [3, 5, 10],
            "include_team_stats": True,
            "include_opponent_stats": True,
            "include_time_features": True,
            "include_categorical_encoding": True,
            "fill_missing": "mean",
            "min_matches_required": 3,
            "handle_outliers": True,
            "outlier_threshold": 3.0
        }

    def get_all_datasets(self):
        """Get all available datasets."""
        if not self.registry_available:
            return []
        
        try:
            return get_all_datasets()
        except Exception as e:
            st.error(f"Error retrieving datasets: {e}")
            return []
    
    def get_dataset(self, dataset_id):
        """Get a specific dataset by ID."""
        if not self.registry_available:
            return None
        
        try:
            return get_dataset(dataset_id)
        except Exception as e:
            st.error(f"Error retrieving dataset {dataset_id}: {e}")
            return None
    
    def get_dataset_preview(self, dataset_id, rows=5):
        """Get a preview of a dataset."""
        if not self.registry_available:
            return pd.DataFrame()
        
        try:
            return get_dataset_preview(dataset_id, rows)
        except Exception as e:
            st.error(f"Error getting preview for dataset {dataset_id}: {e}")
            return pd.DataFrame({'Error': [str(e)]})
    
    def upload_file(self, uploaded_file):
        """Upload a file and register it as a dataset."""
        if not self.registry_available:
            st.error("Dataset registry is not available. File upload failed.")
            return None
        
        try:
            return save_uploaded_dataset(uploaded_file)
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            return None
    
    def import_from_kaggle(self, dataset_ref):
        """Import a dataset from Kaggle."""
        if not self.kaggle_available:
            st.error("Kaggle API is not configured. Please set up Kaggle credentials.")
            return None
        
        with st.spinner(f"Importing dataset from Kaggle: {dataset_ref}..."):
            result = import_kaggle_dataset(dataset_ref)
            
        if isinstance(result, dict) and result.get("status") == "success":
            st.success(f"Successfully imported dataset: {dataset_ref}")
            return result
        else:
            error_msg = result.get("message", "Unknown error") if result else "Failed to import dataset"
            st.error(f"Error importing dataset: {error_msg}")
            return None
    
    def delete_dataset(self, dataset_id):
        """Delete a dataset."""
        if not self.registry_available:
            st.error("Dataset registry is not available. Cannot delete dataset.")
            return False
        
        try:
            return delete_dataset(dataset_id)
        except Exception as e:
            st.error(f"Error deleting dataset {dataset_id}: {e}")
            return False
    
    def verify_kaggle_setup(self):
        """Verify Kaggle API setup."""
        from ui.kaggle_ui import verify_kaggle_setup
        return verify_kaggle_setup()
    
    def process_dataset(self, dataset_id, processing_type='basic'):
        """Process a dataset."""
        if not self.registry_available:
            st.error("Dataset registry is not available. Cannot process dataset.")
            return False
        
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            st.error(f"Dataset {dataset_id} not found.")
            return False
        
        try:
            from src.data.pipeline import run_pipeline
            
            with st.spinner(f"Processing dataset {dataset.get('name')}..."):
                result = run_pipeline(dataset_id, process_type=processing_type)
            
            if result:
                st.success(f"Successfully processed dataset: {dataset.get('name')}")
                return True
            else:
                st.error(f"Failed to process dataset: {dataset.get('name')}")
                return False
        except Exception as e:
            st.error(f"Error processing dataset: {str(e)}")
            logger.error(f"Error processing dataset {dataset_id}: {traceback.format_exc()}")
            return False
    
    def batch_process_datasets(self, dataset_ids, processing_type='basic'):
        """Process multiple datasets."""
        if not self.registry_available:
            st.error("Dataset registry is not available. Cannot process datasets.")
            return {}
        
        results = {}
        
        for dataset_id in dataset_ids:
            dataset = self.get_dataset(dataset_id)
            if not dataset:
                results[dataset_id] = False
                continue
                
            try:
                with st.spinner(f"Processing dataset {dataset.get('name')}..."):
                    from src.data.pipeline import run_pipeline
                    result = run_pipeline(dataset_id, process_type=processing_type)
                
                results[dataset_id] = result
            except Exception as e:
                st.error(f"Error processing dataset {dataset.get('name')}: {str(e)}")
                results[dataset_id] = False
        
        return results
    
    def generate_features(self, dataset_id, feature_type="match_features", config=None):
        """Generate features for a dataset."""
        if not self.registry_available:
            st.error("Dataset registry is not available. Cannot generate features.")
            return False
        
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            st.error(f"Dataset {dataset_id} not found.")
            return False
        
        # Setup paths using the imported DATA_DIR
        processed_dir = os.path.join(DATA_DIR, "processed", dataset_id)
        features_dir = os.path.join(DATA_DIR, "features", dataset_id)
        
        # Create output directory if it doesn't exist
        os.makedirs(features_dir, exist_ok=True)
        
        # Ensure processed data exists
        if not os.path.exists(processed_dir):
            st.error(f"No processed data found for dataset {dataset.get('name')}. Please process the dataset first.")
            return False
        
        try:
            # Use provided config or default
            if config is None:
                config = self.feature_config
                
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            progress_text.text(f"Finding data files for dataset {dataset.get('name')}...")
            progress_bar.progress(0.1)
            
            # For match features
            if feature_type == "match_features":
                # Find match data file
                match_files = [f for f in os.listdir(processed_dir) if "match" in f.lower() or "game" in f.lower()]
                if not match_files:
                    st.error(f"No match data files found in {processed_dir}")
                    return False
                
                # Load match data
                match_file = match_files[0]
                match_path = os.path.join(processed_dir, match_file)
                
                progress_text.text(f"Loading match data from {match_file}...")
                progress_bar.progress(0.2)
                
                # Import the function
                from src.data.features import generate_features_for_dataset
                
                output_path = os.path.join(features_dir, f"{feature_type}.csv")
                
                progress_text.text(f"Generating {feature_type} for {dataset.get('name')}...")
                progress_bar.progress(0.3)
                
                # Generate features
                success, message = generate_features_for_dataset(
                    match_path, 
                    output_path, 
                    feature_type=feature_type,
                    config=config,
                    show_progress=True
                )
                
                progress_bar.progress(1.0)
                
                if success:
                    progress_text.text(f"Successfully generated features: {message}")
                    # Update dataset status
                    update_dataset_status(dataset_id, "features_generated")
                    return True
                else:
                    st.error(f"Failed to generate features: {message}")
                    return False
                
            # For advanced soccer-specific features
            elif feature_type == "advanced_features":
                progress_text.text(f"Finding match and shot data files...")
                progress_bar.progress(0.2)
                
                try:
                    # Import soccer_features module
                    from src.data.soccer_features import load_or_create_advanced_features
                    
                    # Find match data file
                    match_files = [f for f in os.listdir(processed_dir) if "match" in f.lower() or "game" in f.lower()]
                    if not match_files:
                        st.error(f"No match data files found in {processed_dir}")
                        return False
                    
                    # Load match data
                    match_file = match_files[0]
                    match_path = os.path.join(processed_dir, match_file)
                    
                    progress_text.text(f"Loading match data from {match_file}...")
                    progress_bar.progress(0.3)
                    
                    matches_df = pd.read_csv(match_path)
                    
                    # Find shot data file if available
                    shot_files = [f for f in os.listdir(processed_dir) if "shot" in f.lower()]
                    shots_df = None
                    
                    if shot_files:
                        shot_file = shot_files[0]
                        shot_path = os.path.join(processed_dir, shot_file)
                        
                        progress_text.text(f"Loading shot data from {shot_file}...")
                        progress_bar.progress(0.4)
                        
                        shots_df = pd.read_csv(shot_path)
                    
                    # Generate advanced features
                    progress_text.text(f"Generating advanced soccer features...")
                    progress_bar.progress(0.5)
                    
                    features_df = load_or_create_advanced_features(matches_df, shots_df, config=config)
                    
                    # Save features
                    features_path = os.path.join(features_dir, f"{feature_type}.csv")
                    
                    progress_text.text(f"Saving features to {features_path}...")
                    progress_bar.progress(0.9)
                    
                    features_df.to_csv(features_path, index=False)
                    
                    # Save metadata
                    metadata = {
                        "timestamp": datetime.now().isoformat(),
                        "source_dataset": dataset_id,
                        "feature_type": feature_type,
                        "num_samples": len(features_df),
                        "num_features": len(features_df.columns),
                        "configuration": config
                    }
                    
                    metadata_path = os.path.join(features_dir, f"{feature_type}_metadata.json")
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    progress_bar.progress(1.0)
                    progress_text.text(f"Generated {len(features_df)} advanced soccer features")
                    
                    # Update dataset status
                    update_dataset_status(dataset_id, "features_generated")
                    return True
                    
                except ImportError as e:
                    st.error(f"Soccer features module not available: {e}")
                    return False
                except Exception as e:
                    st.error(f"Error generating advanced features: {e}")
                    logger.error(f"Error generating advanced features: {traceback.format_exc()}")
                    return False
            else:
                st.error(f"Unknown feature type: {feature_type}")
                return False
                
        except Exception as e:
            st.error(f"Error generating features: {str(e)}")
            logger.error(f"Error generating features for {dataset_id}: {traceback.format_exc()}")
            return False
    
    def batch_generate_features(self, dataset_ids, feature_type="match_features", config=None):
        """Generate features for multiple datasets."""
        if not self.registry_available:
            st.error("Dataset registry is not available. Cannot generate features.")
            return {}
        
        results = {}
        
        overall_progress = st.progress(0)
        dataset_progress = st.empty()
        
        for i, dataset_id in enumerate(dataset_ids):
            dataset = self.get_dataset(dataset_id)
            if not dataset:
                results[dataset_id] = False
                continue
            
            # Update overall progress
            overall_progress.progress((i) / len(dataset_ids))
            dataset_progress.text(f"Processing dataset {i+1}/{len(dataset_ids)}: {dataset.get('name', dataset_id)}")
            
            # Generate features for this dataset
            try:
                success = self.generate_features(dataset_id, feature_type, config)
                results[dataset_id] = success
            except Exception as e:
                st.error(f"Error processing dataset {dataset.get('name')}: {str(e)}")
                results[dataset_id] = False
        
        # Complete the progress bar
        overall_progress.progress(1.0)
        dataset_progress.text(f"Completed processing {len(dataset_ids)} datasets")
        
        return results
    
    def show_feature_engineering_ui(self, selected_datasets):
        """Show the feature engineering UI."""
        st.subheader("Feature Engineering")
        
        if not selected_datasets:
            st.warning("Please select at least one dataset for feature engineering.")
            return
        
        # Feature type selection
        feature_type = st.selectbox(
            "Select feature type:",
            options=[
                "match_features", 
                "team_form", 
                "player_form", 
                "advanced_features"
            ],
            help="Type of features to generate"
        )
        
        # Feature configuration
        with st.expander("Feature Configuration", expanded=False):
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Window sizes for rolling features
                window_sizes = st.multiselect(
                    "Window sizes for moving averages:",
                    options=[3, 5, 7, 10, 15, 20],
                    default=[3, 5, 10],
                    help="Number of past matches to consider for form features"
                )
                
                # Missing value handling
                fill_missing = st.selectbox(
                    "Handle missing values:",
                    options=["mean", "median", "zero", "none"],
                    index=0,
                    help="Strategy for filling missing numeric values"
                )
                
                # Min matches required
                min_matches = st.slider(
                    "Minimum matches required:",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Minimum number of matches required for feature calculation"
                )
            
            with col2:
                # Include team stats
                include_team_stats = st.checkbox(
                    "Include team statistics",
                    value=True,
                    help="Include team-level statistics in features"
                )
                
                # Include opponent stats
                include_opponent_stats = st.checkbox(
                    "Include opponent statistics",
                    value=True,
                    help="Include opponent-level statistics in features"
                )
                
                # Handle outliers
                handle_outliers = st.checkbox(
                    "Handle outliers",
                    value=True,
                    help="Detect and cap outliers in numeric features"
                )
                
                # Outlier threshold
                outlier_threshold = st.slider(
                    "Outlier threshold (standard deviations):",
                    min_value=1.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    help="Number of standard deviations to consider a value an outlier"
                )
        
        # Build configuration
        config = {
            "window_sizes": window_sizes,
            "include_team_stats": include_team_stats,
            "include_opponent_stats": include_opponent_stats,
            "include_time_features": True,
            "include_categorical_encoding": True,
            "fill_missing": fill_missing,
            "min_matches_required": min_matches,
            "handle_outliers": handle_outliers,
            "outlier_threshold": outlier_threshold
        }
        
        # Update the stored config
        self.feature_config = config
        
        # Execute button
        if st.button("Execute Feature Engineering"):
            st.markdown("### Processing Status")
            
            # Check if datasets have been processed
            datasets_to_process = []
            for dataset_id in selected_datasets:
                dataset = self.get_dataset(dataset_id)
                if not dataset:
                    st.error(f"Dataset {dataset_id} not found.")
                    continue
                
                # Check if dataset has been processed
                processed_dir = os.path.join(DATA_DIR, "processed", dataset_id)
                
                if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
                    st.warning(f"Dataset '{dataset.get('name')}' has not been processed. Processing it first...")
                    processing_result = self.process_dataset(dataset_id)
                    if processing_result:
                        datasets_to_process.append(dataset_id)
                    else:
                        st.error(f"Failed to process dataset '{dataset.get('name')}'. Skipping feature generation.")
                else:
                    datasets_to_process.append(dataset_id)
            
            if not datasets_to_process:
                st.error("No datasets available for feature generation after processing checks.")
                return
            
            # Generate features
            results = self.batch_generate_features(datasets_to_process, feature_type, config)
            
            # Show results
            st.markdown("### Feature Engineering Results")
            
            results_df = pd.DataFrame({
                "Dataset": [self.get_dataset(dataset_id).get('name', dataset_id) for dataset_id in results.keys()],
                "Success": [results[dataset_id] for dataset_id in results.keys()]
            })
            
            st.dataframe(results_df)
            
            # Provide download links for the generated features
            if any(results.values()):
                st.markdown("### Download Generated Features")
                
                for dataset_id, success in results.items():
                    if success:
                        dataset = self.get_dataset(dataset_id)
                        features_path = os.path.join(DATA_DIR, "features", dataset_id, f"{feature_type}.csv")
                        
                        if os.path.exists(features_path):
                            # Show feature preview
                            with st.expander(f"Preview: {dataset.get('name')} features"):
                                try:
                                    features_df = pd.read_csv(features_path)
                                    st.dataframe(features_df.head())
                                    st.text(f"Shape: {features_df.shape[0]} rows, {features_df.shape[1]} columns")
                                except Exception as e:
                                    st.error(f"Error loading feature preview: {e}")
            
            # Show notification
            if all(results.values()):
                st.success("Feature engineering completed successfully for all datasets!")
            elif any(results.values()):
                st.warning("Feature engineering completed with some failures. See results table for details.")
            else:
                st.error("Feature engineering failed for all datasets.")
    
# Function for main data management page
def show_data_management_page():
    """Show the data management page."""
    st.title("Data Management")
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Create tabs for different functions
    tabs = st.tabs([
        "Dataset Browser", 
        "Upload Data", 
        "Import from Kaggle", 
        "Data Processing", 
        "Feature Engineering"
    ])
    
    # Dataset Browser Tab
    with tabs[0]:
        st.subheader("Available Datasets")
        
        # Get all datasets
        datasets = data_manager.get_all_datasets()
        
        if not datasets:
            st.info("No datasets available. Please upload or import data.")
        else:
            # Display datasets in a selectbox with additional info
            selected_dataset = st.selectbox(
                "Select a dataset to view:", 
                options=[ds.get('id') for ds in datasets],
                format_func=lambda x: f"{next((ds.get('name', 'Unknown') for ds in datasets if ds.get('id') == x), 'Unknown')} (ID: {x})"
            )
            
            if selected_dataset:
                # Get the dataset and show details
                dataset = data_manager.get_dataset(selected_dataset)
                
                if dataset:
                    # Display basic info
                    st.markdown(f"### {dataset.get('name', 'Unknown')}")
                    st.markdown(f"**ID**: {dataset.get('id')}")
                    st.markdown(f"**Status**: {dataset.get('status', 'raw')}")
                    st.markdown(f"**Upload Date**: {dataset.get('upload_date', 'Unknown')}")
                    
                    # Display file info
                    st.markdown("#### File Information")
                    st.markdown(f"**Path**: `{dataset.get('path', 'Unknown')}`")
                    
                    # Show preview
                    st.markdown("#### Data Preview")
                    preview = data_manager.get_dataset_preview(selected_dataset)
                    
                    if isinstance(preview, pd.DataFrame):
                        st.dataframe(preview)
                    else:
                        st.info("Preview not available")
                    
                    # Actions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Delete Dataset"):
                            if data_manager.delete_dataset(selected_dataset):
                                st.success(f"Dataset {dataset.get('name')} deleted!")
                                st.experimental_rerun()
                            else:
                                st.error("Failed to delete dataset")
                    
                    with col2:
                        if st.button("Add to Selection"):
                            if 'selected_datasets' not in st.session_state:
                                st.session_state['selected_datasets'] = []
                            
                            if selected_dataset not in st.session_state['selected_datasets']:
                                st.session_state['selected_datasets'].append(selected_dataset)
                                st.success(f"Added {dataset.get('name')} to selection")
                            else:
                                st.info(f"{dataset.get('name')} is already in the selection")
                
                # Show selected datasets
                if 'selected_datasets' in st.session_state and st.session_state['selected_datasets']:
                    st.markdown("### Currently Selected Datasets")
                    
                    # Convert IDs to names
                    selected_names = [
                        next((ds.get('name', 'Unknown') for ds in datasets if ds.get('id') == dataset_id), dataset_id)
                        for dataset_id in st.session_state['selected_datasets']
                    ]
                    
                    # Display as a table
                    selected_df = pd.DataFrame({
                        "Dataset": selected_names,
                        "ID": st.session_state['selected_datasets']
                    })
                    
                    st.dataframe(selected_df)
                    
                    if st.button("Clear Selection"):
                        st.session_state['selected_datasets'] = []
                        st.success("Selection cleared")
                        st.experimental_rerun()
    
    # Upload Data Tab
    with tabs[1]:
        st.subheader("Upload Dataset")
        
        uploaded_file = st.file_uploader("Choose a file to upload:", type=['csv', 'xlsx', 'json'])
        
        if uploaded_file is not None:
            # Show file info
            st.info(f"File: {uploaded_file.name}, Size: {uploaded_file.size} bytes")
            
            # Upload button
            if st.button("Upload Dataset"):
                with st.spinner("Uploading dataset..."):
                    dataset_id = data_manager.upload_file(uploaded_file)
                
                if dataset_id:
                    st.success(f"Dataset uploaded successfully with ID: {dataset_id}")
                else:
                    st.error("Failed to upload dataset")
    
    # Import from Kaggle Tab
    with tabs[2]:
        st.subheader("Import from Kaggle")
        
        if not data_manager.kaggle_available:
            st.warning("Kaggle API is not configured or available.")
            
            # Show setup instructions
            show_kaggle_setup_instructions()
        else:
            # Kaggle dataset reference
            dataset_ref = st.text_input(
                "Enter Kaggle dataset reference (e.g., 'username/dataset-name'):",
                help="You can find this in the URL of the dataset on Kaggle"
            )
            
            # Import button
            if dataset_ref and st.button("Import Dataset"):
                # Import dataset from Kaggle
                result = data_manager.import_from_kaggle(dataset_ref)
                
                if isinstance(result, dict) and result.get("status") == "success":
                    st.success(f"Successfully imported {dataset_ref}")
                    
                    # Add to selection
                    if 'registry_result' in result and 'selected_datasets' not in st.session_state:
                        st.session_state['selected_datasets'] = []
                    
                    if 'registry_result' in result and result['registry_result'] not in st.session_state.get('selected_datasets', []):
                        st.session_state['selected_datasets'].append(result['registry_result'])
                        st.info(f"Added imported dataset to selection")
                else:
                    st.error(f"Failed to import dataset from Kaggle")
    
    # Data Processing Tab
    with tabs[3]:
        st.subheader("Data Processing")
        
        if 'selected_datasets' not in st.session_state or not st.session_state['selected_datasets']:
            st.warning("Please select at least one dataset for processing.")
        else:
            # Processing type
            processing_type = st.selectbox(
                "Select processing type:",
                options=["basic", "clean", "normalize", "advanced"],
                help="Type of processing to apply to the selected datasets"
            )
            
            # Process button
            if st.button("Process Selected Datasets"):
                with st.spinner("Processing datasets..."):
                    results = data_manager.batch_process_datasets(
                        st.session_state['selected_datasets'],
                        processing_type=processing_type
                    )
                
                # Show results
                st.markdown("#### Processing Results")
                
                for dataset_id, result in results.items():
                    dataset = data_manager.get_dataset(dataset_id)
                    if result:
                        st.success(f"Successfully processed {dataset.get('name', dataset_id)}")
                    else:
                        st.error(f"Failed to process {dataset.get('name', dataset_id)}")
    
    # Feature Engineering Tab  
    with tabs[4]:
        data_manager.show_feature_engineering_ui(
            st.session_state.get('selected_datasets', [])
        )

if __name__ == "__main__":
    show_data_management_page() 