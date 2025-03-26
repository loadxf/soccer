"""
Data Manager Module

This module provides a clean interface for managing datasets in the UI.
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path if needed
script_dir = Path(__file__).resolve().parent  # ui directory
project_root = script_dir.parent  # project root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

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

class DataManager:
    """Manager for dataset operations in the UI."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.kaggle_available = KAGGLE_AVAILABLE
        self.registry_available = REGISTRY_AVAILABLE
    
    def upload_file(self, uploaded_file):
        """
        Upload and register a file.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            str: Dataset ID or None if failed
        """
        if not self.registry_available:
            st.error("❌ Dataset registry not available")
            return None
        
        try:
            dataset_id = save_uploaded_dataset(uploaded_file)
            if dataset_id:
                st.success(f"✅ Successfully uploaded {uploaded_file.name}")
                return dataset_id
            else:
                st.error(f"❌ Failed to upload {uploaded_file.name}")
                return None
        except Exception as e:
            st.error(f"❌ Error uploading file: {str(e)}")
            return None
    
    def import_kaggle_dataset(self, dataset_ref):
        """
        Import a dataset from Kaggle.
        
        Args:
            dataset_ref (str): Dataset reference (username/dataset-name)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.kaggle_available:
            st.error("❌ Kaggle credentials not found")
            show_kaggle_setup_instructions()
            return False
        
        try:
            return import_kaggle_dataset(dataset_ref)
        except Exception as e:
            st.error(f"❌ Error importing Kaggle dataset: {str(e)}")
            return False
    
    def get_all_datasets(self):
        """
        Get all available datasets.
        
        Returns:
            list: List of dataset dictionaries
        """
        if not self.registry_available:
            return []
        
        try:
            return get_all_datasets()
        except Exception as e:
            st.error(f"❌ Error getting datasets: {str(e)}")
            return []
    
    def get_dataset_preview(self, dataset_id, rows=5):
        """
        Get a preview of a dataset.
        
        Args:
            dataset_id (str): Dataset ID
            rows (int): Number of rows to preview
            
        Returns:
            DataFrame: Preview of the dataset or None if error
        """
        if not self.registry_available:
            return None
        
        try:
            return get_dataset_preview(dataset_id, rows)
        except Exception as e:
            st.error(f"❌ Error getting dataset preview: {str(e)}")
            return None
    
    def delete_dataset(self, dataset_id):
        """
        Delete a dataset.
        
        Args:
            dataset_id (str): Dataset ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.registry_available:
            st.error("❌ Dataset registry not available")
            return False
        
        try:
            success = delete_dataset(dataset_id)
            if success:
                st.success(f"✅ Dataset deleted successfully")
            else:
                st.error(f"❌ Failed to delete dataset")
            return success
        except Exception as e:
            st.error(f"❌ Error deleting dataset: {str(e)}")
            return False
    
    def process_dataset(self, dataset_id, process_type="process"):
        """
        Process a dataset.
        
        Args:
            dataset_id (str): Dataset ID
            process_type (str): Type of processing
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.registry_available:
            st.error("❌ Dataset registry not available")
            return False
        
        try:
            with st.spinner(f"Processing dataset {dataset_id}..."):
                success = run_pipeline(dataset_id, process_type)
                if success:
                    st.success(f"✅ Dataset processed successfully")
                else:
                    st.error(f"❌ Failed to process dataset")
                return success
        except Exception as e:
            st.error(f"❌ Error processing dataset: {str(e)}")
            return False
    
    def batch_process_datasets(self, dataset_ids, process_type="process"):
        """
        Process multiple datasets.
        
        Args:
            dataset_ids (list): List of dataset IDs
            process_type (str): Type of processing
            
        Returns:
            dict: Results for each dataset ID
        """
        if not self.registry_available:
            st.error("❌ Dataset registry not available")
            return {}
        
        try:
            # Define a processing function for batch processing
            def process_dataset(dataset_id, **kwargs):
                return run_pipeline(dataset_id, process_type=process_type)
            
            # Run batch processing
            with st.spinner(f"Processing {len(dataset_ids)} datasets..."):
                results = batch_process_datasets(dataset_ids, process_dataset)
                
                # Count successes
                success_count = sum(1 for status in results.values() if status)
                
                if success_count > 0:
                    st.success(f"✅ Successfully processed {success_count}/{len(results)} datasets")
                else:
                    st.error("❌ Failed to process datasets")
                
                return results
        except Exception as e:
            st.error(f"❌ Error during batch processing: {str(e)}")
            return {}
    
    def download_football_data(self, leagues=None, seasons=None):
        """
        Download football data.
        
        Args:
            leagues (list): List of leagues (e.g., ["Premier League", "La Liga"])
            seasons (list): List of seasons in UI format (e.g., ["2022/23", "2023/24"])
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.registry_available:
            st.error("❌ Dataset registry not available")
            return False
        
        try:
            with st.spinner("Downloading football data..."):
                # Convert UI league names to backend codes
                league_mapping = {
                    "Premier League": "E0",
                    "Championship": "E1",
                    "La Liga": "SP1",
                    "Bundesliga": "D1",
                    "Serie A": "I1",
                    "Ligue 1": "F1"
                }
                
                league_codes = [league_mapping.get(league, league) for league in leagues] if leagues else None
                
                # Convert UI season format (2022/23) to backend format (20222023)
                backend_seasons = []
                if seasons:
                    for season in seasons:
                        if '/' in season:
                            # UI format: "2022/23" -> Backend format: "20222023"
                            parts = season.split('/')
                            if len(parts) == 2:
                                start_year = parts[0]
                                end_year_short = parts[1]
                                # Handle century change if needed
                                if len(start_year) == 4 and len(end_year_short) == 2:
                                    end_year = f"20{end_year_short}"
                                else:
                                    end_year = end_year_short
                                
                                # Special case for 2024/25 season
                                if start_year == "2024" and end_year_short == "25":
                                    backend_seasons.append("2425")  # Use format expected by football-data.co.uk
                                else:
                                    backend_seasons.append(f"{start_year}{end_year}")
                        else:
                            # Already in backend format
                            backend_seasons.append(season)
                
                # Debug info
                st.info(f"Downloading data for leagues: {league_codes}, seasons: {backend_seasons}")
                
                # Call the backend function
                from src.data.pipeline import download_football_data
                success = download_football_data("football_data", custom_seasons=backend_seasons, custom_leagues=league_codes)
                
                if success:
                    # Get the list of all datasets to show the newly added ones
                    all_datasets = self.get_all_datasets()
                    
                    # Filter to find datasets we just downloaded (using source and league codes)
                    new_datasets = [
                        ds for ds in all_datasets 
                        if ds.get("source") == "football-data.co.uk" and
                           ds.get("status") == "raw" and
                           (not league_codes or ds.get("league") in league_codes) and
                           (not backend_seasons or ds.get("season") in backend_seasons)
                    ]
                    
                    if new_datasets:
                        st.success(f"✅ Successfully downloaded {len(new_datasets)} football datasets")
                        
                        # Show preview of the first dataset
                        if len(new_datasets) > 0:
                            with st.expander("Preview Downloaded Data"):
                                preview_id = new_datasets[0]["id"]
                                preview_df = self.get_dataset_preview(preview_id)
                                if preview_df is not None:
                                    st.dataframe(preview_df)
                                    st.caption(f"Dataset: {new_datasets[0]['name']}")
                                else:
                                    st.error("Could not load preview for this dataset")
                    else:
                        st.warning("✅ Football data downloaded successfully, but no new datasets were found in the registry.")
                else:
                    st.error("❌ Failed to download football data")
                return success
        except Exception as e:
            st.error(f"❌ Error downloading football data: {str(e)}")
            return False
    
    def show_kaggle_setup_instructions(self):
        """Show Kaggle setup instructions."""
        if hasattr(self, '_show_instructions'):
            self._show_instructions()
        else:
            show_kaggle_setup_instructions()
    
    def verify_kaggle_setup(self):
        """
        Verify Kaggle setup.
        
        Returns:
            bool: True if setup is valid, False otherwise
        """
        if not self.kaggle_available:
            st.error("❌ Kaggle credentials not found")
            return False
        
        try:
            return verify_kaggle_setup()
        except Exception as e:
            st.error(f"❌ Error verifying Kaggle setup: {str(e)}")
            return False

def generate_features(dataset_id, feature_type="match_features"):
    """
    Generate features for a dataset.
    
    Args:
        dataset_id: Dataset ID
        feature_type: Type of features to generate
        
    Returns:
        bool: Success status
    """
    try:
        # Verify dataset exists
        dataset_info = get_dataset_info(dataset_id)
        if not dataset_info:
            logger.error(f"Dataset {dataset_id} not found")
            return False
        
        # Get data directory
        try:
            from config.default_config import DATA_DIR
        except ImportError:
            DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent, "data")
        
        processed_dir = os.path.join(DATA_DIR, "processed", dataset_id)
        features_dir = os.path.join(DATA_DIR, "features", dataset_id)
        
        # Create features directory if it doesn't exist
        os.makedirs(features_dir, exist_ok=True)
        
        # Generate features based on type
        if feature_type == "match_features":
            # Basic match features
            logger.info(f"Generating match features for dataset {dataset_id}")
            
            # Find match data file
            match_files = [f for f in os.listdir(processed_dir) if "match" in f.lower() or "game" in f.lower()]
            if not match_files:
                logger.error(f"No match data files found in {processed_dir}")
                return False
            
            # Load match data
            match_file = match_files[0]
            match_path = os.path.join(processed_dir, match_file)
            matches_df = pd.read_csv(match_path)
            
            # Generate basic match features
            from src.data.features import create_match_features
            features_df = create_match_features(matches_df)
            
            # Save features
            features_path = os.path.join(features_dir, f"{feature_type}.csv")
            features_df.to_csv(features_path, index=False)
            
            logger.info(f"Generated {len(features_df)} match features and saved to {features_path}")
            
            return True
        
        elif feature_type == "team_features":
            # Team-level features
            logger.info(f"Generating team features for dataset {dataset_id}")
            
            # Find match data file
            match_files = [f for f in os.listdir(processed_dir) if "match" in f.lower() or "game" in f.lower()]
            if not match_files:
                logger.error(f"No match data files found in {processed_dir}")
                return False
            
            # Load match data
            match_file = match_files[0]
            match_path = os.path.join(processed_dir, match_file)
            matches_df = pd.read_csv(match_path)
            
            # Generate team features
            from src.data.features import create_team_features
            features_df = create_team_features(matches_df)
            
            # Save features
            features_path = os.path.join(features_dir, f"{feature_type}.csv")
            features_df.to_csv(features_path, index=False)
            
            logger.info(f"Generated {len(features_df)} team features and saved to {features_path}")
            
            return True
        
        elif feature_type == "advanced_features":
            # Advanced soccer-specific features
            logger.info(f"Generating advanced soccer features for dataset {dataset_id}")
            
            try:
                # First check if soccer_features module is available
                from src.data.soccer_features import load_or_create_advanced_features
                
                # Find match data file
                match_files = [f for f in os.listdir(processed_dir) if "match" in f.lower() or "game" in f.lower()]
                if not match_files:
                    logger.error(f"No match data files found in {processed_dir}")
                    return False
                
                # Load match data
                match_file = match_files[0]
                match_path = os.path.join(processed_dir, match_file)
                matches_df = pd.read_csv(match_path)
                
                # Find shot data file if available
                shot_files = [f for f in os.listdir(processed_dir) if "shot" in f.lower()]
                shots_df = None
                
                if shot_files:
                    shot_file = shot_files[0]
                    shot_path = os.path.join(processed_dir, shot_file)
                    shots_df = pd.read_csv(shot_path)
                
                # Generate advanced features
                features_df = load_or_create_advanced_features(matches_df, shots_df)
                
                # Save features
                features_path = os.path.join(features_dir, f"{feature_type}.csv")
                features_df.to_csv(features_path, index=False)
                
                logger.info(f"Generated {len(features_df)} advanced soccer features and saved to {features_path}")
                
                return True
            
            except ImportError:
                logger.error("Advanced soccer features module not available")
                return False
            
            except Exception as e:
                logger.error(f"Error generating advanced features: {e}")
                return False
        
        else:
            logger.error(f"Unsupported feature type: {feature_type}")
            return False
    
    except Exception as e:
        logger.error(f"Error generating features: {e}")
        return False 