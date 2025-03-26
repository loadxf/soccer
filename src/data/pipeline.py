"""
Data Pipeline Module

This module provides functions for processing datasets through various pipeline stages.
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from datetime import datetime
import sys
import logging
import urllib.request
import subprocess

# Import project components
from src.utils.logger import get_logger

# Delay Kaggle import to prevent immediate authentication errors
kaggle = None

try:
    from config.default_config import DATA_DIR, KAGGLE_USERNAME, KAGGLE_KEY
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

try:
    from src.data.dataset_registry import (
        get_dataset,
        update_dataset_status,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False

# Setup logger
logger = get_logger("data.pipeline")

# Ensure data directories exist
os.makedirs(DATA_DIR, exist_ok=True)
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DATA_DIR = os.path.join(DATA_DIR, "interim")
FEATURES_DATA_DIR = os.path.join(DATA_DIR, "features")
AUGMENTED_DATA_DIR = os.path.join(DATA_DIR, "augmented")
USER_UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(INTERIM_DATA_DIR, exist_ok=True)
os.makedirs(FEATURES_DATA_DIR, exist_ok=True)
os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)
os.makedirs(USER_UPLOADS_DIR, exist_ok=True)

# Dataset definitions
DATASETS = {
    "transfermarkt": {
        "kaggle_dataset": "davidcariboo/player-scores",
        "description": "Player scores and transfer values from Transfermarkt",
        "files": ["players.csv", "clubs.csv", "games.csv", "appearances.csv", "competitions.csv"],
        "raw_dir": os.path.join(RAW_DATA_DIR, "transfermarkt"),
    },
    "statsbomb": {
        "kaggle_dataset": "hugomathien/soccer",
        "description": "European Soccer Database with +25,000 matches, players & teams attributes",
        "files": ["database.sqlite"],
        "raw_dir": os.path.join(RAW_DATA_DIR, "statsbomb"),
    },
    "fifa": {
        "kaggle_dataset": "stefanoleone992/fifa-22-complete-player-dataset",
        "description": "FIFA 22 complete player dataset",
        "files": ["players_22.csv", "teams_22.csv"],
        "raw_dir": os.path.join(RAW_DATA_DIR, "fifa"),
    },
    "football_data": {
        "url": "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv",
        "description": "Football Results, Statistics & Soccer Betting Odds Data",
        "files": [],  # Will be determined dynamically based on seasons and leagues
        "raw_dir": os.path.join(RAW_DATA_DIR, "football_data"),
        "seasons": [f"{year}{year+1}" for year in range(2010, 2024)] + ["2425"],  # Includes special case for 2024-2025
        "leagues": ["E0", "E1", "SP1", "D1", "I1", "F1"],  # Premier League, Championship, La Liga, Bundesliga, Serie A, Ligue 1
    }
}


def download_kaggle_dataset(dataset_name: str) -> bool:
    """
    Download a dataset from Kaggle.
    
    Args:
        dataset_name: Name of the dataset in the DATASETS dictionary
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    global kaggle
    
    if dataset_name not in DATASETS:
        logger.error(f"Dataset {dataset_name} not found in dataset definitions")
        return False
    
    dataset_info = DATASETS[dataset_name]
    
    if "kaggle_dataset" not in dataset_info:
        logger.error(f"Dataset {dataset_name} does not have a Kaggle dataset defined")
        return False
    
    kaggle_dataset = dataset_info["kaggle_dataset"]
    raw_dir = dataset_info["raw_dir"]
    
    # Ensure raw directory exists
    os.makedirs(raw_dir, exist_ok=True)
    
    try:
        # Import kaggle on-demand to delay authentication until needed
        if kaggle is None:
            try:
                import kaggle as kaggle_module
                kaggle = kaggle_module
            except ImportError:
                logger.error("Kaggle module not installed. Please install with: pip install kaggle")
                return False
        
        # Set Kaggle credentials if they're not in ~/.kaggle/kaggle.json
        if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
            if not KAGGLE_USERNAME or not KAGGLE_KEY:
                logger.error("Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY in environment variables or config.")
                logger.error("For more information: https://github.com/Kaggle/kaggle-api#api-credentials")
                return False
            
            # Create the .kaggle directory if it doesn't exist
            os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
            
            # Create credentials file
            try:
                with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
                    json.dump({
                        "username": KAGGLE_USERNAME,
                        "key": KAGGLE_KEY
                    }, f)
                # Set appropriate permissions
                os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
            except Exception as e:
                logger.error(f"Error creating kaggle.json file: {e}")
                return False
        
        # Log the action
        logger.info(f"Downloading Kaggle dataset: {kaggle_dataset}")
        
        # Authenticate and download
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                kaggle_dataset,
                path=raw_dir,
                unzip=True,
                quiet=False
            )
        except Exception as e:
            logger.error(f"Error in Kaggle API: {e}")
            # Check if this is a credentials error
            if "could not find" in str(e).lower() and "kaggle.json" in str(e).lower():
                logger.error("This appears to be a Kaggle credentials issue.")
                logger.error("Please ensure your Kaggle API credentials are set up correctly.")
                logger.error("See: https://github.com/Kaggle/kaggle-api#api-credentials")
            return False
        
        # Verify that the expected files exist
        for file in dataset_info["files"]:
            file_path = os.path.join(raw_dir, file)
            if not os.path.exists(file_path):
                logger.warning(f"Expected file {file} not found after download")
        
        logger.info(f"Successfully downloaded {dataset_name} dataset to {raw_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading Kaggle dataset {kaggle_dataset}: {e}")
        return False


def download_football_data(dataset_name: str = "football_data", custom_seasons: Optional[List[str]] = None, custom_leagues: Optional[List[str]] = None) -> bool:
    """
    Download soccer data from football-data.co.uk using the FootballDataAPI.
    
    Args:
        dataset_name: Name of the dataset in the DATASETS dictionary
        custom_seasons: Optional list of seasons to download (overrides defaults)
        custom_leagues: Optional list of leagues to download (overrides defaults)
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    if dataset_name != "football_data":
        logger.error(f"Invalid dataset name for football_data: {dataset_name}")
        return False
    
    try:
        # Import the FootballDataAPI
        from src.data.football_api_manager import FootballDataAPI
        
        # Create the API client
        football_api = FootballDataAPI()
        
        # Determine which seasons and leagues to download
        dataset_info = DATASETS[dataset_name]
        seasons = custom_seasons if custom_seasons is not None else dataset_info["seasons"]
        leagues = custom_leagues if custom_leagues is not None else dataset_info["leagues"]
        
        logger.info(f"Downloading {len(seasons)} seasons and {len(leagues)} leagues from football-data.co.uk")
        
        # Download the data
        result = football_api.fetch_all_seasons(seasons=seasons, leagues=leagues)
        
        # Check the results
        if result["status"] == "error":
            logger.error("Failed to download any football data")
            return False
        
        logger.info(f"Downloaded {result['total_files']} files from football-data.co.uk")
        
        # Update the dynamic file list in the dataset info
        DATASETS[dataset_name]["files"] = []
        
        # Scan the raw directory to find all downloaded files
        raw_dir = dataset_info["raw_dir"]
        for season in os.listdir(raw_dir):
            season_dir = os.path.join(raw_dir, season)
            if os.path.isdir(season_dir):
                for league_file in os.listdir(season_dir):
                    if league_file.endswith('.csv'):
                        relative_path = os.path.join(season, league_file)
                        DATASETS[dataset_name]["files"].append(relative_path)
        
        # Register the downloaded data in the dataset registry (if available)
        registered_datasets = []
        if REGISTRY_AVAILABLE:
            try:
                from src.data.dataset_registry import register_dataset
                
                # Register each season/league combination as a separate dataset
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                
                for season in seasons:
                    season_dir = os.path.join(raw_dir, season)
                    if os.path.isdir(season_dir):
                        for league_file in os.listdir(season_dir):
                            if league_file.endswith('.csv'):
                                league_code = league_file.replace('.csv', '')
                                league_name = ""
                                
                                # Get the league name if available
                                try:
                                    from src.data.football_api_manager import AVAILABLE_LEAGUES
                                    league_name = AVAILABLE_LEAGUES.get(league_code, "Unknown")
                                except ImportError:
                                    league_name = league_code
                                
                                file_path = os.path.join(season_dir, league_file)
                                
                                # Read basic info from the file
                                try:
                                    df = pd.read_csv(file_path)
                                    rows = len(df)
                                    columns = df.columns.tolist()
                                except Exception as e:
                                    logger.warning(f"Error reading CSV for registration: {e}")
                                    rows = 0
                                    columns = []
                                
                                # Register dataset
                                dataset_info = {
                                    "name": f"Football Data - {league_name} {season} ({timestamp})",
                                    "description": f"Soccer match data for {league_name} ({league_code}) in season {season}",
                                    "source": "football-data.co.uk",
                                    "filename": league_file,
                                    "path": file_path,
                                    "upload_date": datetime.now().isoformat(),
                                    "rows": rows,
                                    "columns": columns,
                                    "status": "raw",
                                    "file_size": os.path.getsize(file_path),
                                    "dataset_type": "football_data",
                                    "season": season,
                                    "league": league_code,
                                    "league_name": league_name
                                }
                                
                                dataset_id = register_dataset(dataset_info)
                                registered_datasets.append(dataset_id)
                                logger.info(f"Registered dataset {dataset_id}: {dataset_info['name']}")
            except ImportError as e:
                logger.warning(f"Dataset registry not available for registration: {e}")
            except Exception as e:
                logger.error(f"Error registering datasets: {e}")
        
        if result["status"] == "partial":
            logger.warning(f"Partial success: {len(result['errors'])} errors occurred during download")
            for error in result["errors"][:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")
            if len(result["errors"]) > 5:
                logger.warning(f"  - and {len(result['errors']) - 5} more errors")
        
        # Return success (with at least one downloaded file)
        return result["total_files"] > 0
        
    except ImportError:
        logger.error("Failed to import FootballDataAPI. Falling back to legacy download method.")
        
        # Fall back to the legacy download method
        dataset_info = DATASETS[dataset_name]
        raw_dir = dataset_info["raw_dir"]
        
        # Use custom seasons/leagues if provided
        seasons = custom_seasons if custom_seasons is not None else dataset_info["seasons"]
        leagues = custom_leagues if custom_leagues is not None else dataset_info["leagues"]
        
        # Ensure raw directory exists
        os.makedirs(raw_dir, exist_ok=True)
        
        success_count = 0
        total_files = len(seasons) * len(leagues)
        
        try:
            for season in seasons:
                season_dir = os.path.join(raw_dir, season)
                os.makedirs(season_dir, exist_ok=True)
                
                for league in leagues:
                    url = dataset_info["url"].format(season=season, league=league)
                    filename = f"{league}.csv"
                    file_path = os.path.join(season_dir, filename)
                    
                    try:
                        logger.info(f"Downloading {url} to {file_path}")
                        response = requests.get(url, timeout=30)
                        
                        if response.status_code == 200:
                            with open(file_path, 'wb') as f:
                                f.write(response.content)
                            logger.info(f"Successfully downloaded {filename} for season {season}")
                            success_count += 1
                        else:
                            logger.warning(f"Failed to download {filename} for season {season}: HTTP {response.status_code}")
                            
                    except Exception as e:
                        logger.error(f"Error downloading {url}: {e}")
            
            success_rate = success_count / total_files if total_files > 0 else 0
            logger.info(f"Downloaded {success_count}/{total_files} files ({success_rate:.1%} success rate)")
            
            # Add the downloaded files to the dataset info
            DATASETS[dataset_name]["files"] = []
            registered_datasets = []
            
            for season in seasons:
                for league in leagues:
                    file_path_rel = os.path.join(season, f"{league}.csv")
                    file_path_abs = os.path.join(raw_dir, file_path_rel)
                    if os.path.exists(file_path_abs):
                        DATASETS[dataset_name]["files"].append(file_path_rel)
                        
                        # Register the dataset if registry is available
                        if REGISTRY_AVAILABLE:
                            try:
                                from src.data.dataset_registry import register_dataset
                                
                                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                                league_name = league  # Default to code if no mapping available
                                
                                # Read basic info from the file
                                try:
                                    df = pd.read_csv(file_path_abs)
                                    rows = len(df)
                                    columns = df.columns.tolist()
                                except Exception as e:
                                    logger.warning(f"Error reading CSV for registration: {e}")
                                    rows = 0
                                    columns = []
                                
                                # Register dataset
                                dataset_info = {
                                    "name": f"Football Data - {league_name} {season} ({timestamp})",
                                    "description": f"Soccer match data for {league_name} in season {season}",
                                    "source": "football-data.co.uk",
                                    "filename": f"{league}.csv",
                                    "path": file_path_abs,
                                    "upload_date": datetime.now().isoformat(),
                                    "rows": rows,
                                    "columns": columns,
                                    "status": "raw",
                                    "file_size": os.path.getsize(file_path_abs),
                                    "dataset_type": "football_data",
                                    "season": season,
                                    "league": league
                                }
                                
                                dataset_id = register_dataset(dataset_info)
                                registered_datasets.append(dataset_id)
                                logger.info(f"Registered dataset {dataset_id}: {dataset_info['name']}")
                            except ImportError as e:
                                logger.warning(f"Dataset registry not available for registration: {e}")
                            except Exception as e:
                                logger.error(f"Error registering datasets: {e}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error downloading football data: {e}")
            return False


def download_dataset(dataset_name: str, force_refresh: bool = False) -> bool:
    """
    Download a dataset by name.
    This is a simplified interface for the UI to download datasets.
    
    Args:
        dataset_name: Name of the dataset to download
        force_refresh: Whether to force download even if dataset exists
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Early validation
    if dataset_name not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(DATASETS.keys())}")
        return False
    
    dataset_info = DATASETS.get(dataset_name, {})
    
    # Handle different dataset types
    if "kaggle_dataset" in dataset_info:
        # This is a Kaggle dataset
        return download_kaggle_dataset(dataset_name)
    elif dataset_name == "football_data":
        # Football-data.co.uk dataset
        return download_football_data(dataset_name)
    else:
        logger.error(f"Unsupported dataset type for {dataset_name}")
        return False


def process_transfermarkt_data() -> bool:
    """
    Process the Transfermarkt dataset.
    
    Returns:
        bool: True if processing was successful, False otherwise
    """
    dataset_name = "transfermarkt"
    dataset_info = DATASETS[dataset_name]
    raw_dir = dataset_info["raw_dir"]
    processed_dir = os.path.join(PROCESSED_DATA_DIR, dataset_name)
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        logger.info(f"Processing {dataset_name} dataset")
        
        # Load the raw data
        players_df = pd.read_csv(os.path.join(raw_dir, "players.csv"))
        clubs_df = pd.read_csv(os.path.join(raw_dir, "clubs.csv"))
        games_df = pd.read_csv(os.path.join(raw_dir, "games.csv"))
        appearances_df = pd.read_csv(os.path.join(raw_dir, "appearances.csv"))
        competitions_df = pd.read_csv(os.path.join(raw_dir, "competitions.csv"))
        
        # Basic data cleaning
        players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], errors='coerce')
        players_df['current_value'] = players_df['current_value'].str.replace('€', '').str.replace('m', '000000').str.replace('k', '000').astype(float)
        
        # Process games data
        games_df['date'] = pd.to_datetime(games_df['date'], errors='coerce')
        
        # Join competitions to games
        games_comp_df = games_df.merge(competitions_df, on='competition_id', how='left')
        
        # Process appearances data
        appearances_df = appearances_df.merge(games_df[['game_id', 'date', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']], 
                                             on='game_id', how='left')
        
        # Add result (win/loss/draw) column
        appearances_df['result'] = 'draw'
        appearances_df.loc[((appearances_df['home_club_id'] == appearances_df['player_club_id']) & 
                            (appearances_df['home_club_goals'] > appearances_df['away_club_goals'])) | 
                           ((appearances_df['away_club_id'] == appearances_df['player_club_id']) & 
                            (appearances_df['away_club_goals'] > appearances_df['home_club_goals'])), 'result'] = 'win'
        appearances_df.loc[((appearances_df['home_club_id'] == appearances_df['player_club_id']) & 
                            (appearances_df['home_club_goals'] < appearances_df['away_club_goals'])) | 
                           ((appearances_df['away_club_id'] == appearances_df['player_club_id']) & 
                            (appearances_df['away_club_goals'] < appearances_df['home_club_goals'])), 'result'] = 'loss'
        
        # Save processed data
        players_df.to_csv(os.path.join(processed_dir, "players_processed.csv"), index=False)
        clubs_df.to_csv(os.path.join(processed_dir, "clubs_processed.csv"), index=False)
        games_comp_df.to_csv(os.path.join(processed_dir, "games_processed.csv"), index=False)
        appearances_df.to_csv(os.path.join(processed_dir, "appearances_processed.csv"), index=False)
        
        logger.info(f"Successfully processed {dataset_name} dataset")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {dataset_name} dataset: {e}")
        return False


def process_dataset(dataset: Dict[str, Any]) -> bool:
    """
    Clean and preprocess a dataset.
    
    Args:
        dataset: Dataset metadata from registry
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        dataset_id = dataset['id']
        logging.info(f"Processing dataset {dataset['name']}")
        
        # Load the raw dataset
        file_path = dataset['path']
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            logging.error(f"Unsupported file format: {file_path}")
            return False
        
        # Basic processing steps:
        # 1. Drop duplicates
        df = df.drop_duplicates()
        
        # 2. Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # 3. Convert date columns
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        # 4. Save processed dataset
        processed_dir = os.path.join(PROCESSED_DATA_DIR, dataset_id)
        os.makedirs(processed_dir, exist_ok=True)
        
        processed_path = os.path.join(processed_dir, f"processed_{os.path.basename(file_path)}")
        
        # Save in appropriate format
        if processed_path.endswith('.csv'):
            df.to_csv(processed_path, index=False)
        elif processed_path.endswith(('.xlsx', '.xls')):
            df.to_excel(processed_path, index=False)
        elif processed_path.endswith('.json'):
            df.to_json(processed_path, orient='records')
        
        logging.info(f"Processed dataset saved to {processed_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        return False


def run_pipeline(dataset_id: str, process_type: str = "process") -> bool:
    """
    Run the data pipeline on a specific dataset with a specific processing type.
    
    Args:
        dataset_id: The ID of the dataset to process from the registry
        process_type: Type of processing to perform: 'download', 'process', 'features', 'augment'
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not REGISTRY_AVAILABLE:
        logging.error("Dataset registry not available")
        return False
    
    try:
        # Get dataset metadata from registry
        dataset = get_dataset(dataset_id)
        if not dataset:
            logging.error(f"Dataset with ID {dataset_id} not found")
            return False
        
        logging.info(f"Running pipeline for dataset {dataset['name']} (ID: {dataset_id})")
        
        # Define functions to call based on process_type
        if process_type == "download":
            # Download additional data for the dataset
            success = download_additional_data(dataset)
        
        elif process_type == "process":
            # Clean and preprocess the dataset
            success = process_dataset(dataset)
        
        elif process_type == "features":
            # Generate features for the dataset
            success = generate_features(dataset)
        
        elif process_type == "augment":
            # Augment the dataset
            success = augment_dataset(dataset)
        
        else:
            logging.error(f"Unknown process type: {process_type}")
            return False
        
        if success:
            # Update dataset status in registry
            new_status = {
                "download": "downloaded",
                "process": "processed",
                "features": "featured",
                "augment": "augmented"
            }.get(process_type, "processed")
            
            update_dataset_status(dataset_id, new_status)
            logging.info(f"Pipeline completed successfully for dataset {dataset_id}")
            return True
        else:
            logging.error(f"Pipeline failed for dataset {dataset_id}")
            return False
        
    except Exception as e:
        logging.error(f"Error running pipeline for dataset {dataset_id}: {str(e)}")
        return False


def download_additional_data(dataset: Dict[str, Any]) -> bool:
    """
    Download additional data related to a dataset.
    
    Args:
        dataset: Dataset metadata from registry
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        dataset_id = dataset['id']
        logging.info(f"Downloading additional data for {dataset['name']}")
        # Implement dataset-specific download logic here
        
        # For example, if we can determine this is a football dataset:
        if "football" in dataset['name'].lower() or "soccer" in dataset['name'].lower():
            # Download team rankings, player stats, etc.
            pass
        
        return True
    except Exception as e:
        logging.error(f"Error downloading additional data: {str(e)}")
        return False


def generate_features(dataset: Dict[str, Any]) -> bool:
    """
    Generate features for a dataset.
    
    Args:
        dataset: Dataset metadata from registry
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        dataset_id = dataset['id']
        logging.info(f"Generating features for dataset {dataset['name']}")
        
        # Import feature generation functions from features module
        try:
            from src.data.features import calculate_team_form, calculate_player_form
            features_available = True
        except ImportError:
            features_available = False
        
        # Load the processed dataset
        processed_dir = os.path.join(PROCESSED_DATA_DIR, dataset_id)
        processed_files = [f for f in os.listdir(processed_dir) if f.startswith('processed_')]
        
        if not processed_files:
            logging.error(f"No processed files found for dataset {dataset['name']}")
            return False
        
        processed_path = os.path.join(processed_dir, processed_files[0])
        
        if processed_path.endswith('.csv'):
            df = pd.read_csv(processed_path)
        elif processed_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(processed_path)
        elif processed_path.endswith('.json'):
            df = pd.read_json(processed_path)
        
        # Generate basic features if specific feature functions aren't available
        if not features_available:
            # Generate simple features based on column types
            for col in df.columns:
                # Skip ID columns
                if 'id' in col.lower():
                    continue
                
                # Create categorical features
                if df[col].dtype == 'object':
                    df[f"{col}_encoded"] = df[col].astype('category').cat.codes
                
                # Create numeric features
                if df[col].dtype in ['int64', 'float64']:
                    df[f"{col}_normalized"] = (df[col] - df[col].mean()) / df[col].std()
        else:
            # Use specialized feature functions if available
            # For example, if this is football data:
            if ('home_team' in df.columns and 'away_team' in df.columns) or \
               ('team1' in df.columns and 'team2' in df.columns):
                # Apply team form calculation
                df = calculate_team_form(df)
                
                # Apply player form calculation if player data exists
                if 'players' in df.columns or any('player' in col for col in df.columns):
                    df = calculate_player_form(df)
        
        # Save featured dataset
        featured_dir = os.path.join(FEATURES_DATA_DIR, dataset_id)
        os.makedirs(featured_dir, exist_ok=True)
        
        featured_path = os.path.join(featured_dir, f"featured_{os.path.basename(processed_path)}")
        
        # Save in appropriate format
        if featured_path.endswith('.csv'):
            df.to_csv(featured_path, index=False)
        elif featured_path.endswith(('.xlsx', '.xls')):
            df.to_excel(featured_path, index=False)
        elif featured_path.endswith('.json'):
            df.to_json(featured_path, orient='records')
        
        logging.info(f"Featured dataset saved to {featured_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error generating features: {str(e)}")
        return False


def augment_dataset(dataset: Dict[str, Any]) -> bool:
    """
    Augment a dataset using data augmentation techniques.
    
    Args:
        dataset: Dataset metadata from registry
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        dataset_id = dataset['id']
        logging.info(f"Augmenting dataset {dataset['name']}")
        
        # Import augmentation functions
        try:
            from src.data.augmentation import (
                oversample_minority_class,
                undersample_majority_class,
                generate_synthetic_matches
            )
            augmentation_available = True
        except ImportError:
            augmentation_available = False
        
        # Load the featured dataset if available, otherwise processed
        featured_dir = os.path.join(FEATURES_DATA_DIR, dataset_id)
        if os.path.exists(featured_dir):
            feature_files = [f for f in os.listdir(featured_dir) if f.startswith('featured_')]
            if feature_files:
                data_path = os.path.join(featured_dir, feature_files[0])
            else:
                # Fall back to processed data
                processed_dir = os.path.join(PROCESSED_DATA_DIR, dataset_id)
                processed_files = [f for f in os.listdir(processed_dir) if f.startswith('processed_')]
                if not processed_files:
                    logging.error(f"No processed files found for dataset {dataset['name']}")
                    return False
                data_path = os.path.join(processed_dir, processed_files[0])
        else:
            # Fall back to processed data
            processed_dir = os.path.join(PROCESSED_DATA_DIR, dataset_id)
            processed_files = [f for f in os.listdir(processed_dir) if f.startswith('processed_')]
            if not processed_files:
                logging.error(f"No processed files found for dataset {dataset['name']}")
                return False
            data_path = os.path.join(processed_dir, processed_files[0])
        
        # Load the dataset
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        
        # Apply augmentation techniques
        if augmentation_available:
            # Identify target column if it exists
            target_col = None
            for possible_target in ['result', 'outcome', 'target', 'label', 'class']:
                if possible_target in df.columns:
                    target_col = possible_target
                    break
            
            if target_col:
                # Apply oversampling to minority classes
                df = oversample_minority_class(df, target_column=target_col)
                
                # Also generate synthetic matches
                df_synthetic = generate_synthetic_matches(df)
                df = pd.concat([df, df_synthetic], ignore_index=True)
            else:
                # No target column found, just generate synthetic data
                df_synthetic = generate_synthetic_matches(df)
                df = pd.concat([df, df_synthetic], ignore_index=True)
        else:
            # Basic augmentation without specialized functions
            # For example, simple duplication with small random variations
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # Create synthetic data with small random variations
            df_synthetic = df.copy()
            for col in numeric_cols:
                # Add small random noise to numeric columns
                noise = df_synthetic[col].std() * 0.05 * np.random.randn(len(df_synthetic))
                df_synthetic[col] = df_synthetic[col] + noise
            
            # Combine original and synthetic data
            df = pd.concat([df, df_synthetic], ignore_index=True)
        
        # Save augmented dataset
        augmented_dir = os.path.join(AUGMENTED_DATA_DIR, dataset_id)
        os.makedirs(augmented_dir, exist_ok=True)
        
        augmented_path = os.path.join(augmented_dir, f"augmented_{os.path.basename(data_path)}")
        
        # Save in appropriate format
        if augmented_path.endswith('.csv'):
            df.to_csv(augmented_path, index=False)
        elif augmented_path.endswith(('.xlsx', '.xls')):
            df.to_excel(augmented_path, index=False)
        elif augmented_path.endswith('.json'):
            df.to_json(augmented_path, orient='records')
        
        logging.info(f"Augmented dataset saved to {augmented_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error augmenting dataset: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Soccer Prediction Data Pipeline")
    parser.add_argument("--dataset", type=str, default="all", 
                        help="Dataset to process (default: all)")
    parser.add_argument("--download-only", action="store_true", 
                        help="Only download the data without processing")
    parser.add_argument("--process-only", action="store_true", 
                        help="Only process the data without downloading")
    
    args = parser.parse_args()
    
    download = not args.process_only
    process = not args.download_only
    
    if not download and not process:
        print("Error: Must specify at least one of download or process")
        parser.print_help()
        exit(1)
    
    run_pipeline(args.dataset, download, process) 