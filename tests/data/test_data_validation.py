"""
Data validation tests for the Soccer Prediction System.
These tests verify the integrity, quality, and consistency of data 
used throughout the system.
"""

import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Import project components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.pipeline import DATASETS, download_dataset, process_dataset
from src.data.features import load_processed_data, create_feature_datasets
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("tests.data_validation")

# Define paths (same as in src.data.pipeline)
try:
    from config.default_config import DATA_DIR
except ImportError:
    # Fallback defaults if config is not available
    DATA_DIR = os.path.join(Path(__file__).resolve().parent.parent.parent, "data")

# Data directories
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")


class TestDataAvailability(unittest.TestCase):
    """Test case for data availability checks."""
    
    def test_data_directories_exist(self):
        """Test that all required data directories exist."""
        self.assertTrue(os.path.exists(DATA_DIR), f"Main data directory missing: {DATA_DIR}")
        self.assertTrue(os.path.exists(RAW_DATA_DIR), f"Raw data directory missing: {RAW_DATA_DIR}")
        self.assertTrue(os.path.exists(PROCESSED_DATA_DIR), f"Processed data directory missing: {PROCESSED_DATA_DIR}")
        self.assertTrue(os.path.exists(FEATURES_DIR), f"Features directory missing: {FEATURES_DIR}")
    
    def test_raw_dataset_files_exist(self):
        """Test that raw dataset files are available for each configured dataset."""
        for dataset_name, dataset_info in DATASETS.items():
            raw_dir = dataset_info["raw_dir"]
            
            # Skip if raw directory doesn't exist (not downloaded yet)
            if not os.path.exists(raw_dir):
                logger.warning(f"Raw directory for {dataset_name} does not exist: {raw_dir}")
                continue
            
            # Check for required files
            for file in dataset_info["files"]:
                # Handle special case of football_data
                if dataset_name == "football_data" and "/" in file:
                    full_path = os.path.join(raw_dir, file)
                else:
                    full_path = os.path.join(raw_dir, file)
                
                self.assertTrue(
                    os.path.exists(full_path),
                    f"Required file missing for dataset {dataset_name}: {full_path}"
                )


class TestRawDataFormat(unittest.TestCase):
    """Test case for raw data format validation."""
    
    def test_transfermarkt_format(self):
        """Test the format of raw Transfermarkt data."""
        dataset_name = "transfermarkt"
        raw_dir = DATASETS[dataset_name]["raw_dir"]
        
        # Skip if dataset doesn't exist
        if not os.path.exists(raw_dir):
            self.skipTest(f"Transfermarkt dataset not downloaded: {raw_dir}")
        
        # Test players.csv
        players_path = os.path.join(raw_dir, "players.csv")
        if os.path.exists(players_path):
            players_df = pd.read_csv(players_path)
            
            # Check required columns
            required_columns = ['player_id', 'name', 'date_of_birth', 'position']
            for col in required_columns:
                self.assertIn(col, players_df.columns, f"Missing required column in players.csv: {col}")
            
            # Validate data types
            self.assertTrue(pd.api.types.is_numeric_dtype(players_df['player_id']), "player_id should be numeric")
            
            # Check for duplicates in player_id
            self.assertEqual(
                len(players_df['player_id']), 
                len(players_df['player_id'].unique()), 
                "Duplicate player_id found in players.csv"
            )
        
        # Test clubs.csv
        clubs_path = os.path.join(raw_dir, "clubs.csv")
        if os.path.exists(clubs_path):
            clubs_df = pd.read_csv(clubs_path)
            
            # Check required columns
            required_columns = ['club_id', 'name']
            for col in required_columns:
                self.assertIn(col, clubs_df.columns, f"Missing required column in clubs.csv: {col}")
            
            # Validate data types
            self.assertTrue(pd.api.types.is_numeric_dtype(clubs_df['club_id']), "club_id should be numeric")
            
            # Check for duplicates in club_id
            self.assertEqual(
                len(clubs_df['club_id']), 
                len(clubs_df['club_id'].unique()), 
                "Duplicate club_id found in clubs.csv"
            )
        
        # Test games.csv
        games_path = os.path.join(raw_dir, "games.csv")
        if os.path.exists(games_path):
            games_df = pd.read_csv(games_path)
            
            # Check required columns
            required_columns = ['game_id', 'date', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']
            for col in required_columns:
                self.assertIn(col, games_df.columns, f"Missing required column in games.csv: {col}")
            
            # Validate data types
            self.assertTrue(pd.api.types.is_numeric_dtype(games_df['game_id']), "game_id should be numeric")
            self.assertTrue(pd.api.types.is_numeric_dtype(games_df['home_club_id']), "home_club_id should be numeric")
            self.assertTrue(pd.api.types.is_numeric_dtype(games_df['away_club_id']), "away_club_id should be numeric")
            
            # Check for duplicates in game_id
            self.assertEqual(
                len(games_df['game_id']), 
                len(games_df['game_id'].unique()), 
                "Duplicate game_id found in games.csv"
            )
    
    def test_football_data_format(self):
        """Test the format of raw football-data.co.uk data."""
        dataset_name = "football_data"
        raw_dir = DATASETS[dataset_name]["raw_dir"]
        
        # Skip if dataset doesn't exist
        if not os.path.exists(raw_dir):
            self.skipTest(f"Football Data dataset not downloaded: {raw_dir}")
        
        # Take a sample file to test
        season_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
        if not season_dirs:
            self.skipTest("No season directories found in Football Data dataset")
        
        # Find the first available CSV file
        csv_path = None
        for season in season_dirs:
            season_path = os.path.join(raw_dir, season)
            csv_files = [f for f in os.listdir(season_path) if f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(season_path, csv_files[0])
                break
        
        if not csv_path:
            self.skipTest("No CSV files found in Football Data dataset")
        
        # Test the CSV file
        df = pd.read_csv(csv_path)
        
        # Check required columns (common in football-data.co.uk files)
        required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        for col in required_columns:
            self.assertIn(col, df.columns, f"Missing required column in {os.path.basename(csv_path)}: {col}")
        
        # Validate match result is consistent with goals
        for _, row in df.iterrows():
            if 'FTR' in row and 'FTHG' in row and 'FTAG' in row:
                if row['FTHG'] > row['FTAG']:
                    self.assertEqual(row['FTR'], 'H', f"Match result should be 'H' when home goals > away goals")
                elif row['FTHG'] < row['FTAG']:
                    self.assertEqual(row['FTR'], 'A', f"Match result should be 'A' when home goals < away goals")
                else:
                    self.assertEqual(row['FTR'], 'D', f"Match result should be 'D' when home goals = away goals")


class TestProcessedData(unittest.TestCase):
    """Test case for processed data validation."""
    
    def test_processed_data_exists(self):
        """Test that processed data exists for each dataset that has been processed."""
        for dataset_name in DATASETS.keys():
            processed_dir = os.path.join(PROCESSED_DATA_DIR, dataset_name)
            
            # Skip if processed directory doesn't exist
            if not os.path.exists(processed_dir):
                logger.warning(f"Processed directory for {dataset_name} does not exist: {processed_dir}")
                continue
            
            # Check that at least one CSV file exists
            csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
            self.assertTrue(
                len(csv_files) > 0,
                f"No processed CSV files found for dataset {dataset_name} in {processed_dir}"
            )
    
    def test_processed_data_integrity(self):
        """Test the integrity of processed data."""
        for dataset_name in DATASETS.keys():
            processed_dir = os.path.join(PROCESSED_DATA_DIR, dataset_name)
            
            # Skip if processed directory doesn't exist
            if not os.path.exists(processed_dir):
                continue
            
            # Load all processed data for the dataset
            try:
                data_dict = load_processed_data(dataset_name)
            except (FileNotFoundError, Exception) as e:
                logger.warning(f"Could not load processed data for {dataset_name}: {e}")
                continue
            
            # Test each dataframe in the dataset
            for name, df in data_dict.items():
                # Check that dataframe is not empty
                self.assertFalse(df.empty, f"Processed dataframe {name} is empty for dataset {dataset_name}")
                
                # Check for NaN values in key columns
                if 'match_id' in df.columns:
                    self.assertEqual(0, df['match_id'].isna().sum(), f"NaN values found in match_id column of {name}")
                
                if 'team_id' in df.columns:
                    self.assertEqual(0, df['team_id'].isna().sum(), f"NaN values found in team_id column of {name}")
                
                if 'player_id' in df.columns:
                    self.assertEqual(0, df['player_id'].isna().sum(), f"NaN values found in player_id column of {name}")
                
                if 'date' in df.columns:
                    self.assertEqual(0, df['date'].isna().sum(), f"NaN values found in date column of {name}")


class TestFeatureEngineering(unittest.TestCase):
    """Test case for feature engineering validation."""
    
    def test_feature_datasets_exist(self):
        """Test that feature datasets exist."""
        for dataset_name in DATASETS.keys():
            features_dir = os.path.join(FEATURES_DIR, dataset_name)
            
            # Skip if features directory doesn't exist
            if not os.path.exists(features_dir):
                logger.warning(f"Features directory for {dataset_name} does not exist: {features_dir}")
                continue
            
            # Check that at least one CSV file exists
            csv_files = [f for f in os.listdir(features_dir) if f.endswith('.csv')]
            self.assertTrue(
                len(csv_files) > 0,
                f"No feature CSV files found for dataset {dataset_name} in {features_dir}"
            )
    
    def test_feature_pipelines_exist(self):
        """Test that feature transformation pipelines exist."""
        for dataset_name in DATASETS.keys():
            features_dir = os.path.join(FEATURES_DIR, dataset_name)
            
            # Skip if features directory doesn't exist
            if not os.path.exists(features_dir):
                continue
            
            # Check that at least one pipeline file exists
            pipeline_files = [f for f in os.listdir(features_dir) if f.endswith('.joblib')]
            self.assertTrue(
                len(pipeline_files) > 0,
                f"No feature pipeline files found for dataset {dataset_name} in {features_dir}"
            )
    
    def test_feature_integrity(self):
        """Test the integrity of engineered features."""
        for dataset_name in DATASETS.keys():
            features_dir = os.path.join(FEATURES_DIR, dataset_name)
            
            # Skip if features directory doesn't exist
            if not os.path.exists(features_dir):
                continue
            
            # Check each feature CSV file
            csv_files = [f for f in os.listdir(features_dir) if f.endswith('.csv')]
            for file in csv_files:
                file_path = os.path.join(features_dir, file)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Check that dataframe is not empty
                    self.assertFalse(df.empty, f"Feature dataframe {file} is empty for dataset {dataset_name}")
                    
                    # Check for excessive NaN values (more than 20% in any column)
                    na_percentage = df.isna().mean() * 100
                    columns_with_many_nans = na_percentage[na_percentage > 20].index.tolist()
                    self.assertEqual(
                        0,
                        len(columns_with_many_nans),
                        f"These columns in {file} have >20% NaN values: {columns_with_many_nans}"
                    )
                    
                    # For match feature datasets, check that we have both teams and outcome
                    if 'match_features' in file:
                        # Check essential columns
                        essential_columns = ['match_id', 'home_team_id', 'away_team_id']
                        for col in essential_columns:
                            self.assertIn(col, df.columns, f"Missing essential column {col} in {file}")
                        
                        # At least one outcome column should exist
                        outcome_columns = ['home_goals', 'away_goals', 'result', 'home_win', 'draw', 'away_win']
                        self.assertTrue(
                            any(col in df.columns for col in outcome_columns),
                            f"No outcome column found in {file}"
                        )
                
                except Exception as e:
                    self.fail(f"Error testing feature file {file}: {e}")


class TestEndToEndDataPipeline(unittest.TestCase):
    """Test case for end-to-end data pipeline validation."""
    
    def test_pipeline_steps(self):
        """Test that the data pipeline can execute all steps for a sample dataset."""
        # Use a small dataset for testing
        dataset_name = "football_data"
        
        # Download dataset if needed (only test a single season/league for speed)
        DATASETS[dataset_name]["seasons"] = [DATASETS[dataset_name]["seasons"][0]]  # Use only first season
        DATASETS[dataset_name]["leagues"] = [DATASETS[dataset_name]["leagues"][0]]  # Use only first league
        
        # Try to download data (skip if fails)
        try:
            success = download_dataset(dataset_name)
            if not success:
                self.skipTest(f"Failed to download test data for {dataset_name}")
        except Exception as e:
            self.skipTest(f"Error downloading test data: {e}")
        
        # Test processing
        try:
            success = process_dataset(dataset_name)
            self.assertTrue(success, f"Failed to process dataset {dataset_name}")
        except Exception as e:
            self.fail(f"Error processing dataset {dataset_name}: {e}")
        
        # Test feature engineering
        try:
            feature_datasets = create_feature_datasets(dataset_name)
            self.assertIsInstance(feature_datasets, dict, "Feature datasets should be returned as a dictionary")
            self.assertTrue(len(feature_datasets) > 0, "No feature datasets created")
        except Exception as e:
            self.fail(f"Error creating feature datasets for {dataset_name}: {e}")


if __name__ == '__main__':
    unittest.main() 