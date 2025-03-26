#!/usr/bin/env python3
"""
Database backup and restore utility for the Soccer Prediction System.
This script provides functionality for creating, listing, and restoring database backups.
"""

import os
import sys
import argparse
import datetime
import subprocess
import shutil
import glob
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils.logger import get_logger
try:
    from config.default_config import (
        DB_TYPE, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD,
        MONGO_URI
    )
except ImportError:
    # Fallback defaults if config is not available
    import os
    DB_TYPE = os.getenv("DB_TYPE", "postgres")
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = int(os.getenv("DB_PORT", 5432))
    DB_NAME = os.getenv("DB_NAME", "soccer_prediction")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/soccer_prediction")

# Setup logger
logger = get_logger("db_backup")

# Constants
BACKUP_DIR = os.path.join(project_root, "backups")
MAX_BACKUPS = 10  # Maximum number of backups to keep by default

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message: str) -> None:
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")

def print_success(message: str) -> None:
    """Print a formatted success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_error(message: str) -> None:
    """Print a formatted error message."""
    print(f"{Colors.RED}✗ {message}{Colors.ENDC}")

def print_warning(message: str) -> None:
    """Print a formatted warning message."""
    print(f"{Colors.YELLOW}! {message}{Colors.ENDC}")

def print_info(message: str) -> None:
    """Print a formatted info message."""
    print(f"{Colors.BLUE}ℹ {message}{Colors.ENDC}")

def ensure_backup_dir() -> None:
    """Ensure the backup directory exists."""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        logger.info(f"Created backup directory at {BACKUP_DIR}")

def generate_backup_filename(prefix: str = "backup") -> str:
    """Generate a filename for a backup based on current date and time."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"

def backup_postgres() -> str:
    """
    Create a backup of a PostgreSQL database.
    
    Returns:
        Path to the created backup file
    """
    print_info(f"Starting PostgreSQL backup for database '{DB_NAME}'...")
    backup_name = generate_backup_filename("postgres")
    backup_file = os.path.join(BACKUP_DIR, f"{backup_name}.sql")
    
    # Create pg_dump command
    cmd = [
        "pg_dump",
        "-h", DB_HOST,
        "-p", str(DB_PORT),
        "-U", DB_USER,
        "-d", DB_NAME,
        "-F", "c",  # Custom format (compressed)
        "-f", backup_file
    ]
    
    # Set PGPASSWORD environment variable for authentication
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD
    
    try:
        # Run pg_dump command
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Create a metadata file with backup information
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "db_type": "postgres",
            "db_name": DB_NAME,
            "backup_file": os.path.basename(backup_file),
            "size": os.path.getsize(backup_file)
        }
        
        metadata_file = os.path.join(BACKUP_DIR, f"{backup_name}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print_success(f"PostgreSQL backup created successfully: {backup_file}")
        logger.info(f"PostgreSQL backup created: {backup_file}")
        return backup_file
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        print_error(f"PostgreSQL backup failed: {error_message}")
        logger.error(f"PostgreSQL backup failed: {error_message}")
        raise

def backup_mysql() -> str:
    """
    Create a backup of a MySQL database.
    
    Returns:
        Path to the created backup file
    """
    print_info(f"Starting MySQL backup for database '{DB_NAME}'...")
    backup_name = generate_backup_filename("mysql")
    backup_file = os.path.join(BACKUP_DIR, f"{backup_name}.sql")
    
    # Create mysqldump command
    cmd = [
        "mysqldump",
        "-h", DB_HOST,
        "-P", str(DB_PORT),
        "-u", DB_USER,
        f"--password={DB_PASSWORD}",
        "--single-transaction",
        "--routines",
        "--triggers",
        "--events",
        DB_NAME
    ]
    
    try:
        # Run mysqldump command and redirect output to file
        with open(backup_file, 'w') as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.PIPE)
        
        # Create a metadata file with backup information
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "db_type": "mysql",
            "db_name": DB_NAME,
            "backup_file": os.path.basename(backup_file),
            "size": os.path.getsize(backup_file)
        }
        
        metadata_file = os.path.join(BACKUP_DIR, f"{backup_name}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print_success(f"MySQL backup created successfully: {backup_file}")
        logger.info(f"MySQL backup created: {backup_file}")
        return backup_file
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        print_error(f"MySQL backup failed: {error_message}")
        logger.error(f"MySQL backup failed: {error_message}")
        raise

def backup_sqlite() -> str:
    """
    Create a backup of a SQLite database.
    
    Returns:
        Path to the created backup file
    """
    print_info(f"Starting SQLite backup for database '{DB_NAME}'...")
    backup_name = generate_backup_filename("sqlite")
    sqlite_db_path = f"{DB_NAME}.db"
    backup_file = os.path.join(BACKUP_DIR, f"{backup_name}.db")
    
    try:
        # Simply copy the SQLite file
        shutil.copy2(sqlite_db_path, backup_file)
        
        # Create a metadata file with backup information
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "db_type": "sqlite",
            "db_name": DB_NAME,
            "backup_file": os.path.basename(backup_file),
            "size": os.path.getsize(backup_file)
        }
        
        metadata_file = os.path.join(BACKUP_DIR, f"{backup_name}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print_success(f"SQLite backup created successfully: {backup_file}")
        logger.info(f"SQLite backup created: {backup_file}")
        return backup_file
    
    except Exception as e:
        print_error(f"SQLite backup failed: {str(e)}")
        logger.error(f"SQLite backup failed: {str(e)}")
        raise

def backup_mongodb() -> str:
    """
    Create a backup of a MongoDB database.
    
    Returns:
        Path to the created backup file
    """
    print_info(f"Starting MongoDB backup...")
    backup_name = generate_backup_filename("mongodb")
    backup_dir = os.path.join(BACKUP_DIR, backup_name)
    
    # Parse MongoDB URI to extract database name
    from urllib.parse import urlparse
    parsed_uri = urlparse(MONGO_URI)
    mongo_db_name = parsed_uri.path.lstrip('/')
    
    # If no database name is provided in the URI, use the default
    if not mongo_db_name:
        mongo_db_name = DB_NAME
    
    # Extract host and port from URI
    mongo_host = parsed_uri.hostname or "127.0.0.1"
    mongo_port = parsed_uri.port or 27017
    
    try:
        # Create directory for MongoDB backup
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create mongodump command
        cmd = [
            "mongodump",
            "--host", mongo_host,
            "--port", str(mongo_port),
            "--db", mongo_db_name,
            "--out", backup_dir
        ]
        
        # Add authentication if provided in URI
        if parsed_uri.username and parsed_uri.password:
            cmd.extend(["--username", parsed_uri.username])
            cmd.extend(["--password", parsed_uri.password])
            cmd.extend(["--authenticationDatabase", "admin"])
        
        # Run mongodump command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Zip the backup directory
        backup_zip = f"{backup_dir}.zip"
        shutil.make_archive(backup_dir, 'zip', backup_dir)
        
        # Remove the directory, keeping only the zip file
        shutil.rmtree(backup_dir)
        
        # Create a metadata file with backup information
        metadata = {
            "timestamp": datetime.datetime.now().isoformat(),
            "db_type": "mongodb",
            "db_name": mongo_db_name,
            "backup_file": os.path.basename(backup_zip),
            "size": os.path.getsize(backup_zip)
        }
        
        metadata_file = os.path.join(BACKUP_DIR, f"{backup_name}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print_success(f"MongoDB backup created successfully: {backup_zip}")
        logger.info(f"MongoDB backup created: {backup_zip}")
        return backup_zip
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        print_error(f"MongoDB backup failed: {error_message}")
        logger.error(f"MongoDB backup failed: {error_message}")
        raise

def create_backup(db_type: Optional[str] = None) -> str:
    """
    Create a backup of the database.
    
    Args:
        db_type: Database type to backup (if None, use the configured DB_TYPE)
        
    Returns:
        Path to the created backup file
    """
    ensure_backup_dir()
    
    db_type = db_type or DB_TYPE
    
    if db_type == "postgres":
        return backup_postgres()
    elif db_type == "mysql":
        return backup_mysql()
    elif db_type == "sqlite":
        return backup_sqlite()
    elif db_type == "mongodb":
        return backup_mongodb()
    else:
        error_msg = f"Unsupported database type: {db_type}"
        print_error(error_msg)
        logger.error(error_msg)
        raise ValueError(error_msg)

def list_backups() -> List[Dict[str, Any]]:
    """
    List all available backups.
    
    Returns:
        List of dictionaries with backup information
    """
    ensure_backup_dir()
    
    # Find all metadata files
    metadata_files = glob.glob(os.path.join(BACKUP_DIR, "*.json"))
    
    backups = []
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if the backup file still exists
            backup_file = os.path.join(BACKUP_DIR, metadata["backup_file"])
            if not os.path.exists(backup_file):
                continue
            
            # Add file path to metadata
            metadata["file_path"] = backup_file
            backups.append(metadata)
        
        except Exception as e:
            logger.warning(f"Error reading metadata file {metadata_file}: {e}")
    
    # Sort backups by timestamp (newest first)
    backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return backups

def print_backups() -> None:
    """Print a formatted list of available backups."""
    backups = list_backups()
    
    if not backups:
        print_warning("No backups found.")
        return
    
    print_header("Available Backups")
    print(f"{'ID':<3} | {'Date':<19} | {'Type':<8} | {'Database':<15} | {'Size':<10} | {'File'}")
    print("-" * 80)
    
    for i, backup in enumerate(backups):
        # Format timestamp
        timestamp = datetime.datetime.fromisoformat(backup["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        
        # Format size
        size_bytes = backup["size"]
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        print(f"{i+1:<3} | {timestamp} | {backup['db_type']:<8} | {backup['db_name']:<15} | {size_str:<10} | {backup['backup_file']}")

def restore_postgres(backup_file: str) -> None:
    """
    Restore a PostgreSQL database from a backup file.
    
    Args:
        backup_file: Path to the backup file
    """
    print_info(f"Starting PostgreSQL restore for database '{DB_NAME}'...")
    
    # Create pg_restore command
    cmd = [
        "pg_restore",
        "-h", DB_HOST,
        "-p", str(DB_PORT),
        "-U", DB_USER,
        "-d", DB_NAME,
        "-c",  # Clean (drop) database objects before recreating them
        backup_file
    ]
    
    # Set PGPASSWORD environment variable for authentication
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD
    
    try:
        # Run pg_restore command
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print_success(f"PostgreSQL restore completed successfully.")
        logger.info(f"PostgreSQL restore completed: {backup_file}")
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        print_error(f"PostgreSQL restore failed: {error_message}")
        logger.error(f"PostgreSQL restore failed: {error_message}")
        raise

def restore_mysql(backup_file: str) -> None:
    """
    Restore a MySQL database from a backup file.
    
    Args:
        backup_file: Path to the backup file
    """
    print_info(f"Starting MySQL restore for database '{DB_NAME}'...")
    
    # Create mysql command to restore
    cmd = [
        "mysql",
        "-h", DB_HOST,
        "-P", str(DB_PORT),
        "-u", DB_USER,
        f"--password={DB_PASSWORD}",
        DB_NAME
    ]
    
    try:
        # Run mysql command and pipe the backup file to it
        with open(backup_file, 'r') as f:
            subprocess.run(cmd, check=True, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print_success(f"MySQL restore completed successfully.")
        logger.info(f"MySQL restore completed: {backup_file}")
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        print_error(f"MySQL restore failed: {error_message}")
        logger.error(f"MySQL restore failed: {error_message}")
        raise

def restore_sqlite(backup_file: str) -> None:
    """
    Restore a SQLite database from a backup file.
    
    Args:
        backup_file: Path to the backup file
    """
    print_info(f"Starting SQLite restore for database '{DB_NAME}'...")
    
    sqlite_db_path = f"{DB_NAME}.db"
    
    try:
        # Create a backup of the current database before restoring
        if os.path.exists(sqlite_db_path):
            temp_backup = f"{sqlite_db_path}.bak.{int(time.time())}"
            shutil.copy2(sqlite_db_path, temp_backup)
            print_info(f"Created backup of current database: {temp_backup}")
        
        # Copy the backup file to the database location
        shutil.copy2(backup_file, sqlite_db_path)
        
        print_success(f"SQLite restore completed successfully.")
        logger.info(f"SQLite restore completed: {backup_file}")
    
    except Exception as e:
        print_error(f"SQLite restore failed: {str(e)}")
        logger.error(f"SQLite restore failed: {str(e)}")
        raise

def restore_mongodb(backup_file: str) -> None:
    """
    Restore a MongoDB database from a backup file.
    
    Args:
        backup_file: Path to the backup file (zip)
    """
    print_info(f"Starting MongoDB restore...")
    
    # Parse MongoDB URI to extract database name
    from urllib.parse import urlparse
    parsed_uri = urlparse(MONGO_URI)
    mongo_db_name = parsed_uri.path.lstrip('/')
    
    # If no database name is provided in the URI, use the default
    if not mongo_db_name:
        mongo_db_name = DB_NAME
    
    # Extract host and port from URI
    mongo_host = parsed_uri.hostname or "127.0.0.1"
    mongo_port = parsed_uri.port or 27017
    
    # Create a temporary directory for extraction
    temp_dir = os.path.join(BACKUP_DIR, f"temp_restore_{int(time.time())}")
    
    try:
        # Extract the zip file
        os.makedirs(temp_dir, exist_ok=True)
        shutil.unpack_archive(backup_file, temp_dir, 'zip')
        
        # Create mongorestore command
        cmd = [
            "mongorestore",
            "--host", mongo_host,
            "--port", str(mongo_port),
            "--db", mongo_db_name,
            "--drop",  # Drop existing collections before restoring
            os.path.join(temp_dir, mongo_db_name)
        ]
        
        # Add authentication if provided in URI
        if parsed_uri.username and parsed_uri.password:
            cmd.extend(["--username", parsed_uri.username])
            cmd.extend(["--password", parsed_uri.password])
            cmd.extend(["--authenticationDatabase", "admin"])
        
        # Run mongorestore command
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print_success(f"MongoDB restore completed successfully.")
        logger.info(f"MongoDB restore completed: {backup_file}")
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        print_error(f"MongoDB restore failed: {error_message}")
        logger.error(f"MongoDB restore failed: {error_message}")
        raise
    
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def restore_backup(backup_id: Optional[int] = None, backup_file: Optional[str] = None, db_type: Optional[str] = None) -> None:
    """
    Restore a database from a backup.
    
    Args:
        backup_id: ID of the backup to restore (from the list_backups output)
        backup_file: Path to the backup file (alternative to backup_id)
        db_type: Database type (required if backup_file is provided)
    """
    ensure_backup_dir()
    
    # Determine the backup file and database type
    if backup_id is not None:
        backups = list_backups()
        if not backups:
            print_error("No backups found.")
            return
        
        if backup_id < 1 or backup_id > len(backups):
            print_error(f"Invalid backup ID. Must be between 1 and {len(backups)}.")
            return
        
        # Get the selected backup
        backup = backups[backup_id - 1]
        backup_file = backup["file_path"]
        db_type = backup["db_type"]
    
    elif backup_file is not None:
        if not os.path.exists(backup_file):
            print_error(f"Backup file not found: {backup_file}")
            return
        
        if db_type is None:
            print_error("Database type must be specified when providing a backup file.")
            return
    
    else:
        print_error("Either backup_id or backup_file must be provided.")
        return
    
    # Ask for confirmation
    print_warning(f"You are about to restore database '{DB_NAME}' from backup: {os.path.basename(backup_file)}")
    print_warning("This will OVERWRITE the current database. All existing data will be lost.")
    
    confirmation = input("Are you sure you want to continue? (y/N): ")
    if confirmation.lower() not in ["y", "yes"]:
        print_info("Restore operation cancelled.")
        return
    
    # Perform the restore based on database type
    if db_type == "postgres":
        restore_postgres(backup_file)
    elif db_type == "mysql":
        restore_mysql(backup_file)
    elif db_type == "sqlite":
        restore_sqlite(backup_file)
    elif db_type == "mongodb":
        restore_mongodb(backup_file)
    else:
        print_error(f"Unsupported database type: {db_type}")
        return

def clean_old_backups(keep: int = MAX_BACKUPS) -> None:
    """
    Remove old backups, keeping only the specified number of most recent backups.
    
    Args:
        keep: Number of backups to keep
    """
    backups = list_backups()
    
    if len(backups) <= keep:
        print_info(f"No old backups to remove (keeping {keep} backups).")
        return
    
    # Identify backups to remove (oldest first)
    backups_to_remove = backups[keep:]
    
    print_info(f"Removing {len(backups_to_remove)} old backups (keeping {keep} most recent)...")
    
    for backup in backups_to_remove:
        try:
            # Remove the backup file
            backup_file = backup["file_path"]
            os.remove(backup_file)
            
            # Remove the metadata file
            backup_name = os.path.splitext(backup["backup_file"])[0]
            metadata_file = os.path.join(BACKUP_DIR, f"{backup_name}.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            print_success(f"Removed old backup: {backup['backup_file']}")
            logger.info(f"Removed old backup: {backup_file}")
        
        except Exception as e:
            print_error(f"Error removing backup {backup['backup_file']}: {str(e)}")
            logger.error(f"Error removing backup {backup['file_path']}: {str(e)}")
    
    print_success(f"Backup cleanup complete. Kept {keep} most recent backups.")

def main() -> None:
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Database backup and restore utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create a new database backup")
    backup_parser.add_argument("--type", choices=["postgres", "mysql", "sqlite", "mongodb"], 
                            help="Database type (default: from config)")
    backup_parser.add_argument("--clean", action="store_true", 
                            help="Clean old backups after creating a new one")
    backup_parser.add_argument("--keep", type=int, default=MAX_BACKUPS, 
                            help=f"Number of backups to keep when cleaning (default: {MAX_BACKUPS})")
    
    # List command
    subparsers.add_parser("list", help="List available backups")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore database from a backup")
    restore_parser.add_argument("--id", type=int, help="Backup ID to restore (from the list output)")
    restore_parser.add_argument("--file", help="Path to a backup file to restore")
    restore_parser.add_argument("--type", choices=["postgres", "mysql", "sqlite", "mongodb"], 
                             help="Database type (required if --file is specified)")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Remove old backups")
    clean_parser.add_argument("--keep", type=int, default=MAX_BACKUPS, 
                           help=f"Number of backups to keep (default: {MAX_BACKUPS})")
    
    args = parser.parse_args()
    
    if args.command == "backup":
        backup_file = create_backup(args.type)
        if args.clean:
            clean_old_backups(args.keep)
    
    elif args.command == "list":
        print_backups()
    
    elif args.command == "restore":
        restore_backup(args.id, args.file, args.type)
    
    elif args.command == "clean":
        clean_old_backups(args.keep)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 