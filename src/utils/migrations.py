"""
Database migration utilities for the Soccer Prediction System.
Handles database schema version control and migrations.
"""

import os
import sys
import importlib
import logging
import datetime
import hashlib
import re
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Session

from src.utils.logger import get_logger
from src.utils.db import get_db, db_session, engine

# Setup logger
logger = get_logger("migrations")

# Constants
MIGRATIONS_TABLE = "schema_migrations"
MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations")


def ensure_migrations_table_exists() -> None:
    """
    Ensure the schema_migrations table exists in the database.
    This table tracks which migrations have been applied.
    """
    if not os.path.exists(MIGRATIONS_DIR):
        os.makedirs(MIGRATIONS_DIR)
        logger.info(f"Created migrations directory at {MIGRATIONS_DIR}")
        
    metadata = sa.MetaData()
    
    # Create migrations table if it doesn't exist
    if not sa.inspect(engine).has_table(MIGRATIONS_TABLE):
        migrations_table = sa.Table(
            MIGRATIONS_TABLE,
            metadata,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("version", sa.String(255), nullable=False, unique=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("applied_at", sa.DateTime, default=datetime.datetime.utcnow),
            sa.Column("checksum", sa.String(64), nullable=False),
            sa.Column("execution_time", sa.Float, nullable=False),
        )
        
        metadata.create_all(engine)
        logger.info(f"Created {MIGRATIONS_TABLE} table")


def get_applied_migrations() -> List[str]:
    """
    Get a list of all migrations that have been applied to the database.
    
    Returns:
        List of migration version strings
    """
    ensure_migrations_table_exists()
    
    with db_session() as session:
        query = sa.text(f"SELECT version FROM {MIGRATIONS_TABLE} ORDER BY id")
        result = session.execute(query)
        return [row[0] for row in result]


def get_available_migrations() -> List[Tuple[str, str, str]]:
    """
    Get a list of all available migrations from the migrations directory.
    
    Returns:
        List of tuples (version, name, full_path)
    """
    if not os.path.exists(MIGRATIONS_DIR):
        return []
    
    migrations = []
    migration_pattern = re.compile(r"^V(\d+)__(.+)\.py$")
    
    for filename in sorted(os.listdir(MIGRATIONS_DIR)):
        match = migration_pattern.match(filename)
        if match and not filename.startswith("__"):
            version = match.group(1)
            name = match.group(2).replace("_", " ")
            full_path = os.path.join(MIGRATIONS_DIR, filename)
            migrations.append((version, name, full_path))
    
    return migrations


def calculate_file_checksum(file_path: str) -> str:
    """
    Calculate a checksum for a migration file.
    
    Args:
        file_path: Path to the migration file
        
    Returns:
        Hex string checksum
    """
    with open(file_path, 'rb') as f:
        file_contents = f.read()
        return hashlib.sha256(file_contents).hexdigest()


def create_migration(name: str) -> str:
    """
    Create a new migration file.
    
    Args:
        name: Name of the migration (will be converted to snake_case)
        
    Returns:
        Path to the created migration file
    """
    ensure_migrations_table_exists()
    
    # Convert name to snake_case
    snake_name = name.lower().replace(" ", "_")
    
    # Get the next version number
    available_migrations = get_available_migrations()
    if available_migrations:
        last_version = max(int(version) for version, _, _ in available_migrations)
        new_version = last_version + 1
    else:
        new_version = 1
    
    # Format with leading zeros
    version_str = f"{new_version:04d}"
    
    # Create the migration file
    filename = f"V{version_str}__{snake_name}.py"
    file_path = os.path.join(MIGRATIONS_DIR, filename)
    
    with open(file_path, 'w') as f:
        f.write(f'''"""
Migration: {name}
Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def upgrade(engine):
    """
    Upgrade database schema.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    connection = engine.connect()
    metadata = MetaData()
    
    # Create or alter tables here
    # Example:
    # teams = Table('teams', metadata,
    #     Column('id', Integer, primary_key=True),
    #     Column('name', String(255), nullable=False),
    #     Column('country', String(255), nullable=False),
    # )
    # teams.create(connection)
    
    connection.close()


def downgrade(engine):
    """
    Downgrade database schema.
    
    Args:
        engine: SQLAlchemy engine instance
    """
    connection = engine.connect()
    metadata = MetaData()
    
    # Drop or alter tables here
    # Example:
    # connection.execute(text("DROP TABLE IF EXISTS teams"))
    
    connection.close()
''')
    
    logger.info(f"Created migration file: {file_path}")
    return file_path


def apply_migration(version: str, name: str, file_path: str) -> None:
    """
    Apply a single migration to the database.
    
    Args:
        version: Migration version
        name: Migration name
        file_path: Path to the migration file
    """
    logger.info(f"Applying migration {version}: {name}")
    
    # Calculate checksum
    checksum = calculate_file_checksum(file_path)
    
    # Import the migration module
    module_name = os.path.basename(file_path)[:-3]  # Remove .py extension
    
    # Add migrations directory to path if not already there
    if MIGRATIONS_DIR not in sys.path:
        sys.path.insert(0, os.path.dirname(MIGRATIONS_DIR))
    
    try:
        # Import the migration module
        module = importlib.import_module(f"migrations.{module_name}")
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Apply the migration
        module.upgrade(engine)
        
        # Calculate execution time
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        
        # Record the migration in the schema_migrations table
        with db_session() as session:
            query = sa.text(f"""
                INSERT INTO {MIGRATIONS_TABLE} (version, name, applied_at, checksum, execution_time)
                VALUES (:version, :name, :applied_at, :checksum, :execution_time)
            """)
            session.execute(query, {
                "version": version,
                "name": name,
                "applied_at": datetime.datetime.utcnow(),
                "checksum": checksum,
                "execution_time": execution_time
            })
        
        logger.info(f"Migration {version} applied successfully in {execution_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error applying migration {version}: {e}")
        raise


def revert_migration(version: str) -> None:
    """
    Revert a specific migration.
    
    Args:
        version: Migration version to revert
    """
    applied_migrations = get_applied_migrations()
    if version not in applied_migrations:
        logger.error(f"Migration {version} is not applied, cannot revert")
        return
    
    # Find the migration file
    for v, name, file_path in get_available_migrations():
        if v == version:
            logger.info(f"Reverting migration {version}: {name}")
            
            # Import the migration module
            module_name = os.path.basename(file_path)[:-3]  # Remove .py extension
            
            # Add migrations directory to path if not already there
            if MIGRATIONS_DIR not in sys.path:
                sys.path.insert(0, os.path.dirname(MIGRATIONS_DIR))
            
            try:
                # Import the migration module
                module = importlib.import_module(f"migrations.{module_name}")
                
                # Apply the downgrade
                module.downgrade(engine)
                
                # Remove the migration from the schema_migrations table
                with db_session() as session:
                    query = sa.text(f"DELETE FROM {MIGRATIONS_TABLE} WHERE version = :version")
                    session.execute(query, {"version": version})
                
                logger.info(f"Migration {version} reverted successfully")
                return
            
            except Exception as e:
                logger.error(f"Error reverting migration {version}: {e}")
                raise
    
    logger.error(f"Migration file for version {version} not found")


def migrate(target_version: Optional[str] = None) -> None:
    """
    Apply all pending migrations or up to a specific version.
    
    Args:
        target_version: Optional target version to migrate to
    """
    ensure_migrations_table_exists()
    
    applied_migrations = get_applied_migrations()
    available_migrations = get_available_migrations()
    
    # Filter to migrations that haven't been applied yet
    pending_migrations = [
        (version, name, path) for version, name, path in available_migrations
        if version not in applied_migrations
    ]
    
    # Sort by version
    pending_migrations.sort(key=lambda x: int(x[0]))
    
    # If target version is specified, filter to migrations up to that version
    if target_version:
        pending_migrations = [
            m for m in pending_migrations if int(m[0]) <= int(target_version)
        ]
    
    if not pending_migrations:
        logger.info("No pending migrations to apply")
        return
    
    logger.info(f"Found {len(pending_migrations)} pending migrations")
    
    for version, name, file_path in pending_migrations:
        apply_migration(version, name, file_path)


def rollback(steps: int = 1) -> None:
    """
    Rollback the specified number of migrations.
    
    Args:
        steps: Number of migrations to roll back
    """
    ensure_migrations_table_exists()
    
    applied_migrations = get_applied_migrations()
    
    if not applied_migrations:
        logger.info("No migrations to roll back")
        return
    
    # Get the most recent migrations to roll back
    migrations_to_revert = applied_migrations[-steps:]
    
    logger.info(f"Rolling back {len(migrations_to_revert)} migrations")
    
    # Rollback in reverse order (most recent first)
    for version in reversed(migrations_to_revert):
        revert_migration(version)


def get_migration_status() -> List[Dict[str, Any]]:
    """
    Get the status of all migrations.
    
    Returns:
        List of dictionaries with migration status information
    """
    ensure_migrations_table_exists()
    
    applied_migrations = get_applied_migrations()
    available_migrations = get_available_migrations()
    
    status = []
    
    for version, name, file_path in available_migrations:
        applied = version in applied_migrations
        
        migration_info = {
            "version": version,
            "name": name,
            "status": "Applied" if applied else "Pending",
            "file": os.path.basename(file_path)
        }
        
        if applied:
            # Get additional info for applied migrations
            with db_session() as session:
                query = sa.text(f"SELECT applied_at, execution_time FROM {MIGRATIONS_TABLE} WHERE version = :version")
                result = session.execute(query, {"version": version}).fetchone()
                if result:
                    migration_info["applied_at"] = result[0].strftime("%Y-%m-%d %H:%M:%S")
                    migration_info["execution_time"] = f"{result[1]:.2f}s"
        
        status.append(migration_info)
    
    return status


def print_migration_status() -> None:
    """Print the status of all migrations in a table format."""
    status = get_migration_status()
    
    if not status:
        print("No migrations found")
        return
    
    # Calculate column widths
    version_width = max(len("Version"), max(len(m["version"]) for m in status))
    name_width = max(len("Name"), max(len(m["name"]) for m in status))
    status_width = max(len("Status"), max(len(m["status"]) for m in status))
    
    # Print header
    print(f"{'Version':<{version_width}} | {'Name':<{name_width}} | {'Status':<{status_width}} | {'Applied At':<19} | {'Time':<8}")
    print("-" * (version_width + name_width + status_width + 43))
    
    # Print migrations
    for migration in status:
        applied_at = migration.get("applied_at", "")
        execution_time = migration.get("execution_time", "")
        print(f"{migration['version']:<{version_width}} | {migration['name']:<{name_width}} | {migration['status']:<{status_width}} | {applied_at:<19} | {execution_time:<8}")


if __name__ == "__main__":
    """
    Command-line interface for migration management.
    
    Usage:
        python -m src.utils.migrations <command> [options]
        
    Commands:
        status                  - Show migration status
        create <name>           - Create a new migration
        migrate [version]       - Apply pending migrations (up to version if specified)
        rollback [steps]        - Rollback migrations (default: 1 step)
        apply <version>         - Apply a specific migration
        revert <version>        - Revert a specific migration
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Database migration utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status command
    subparsers.add_parser("status", help="Show migration status")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new migration")
    create_parser.add_argument("name", help="Name of the migration")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Apply pending migrations")
    migrate_parser.add_argument("version", nargs="?", help="Target version to migrate to")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("steps", nargs="?", type=int, default=1, help="Number of migrations to roll back")
    
    # Apply command
    apply_parser = subparsers.add_parser("apply", help="Apply a specific migration")
    apply_parser.add_argument("version", help="Migration version to apply")
    
    # Revert command
    revert_parser = subparsers.add_parser("revert", help="Revert a specific migration")
    revert_parser.add_argument("version", help="Migration version to revert")
    
    args = parser.parse_args()
    
    if args.command == "status":
        print_migration_status()
    
    elif args.command == "create":
        create_migration(args.name)
    
    elif args.command == "migrate":
        migrate(args.version)
    
    elif args.command == "rollback":
        rollback(args.steps)
    
    elif args.command == "apply":
        # Find the migration
        for version, name, file_path in get_available_migrations():
            if version == args.version:
                apply_migration(version, name, file_path)
                break
        else:
            logger.error(f"Migration version {args.version} not found")
    
    elif args.command == "revert":
        revert_migration(args.version)
    
    else:
        parser.print_help() 