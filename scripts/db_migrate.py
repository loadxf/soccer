#!/usr/bin/env python3
"""
Database Migration Script for Soccer Prediction System

This script provides a user-friendly command-line interface for managing
database migrations. It wraps the functionality in src.utils.migrations
and provides colorful output and additional user-friendly features.
"""

import os
import sys
import argparse
import datetime
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import migration utilities
from src.utils.migrations import (
    create_migration, 
    migrate, 
    rollback, 
    get_migration_status, 
    apply_migration, 
    revert_migration,
    get_available_migrations,
    ensure_migrations_table_exists
)
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("db_migrate")

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


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {text} ==={Colors.ENDC}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}! {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")


def print_migration_status() -> None:
    """Print the status of all migrations in a color-coded table."""
    status = get_migration_status()
    
    if not status:
        print_info("No migrations found")
        return
    
    # Calculate column widths
    version_width = max(len("Version"), max(len(m["version"]) for m in status))
    name_width = max(len("Name"), max(len(m["name"]) for m in status))
    status_width = max(len("Status"), max(len(m["status"]) for m in status))
    applied_width = 19  # Fixed width for datetime
    time_width = 8      # Fixed width for execution time
    
    # Print header
    header = f"{Colors.BOLD}{'Version':<{version_width}} | {'Name':<{name_width}} | {'Status':<{status_width}} | {'Applied At':<{applied_width}} | {'Time':<{time_width}}{Colors.ENDC}"
    print(header)
    print("-" * (version_width + name_width + status_width + applied_width + time_width + 12))
    
    # Print migrations
    for migration in status:
        applied_at = migration.get("applied_at", "")
        execution_time = migration.get("execution_time", "")
        
        # Color-code status
        status_str = migration["status"]
        if status_str == "Applied":
            status_colored = f"{Colors.GREEN}{status_str}{Colors.ENDC}"
        else:
            status_colored = f"{Colors.YELLOW}{status_str}{Colors.ENDC}"
        
        # Format line
        line = f"{migration['version']:<{version_width}} | {migration['name']:<{name_width}} | {status_colored:<{status_width + len(Colors.GREEN) + len(Colors.ENDC)}} | {applied_at:<{applied_width}} | {execution_time:<{time_width}}"
        print(line)


def create_new_migration(name: str) -> None:
    """Create a new migration file with user-friendly output."""
    print_header(f"Creating Migration: {name}")
    
    try:
        file_path = create_migration(name)
        print_success(f"Migration file created: {os.path.basename(file_path)}")
        print_info(f"Edit the file at: {file_path}")
    except Exception as e:
        print_error(f"Failed to create migration: {e}")
        sys.exit(1)


def run_migrations(target_version: Optional[str] = None) -> None:
    """Run pending migrations with user-friendly output."""
    if target_version:
        print_header(f"Applying Migrations (up to version {target_version})")
    else:
        print_header("Applying All Pending Migrations")
    
    try:
        # Get status before migrations
        before_status = get_migration_status()
        applied_before = {m["version"] for m in before_status if m["status"] == "Applied"}
        
        # Run migrations
        migrate(target_version)
        
        # Get status after migrations
        after_status = get_migration_status()
        applied_after = {m["version"] for m in after_status if m["status"] == "Applied"}
        
        # Calculate newly applied migrations
        newly_applied = applied_after - applied_before
        
        if newly_applied:
            print_success(f"Successfully applied {len(newly_applied)} migration(s)")
            
            # Show details of applied migrations
            for m in after_status:
                if m["version"] in newly_applied:
                    print_info(f"Applied version {m['version']}: {m['name']} ({m.get('execution_time', '?')})")
        else:
            print_info("No migrations were applied")
    
    except Exception as e:
        print_error(f"Migration failed: {e}")
        sys.exit(1)


def run_rollback(steps: int = 1) -> None:
    """Rollback migrations with user-friendly output."""
    print_header(f"Rolling Back {steps} Migration(s)")
    
    try:
        # Get status before rollback
        before_status = get_migration_status()
        applied_before = {m["version"] for m in before_status if m["status"] == "Applied"}
        
        if not applied_before:
            print_info("No migrations to roll back")
            return
        
        # Get the migrations that will be rolled back
        to_rollback = sorted(list(applied_before))[-steps:]
        
        # Confirm rollback with user
        if to_rollback:
            print_warning("The following migrations will be rolled back:")
            for version in to_rollback:
                for m in before_status:
                    if m["version"] == version:
                        print(f"  - {version}: {m['name']}")
            
            confirm = input(f"\n{Colors.YELLOW}Are you sure you want to proceed? [y/N]: {Colors.ENDC}")
            if confirm.lower() != 'y':
                print_info("Rollback cancelled")
                return
        
        # Run rollback
        rollback(steps)
        
        # Get status after rollback
        after_status = get_migration_status()
        applied_after = {m["version"] for m in after_status if m["status"] == "Applied"}
        
        # Calculate rolled back migrations
        rolled_back = applied_before - applied_after
        
        if rolled_back:
            print_success(f"Successfully rolled back {len(rolled_back)} migration(s)")
            
            # Show details of rolled back migrations
            for version in rolled_back:
                for m in before_status:
                    if m["version"] == version:
                        print_info(f"Rolled back version {version}: {m['name']}")
        else:
            print_info("No migrations were rolled back")
    
    except Exception as e:
        print_error(f"Rollback failed: {e}")
        sys.exit(1)


def apply_specific_migration(version: str) -> None:
    """Apply a specific migration."""
    # Find the migration
    for v, name, file_path in get_available_migrations():
        if v == version:
            print_header(f"Applying Migration: {version} ({name})")
            
            try:
                apply_migration(version, name, file_path)
                print_success(f"Successfully applied migration {version}")
                return
            except Exception as e:
                print_error(f"Failed to apply migration {version}: {e}")
                sys.exit(1)
    
    print_error(f"Migration version {version} not found")
    sys.exit(1)


def revert_specific_migration(version: str) -> None:
    """Revert a specific migration."""
    print_header(f"Reverting Migration: {version}")
    
    try:
        # Confirm revert with user
        confirm = input(f"{Colors.YELLOW}Are you sure you want to revert migration {version}? [y/N]: {Colors.ENDC}")
        if confirm.lower() != 'y':
            print_info("Revert cancelled")
            return
        
        revert_migration(version)
        print_success(f"Successfully reverted migration {version}")
    except Exception as e:
        print_error(f"Failed to revert migration {version}: {e}")
        sys.exit(1)


def initialize_db() -> None:
    """Initialize the database migration system."""
    print_header("Initializing Database Migration System")
    
    try:
        ensure_migrations_table_exists()
        print_success("Migration system initialized successfully")
    except Exception as e:
        print_error(f"Failed to initialize migration system: {e}")
        sys.exit(1)


def main() -> None:
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(
        description="Soccer Prediction System - Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python db_migrate.py status
  python db_migrate.py create "Add users table"
  python db_migrate.py migrate
  python db_migrate.py rollback 2
  python db_migrate.py apply 0001
  python db_migrate.py revert 0001
  python db_migrate.py init
        """
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Status command
    subparsers.add_parser(
        "status", 
        help="Show migration status"
    )
    
    # Create command
    create_parser = subparsers.add_parser(
        "create", 
        help="Create a new migration"
    )
    create_parser.add_argument(
        "name", 
        help="Name of the migration (will be converted to snake_case)"
    )
    
    # Migrate command
    migrate_parser = subparsers.add_parser(
        "migrate", 
        help="Apply pending migrations"
    )
    migrate_parser.add_argument(
        "version", 
        nargs="?", 
        help="Target version to migrate to (optional)"
    )
    
    # Rollback command
    rollback_parser = subparsers.add_parser(
        "rollback", 
        help="Rollback migrations"
    )
    rollback_parser.add_argument(
        "steps", 
        nargs="?", 
        type=int, 
        default=1, 
        help="Number of migrations to roll back (default: 1)"
    )
    
    # Apply command
    apply_parser = subparsers.add_parser(
        "apply", 
        help="Apply a specific migration"
    )
    apply_parser.add_argument(
        "version", 
        help="Migration version to apply"
    )
    
    # Revert command
    revert_parser = subparsers.add_parser(
        "revert", 
        help="Revert a specific migration"
    )
    revert_parser.add_argument(
        "version", 
        help="Migration version to revert"
    )
    
    # Init command
    subparsers.add_parser(
        "init", 
        help="Initialize the database migration system"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "status":
        print_migration_status()
    
    elif args.command == "create":
        create_new_migration(args.name)
    
    elif args.command == "migrate":
        run_migrations(args.version)
    
    elif args.command == "rollback":
        run_rollback(args.steps)
    
    elif args.command == "apply":
        apply_specific_migration(args.version)
    
    elif args.command == "revert":
        revert_specific_migration(args.version)
    
    elif args.command == "init":
        initialize_db()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 