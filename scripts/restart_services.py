#!/usr/bin/env python3
"""
Restart Docker Services Script

This script:
1. Stops all Docker services
2. Fixes permissions issues
3. Rebuilds and restarts all services
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("restart-services")

# Get project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

def run_command(command, cwd=None):
    """Run a shell command and log the output."""
    logger.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd if cwd else project_root,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.info(f"Command output: {result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"Command errors: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        return False

def fix_permissions():
    """Run the permissions fix script."""
    logger.info("Running permissions fix script")
    fix_script = script_dir / "fix_permissions.py"
    if not fix_script.exists():
        logger.error("Permissions fix script not found!")
        return False
    
    return run_command(f"python {fix_script}")

def stop_services():
    """Stop all Docker services."""
    logger.info("Stopping Docker services")
    return run_command("docker compose down")

def rebuild_services():
    """Rebuild all Docker services."""
    logger.info("Rebuilding Docker services")
    return run_command("docker compose build")

def start_services():
    """Start all Docker services."""
    logger.info("Starting Docker services")
    return run_command("docker compose up -d")

def main():
    """Main function to restart services."""
    logger.info("Starting service restart process")
    
    # Stop services
    if not stop_services():
        logger.warning("Failed to stop services, continuing anyway")
    
    # Fix permissions
    if not fix_permissions():
        logger.warning("Failed to fix permissions, continuing anyway")
    
    # Rebuild services
    if not rebuild_services():
        logger.error("Failed to rebuild services")
        return 1
    
    # Start services
    if not start_services():
        logger.error("Failed to start services")
        return 1
    
    logger.info("Service restart completed successfully")
    
    # Wait for services to be ready
    logger.info("Waiting for services to initialize...")
    time.sleep(10)
    
    # Check if UI service is running
    ui_running = run_command("docker compose ps ui | grep running")
    if ui_running:
        logger.info("UI service is running!")
        print("\n" + "-"*50)
        print("Services have been restarted successfully!")
        print("You can now access the application at:")
        print("http://localhost:8501 (or at your server's IP address)")
        print("-"*50 + "\n")
    else:
        logger.warning("UI service may not be running properly")
        print("\n" + "-"*50)
        print("Warning: UI service may not be running properly!")
        print("Check Docker logs for more information:")
        print("docker compose logs ui")
        print("-"*50 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 