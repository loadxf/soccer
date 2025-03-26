#!/usr/bin/env python
"""
Run CI Pipeline Script

This script simulates the CI/CD pipeline locally for testing purposes.
It runs the linting, testing, and building steps that would be performed
in the GitHub Actions workflow.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CIPipeline:
    """Class to handle CI pipeline operations."""
    
    def __init__(self, skip_lint=False, skip_tests=False, skip_build=False):
        """
        Initialize CI pipeline.
        
        Args:
            skip_lint (bool): Skip linting step
            skip_tests (bool): Skip testing step
            skip_build (bool): Skip build step
        """
        self.skip_lint = skip_lint
        self.skip_tests = skip_tests
        self.skip_build = skip_build
        
        # Configure directories
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.logs_dir = os.path.join(self.root_dir, "logs")
        
        # Ensure logs directory exists
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.lint_log = os.path.join(self.logs_dir, f"lint_{timestamp}.log")
        self.test_log = os.path.join(self.logs_dir, f"test_{timestamp}.log")
        self.build_log = os.path.join(self.logs_dir, f"build_{timestamp}.log")
        
    def run_command(self, command, log_file=None, check=True):
        """
        Run a shell command.
        
        Args:
            command (list): Command to run as list of strings
            log_file (str): Path to log file
            check (bool): Whether to check for command success
            
        Returns:
            bool: True if command succeeded, False otherwise
        """
        logger.info(f"Running command: {' '.join(command)}")
        
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command, 
                    stdout=f, 
                    stderr=subprocess.STDOUT, 
                    cwd=self.root_dir,
                    check=False
                )
        else:
            result = subprocess.run(
                command, 
                cwd=self.root_dir, 
                check=False
            )
            
        if result.returncode != 0:
            logger.error(f"Command failed with exit code {result.returncode}")
            if check:
                logger.error(f"Check {log_file} for details")
                return False
        else:
            logger.info("Command succeeded")
        
        return result.returncode == 0
    
    def run_lint(self):
        """Run linting step."""
        if self.skip_lint:
            logger.info("Skipping lint step")
            return True
        
        logger.info("Running linting")
        
        # First, run with strict rules that should fail the build
        lint_success = self.run_command(
            ["flake8", "--count", "--select=E9,F63,F7,F82", "--show-source", "."],
            self.lint_log,
            check=True
        )
        if not lint_success:
            return False
        
        # Then run with more permissive rules that are just warnings
        self.run_command(
            ["flake8", "--count", "--exit-zero", "--max-complexity=10", "--max-line-length=127", "."],
            self.lint_log,
            check=False
        )
        
        return True
    
    def run_tests(self):
        """Run testing step."""
        if self.skip_tests:
            logger.info("Skipping test step")
            return True
        
        logger.info("Running tests")
        
        test_success = self.run_command(
            ["pytest", "tests/", "--cov=src/", "--cov-report=xml", "-v"],
            self.test_log,
            check=True
        )
        
        return test_success
    
    def run_build(self):
        """Run build step."""
        if self.skip_build:
            logger.info("Skipping build step")
            return True
        
        logger.info("Running build step")
        
        build_success = self.run_command(
            ["pip", "install", "build", "wheel"],
            self.build_log,
            check=True
        )
        if not build_success:
            return False
            
        build_success = self.run_command(
            ["python", "-m", "build"],
            self.build_log,
            check=True
        )
        
        return build_success
    
    def run_pipeline(self):
        """Run full CI pipeline."""
        logger.info("Starting CI pipeline")
        
        # Run lint
        lint_success = self.run_lint()
        if not lint_success:
            logger.error("Linting failed. Pipeline stopping.")
            return False
        
        # Run tests
        test_success = self.run_tests()
        if not test_success:
            logger.error("Tests failed. Pipeline stopping.")
            return False
        
        # Run build
        build_success = self.run_build()
        if not build_success:
            logger.error("Build failed. Pipeline stopping.")
            return False
        
        logger.info("CI pipeline completed successfully")
        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run CI pipeline locally")
    parser.add_argument("--skip-lint", action="store_true", help="Skip linting step")
    parser.add_argument("--skip-tests", action="store_true", help="Skip testing step")
    parser.add_argument("--skip-build", action="store_true", help="Skip build step")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = CIPipeline(
        skip_lint=args.skip_lint,
        skip_tests=args.skip_tests,
        skip_build=args.skip_build
    )
    
    success = pipeline.run_pipeline()
    sys.exit(0 if success else 1) 