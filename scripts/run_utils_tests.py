#!/usr/bin/env python
"""
Script to run utility tests for the Soccer Prediction System.
"""

import os
import sys
import unittest
import argparse
from datetime import datetime

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project components
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("scripts.run_utils_tests")


def run_utils_tests(test_module=None, verbose=False):
    """
    Run utility module tests.
    
    Args:
        test_module: Optional name of specific test module to run
        verbose: Whether to show verbose output
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Import test modules
    from tests.utils.test_logger import TestLogger
    from tests.utils.test_auth import TestAuth
    from tests.utils.test_db import TestDatabaseUtils
    from tests.utils.test_rate_limit import TestRateLimiter, TestRateLimitMiddleware
    
    # Create test suite
    suite = unittest.TestSuite()
    
    if test_module:
        # Run specific test module
        test_modules = {
            "logger": [TestLogger],
            "auth": [TestAuth],
            "db": [TestDatabaseUtils],
            "rate_limit": [TestRateLimiter, TestRateLimitMiddleware]
        }
        
        if test_module not in test_modules:
            logger.error(f"Unknown test module: {test_module}")
            logger.info(f"Available test modules: {', '.join(test_modules.keys())}")
            return False
        
        logger.info(f"Running test module: {test_module}")
        for test_class in test_modules[test_module]:
            suite.addTest(unittest.makeSuite(test_class))
    else:
        # Run all test modules
        logger.info("Running all utility tests")
        suite.addTest(unittest.makeSuite(TestLogger))
        suite.addTest(unittest.makeSuite(TestAuth))
        suite.addTest(unittest.makeSuite(TestDatabaseUtils))
        suite.addTest(unittest.makeSuite(TestRateLimiter))
        suite.addTest(unittest.makeSuite(TestRateLimitMiddleware))
    
    # Run tests
    verbosity = 2 if verbose else 1
    start_time = datetime.now()
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Log results
    logger.info(f"Tests completed in {duration:.2f} seconds")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    # Return success status
    return len(result.errors) == 0 and len(result.failures) == 0


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run utility tests")
    parser.add_argument("--test-module", type=str, help="Specific test module to run (logger, auth, db, rate_limit)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()
    
    # Run tests
    success = run_utils_tests(args.test_module, args.verbose)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 