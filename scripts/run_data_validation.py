#!/usr/bin/env python
"""
Script to run data validation tests for the Soccer Prediction System.
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
logger = get_logger("scripts.run_data_validation")


def run_data_validation(test_class=None, verbose=False):
    """
    Run data validation tests.
    
    Args:
        test_class: Optional name of specific test class to run
        verbose: Whether to show verbose output
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Import test modules
    from tests.data.test_data_validation import (
        TestDataAvailability,
        TestRawDataFormat,
        TestProcessedData,
        TestFeatureEngineering,
        TestEndToEndDataPipeline
    )
    
    # Create test suite
    suite = unittest.TestSuite()
    
    if test_class:
        # Run specific test class
        test_classes = {
            "availability": TestDataAvailability,
            "raw_format": TestRawDataFormat,
            "processed": TestProcessedData,
            "features": TestFeatureEngineering,
            "pipeline": TestEndToEndDataPipeline
        }
        
        if test_class not in test_classes:
            logger.error(f"Unknown test class: {test_class}")
            logger.info(f"Available test classes: {', '.join(test_classes.keys())}")
            return False
        
        logger.info(f"Running test class: {test_class}")
        suite.addTest(unittest.makeSuite(test_classes[test_class]))
    else:
        # Run all test classes
        logger.info("Running all data validation tests")
        suite.addTest(unittest.makeSuite(TestDataAvailability))
        suite.addTest(unittest.makeSuite(TestRawDataFormat))
        suite.addTest(unittest.makeSuite(TestProcessedData))
        suite.addTest(unittest.makeSuite(TestFeatureEngineering))
        suite.addTest(unittest.makeSuite(TestEndToEndDataPipeline))
    
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
    parser = argparse.ArgumentParser(description="Run data validation tests")
    parser.add_argument("--test-class", type=str, help="Specific test class to run (availability, raw_format, processed, features, pipeline)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    args = parser.parse_args()
    
    # Run tests
    success = run_data_validation(args.test_class, args.verbose)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 