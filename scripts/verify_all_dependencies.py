#!/usr/bin/env python
"""
Verify All Required Dependencies
This script checks that all dependencies, including those that were previously optional,
are correctly installed and available for use.
"""

import importlib
import sys
import os
from typing import Dict, List, Tuple

def check_package(package_name: str) -> Tuple[bool, str]:
    """Check if a package is installed and get its version"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "Not installed"

def main():
    """Main function to check all dependencies"""
    print("\n=== Soccer Prediction System Dependency Check ===\n")
    
    # Define all critical packages to verify
    packages_to_check = {
        "Core ML Libraries": [
            "tensorflow", "torch", "xgboost", "lightgbm", "catboost", 
            "sklearn", "prophet", "optuna", "hyperopt"
        ],
        "Data Processing": [
            "pandas", "numpy", "scipy", "matplotlib", "seaborn", 
            "plotly", "kaggle", "dvc"
        ],
        "Web & API": [
            "fastapi", "streamlit", "requests", "beautifulsoup4", 
            "selenium", "scrapy"
        ],
        "Utilities": [
            "joblib", "tqdm", "psutil"
        ]
    }
    
    # Track overall status
    all_installed = True
    missing_packages = []
    results = {}
    
    # Check each package category
    for category, packages in packages_to_check.items():
        print(f"\n{category}:")
        print("-" * 50)
        
        results[category] = {}
        category_all_installed = True
        
        for package in packages:
            installed, version = check_package(package)
            status = f"✅ v{version}" if installed else "❌ MISSING"
            results[category][package] = {"installed": installed, "version": version}
            
            # Update status
            if not installed:
                all_installed = False
                category_all_installed = False
                missing_packages.append(package)
            
            # Print status with consistent formatting
            package_str = f"{package}".ljust(25)
            print(f"  {package_str} {status}")
        
        # Print category summary
        if category_all_installed:
            print(f"  All {category} packages installed successfully!")
        else:
            print(f"  Warning: Some {category} packages are missing!")
    
    # Print overall summary
    print("\n=== Summary ===")
    if all_installed:
        print("\n✅ SUCCESS: All required dependencies are installed!")
    else:
        print(f"\n❌ FAILURE: The following {len(missing_packages)} packages are missing:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease run: pip install -r requirements.txt")
    
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"Executable path: {sys.executable}")
    
    # Return success status
    return all_installed

if __name__ == "__main__":
    success = main()
    # Exit with appropriate code for CI/CD pipelines
    sys.exit(0 if success else 1) 