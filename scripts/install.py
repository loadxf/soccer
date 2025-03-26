#!/usr/bin/env python
"""
Soccer Prediction System - Installation Script
This script installs the required dependencies and sets up the environment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Soccer Prediction System installation script")
    parser.add_argument("--all", action="store_true", help="Install all dependencies, including optional ones")
    parser.add_argument("--ml", action="store_true", help="Install machine learning dependencies")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--no-venv", action="store_true", help="Don't create a virtual environment")
    return parser.parse_args()

def create_virtual_env():
    """Create a virtual environment"""
    print("Creating virtual environment...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"])
        print("Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("Failed to create virtual environment")
        return False

def install_dependencies(include_optional=False, include_ml=False, include_dev=False):
    """Install dependencies"""
    print("Installing dependencies...")
    
    # Determine pip command
    if os.path.exists(".venv"):
        if sys.platform.startswith('win'):
            pip_cmd = [os.path.join(".venv", "Scripts", "pip")]
        else:
            pip_cmd = [os.path.join(".venv", "bin", "pip")]
    else:
        pip_cmd = [sys.executable, "-m", "pip"]
    
    # Upgrade pip
    subprocess.check_call(pip_cmd + ["install", "--upgrade", "pip"])
    
    # Install core requirements
    print("Installing core dependencies...")
    subprocess.check_call(pip_cmd + ["install", "-r", "requirements.txt"])
    
    # Install optional dependencies if requested
    if include_optional:
        print("Installing optional dependencies...")
        subprocess.check_call(pip_cmd + ["install", "-r", "optional-requirements.txt"])
    
    # Install ML dependencies if requested
    if include_ml and not include_optional:  # Skip if already installing optional
        print("Installing ML dependencies...")
        # Install just the ML dependencies from the optional requirements
        subprocess.check_call(pip_cmd + ["install", "tensorflow", "xgboost", "torch", "shap", "lime"])
    
    # Install development dependencies if requested
    if include_dev:
        print("Installing development dependencies...")
        subprocess.check_call(pip_cmd + ["install", "black", "flake8", "isort", "mypy", "pytest", "pre-commit"])
    
    print("Dependencies installed successfully")

def create_directories():
    """Create necessary directories"""
    print("Creating necessary directories...")
    dirs = [
        "data/raw", 
        "data/processed", 
        "data/features", 
        "data/models", 
        "data/evaluation", 
        "data/predictions",
        "logs"
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    print("Directories created successfully")

def check_streamlit():
    """Check if Streamlit is installed and working"""
    print("Checking Streamlit installation...")
    
    if os.path.exists(".venv"):
        if sys.platform.startswith('win'):
            streamlit_cmd = os.path.join(".venv", "Scripts", "streamlit.exe")
        else:
            streamlit_cmd = os.path.join(".venv", "bin", "streamlit")
    else:
        streamlit_cmd = "streamlit"
    
    try:
        # Check if streamlit command works
        if not os.path.exists(streamlit_cmd) and streamlit_cmd != "streamlit":
            # Try using Python module
            subprocess.check_call([sys.executable, "-m", "streamlit", "--version"])
        else:
            subprocess.check_call([streamlit_cmd, "--version"])
        print("Streamlit is installed and working")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: Streamlit is not working correctly")
        return False

def main():
    """Main function"""
    args = parse_args()
    
    # Create virtual environment if needed
    if not args.no_venv and not os.path.exists(".venv"):
        if not create_virtual_env():
            print("WARNING: Continuing without virtual environment")
    
    # Install dependencies
    try:
        install_dependencies(
            include_optional=args.all,
            include_ml=args.ml,
            include_dev=args.dev
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return 1
    
    # Create directories
    create_directories()
    
    # Check Streamlit
    check_streamlit()
    
    print("\nSetup completed successfully!")
    print("\nTo activate the virtual environment:")
    if sys.platform.startswith('win'):
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")
    
    print("\nTo run the application:")
    print("  python scripts/run_with_ui.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 