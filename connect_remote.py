#!/usr/bin/env python3
"""
Remote Connection Utility for Soccer Prediction System.

This script loads the remote server configuration and starts the UI application
connected to the remote API server.
"""

import os
import sys
import subprocess
from pathlib import Path
import time

# Set script directory
script_dir = Path(__file__).resolve().parent

def load_env_file(env_file):
    """Load environment variables from a file."""
    env_path = script_dir / env_file
    
    if not env_path.exists():
        print(f"Error: Environment file {env_file} not found.")
        return False
        
    # Parse .env.remote file and set environment variables
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Split at the first equals sign
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value
                print(f"Set environment variable: {key}={value}")
    
    return True

def check_remote_api():
    """Check if the remote API is accessible."""
    import requests
    
    remote_host = os.environ.get('REMOTE_API_HOST')
    api_port = os.environ.get('API_PORT', '8000')
    
    if not remote_host:
        print("Error: REMOTE_API_HOST environment variable not set.")
        return False
    
    # Try multiple endpoint patterns
    endpoints = ["/health", "/api/v1/health", "/api/health"]
    
    for endpoint in endpoints:
        try:
            url = f"http://{remote_host}:{api_port}{endpoint}"
            print(f"Testing API endpoint: {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ Remote API accessible at: {url}")
                print(f"Response: {response.json()}")
                return True
        except Exception as e:
            print(f"❌ Failed to connect to {url}: {str(e)}")
    
    print("❌ Could not connect to the remote API server.")
    print(f"Please verify that the API server is running at {remote_host}:{api_port}")
    print("and that the endpoint is accessible from your network.")
    return False

def start_ui_with_remote_connection():
    """Start the UI application connected to the remote API server."""
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is installed.")
    except ImportError:
        print("❌ Streamlit is not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
    
    # Set up command to run the UI
    ui_file = script_dir / "ui" / "app.py"
    
    if not ui_file.exists():
        print(f"Error: UI application file not found at {ui_file}")
        return False
    
    print("\n=== Starting UI with Remote Connection ===")
    print(f"UI will connect to API at http://{os.environ.get('REMOTE_API_HOST')}:{os.environ.get('API_PORT', '8000')}")
    print("Starting Streamlit...")
    
    # Start Streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui_file), "--server.port=8501"]
    
    # Print command for debugging
    print(f"Running command: {' '.join(cmd)}")
    
    # Run Streamlit
    process = subprocess.Popen(cmd)
    
    # Open browser
    time.sleep(2)  # Give Streamlit time to start
    try:
        import webbrowser
        webbrowser.open("http://localhost:8501")
    except:
        pass
    
    try:
        # Wait for Streamlit process to finish
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping UI application...")
        process.terminate()
    
    return True

def main():
    """Main entry point."""
    print("=== Soccer Prediction System - Remote Connection Utility ===")
    
    # Load remote environment configuration
    if not load_env_file(".env.remote"):
        sys.exit(1)
    
    # Check if remote API is accessible
    if not check_remote_api():
        response = input("Could not connect to remote API. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Start UI application
    start_ui_with_remote_connection()

if __name__ == "__main__":
    main() 