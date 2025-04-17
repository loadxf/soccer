"""
Smart startup script for the Soccer Prediction System.
This script intelligently manages service startup:
1. Checks if API server is already running before attempting to start it
2. Starts the UI with --no-api flag to prevent port conflicts
3. Opens browser with session initialization helper to prevent browser cache issues
"""

import subprocess
import sys
import time
import requests
import os
import webbrowser
import socket
from pathlib import Path

# Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_VERSION = "v1"  # Add the API version
API_CHECK_URL = f"http://127.0.0.1:{API_PORT}/api/{API_VERSION}/health"
UI_URL = "http://127.0.0.1:8501?force_reset=true"  # Force session reset
RESET_URL = str(Path("reset_app.html").absolute().as_uri())  # Use reset page first
ADVANCED_FIX_URL = str(Path("fix_session_errors.html").absolute().as_uri())  # Advanced session fix
MAX_RETRIES = 10
RETRY_DELAY = 2  # seconds
SKIP_RESET = False  # Set to True to skip the reset page

def is_port_in_use(port, host='0.0.0.0'):
    """Check if a port is already in use on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            # If we get here, the port is free
            return False
        except socket.error:
            # Port is already in use
            return True

def is_api_running():
    """Check if the API server is already running and responding to health checks."""
    try:
        response = requests.get(API_CHECK_URL, timeout=2)
        if response.status_code == 200:
            try:
                data = response.json()
                return True
            except:
                pass
        return False
    except Exception:
        return False

def is_api_ready():
    """Check if the API server is ready to accept connections."""
    try:
        print(f"Checking API at: {API_CHECK_URL}")
        response = requests.get(API_CHECK_URL, timeout=5)
        status_code = response.status_code
        print(f"Got status code: {status_code}")
        
        if status_code == 200:
            try:
                data = response.json()
                print(f"API response: {data}")
                return True
            except Exception as e:
                print(f"Error parsing API response: {e}")
                return False
        return False
    except Exception as e:
        print(f"Error checking API: {e}")
        return False

def start_api():
    """Start the API server in a separate process with intelligent handling."""
    print("\n===== API SERVER DETECTION =====")
    
    # First, check if API is already running
    if is_api_running():
        print("\n✅ API server is already running and healthy!")
        print(f"API available at: http://{API_HOST}:{API_PORT}/api/{API_VERSION}/")
        # Return a dummy process that can be terminated safely
        class DummyProcess:
            def poll(self): return None
            def terminate(self): pass
        return DummyProcess()
    
    # API not running, but check if port is in use by something else
    if is_port_in_use(API_PORT, API_HOST):
        print(f"\n⚠️ Port {API_PORT} is in use, but API health check failed!")
        print("This likely means another application is using this port.")
        print("\nOptions:")
        print("1. Stop the other application using port 8000")
        print("2. Continue without API (some features won't work)")
        
        try:
            response = input("\nDo you want to continue without the API? (y/n): ").strip().lower()
            if response == 'y':
                print("Continuing without API server...")
                class DummyProcess:
                    def poll(self): return None
                    def terminate(self): pass
                return DummyProcess()
            else:
                print("Exiting...")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(1)
    
    print("\n===== STARTING API SERVER =====")
    # Check if Python 3.11 environment is available
    api_env_exists = os.path.exists("api_env") and os.path.exists("api_env/Scripts/python.exe")
    
    if api_env_exists:
        print("Using Python 3.11 environment...")
        # Use absolute path for run_api.bat
        run_api_path = os.path.abspath("run_api.bat")
        print(f"Running: {run_api_path}")
        api_process = subprocess.Popen([run_api_path], shell=True)
    else:
        print("Using current Python environment...")
        api_process = subprocess.Popen([sys.executable, "main.py", "api", "--start"])
    
    # Wait for API to become available
    print(f"Waiting for API server to start at {API_CHECK_URL}...")
    for i in range(MAX_RETRIES):
        if is_api_ready():
            print("\n✅ API server is ready!\n")
            return api_process
        
        print(f"API not ready yet, retrying in {RETRY_DELAY} seconds... ({i+1}/{MAX_RETRIES})")
        time.sleep(RETRY_DELAY)
    
    print("\n⚠️ Warning: API server did not respond in the expected time.")
    print("The UI will be started anyway, but it might not connect to the API initially.")
    return api_process

def start_ui():
    """Start the UI application without launching its own API server."""
    print("\n===== STARTING UI APPLICATION =====")
    # Use --no-api flag to avoid starting a second API server
    # Add --server.address=0.0.0.0 parameter to make the UI accessible from any network interface
    ui_process = subprocess.Popen([sys.executable, "main.py", "ui", "--start", "--no-api", "--", "--server.address=0.0.0.0"])
    return ui_process

def open_browser():
    """
    Open the browser with the reset page first to clear cache and then redirect to the Streamlit app.
    This helps prevent 'SessionInfo not initialized' errors by ensuring a clean browser state.
    """
    print("\n===== OPENING BROWSER =====")
    
    if SKIP_RESET:
        print(f"Opening UI directly in browser with forced session reset: {UI_URL}")
        webbrowser.open(UI_URL)
    else:
        reset_file = Path("reset_app.html").absolute()
        if reset_file.exists():
            print(f"Opening browser cache reset tool first: {RESET_URL}")
            print("This will clear browser cache and storage before redirecting to the app")
            webbrowser.open(RESET_URL)
        else:
            print(f"Reset tool not found, opening UI directly: {UI_URL}")
            webbrowser.open(UI_URL)
            print("\n⚠️ If you experience a 'SessionInfo not initialized' error:")
            print("1. Run fix_session_errors.bat for an advanced browser cleaning tool")
            print("2. Run clear_browser_cache.bat for basic browser cache reset")
            print("3. Or try a hard refresh with Ctrl+F5")
    
    # Also provide direct API test links
    print("\n===== API TEST LINKS =====")
    print(f"API Health Check: http://{API_HOST}:{API_PORT}/api/{API_VERSION}/health")
    print(f"API Documentation: http://{API_HOST}:{API_PORT}/api/{API_VERSION}/docs")

def main():
    """Start both the API and UI with proper sequencing and intelligent handling."""
    print("Soccer Prediction System - Smart Startup")
    print("=========================================")
    
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-reset":
        global SKIP_RESET
        SKIP_RESET = True
        print("Skipping browser reset page (--skip-reset flag detected)")
    
    try:
        # Start API first with intelligent detection
        api_process = start_api()
        
        # Then start UI with --no-api flag
        ui_process = start_ui()
        
        # Wait to ensure UI is starting up
        time.sleep(5)
        
        # Open browser with reset mechanism to prevent SessionInfo errors
        open_browser()
        
        print("\n✅ System is running!")
        print(f"API: http://{API_HOST}:{API_PORT}/api/{API_VERSION}/")
        print(f"API Docs: http://{API_HOST}:{API_PORT}/api/{API_VERSION}/docs")
        print("UI: http://127.0.0.1:8501/")
        print("\nPress Ctrl+C to stop all services...")
        
        print("\nBrowser Troubleshooting:")
        print("========================")
        print("If you see 'Tried to use SessionInfo before it was initialized' error:")
        print("1. Run fix_session_errors.bat for an advanced browser cleaning tool")
        print("2. Run clear_browser_cache.bat to reset browser cache")
        print("3. Try accessing the UI in an incognito/private window")
        print("4. Use Ctrl+F5 to perform a hard refresh")
        print("5. Try using 127.0.0.1 instead of localhost in the URL")
        print("6. Close and reopen your browser completely\n")
        
        # Wait for either process to exit
        while api_process.poll() is None and ui_process.poll() is None:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping services...")
    finally:
        # Clean up processes
        try:
            api_process.terminate()
            ui_process.terminate()
            print("All services stopped.")
        except Exception as e:
            print(f"Error stopping services: {e}")

if __name__ == "__main__":
    main() 