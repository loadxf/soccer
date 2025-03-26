#!/usr/bin/env python
"""
Script to run both the Soccer Prediction System API and UI
"""

import os
import sys
import time
import subprocess
import signal
import threading
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root))

from config.default_config import HOST, PORT, DEBUG

def run_api():
    """Run the API server"""
    print("Starting API server...")
    api_process = subprocess.Popen(
        ["python", "main.py", "api", "--start", "--host", HOST, "--port", str(PORT)],
        cwd=str(project_root)
    )
    return api_process

def run_ui():
    """Run the UI server"""
    print("Starting UI...")
    
    # Check if we're in a virtual environment and use the appropriate streamlit path
    venv_path = os.environ.get('VIRTUAL_ENV')
    is_windows = sys.platform.startswith('win')
    
    if venv_path:
        # If in virtual environment, use its streamlit executable
        if is_windows:
            streamlit_cmd = os.path.join(venv_path, 'Scripts', 'streamlit.exe')
        else:
            streamlit_cmd = os.path.join(venv_path, 'bin', 'streamlit')
    else:
        # Check if .venv directory exists at project root
        venv_dir = os.path.join(project_root, '.venv')
        if os.path.exists(venv_dir):
            if is_windows:
                streamlit_cmd = os.path.join(venv_dir, 'Scripts', 'streamlit.exe') 
            else:
                streamlit_cmd = os.path.join(venv_dir, 'bin', 'streamlit')
        else:
            # Fall back to global streamlit
            streamlit_cmd = "streamlit"
    
    # Check if the streamlit command exists
    if not os.path.exists(streamlit_cmd) and not streamlit_cmd == "streamlit":
        print(f"Warning: Streamlit not found at {streamlit_cmd}")
        print("Trying alternative methods to locate streamlit...")
        
        # Try using Python module directly
        streamlit_cmd = [sys.executable, "-m", "streamlit"]
    else:
        streamlit_cmd = [streamlit_cmd]
    
    try:
        # Run with streamlit command we found
        ui_process = subprocess.Popen(
            streamlit_cmd + ["run", "ui/app.py"],
            cwd=str(project_root)
        )
        return ui_process
    except FileNotFoundError:
        print("Error: Could not find streamlit. Please install it with 'pip install streamlit'")
        return None

def main():
    """Main function"""
    print("Soccer Prediction System with UI")
    print("================================")
    
    # Start API server
    api_process = run_api()
    
    # Wait for API to start
    print("Waiting for API to start...")
    time.sleep(5)
    
    # Start UI
    ui_process = run_ui()
    
    if ui_process is None:
        print("Failed to start UI. Stopping API...")
        if api_process:
            api_process.terminate()
        return
    
    try:
        print("\nServers running:")
        print(f"- API: http://{HOST}:{PORT}/api/v1/")
        print(f"- API Documentation: http://{HOST}:{PORT}/api/v1/docs")
        print("- UI: http://127.0.0.1:8501/")
        print("\nPress Ctrl+C to stop both servers...")
        
        # Keep running until interrupted
        api_process.wait()
    
    except KeyboardInterrupt:
        print("\nStopping servers...")
    
    finally:
        # Stop both processes
        if api_process:
            api_process.terminate()
        
        if ui_process:
            ui_process.terminate()
        
        print("Servers stopped.")

if __name__ == "__main__":
    main() 