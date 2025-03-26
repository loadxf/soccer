"""
Script to safely restart the API server.
This script will:
1. Check for and stop any running API server processes
2. Start a fresh API server instance
"""

import subprocess
import os
import sys
import time
import signal
import psutil
import platform
import webbrowser
from pathlib import Path

def find_api_processes():
    """Find and return any running API processes."""
    api_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check for uvicorn processes that might be our API
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info['cmdline']
                if cmdline and any('uvicorn' in cmd.lower() for cmd in cmdline if cmd):
                    if any('api' in cmd.lower() for cmd in cmdline if cmd):
                        api_processes.append(proc)
            
            # Check for run_api.bat processes
            if platform.system() == 'Windows':
                if proc.info['name'] and 'cmd.exe' in proc.info['name'].lower():
                    cmdline = proc.info['cmdline']
                    if cmdline and any('run_api.bat' in cmd.lower() for cmd in cmdline if cmd):
                        api_processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return api_processes

def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # First try to terminate gracefully
        for child in children:
            try:
                child.terminate()
            except:
                pass
        
        try:
            parent.terminate()
        except:
            pass
        
        # Give processes time to terminate
        gone, alive = psutil.wait_procs(children + [parent], timeout=3)
        
        # If still alive, force kill
        for p in alive:
            try:
                p.kill()
            except:
                pass
                
        print(f"✅ Process {pid} and its children terminated")
    except:
        print(f"❌ Failed to terminate process {pid}")

def stop_api_server():
    """Stop any running API server processes."""
    print("\n===== STOPPING API SERVER =====")
    api_processes = find_api_processes()
    
    if not api_processes:
        print("No running API server processes found.")
        return
    
    print(f"Found {len(api_processes)} API server processes.")
    
    for proc in api_processes:
        print(f"Stopping process {proc.info['pid']} ({' '.join([str(x) for x in proc.info['cmdline'][:2]])})")
        kill_process_tree(proc.info['pid'])
    
    # Wait to ensure processes are fully stopped
    time.sleep(2)
    
    # Check again
    if find_api_processes():
        print("⚠️ Warning: Some API processes may still be running.")
    else:
        print("✅ All API processes have been stopped.")

def start_api_server():
    """Start the API server using the standard method."""
    print("\n===== STARTING API SERVER =====")
    
    # Check if Python 3.11 environment is available
    api_env_exists = os.path.exists("api_env") and os.path.exists("api_env/Scripts/python.exe")
    
    if api_env_exists:
        print("Using Python 3.11 environment...")
        
        # Use absolute path for run_api.bat
        run_api_path = os.path.abspath("run_api.bat")
        print(f"Running: {run_api_path}")
        
        # Start in a new window so it stays running after this script exits
        if platform.system() == 'Windows':
            subprocess.Popen(["start", "cmd", "/k", run_api_path], shell=True)
        else:
            subprocess.Popen([run_api_path], shell=True)
    else:
        print("Using current Python environment...")
        
        if platform.system() == 'Windows':
            subprocess.Popen(["start", "cmd", "/k", sys.executable, "main.py", "api", "--start"], shell=True)
        else:
            subprocess.Popen([sys.executable, "main.py", "api", "--start"])
    
    # Wait for API to start
    print("Waiting for API server to start...")
    time.sleep(5)

def main():
    """Main function to restart the API server."""
    print("==== API SERVER RESTART UTILITY ====")
    
    try:
        # Install psutil if not available
        try:
            import psutil
        except ImportError:
            print("Installing required package: psutil")
            subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
            print("psutil installed successfully. Continuing...")
        
        # Stop any running API servers
        stop_api_server()
        
        # Start a fresh API server
        start_api_server()
        
        print("\n==== API SERVER RESTARTED ====")
        print("✅ The API server has been restarted.")
        print("API Documentation: http://127.0.0.1:8080/api/v1/docs")
        
        # Open the API docs
        webbrowser.open("http://127.0.0.1:8080/api/v1/docs")
        
    except Exception as e:
        print(f"Error restarting API server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 