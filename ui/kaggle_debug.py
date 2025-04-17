#!/usr/bin/env python3
"""
Kaggle Credentials Debug Script

This script helps diagnose issues with Kaggle credentials in a Docker environment.
"""

import os
import sys
import json
import traceback

def main():
    print("Kaggle Credentials Debug Script")
    print("===============================")
    
    # Check Python environment
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Check environment variables
    print("\nEnvironment Variables:")
    print(f"KAGGLE_USERNAME: {'Set' if os.environ.get('KAGGLE_USERNAME') else 'Not set'}")
    print(f"KAGGLE_KEY: {'Set' if os.environ.get('KAGGLE_KEY') else 'Not set'}")
    print(f"KAGGLE_CONFIG_DIR: {os.environ.get('KAGGLE_CONFIG_DIR', 'Not set')}")
    
    # Check for kaggle.json in common locations
    print("\nChecking for kaggle.json:")
    locations = [
        "/root/.kaggle/kaggle.json",
        os.path.expanduser("~/.kaggle/kaggle.json"),
        "./.kaggle/kaggle.json"
    ]
    
    for loc in locations:
        if os.path.exists(loc):
            print(f"✅ Found at: {loc}")
            # Check permissions
            try:
                perms = oct(os.stat(loc).st_mode)[-3:]
                print(f"   File permissions: {perms}")
                
                # Check if file is valid JSON
                try:
                    with open(loc, 'r') as f:
                        creds = json.load(f)
                    username = creds.get('username', 'Not found')
                    key = creds.get('key', 'Not found')
                    print(f"   Username: {username}")
                    print(f"   Key: {'*' * 5 + key[-4:] if key != 'Not found' else key}")
                    
                    # Set environment variables
                    os.environ['KAGGLE_USERNAME'] = creds.get('username', '')
                    os.environ['KAGGLE_KEY'] = creds.get('key', '')
                    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(loc)
                    
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON in {loc}")
                except Exception as e:
                    print(f"❌ Error reading {loc}: {str(e)}")
            except Exception as e:
                print(f"❌ Error checking {loc}: {str(e)}")
        else:
            print(f"❌ Not found at: {loc}")
    
    # Try importing and authenticating with Kaggle
    print("\nTesting Kaggle API:")
    try:
        print("Importing kaggle module...")
        import kaggle
        print("✅ Successfully imported kaggle module")
        
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        print("Authenticating with Kaggle API...")
        api.authenticate()
        print("✅ Successfully authenticated with Kaggle API")
        
        # Try listing datasets
        print("Listing datasets...")
        datasets = api.dataset_list(search="soccer", max_size=5)
        print(f"✅ Successfully listed {len(datasets)} datasets")
        for i, dataset in enumerate(datasets):
            print(f"   {i+1}. {dataset.ref}: {dataset.title}")
            
    except ImportError as e:
        print(f"❌ Failed to import Kaggle: {str(e)}")
    except Exception as e:
        print(f"❌ Error with Kaggle API: {str(e)}")
        print(traceback.format_exc())
    
    print("\nDebug complete")

if __name__ == "__main__":
    main() 