#!/usr/bin/env python3
"""
UI to API Connection Checker

This script checks the connection between the UI and API using both localhost and 127.0.0.1
to help diagnose connection issues.
"""

import requests
import sys
import json
import time
from colorama import init, Fore, Style

# Initialize colorama
init()

API_PORT = 8080
UI_PORT = 8501

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")
    print("=" * 80)

def print_success(text):
    """Print a success message."""
    print(f"{Fore.GREEN}{text}{Style.RESET_ALL}")

def print_error(text):
    """Print an error message."""
    print(f"{Fore.RED}{text}{Style.RESET_ALL}")

def print_warning(text):
    """Print a warning message."""
    print(f"{Fore.YELLOW}{text}{Style.RESET_ALL}")

def print_info(text):
    """Print an info message."""
    print(f"{Fore.WHITE}{text}{Style.RESET_ALL}")

def check_endpoint(url, description):
    """Check if an endpoint is accessible."""
    try:
        print_info(f"Testing {description}: {url}")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print_success(f"✓ Success! Status code: {response.status_code}")
            try:
                data = response.json()
                print_info(f"Response: {json.dumps(data, indent=2)}")
            except:
                print_info(f"Response: {response.text[:100]}...")
            return True
        else:
            print_error(f"✗ Failed: Status code: {response.status_code}")
            print_info(f"Response: {response.text[:100]}...")
            return False
    except requests.exceptions.ConnectionError:
        print_error(f"✗ Connection error. Server not running or refusing connection.")
        return False
    except requests.exceptions.Timeout:
        print_error(f"✗ Request timed out.")
        return False
    except Exception as e:
        print_error(f"✗ Error: {str(e)}")
        return False

def check_cors(origin_url, api_url):
    """Check CORS configuration."""
    try:
        print_info(f"Testing CORS from {origin_url} to {api_url}")
        headers = {
            'Origin': origin_url,
            'Access-Control-Request-Method': 'GET',
            'Access-Control-Request-Headers': 'Content-Type'
        }
        response = requests.options(api_url, headers=headers, timeout=5)
        
        if 'Access-Control-Allow-Origin' in response.headers:
            allow_origin = response.headers['Access-Control-Allow-Origin']
            if allow_origin == '*' or origin_url in allow_origin:
                print_success(f"✓ CORS properly configured. Allow-Origin: {allow_origin}")
                return True
            else:
                print_error(f"✗ CORS misconfigured. Allow-Origin: {allow_origin}")
                return False
        else:
            print_error(f"✗ CORS headers not found in response")
            return False
    except Exception as e:
        print_error(f"✗ CORS check error: {str(e)}")
        return False

def main():
    print_header("Soccer Prediction System - Connection Checker")
    print_info("This tool checks connections between UI and API using different hostnames")
    
    # Check API health with localhost
    localhost_api_url = f"http://localhost:{API_PORT}/api/v1/health"
    localhost_success = check_endpoint(localhost_api_url, "API health (localhost)")
    
    # Check API health with 127.0.0.1
    ip_api_url = f"http://127.0.0.1:{API_PORT}/api/v1/health"
    ip_success = check_endpoint(ip_api_url, "API health (127.0.0.1)")
    
    # Check UI status
    localhost_ui_url = f"http://localhost:{UI_PORT}/api/healthz"
    ui_success = check_endpoint(localhost_ui_url, "UI health")
    
    if not localhost_success and not ip_success:
        print_error("\nAPI server is not accessible with either hostname!")
        print_warning("Possible reasons:")
        print_info("1. API server is not running")
        print_info("2. API server is running on a different port")
        print_info("3. Firewall is blocking connections")
        print_info("\nTo start API server: python main.py api --start")
    
    if not ui_success:
        print_error("\nUI server is not accessible!")
        print_warning("Possible reasons:")
        print_info("1. UI server is not running")
        print_info("2. UI server is running on a different port")
        print_info("\nTo start UI server: python main.py ui --start")
    
    # Check CORS configuration if both servers are running
    if (localhost_success or ip_success) and ui_success:
        print_header("CORS Configuration Check")
        
        ui_localhost_origin = f"http://localhost:{UI_PORT}"
        ui_ip_origin = f"http://127.0.0.1:{UI_PORT}"
        
        if localhost_success:
            check_cors(ui_localhost_origin, localhost_api_url)
            check_cors(ui_ip_origin, localhost_api_url)
        
        if ip_success:
            check_cors(ui_localhost_origin, ip_api_url)
            check_cors(ui_ip_origin, ip_api_url)
    
    print_header("Connection Summary")
    print_info(f"API (localhost): {'✓' if localhost_success else '✗'}")
    print_info(f"API (127.0.0.1): {'✓' if ip_success else '✗'}")
    print_info(f"UI server: {'✓' if ui_success else '✗'}")
    
    if localhost_success and ip_success and ui_success:
        print_success("\nAll services are running and accessible!")
    else:
        print_warning("\nSome services are not running or have connection issues.")
        
    print_info("\nFor detailed troubleshooting, see: localhost_vs_127001_recommendations.md")

if __name__ == "__main__":
    main() 