#!/usr/bin/env python3
"""
API Connectivity Diagnostic Tool

This script performs comprehensive checks on the API server's endpoints
and provides detailed diagnostics about connection issues.
"""

import requests
import sys
import os
import socket
import json
import time
from pathlib import Path

# Configure colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 50}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(50)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 50}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}! {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}> {text}{Colors.ENDC}")

# Add project root to path
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

# Get hosts to test
def get_hosts_to_test():
    """Get a list of hosts to test based on environment and configuration"""
    hosts = ["localhost", "127.0.0.1"]
    
    # Check .env.remote for REMOTE_API_HOST
    env_remote_path = script_dir / ".env.remote"
    if env_remote_path.exists():
        with open(env_remote_path, 'r') as f:
            for line in f:
                if line.strip().startswith("REMOTE_API_HOST="):
                    remote_host = line.strip().split("=", 1)[1].strip()
                    if remote_host and remote_host != "YOUR_SERVER_IP_OR_HOSTNAME":
                        hosts.append(remote_host)
    
    # Check environment variable
    remote_host_env = os.environ.get("REMOTE_API_HOST")
    if remote_host_env and remote_host_env not in hosts:
        hosts.append(remote_host_env)
        
    return hosts

# Get port to test
def get_port_to_test():
    """Get port to test from environment or configuration"""
    # Check .env.remote for API_PORT
    env_remote_path = script_dir / ".env.remote"
    if env_remote_path.exists():
        with open(env_remote_path, 'r') as f:
            for line in f:
                if line.strip().startswith("API_PORT="):
                    port = line.strip().split("=", 1)[1].strip()
                    try:
                        return int(port)
                    except ValueError:
                        pass
    
    # Check environment variable
    port_env = os.environ.get("API_PORT")
    if port_env:
        try:
            return int(port_env)
        except ValueError:
            pass
    
    # Default
    return 8000

def test_endpoint(url, timeout=5):
    """Test an API endpoint and return detailed results"""
    print_info(f"Testing endpoint: {url}")
    
    start_time = time.time()
    try:
        response = requests.get(url, timeout=timeout)
        elapsed = time.time() - start_time
        
        # Process successful request
        try:
            data = response.json()
            result = {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "elapsed_ms": int(elapsed * 1000),
                "data": data,
                "error": None
            }
        except json.JSONDecodeError:
            result = {
                "success": False,
                "status_code": response.status_code,
                "elapsed_ms": int(elapsed * 1000),
                "data": None,
                "error": "Invalid JSON response"
            }
        
        if result["success"]:
            print_success(f"Success! Status {response.status_code}, {result['elapsed_ms']}ms")
        else:
            print_error(f"Failed with status {response.status_code}, {result['elapsed_ms']}ms")
        
        return result
        
    except requests.exceptions.Timeout:
        print_error(f"Timeout after {timeout} seconds")
        return {
            "success": False,
            "status_code": None,
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "data": None,
            "error": "Request timed out"
        }
    except requests.exceptions.ConnectionError:
        print_error("Connection refused or server not available")
        return {
            "success": False,
            "status_code": None,
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "data": None,
            "error": "Connection refused"
        }
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return {
            "success": False,
            "status_code": None,
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "data": None,
            "error": str(e)
        }

def test_network_connectivity(host, port=8000):
    """Test basic network connectivity to a host and port"""
    print_info(f"Testing network connectivity to {host}:{port}")
    
    # Test DNS resolution
    try:
        ip_address = socket.gethostbyname(host)
        if ip_address == host:
            print_success(f"Host {host} is already an IP address")
        else:
            print_success(f"Host {host} resolves to {ip_address}")
    except socket.gaierror:
        print_error(f"Cannot resolve hostname {host}")
        return False
    
    # Test ICMP ping (simple implementation)
    if os.name == "nt":  # Windows
        ping_cmd = f"ping -n 1 -w 1000 {host}"
    else:  # Linux/Mac
        ping_cmd = f"ping -c 1 -W 1 {host}"
    
    result = os.system(ping_cmd + " > nul 2>&1" if os.name == "nt" else ping_cmd + " > /dev/null 2>&1")
    if result == 0:
        print_success(f"Host {host} responds to ping")
    else:
        print_warning(f"Host {host} does not respond to ping (might be blocked by firewall)")
    
    # Test TCP connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect((host, port))
        print_success(f"Port {port} is open on {host}")
        s.close()
        return True
    except socket.error:
        print_error(f"Cannot connect to port {port} on {host}")
        s.close()
        return False

def main():
    print_header("API CONNECTIVITY DIAGNOSTIC TOOL")
    
    # Import API module and report configuration
    try:
        from ui.api_service import SoccerPredictionAPI, API_BASE_URL, REQUEST_TIMEOUT
        print_info("Imported API service module successfully")
        print_info(f"API_BASE_URL: {API_BASE_URL}")
        print_info(f"REQUEST_TIMEOUT: {REQUEST_TIMEOUT}")
    except ImportError as e:
        print_error(f"Failed to import API service module: {e}")
        from_ui = input("Are you running this from the 'ui' directory? (y/n): ")
        if from_ui.lower() == 'y':
            print_warning("Try running this script from the project root directory")
        sys.exit(1)
    
    # Get hosts and ports to test
    hosts = get_hosts_to_test()
    port = get_port_to_test()
    
    print_header("NETWORK CONNECTIVITY TESTS")
    
    # Test connectivity to all hosts
    for host in hosts:
        result = test_network_connectivity(host, port)
        print()
    
    print_header("API ENDPOINT TESTS")
    
    # Test multiple endpoint variations for all hosts
    results = []
    
    # Test patterns
    patterns = [
        "/health",
        "/api/v1/health",
        "/api/health"
    ]
    
    successful_endpoints = []
    
    # Test all combinations
    for host in hosts:
        for pattern in patterns:
            url = f"http://{host}:{port}{pattern}"
            result = test_endpoint(url)
            results.append((url, result))
            
            if result["success"]:
                successful_endpoints.append(url)
            
            print()
    
    # Test using API service
    print_header("API SERVICE TESTS")
    
    # Try using the API service directly
    print_info("Testing API service health check")
    try:
        api_health = SoccerPredictionAPI.check_health()
        print(f"API health result: {json.dumps(api_health, indent=2)}")
        if api_health.get("status") == "online":
            print_success("API service health check succeeded")
        else:
            print_error(f"API service returned offline status: {api_health.get('message', 'Unknown error')}")
    except Exception as e:
        print_error(f"API service health check failed with error: {e}")
    
    # Summary
    print_header("DIAGNOSTIC SUMMARY")
    
    if successful_endpoints:
        print_success(f"Found {len(successful_endpoints)} working endpoint(s):")
        for endpoint in successful_endpoints:
            print_success(f"  ✓ {endpoint}")
            
        # Suggest config updates based on working endpoints
        print_info("\nRecommended actions:")
        if "/health" in successful_endpoints[0]:
            print_info("1. Ensure ui/api_service.py SoccerPredictionAPI.check_health() uses '/health'")
        elif "/api/v1/health" in successful_endpoints[0]:
            print_info("1. Ensure ui/api_service.py SoccerPredictionAPI.check_health() uses '/api/v1/health'")
        
        print_info("2. Update ui/app.py check_api_health_with_retries() to use the working endpoint")
        print_info("3. Make sure REMOTE_API_HOST is properly set in your environment")
    else:
        print_error("No working API endpoints found!")
        print_info("\nTroubleshooting steps:")
        print_info("1. Check if the API server is running")
        print_info("2. Verify firewall settings are allowing connections")
        print_info("3. Ensure the API server is binding to the correct interface (0.0.0.0)")
        print_info("4. Check port configuration (default 8000)")
        print_info("5. Review API server logs for errors")

if __name__ == "__main__":
    main() 