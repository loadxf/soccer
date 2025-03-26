#!/usr/bin/env python3
"""
Runner script for the Soccer Prediction System performance tests.
This script provides an easy interface to run specific or all performance tests.
"""

import os
import sys
import argparse
import subprocess
import json
import time
from datetime import datetime

# Performance test modules
TEST_MODULES = [
    "test_benchmarks.py",
    "test_model_inference.py",
    "test_data_processing.py"
]

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def run_locust_test(headless=True, users=100, spawn_rate=10, runtime="60s", host="http://127.0.0.1:8000", tags=None):
    """Run a Locust load test."""
    print(f"\n{'='*80}")
    print(f"Running Locust load test with {users} users, spawn rate {spawn_rate}, runtime {runtime}")
    print(f"{'='*80}\n")
    
    cmd = [
        "locust",
        "-f", "locustfile.py",
        "--host", host
    ]
    
    if headless:
        cmd.extend([
            "--headless",
            "-u", str(users),
            "-r", str(spawn_rate),
            "--run-time", runtime
        ])
        
        if tags:
            cmd.extend(["--tags", tags])
    
    # Run the locust command
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        print("\nLocust test completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"\nError running Locust test: {e}")
        return False
    except KeyboardInterrupt:
        print("\nLocust test interrupted by user")
        return False
    
    return True


def run_pytest_benchmarks(module=None, group=None, verbose=True, json_output=True):
    """Run pytest benchmarks and save results."""
    print(f"\n{'='*80}")
    if module:
        print(f"Running benchmarks from module: {module}")
    else:
        print(f"Running all benchmarks")
    if group:
        print(f"Filtering by group: {group}")
    print(f"{'='*80}\n")
    
    # Build the command
    cmd = ["pytest"]
    
    # Add module or modules
    if module:
        cmd.append(module)
    else:
        cmd.extend(TEST_MODULES)
    
    # Add benchmark options
    cmd.append("--benchmark-only")
    
    if group:
        cmd.append(f"--benchmark-group={group}")
    
    if verbose:
        cmd.append("-v")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if json_output:
        json_path = f"benchmark_results_{timestamp}.json"
        cmd.append(f"--benchmark-json={json_path}")
    
    # Run the pytest command
    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        print("\nBenchmark tests completed successfully")
        if json_output:
            print(f"Results saved to {json_path}")
    except subprocess.CalledProcessError as e:
        print(f"\nError running benchmark tests: {e}")
        return False
    except KeyboardInterrupt:
        print("\nBenchmark tests interrupted by user")
        return False
    
    return True


def run_all_tests(host="http://127.0.0.1:8000"):
    """Run all performance tests (benchmarks and load tests)."""
    print(f"\n{'='*80}")
    print(f"Running all performance tests")
    print(f"{'='*80}\n")
    
    # First run all benchmarks
    benchmark_success = run_pytest_benchmarks(verbose=True)
    
    # Then run load tests
    load_test_success = run_locust_test(
        headless=True, 
        users=50, 
        spawn_rate=5, 
        runtime="30s",
        host=host
    )
    
    return benchmark_success and load_test_success


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Soccer Prediction System Performance Tests")
    
    # Create subparsers for different test types
    subparsers = parser.add_subparsers(dest="command", help="Test type")
    
    # Benchmark tests parser
    benchmark_parser = subparsers.add_parser("benchmark", help="Run component benchmark tests")
    benchmark_parser.add_argument("--module", choices=TEST_MODULES, help="Specific test module to run")
    benchmark_parser.add_argument("--group", help="Benchmark group to run")
    benchmark_parser.add_argument("--no-json", action="store_true", help="Disable JSON output")
    
    # Load tests parser
    load_parser = subparsers.add_parser("load", help="Run API load tests with Locust")
    load_parser.add_argument("--no-headless", action="store_true", help="Run in UI mode")
    load_parser.add_argument("--users", type=int, default=100, help="Number of users to simulate")
    load_parser.add_argument("--spawn-rate", type=int, default=10, help="User spawn rate per second")
    load_parser.add_argument("--runtime", default="60s", help="Test run time (e.g. 60s, 10m)")
    load_parser.add_argument("--host", default="http://127.0.0.1:8000", help="Target host")
    load_parser.add_argument("--tags", help="Comma-separated list of tags to include")
    
    # All tests parser
    all_parser = subparsers.add_parser("all", help="Run all performance tests")
    all_parser.add_argument("--host", default="http://127.0.0.1:8000", help="Target host for load tests")
    
    # Parse args
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return
    
    # Run specified tests
    if args.command == "benchmark":
        run_pytest_benchmarks(
            module=args.module,
            group=args.group,
            verbose=True,
            json_output=not args.no_json
        )
    elif args.command == "load":
        run_locust_test(
            headless=not args.no_headless,
            users=args.users,
            spawn_rate=args.spawn_rate,
            runtime=args.runtime,
            host=args.host,
            tags=args.tags
        )
    elif args.command == "all":
        run_all_tests(host=args.host)


if __name__ == "__main__":
    main() 