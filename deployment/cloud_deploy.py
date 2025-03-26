#!/usr/bin/env python3
"""
Soccer Prediction System - Cloud Deployment Utility

This script provides a unified interface for deploying the Soccer Prediction System
to various cloud providers (AWS, Azure, GCP). It offers a command-line interface
with options for specific configurations and deployment types.
"""

import argparse
import os
import subprocess
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cloud_deploy")

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOYMENT_TYPES = ["full", "infra", "app", "frontend", "monitoring"]
PROVIDERS = ["aws", "azure", "gcp"]
DEFAULT_CONFIG = {
    "aws": {
        "region": "us-east-1",
        "environment": "dev",
        "project_name": "soccer-prediction"
    },
    "azure": {
        "region": "eastus",
        "environment": "dev",
        "project_name": "soccer-prediction"
    },
    "gcp": {
        "region": "us-central1",
        "zone": "us-central1-a",
        "environment": "dev",
        "project_name": "soccer-prediction"
    }
}


def check_prerequisites(provider: str) -> bool:
    """
    Check if the required tools are installed for the specified provider.
    
    Args:
        provider: The cloud provider (aws, azure, gcp)
        
    Returns:
        True if all requirements are met, False otherwise
    """
    requirements = {
        "all": ["terraform", "docker", "git"],
        "aws": ["aws"],
        "azure": ["az"],
        "gcp": ["gcloud"]
    }
    
    # Check common requirements
    missing = []
    for cmd in requirements["all"]:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    
    # Check provider-specific requirements
    for cmd in requirements.get(provider, []):
        if shutil.which(cmd) is None:
            missing.append(cmd)
    
    if missing:
        logger.error(f"Missing required tools: {', '.join(missing)}")
        logger.error(f"Please install these tools and try again.")
        return False
    
    return True


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_file: Path to the config file (optional)
        
    Returns:
        Configuration dictionary
    """
    if not config_file:
        return DEFAULT_CONFIG
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return DEFAULT_CONFIG


def save_config(config: Dict[str, Any], config_file: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_file: Path to save the config file
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config file: {e}")
        return False


def run_command(command: List[str], capture_output: bool = False) -> Union[bool, str]:
    """
    Run a shell command.
    
    Args:
        command: Command list to run
        capture_output: Whether to capture and return the command output
        
    Returns:
        True/False for success/failure, or the command output if capture_output is True
    """
    try:
        if capture_output:
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.stdout.decode('utf-8')
        else:
            subprocess.run(command, check=True)
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}")
        logger.error(f"Error: {e}")
        if capture_output:
            return e.stderr.decode('utf-8')
        return False


def deploy_aws(args: argparse.Namespace, config: Dict[str, Any]) -> bool:
    """
    Deploy to AWS.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        True if deployment successful, False otherwise
    """
    logger.info("Deploying to AWS...")
    
    # Build command
    cmd = [
        os.path.join(SCRIPT_DIR, "deploy-aws.sh"),
        "--region", args.region or config["aws"]["region"],
        "--environment", args.environment or config["aws"]["environment"],
        "--project-name", args.project_name or config["aws"]["project_name"]
    ]
    
    # Add optional arguments
    if args.profile:
        cmd.extend(["--profile", args.profile])
    if args.destroy:
        cmd.append("--destroy")
    if args.verbose:
        cmd.append("--verbose")
    
    # Add deployment type flags
    if args.deployment_type != "full":
        if args.deployment_type != "infra":
            cmd.append("--skip-infra")
        if args.deployment_type != "app":
            cmd.append("--skip-backend")
        if args.deployment_type != "frontend":
            cmd.append("--skip-frontend")
    
    # Run the AWS deployment script
    return run_command(cmd)


def deploy_azure(args: argparse.Namespace, config: Dict[str, Any]) -> bool:
    """
    Deploy to Azure.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        True if deployment successful, False otherwise
    """
    logger.info("Deploying to Azure...")
    
    # Build command
    cmd = [
        os.path.join(SCRIPT_DIR, "deploy-azure.sh"),
        "--region", args.region or config["azure"]["region"],
        "--environment", args.environment or config["azure"]["environment"],
        "--project-name", args.project_name or config["azure"]["project_name"]
    ]
    
    # Add optional arguments
    if args.subscription:
        cmd.extend(["--subscription", args.subscription])
    if args.destroy:
        cmd.append("--destroy")
    if args.verbose:
        cmd.append("--verbose")
    
    # Add deployment type flags
    if args.deployment_type != "full":
        if args.deployment_type != "infra":
            cmd.append("--skip-infra")
        if args.deployment_type != "app":
            cmd.append("--skip-backend")
        if args.deployment_type != "frontend":
            cmd.append("--skip-frontend")
    
    # Run the Azure deployment script
    return run_command(cmd)


def deploy_gcp(args: argparse.Namespace, config: Dict[str, Any]) -> bool:
    """
    Deploy to GCP.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        True if deployment successful, False otherwise
    """
    logger.info("Deploying to GCP...")
    
    # Build command
    cmd = [
        os.path.join(SCRIPT_DIR, "deploy-gcp.sh"),
        "--region", args.region or config["gcp"]["region"],
        "--zone", args.zone or config["gcp"]["zone"],
        "--environment", args.environment or config["gcp"]["environment"],
        "--project-name", args.project_name or config["gcp"]["project_name"]
    ]
    
    # Add optional arguments
    if args.project_id:
        cmd.extend(["--project", args.project_id])
    if args.destroy:
        cmd.append("--destroy")
    if args.verbose:
        cmd.append("--verbose")
    
    # Add deployment type flags
    if args.deployment_type != "full":
        if args.deployment_type != "infra":
            cmd.append("--skip-infra")
        if args.deployment_type != "app":
            cmd.append("--skip-backend")
        if args.deployment_type != "frontend":
            cmd.append("--skip-frontend")
    
    # Run the GCP deployment script
    return run_command(cmd)


def deploy(args: argparse.Namespace) -> bool:
    """
    Deploy to the specified cloud provider.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if deployment successful, False otherwise
    """
    # Load configuration
    config = load_config(args.config)
    
    # Check prerequisites
    if not check_prerequisites(args.provider):
        return False
    
    # Deploy to the specified provider
    if args.provider == "aws":
        return deploy_aws(args, config)
    elif args.provider == "azure":
        return deploy_azure(args, config)
    elif args.provider == "gcp":
        return deploy_gcp(args, config)
    else:
        logger.error(f"Unsupported provider: {args.provider}")
        return False


def setup_environment() -> bool:
    """
    Setup the deployment environment.
    
    Returns:
        True if setup successful, False otherwise
    """
    logger.info("Setting up deployment environment...")
    
    # Run deploy.sh with --setup flag
    cmd = [os.path.join(SCRIPT_DIR, "deploy.sh"), "--setup"]
    return run_command(cmd)


def get_deployment_status(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Get the status of deployed resources.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with status information
    """
    logger.info(f"Getting deployment status for {args.provider}...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Build command based on provider
    if args.provider == "aws":
        cmd = [
            "aws", "cloudformation", "describe-stacks",
            "--region", args.region or config["aws"]["region"],
            "--stack-name", f"{args.project_name or config['aws']['project_name']}-{args.environment or config['aws']['environment']}"
        ]
        if args.profile:
            cmd.extend(["--profile", args.profile])
    elif args.provider == "azure":
        cmd = [
            "az", "group", "show",
            "--name", f"{args.project_name or config['azure']['project_name']}-{args.environment or config['azure']['environment']}"
        ]
        if args.subscription:
            cmd.extend(["--subscription", args.subscription])
    elif args.provider == "gcp":
        project_id = args.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        if not project_id:
            logger.error("No GCP project ID specified. Please use --project-id or set the GOOGLE_CLOUD_PROJECT environment variable.")
            return {}
        cmd = [
            "gcloud", "deployment-manager", "deployments", "describe",
            f"{args.project_name or config['gcp']['project_name']}-{args.environment or config['gcp']['environment']}",
            "--project", project_id
        ]
    else:
        logger.error(f"Unsupported provider: {args.provider}")
        return {}
    
    # Run the command and parse the output
    output = run_command(cmd, capture_output=True)
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # Just return the raw output if it's not JSON
            return {"output": output}
    
    return {}


def main() -> int:
    """
    Main function to parse command line arguments and execute the requested action.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = argparse.ArgumentParser(
        description="Soccer Prediction System - Cloud Deployment Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # General options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--config", help="Path to configuration file")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup deployment environment")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to a cloud provider")
    deploy_parser.add_argument("--provider", required=True, choices=PROVIDERS, help="Cloud provider to deploy to")
    deploy_parser.add_argument("--deployment-type", default="full", choices=DEPLOYMENT_TYPES, help="Type of deployment to perform")
    deploy_parser.add_argument("--destroy", action="store_true", help="Destroy resources instead of creating them")
    deploy_parser.add_argument("--environment", help="Deployment environment (default based on provider)")
    deploy_parser.add_argument("--project-name", help="Project name (default based on provider)")
    deploy_parser.add_argument("--region", help="Region to deploy to (default based on provider)")
    
    # Provider-specific arguments
    deploy_parser.add_argument("--profile", help="AWS profile to use (AWS only)")
    deploy_parser.add_argument("--subscription", help="Azure subscription ID (Azure only)")
    deploy_parser.add_argument("--project-id", help="GCP project ID (GCP only)")
    deploy_parser.add_argument("--zone", help="GCP zone (GCP only)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get deployment status")
    status_parser.add_argument("--provider", required=True, choices=PROVIDERS, help="Cloud provider to get status for")
    status_parser.add_argument("--environment", help="Deployment environment (default based on provider)")
    status_parser.add_argument("--project-name", help="Project name (default based on provider)")
    status_parser.add_argument("--region", help="Region to get status for (default based on provider)")
    
    # Provider-specific arguments
    status_parser.add_argument("--profile", help="AWS profile to use (AWS only)")
    status_parser.add_argument("--subscription", help="Azure subscription ID (Azure only)")
    status_parser.add_argument("--project-id", help="GCP project ID (GCP only)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute requested command
    if args.command == "setup":
        success = setup_environment()
    elif args.command == "deploy":
        success = deploy(args)
    elif args.command == "status":
        status = get_deployment_status(args)
        if status:
            print(json.dumps(status, indent=2))
            success = True
        else:
            success = False
    else:
        parser.print_help()
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 