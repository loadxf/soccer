#!/bin/bash
set -e

# Soccer Prediction System - Main Deployment Script
# This script serves as an entry point for all cloud provider-specific deployment scripts

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display script usage
function display_usage() {
    echo -e "${BOLD}Soccer Prediction System Deployment Tool${NC}"
    echo
    echo -e "${BOLD}Usage:${NC} $0 [OPTIONS] [PROVIDER_OPTIONS]"
    echo -e "${BOLD}Options:${NC}"
    echo -e "  -h, --help                  Display this help message"
    echo -e "  -p, --provider PROVIDER     Cloud provider to deploy to (aws, azure, gcp)"
    echo -e "  --setup                     Setup deployment environment"
    echo -e "  --version                   Display version information"
    echo
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  $0 --provider aws --region us-east-1"
    echo -e "  $0 --provider azure --subscription YOUR_SUBSCRIPTION_ID"
    echo -e "  $0 --provider gcp --project YOUR_PROJECT_ID"
    echo
    echo -e "${BOLD}For provider-specific options, run:${NC}"
    echo -e "  $0 --provider aws --help"
    echo -e "  $0 --provider azure --help"
    echo -e "  $0 --provider gcp --help"
}

# Default values
PROVIDER=""
SETUP=false

# Parse command line arguments
while [[ $# -gt 0 ]] && [[ "$1" == -* ]]; do
    key="$1"
    case $key in
        -h|--help)
            display_usage
            exit 0
            ;;
        -p|--provider)
            PROVIDER="$2"
            shift 2
            ;;
        --setup)
            SETUP=true
            shift
            ;;
        --version)
            echo -e "Soccer Prediction System Deployment Tool v1.0.0"
            exit 0
            ;;
        *)
            # If it's not one of our flags, it's probably a provider flag, so break
            break
            ;;
    esac
done

# Function to check if a command exists
function command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup deployment environment
if [ "$SETUP" = true ]; then
    echo -e "${YELLOW}Setting up deployment environment...${NC}"
    
    echo -e "${YELLOW}Checking required tools...${NC}"
    
    # Check for basic tools
    if ! command_exists git; then
        echo -e "${RED}Git is not installed. Please install Git.${NC}"
    else
        echo -e "${GREEN}Git found!${NC}"
    fi
    
    if ! command_exists terraform; then
        echo -e "${RED}Terraform is not installed. Please install Terraform v1.2.0 or higher.${NC}"
    else
        TERRAFORM_VERSION=$(terraform version -json | grep -o '"terraform_version":"[^"]*' | cut -d '"' -f 4)
        echo -e "${GREEN}Terraform found! Version: $TERRAFORM_VERSION${NC}"
    fi
    
    if ! command_exists docker; then
        echo -e "${RED}Docker is not installed. Please install Docker.${NC}"
    else
        DOCKER_VERSION=$(docker --version | cut -d ' ' -f 3 | tr -d ',')
        echo -e "${GREEN}Docker found! Version: $DOCKER_VERSION${NC}"
    fi
    
    # Check for cloud provider tools
    echo -e "\n${YELLOW}Checking cloud provider tools...${NC}"
    
    if ! command_exists aws; then
        echo -e "${RED}AWS CLI is not installed. Required for AWS deployments.${NC}"
    else
        AWS_VERSION=$(aws --version | cut -d ' ' -f 1 | cut -d '/' -f 2)
        echo -e "${GREEN}AWS CLI found! Version: $AWS_VERSION${NC}"
    fi
    
    if ! command_exists az; then
        echo -e "${RED}Azure CLI is not installed. Required for Azure deployments.${NC}"
    else
        AZ_VERSION=$(az version | grep '"azure-cli"' | cut -d ':' -f 2 | tr -d '", ')
        echo -e "${GREEN}Azure CLI found! Version: $AZ_VERSION${NC}"
    fi
    
    if ! command_exists gcloud; then
        echo -e "${RED}Google Cloud SDK is not installed. Required for GCP deployments.${NC}"
    else
        GCLOUD_VERSION=$(gcloud version | grep "Google Cloud SDK" | cut -d ' ' -f 4)
        echo -e "${GREEN}Google Cloud SDK found! Version: $GCLOUD_VERSION${NC}"
    fi
    
    echo -e "\n${YELLOW}Setting file permissions...${NC}"
    chmod +x "$SCRIPT_DIR/deploy-aws.sh" 2>/dev/null || echo -e "${RED}Could not set executable permission for deploy-aws.sh${NC}"
    chmod +x "$SCRIPT_DIR/deploy-azure.sh" 2>/dev/null || echo -e "${RED}Could not set executable permission for deploy-azure.sh${NC}"
    chmod +x "$SCRIPT_DIR/deploy-gcp.sh" 2>/dev/null || echo -e "${RED}Could not set executable permission for deploy-gcp.sh${NC}"
    
    echo -e "\n${GREEN}Setup completed!${NC}"
    exit 0
fi

# Validate provider
if [ -z "$PROVIDER" ]; then
    echo -e "${RED}Error: Please specify a cloud provider with --provider.${NC}"
    display_usage
    exit 1
fi

# Call the appropriate provider-specific script
case $PROVIDER in
    aws)
        SCRIPT_PATH="$SCRIPT_DIR/deploy-aws.sh"
        ;;
    azure)
        SCRIPT_PATH="$SCRIPT_DIR/deploy-azure.sh"
        ;;
    gcp)
        SCRIPT_PATH="$SCRIPT_DIR/deploy-gcp.sh"
        ;;
    *)
        echo -e "${RED}Error: Invalid provider '$PROVIDER'. Supported providers are: aws, azure, gcp.${NC}"
        display_usage
        exit 1
        ;;
esac

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}Error: Deployment script for $PROVIDER not found at $SCRIPT_PATH.${NC}"
    exit 1
fi

# Make sure the script is executable
chmod +x "$SCRIPT_PATH" 2>/dev/null

# Execute the provider-specific script, passing along any additional arguments
echo -e "${YELLOW}Executing deployment script for ${BOLD}$PROVIDER${NC}..."
"$SCRIPT_PATH" "$@"
EXIT_CODE=$?

# Exit with the same exit code as the provider script
exit $EXIT_CODE 