#!/bin/bash
set -e

# Soccer Prediction System - Azure Deployment Script
# This script automates the deployment of the Soccer Prediction System to Azure

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display script usage
function display_usage() {
    echo -e "${BOLD}Usage:${NC} $0 [OPTIONS]"
    echo -e "${BOLD}Options:${NC}"
    echo -e "  -h, --help                  Display this help message"
    echo -e "  -s, --subscription ID       Azure subscription ID"
    echo -e "  -r, --region REGION         Azure region to deploy to (default: eastus)"
    echo -e "  -e, --environment ENV       Deployment environment (default: dev)"
    echo -e "  -n, --project-name NAME     Project name (default: soccer-prediction)"
    echo -e "  -d, --destroy               Destroy the infrastructure instead of creating it"
    echo -e "  -i, --skip-infra            Skip infrastructure deployment"
    echo -e "  -b, --skip-backend          Skip backend deployment"
    echo -e "  -f, --skip-frontend         Skip frontend deployment"
    echo -e "  -v, --verbose               Enable verbose output"
    echo
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  $0 --subscription 00000000-0000-0000-0000-000000000000 --region westeurope"
    echo -e "  $0 --environment production"
    echo -e "  $0 --destroy"
}

# Default values
SUBSCRIPTION_ID=""
AZURE_REGION="eastus"
ENVIRONMENT="dev"
PROJECT_NAME="soccer-prediction"
DESTROY=false
SKIP_INFRA=false
SKIP_BACKEND=false
SKIP_FRONTEND=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            display_usage
            exit 0
            ;;
        -s|--subscription)
            SUBSCRIPTION_ID="$2"
            shift 2
            ;;
        -r|--region)
            AZURE_REGION="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -d|--destroy)
            DESTROY=true
            shift
            ;;
        -i|--skip-infra)
            SKIP_INFRA=true
            shift
            ;;
        -b|--skip-backend)
            SKIP_BACKEND=true
            shift
            ;;
        -f|--skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            display_usage
            exit 1
            ;;
    esac
done

# Function to run commands with verbosity control
function run_cmd() {
    if [ "$VERBOSE" = true ]; then
        eval "$@"
    else
        eval "$@" > /dev/null 2>&1
    fi
}

# Function to check if a command exists
function command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if required tools are installed
echo -e "${YELLOW}Checking required tools...${NC}"

if ! command_exists terraform; then
    echo -e "${RED}Error: Terraform is not installed. Please install Terraform v1.2.0 or higher.${NC}"
    exit 1
fi

if ! command_exists az; then
    echo -e "${RED}Error: Azure CLI is not installed. Please install the Azure CLI.${NC}"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker.${NC}"
    if [ "$SKIP_BACKEND" = false ]; then
        exit 1
    else
        echo -e "${YELLOW}Warning: Docker is required for backend deployment, but --skip-backend is set.${NC}"
    fi
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AZURE_DIR="$SCRIPT_DIR/azure"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set subscription if provided
if [ -n "$SUBSCRIPTION_ID" ]; then
    echo -e "${YELLOW}Setting Azure subscription...${NC}"
    run_cmd "az account set --subscription $SUBSCRIPTION_ID"
else
    # Check if user is logged in to Azure
    if ! az account show &>/dev/null; then
        echo -e "${YELLOW}You are not logged in to Azure. Running 'az login'...${NC}"
        az login
    fi
    
    # If still no subscription ID, get the current one
    if [ -z "$SUBSCRIPTION_ID" ]; then
        SUBSCRIPTION_ID=$(az account show --query id -o tsv)
        echo -e "${YELLOW}Using current Azure subscription: ${BOLD}$SUBSCRIPTION_ID${NC}"
    fi
fi

# Change to the Azure directory
cd "$AZURE_DIR"

# Check if terraform.tfvars exists, if not create from example
if [ ! -f terraform.tfvars ]; then
    if [ -f terraform.tfvars.example ]; then
        echo -e "${YELLOW}Creating terraform.tfvars from example...${NC}"
        cp terraform.tfvars.example terraform.tfvars
        # Replace default values
        sed -i "s/location = .*/location = \"$AZURE_REGION\"/" terraform.tfvars
        sed -i "s/environment = .*/environment = \"$ENVIRONMENT\"/" terraform.tfvars
        sed -i "s/project_name = .*/project_name = \"$PROJECT_NAME\"/" terraform.tfvars
    else
        echo -e "${RED}Error: terraform.tfvars.example not found.${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Please review and edit terraform.tfvars with your specific settings.${NC}"
    echo -e "${YELLOW}Then run this script again.${NC}"
    exit 0
fi

# Infrastructure deployment
if [ "$SKIP_INFRA" = false ]; then
    if [ "$DESTROY" = true ]; then
        echo -e "${RED}${BOLD}WARNING: You are about to destroy all resources in $ENVIRONMENT environment!${NC}"
        read -p "Are you sure you want to continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Operation cancelled.${NC}"
            exit 0
        fi

        echo -e "${YELLOW}Destroying infrastructure...${NC}"
        terraform init
        terraform destroy -auto-approve -var="location=$AZURE_REGION" -var="environment=$ENVIRONMENT" -var="project_name=$PROJECT_NAME"
        echo -e "${GREEN}Infrastructure destroyed successfully!${NC}"
        exit 0
    else
        echo -e "${YELLOW}Initializing Terraform...${NC}"
        run_cmd "terraform init"

        echo -e "${YELLOW}Planning infrastructure deployment...${NC}"
        run_cmd "terraform plan -var=\"location=$AZURE_REGION\" -var=\"environment=$ENVIRONMENT\" -var=\"project_name=$PROJECT_NAME\" -out=tfplan"

        echo -e "${YELLOW}Deploying infrastructure...${NC}"
        run_cmd "terraform apply tfplan"

        echo -e "${GREEN}Infrastructure deployed successfully!${NC}"
        
        # Get output values
        ACR_NAME=$(terraform output -raw acr_name)
        ACR_LOGIN_SERVER=$(terraform output -raw acr_login_server)
        STORAGE_ACCOUNT_NAME=$(terraform output -raw storage_account_name)
        STORAGE_ACCOUNT_KEY=$(terraform output -raw storage_account_key)
        APP_SERVICE_URL=$(terraform output -raw app_service_url)
        CDN_ENDPOINT_URL=$(terraform output -raw cdn_endpoint_url || echo "")
        DB_SERVER_NAME=$(terraform output -raw db_server_name)
        DB_CONNECTION_STRING=$(terraform output -raw db_connection_string)
    fi
else
    echo -e "${YELLOW}Skipping infrastructure deployment...${NC}"
    
    # Get output values even when skipping infrastructure
    if [[ -f .terraform.lock.hcl ]]; then
        ACR_NAME=$(terraform output -raw acr_name 2>/dev/null || echo "")
        ACR_LOGIN_SERVER=$(terraform output -raw acr_login_server 2>/dev/null || echo "")
        STORAGE_ACCOUNT_NAME=$(terraform output -raw storage_account_name 2>/dev/null || echo "")
        STORAGE_ACCOUNT_KEY=$(terraform output -raw storage_account_key 2>/dev/null || echo "")
        APP_SERVICE_URL=$(terraform output -raw app_service_url 2>/dev/null || echo "")
        CDN_ENDPOINT_URL=$(terraform output -raw cdn_endpoint_url 2>/dev/null || echo "")
        DB_SERVER_NAME=$(terraform output -raw db_server_name 2>/dev/null || echo "")
        DB_CONNECTION_STRING=$(terraform output -raw db_connection_string 2>/dev/null || echo "")
    else
        echo -e "${RED}Error: Terraform not initialized, cannot get output values.${NC}"
        echo -e "${RED}Please run without --skip-infra first, or run terraform init manually.${NC}"
        exit 1
    fi
fi

# Backend deployment
if [ "$SKIP_BACKEND" = false ] && [ -n "$ACR_LOGIN_SERVER" ]; then
    echo -e "${YELLOW}Deploying backend application...${NC}"
    
    # Log in to ACR
    echo -e "${YELLOW}Logging in to Azure Container Registry...${NC}"
    run_cmd "az acr login --name $ACR_NAME"
    
    # Build backend Docker image
    echo -e "${YELLOW}Building backend Docker image...${NC}"
    cd "$PROJECT_ROOT"
    IMAGE_TAG="${ACR_LOGIN_SERVER}/soccer-prediction-backend:latest"
    run_cmd "docker build -t $IMAGE_TAG ."
    
    # Push image to ACR
    echo -e "${YELLOW}Pushing backend Docker image to ACR...${NC}"
    run_cmd "docker push $IMAGE_TAG"
    
    # Update App Service with the new image
    echo -e "${YELLOW}Updating App Service...${NC}"
    APP_SERVICE_NAME="${PROJECT_NAME}-api-${ENVIRONMENT}"
    RESOURCE_GROUP="${PROJECT_NAME}-${ENVIRONMENT}"
    run_cmd "az webapp config container set --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP --docker-custom-image-name $IMAGE_TAG --docker-registry-server-url https://${ACR_LOGIN_SERVER}"
    
    # Restart App Service
    echo -e "${YELLOW}Restarting App Service...${NC}"
    run_cmd "az webapp restart --name $APP_SERVICE_NAME --resource-group $RESOURCE_GROUP"
    
    echo -e "${GREEN}Backend deployed successfully!${NC}"
else
    echo -e "${YELLOW}Skipping backend deployment...${NC}"
fi

# Frontend deployment
if [ "$SKIP_FRONTEND" = false ] && [ -n "$STORAGE_ACCOUNT_NAME" ]; then
    echo -e "${YELLOW}Deploying frontend application...${NC}"
    
    # Navigate to frontend directory
    cd "$PROJECT_ROOT/src/frontend"
    
    # Install dependencies and build
    echo -e "${YELLOW}Building frontend...${NC}"
    run_cmd "npm install"
    run_cmd "npm run build"
    
    # Deploy to Azure Storage
    echo -e "${YELLOW}Uploading frontend to Azure Storage...${NC}"
    run_cmd "az storage blob upload-batch --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_ACCOUNT_KEY --destination '\$web' --source build/ --overwrite"
    
    # Purge CDN endpoint if CDN is used
    if [ -n "$CDN_ENDPOINT_URL" ]; then
        echo -e "${YELLOW}Purging CDN cache...${NC}"
        CDN_PROFILE_NAME="${PROJECT_NAME}-cdn-${ENVIRONMENT}"
        CDN_ENDPOINT_NAME="${PROJECT_NAME}-endpoint-${ENVIRONMENT}"
        RESOURCE_GROUP="${PROJECT_NAME}-${ENVIRONMENT}"
        run_cmd "az cdn endpoint purge --content-paths '/*' --profile-name $CDN_PROFILE_NAME --name $CDN_ENDPOINT_NAME --resource-group $RESOURCE_GROUP"
    fi
    
    echo -e "${GREEN}Frontend deployed successfully!${NC}"
else
    echo -e "${YELLOW}Skipping frontend deployment...${NC}"
fi

# Print summary
echo -e "\n${GREEN}${BOLD}Deployment Summary:${NC}"
echo -e "${BOLD}Environment:${NC} $ENVIRONMENT"
echo -e "${BOLD}Region:${NC} $AZURE_REGION"
echo -e "${BOLD}Project Name:${NC} $PROJECT_NAME"

if [ -n "$APP_SERVICE_URL" ]; then
    echo -e "${BOLD}Backend API URL:${NC} $APP_SERVICE_URL/api/v1"
fi

if [ -n "$CDN_ENDPOINT_URL" ]; then
    echo -e "${BOLD}Frontend URL:${NC} $CDN_ENDPOINT_URL"
elif [ -n "$STORAGE_ACCOUNT_NAME" ]; then
    echo -e "${BOLD}Frontend URL:${NC} https://$STORAGE_ACCOUNT_NAME.z$AZURE_REGION.web.core.windows.net"
fi

if [ -n "$DB_SERVER_NAME" ]; then
    echo -e "${BOLD}Database Server:${NC} $DB_SERVER_NAME"
fi

echo -e "\n${GREEN}${BOLD}Deployment completed successfully!${NC}" 