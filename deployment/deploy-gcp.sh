#!/bin/bash
set -e

# Soccer Prediction System - GCP Deployment Script
# This script automates the deployment of the Soccer Prediction System to Google Cloud Platform

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
    echo -e "  -p, --project PROJECT_ID    GCP project ID"
    echo -e "  -r, --region REGION         GCP region to deploy to (default: us-central1)"
    echo -e "  -z, --zone ZONE             GCP zone to deploy to (default: us-central1-a)"
    echo -e "  -e, --environment ENV       Deployment environment (default: dev)"
    echo -e "  -n, --project-name NAME     Project name (default: soccer-prediction)"
    echo -e "  -d, --destroy               Destroy the infrastructure instead of creating it"
    echo -e "  -i, --skip-infra            Skip infrastructure deployment"
    echo -e "  -b, --skip-backend          Skip backend deployment"
    echo -e "  -f, --skip-frontend         Skip frontend deployment"
    echo -e "  -v, --verbose               Enable verbose output"
    echo
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  $0 --project soccer-prediction-123 --region us-west1 --zone us-west1-b"
    echo -e "  $0 --environment production"
    echo -e "  $0 --destroy"
}

# Default values
GCP_PROJECT_ID=""
GCP_REGION="us-central1"
GCP_ZONE="us-central1-a"
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
        -p|--project)
            GCP_PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            GCP_REGION="$2"
            shift 2
            ;;
        -z|--zone)
            GCP_ZONE="$2"
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

if ! command_exists gcloud; then
    echo -e "${RED}Error: Google Cloud SDK is not installed. Please install the gcloud CLI.${NC}"
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
GCP_DIR="$SCRIPT_DIR/gcp"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Set GCP project if provided
if [ -z "$GCP_PROJECT_ID" ]; then
    # Try to get the current GCP project
    GCP_PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$GCP_PROJECT_ID" ]; then
        echo -e "${RED}Error: No GCP project ID provided or configured.${NC}"
        echo -e "${YELLOW}Use --project to specify a project ID or run 'gcloud init' to configure a default project.${NC}"
        exit 1
    else
        echo -e "${YELLOW}Using current GCP project: ${BOLD}$GCP_PROJECT_ID${NC}"
    fi
else
    echo -e "${YELLOW}Setting GCP project to: ${BOLD}$GCP_PROJECT_ID${NC}"
    run_cmd "gcloud config set project $GCP_PROJECT_ID"
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
    echo -e "${YELLOW}You are not logged in to GCP. Running 'gcloud auth login'...${NC}"
    gcloud auth login
fi

# Change to the GCP directory
cd "$GCP_DIR"

# Check if terraform.tfvars exists, if not create from example
if [ ! -f terraform.tfvars ]; then
    if [ -f terraform.tfvars.example ]; then
        echo -e "${YELLOW}Creating terraform.tfvars from example...${NC}"
        cp terraform.tfvars.example terraform.tfvars
        # Replace default values
        sed -i "s/project_id = .*/project_id = \"$GCP_PROJECT_ID\"/" terraform.tfvars
        sed -i "s/region = .*/region = \"$GCP_REGION\"/" terraform.tfvars
        sed -i "s/zone = .*/zone = \"$GCP_ZONE\"/" terraform.tfvars
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
        run_cmd "terraform init"
        run_cmd "terraform destroy -auto-approve -var=\"project_id=$GCP_PROJECT_ID\" -var=\"region=$GCP_REGION\" -var=\"zone=$GCP_ZONE\" -var=\"environment=$ENVIRONMENT\" -var=\"project_name=$PROJECT_NAME\""
        echo -e "${GREEN}Infrastructure destroyed successfully!${NC}"
        exit 0
    else
        echo -e "${YELLOW}Initializing Terraform...${NC}"
        run_cmd "terraform init"

        echo -e "${YELLOW}Planning infrastructure deployment...${NC}"
        run_cmd "terraform plan -var=\"project_id=$GCP_PROJECT_ID\" -var=\"region=$GCP_REGION\" -var=\"zone=$GCP_ZONE\" -var=\"environment=$ENVIRONMENT\" -var=\"project_name=$PROJECT_NAME\" -out=tfplan"

        echo -e "${YELLOW}Deploying infrastructure...${NC}"
        run_cmd "terraform apply tfplan"

        echo -e "${GREEN}Infrastructure deployed successfully!${NC}"
        
        # Get output values
        GCR_REPOSITORY=$(terraform output -raw gcr_repository_url)
        GKE_CLUSTER_NAME=$(terraform output -raw gke_cluster_name)
        GKE_CLUSTER_LOCATION=$(terraform output -raw gke_cluster_location)
        CLOUD_STORAGE_BUCKET=$(terraform output -raw cloud_storage_bucket)
        CLOUD_CDN_URL=$(terraform output -raw cloud_cdn_url || echo "")
        CLOUD_SQL_INSTANCE=$(terraform output -raw cloud_sql_instance)
        CLOUD_SQL_CONNECTION_NAME=$(terraform output -raw cloud_sql_connection_name)
    fi
else
    echo -e "${YELLOW}Skipping infrastructure deployment...${NC}"
    
    # Get output values even when skipping infrastructure
    if [[ -f .terraform.lock.hcl ]]; then
        GCR_REPOSITORY=$(terraform output -raw gcr_repository_url 2>/dev/null || echo "")
        GKE_CLUSTER_NAME=$(terraform output -raw gke_cluster_name 2>/dev/null || echo "")
        GKE_CLUSTER_LOCATION=$(terraform output -raw gke_cluster_location 2>/dev/null || echo "")
        CLOUD_STORAGE_BUCKET=$(terraform output -raw cloud_storage_bucket 2>/dev/null || echo "")
        CLOUD_CDN_URL=$(terraform output -raw cloud_cdn_url 2>/dev/null || echo "")
        CLOUD_SQL_INSTANCE=$(terraform output -raw cloud_sql_instance 2>/dev/null || echo "")
        CLOUD_SQL_CONNECTION_NAME=$(terraform output -raw cloud_sql_connection_name 2>/dev/null || echo "")
    else
        echo -e "${RED}Error: Terraform not initialized, cannot get output values.${NC}"
        echo -e "${RED}Please run without --skip-infra first, or run terraform init manually.${NC}"
        exit 1
    fi
fi

# Get GKE credentials if needed for backend deployment
if [ "$SKIP_BACKEND" = false ] && [ -n "$GKE_CLUSTER_NAME" ]; then
    echo -e "${YELLOW}Getting credentials for GKE cluster...${NC}"
    run_cmd "gcloud container clusters get-credentials $GKE_CLUSTER_NAME --region $GKE_CLUSTER_LOCATION --project $GCP_PROJECT_ID"
fi

# Backend deployment
if [ "$SKIP_BACKEND" = false ] && [ -n "$GCR_REPOSITORY" ]; then
    echo -e "${YELLOW}Deploying backend application...${NC}"
    
    # Configure Docker to use GCR
    echo -e "${YELLOW}Configuring Docker to use Google Container Registry...${NC}"
    run_cmd "gcloud auth configure-docker -q"
    
    # Build backend Docker image
    echo -e "${YELLOW}Building backend Docker image...${NC}"
    cd "$PROJECT_ROOT"
    IMAGE_TAG="${GCR_REPOSITORY}/backend:latest"
    run_cmd "docker build -t $IMAGE_TAG ."
    
    # Push image to GCR
    echo -e "${YELLOW}Pushing backend Docker image to GCR...${NC}"
    run_cmd "docker push $IMAGE_TAG"
    
    # Check if Kubernetes configuration exists
    K8S_DIR="$GCP_DIR/k8s"
    if [ -d "$K8S_DIR" ]; then
        echo -e "${YELLOW}Applying Kubernetes configurations...${NC}"
        
        # Update backend image in deployment
        BACKEND_DEPLOYMENT="$K8S_DIR/backend-deployment.yaml"
        if [ -f "$BACKEND_DEPLOYMENT" ]; then
            echo -e "${YELLOW}Updating backend image in deployment...${NC}"
            sed -i "s|image: .*|image: $IMAGE_TAG|g" "$BACKEND_DEPLOYMENT"
            
            # Apply Kubernetes configurations
            echo -e "${YELLOW}Applying Kubernetes configurations...${NC}"
            run_cmd "kubectl apply -f $K8S_DIR/"
            
            # Wait for deployment to be ready
            echo -e "${YELLOW}Waiting for backend deployment to be ready...${NC}"
            DEPLOYMENT_NAME="${PROJECT_NAME}-backend-${ENVIRONMENT}"
            run_cmd "kubectl rollout status deployment/$DEPLOYMENT_NAME"
        else
            echo -e "${RED}Error: Backend deployment file not found at $BACKEND_DEPLOYMENT${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: Kubernetes configuration directory not found at $K8S_DIR${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Backend deployed successfully!${NC}"
else
    echo -e "${YELLOW}Skipping backend deployment...${NC}"
fi

# Frontend deployment
if [ "$SKIP_FRONTEND" = false ] && [ -n "$CLOUD_STORAGE_BUCKET" ]; then
    echo -e "${YELLOW}Deploying frontend application...${NC}"
    
    # Navigate to frontend directory
    cd "$PROJECT_ROOT/src/frontend"
    
    # Install dependencies and build
    echo -e "${YELLOW}Building frontend...${NC}"
    run_cmd "npm install"
    run_cmd "npm run build"
    
    # Deploy to Cloud Storage
    echo -e "${YELLOW}Uploading frontend to Cloud Storage...${NC}"
    run_cmd "gsutil -m rsync -d -r build/ gs://$CLOUD_STORAGE_BUCKET/"
    
    # Set appropriate Cache-Control headers
    echo -e "${YELLOW}Setting Cache-Control headers...${NC}"
    run_cmd "gsutil -m setmeta -h \"Cache-Control:public, max-age=3600\" gs://$CLOUD_STORAGE_BUCKET/*.html"
    run_cmd "gsutil -m setmeta -h \"Cache-Control:public, max-age=86400\" gs://$CLOUD_STORAGE_BUCKET/static/**"
    
    echo -e "${GREEN}Frontend deployed successfully!${NC}"
else
    echo -e "${YELLOW}Skipping frontend deployment...${NC}"
fi

# Get backend service URL
BACKEND_URL=""
if [ -n "$GKE_CLUSTER_NAME" ]; then
    echo -e "${YELLOW}Getting backend service URL...${NC}"
    BACKEND_SERVICE="${PROJECT_NAME}-backend-service-${ENVIRONMENT}"
    BACKEND_URL=$(kubectl get service $BACKEND_SERVICE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    if [ -z "$BACKEND_URL" ]; then
        BACKEND_URL=$(kubectl get service $BACKEND_SERVICE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    fi
fi

# Print summary
echo -e "\n${GREEN}${BOLD}Deployment Summary:${NC}"
echo -e "${BOLD}Environment:${NC} $ENVIRONMENT"
echo -e "${BOLD}Project ID:${NC} $GCP_PROJECT_ID"
echo -e "${BOLD}Region:${NC} $GCP_REGION"
echo -e "${BOLD}Project Name:${NC} $PROJECT_NAME"

if [ -n "$BACKEND_URL" ]; then
    echo -e "${BOLD}Backend API URL:${NC} http://$BACKEND_URL/api/v1"
fi

if [ -n "$CLOUD_CDN_URL" ]; then
    echo -e "${BOLD}Frontend URL:${NC} $CLOUD_CDN_URL"
elif [ -n "$CLOUD_STORAGE_BUCKET" ]; then
    echo -e "${BOLD}Frontend URL:${NC} https://storage.googleapis.com/$CLOUD_STORAGE_BUCKET/index.html"
fi

if [ -n "$CLOUD_SQL_INSTANCE" ]; then
    echo -e "${BOLD}Database Instance:${NC} $CLOUD_SQL_INSTANCE"
fi

if [ -n "$CLOUD_SQL_CONNECTION_NAME" ]; then
    echo -e "${BOLD}SQL Connection Name:${NC} $CLOUD_SQL_CONNECTION_NAME"
fi

echo -e "\n${GREEN}${BOLD}Deployment completed successfully!${NC}" 