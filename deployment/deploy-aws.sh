#!/bin/bash
set -e

# Soccer Prediction System - AWS Deployment Script
# This script automates the deployment of the Soccer Prediction System to AWS

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
    echo -e "  -p, --profile PROFILE       AWS profile to use"
    echo -e "  -r, --region REGION         AWS region to deploy to (default: us-east-1)"
    echo -e "  -e, --environment ENV       Deployment environment (default: dev)"
    echo -e "  -n, --project-name NAME     Project name (default: soccer-prediction)"
    echo -e "  -d, --destroy               Destroy the infrastructure instead of creating it"
    echo -e "  -s, --skip-infra            Skip infrastructure deployment"
    echo -e "  -b, --skip-backend          Skip backend deployment"
    echo -e "  -f, --skip-frontend         Skip frontend deployment"
    echo -e "  -v, --verbose               Enable verbose output"
    echo
    echo -e "${BOLD}Examples:${NC}"
    echo -e "  $0 --profile myprofile --region us-west-2"
    echo -e "  $0 --environment production"
    echo -e "  $0 --destroy"
}

# Default values
AWS_PROFILE=""
AWS_REGION="us-east-1"
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
        -p|--profile)
            AWS_PROFILE="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
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
        -s|--skip-infra)
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

# Set AWS profile if provided
if [ -n "$AWS_PROFILE" ]; then
    export AWS_PROFILE="$AWS_PROFILE"
    echo -e "${YELLOW}Using AWS profile: ${BOLD}$AWS_PROFILE${NC}"
fi

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

if ! command_exists aws; then
    echo -e "${RED}Error: AWS CLI is not installed. Please install the AWS CLI.${NC}"
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

if ! command_exists jq; then
    echo -e "${RED}Error: jq is not installed. Please install jq.${NC}"
    exit 1
fi

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AWS_DIR="$SCRIPT_DIR/aws"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to the AWS directory
cd "$AWS_DIR"

# Check if terraform.tfvars exists, if not create from example
if [ ! -f terraform.tfvars ]; then
    if [ -f terraform.tfvars.example ]; then
        echo -e "${YELLOW}Creating terraform.tfvars from example...${NC}"
        cp terraform.tfvars.example terraform.tfvars
        # Replace default values
        sed -i "s/aws_region = .*/aws_region = \"$AWS_REGION\"/" terraform.tfvars
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
        terraform destroy -auto-approve -var="aws_region=$AWS_REGION" -var="environment=$ENVIRONMENT" -var="project_name=$PROJECT_NAME"
        echo -e "${GREEN}Infrastructure destroyed successfully!${NC}"
        exit 0
    else
        echo -e "${YELLOW}Initializing Terraform...${NC}"
        run_cmd "terraform init"

        echo -e "${YELLOW}Planning infrastructure deployment...${NC}"
        run_cmd "terraform plan -var=\"aws_region=$AWS_REGION\" -var=\"environment=$ENVIRONMENT\" -var=\"project_name=$PROJECT_NAME\" -out=tfplan"

        echo -e "${YELLOW}Deploying infrastructure...${NC}"
        run_cmd "terraform apply tfplan"

        echo -e "${GREEN}Infrastructure deployed successfully!${NC}"
        
        # Get output values
        BACKEND_ECR_REPO=$(terraform output -raw backend_ecr_repository_url)
        FRONTEND_ECR_REPO=$(terraform output -raw frontend_ecr_repository_url || echo "")
        S3_BUCKET=$(terraform output -raw s3_bucket_name)
        ALB_DNS_NAME=$(terraform output -raw alb_dns_name)
        CLOUDFRONT_DOMAIN=$(terraform output -raw cloudfront_domain_name || echo "")
        RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
    fi
else
    echo -e "${YELLOW}Skipping infrastructure deployment...${NC}"
    
    # Get output values even when skipping infrastructure
    if [[ -f .terraform.lock.hcl ]]; then
        BACKEND_ECR_REPO=$(terraform output -raw backend_ecr_repository_url 2>/dev/null || echo "")
        FRONTEND_ECR_REPO=$(terraform output -raw frontend_ecr_repository_url 2>/dev/null || echo "")
        S3_BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null || echo "")
        ALB_DNS_NAME=$(terraform output -raw alb_dns_name 2>/dev/null || echo "")
        CLOUDFRONT_DOMAIN=$(terraform output -raw cloudfront_domain_name 2>/dev/null || echo "")
        RDS_ENDPOINT=$(terraform output -raw rds_endpoint 2>/dev/null || echo "")
    else
        echo -e "${RED}Error: Terraform not initialized, cannot get output values.${NC}"
        echo -e "${RED}Please run without --skip-infra first, or run terraform init manually.${NC}"
        exit 1
    fi
fi

# Backend deployment
if [ "$SKIP_BACKEND" = false ] && [ -n "$BACKEND_ECR_REPO" ]; then
    echo -e "${YELLOW}Deploying backend application...${NC}"
    
    # Get ECR login
    echo -e "${YELLOW}Logging in to ECR...${NC}"
    run_cmd "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $BACKEND_ECR_REPO"
    
    # Build backend Docker image
    echo -e "${YELLOW}Building backend Docker image...${NC}"
    cd "$PROJECT_ROOT"
    run_cmd "docker build -t $BACKEND_ECR_REPO:latest ."
    
    # Push image to ECR
    echo -e "${YELLOW}Pushing backend Docker image to ECR...${NC}"
    run_cmd "docker push $BACKEND_ECR_REPO:latest"
    
    # Restart ECS service to pick up the new image
    echo -e "${YELLOW}Updating ECS service...${NC}"
    SERVICE_NAME="${PROJECT_NAME}-backend-service-${ENVIRONMENT}"
    CLUSTER_NAME="${PROJECT_NAME}-cluster-${ENVIRONMENT}"
    run_cmd "aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --force-new-deployment --region $AWS_REGION"
    
    echo -e "${GREEN}Backend deployed successfully!${NC}"
else
    echo -e "${YELLOW}Skipping backend deployment...${NC}"
fi

# Frontend deployment
if [ "$SKIP_FRONTEND" = false ] && [ -n "$S3_BUCKET" ]; then
    echo -e "${YELLOW}Deploying frontend application...${NC}"
    
    # Navigate to frontend directory
    cd "$PROJECT_ROOT/src/frontend"
    
    # Install dependencies and build
    echo -e "${YELLOW}Building frontend...${NC}"
    run_cmd "npm install"
    run_cmd "npm run build"
    
    # Deploy to S3
    echo -e "${YELLOW}Uploading frontend to S3...${NC}"
    run_cmd "aws s3 sync build/ s3://$S3_BUCKET --delete"
    
    # Invalidate CloudFront cache if CloudFront is used
    if [ -n "$CLOUDFRONT_DOMAIN" ]; then
        echo -e "${YELLOW}Invalidating CloudFront cache...${NC}"
        DISTRIBUTION_ID=$(aws cloudfront list-distributions --query "DistributionList.Items[?DomainName=='$CLOUDFRONT_DOMAIN'].Id" --output text)
        if [ -n "$DISTRIBUTION_ID" ]; then
            run_cmd "aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths \"/*\""
        else
            echo -e "${RED}Warning: Could not find CloudFront distribution ID.${NC}"
        fi
    fi
    
    echo -e "${GREEN}Frontend deployed successfully!${NC}"
else
    echo -e "${YELLOW}Skipping frontend deployment...${NC}"
fi

# Print summary
echo -e "\n${GREEN}${BOLD}Deployment Summary:${NC}"
echo -e "${BOLD}Environment:${NC} $ENVIRONMENT"
echo -e "${BOLD}Region:${NC} $AWS_REGION"
echo -e "${BOLD}Project Name:${NC} $PROJECT_NAME"

if [ -n "$ALB_DNS_NAME" ]; then
    echo -e "${BOLD}Backend API URL:${NC} http://$ALB_DNS_NAME/api/v1"
fi

if [ -n "$CLOUDFRONT_DOMAIN" ]; then
    echo -e "${BOLD}Frontend URL:${NC} https://$CLOUDFRONT_DOMAIN"
elif [ -n "$S3_BUCKET" ]; then
    echo -e "${BOLD}Frontend URL:${NC} http://$S3_BUCKET.s3-website-$AWS_REGION.amazonaws.com"
fi

if [ -n "$RDS_ENDPOINT" ]; then
    echo -e "${BOLD}Database Endpoint:${NC} $RDS_ENDPOINT"
fi

echo -e "\n${GREEN}${BOLD}Deployment completed successfully!${NC}" 