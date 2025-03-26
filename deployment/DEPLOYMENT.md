# Soccer Prediction System - Deployment Guide

This document describes how to deploy the Soccer Prediction System to various cloud providers.

## Prerequisites

Before deploying the Soccer Prediction System, ensure you have the following tools installed:

- **Git**: For source code management
- **Docker**: For building and pushing container images
- **Terraform** (v1.2.0+): For infrastructure provisioning
- **Cloud Provider CLI Tools**:
  - AWS CLI (for AWS deployment)
  - Azure CLI (for Azure deployment)
  - Google Cloud SDK (for GCP deployment)

You can verify your environment setup by running:

```bash
./deploy.sh --setup
```

## Deployment Overview

The Soccer Prediction System can be deployed to the following cloud providers:

- Amazon Web Services (AWS)
- Microsoft Azure
- Google Cloud Platform (GCP)

Each deployment creates the following resources:

1. **Infrastructure**:
   - Container registry for backend images
   - Storage for frontend static files
   - Managed database service
   - Load balancers/API gateways
   - Networking components (VPC, subnets, etc.)
   - Monitoring and logging resources

2. **Backend Application**:
   - Containerized API service
   - Database migrations
   - Authentication and security setup

3. **Frontend Application**:
   - Static web application
   - CDN configuration (if applicable)

## Deployment Architecture

### AWS Architecture
- **Compute**: AWS ECS (Elastic Container Service)
- **Database**: Amazon RDS (PostgreSQL)
- **Storage**: Amazon S3 for frontend
- **CDN**: Amazon CloudFront
- **Registry**: Amazon ECR
- **Monitoring**: CloudWatch

### Azure Architecture
- **Compute**: Azure App Service
- **Database**: Azure Database for PostgreSQL
- **Storage**: Azure Blob Storage for frontend
- **CDN**: Azure CDN
- **Registry**: Azure Container Registry
- **Monitoring**: Azure Monitor

### GCP Architecture
- **Compute**: Google Kubernetes Engine (GKE)
- **Database**: Cloud SQL (PostgreSQL)
- **Storage**: Cloud Storage for frontend
- **CDN**: Cloud CDN
- **Registry**: Google Container Registry
- **Monitoring**: Cloud Monitoring

## Main Deployment Script

The main entry point for deployment is the `deploy.sh` script, which provides a unified interface for deploying to any of the supported cloud providers.

### Basic Usage

```bash
# Deploy to AWS
./deploy.sh --provider aws [options]

# Deploy to Azure
./deploy.sh --provider azure [options]

# Deploy to Google Cloud
./deploy.sh --provider gcp [options]
```

### Common Options

- `--help`: Display help information
- `--setup`: Verify and setup the deployment environment
- `--version`: Display version information

## AWS Deployment

The AWS deployment script (`deploy-aws.sh`) provisions infrastructure in AWS and deploys the application.

### Prerequisites

- AWS CLI installed and configured
- IAM credentials with sufficient permissions

### Options

```bash
./deploy.sh --provider aws [OPTIONS]
```

Options:
- `--profile PROFILE`: AWS profile to use
- `--region REGION`: AWS region to deploy to (default: us-east-1)
- `--environment ENV`: Deployment environment (default: dev)
- `--project-name NAME`: Project name (default: soccer-prediction)
- `--destroy`: Destroy the infrastructure instead of creating it
- `--skip-infra`: Skip infrastructure deployment
- `--skip-backend`: Skip backend deployment
- `--skip-frontend`: Skip frontend deployment
- `--verbose`: Enable verbose output

### Example

```bash
# Deploy to AWS in us-west-2 region with production environment
./deploy.sh --provider aws --profile my-aws-profile --region us-west-2 --environment production
```

## Azure Deployment

The Azure deployment script (`deploy-azure.sh`) provisions infrastructure in Azure and deploys the application.

### Prerequisites

- Azure CLI installed and logged in
- Subscription with sufficient permissions

### Options

```bash
./deploy.sh --provider azure [OPTIONS]
```

Options:
- `--subscription ID`: Azure subscription ID
- `--region REGION`: Azure region to deploy to (default: eastus)
- `--environment ENV`: Deployment environment (default: dev)
- `--project-name NAME`: Project name (default: soccer-prediction)
- `--destroy`: Destroy the infrastructure instead of creating it
- `--skip-infra`: Skip infrastructure deployment
- `--skip-backend`: Skip backend deployment
- `--skip-frontend`: Skip frontend deployment
- `--verbose`: Enable verbose output

### Example

```bash
# Deploy to Azure in westeurope region with staging environment
./deploy.sh --provider azure --subscription 00000000-0000-0000-0000-000000000000 --region westeurope --environment staging
```

## GCP Deployment

The GCP deployment script (`deploy-gcp.sh`) provisions infrastructure in Google Cloud and deploys the application.

### Prerequisites

- Google Cloud SDK installed and configured
- Project with sufficient permissions and APIs enabled

### Options

```bash
./deploy.sh --provider gcp [OPTIONS]
```

Options:
- `--project PROJECT_ID`: GCP project ID
- `--region REGION`: GCP region to deploy to (default: us-central1)
- `--zone ZONE`: GCP zone to deploy to (default: us-central1-a)
- `--environment ENV`: Deployment environment (default: dev)
- `--project-name NAME`: Project name (default: soccer-prediction)
- `--destroy`: Destroy the infrastructure instead of creating it
- `--skip-infra`: Skip infrastructure deployment
- `--skip-backend`: Skip backend deployment
- `--skip-frontend`: Skip frontend deployment
- `--verbose`: Enable verbose output

### Example

```bash
# Deploy to GCP in us-west1 with test environment
./deploy.sh --provider gcp --project soccer-prediction-123 --region us-west1 --zone us-west1-b --environment test
```

## Customizing Deployment

### Terraform Variables

For each provider, you can customize the deployment by modifying the `terraform.tfvars` file in the provider-specific directory. If the file doesn't exist, it will be created from the example file during the first run.

Variables you can customize include:
- Instance sizes
- Storage capacities
- Networking settings
- Database configurations
- Region-specific settings

### Deploy Only Specific Components

You can deploy only specific components of the system:

```bash
# Deploy only infrastructure
./deploy.sh --provider aws --skip-backend --skip-frontend

# Deploy only backend
./deploy.sh --provider aws --skip-infra --skip-frontend

# Deploy only frontend
./deploy.sh --provider aws --skip-infra --skip-backend
```

## Destroying Resources

To destroy all deployed resources:

```bash
./deploy.sh --provider aws --destroy
```

This will prompt for confirmation before destroying any resources.

## Troubleshooting

### Common Issues

1. **Terraform State Conflicts**:
   - Error: If you see errors about Terraform state, ensure you're not running multiple deployments simultaneously.
   - Solution: Check the Terraform state files in each provider directory.

2. **Permission Issues**:
   - Error: "Access denied" or "Insufficient privileges"
   - Solution: Ensure your cloud credentials have the necessary permissions.

3. **Resource Limits**:
   - Error: "Quota exceeded" or "Resource limit reached"
   - Solution: Request quota increases from your cloud provider or use smaller/fewer resources.

4. **Network Issues**:
   - Error: Timeout when connecting to cloud APIs
   - Solution: Check your internet connection and firewall settings.

### Getting Help

If you encounter issues not covered in this document:

1. Check the detailed logs by running with the `--verbose` flag
2. Review the cloud provider's documentation
3. Contact the development team with the full error output 