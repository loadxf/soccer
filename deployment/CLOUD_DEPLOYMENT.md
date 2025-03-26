# Soccer Prediction System - Cloud Deployment

This document provides detailed instructions for deploying the Soccer Prediction System to various cloud providers using the unified cloud deployment tool.

## Overview

The Soccer Prediction System can be deployed to the following cloud providers:

- Amazon Web Services (AWS)
- Microsoft Azure
- Google Cloud Platform (GCP)

The deployment process is managed through a unified Python script that handles the coordination of cloud-specific resources and configurations.

## Prerequisites

Before deploying the Soccer Prediction System, ensure you have the following prerequisites:

### Required Tools

- **Git**: For source code management
- **Docker**: For building and pushing container images
- **Terraform** (v1.2.0+): For infrastructure provisioning
- **Python 3.8+**: For running the deployment scripts

### Cloud Provider Tools

Depending on your target cloud provider, you'll need one or more of the following:

- **AWS CLI**: For AWS deployment
- **Azure CLI**: For Azure deployment
- **Google Cloud SDK**: For GCP deployment

### Authentication

Ensure you're authenticated with your cloud provider:

- **AWS**: Configure AWS CLI with `aws configure` or use a named profile
- **Azure**: Login with `az login`
- **GCP**: Login with `gcloud auth login`

## Quick Start

For a quick deployment, follow these steps:

1. **Setup Deployment Environment**:
   ```bash
   python cloud_deploy.py setup
   ```

2. **Deploy to a Cloud Provider**:
   ```bash
   # AWS Example
   python cloud_deploy.py deploy --provider aws --region us-east-1
   
   # Azure Example
   python cloud_deploy.py deploy --provider azure --subscription YOUR_SUBSCRIPTION_ID
   
   # GCP Example
   python cloud_deploy.py deploy --provider gcp --project-id YOUR_PROJECT_ID
   ```

3. **Check Deployment Status**:
   ```bash
   python cloud_deploy.py status --provider aws
   ```

## Detailed Usage

The `cloud_deploy.py` script provides a unified interface for managing deployments across multiple cloud providers.

### General Options

- `--verbose`: Enable verbose output
- `--config`: Path to a custom configuration file

### Setup Command

The setup command verifies the environment and ensures all required tools are installed:

```bash
python cloud_deploy.py setup
```

### Deploy Command

The deploy command deploys the application to the specified cloud provider:

```bash
python cloud_deploy.py deploy [OPTIONS]
```

#### Required Arguments

- `--provider`: Cloud provider to deploy to (`aws`, `azure`, or `gcp`)

#### Common Options

- `--deployment-type`: Type of deployment to perform (`full`, `infra`, `app`, `frontend`, or `monitoring`)
- `--destroy`: Destroy resources instead of creating them
- `--environment`: Deployment environment (e.g., `dev`, `staging`, `prod`)
- `--project-name`: Project name
- `--region`: Region to deploy to

#### Provider-Specific Options

AWS:
- `--profile`: AWS profile to use

Azure:
- `--subscription`: Azure subscription ID

GCP:
- `--project-id`: GCP project ID
- `--zone`: GCP zone

### Status Command

The status command retrieves the current status of deployed resources:

```bash
python cloud_deploy.py status [OPTIONS]
```

Required and optional arguments are similar to the deploy command.

## Deployment Examples

### AWS Deployment Example

```bash
# Full deployment to AWS
python cloud_deploy.py deploy --provider aws --region us-east-1 --environment prod

# Deploy only infrastructure
python cloud_deploy.py deploy --provider aws --deployment-type infra --region us-east-1

# Destroy resources
python cloud_deploy.py deploy --provider aws --destroy --region us-east-1
```

### Azure Deployment Example

```bash
# Full deployment to Azure
python cloud_deploy.py deploy --provider azure --subscription 00000000-0000-0000-0000-000000000000 --region eastus --environment prod

# Deploy only application code
python cloud_deploy.py deploy --provider azure --deployment-type app --subscription 00000000-0000-0000-0000-000000000000
```

### GCP Deployment Example

```bash
# Full deployment to GCP
python cloud_deploy.py deploy --provider gcp --project-id soccer-prediction --region us-central1 --zone us-central1-a --environment prod

# Deploy only frontend
python cloud_deploy.py deploy --provider gcp --deployment-type frontend --project-id soccer-prediction
```

## Configuration Files

The deployment can be customized using a JSON configuration file. Create a file with the following structure:

```json
{
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
```

Then use it with:

```bash
python cloud_deploy.py --config my_config.json deploy --provider aws
```

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

## Troubleshooting

### Common Issues

1. **Missing Prerequisites**:
   - Error: "Missing required tools"
   - Solution: Install the missing tools as indicated in the error message

2. **Authentication Issues**:
   - Error: "Not authenticated" or "Permission denied"
   - Solution: Ensure you're properly authenticated with your cloud provider

3. **Deployment Failures**:
   - Error: Various error messages from the provider-specific scripts
   - Solution: Check the logs by using the `--verbose` flag, review the error message, and consult the cloud provider's documentation

### Debugging

For more detailed debugging information, use the `--verbose` flag:

```bash
python cloud_deploy.py deploy --provider aws --verbose
```

This will show detailed logs from all underlying scripts and commands.

## Continuous Integration/Deployment

The cloud deployment tool can be integrated into CI/CD pipelines:

### GitHub Actions Example

```yaml
name: Deploy to Cloud

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
          
      - name: Deploy to AWS
        run: |
          cd deployment
          python cloud_deploy.py deploy --provider aws --environment ${{ github.ref == 'refs/heads/main' && 'prod' || 'dev' }}
```

## Best Practices

1. **Environment Isolation**: Use different environments (dev, staging, prod) to isolate deployments
2. **Version Control**: Version your infrastructure code alongside your application code
3. **Secrets Management**: Never store credentials in your code; use environment variables or secret management services
4. **Regular Updates**: Keep your cloud provider tools and Terraform updated to the latest versions
5. **Monitoring**: Set up monitoring and alerting for your deployed resources

## Additional Resources

- [AWS Documentation](https://docs.aws.amazon.com/)
- [Azure Documentation](https://docs.microsoft.com/azure/)
- [GCP Documentation](https://cloud.google.com/docs/)
- [Terraform Documentation](https://www.terraform.io/docs/) 