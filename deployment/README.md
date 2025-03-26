# Soccer Prediction System - Cloud Deployment Scripts

This directory contains Terraform configurations to deploy the Soccer Prediction System to different cloud providers. The system includes:

- A backend API service built with FastAPI
- A PostgreSQL database
- A frontend web application built with React
- Monitoring and logging infrastructure

## Available Cloud Providers

- [AWS (Amazon Web Services)](#aws-deployment)
- [GCP (Google Cloud Platform)](#gcp-deployment)
- [Azure (Microsoft Azure)](#azure-deployment)

## Prerequisites

Before deploying to any cloud provider, you will need:

1. [Terraform](https://www.terraform.io/downloads.html) (version >= 1.2.0)
2. Cloud provider CLI tool:
   - [AWS CLI](https://aws.amazon.com/cli/) for AWS
   - [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) for GCP
   - [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli) for Azure
3. Cloud provider account with appropriate permissions
4. Docker for building and pushing container images

## AWS Deployment

### Configuration

1. Navigate to the AWS directory:
   ```
   cd aws
   ```

2. Create a `terraform.tfvars` file based on the example:
   ```
   cp terraform.tfvars.example terraform.tfvars
   ```

3. Edit `terraform.tfvars` with your specific configuration.

### Authentication

Set up your AWS credentials:

```bash
aws configure
```

Or use environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
```

### Deployment

1. Initialize Terraform:
   ```bash
   terraform init
   ```

2. Create an execution plan:
   ```bash
   terraform plan
   ```

3. Apply the Terraform configuration:
   ```bash
   terraform apply
   ```

4. After deployment, note the outputs for resource endpoints.

### Building and Deploying Applications

1. Build and push the backend Docker image:
   ```bash
   aws ecr get-login-password --region $(terraform output -raw aws_region) | docker login --username AWS --password-stdin $(terraform output -raw backend_ecr_repository_url)
   docker build -t $(terraform output -raw backend_ecr_repository_url):latest ../../src/
   docker push $(terraform output -raw backend_ecr_repository_url):latest
   ```

2. Build and deploy the frontend:
   ```bash
   cd ../../src/frontend
   npm install
   npm run build
   aws s3 sync build/ s3://$(terraform output -raw s3_bucket_name)
   ```

3. Invalidate CloudFront cache (if needed):
   ```bash
   aws cloudfront create-invalidation --distribution-id $(terraform output -raw cloudfront_distribution_id) --paths "/*"
   ```

## GCP Deployment

### Configuration

1. Navigate to the GCP directory:
   ```
   cd gcp
   ```

2. Create a `terraform.tfvars` file based on the example:
   ```
   cp terraform.tfvars.example terraform.tfvars
   ```

3. Edit `terraform.tfvars` with your specific configuration.

### Authentication

Set up your GCP credentials:

```bash
gcloud auth login
gcloud config set project your-project-id
gcloud auth application-default login
```

### Deployment

1. Initialize Terraform:
   ```bash
   terraform init
   ```

2. Create an execution plan:
   ```bash
   terraform plan
   ```

3. Apply the Terraform configuration:
   ```bash
   terraform apply
   ```

4. After deployment, note the outputs for resource endpoints.

### Building and Deploying Applications

1. Configure kubectl to use the GKE cluster:
   ```bash
   gcloud container clusters get-credentials $(terraform output -raw kubernetes_cluster_name) --zone $(terraform output -raw zone) --project $(terraform output -raw project_id)
   ```

2. Build and push the backend Docker image:
   ```bash
   docker build -t $(terraform output -raw artifact_registry_repository)/backend:latest ../../src/
   docker push $(terraform output -raw artifact_registry_repository)/backend:latest
   ```

3. Deploy Kubernetes resources:
   ```bash
   kubectl apply -f k8s/
   ```

4. Build and deploy the frontend:
   ```bash
   cd ../../src/frontend
   npm install
   npm run build
   gsutil rsync -r build/ $(terraform output -raw bucket_url)
   ```

## Azure Deployment

### Configuration

1. Navigate to the Azure directory:
   ```
   cd azure
   ```

2. Create a `terraform.tfvars` file based on the example:
   ```
   cp terraform.tfvars.example terraform.tfvars
   ```

3. Edit `terraform.tfvars` with your specific configuration.

### Authentication

Set up your Azure credentials:

```bash
az login
az account set --subscription "your-subscription-id"
```

### Deployment

1. Initialize Terraform:
   ```bash
   terraform init
   ```

2. Create an execution plan:
   ```bash
   terraform plan
   ```

3. Apply the Terraform configuration:
   ```bash
   terraform apply
   ```

4. After deployment, note the outputs for resource endpoints.

### Building and Deploying Applications

1. Configure kubectl to use the AKS cluster:
   ```bash
   az aks get-credentials --resource-group $(terraform output -raw resource_group_name) --name $(terraform output -raw kubernetes_cluster_name)
   ```

2. Log in to Azure Container Registry:
   ```bash
   az acr login --name $(terraform output -raw acr_login_server)
   ```

3. Build and push the backend Docker image:
   ```bash
   docker build -t $(terraform output -raw acr_login_server)/backend:latest ../../src/
   docker push $(terraform output -raw acr_login_server)/backend:latest
   ```

4. Deploy Kubernetes resources:
   ```bash
   kubectl apply -f k8s-azure/
   ```

5. Build and deploy the frontend:
   ```bash
   cd ../../src/frontend
   npm install
   npm run build
   az storage blob upload-batch --account-name $(terraform output -raw storage_account_name) --auth-mode key --destination '$web' --source ./build
   ```

## Cleanup

To remove all deployed resources:

```bash
terraform destroy
```

## Kubernetes Resource Files

The `k8s/` and `k8s-azure/` directories contain the Kubernetes manifest files needed to deploy the application:

- `backend-deployment.yaml`: Deployment for the backend API
- `backend-service.yaml`: Service to expose the backend API
- `backend-ingress.yaml`: Ingress for routing to the backend API
- `config.yaml`: ConfigMap for application configuration
- `secrets.yaml`: Sample Secret template (do not commit actual secrets)

## Additional Resources

- [Terraform Documentation](https://www.terraform.io/docs/)
- [AWS Documentation](https://docs.aws.amazon.com/)
- [GCP Documentation](https://cloud.google.com/docs)
- [Azure Documentation](https://docs.microsoft.com/en-us/azure/)

## Troubleshooting

### Common Issues

- **Terraform state lock**: If a previous Terraform operation was interrupted, you might need to unlock the state:
  ```bash
  terraform force-unlock LOCK_ID
  ```

- **Permission issues**: Ensure your cloud provider account has the necessary permissions for resource creation.

- **Resource quotas**: Some cloud resources have limits or quotas. Check your cloud provider's quotas if resource creation fails.

- **Network issues**: Ensure your local network allows connections to the cloud provider API endpoints.

For specific issues, refer to the cloud provider documentation or open an issue in the project repository. 