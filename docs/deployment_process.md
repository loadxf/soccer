# Soccer Prediction System - Deployment Process Guide

This document provides a comprehensive, step-by-step guide for deploying the Soccer Prediction System to various environments. It consolidates information from other deployment documentation to provide a complete picture of the deployment process.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Environment Setup](#local-development-environment-setup)
4. [Containerized Deployment with Docker](#containerized-deployment-with-docker)
5. [Cloud Deployment Process](#cloud-deployment-process)
   - [AWS Deployment](#aws-deployment)
   - [Azure Deployment](#azure-deployment)
   - [GCP Deployment](#gcp-deployment)
6. [Database Migration Process](#database-migration-process)
7. [Monitoring and Alerting Setup](#monitoring-and-alerting-setup)
8. [Backup and Restore Procedures](#backup-and-restore-procedures)
9. [Troubleshooting Deployment Issues](#troubleshooting-deployment-issues)
10. [Deployment Verification Checklist](#deployment-verification-checklist)

## Deployment Overview

The Soccer Prediction System consists of several components that need to be deployed together:

1. **Backend API**: FastAPI application for predictions and data serving
2. **Frontend UI**: React-based web application
3. **Database**: PostgreSQL database for storing match data, predictions, and user information
4. **Monitoring**: Prometheus and Grafana for metrics and alerts
5. **Infrastructure**: Networking, security, and scaling components

Each deployment target (local, Docker, or cloud provider) requires specific configurations and processes, but follows a common pattern:

1. Environment setup and prerequisites
2. Infrastructure provisioning
3. Database setup and migration
4. Backend API deployment
5. Frontend UI deployment
6. Monitoring and logging configuration
7. Verification and testing

## Prerequisites

Before deploying to any environment, ensure you have the following prerequisites installed:

### Common Requirements

- **Git**: For source code management
- **Python 3.9+**: For running the backend and deployment scripts
- **Node.js 16+**: For building the frontend
- **PostgreSQL 13+**: For database (local development only)

### Environment-Specific Requirements

For **Local Development**:
- Python virtual environment tools (venv)
- npm or yarn

For **Docker Deployment**:
- Docker Engine (20.10.0+)
- Docker Compose (2.0.0+)

For **Cloud Deployment**:
- Terraform (1.2.0+)
- Cloud provider CLI tools:
  - AWS CLI for AWS
  - Azure CLI for Azure
  - Google Cloud SDK for GCP
- Docker for building container images

## Local Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-organization/soccer-prediction.git
   cd soccer-prediction
   ```

2. **Set up the Python environment**:
   ```bash
   python -m venv .venv
   # On Linux/macOS
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file with your local configuration
   ```

4. **Initialize the database**:
   ```bash
   python scripts/init_db.py
   ```

5. **Start the backend API server**:
   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Install frontend dependencies and start the development server**:
   ```bash
   cd src/frontend
   npm install
   npm start
   ```

7. **Access the application**:
   - Frontend: http://127.0.0.1:3000
   - API: http://127.0.0.1:8000
   - API Documentation: http://127.0.0.1:8000/docs

## Containerized Deployment with Docker

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-organization/soccer-prediction.git
   cd soccer-prediction
   ```

2. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

3. **Build and start containers**:
   ```bash
   docker-compose up -d --build
   ```

4. **Verify deployment**:
   ```bash
   # Check if all containers are running
   docker-compose ps
   
   # Check backend logs
   docker-compose logs -f backend
   ```

5. **Access the application**:
   - Frontend: http://127.0.0.1
   - API: http://127.0.0.1/api
   - API Documentation: http://127.0.0.1/api/docs
   - Prometheus: http://127.0.0.1:9090
   - Grafana: http://127.0.0.1:3000

6. **Stop the containers**:
   ```bash
   docker-compose down
   ```

## Cloud Deployment Process

The Soccer Prediction System can be deployed to AWS, Azure, or GCP using our unified cloud deployment script `cloud_deploy.py`.

### Cloud Deployment Preparation

1. **Ensure prerequisites are installed**:
   ```bash
   # Verify prerequisites
   cd deployment
   python cloud_deploy.py setup
   ```

2. **Configure authentication for your cloud provider**:
   
   **For AWS**:
   ```bash
   aws configure
   # Or
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_REGION="us-east-1"
   ```
   
   **For Azure**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```
   
   **For GCP**:
   ```bash
   gcloud auth login
   gcloud config set project your-project-id
   gcloud auth application-default login
   ```

3. **Customize deployment configuration (optional)**:
   Create a custom configuration file (e.g., `my_deployment_config.json`):
   ```json
   {
     "aws": {
       "region": "us-east-1",
       "environment": "prod",
       "project_name": "soccer-prediction"
     }
   }
   ```

### AWS Deployment

1. **Deploy infrastructure and application**:
   ```bash
   cd deployment
   python cloud_deploy.py deploy --provider aws --region us-east-1 --environment prod
   ```

2. **For a specific deployment component**:
   ```bash
   # Deploy only infrastructure
   python cloud_deploy.py deploy --provider aws --deployment-type infra --region us-east-1
   
   # Deploy only application
   python cloud_deploy.py deploy --provider aws --deployment-type app --region us-east-1
   
   # Deploy only frontend
   python cloud_deploy.py deploy --provider aws --deployment-type frontend --region us-east-1
   ```

3. **Check deployment status**:
   ```bash
   python cloud_deploy.py status --provider aws
   ```

4. **Access deployed resources**:
   ```bash
   # Get resource information
   python cloud_deploy.py output --provider aws
   ```

### Azure Deployment

1. **Deploy infrastructure and application**:
   ```bash
   cd deployment
   python cloud_deploy.py deploy --provider azure --subscription your-subscription-id --region eastus --environment prod
   ```

2. **For a specific deployment component**:
   ```bash
   # Deploy only infrastructure
   python cloud_deploy.py deploy --provider azure --deployment-type infra --subscription your-subscription-id
   
   # Deploy only application
   python cloud_deploy.py deploy --provider azure --deployment-type app --subscription your-subscription-id
   
   # Deploy only frontend
   python cloud_deploy.py deploy --provider azure --deployment-type frontend --subscription your-subscription-id
   ```

3. **Check deployment status**:
   ```bash
   python cloud_deploy.py status --provider azure
   ```

4. **Access deployed resources**:
   ```bash
   # Get resource information
   python cloud_deploy.py output --provider azure
   ```

### GCP Deployment

1. **Deploy infrastructure and application**:
   ```bash
   cd deployment
   python cloud_deploy.py deploy --provider gcp --project-id your-project-id --region us-central1 --zone us-central1-a --environment prod
   ```

2. **For a specific deployment component**:
   ```bash
   # Deploy only infrastructure
   python cloud_deploy.py deploy --provider gcp --deployment-type infra --project-id your-project-id
   
   # Deploy only application
   python cloud_deploy.py deploy --provider gcp --deployment-type app --project-id your-project-id
   
   # Deploy only frontend
   python cloud_deploy.py deploy --provider gcp --deployment-type frontend --project-id your-project-id
   ```

3. **Check deployment status**:
   ```bash
   python cloud_deploy.py status --provider gcp
   ```

4. **Access deployed resources**:
   ```bash
   # Get resource information
   python cloud_deploy.py output --provider gcp
   ```

## Database Migration Process

Database migrations ensure that your database schema stays in sync with your application code during deployment.

### Automatic Migration During Deployment

By default, migrations run automatically during deployment. This behavior can be controlled with:

```bash
python cloud_deploy.py deploy --provider aws --skip-migrations
```

### Manual Migration Process

For manual migration execution:

1. **Locally**:
   ```bash
   python scripts/run_migrations.py
   ```

2. **On a deployed instance**:
   ```bash
   # For AWS
   aws ecs run-task --cluster soccer-prediction --task-definition soccer-prediction-migrations --launch-type FARGATE
   
   # For Azure
   az container create --resource-group soccer-prediction-rg --name migration-job --image <acr-name>.azurecr.io/soccer-prediction:latest --cpu 1 --memory 1 --command-line "python scripts/run_migrations.py"
   
   # For GCP
   kubectl create job --from=cronjob/migration-job migration-manual
   ```

### Rolling Back Migrations

To roll back migrations:

```bash
# Locally
python scripts/run_migrations.py --rollback

# For cloud deployments, follow similar patterns as for running migrations
```

## Monitoring and Alerting Setup

The Soccer Prediction System uses Prometheus and Grafana for monitoring and alerting.

### Accessing Monitoring Dashboards

**Local Docker Deployment**:
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000

**Cloud Deployments**:
- Use the URLs from the deployment output:
  ```bash
  python cloud_deploy.py output --provider <aws|azure|gcp>
  ```

### Configuring Custom Alerts

1. **Log into Grafana** with the default credentials:
   - Username: admin
   - Password: admin (you'll be prompted to change this)

2. **Navigate to Alerting** in the sidebar

3. **Create a new alert rule**:
   - Set condition (e.g., API response time > 500ms)
   - Set evaluation interval
   - Add notification channel (e.g., email, Slack)

4. **Save the alert rule**

## Backup and Restore Procedures

Regular backups are essential for data safety and disaster recovery.

### Automated Backups

Automated backups are configured during deployment and include:

- Database backups (daily)
- Application state backups (weekly)
- Infrastructure configuration backups (on change)

### Manual Backup Procedure

1. **Database Backup**:
   ```bash
   # Local PostgreSQL
   pg_dump -U <username> -d soccer_prediction > backup.sql
   
   # AWS RDS
   aws rds create-db-snapshot --db-instance-identifier <instance-id> --db-snapshot-identifier <snapshot-name>
   
   # Azure Database for PostgreSQL
   az postgres server backup create --resource-group <resource-group> --server-name <server-name> --backup-name <backup-name>
   
   # GCP Cloud SQL
   gcloud sql backups create --instance <instance-name>
   ```

2. **Application State Backup**:
   ```bash
   # Local
   tar -czvf app_state_backup.tar.gz data/
   
   # AWS S3
   aws s3 sync data/ s3://soccer-prediction-backups/app-state/$(date +%Y-%m-%d)/
   
   # Azure Blob Storage
   az storage blob upload-batch --account-name <storage-account> --destination app-state/$(date +%Y-%m-%d) --source data/
   
   # GCP Cloud Storage
   gsutil -m cp -r data/ gs://soccer-prediction-backups/app-state/$(date +%Y-%m-%d)/
   ```

### Restore Procedure

1. **Database Restore**:
   ```bash
   # Local PostgreSQL
   psql -U <username> -d soccer_prediction < backup.sql
   
   # AWS RDS
   aws rds restore-db-instance-from-db-snapshot --db-instance-identifier <new-instance-id> --db-snapshot-identifier <snapshot-name>
   
   # Azure Database for PostgreSQL
   az postgres server restore --resource-group <resource-group> --name <new-server-name> --source-server <server-name> --restore-point-in-time <time>
   
   # GCP Cloud SQL
   gcloud sql instances restore <instance-name> --backup <backup-id>
   ```

2. **Application State Restore**:
   ```bash
   # Local
   tar -xzvf app_state_backup.tar.gz -C ./
   
   # AWS S3
   aws s3 sync s3://soccer-prediction-backups/app-state/<date>/ data/
   
   # Azure Blob Storage
   az storage blob download-batch --account-name <storage-account> --source app-state/<date> --destination data/
   
   # GCP Cloud Storage
   gsutil -m cp -r gs://soccer-prediction-backups/app-state/<date>/ data/
   ```

## Troubleshooting Deployment Issues

### Common Issues and Solutions

#### Infrastructure Provisioning Failures

**Issue**: Terraform fails with permission errors.
**Solution**: Check that your cloud credentials have the necessary permissions. Verify IAM roles/policies.

**Issue**: Resource quota limits reached.
**Solution**: Request quota increases from your cloud provider or adjust resource allocations.

#### Container Image Failures

**Issue**: Docker build fails.
**Solution**: Check Dockerfile for errors, ensure dependencies are available, and verify build context.

**Issue**: Container exits immediately after startup.
**Solution**: Check container logs (`docker logs <container-id>` or cloud provider equivalent).

#### Database Connection Issues

**Issue**: Application cannot connect to database.
**Solution**: Verify database credentials, check network/firewall rules, ensure database is running.

#### Frontend Deployment Issues

**Issue**: Frontend not loading or showing errors.
**Solution**: Check browser console for errors, verify API endpoint configuration, ensure build process completed.

### Getting Help

For more assistance:

1. Check detailed logs:
   ```bash
   python cloud_deploy.py deploy --provider <provider> --verbose
   ```

2. Review specific component logs:
   ```bash
   # Docker logs
   docker-compose logs -f <service-name>
   
   # AWS CloudWatch
   aws logs get-log-events --log-group-name <log-group> --log-stream-name <log-stream>
   
   # Azure Log Analytics
   az monitor log-analytics query --workspace <workspace-id> --query <query>
   
   # GCP Cloud Logging
   gcloud logging read "resource.type=<resource-type>"
   ```

## Deployment Verification Checklist

Before considering the deployment complete, verify the following:

### Backend API Verification

- [ ] API health endpoint (`/api/health`) returns 200 OK
- [ ] Authentication endpoints function correctly
- [ ] CRUD operations work as expected
- [ ] Prediction endpoints return valid results

### Frontend Verification

- [ ] Login/registration works
- [ ] Dashboard loads with correct data
- [ ] Match prediction interface functions correctly
- [ ] Visualizations render properly
- [ ] Mobile responsiveness works as expected

### Database Verification

- [ ] Migrations applied successfully
- [ ] Data integrity maintained
- [ ] Performance metrics within expected ranges

### Monitoring Verification

- [ ] Prometheus targets are all up
- [ ] Grafana dashboards show data
- [ ] Alert configurations are active

### Security Verification

- [ ] SSL/TLS certificates are valid
- [ ] Authentication is required for protected endpoints
- [ ] Rate limiting is working
- [ ] No sensitive information is exposed

After completing this checklist, the deployment can be considered successful and ready for use. 