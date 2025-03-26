# Soccer Prediction System - Deployment Guide

This document provides comprehensive instructions for deploying the Soccer Prediction System in various environments. The system consists of a Python backend API built with FastAPI, a React frontend, and requires a database for storing predictions and model data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Deployment](#local-development-deployment)
- [Production Deployment Options](#production-deployment-options)
  - [Docker Deployment](#docker-deployment)
  - [Cloud Deployment](#cloud-deployment)
    - [AWS Deployment](#aws-deployment)
    - [GCP Deployment](#gcp-deployment)
    - [Azure Deployment](#azure-deployment)
  - [On-Premises Deployment](#on-premises-deployment)
- [Database Setup](#database-setup)
- [Environment Configuration](#environment-configuration)
- [SSL/TLS Configuration](#ssltls-configuration)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Backup and Restore Procedures](#backup-and-restore-procedures)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying the Soccer Prediction System, ensure you have the following prerequisites:

- Python 3.9 or higher
- Node.js 16.x or higher
- PostgreSQL 13 or higher
- Docker and Docker Compose (for containerized deployment)
- Git

## Local Development Deployment

For local development, follow these steps:

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-organization/soccer-prediction.git
   cd soccer-prediction
   ```

2. **Set up the Python environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the environment variables**

   Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file with your local configuration.

4. **Initialize the database**

   ```bash
   python scripts/init_db.py
   ```

5. **Start the backend API server**

   ```bash
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Install frontend dependencies and start the frontend development server**

   ```bash
   cd src/frontend
   npm install
   npm start
   ```

   The frontend will be available at http://127.0.0.1:3000.

## Production Deployment Options

### Docker Deployment

We provide a Docker Compose setup for easy deployment.

1. **Build and start the containers**

   ```bash
   docker-compose up -d --build
   ```

   This will start:
   - PostgreSQL database
   - Backend API server
   - Frontend serving with Nginx
   - Monitoring services (Prometheus and Grafana)

2. **Access the application**

   The application will be available at http://127.0.0.1 (port 80).
   The API documentation will be available at http://127.0.0.1/api/docs.

3. **Stop the containers**

   ```bash
   docker-compose down
   ```

### Cloud Deployment

#### AWS Deployment

1. **Set up Infrastructure with Terraform**

   We provide Terraform scripts in the `deployment/aws` directory.

   ```bash
   cd deployment/aws
   terraform init
   terraform apply
   ```

   This sets up:
   - VPC with public and private subnets
   - ECS Cluster for container orchestration
   - RDS PostgreSQL instance
   - Load Balancer and security groups
   - S3 bucket for frontend assets
   - CloudFront for content delivery

2. **Deploy the Application**

   After the infrastructure is set up, deploy the application using AWS CLI:

   ```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <aws-account-id>.dkr.ecr.<region>.amazonaws.com
   docker-compose -f docker-compose.aws.yml build
   docker-compose -f docker-compose.aws.yml push
   aws ecs update-service --cluster soccer-prediction-cluster --service soccer-prediction-service --force-new-deployment
   ```

3. **Frontend Deployment**

   ```bash
   cd src/frontend
   npm run build
   aws s3 sync build/ s3://soccer-prediction-frontend
   aws cloudfront create-invalidation --distribution-id <distribution-id> --paths "/*"
   ```

#### GCP Deployment

1. **Set up GCP Project**

   ```bash
   gcloud config set project <your-project-id>
   gcloud services enable compute.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com sqladmin.googleapis.com
   ```

2. **Deploy to Google Kubernetes Engine (GKE)**

   ```bash
   cd deployment/gcp
   gcloud container clusters create soccer-prediction-cluster --num-nodes=3 --zone=us-central1-a
   gcloud container clusters get-credentials soccer-prediction-cluster --zone=us-central1-a
   
   # Create Cloud SQL instance
   gcloud sql instances create soccer-prediction-db --tier=db-g1-small --region=us-central1
   
   # Deploy the application using Kubernetes manifests
   kubectl apply -f k8s/
   ```

3. **Set up HTTPS with Cloud Load Balancer**

   ```bash
   kubectl apply -f k8s/ingress.yaml
   ```

#### Azure Deployment

1. **Set up Azure Resources**

   ```bash
   cd deployment/azure
   az group create --name soccer-prediction-rg --location eastus
   az acr create --resource-group soccer-prediction-rg --name soccerpredictionacr --sku Basic
   az aks create --resource-group soccer-prediction-rg --name soccer-prediction-aks --node-count 3 --enable-addons monitoring --generate-ssh-keys
   az aks get-credentials --resource-group soccer-prediction-rg --name soccer-prediction-aks
   ```

2. **Deploy the Application**

   ```bash
   # Build and push Docker images
   az acr login --name soccerpredictionacr
   docker-compose -f docker-compose.azure.yml build
   docker-compose -f docker-compose.azure.yml push
   
   # Deploy to AKS
   kubectl apply -f k8s-azure/
   ```

### On-Premises Deployment

For on-premises deployment, you can use:

1. **Docker Compose (recommended for smaller deployments)**

   Follow the Docker Deployment instructions above.

2. **Kubernetes (for larger scale deployments)**

   ```bash
   # Assuming you have a Kubernetes cluster
   kubectl apply -f deployment/k8s/
   ```

3. **Traditional Deployment**

   For deploying directly on VMs or bare metal:

   - Deploy PostgreSQL on a dedicated server
   - Deploy the backend API using Gunicorn and Nginx:
     ```bash
     gunicorn src.api.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
     ```
   - Deploy the frontend as static files with Nginx:
     ```bash
     cd src/frontend
     npm run build
     # Copy the build directory to your web server's root
     ```

## Database Setup

### PostgreSQL Setup

1. **Create the database and user**

   ```sql
   CREATE DATABASE soccer_prediction;
   CREATE USER soccer_user WITH ENCRYPTED PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE soccer_prediction TO soccer_user;
   ```

2. **Run migrations**

   ```bash
   python scripts/db_migrations.py
   ```

3. **Indexes and Optimization**

   We recommend creating the following indexes for performance:

   ```sql
   CREATE INDEX idx_match_date ON matches (match_date);
   CREATE INDEX idx_team_name ON teams (name);
   CREATE INDEX idx_prediction_user ON predictions (user_id);
   ```

## Environment Configuration

The following environment variables should be configured:

```
# Database Configuration
DATABASE_URL=postgresql://user:password@host:port/dbname

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_SECRET_KEY=your_secure_secret_key

# JWT Authentication
JWT_SECRET_KEY=your_jwt_secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440

# Model Configuration
MODEL_VERSION=v1.2.3
MODEL_DATA_PATH=/path/to/model/data

# Frontend Configuration
REACT_APP_API_URL=https://api.example.com
```

## SSL/TLS Configuration

For production deployments, always use HTTPS:

1. **Generate SSL certificate** (or use Let's Encrypt)

   ```bash
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
   ```

2. **Configure Nginx**

   ```nginx
   server {
       listen 443 ssl;
       server_name yourdomain.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       # Other SSL settings
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_prefer_server_ciphers on;
       ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
       
       # Frontend
       location / {
           root /usr/share/nginx/html;
           try_files $uri $uri/ /index.html;
       }
       
       # API
       location /api/ {
           proxy_pass http://backend:8000/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   
   # Redirect HTTP to HTTPS
   server {
       listen 80;
       server_name yourdomain.com;
       return 301 https://$host$request_uri;
   }
   ```

## Monitoring and Alerting

The system includes:

1. **Prometheus** for metrics collection
2. **Grafana** for visualization
3. **Alert Manager** for alerting

To set up monitoring:

1. **Access Grafana** at http://your-domain/grafana (default credentials: admin/admin)

2. **Import dashboards** from the `monitoring/dashboards` directory

3. **Configure alerting** in Grafana or directly in Prometheus Alert Manager

   ```yaml
   # alertmanager.yml example
   receivers:
     - name: 'team-email'
       email_configs:
         - to: 'team@example.com'
           from: 'alertmanager@example.com'
           smarthost: 'smtp.example.com:587'
           auth_username: 'alertmanager'
           auth_password: 'password'
   
   route:
     group_by: ['alertname', 'cluster', 'service']
     group_wait: 30s
     group_interval: 5m
     repeat_interval: 3h
     receiver: 'team-email'
   ```

## Backup and Restore Procedures

### Database Backup

1. **Automated daily backups**

   ```bash
   pg_dump -U username -d soccer_prediction | gzip > /backups/soccer_prediction_$(date +%Y-%m-%d).sql.gz
   ```

   Add this to a cron job:

   ```
   0 2 * * * /path/to/backup_script.sh
   ```

2. **Restore from backup**

   ```bash
   gunzip -c /backups/soccer_prediction_2023-05-15.sql.gz | psql -U username -d soccer_prediction
   ```

### Model Backup

1. **Backup trained models**

   ```bash
   rsync -av /models/ /backups/models_$(date +%Y-%m-%d)/
   ```

## CI/CD Integration

The project includes GitHub Actions workflows for CI/CD:

1. **Continuous Integration**: Runs tests on every pull request
2. **Continuous Deployment**: Deploys to staging or production environments

To set up GitHub Actions, the following secrets need to be configured:

- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for AWS deployments
- `GCP_PROJECT_ID` and `GCP_SA_KEY` for GCP deployments
- `AZURE_CREDENTIALS` for Azure deployments
- `DOCKER_USERNAME` and `DOCKER_PASSWORD` for Docker Hub

## Troubleshooting

### Common Issues

1. **Database Connection Issues**

   Check:
   - Database credentials in environment variables
   - Network connectivity and firewall rules
   - PostgreSQL logs for any authentication errors

2. **API Startup Issues**

   Check:
   - Python environment and required packages
   - Environment variables configuration
   - API logs for detailed error messages

3. **Frontend Loading Issues**

   Check:
   - Browser console for JavaScript errors
   - CORS configuration if API and frontend are on different domains
   - Network tab for failed API requests

### Logs

Key log locations:

- API logs: `/var/log/soccer-prediction/api.log`
- Database logs: `/var/log/postgresql/postgresql.log`
- Nginx logs: `/var/log/nginx/access.log` and `/var/log/nginx/error.log`

For Docker deployments, use:

```bash
docker-compose logs -f
```

For Kubernetes deployments, use:

```bash
kubectl logs -f deployment/soccer-prediction-api
```

---

For additional support, please file an issue on the GitHub repository or contact the development team at team@example.com. 