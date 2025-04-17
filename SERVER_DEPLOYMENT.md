# Server Deployment Guide

This guide provides instructions for deploying the Soccer Prediction System on a server and troubleshooting common issues.

## Prerequisites

- A server running Linux (Ubuntu/Debian recommended)
- Docker and Docker Compose installed
- Ports 8000 (API), 8501 (Streamlit UI), and others needed by your services
- Git to clone the repository

## Setup Instructions

### 1. Install Docker and Docker Compose

```bash
# Update package lists
sudo apt update

# Install required packages
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Update package lists again
sudo apt update

# Install Docker CE
sudo apt install -y docker-ce

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.15.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add your user to the docker group (to avoid using sudo with docker)
sudo usermod -aG docker $USER

# Apply changes to current session
newgrp docker
```

### 2. Clone and Deploy the Application

```bash
# Clone the repository
git clone https://github.com/yourusername/soccer-prediction-system.git
cd soccer-prediction-system

# Build and start the containers
docker compose build
docker compose up -d
```

### 3. Configure Firewall (if applicable)

```bash
# Allow HTTP, Streamlit and API ports
sudo ufw allow 80/tcp
sudo ufw allow 8501/tcp
sudo ufw allow 8000/tcp

# If you've configured HTTPS
sudo ufw allow 443/tcp

# Reload the firewall
sudo ufw reload
```

## Troubleshooting Common Issues

### Docker Networking Issues

If you encounter networking errors like "failed to set up container networking" or "bridge port not forwarding", follow these steps:

1. **Make the network fix script executable and run it:**
   ```bash
   chmod +x scripts/fix_docker_network.sh
   ./scripts/fix_docker_network.sh
   ```

2. **If the script doesn't resolve the issue, try manual steps:**
   ```bash
   # Stop Docker completely
   sudo systemctl stop docker
   
   # Clean up network interfaces
   sudo ip link set dev docker0 down
   sudo ip link delete docker0
   
   # Reset iptables for Docker
   sudo iptables -t nat -F
   
   # Restart Docker
   sudo systemctl start docker
   
   # Prune Docker networks
   docker network prune -f
   
   # Try starting the containers again
   docker compose up -d
   ```

3. **Check if you have a subnet conflict:**
   If your server uses an IP range that conflicts with Docker's default network, modify the docker-compose.yml file to use a different subnet.

4. **System-level network issues:**
   ```bash
   # Check system logs for network-related errors
   sudo dmesg | grep -i net
   
   # Verify network interface status
   ip addr show
   
   # Check if iptables is functioning correctly
   sudo iptables -L
   ```

### Streamlit Connection Issues

If you can't access the Streamlit UI on port 8501, follow these steps:

#### 1. Check Container Status

```bash
# Make the troubleshooting script executable
chmod +x scripts/check_streamlit_connectivity.sh

# Run the troubleshooting script
./scripts/check_streamlit_connectivity.sh
```

#### 2. Verify Streamlit Logs

```bash
# Check UI container logs
docker logs soccer-ui-1
```

#### 3. Common Issues and Solutions

##### Streamlit Not Binding to 0.0.0.0

Ensure Streamlit is configured to listen on all interfaces by checking docker-compose.yml:
- The `--server.address=0.0.0.0` parameter must be passed to Streamlit
- Environment variables should include `STREAMLIT_SERVER_ADDRESS=0.0.0.0`

##### Firewall Blocking Access

Check if your firewall is blocking port 8501:
```bash
sudo ufw status
```
If 8501 is not allowed, add it:
```bash
sudo ufw allow 8501/tcp
```

##### Network Configuration

If you're behind a proxy or accessing from a different network:
1. Try using the server's IP address directly: `http://YOUR_SERVER_IP:8501`
2. If that doesn't work, set up a reverse proxy using Nginx (see below)

#### 4. Setting Up Nginx as a Reverse Proxy (Optional)

If direct access doesn't work, you can set up Nginx to proxy requests to Streamlit:

```bash
# Install Nginx
sudo apt install -y nginx

# Copy the provided Nginx configuration
sudo cp nginx-streamlit.conf /etc/nginx/sites-available/streamlit

# Edit the configuration to match your server name
sudo nano /etc/nginx/sites-available/streamlit

# Create a symbolic link to enable the site
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/

# Test Nginx configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

Now you should be able to access Streamlit through your domain or IP without specifying a port.

## Monitoring and Maintenance

### View Container Status

```bash
docker compose ps
```

### Restart Services

```bash
# Restart a specific service
docker compose restart ui

# Restart all services
docker compose restart
```

### Update the Application

```bash
# Pull the latest changes
git pull

# Rebuild and restart containers
docker compose down
docker compose build
docker compose up -d
```

## Security Considerations

1. For production deployments, consider:
   - Setting up HTTPS with Let's Encrypt
   - Implementing proper authentication
   - Restricting access to administrative interfaces

2. Keep Docker and the server OS updated:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

3. Configure a proper backup strategy for your data volumes.

## Support and Resources

If you encounter any issues not covered by this guide, please:
1. Check the official Docker and Streamlit documentation
2. Search for similar issues in our GitHub repository
3. Contact your system administrator or open a new issue in our repository 