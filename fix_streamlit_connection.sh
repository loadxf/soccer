#!/bin/bash

echo "=== Streamlit Connection Troubleshooting Tool ==="

# Function to check if Docker is running
check_docker() {
  echo "Checking if Docker is running..."
  if docker info > /dev/null 2>&1; then
    echo "✅ Docker is running"
    return 0
  else
    echo "❌ Docker is not running"
    return 1
  fi
}

# Function to check if the UI container exists and its status
check_ui_container() {
  echo "Checking UI container status..."
  
  # First see if the container exists
  local container_id=$(docker ps -a --filter "name=soccer-ui" --format "{{.ID}}")
  
  if [ -z "$container_id" ]; then
    echo "❌ UI container doesn't exist"
    return 1
  fi
  
  # Check if it's running
  local status=$(docker inspect --format "{{.State.Status}}" $container_id)
  
  if [ "$status" == "running" ]; then
    echo "✅ UI container is running (ID: $container_id)"
    return 0
  else
    echo "❌ UI container exists but is not running (Status: $status, ID: $container_id)"
    
    # Print the exit code and reason if available
    local exit_code=$(docker inspect --format "{{.State.ExitCode}}" $container_id)
    local error=$(docker inspect --format "{{.State.Error}}" $container_id)
    
    echo "   Exit Code: $exit_code"
    [ ! -z "$error" ] && echo "   Error: $error"
    
    # Get the last few lines of logs
    echo "   Last log entries:"
    docker logs --tail 10 $container_id
    
    return 1
  fi
}

# Function to check network connectivity to the UI container
check_network() {
  echo "Checking network connectivity..."
  
  local container_id=$(docker ps --filter "name=soccer-ui" --format "{{.ID}}")
  
  if [ -z "$container_id" ]; then
    echo "❌ Cannot check network - UI container is not running"
    return 1
  fi
  
  # Get container's IP address
  local ip=$(docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $container_id)
  echo "   Container IP: $ip"
  
  # Check port binding
  local ports=$(docker inspect --format '{{range $p, $conf := .NetworkSettings.Ports}}{{$p}} -> {{(index $conf 0).HostPort}}{{println}}{{end}}' $container_id)
  echo "   Port Mappings:"
  echo "$ports"
  
  # Check if the port is actually listening in the container
  echo "Checking if Streamlit is listening inside the container..."
  if docker exec $container_id netstat -tuln | grep -q ":8501"; then
    echo "✅ Streamlit is listening on port 8501 inside the container"
  else
    echo "❌ Streamlit is NOT listening on port 8501 inside the container"
  fi
  
  # Check host port connectivity
  echo "Checking if port 8501 is accessible on localhost..."
  if nc -z localhost 8501 2>/dev/null; then
    echo "✅ Port 8501 is accessible on localhost"
  else
    echo "❌ Port 8501 is NOT accessible on localhost"
    echo "   This could be due to the container not running, Streamlit not binding to 0.0.0.0, or a firewall issue"
  fi
  
  # Check firewall
  echo "Checking firewall status..."
  if command -v ufw > /dev/null; then
    ufw status | grep 8501
    if ! ufw status | grep -q "8501"; then
      echo "   Port 8501 might not be allowed in the firewall"
    fi
  else
    echo "   UFW not installed, skipping firewall check"
  fi
}

# Function to apply fixes
apply_fixes() {
  echo "Applying potential fixes..."
  
  # 1. Fix docker-compose.yml to ensure proper Streamlit settings
  echo "1. Ensuring proper Streamlit settings in docker-compose.yml..."
  
  # Create backup
  cp docker-compose.yml docker-compose.yml.bak
  
  # Check if this is needed based on the OS
  if [ "$(uname)" == "Darwin" ]; then
    # macOS version
    sed -i '' 's/--server.address=127.0.0.1/--server.address=0.0.0.0/g' docker-compose.yml
  else
    # Linux version
    sed -i 's/--server.address=127.0.0.1/--server.address=0.0.0.0/g' docker-compose.yml
  fi
  
  # 2. Ensure environment variables are set for Streamlit
  echo "2. Ensuring proper environment variables are set..."
  if grep -q "STREAMLIT_SERVER_ADDRESS" docker-compose.yml; then
    echo "   STREAMLIT_SERVER_ADDRESS is already set in docker-compose.yml"
  else
    echo "   Adding STREAMLIT_SERVER_ADDRESS to environment section..."
    # This is more complex and should be done carefully - we'll skip automatic editing
    echo "   Please manually add: '- STREAMLIT_SERVER_ADDRESS=0.0.0.0' to the environment section"
  fi
  
  echo "3. Restarting the UI container..."
  docker compose stop ui
  docker compose up -d ui
  
  echo "4. Checking if the container started successfully..."
  sleep 5
  if docker ps | grep -q "soccer-ui"; then
    echo "✅ UI container started successfully"
    docker logs --tail 20 $(docker ps --filter "name=soccer-ui" --format "{{.ID}}")
  else
    echo "❌ UI container failed to start"
    docker compose logs ui
  fi
}

# Main execution
echo "Starting Streamlit connection diagnostics..."
echo

check_docker
echo

check_ui_container
echo

check_network
echo

echo "Do you want to apply potential fixes? (y/n)"
read -p "> " apply_fix

if [ "$apply_fix" == "y" ] || [ "$apply_fix" == "Y" ]; then
  apply_fixes
else
  echo "No fixes applied. You can run this script again with './fix_streamlit_connection.sh' if needed."
fi

echo
echo "=== Diagnostic Complete ==="
echo
echo "If the issue persists, consider:"
echo "1. Running the UI without Docker: 'streamlit run ui/app.py --server.address=0.0.0.0'"
echo "2. Checking for proxy or network issues between your client and the server"
echo "3. Running the entropy fix script: './fix_ui_entropy.sh'" 