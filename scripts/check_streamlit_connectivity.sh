#!/bin/bash
# Script to check Streamlit connectivity issues

echo "=== Streamlit Connectivity Check ==="
echo ""

# Check if Docker is running
echo "1. Checking Docker status..."
if docker info > /dev/null 2>&1; then
  echo "✅ Docker is running"
else
  echo "❌ Docker is not running. Please start Docker with: sudo systemctl start docker"
  exit 1
fi

# Check if UI container is running
echo ""
echo "2. Checking UI container status..."
UI_CONTAINER=$(docker ps --filter "name=soccer-ui" -q)
if [ -z "$UI_CONTAINER" ]; then
  echo "❌ UI container is not running"
  echo "   Try restarting with: docker compose up -d ui"
else
  echo "✅ UI container is running (Container ID: $UI_CONTAINER)"
  
  # Get container details
  CONTAINER_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $UI_CONTAINER)
  echo "   Container IP: $CONTAINER_IP"
  
  # Check if port 8501 is open in the container
  echo ""
  echo "3. Checking if Streamlit is listening on port 8501 inside the container..."
  if docker exec $UI_CONTAINER netstat -tulpn 2>/dev/null | grep -q 8501; then
    echo "✅ Streamlit is listening on port 8501 inside the container"
  else
    echo "❌ Streamlit is NOT listening on port 8501 inside the container"
    echo "   Check logs with: docker logs $UI_CONTAINER"
  fi
  
  # Check container logs for common Streamlit issues
  echo ""
  echo "4. Checking container logs for Streamlit errors..."
  if docker logs $UI_CONTAINER 2>&1 | grep -q "Error"; then
    echo "⚠️ Found errors in container logs:"
    docker logs $UI_CONTAINER 2>&1 | grep -i "error" | tail -5
  else
    echo "✅ No obvious errors found in container logs"
  fi
fi

# Check if port 8501 is accessible
echo ""
echo "5. Checking if port 8501 is accessible on the server..."
if command -v nc > /dev/null; then
  if nc -z localhost 8501; then
    echo "✅ Port 8501 is accessible on localhost"
  else
    echo "❌ Port 8501 is NOT accessible on localhost"
  fi
else
  echo "⚠️ 'nc' command not found, using curl instead"
  if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Port 8501 is accessible on localhost"
  else
    echo "❌ Port 8501 is NOT accessible on localhost"
  fi
fi

# Check firewall status
echo ""
echo "6. Checking firewall status..."
if command -v ufw > /dev/null; then
  UFW_STATUS=$(sudo ufw status | grep Status | awk '{print $2}')
  echo "UFW Status: $UFW_STATUS"
  if [ "$UFW_STATUS" = "active" ]; then
    if sudo ufw status | grep -q "8501"; then
      echo "✅ Port 8501 is allowed in UFW"
    else
      echo "⚠️ Port 8501 might be blocked by UFW. Consider adding a rule:"
      echo "   sudo ufw allow 8501/tcp"
    fi
  fi
elif command -v firewall-cmd > /dev/null; then
  if sudo firewall-cmd --list-ports | grep -q "8501"; then
    echo "✅ Port 8501 is allowed in firewalld"
  else
    echo "⚠️ Port 8501 might be blocked by firewalld. Consider adding a rule:"
    echo "   sudo firewall-cmd --permanent --add-port=8501/tcp"
    echo "   sudo firewall-cmd --reload"
  fi
else
  echo "ℹ️ No common firewall detected"
fi

# Get server external IP
echo ""
echo "7. Server Information:"
HOST_IP=$(hostname -I | awk '{print $1}')
echo "   Internal IP: $HOST_IP"
EXTERNAL_IP=$(curl -s https://ipinfo.io/ip)
echo "   External IP: $EXTERNAL_IP"

# Display summary and suggestions
echo ""
echo "=== Summary & Suggestions ==="
echo ""
echo "If you still can't access Streamlit, try these solutions:"
echo ""
echo "1. Access via the server's IP address:"
echo "   http://$HOST_IP:8501"
echo ""
echo "2. Ensure port 8501 is open on your server's firewall:"
echo "   sudo ufw allow 8501/tcp"
echo "   or"
echo "   sudo firewall-cmd --permanent --add-port=8501/tcp"
echo "   sudo firewall-cmd --reload"
echo ""
echo "3. Try restarting the UI container:"
echo "   docker compose restart ui"
echo ""
echo "4. Check detailed container logs:"
echo "   docker logs soccer-ui-1"
echo ""
echo "5. Verify network connectivity to the server from your browser:"
echo "   - Can you ping the server?"
echo "   - Are there any network restrictions between your browser and the server?"
echo ""
echo "6. If all else fails, try exposing Streamlit through Nginx or a reverse proxy."

# Make the script exit with success
exit 0 