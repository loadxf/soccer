#!/bin/bash
# Test API connectivity and diagnose common issues

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

# Print header
echo -e "${BLUE}=======================================${RESET}"
echo -e "${BLUE}      API CONNECTIVITY TEST TOOL      ${RESET}"
echo -e "${BLUE}=======================================${RESET}"
echo ""

# Get remote host from arguments or environment variable or use default
if [ $# -gt 0 ]; then
    HOST=$1
else
    # Try to get from .env.remote file first
    if [ -f .env.remote ]; then
        HOST=$(grep REMOTE_API_HOST .env.remote | cut -d '=' -f2)
        if [ "$HOST" = "YOUR_SERVER_IP_OR_HOSTNAME" ]; then
            # Not configured yet
            HOST=""
        fi
    fi
    
    # If not found in .env.remote, try regular .env
    if [ -z "$HOST" ] && [ -f .env ]; then
        HOST=$(grep REMOTE_API_HOST .env | cut -d '=' -f2)
    fi
    
    # If still not found, use environment variable
    if [ -z "$HOST" ]; then
        HOST=$REMOTE_API_HOST
    fi
    
    # Default to localhost if still not found
    if [ -z "$HOST" ]; then
        HOST="localhost"
    fi
fi

# Get port from arguments or use default
if [ $# -gt 1 ]; then
    PORT=$2
else
    # Try to get from .env.remote file first
    if [ -f .env.remote ]; then
        PORT=$(grep API_PORT .env.remote | cut -d '=' -f2)
    fi
    
    # If not found in .env.remote, try regular .env
    if [ -z "$PORT" ] && [ -f .env ]; then
        PORT=$(grep API_PORT .env | cut -d '=' -f2)
    fi
    
    # If still not found, use environment variable
    if [ -z "$PORT" ]; then
        PORT=$API_PORT
    fi
    
    # Default to 8000 if still not found
    if [ -z "$PORT" ]; then
        PORT=8000
    fi
fi

echo -e "${BLUE}Testing API connectivity to:${RESET} $HOST:$PORT"
echo ""

# Function to test a URL
test_url() {
    local url=$1
    local description=$2
    
    echo -e "${YELLOW}Testing $description:${RESET} $url"
    
    # First try curl with timeout
    response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url" 2>/dev/null)
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ Success!${RESET} API responds with HTTP 200"
        echo -e "${BLUE}Fetching API response content:${RESET}"
        curl -s "$url" | grep -v "curl" | grep -v "wget"
        echo ""
        return 0
    elif [ "$response" = "000" ]; then
        echo -e "${RED}✗ Failed!${RESET} Could not connect to the API (connection refused or timeout)"
        return 1
    else
        echo -e "${RED}✗ Failed!${RESET} API responded with HTTP $response"
        return 1
    fi
}

# Test both root health endpoint and API prefixed endpoint
if ! test_url "http://$HOST:$PORT/health" "root health endpoint"; then
    echo -e "${YELLOW}Trying API prefixed health endpoint...${RESET}"
    if ! test_url "http://$HOST:$PORT/api/v1/health" "API prefixed health endpoint"; then
        echo -e "${RED}Both health endpoints failed.${RESET}"
    fi
fi

echo -e "${BLUE}=======================================${RESET}"
echo -e "${BLUE}          NETWORK DIAGNOSTICS         ${RESET}"
echo -e "${BLUE}=======================================${RESET}"
echo ""

# Test basic network connectivity
echo -e "${YELLOW}Testing basic network connectivity to $HOST:${RESET}"
if ping -c 1 $HOST >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Host is reachable via ping${RESET}"
else
    echo -e "${RED}✗ Cannot ping host${RESET} - Check networking and firewall rules"
fi

# Check if port is open
echo -e "${YELLOW}Testing if port $PORT is open on $HOST:${RESET}"
if command -v nc >/dev/null 2>&1; then
    if nc -z -w5 $HOST $PORT >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Port $PORT is open${RESET}"
    else
        echo -e "${RED}✗ Port $PORT is closed${RESET} - Check if API server is running and firewall rules"
    fi
else
    echo -e "${YELLOW}! Netcat not available${RESET} - Cannot check port connectivity"
fi

echo ""
echo -e "${BLUE}=======================================${RESET}"
echo -e "${BLUE}         TROUBLESHOOTING TIPS         ${RESET}"
echo -e "${BLUE}=======================================${RESET}"
echo ""

echo -e "${YELLOW}If the API is not responding, try the following:${RESET}"
echo ""
echo -e "1. ${BLUE}Check if API server is running:${RESET}"
echo "   - Run 'ps aux | grep api_server.py' or 'ps aux | grep simple_api_server.py'"
echo ""
echo -e "2. ${BLUE}Start the API server:${RESET}"
echo "   - Run 'python main.py api --start'"
echo "   - or 'python simple_api_server.py'"
echo ""
echo -e "3. ${BLUE}Check firewall rules:${RESET}"
echo "   - Run 'sudo ufw status' to check Ubuntu firewall"
echo "   - Ensure port $PORT is allowed: 'sudo ufw allow $PORT/tcp'"
echo ""
echo -e "4. ${BLUE}Configure environment variables:${RESET}"
echo "   - Edit .env.remote and set REMOTE_API_HOST to your server's IP/hostname"
echo "   - Run 'export REMOTE_API_HOST=your_server_ip_or_hostname'"
echo ""
echo -e "5. ${BLUE}Check logs for errors:${RESET}"
echo "   - Look at 'app.log' or use 'journalctl' if using systemd"
echo ""
echo -e "${BLUE}=======================================${RESET}"

# Final instructions
echo ""
echo -e "${YELLOW}To use this script:${RESET}"
echo -e "  Basic usage: ${GREEN}./test_api.sh${RESET}"
echo -e "  Specify host: ${GREEN}./test_api.sh your_server_ip${RESET}"
echo -e "  Specify host and port: ${GREEN}./test_api.sh your_server_ip 8000${RESET}" 