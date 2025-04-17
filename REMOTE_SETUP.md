# Remote Server Setup Guide

This document provides detailed instructions for setting up and connecting to the Soccer Prediction System running on a remote server.

## Remote Server Information

- Remote Server URL: http://103.163.186.204:8501
- API Port: 8000
- UI Port: 8501

## Configuration Files

We've created the following files to simplify remote server connections:

1. `.env.remote` - Contains environment variables for remote connection
2. `connect_remote.py` - Python utility script to load configuration and connect
3. `connect_remote.bat` - Batch file for Windows users to easily run the connection utility

## Connection Methods

### Method 1: Access the Remote UI Directly

The simplest method is to access the remote UI directly through your browser:

1. Open your web browser
2. Navigate to: http://103.163.186.204:8501
3. The UI should connect to the API automatically

### Method 2: Run a Local UI Connected to Remote API

If you need to run the UI locally but connect to the remote API:

1. Ensure you have the latest code from the repository
2. Update `.env.remote` with the correct server IP (currently set to 103.163.186.204)
3. Run the connection utility:
   - Windows: Double-click `connect_remote.bat`
   - Linux/Mac: Run `python connect_remote.py`
4. The local UI will start and connect to the remote API
5. Access the UI in your browser at: http://localhost:8501

## Troubleshooting

### API Connection Issues

If you experience connection issues:

1. Verify the API server is running on the remote machine
   ```bash
   ssh user@103.163.186.204
   netstat -tuln | grep 8000
   ```

2. Check that ports are open in the firewall
   ```bash
   ssh user@103.163.186.204
   sudo ufw status
   ```
   
   If necessary, open the ports:
   ```bash
   sudo ufw allow 8000/tcp
   sudo ufw allow 8501/tcp
   ```

3. Run the diagnostic tool
   ```bash
   python diagnose_api.py
   ```

4. Try direct API connection test
   ```bash
   python debug_api_connection.py
   ```

### Browser Cache Issues

If you encounter "SessionInfo not initialized" errors or other UI issues:

1. Use the "Fix Browser Issues" button in the sidebar of the UI
2. Try accessing with a different browser or in incognito/private mode
3. Run `fix_session_errors.bat` (Windows) for advanced browser cleaning
4. Try using 127.0.0.1 instead of localhost in the URL

## Checking API Status

You can directly check the API status by visiting:
- http://103.163.186.204:8000/health
- http://103.163.186.204:8000/api/v1/health

## Server Maintenance

For server administrators:

1. Restarting the API server:
   ```bash
   ssh user@103.163.186.204
   cd /path/to/soccer-prediction-system
   python main.py api --restart
   ```

2. Updating to the latest code:
   ```bash
   ssh user@103.163.186.204
   cd /path/to/soccer-prediction-system
   git pull
   pip install -r requirements.txt
   python main.py api --restart
   ```

3. Checking logs:
   ```bash
   ssh user@103.163.186.204
   cd /path/to/soccer-prediction-system
   tail -f logs/api.log
   ``` 