@echo off
echo Starting Soccer Prediction API Server with Python 3.11...

REM Set environment variables
set API_PORT=8000
set HOST=0.0.0.0

REM Activate the Python 3.11 environment if it exists
if exist api_env\Scripts\activate.bat (
    call api_env\Scripts\activate.bat
) else (
    echo Warning: Python 3.11 environment not found, using system Python
)

REM Run the simple API server
python simple_api_server.py

REM If simple_api_server.py doesn't exist, try the main.py approach
if %ERRORLEVEL% NEQ 0 (
    echo Simple API server failed, trying main.py approach...
    python main.py api --start
)

REM Keep the window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error starting API server. Press any key to exit...
    pause > nul
) 