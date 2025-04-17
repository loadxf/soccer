@echo off
echo Soccer Prediction System - Remote Connection Utility
echo ================================================
echo This will connect to the remote server at %REMOTE_API_HOST%

rem Run the remote connection utility
python connect_remote.py

rem If there's an error, pause to show the message
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error connecting to remote server. See details above.
    pause
) 