@echo off
echo Soccer Prediction System - Advanced Session Fix Tool
echo ---------------------------------------------------
echo This tool will help fix persistent "SessionInfo not initialized" errors by
echo aggressively clearing browser storage and resetting connection states.
echo.

REM Get the full path to the HTML file
set HTML_FILE=%~dp0fix_session_errors.html
set HTML_FILE_URL=file:///%HTML_FILE:\=/%

echo Opening advanced session fix tool in your default browser...
echo.

REM Open the browser with the HTML file
start "" "%HTML_FILE_URL%"

echo If your browser doesn't open automatically, please open this file manually:
echo %HTML_FILE%
echo.

echo After running the tool, try accessing the application again at:
echo http://127.0.0.1:8501
echo.

pause 