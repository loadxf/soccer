@echo off
echo Soccer Prediction System - Browser Cache Reset Tool
echo ===================================================
echo.
echo This tool will help fix the "Tried to use SessionInfo before it was initialized" error
echo by clearing your browser's cache and storage data.
echo.

REM Check if we need to restart the UI
set restart_ui=0
if "%1"=="-restart" set restart_ui=1
if "%1"=="--restart" set restart_ui=1

REM Find the absolute path to the reset_app.html file
set script_dir=%~dp0
set reset_file=%script_dir%reset_app.html
echo Reset page located at: %reset_file%
echo.

REM Check if reset_app.html exists
if not exist "%reset_file%" (
    echo ERROR: reset_app.html not found!
    echo Make sure you're running this script from the project root directory.
    echo.
    pause
    exit /b 1
)

echo Opening reset tool in your default browser...
start "" "%reset_file%"
echo.

echo If the automatic tool doesn't fix the issue, try these manual steps:
echo.
echo 1. Clear your browser cache:
echo    - Chrome:  Press Ctrl+Shift+Delete
echo    - Firefox: Press Ctrl+Shift+Delete
echo    - Edge:    Press Ctrl+Shift+Delete
echo.
echo 2. Check these specific items to clear:
echo    - Cookies and other site data
echo    - Cached images and files
echo.
echo 3. Use a "hard refresh" by pressing: Ctrl+F5
echo.
echo 4. Try using a private/incognito browser window
echo.

REM If restart flag is set, restart the UI
if %restart_ui%==1 (
    echo Restarting the UI application...
    echo.
    
    REM Kill existing Streamlit processes
    taskkill /f /im streamlit.exe >nul 2>&1
    
    REM Wait a moment
    timeout /t 2 /nobreak >nul
    
    REM Start the application
    start "" python main.py ui --start --no-api
    
    echo UI has been restarted. Please try accessing it again.
)

echo If you're still experiencing issues, try restarting the application:
echo.
echo    %script_dir%clear_browser_cache.bat --restart
echo.

REM Keep the window open
pause 