@echo off
echo Starting Soccer Prediction System...

:: Add the project root to PYTHONPATH
set PYTHONPATH=%~dp0;%PYTHONPATH%

:: Check if port 8501 is already in use
netstat -aon | findstr :8501 > nul
if %errorlevel% equ 0 (
    echo WARNING: Port 8501 is already in use.
    echo You may need to close the existing Streamlit app or use a different port.
    echo To use a different port, run: streamlit run ui/app.py --server.port=8502 --server.address=0.0.0.0
    echo.
)

:: Start the Streamlit app
echo Starting Streamlit app...
streamlit run ui/app.py --server.address=0.0.0.0

pause 