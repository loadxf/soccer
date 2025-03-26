# setup_api_env.ps1
# Script to set up a Python 3.11 environment for the Soccer Prediction API

$ErrorActionPreference = "Stop"
$pythonVersion = "3.11.8"
$pythonUrl = "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-amd64.exe"
$apiEnvPath = ".\api_env"

Write-Host "Soccer Prediction System - API Environment Setup" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python 3.11 is available through py launcher
$python311Available = $false
try {
    $pythonVersionOutput = & py -3.11 --version 2>&1
    if ($pythonVersionOutput -match "3\.11") {
        $python311Available = $true
        Write-Host "Python 3.11 found through py launcher" -ForegroundColor Green
    }
} catch {}

if (-not $python311Available) {
    Write-Host "Python 3.11 not found. You need to install it manually." -ForegroundColor Yellow
    Write-Host "Please download and install Python 3.11 from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installing Python 3.11, run this script again." -ForegroundColor Yellow
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path $apiEnvPath)) {
    Write-Host "Creating new virtual environment at $apiEnvPath..." -ForegroundColor Cyan
    & py -3.11 -m venv $apiEnvPath
} else {
    Write-Host "Virtual environment already exists at $apiEnvPath" -ForegroundColor Green
}

# Activate the environment and install dependencies
Write-Host "Installing required packages in the virtual environment..." -ForegroundColor Cyan
$activateScript = Join-Path $apiEnvPath "Scripts\Activate.ps1"
& $activateScript
& python -m pip install --upgrade pip
& pip install "fastapi==0.95.2" "uvicorn==0.22.0" "pydantic==1.10.8"

Write-Host ""
Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the API server, run:" -ForegroundColor Cyan
Write-Host ".\api_env\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "python api_server.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "Or use the provided run_api.bat script:" -ForegroundColor Cyan
Write-Host ".\run_api.bat" -ForegroundColor Yellow 