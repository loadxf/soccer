# Soccer Prediction System API Setup

This guide explains how to set up and run the Soccer Prediction System API using Python 3.11, which resolves compatibility issues with newer Python versions (3.12+).

## Background

The Soccer Prediction System API relies on FastAPI and pydantic, which have compatibility issues with Python 3.12. To resolve this, we've created a separate Python 3.11 environment that can run the API without issues.

## Setup Instructions

### Automatic Setup

1. Make sure PowerShell is installed on your system
2. Run the setup script:
   ```
   python main.py api --setup-py311
   ```
   This will:
   - Check if Python 3.11 is installed
   - Create a virtual environment called `api_env`
   - Install required packages in the environment

### Manual Setup

If you prefer to set up manually:

1. Install Python 3.11 from [python.org](https://www.python.org/downloads/)
2. Create a virtual environment:
   ```
   python3.11 -m venv api_env
   ```
3. Activate the environment:
   ```
   .\api_env\Scripts\activate
   ```
4. Install required packages:
   ```
   pip install fastapi==0.95.2 uvicorn==0.22.0 pydantic==1.10.8
   ```

## Running the API

### Using the Main Command

After setup is complete, you can start the API with:
```
python main.py api --start
```

The system will automatically detect and use the Python 3.11 environment if available.

### Using the Batch File

Alternatively, you can start the API directly with:
```
.\run_api.bat
```

## API Endpoints

Once running, the API provides these endpoints:

- API Health Check: http://127.0.0.1:8080/api/v1/health
- API Documentation: http://127.0.0.1:8080/api/v1/docs
- Teams List: http://127.0.0.1:8080/api/v1/teams
- Matches List: http://127.0.0.1:8080/api/v1/matches
- Prediction Models: http://127.0.0.1:8080/api/v1/predictions/models

## Troubleshooting

If you encounter issues:

1. Ensure Python 3.11 is installed and in your PATH
2. Check that the `api_env` directory exists
3. Try running the setup script again
4. If all else fails, you can run the API directly with:
   ```
   .\api_env\Scripts\python api_server.py
   ``` 