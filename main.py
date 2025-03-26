"""
Main entry point for the Soccer Prediction System.
This script provides a command-line interface for all available functionality.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import socket

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("main")

# Explainability module will be imported only when needed
# from src.models.explainability import generate_model_explanations 

def handle_data_command(args):
    """Handle data-related commands."""
    if args.load:
        from src.data.data_loader import load_data
        load_data(args.source)
    elif args.clean:
        from src.data.data_cleaner import clean_data
        clean_data()
    elif args.process:
        from src.data.data_processor import process_data
        process_data()
    elif args.validate:
        from src.data.data_validator import validate_data
        validate_data()
    else:
        logger.error("No valid data command specified.")

def handle_model_command(args):
    """Handle model-related commands."""
    if args.train:
        from src.models.model_trainer import train_model
        train_model(args.model_type)
    elif args.evaluate:
        from src.models.model_evaluator import evaluate_model
        evaluate_model(args.model_id)
    elif args.predict:
        from src.models.predictor import predict
        predict(args.match_id, args.model_id)
    elif args.explain:
        # Import explainability module only when needed
        try:
            from src.models.explainability import generate_model_explanations
            generate_model_explanations(args.model_id)
        except ImportError as e:
            logger.error(f"Could not import explainability module: {e}")
            logger.error("Make sure you have the required packages installed.")
    else:
        logger.error("No valid model command specified.")

def is_py311_env_available():
    """Check if Python 3.11 environment is available."""
    return os.path.exists("api_env") and os.path.exists("api_env/Scripts/python.exe")

def run_api_with_py311():
    """Run the API server using Python 3.11 environment."""
    try:
        logger.info("Starting API server using Python 3.11 environment...")
        
        # Use subprocess to run the batch file
        result = subprocess.run(["run_api.bat"], shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running API with Python 3.11: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running API with Python 3.11: {e}")
        return False

def start_simple_api_server():
    """Start a simplified API server directly."""
    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        # Create FastAPI app
        app = FastAPI(
            title="Soccer Prediction System API",
            description="REST API for soccer match predictions (simplified)",
            version="0.1.0",
            docs_url="/api/v1/docs",
            redoc_url="/api/v1/redoc",
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1:3000", "http://127.0.0.1:3000", "http://127.0.0.1:8501"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Health check endpoint
        @app.get("/api/v1/health", tags=["Health"])
        async def health_check():
            """Check API health."""
            return {
                "status": "ok",
                "api": "up",
                "database": "unknown",
                "models": "unknown",
                "version": "0.1.0"
            }
        
        # Root endpoint
        @app.get("/api/v1/", tags=["Root"])
        async def root():
            """Root endpoint."""
            return {
                "message": "Welcome to the Soccer Prediction System API (Simplified)",
                "version": "0.1.0",
                "docs_url": "/api/v1/docs"
            }
        
        # Teams endpoint
        @app.get("/api/v1/teams", tags=["Teams"])
        async def get_teams():
            """Get a list of teams."""
            return [
                {"id": 1, "name": "Team A", "country": "Country A"},
                {"id": 2, "name": "Team B", "country": "Country B"},
                {"id": 3, "name": "Team C", "country": "Country C"},
            ]
        
        # Matches endpoint
        @app.get("/api/v1/matches", tags=["Matches"])
        async def get_matches():
            """Get a list of matches."""
            return [
                {"id": 1, "home_team": "Team A", "away_team": "Team B", "date": "2025-01-01"},
                {"id": 2, "home_team": "Team C", "away_team": "Team A", "date": "2025-01-08"},
            ]
        
        # Prediction models endpoint
        @app.get("/api/v1/predictions/models", tags=["Predictions"])
        async def get_prediction_models():
            """Get available prediction models."""
            return [
                {"id": 1, "name": "Ensemble", "type": "ensemble", "version": "1.0"},
                {"id": 2, "name": "Logistic Regression", "type": "baseline", "version": "1.0"},
            ]
        
        # Start the server
        logger.info("Starting simplified API server at http://0.0.0.0:8000")
        logger.info("API documentation: http://0.0.0.0:8000/api/v1/docs")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return True
    except Exception as e:
        logger.error(f"Error starting simplified API server: {e}")
        return False

def is_api_running(port=8080):
    """Check if the API is already running on the specified port."""
    try:
        # Try to connect to the API
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', port))
            s.sendall(b'GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n')
            response = s.recv(4096)
            # If we got a response that looks like HTTP, the API is probably running
            return b'HTTP/1.1' in response
    except (socket.error, ConnectionRefusedError):
        # If connection fails, the API is not running
        return False

def handle_api_command(args):
    """Handle API-related commands."""
    if args.start:
        # First, check if Python 3.11 environment is available
        if is_py311_env_available():
            logger.info("Found Python 3.11 environment. Using it to start the API server.")
            return run_api_with_py311()
            
        if args.simple:
            # Use the simplified server implementation
            start_simple_api_server()
        else:
            try:
                # Try using the standard server module
                from src.api.server import start_server
                start_server()
            except ImportError as e:
                logger.error(f"Could not import API server module: {e}")
                logger.error("Falling back to simplified API server")
                start_simple_api_server()
    elif args.stop:
        from src.api.server import stop_server
        stop_server()
    elif args.reload:
        from src.api.server import reload_server
        reload_server()
    else:
        logger.error("No valid API command specified.")

def handle_ui_command(args):
    """Handle UI-related commands."""
    if args.start:
        try:
            # Add the project root to Python path
            import sys
            from pathlib import Path
            project_root = Path(__file__).resolve().parent
            sys.path.append(str(project_root))
            
            # Check if we should start the API
            if not args.no_api:
                # First check if API is already running
                if is_api_running():
                    logger.info("API is already running on port 8080. Skipping API startup.")
                else:
                    # Try to start the API server
                    logger.info("UI is starting the API server...")
                    try:
                        if is_py311_env_available():
                            run_api_with_py311()
                        else:
                            try:
                                from src.api.server import start_server
                                start_server()
                            except ImportError:
                                start_simple_api_server()
                    except Exception as api_error:
                        logger.error(f"UI could not start API server: {api_error}")
                        logger.warning("Starting UI without API server. Some features may not work.")
            else:
                logger.info("Skipping API server startup due to --no-api flag")
            
            # Try to import and start the UI app
            from ui.app import start_app
            start_app()
        except ImportError as e:
            if "streamlit" in str(e):
                logger.error("Streamlit is not installed. Installing required packages...")
                try:
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
                    logger.info("Streamlit installed successfully. Retrying UI startup...")
                    from ui.app import start_app
                    start_app()
                except Exception as install_error:
                    logger.error(f"Failed to install Streamlit: {install_error}")
                    logger.error("Please install manually: pip install streamlit")
            else:
                logger.error(f"Could not import UI app module: {e}")
                logger.error("Make sure the 'ui' directory exists and contains app.py")
    else:
        logger.error("No valid UI command specified.")

def main():
    """Entry point function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Soccer Prediction System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Data management commands
    data_parser = subparsers.add_parser("data", help="Data management commands")
    data_parser.add_argument("--load", action="store_true", help="Load data from source")
    data_parser.add_argument("--clean", action="store_true", help="Clean the loaded data")
    data_parser.add_argument("--process", action="store_true", help="Process data for model training")
    data_parser.add_argument("--validate", action="store_true", help="Validate data quality")
    data_parser.add_argument("--source", type=str, help="Data source path or URL")
    
    # Model management commands
    model_parser = subparsers.add_parser("model", help="Model management commands")
    model_parser.add_argument("--train", action="store_true", help="Train a new model")
    model_parser.add_argument("--evaluate", action="store_true", help="Evaluate a model")
    model_parser.add_argument("--predict", action="store_true", help="Make predictions")
    model_parser.add_argument("--explain", action="store_true", help="Generate model explanations")
    model_parser.add_argument("--model-type", type=str, help="Type of model to train")
    model_parser.add_argument("--model-id", type=str, help="Model ID to use")
    model_parser.add_argument("--match-id", type=str, help="Match ID for prediction")
    
    # API server commands
    api_parser = subparsers.add_parser("api", help="API server commands")
    api_parser.add_argument("--start", action="store_true", help="Start the API server")
    api_parser.add_argument("--stop", action="store_true", help="Stop the API server")
    api_parser.add_argument("--reload", action="store_true", help="Reload the API server")
    api_parser.add_argument("--simple", action="store_true", help="Use simplified API server")
    api_parser.add_argument("--setup-py311", action="store_true", help="Set up Python 3.11 environment for API")
    
    # UI commands
    ui_parser = subparsers.add_parser("ui", help="UI commands")
    ui_parser.add_argument("--start", action="store_true", help="Start the UI application")
    ui_parser.add_argument("--no-api", action="store_true", help="Skip starting the API server with the UI")
    
    args = parser.parse_args()
    
    # Execute the appropriate function based on the command
    if args.command == "data":
        handle_data_command(args)
    elif args.command == "model":
        handle_model_command(args)
    elif args.command == "api":
        if args.setup_py311:
            logger.info("Setting up Python 3.11 environment for API server...")
            try:
                subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", "setup_api_env.ps1"], check=True)
            except Exception as e:
                logger.error(f"Error setting up Python 3.11 environment: {e}")
        else:
            handle_api_command(args)
    elif args.command == "ui":
        handle_ui_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 