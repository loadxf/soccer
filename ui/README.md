# Soccer Prediction System UI

This is the user interface for the Soccer Prediction System, built with Streamlit.

## Features

- Interactive data management
- Model training configuration
- Match predictions
- Model evaluation and comparison
- Prediction explanations

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure the backend API is running:
   ```bash
   python main.py api --start
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run ui/app.py
   ```

## Directory Structure

- `ui/app.py` - Main Streamlit application
- `ui/api_service.py` - Service for interacting with the backend API
- `ui/assets/` - Images and other static assets

## Usage

The UI provides a simple interface for all major features of the Soccer Prediction System:

### Data Management
- Import data from various sources
- Process and clean data
- Generate features

### Model Training
- Configure and train various ML models
- Set hyperparameters
- Monitor training progress

### Predictions
- Make predictions for custom matches
- Predict upcoming fixtures
- Batch prediction

### Model Evaluation
- View model performance metrics
- Compare different models
- Analyze feature importance

### Explanations
- Understand model predictions with SHAP
- Generate LIME explanations
- View Partial Dependence Plots

## Development

To add new features to the UI:

1. Create new page components in the appropriate sections
2. Update the API service if needed
3. Run the app and test your changes

## Configuration

The UI connects to the backend API using the settings in `api_service.py`. By default, it connects to `http://127.0.0.1:8000/api/v1`. 