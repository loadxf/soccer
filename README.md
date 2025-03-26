# Soccer Prediction System

A machine learning platform for predicting soccer match outcomes with a user-friendly interface and advanced statistical models.

## Features

- **Data Management**: Import, process, and explore soccer datasets from various sources
- **Advanced Feature Engineering**: Create specialized soccer-specific features based on academic research
- **Model Training**: Configure and train prediction models using various algorithms and statistical approaches
- **Soccer-Specific Models**: Implements academic research-backed statistical models like Dixon-Coles
- **Ensemble Methods**: Combine multiple models for improved prediction accuracy
- **Predictions**: Generate predictions for upcoming matches with confidence scores
- **Model Evaluation**: Compare and analyze model performance with comprehensive metrics
- **Explainability**: Understand the factors influencing predictions

## Installation

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/soccer-prediction-system.git
cd soccer-prediction-system

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional packages for advanced features
pip install -r optional-requirements.txt
```

### Docker Installation

The system provides a complete Docker setup for easy deployment and scaling:

1. **Prerequisites**:
   - [Docker](https://docs.docker.com/get-docker/)
   - [Docker Compose](https://docs.docker.com/compose/install/)

2. **Build and Run**:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/soccer-prediction-system.git
   cd soccer-prediction-system
   
   # Create environment file
   cp .env.example .env
   
   # Build and start the containers
   docker-compose up -d
   ```

3. **Access the application**:
   - UI: http://localhost:8501
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Monitoring Dashboard: http://localhost:3000 (Grafana, admin/admin)
   - Database Admin: http://localhost:5050 (pgAdmin, admin@example.com/admin)

## Setup

### Environment Configuration

The system uses environment variables for configuration. Copy the example file and modify as needed:

```bash
cp .env.example .env
```

### Network Accessibility

The application is configured to listen on all network interfaces (0.0.0.0) by default, making it accessible from other machines on your network:

- When running locally: Access via `http://your-machine-ip:8501`
- When running in Docker: Access via `http://your-machine-ip:8501`
- For development: Access via `http://localhost:8501` or `http://127.0.0.1:8501`

To restrict access to localhost only, modify the startup commands to use `--server.address=127.0.0.1` instead.

### Kaggle Integration (Optional)

To use Kaggle datasets:

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
2. Go to your Kaggle account settings (click on your profile picture → Account)
3. Scroll down to the API section and click "Create New API Token"
4. This will download a `kaggle.json` file with your credentials
5. Create a `.kaggle` directory in your home folder and move the file there:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
6. Ensure proper permissions (for Linux/Mac): `chmod 600 ~/.kaggle/kaggle.json`

## Usage

### Starting the Application

#### Local Deployment

```bash
# Easy start with batch file (Windows)
start_system.bat

# Or use Python script (cross-platform)
python start_system.py

# Or start individual components
python main.py api --start   # Start the API server
streamlit run ui/app.py      # Start the UI server
```

Then open your browser to http://localhost:8501 to access the UI.

#### Docker Deployment

```bash
# Start all services
docker-compose up -d

# Start only specific services
docker-compose up -d app ui db

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

#### Docker Services

The Docker setup provides the following services:

| Service | Description | Port |
|---------|-------------|------|
| app | Main API server | 8000 |
| ui | Streamlit user interface | 8501 |
| db | PostgreSQL database | 5432 |
| pgadmin | Database administration | 5050 |
| prometheus | Metrics collection | 9090 |
| alertmanager | Alert management | 9093 |
| grafana | Monitoring dashboards | 3000 |
| node-exporter | Host metrics exporter | 9100 |
| cadvisor | Container metrics | 8080 |
| postgres-exporter | Database metrics | 9187 |

### Working with Large Files

For files larger than Streamlit's 1000MB upload limit, we provide a command-line utility:

```bash
# Upload a large file to the system
python scripts/upload_large_file.py path/to/your/large_dataset.csv

# Optionally specify a destination directory
python scripts/upload_large_file.py path/to/your/large_dataset.csv data/custom_dir

# When using Docker
docker-compose exec app python scripts/upload_large_file.py /path/to/file.csv
```

### Docker Data Persistence

Data is persisted across container restarts using Docker volumes:

- `postgres_data`: Database files
- `prometheus_data`: Monitoring metrics
- `grafana_data`: Dashboard configurations

### Docker Monitoring and Scaling

For production deployments, the system includes:

1. **Health Checks**: Automatic container restart on failure
2. **Metrics Collection**: Prometheus for system and application metrics
3. **Visualization**: Grafana dashboards for monitoring
4. **Alerting**: AlertManager for notification on critical events

To scale the API server:
```bash
docker-compose up -d --scale app=3
```

## Model Training Strategies

The system offers several model training approaches, each with specific strengths:

### Traditional Machine Learning Models

- **Logistic Regression**: Fast baseline model for classification problems
- **Random Forest**: Ensemble of decision trees, good for handling non-linear relationships
- **XGBoost**: Gradient boosting implementation, often achieves state-of-the-art performance

These models use standard feature sets including:
- `match_features`: Basic match statistics (goals, shots, etc.)
- `team_features`: Team-level aggregated statistics
- `advanced_features`: Specialized soccer metrics

### Soccer-Specific Statistical Models

- **Dixon-Coles Model**: Modified Poisson distribution model specifically designed for soccer score prediction
  - Incorporates team attack/defense parameters
  - Accounts for home advantage
  - Includes low-score correction factor
  - Time-weighted match importance based on recency
  
This model is trained directly on match results rather than derived features and has been proven effective in academic research.

### Advanced Soccer Features

The system includes specialized feature engineering for soccer:

- **Time-weighted Form**: Exponential weighting of team performance based on recency
- **Expected Goals (xG)**: Statistical measure of shot quality and goal probability
- **Bayesian Strength Indicators**: Advanced team strength estimation
- **Team Style Metrics**: Indicators of team playing style and tactical approach
- **Market-derived Features**: Signals from betting markets

### Ensemble Methods

Combine multiple models to improve prediction accuracy:

- **Voting**: Equal or weighted voting from multiple models
- **Stacking**: Uses a meta-model to combine predictions
- **Performance-weighted**: Dynamically adjusts model weights based on historical performance

## Workflow

1. **Data Import**: Upload datasets or import from Kaggle/Football API
2. **Data Processing**: Clean and transform raw data into appropriate formats
3. **Feature Engineering**: Generate standard or advanced features
4. **Model Training**: Select and train models with optional hyperparameter tuning
5. **Model Evaluation**: Assess model performance on test data
6. **Prediction**: Generate predictions for upcoming matches
7. **Ensemble Creation**: Combine models for improved accuracy

## Project Structure

```
soccer-prediction-system/
├── data/                      # Data storage
│   ├── raw/                   # Original, immutable data
│   ├── processed/             # Cleaned and transformed data
│   ├── features/              # Generated features
│   ├── models/                # Trained model storage
│   │   └── distributions/     # Soccer distribution models
│   ├── uploads/               # User uploaded files
│   └── kaggle_imports/        # Datasets imported from Kaggle
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── pipeline.py        # Data processing pipeline
│   │   ├── soccer_features.py # Advanced soccer feature engineering
│   │   ├── features.py        # Standard feature engineering
│   │   └── kaggle_manager.py  # Kaggle integration
│   ├── models/                # Machine learning models
│   │   ├── training.py        # Model training routines
│   │   ├── baseline.py        # Standard ML models
│   │   ├── soccer_distributions.py # Soccer-specific distribution models
│   │   ├── ensemble.py        # Ensemble model framework
│   │   ├── evaluation.py      # Model evaluation metrics
│   │   ├── hyperopt.py        # Hyperparameter optimization
│   │   └── advanced.py        # Advanced ML models
│   ├── utils/                 # Utility functions
│   └── api/                   # API server implementation
├── ui/                        # User interface
│   ├── app.py                 # Main Streamlit application
│   ├── data_manager.py        # Dataset management UI
│   └── kaggle_ui.py           # Kaggle integration UI
├── scripts/                   # Utility scripts
├── tests/                     # Test suite
├── docker-compose.yml         # Docker services configuration
├── Dockerfile                 # Main API Docker configuration
├── ui/Dockerfile              # UI Docker configuration
├── monitoring/                # Monitoring configurations
├── start_system.py            # System startup script
├── start_system.bat           # Windows startup script
├── main.py                    # Command-line interface
└── requirements.txt           # Project dependencies
```

## Troubleshooting

### Browser Issues

If you encounter UI issues:
- Run `fix_session_errors.bat` for advanced browser cleaning
- Try accessing with `127.0.0.1:8501` instead of `localhost:8501`
- Use incognito/private browsing mode

### Port Conflicts

If port 8501 is already in use:
- Close any running Streamlit applications
- Specify a different port: `streamlit run ui/app.py --server.port=8502`

### Docker Issues

- **Container won't start**: Check logs with `docker-compose logs <service_name>`
- **Database connection error**: Ensure the database container is running with `docker-compose ps`
- **UI can't connect to API**: Check the network settings in docker-compose.yml
- **Reset all containers and data**: Run `docker-compose down -v && docker-compose up -d`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 