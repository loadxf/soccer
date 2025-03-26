# Soccer Prediction System - Developer Onboarding Guide

Welcome to the Soccer Prediction System development team! This guide will help you get set up with the codebase, understand the project architecture, and contribute effectively to the project.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Development Environment Setup](#development-environment-setup)
4. [Project Architecture](#project-architecture)
5. [Development Workflow](#development-workflow)
6. [Coding Standards](#coding-standards)
7. [Testing Guidelines](#testing-guidelines)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Common Development Tasks](#common-development-tasks)
10. [Troubleshooting](#troubleshooting)
11. [Resources and References](#resources-and-references)

## Introduction

The Soccer Prediction System is a comprehensive platform for predicting soccer match outcomes using machine learning. The project combines:

- **Data Engineering**: Collection, cleaning, and transformation of soccer data
- **Model Development**: Training and evaluation of prediction models
- **API Service**: FastAPI-based backend for serving predictions
- **Frontend Application**: React-based UI for interacting with the system

As a developer, you may work on one or more of these components. This guide will help you understand how these components fit together and how to contribute to each area.

## Getting Started

### Project Overview

- **GitHub Repository**: [https://github.com/your-organization/soccer-prediction](https://github.com/your-organization/soccer-prediction)
- **Issue Tracker**: GitHub Issues
- **Documentation**: Located in the `docs/` directory
- **Communication**: Team Slack channel #soccer-prediction

### First Week Checklist

For your first week, we recommend:

1. Set up your development environment (see [Development Environment Setup](#development-environment-setup))
2. Complete a walkthrough of the system with your onboarding buddy
3. Run the system locally and explore its functionality
4. Pick a small, well-defined issue to work on
5. Submit your first pull request

## Development Environment Setup

### Prerequisites

- **Git**: For source code management
- **Python 3.9+**: For backend development
- **Node.js 16+**: For frontend development
- **Docker**: For containerized development and testing
- **PostgreSQL 13+**: For local database (if not using Docker)

### Initial Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-organization/soccer-prediction.git
   cd soccer-prediction
   ```

2. **Set up the Python environment**:
   ```bash
   python -m venv .venv
   # On Linux/macOS
   source .venv/bin/activate
   # On Windows
   .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file with your local configuration
   ```

4. **Initialize the database**:
   ```bash
   python scripts/init_db.py
   ```

5. **Load sample data** (optional, but recommended for development):
   ```bash
   python scripts/load_sample_data.py
   ```

6. **Set up frontend development environment**:
   ```bash
   cd src/frontend
   npm install
   ```

### Using Docker for Development

For a fully containerized development environment:

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# Access services:
# - API: http://127.0.0.1:8000
# - Frontend: http://127.0.0.1:3000
# - Database: 127.0.0.1:5432

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Code Editor Setup

We recommend using VSCode with the following extensions:

- Python
- ESLint
- Prettier
- Docker
- GitLens

VSCode settings for the project are included in `.vscode/settings.json`.

## Project Architecture

### Directory Structure

```
soccer/
├── config/                   # Configuration files
│   └── default_config.py     # Default configuration settings
├── data/                     # Data storage
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Processed datasets
│   ├── features/             # Feature datasets
│   ├── models/               # Trained models
│   └── predictions/          # Prediction history
├── deployment/               # Deployment scripts and configurations
│   ├── aws/                  # AWS-specific deployment
│   ├── azure/                # Azure-specific deployment
│   └── gcp/                  # GCP-specific deployment
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
├── src/                      # Source code
│   ├── api/                  # API implementation
│   ├── data/                 # Data processing modules
│   ├── models/               # Model implementation
│   ├── frontend/             # React frontend
│   └── utils/                # Utility modules
├── tests/                    # Test directory
├── monitoring/               # Monitoring configurations
├── .github/                  # GitHub Actions workflows
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
└── docker-compose.yml        # Docker Compose configuration
```

### Component Overview

#### Backend (Python)

- **FastAPI**: Web framework for building the API
- **SQLAlchemy**: ORM for database interactions
- **Pydantic**: Data validation and settings management
- **Pandas/NumPy**: Data processing and analysis
- **Scikit-learn/PyTorch**: Model training and evaluation

#### Frontend (JavaScript/React)

- **React**: Frontend library
- **Material-UI**: Component library
- **Redux**: State management
- **React Query**: Data fetching and caching
- **Recharts**: Data visualization

#### DevOps

- **Docker**: Containerization
- **GitHub Actions**: CI/CD
- **Terraform**: Infrastructure as code
- **Prometheus/Grafana**: Monitoring and alerting

## Development Workflow

### Git Workflow

We follow a GitHub Flow workflow:

1. **Create a branch** from `main` for each feature or bug fix
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes**, committing frequently with clear messages
   ```bash
   git commit -m "Add feature X to component Y"
   ```

3. **Push your branch** to GitHub
   ```bash
   git push -u origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub with a clear description of your changes

5. **Address review feedback** and update your PR as needed

6. **Merge the PR** once approved (maintainers will handle this)

### Branch Naming Conventions

- `feature/description`: For new features
- `fix/description`: For bug fixes
- `refactor/description`: For code refactoring
- `docs/description`: For documentation changes
- `test/description`: For adding or modifying tests

### Commit Messages

Follow a clear and consistent format for commit messages:

```
type(scope): Short description

Longer description if needed
```

Types include: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(models): Add gradient boosting classifier

Implemented a gradient boosting classifier with hyperparameter tuning
for improved prediction accuracy on Premier League matches.
```

## Coding Standards

### Python Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [pylint](https://pylint.org/) for linting
- Use type hints for all function signatures

Example:
```python
from typing import List, Optional

def process_matches(match_ids: List[int], include_stats: bool = False) -> Optional[dict]:
    """
    Process match data for the given match IDs.
    
    Args:
        match_ids: List of match IDs to process
        include_stats: Whether to include detailed statistics
        
    Returns:
        Processed match data or None if no matches found
    """
    # Implementation
```

### JavaScript/TypeScript Coding Standards

- Use ESLint with the project configuration
- Use Prettier for code formatting
- Follow Airbnb JavaScript Style Guide
- Use TypeScript for type safety

Example:
```typescript
interface MatchProps {
  matchId: number;
  showDetails?: boolean;
}

const MatchCard: React.FC<MatchProps> = ({ matchId, showDetails = false }) => {
  // Implementation
};
```

### Documentation Standards

- Add docstrings to all Python functions, classes, and modules
- Use JSDoc for JavaScript functions and components
- Keep README and documentation files up to date
- Document API endpoints with OpenAPI/Swagger

## Testing Guidelines

### Backend Testing

- **Unit Tests**: Use pytest for unit testing
  ```bash
  # Run all tests
  pytest
  
  # Run specific test module
  pytest tests/models/test_prediction.py
  
  # Run with coverage report
  pytest --cov=src
  ```

- **Integration Tests**: Test API endpoints and database interactions
  ```bash
  pytest tests/integration/
  ```

- **Model Tests**: Validate model training and prediction
  ```bash
  pytest tests/models/
  ```

### Frontend Testing

- **Unit Tests**: Use Jest for component and utility testing
  ```bash
  cd src/frontend
  npm test
  
  # Run specific test
  npm test -- -t "MatchPrediction component"
  ```

- **E2E Tests**: Use Cypress for end-to-end testing
  ```bash
  cd src/frontend
  npm run cypress:open  # UI mode
  npm run cypress:run   # Headless mode
  ```

### Test-Driven Development

We encourage TDD practices:
1. Write a failing test for the functionality you want to implement
2. Implement the minimal code to make the test pass
3. Refactor your code while keeping tests passing

## CI/CD Pipeline

Our continuous integration and deployment pipeline is implemented using GitHub Actions.

### CI Workflow

On every pull request:
1. **Lint**: Check code style and quality
2. **Test**: Run unit and integration tests
3. **Build**: Build Docker images
4. **Security Scan**: Check for security vulnerabilities

### CD Workflow

On merge to main:
1. **Build**: Build and tag Docker images
2. **Push**: Push images to container registry
3. **Deploy**: Deploy to staging environment
4. **Smoke Test**: Run basic validation tests
5. **Manual Approval**: Required for production deployment
6. **Production Deploy**: Deploy to production environment

### Environment Strategy

- **Development**: Local environment for development
- **Staging**: Cloud environment for testing before production
- **Production**: Live environment for end users

## Common Development Tasks

### Adding a New Model

1. Create a new file in `src/models/` with your model implementation
2. Implement the required interfaces (`BaseModel`, `Trainable`, etc.)
3. Add unit tests in `tests/models/`
4. Register the model in `src/models/__init__.py`
5. Add the model to the benchmarking tool in `scripts/benchmark_models.py`

Example:
```python
from src.models.base import BaseModel

class NewPredictionModel(BaseModel):
    def __init__(self, params=None):
        self.params = params or {}
        self.model = None
    
    def train(self, X, y):
        # Implementation
        
    def predict(self, X):
        # Implementation
```

### Adding a New API Endpoint

1. Create a new route file in `src/api/routes/` or add to an existing file
2. Define request and response models using Pydantic
3. Implement the endpoint function
4. Add tests in `tests/api/`

Example:
```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/predictions", tags=["predictions"])

class PredictionRequest(BaseModel):
    match_id: int
    features: dict

class PredictionResponse(BaseModel):
    match_id: int
    home_win_probability: float
    draw_probability: float
    away_win_probability: float

@router.post("/", response_model=PredictionResponse)
async def create_prediction(request: PredictionRequest):
    # Implementation
```

### Adding a New Frontend Component

1. Create a new component file in `src/frontend/src/components/`
2. Implement the component using React and TypeScript
3. Add styles using Material-UI or CSS modules
4. Add tests in a corresponding `__tests__` directory

Example:
```tsx
import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { useQuery } from 'react-query';
import { fetchMatchDetails } from '../api/matches';

interface MatchDetailsProps {
  matchId: number;
}

export const MatchDetails: React.FC<MatchDetailsProps> = ({ matchId }) => {
  const { data, isLoading, error } = useQuery(['match', matchId], () => fetchMatchDetails(matchId));
  
  if (isLoading) return <CircularProgress />;
  if (error) return <Typography color="error">Error loading match details</Typography>;
  
  return (
    <Box>
      {/* Component implementation */}
    </Box>
  );
};
```

### Working with Data

1. **Accessing Sample Data**:
   ```python
   import pandas as pd
   from src.utils.data import get_data_path
   
   matches_df = pd.read_csv(get_data_path('processed/matches.csv'))
   ```

2. **Adding New Data Sources**:
   - Add data connector in `src/data/sources/`
   - Add data transformation in `src/data/transformations/`
   - Update the pipeline in `src/data/pipeline.py`

## Troubleshooting

### Common Issues

#### Database Connection Issues

**Issue**: Cannot connect to the database
**Solution**:
- Check your `.env` file for correct database credentials
- Ensure PostgreSQL is running (`docker-compose ps` or `pg_isready`)
- Try connecting with a database client to verify credentials

#### Docker Issues

**Issue**: Container fails to start
**Solution**:
- Check container logs: `docker-compose logs -f <service-name>`
- Verify environment variables and volumes
- Ensure ports are not already in use

#### Model Training Issues

**Issue**: Model training fails or produces poor results
**Solution**:
- Check input data shape and quality
- Verify feature preprocessing steps
- Check for data leakage in your pipeline
- Look for class imbalance issues

### Debug Tools and Techniques

- **API Debugging**: Use the `/debug` endpoints (development only)
- **Model Inspection**: Use the model inspector in `scripts/inspect_model.py`
- **Performance Profiling**: Use `scripts/profile_api.py` to identify bottlenecks

## Resources and References

### Internal Documentation

- [API Documentation](./api/README.md)
- [Model Architecture](./model_architecture.md)
- [Deployment Process](./deployment_process.md)
- [Database Migrations](./database_migrations.md)

### External Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/getting-started.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Material-UI Documentation](https://mui.com/getting-started/usage/)

### Learning Resources

- [Machine Learning Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Clean Code in Python](https://www.oreilly.com/library/view/clean-code-in/9781788835831/)
- [Effective TypeScript](https://effectivetypescript.com/)

### Contact Information

- **Tech Lead**: tech.lead@example.com
- **Data Science Lead**: ds.lead@example.com
- **DevOps Contact**: devops@example.com
- **Slack Channel**: #soccer-prediction 