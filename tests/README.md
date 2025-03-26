# Soccer Prediction System Tests

This directory contains tests for the Soccer Prediction System. The tests are organized by component:

- `api/`: Integration tests for the API endpoints
- `data/`: Tests for data processing and feature engineering
- `models/`: Tests for model training and prediction
- `utils/`: Tests for utility functions and services

## Running the Tests

### Prerequisites

Before running the tests, make sure you have installed all dependencies:

```bash
pip install -r requirements.txt
```

### Running All Tests

To run all tests:

```bash
python -m unittest discover -s tests
```

### Running Tests for a Specific Module

To run tests for a specific module:

```bash
python -m unittest discover -s tests/api  # Run API tests only
python -m unittest discover -s tests/models  # Run model tests only
python -m unittest discover -s tests/data  # Run data processing tests only
python -m unittest discover -s tests/utils  # Run utility tests only
```

### Running a Specific Test File

To run a specific test file:

```bash
python -m unittest tests/api/test_integration.py
```

### Running a Specific Test Class or Method

To run a specific test class:

```bash
python -m unittest tests.api.test_integration.TestAPIIntegration
```

To run a specific test method:

```bash
python -m unittest tests.api.test_integration.TestAPIIntegration.test_root_endpoint
```

## Docker Testing

You can also run the tests inside Docker:

```bash
docker build -f Dockerfile.test -t soccer-prediction-tests .
docker run soccer-prediction-tests
```

## Test Coverage

To generate a test coverage report:

```bash
pip install coverage
coverage run -m unittest discover -s tests
coverage report
coverage html  # Generates HTML report in htmlcov/
```

## Test Structure

Each test module follows a similar structure:

1. Unit tests for basic functionality
2. Integration tests for component interactions
3. Mock objects used to isolate dependencies
4. Setup and teardown methods to ensure test isolation

## API Tests

The API tests verify that:

- Endpoints return the expected status codes and data
- Authentication and authorization work correctly
- Rate limiting prevents abuse
- Error handling functions properly

## Data Processing Tests

Data processing tests verify that:

- Data is loaded correctly from various sources
- Cleaning and transformation produce the expected results
- Feature engineering creates the right features
- Data validation catches errors appropriately

## Model Tests

Model tests verify that:

- Models train correctly
- Predictions are in the expected format
- Model evaluation metrics work correctly
- Models can be saved and loaded properly

## Utility Tests

Utility tests verify that:

- Authentication works correctly
- Database connections function properly
- Logging records expected information
- Configuration loading works correctly

## Adding New Tests

When adding new functionality, follow these guidelines for testing:

1. Create unit tests for individual functions/methods
2. Create integration tests for interactions between components
3. Use mock objects to isolate dependencies
4. Add appropriate assertions to verify correctness
5. Keep tests independent and isolated 