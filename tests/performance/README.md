# Performance Testing

This directory contains performance tests for the Soccer Prediction System, measuring both API load testing and benchmark tests for critical components.

## Tools Used

- **Locust**: For API load testing
- **pytest-benchmark**: For benchmarking critical functions and components

## Running Load Tests

Load tests simulate multiple users accessing the API concurrently to measure:
- Response times under various loads
- System stability under sustained load
- Maximum throughput capability
- Identification of performance bottlenecks

### Starting the Locust Server

```bash
# Start the Locust server 
cd tests/performance
locust -f locustfile.py
```

Then open your browser at http://127.0.0.1:8089 to access the Locust web interface.

### Running Headless Load Tests

For CI/CD pipelines or automated testing, you can run Locust in headless mode:

```bash
locust -f locustfile.py --headless -u 100 -r 10 --run-time 30s
```

Parameters:
- `-u`: Number of users to simulate
- `-r`: Spawn rate (users per second)
- `--run-time`: Duration of the test

## Running Component Benchmarks

Benchmark tests measure the performance of individual components:

```bash
# Run all benchmarks
pytest tests/performance/test_benchmarks.py -v

# Run specific benchmark group
pytest tests/performance/test_benchmarks.py::TestModelBenchmarks -v
```

## Available Tests

### Load Tests
- `locustfile.py`: Main load testing file for API endpoints
- `custom_load_profiles.py`: Custom load profiles for simulating different user patterns

### Benchmark Tests
- `test_benchmarks.py`: Benchmarks for critical components
- `test_model_inference.py`: Specific benchmarks for model inference performance
- `test_data_processing.py`: Benchmarks for data processing operations

## Interpreting Results

Results are compared against established baselines to identify:
- Performance regressions
- Opportunities for optimization
- Scalability limitations

## Adding New Tests

When adding new functionality, consider adding corresponding performance tests:
1. For API endpoints, add new tasks to `locustfile.py`
2. For critical functions, add benchmark tests to the appropriate test file 