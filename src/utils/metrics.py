"""
Metrics collection and monitoring utility for the Soccer Prediction System.

This module provides metrics collection functionality using Prometheus client library
for monitoring API performance, request rates, errors, and resource usage.
"""

import time
from typing import Callable, Dict, Optional, List, Union
import os
import platform
import psutil
import functools

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
from prometheus_client import push_to_gateway, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

# Configure logger
logger = get_logger("metrics")

# Try to import configuration
try:
    from config.default_config import (
        METRICS_ENABLED, 
        PUSH_GATEWAY_URL, 
        PUSH_GATEWAY_JOB,
        COLLECT_DEFAULT_METRICS,
        API_PREFIX
    )
except ImportError:
    # Fallback to defaults if config is not available
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    PUSH_GATEWAY_URL = os.getenv("PUSH_GATEWAY_URL", "")
    PUSH_GATEWAY_JOB = os.getenv("PUSH_GATEWAY_JOB", "soccer_prediction_api")
    COLLECT_DEFAULT_METRICS = os.getenv("COLLECT_DEFAULT_METRICS", "true").lower() == "true"
    API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

# Create registry
registry = CollectorRegistry()

# Define metrics
HTTP_REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=registry
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0, float("inf")),
    registry=registry
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently in progress",
    ["method", "endpoint"],
    registry=registry
)

HTTP_RESPONSE_SIZE = Summary(
    "http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "endpoint"],
    registry=registry
)

CACHE_HIT_COUNTER = Counter(
    "cache_hits_total",
    "Total number of cache hits",
    ["endpoint"],
    registry=registry
)

CACHE_MISS_COUNTER = Counter(
    "cache_misses_total",
    "Total number of cache misses",
    ["endpoint"],
    registry=registry
)

DB_QUERY_DURATION = Histogram(
    "db_query_duration_seconds",
    "Database query duration in seconds",
    ["query_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")),
    registry=registry
)

PREDICTION_REQUEST_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["model", "match_type"],
    registry=registry
)

PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds",
    "Time taken to generate a prediction",
    ["model", "match_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")),
    registry=registry
)

ERROR_COUNTER = Counter(
    "errors_total",
    "Total number of errors",
    ["error_type", "endpoint"],
    registry=registry
)

API_ERROR_COUNTER = Counter(
    "api_errors_total",
    "Total number of API errors",
    ["error_code", "endpoint", "method"],
    registry=registry
)

AUTH_REQUEST_COUNTER = Counter(
    "auth_requests_total",
    "Total number of authentication requests",
    ["status", "auth_type"],
    registry=registry
)

# System metrics
SYSTEM_CPU_USAGE = Gauge(
    "system_cpu_usage",
    "Current CPU usage percentage",
    registry=registry
)

SYSTEM_MEMORY_USAGE = Gauge(
    "system_memory_usage_bytes",
    "Current memory usage in bytes",
    registry=registry
)

SYSTEM_MEMORY_TOTAL = Gauge(
    "system_memory_total_bytes",
    "Total system memory in bytes",
    registry=registry
)

PROCESS_CPU_USAGE = Gauge(
    "process_cpu_usage",
    "Current process CPU usage percentage",
    registry=registry
)

PROCESS_MEMORY_USAGE = Gauge(
    "process_memory_usage_bytes",
    "Current process memory usage in bytes",
    registry=registry
)

PROCESS_OPEN_FDS = Gauge(
    "process_open_file_descriptors",
    "Number of open file descriptors",
    registry=registry
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        logger.info("Metrics middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get the route path to use as the endpoint label
        route = request.url.path
        method = request.method
        
        # Skip metrics endpoint itself to avoid recursion
        if route == f"{API_PREFIX}/metrics":
            return await call_next(request)
        
        # Normalize API routes for better metric grouping
        # e.g., /api/v1/teams/123 becomes /api/v1/teams/{id}
        if API_PREFIX in route:
            parts = route.split('/')
            for i, part in enumerate(parts):
                # If this part looks like an ID (numeric), replace it with {id}
                if part.isdigit() and i > 0:
                    parts[i] = "{id}"
            route = '/'.join(parts)
        
        # Track in-progress requests
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=route).inc()
        
        # Record start time
        start_time = time.time()
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Record end time and calculate duration
            duration = time.time() - start_time
            status_code = response.status_code
            
            # Update metrics
            HTTP_REQUEST_COUNTER.labels(
                method=method, endpoint=route, status_code=status_code
            ).inc()
            
            HTTP_REQUEST_DURATION.labels(
                method=method, endpoint=route
            ).observe(duration)
            
            # Try to get response size
            try:
                if hasattr(response, "headers") and "content-length" in response.headers:
                    content_length = int(response.headers["content-length"])
                    HTTP_RESPONSE_SIZE.labels(method=method, endpoint=route).observe(content_length)
            except (ValueError, TypeError, KeyError):
                # If we can't get the size, just skip this metric
                pass
            
            # Check for cache hit/miss
            if hasattr(response, "headers") and "X-Cache" in response.headers:
                if response.headers["X-Cache"] == "HIT":
                    CACHE_HIT_COUNTER.labels(endpoint=route).inc()
                elif response.headers["X-Cache"] == "MISS":
                    CACHE_MISS_COUNTER.labels(endpoint=route).inc()
            
            # Track errors if status code indicates an error
            if status_code >= 400:
                error_type = "client_error" if status_code < 500 else "server_error"
                ERROR_COUNTER.labels(error_type=error_type, endpoint=route).inc()
                
                # If there's an error_code in the response, track it
                try:
                    if (
                        hasattr(response, "body") 
                        and response.body 
                        and b"error_code" in response.body
                    ):
                        import json
                        error_data = json.loads(response.body)
                        if "error_code" in error_data:
                            API_ERROR_COUNTER.labels(
                                error_code=error_data["error_code"],
                                endpoint=route,
                                method=method
                            ).inc()
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                    # Skip if we can't parse the error
                    pass
            
            return response
            
        except Exception as e:
            # Record errors for exceptions
            ERROR_COUNTER.labels(
                error_type=type(e).__name__,
                endpoint=route
            ).inc()
            
            # Re-raise the exception
            raise
            
        finally:
            # Always decrement in-progress counter
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=route).dec()


def collect_system_metrics():
    """Collect system metrics like CPU and memory usage."""
    # System-wide CPU usage
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    
    # System memory
    memory = psutil.virtual_memory()
    SYSTEM_MEMORY_USAGE.set(memory.used)
    SYSTEM_MEMORY_TOTAL.set(memory.total)
    
    # Process-specific metrics
    process = psutil.Process(os.getpid())
    
    # Process CPU usage
    PROCESS_CPU_USAGE.set(process.cpu_percent(interval=0.1))
    
    # Process memory usage
    PROCESS_MEMORY_USAGE.set(process.memory_info().rss)
    
    # Open file descriptors (Unix-like systems only)
    if platform.system() != "Windows":
        PROCESS_OPEN_FDS.set(process.num_fds())


def push_metrics():
    """Push metrics to a Prometheus Pushgateway."""
    if not PUSH_GATEWAY_URL:
        logger.warning("PUSH_GATEWAY_URL not configured, skipping push")
        return
    
    try:
        # Collect system metrics before pushing
        collect_system_metrics()
        
        # Push metrics to gateway
        push_to_gateway(
            PUSH_GATEWAY_URL, 
            job=PUSH_GATEWAY_JOB,
            registry=registry
        )
        logger.debug(f"Metrics pushed to gateway: {PUSH_GATEWAY_URL}")
    except Exception as e:
        logger.error(f"Failed to push metrics: {e}")


def timed(metric: Optional[Histogram] = None, labels: Optional[Dict] = None):
    """
    Decorator to time function execution and record it to a Histogram metric.
    
    Args:
        metric: The Histogram metric to record to
        labels: Dictionary of labels for the metric
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # If metrics are disabled or no metric is provided, just call the function
            if not METRICS_ENABLED or metric is None:
                return func(*args, **kwargs)
            
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapped
    return decorator


def track_prediction(model: str, match_type: str):
    """
    Decorator to track prediction requests.
    
    Args:
        model: The prediction model name
        match_type: The type of match prediction (single, custom, batch)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # If metrics are disabled, just call the function
            if not METRICS_ENABLED:
                return func(*args, **kwargs)
            
            # Increment counter
            PREDICTION_REQUEST_COUNTER.labels(model=model, match_type=match_type).inc()
            
            # Measure prediction time
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                PREDICTION_DURATION.labels(model=model, match_type=match_type).observe(duration)
        return wrapped
    return decorator


def track_db_query(query_type: str):
    """
    Decorator to track database query execution time.
    
    Args:
        query_type: The type of database query
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # If metrics are disabled, just call the function
            if not METRICS_ENABLED:
                return func(*args, **kwargs)
            
            # Measure query time
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                DB_QUERY_DURATION.labels(query_type=query_type).observe(duration)
        return wrapped
    return decorator


def get_metrics():
    """Generate Prometheus metrics output."""
    # Collect system metrics
    collect_system_metrics()
    
    # Return metrics in Prometheus format
    return generate_latest(registry)


# Initialize metrics collection
if METRICS_ENABLED:
    logger.info("Metrics collection enabled")
    
    # Set initial values for system metrics
    if COLLECT_DEFAULT_METRICS:
        collect_system_metrics()
        logger.info("System metrics collection initialized")
else:
    logger.info("Metrics collection disabled") 