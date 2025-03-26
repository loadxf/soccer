"""
Monitoring and Alerting Utility for Soccer Prediction System

This module provides centralized monitoring and metrics collection functionality 
for the Soccer Prediction System, including:
- Prometheus metrics collection
- Custom metrics for ML model performance
- Health check utilities
- Alerting integrations

Usage:
    from src.utils.monitoring import metrics, monitor_endpoint, record_prediction_metrics
"""

import os
import time
import socket
import smtplib
import asyncio
import platform
import traceback
import functools
import subprocess
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from enum import Enum

import aiohttp
import psutil
import redis
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from fastapi import FastAPI, Request, Response, status
import requests
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, 
    start_http_server, push_to_gateway,
    REGISTRY, PROCESS_COLLECTOR, PLATFORM_COLLECTOR, CollectorRegistry
)

from src.utils.logger import get_logger
from src.utils.db import db_session

# Configure logger
logger = get_logger("monitoring")

# Try to import configuration
try:
    from config.default_config import (
        METRICS_ENABLED,
        HEALTH_CHECK_INTERVAL,
        ENABLE_EMAIL_ALERTS,
        ALERT_EMAIL_FROM,
        ALERT_EMAIL_TO,
        SMTP_SERVER,
        SMTP_PORT,
        SMTP_USER,
        SMTP_PASSWORD,
        SMTP_USE_TLS,
        MONITOR_EXTERNAL_SERVICES,
        EXTERNAL_SERVICE_TIMEOUT,
        SLOW_REQUEST_THRESHOLD,
        LOG_SLOW_REQUESTS,
        REDIS_HOST,
        REDIS_PORT,
        REDIS_DB,
        VERSION
    )
except ImportError:
    # Fallback to defaults if config is not available
    METRICS_ENABLED = True
    HEALTH_CHECK_INTERVAL = 60
    ENABLE_EMAIL_ALERTS = False
    ALERT_EMAIL_FROM = "alerts@example.com"
    ALERT_EMAIL_TO = "admin@example.com"
    SMTP_SERVER = "smtp.example.com"
    SMTP_PORT = 587
    SMTP_USER = ""
    SMTP_PASSWORD = ""
    SMTP_USE_TLS = True
    MONITOR_EXTERNAL_SERVICES = True
    EXTERNAL_SERVICE_TIMEOUT = 5.0
    SLOW_REQUEST_THRESHOLD = 1.0
    LOG_SLOW_REQUESTS = True
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379
    REDIS_DB = 0
    VERSION = "0.1.0"

# System constants
START_TIME = datetime.now()
HOSTNAME = socket.gethostname()
SERVICE_STATUS_CHECK_INTERVAL = 60  # seconds
CRITICAL_CPU_THRESHOLD = 90  # percentage
CRITICAL_MEMORY_THRESHOLD = 90  # percentage
CRITICAL_DISK_THRESHOLD = 90  # percentage

# Global health status
health_status = {
    "status": "starting",
    "version": VERSION,
    "time": datetime.now().isoformat(),
    "uptime": "0s",
    "hostname": HOSTNAME,
    "services": {
        "database": "unknown",
        "redis": "unknown",
        "models": "unknown"
    },
    "system": {
        "cpu": 0,
        "memory": 0,
        "disk": 0
    },
    "last_check": datetime.now().isoformat()
}

# Track alerts to prevent spamming
last_alerts = {}

# Define metric collectors
REQUEST_COUNT = Counter(
    "http_requests_total", 
    "Total HTTP Requests Count",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", 
    "HTTP Request Latency",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf"))
)

DB_QUERY_LATENCY = Histogram(
    "db_query_duration_seconds", 
    "Database Query Latency",
    ["query_type", "table"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, float("inf"))
)

API_ERRORS = Counter(
    "api_errors_total", 
    "Total API Errors",
    ["method", "endpoint", "exception_type"]
)

MODEL_PREDICTION_LATENCY = Histogram(
    "model_prediction_duration_seconds", 
    "Model Prediction Latency",
    ["model_name", "model_version"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, float("inf"))
)

MODEL_PREDICTION_COUNT = Counter(
    "model_predictions_total", 
    "Total Model Predictions",
    ["model_name", "model_version", "prediction_type"]
)

MODEL_ACCURACY = Gauge(
    "model_accuracy", 
    "Model Accuracy",
    ["model_name", "model_version", "metric_type"]
)

SYSTEM_MEMORY_USAGE = Gauge(
    "system_memory_usage_bytes", 
    "System Memory Usage in Bytes",
    ["type"]
)

SYSTEM_CPU_USAGE = Gauge(
    "system_cpu_usage_percent", 
    "System CPU Usage Percentage",
    ["cpu"]
)

API_REQUESTS_IN_PROGRESS = Gauge(
    "api_requests_in_progress", 
    "API Requests currently being processed",
    ["method", "endpoint"]
)

DATA_PIPELINE_DURATION = Histogram(
    "data_pipeline_duration_seconds", 
    "Data Pipeline Processing Duration",
    ["pipeline_name", "stage"]
)

# Alert thresholds
class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertManager:
    """Manages alerts for monitoring system"""
    
    def __init__(self):
        self.alert_endpoints = {
            "slack": os.getenv("SLACK_WEBHOOK_URL"),
            "pagerduty": os.getenv("PAGERDUTY_INTEGRATION_KEY"),
            "email": os.getenv("ALERT_EMAIL")
        }
        self.enabled = os.getenv("ENABLE_ALERTS", "true").lower() == "true"
    
    def send_alert(
        self, 
        title: str, 
        message: str, 
        severity: AlertSeverity = AlertSeverity.WARNING,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send an alert through configured channels
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            context: Additional context/metadata for the alert
            
        Returns:
            bool: Success status
        """
        if not self.enabled:
            logger.info(f"Alert triggered but alerting disabled: {title}")
            return False
            
        context = context or {}
        context.update({
            "severity": severity.value,
            "timestamp": time.time(),
            "environment": os.getenv("APP_ENV", "development")
        })
        
        success = False
        
        # Log all alerts
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }
        log_method[severity](f"ALERT - {title}: {message}")
        
        # Send to Slack if configured
        if self.alert_endpoints["slack"]:
            try:
                self._send_slack_alert(title, message, severity, context)
                success = True
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {str(e)}")
        
        # Send to PagerDuty if configured and high severity
        if self.alert_endpoints["pagerduty"] and severity in (AlertSeverity.ERROR, AlertSeverity.CRITICAL):
            try:
                self._send_pagerduty_alert(title, message, severity, context)
                success = True
            except Exception as e:
                logger.error(f"Failed to send PagerDuty alert: {str(e)}")
        
        # Send email if configured
        if self.alert_endpoints["email"]:
            try:
                self._send_email_alert(title, message, severity, context)
                success = True
            except Exception as e:
                logger.error(f"Failed to send email alert: {str(e)}")
                
        return success
    
    def _send_slack_alert(
        self, 
        title: str, 
        message: str, 
        severity: AlertSeverity,
        context: Dict[str, Any]
    ) -> None:
        """Send alert to Slack"""
        color = {
            AlertSeverity.INFO: "#36a64f",  # green
            AlertSeverity.WARNING: "#ffcc00",  # yellow
            AlertSeverity.ERROR: "#ff9966",  # orange
            AlertSeverity.CRITICAL: "#ff0000",  # red
        }
        
        payload = {
            "attachments": [
                {
                    "fallback": f"{severity.value.upper()}: {title}",
                    "color": color[severity],
                    "title": f"{severity.value.upper()}: {title}",
                    "text": message,
                    "fields": [
                        {"title": key, "value": str(value), "short": True}
                        for key, value in context.items() if key not in ("severity", "timestamp")
                    ],
                    "footer": f"Soccer Prediction System | {context['environment']}",
                    "ts": int(context["timestamp"])
                }
            ]
        }
        
        response = requests.post(self.alert_endpoints["slack"], json=payload)
        if response.status_code != 200:
            raise ValueError(f"Slack API returned {response.status_code}: {response.text}")
    
    def _send_pagerduty_alert(
        self, 
        title: str, 
        message: str, 
        severity: AlertSeverity,
        context: Dict[str, Any]
    ) -> None:
        """Send alert to PagerDuty"""
        event_action = "trigger"
        
        pd_severity = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }
        
        payload = {
            "routing_key": self.alert_endpoints["pagerduty"],
            "event_action": event_action,
            "payload": {
                "summary": title,
                "severity": pd_severity[severity],
                "source": f"Soccer Prediction System - {context['environment']}",
                "custom_details": {
                    "message": message,
                    **{k: v for k, v in context.items() if k not in ("severity", "timestamp")}
                }
            }
        }
        
        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code != 202:
            raise ValueError(f"PagerDuty API returned {response.status_code}: {response.text}")
    
    def _send_email_alert(
        self, 
        title: str, 
        message: str, 
        severity: AlertSeverity,
        context: Dict[str, Any]
    ) -> None:
        """Send email alert using SMTP or a service like SendGrid"""
        # This is a placeholder for email sending logic
        # In a real implementation, this would use smtplib or a service like SendGrid
        logger.info(f"Would send email to {self.alert_endpoints['email']}: {title} - {message}")
        # Implementation would go here


# Create a global AlertManager instance
alert_manager = AlertManager()


def get_formatted_uptime() -> str:
    """Get uptime in a human-readable format."""
    uptime = datetime.now() - START_TIME
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    
    parts.append(f"{seconds}s")
    
    return " ".join(parts)


def get_system_health() -> Dict[str, float]:
    """Get system health metrics."""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Disk usage
    disk = psutil.disk_usage('/')
    disk_percent = disk.percent
    
    return {
        "cpu": cpu_percent,
        "memory": memory_percent,
        "disk": disk_percent
    }


async def check_database_health() -> str:
    """Check the health of the database connection."""
    try:
        # Execute a simple query to check the connection
        with db_session() as session:
            result = session.execute(text("SELECT 1")).fetchone()
            if result and result[0] == 1:
                return "up"
            else:
                logger.warning("Database health check failed: unexpected result")
                return "degraded"
    except SQLAlchemyError as e:
        logger.error(f"Database health check failed: {e}")
        return "down"
    except Exception as e:
        logger.error(f"Unexpected error in database health check: {e}")
        return "error"


async def check_redis_health() -> str:
    """Check the health of the Redis connection."""
    try:
        # Connect to Redis
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            socket_timeout=2
        )
        
        # Ping to check the connection
        if r.ping():
            # Try a simple SET/GET operation
            test_key = "health_check_test"
            test_value = f"test_{int(time.time())}"
            r.set(test_key, test_value, ex=10)  # Expire in 10 seconds
            
            retrieved = r.get(test_key)
            if retrieved and retrieved.decode('utf-8') == test_value:
                return "up"
            else:
                logger.warning("Redis health check failed: SET/GET test failed")
                return "degraded"
        else:
            logger.warning("Redis health check failed: ping failed")
            return "degraded"
    except redis.RedisError as e:
        logger.error(f"Redis health check failed: {e}")
        return "down"
    except Exception as e:
        logger.error(f"Unexpected error in Redis health check: {e}")
        return "error"


async def check_models_health() -> str:
    """Check if the prediction models are loaded and working."""
    try:
        # Import here to avoid circular imports
        from src.models.prediction import prediction_service
        
        # Check if models are available
        models = prediction_service.get_available_models()
        if not models:
            logger.warning("Models health check failed: no models available")
            return "degraded"
        
        # Try a simple prediction
        prediction = prediction_service.predict_match(
            home_team_id=1,
            away_team_id=2,
        )
        
        if prediction:
            return "up"
        else:
            logger.warning("Models health check failed: prediction failed")
            return "degraded"
    except ImportError:
        logger.error("Models health check failed: prediction_service not available")
        return "unknown"
    except Exception as e:
        logger.error(f"Unexpected error in models health check: {e}")
        return "error"


async def check_external_service(url: str, timeout: float = EXTERNAL_SERVICE_TIMEOUT) -> Tuple[bool, float]:
    """
    Check if an external service is reachable.
    
    Args:
        url: The URL to check
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (is_healthy, response_time)
    """
    try:
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                duration = time.time() - start_time
                is_healthy = 200 <= response.status < 500
                return is_healthy, duration
    except asyncio.TimeoutError:
        logger.warning(f"External service check timed out: {url}")
        return False, timeout
    except Exception as e:
        logger.error(f"External service check failed: {url} - {e}")
        return False, 0


async def update_health_status():
    """Update the global health status."""
    global health_status
    
    # Get system health
    system_health = get_system_health()
    
    # Check services health (concurrently)
    db_status_task = asyncio.create_task(check_database_health())
    redis_status_task = asyncio.create_task(check_redis_health())
    models_status_task = asyncio.create_task(check_models_health())
    
    db_status = await db_status_task
    redis_status = await redis_status_task
    models_status = await models_status_task
    
    # Determine overall status
    services_status = [db_status, redis_status, models_status]
    if any(status == "down" for status in services_status):
        overall_status = "error"
    elif any(status == "degraded" for status in services_status):
        overall_status = "degraded"
    elif any(status == "unknown" for status in services_status):
        overall_status = "unknown"
    else:
        overall_status = "ok"
    
    # Update global status
    health_status = {
        "status": overall_status,
        "version": VERSION,
        "time": datetime.now().isoformat(),
        "uptime": get_formatted_uptime(),
        "hostname": HOSTNAME,
        "services": {
            "database": db_status,
            "redis": redis_status,
            "models": models_status
        },
        "system": system_health,
        "last_check": datetime.now().isoformat()
    }
    
    # Log health status
    logger.debug(f"Health status: {overall_status}")
    
    # Check for critical issues
    check_for_critical_issues(health_status)


def check_for_critical_issues(status: Dict[str, Any]):
    """
    Check for critical issues and trigger alerts if needed.
    
    Args:
        status: Current health status
    """
    alerts = []
    
    # Check service status
    for service, service_status in status["services"].items():
        if service_status == "down":
            alerts.append(f"Service {service} is DOWN")
    
    # Check system metrics
    if status["system"]["cpu"] > CRITICAL_CPU_THRESHOLD:
        alerts.append(f"Critical CPU usage: {status['system']['cpu']}%")
    
    if status["system"]["memory"] > CRITICAL_MEMORY_THRESHOLD:
        alerts.append(f"Critical memory usage: {status['system']['memory']}%")
    
    if status["system"]["disk"] > CRITICAL_DISK_THRESHOLD:
        alerts.append(f"Critical disk usage: {status['system']['disk']}%")
    
    # Send alerts if needed
    if alerts:
        send_alerts(alerts)


def send_alerts(issues: List[str]):
    """
    Send alerts for critical issues.
    
    Args:
        issues: List of critical issues
    """
    # Check if we should send email alerts
    if not ENABLE_EMAIL_ALERTS:
        logger.warning(f"Critical issues detected but email alerts are disabled: {', '.join(issues)}")
        return
    
    # Check for alert throttling
    now = datetime.now()
    for issue in issues:
        if issue in last_alerts:
            # Only alert once per hour for the same issue
            if now - last_alerts[issue] < timedelta(hours=1):
                logger.info(f"Skipping alert for {issue} - already sent recently")
                continue
        
        # Update last alert time
        last_alerts[issue] = now
        
        # Send email alert
        send_email_alert(issue)
        
        # Log alert
        logger.warning(f"Sent alert for: {issue}")


def send_email_alert(issue: str):
    """
    Send an email alert for a critical issue.
    
    Args:
        issue: The critical issue
    """
    if not SMTP_SERVER or not SMTP_USER or not SMTP_PASSWORD:
        logger.error("Cannot send email alert - SMTP settings not configured")
        return
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = ALERT_EMAIL_FROM
        msg['To'] = ALERT_EMAIL_TO
        msg['Subject'] = f"ALERT: Soccer Prediction API - {issue}"
        
        # Build email body
        body = f"""
        <html>
        <body>
            <h2>Critical Alert: Soccer Prediction API</h2>
            <p><strong>Issue:</strong> {issue}</p>
            <p><strong>Time:</strong> {datetime.now().isoformat()}</p>
            <p><strong>Server:</strong> {HOSTNAME}</p>
            <p><strong>System Status:</strong></p>
            <ul>
                <li>CPU: {health_status['system']['cpu']}%</li>
                <li>Memory: {health_status['system']['memory']}%</li>
                <li>Disk: {health_status['system']['disk']}%</li>
            </ul>
            <p><strong>Service Status:</strong></p>
            <ul>
                <li>Database: {health_status['services']['database']}</li>
                <li>Redis: {health_status['services']['redis']}</li>
                <li>Models: {health_status['services']['models']}</li>
            </ul>
            <p>Please check the system immediately.</p>
        </body>
        </html>
        """
        
        # Attach body
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        if SMTP_USE_TLS:
            server.starttls()
        
        # Login
        server.login(SMTP_USER, SMTP_PASSWORD)
        
        # Send email
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email alert sent for: {issue}")
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")


async def health_check_worker():
    """Background worker to periodically check system health."""
    while True:
        try:
            await update_health_status()
        except Exception as e:
            logger.error(f"Error in health check worker: {e}")
            traceback.print_exc()
        
        # Wait for next check
        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


def get_current_health() -> Dict[str, Any]:
    """Get the current health status."""
    return health_status


def slow_request_monitor():
    """
    Decorator to monitor slow requests.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapped(request: Request, *args, **kwargs):
            # Skip monitoring if disabled
            if not LOG_SLOW_REQUESTS:
                return await func(request, *args, **kwargs)
            
            # Record start time
            start_time = time.time()
            
            # Process the request
            response = await func(request, *args, **kwargs)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log slow requests
            if duration > SLOW_REQUEST_THRESHOLD:
                logger.warning(
                    f"Slow request detected: {request.method} {request.url.path} - {duration:.2f}s"
                )
            
            return response
        return wrapped
    return decorator


def start_monitoring_services(app: FastAPI):
    """
    Start monitoring services for a FastAPI application.
    
    Args:
        app: The FastAPI application
    """
    # Only register if monitoring is enabled
    if not METRICS_ENABLED:
        logger.info("Monitoring disabled, skipping service startup")
        return
    
    @app.on_event("startup")
    async def start_monitoring():
        # Start health check worker
        asyncio.create_task(health_check_worker())
        logger.info("Health check worker started")


# Simple service for calculating metrics
class HealthService:
    @classmethod
    async def get_health(cls) -> Dict[str, Any]:
        """Get the current health status."""
        return get_current_health()


def monitor_endpoint(method: str, endpoint: str):
    """
    Decorator to monitor FastAPI endpoint performance
    
    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: Endpoint path
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            # Increment in-progress counter
            API_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
            
            # Record start time
            start_time = time.time()
            status_code = 500  # Default to error
            
            try:
                # Call the original function
                response = await func(*args, **kwargs)
                
                # Get status code from the response
                if hasattr(response, "status_code"):
                    status_code = response.status_code
                elif isinstance(response, Response):
                    status_code = response.status_code
                else:
                    status_code = 200
                
                return response
            except Exception as e:
                # Record exception
                exception_type = type(e).__name__
                API_ERRORS.labels(
                    method=method, 
                    endpoint=endpoint,
                    exception_type=exception_type
                ).inc()
                
                # Send alert for 5xx errors
                if hasattr(e, "status_code") and e.status_code >= 500:
                    status_code = e.status_code
                    alert_manager.send_alert(
                        title=f"API Error: {endpoint}",
                        message=f"Exception in API endpoint: {str(e)}\n{traceback.format_exc()}",
                        severity=AlertSeverity.ERROR,
                        context={
                            "method": method,
                            "endpoint": endpoint,
                            "exception_type": exception_type,
                            "status_code": status_code
                        }
                    )
                
                # Re-raise the exception
                raise
            finally:
                # Record metrics
                duration = time.time() - start_time
                REQUEST_COUNT.labels(
                    method=method, 
                    endpoint=endpoint,
                    status=status_code
                ).inc()
                REQUEST_LATENCY.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
                
                # Alert on slow requests
                if duration > 5.0:  # 5 seconds threshold
                    alert_manager.send_alert(
                        title="Slow API Request",
                        message=f"Request to {method} {endpoint} took {duration:.2f}s",
                        severity=AlertSeverity.WARNING,
                        context={
                            "method": method,
                            "endpoint": endpoint,
                            "duration": f"{duration:.2f}s",
                            "status_code": status_code
                        }
                    )
                
                # Decrement in-progress counter
                API_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
        
        return wrapper
    return decorator


def monitor_database(query_type: str, table: str):
    """
    Decorator to monitor database query performance
    
    Args:
        query_type: Type of query (SELECT, INSERT, etc.)
        table: Database table name
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                DB_QUERY_LATENCY.labels(
                    query_type=query_type,
                    table=table
                ).observe(duration)
                
                # Alert on very slow database queries
                if duration > 1.0:  # 1 second threshold
                    alert_manager.send_alert(
                        title="Slow Database Query",
                        message=f"{query_type} query on {table} took {duration:.2f}s",
                        severity=AlertSeverity.WARNING,
                        context={
                            "query_type": query_type,
                            "table": table,
                            "duration": f"{duration:.2f}s"
                        }
                    )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                DB_QUERY_LATENCY.labels(
                    query_type=query_type,
                    table=table
                ).observe(duration)
                
                # Alert on very slow database queries
                if duration > 1.0:  # 1 second threshold
                    alert_manager.send_alert(
                        title="Slow Database Query",
                        message=f"{query_type} query on {table} took {duration:.2f}s",
                        severity=AlertSeverity.WARNING,
                        context={
                            "query_type": query_type,
                            "table": table,
                            "duration": f"{duration:.2f}s"
                        }
                    )
        
        # Choose the appropriate wrapper based on if the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def record_prediction_metrics(model_name: str, model_version: str, prediction_type: str, duration: float) -> None:
    """
    Record metrics for model predictions
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        prediction_type: Type of prediction
        duration: Prediction duration in seconds
    """
    MODEL_PREDICTION_COUNT.labels(
        model_name=model_name,
        model_version=model_version,
        prediction_type=prediction_type
    ).inc()
    
    MODEL_PREDICTION_LATENCY.labels(
        model_name=model_name,
        model_version=model_version
    ).observe(duration)


def update_model_accuracy(model_name: str, model_version: str, accuracy: float, metric_type: str = "accuracy") -> None:
    """
    Update model accuracy metrics
    
    Args:
        model_name: Name of the model
        model_version: Version of the model
        accuracy: Accuracy value (0.0 - 1.0)
        metric_type: Type of accuracy metric
    """
    MODEL_ACCURACY.labels(
        model_name=model_name,
        model_version=model_version,
        metric_type=metric_type
    ).set(accuracy)


def record_pipeline_duration(pipeline_name: str, stage: str, duration: float) -> None:
    """
    Record data pipeline processing duration
    
    Args:
        pipeline_name: Name of the pipeline
        stage: Pipeline stage
        duration: Processing duration in seconds
    """
    DATA_PIPELINE_DURATION.labels(
        pipeline_name=pipeline_name,
        stage=stage
    ).observe(duration)


def monitor_system_metrics(interval: int = 15) -> None:
    """
    Start a background thread to monitor system metrics
    
    Args:
        interval: Polling interval in seconds
    """
    import threading
    
    def update_metrics():
        while True:
            try:
                # Update memory metrics
                memory = psutil.virtual_memory()
                SYSTEM_MEMORY_USAGE.labels(type="total").set(memory.total)
                SYSTEM_MEMORY_USAGE.labels(type="available").set(memory.available)
                SYSTEM_MEMORY_USAGE.labels(type="used").set(memory.used)
                
                # Update CPU metrics
                for i, percentage in enumerate(psutil.cpu_percent(percpu=True)):
                    SYSTEM_CPU_USAGE.labels(cpu=f"cpu{i}").set(percentage)
                SYSTEM_CPU_USAGE.labels(cpu="total").set(psutil.cpu_percent())
                
                # Check for high resource usage
                if memory.percent > 90:
                    alert_manager.send_alert(
                        title="High Memory Usage",
                        message=f"System memory usage is at {memory.percent:.1f}%",
                        severity=AlertSeverity.WARNING,
                        context={"memory_percent": f"{memory.percent:.1f}%"}
                    )
                
                total_cpu = psutil.cpu_percent()
                if total_cpu > 90:
                    alert_manager.send_alert(
                        title="High CPU Usage",
                        message=f"System CPU usage is at {total_cpu:.1f}%",
                        severity=AlertSeverity.WARNING,
                        context={"cpu_percent": f"{total_cpu:.1f}%"}
                    )
                
                # Sleep for the interval
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error updating system metrics: {str(e)}")
                time.sleep(interval)
    
    # Start the metrics thread
    metrics_thread = threading.Thread(target=update_metrics, daemon=True)
    metrics_thread.start()


def setup_prometheus_metrics(port: int = 9090) -> None:
    """
    Start Prometheus metrics HTTP server
    
    Args:
        port: HTTP port for metrics endpoint
    """
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")


def push_to_prometheus_gateway(job_name: str, gateway_url: str = "127.0.0.1:9091") -> None:
    """
    Push metrics to Prometheus Pushgateway
    
    Args:
        job_name: Job name for grouping metrics
        gateway_url: URL of the Prometheus Pushgateway
    """
    try:
        push_to_gateway(gateway_url, job=job_name, registry=REGISTRY)
    except Exception as e:
        logger.error(f"Failed to push metrics to gateway: {str(e)}")


def health_check() -> Dict[str, Any]:
    """
    Perform a health check of system components
    
    Returns:
        dict: Health check results
    """
    results = {
        "status": "ok",
        "timestamp": time.time(),
        "components": {
            "system": {"status": "ok"},
            "api": {"status": "ok"},
            "database": {"status": "ok"},
            "models": {"status": "ok"}
        }
    }
    
    # Check system resources
    try:
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            results["components"]["system"] = {
                "status": "warning",
                "message": f"High memory usage: {memory.percent:.1f}%"
            }
            results["status"] = "warning"
    except Exception as e:
        results["components"]["system"] = {
            "status": "unknown",
            "message": f"Failed to check system resources: {str(e)}"
        }
    
    # Database check would go here
    # Model health check would go here
    
    return results


# Import asyncio at the end to avoid circular imports
import asyncio 