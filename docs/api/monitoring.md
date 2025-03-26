# Metrics and Monitoring

This documentation describes the metrics and monitoring system for the Soccer Prediction System API.

## Overview

The Soccer Prediction System API includes a comprehensive metrics and monitoring system that provides real-time insights into the performance, health, and usage patterns of the API. The system is built using industry-standard tools and best practices to ensure reliability and observability.

## Key Components

The metrics and monitoring system consists of the following key components:

1. **Metrics Collection**: Using Prometheus client library to collect custom metrics from the API
2. **Health Checks**: Monitoring the health of API and its dependencies
3. **Alerting**: Notifications for critical issues
4. **Dashboard**: Grafana dashboard for visualization

## Metrics Collection

The API collects various metrics to provide insights into its performance and usage:

### HTTP Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total number of HTTP requests, labeled by method, endpoint, and status code |
| `http_request_duration_seconds` | Histogram | HTTP request duration in seconds, labeled by method and endpoint |
| `http_requests_in_progress` | Gauge | Number of HTTP requests currently in progress, labeled by method and endpoint |
| `http_response_size_bytes` | Summary | HTTP response size in bytes, labeled by method and endpoint |

### Cache Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `cache_hits_total` | Counter | Total number of cache hits, labeled by endpoint |
| `cache_misses_total` | Counter | Total number of cache misses, labeled by endpoint |

### Database Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `db_query_duration_seconds` | Histogram | Database query duration in seconds, labeled by query type |

### Prediction Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter | Total number of prediction requests, labeled by model and match type |
| `prediction_duration_seconds` | Histogram | Time taken to generate a prediction, labeled by model and match type |

### Error Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `errors_total` | Counter | Total number of errors, labeled by error type and endpoint |
| `api_errors_total` | Counter | Total number of API errors, labeled by error code, endpoint, and method |
| `auth_requests_total` | Counter | Total number of authentication requests, labeled by status and auth type |

### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `system_cpu_usage` | Gauge | Current CPU usage percentage |
| `system_memory_usage_bytes` | Gauge | Current memory usage in bytes |
| `system_memory_total_bytes` | Gauge | Total system memory in bytes |
| `process_cpu_usage` | Gauge | Current process CPU usage percentage |
| `process_memory_usage_bytes` | Gauge | Current process memory usage in bytes |
| `process_open_file_descriptors` | Gauge | Number of open file descriptors |

## Accessing Metrics

The metrics are exposed through a Prometheus-compatible endpoint at:

```
/api/v1/metrics
```

This endpoint returns metrics in the Prometheus text format, which can be scraped by Prometheus or other compatible monitoring systems. The endpoint does not require authentication to allow for easy scraping by monitoring systems.

## Health Checks

The health check system monitors the health of the API and its dependencies. The health check endpoint is available at:

```
/api/v1/health
```

The health check returns a JSON response with the following information:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "time": "2023-03-15T12:00:00.000Z",
  "uptime": "10d 2h 30m 15s",
  "hostname": "api-server-1",
  "services": {
    "database": "up",
    "redis": "up",
    "models": "up"
  },
  "system": {
    "cpu": 25.5,
    "memory": 60.2,
    "disk": 45.0
  },
  "last_check": "2023-03-15T12:00:00.000Z"
}
```

### Service Status Values

| Value | Description |
|-------|-------------|
| `up` | Service is fully operational |
| `degraded` | Service is operational but not performing optimally |
| `down` | Service is not operational |
| `unknown` | Service status could not be determined |
| `error` | An error occurred while checking the service |

## Alerting System

The monitoring system includes an alerting capability that can send email notifications when critical issues are detected. Alerts are generated for:

- Service outages (database, Redis, model service)
- Critical CPU usage (above 90%)
- Critical memory usage (above 90%)
- Critical disk usage (above 90%)

### Configuring Alerts

Alerts can be configured through environment variables or the configuration file:

```
ENABLE_EMAIL_ALERTS=true
ALERT_EMAIL_FROM=alerts@example.com
ALERT_EMAIL_TO=admin@example.com
SMTP_SERVER=smtp.example.com
SMTP_PORT=587
SMTP_USER=user
SMTP_PASSWORD=password
SMTP_USE_TLS=true
```

## Dashboard

The metrics and monitoring system includes a pre-configured Grafana dashboard for visualization. The dashboard is available in `monitoring/dashboards/api_dashboard.json` and can be imported into Grafana.

The dashboard provides visualizations for:

- HTTP request rates and status codes
- API response times
- System resource usage
- Error rates
- Prediction service performance
- Cache hit/miss rates
- Service health status

## Deployment

### Prometheus

To collect and store metrics, deploy Prometheus with a configuration that includes:

```yaml
scrape_configs:
  - job_name: 'soccer-prediction-api'
    scrape_interval: 15s
    metrics_path: /api/v1/metrics
    static_configs:
      - targets: ['api-host:8000']
```

### Grafana

1. Deploy Grafana with Prometheus as a data source
2. Import the dashboard JSON from `monitoring/dashboards/api_dashboard.json`

## Configuration

The metrics and monitoring system can be configured through environment variables or the configuration file:

```
METRICS_ENABLED=true
COLLECT_DEFAULT_METRICS=true
PUSH_GATEWAY_URL=http://pushgateway:9091
PUSH_GATEWAY_JOB=soccer_prediction_api
METRICS_EXPORT_INTERVAL=15
HEALTH_CHECK_INTERVAL=60
ENABLE_PROMETHEUS_ENDPOINT=true
SLOW_REQUEST_THRESHOLD=1.0
LOG_SLOW_REQUESTS=true
MONITOR_EXTERNAL_SERVICES=true
EXTERNAL_SERVICE_TIMEOUT=5.0
```

## Production Best Practices

1. **Security**: Consider adding authentication to the metrics endpoint in production environments.
2. **Resource Usage**: Monitor the resource usage of the metrics collection itself, as it can add overhead.
3. **Retention**: Configure appropriate retention periods for metrics data to manage storage requirements.
4. **Aggregation**: Use Prometheus recording rules to pre-compute expensive queries.
5. **Dashboards**: Create additional dashboards for different stakeholders (developers, operations, business).

## Troubleshooting

### Common Issues

1. **High memory usage**: Check if metrics collection is storing too many unique label combinations. Consider reducing the cardinality of labels.
2. **Missing metrics**: Ensure the metrics endpoint is accessible and that Prometheus is correctly configured to scrape it.
3. **Alert spam**: Adjust thresholds or add alert throttling to reduce excessive notifications.

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)
- [FastAPI Monitoring Best Practices](https://fastapi.tiangolo.com/advanced/monitoring/) 