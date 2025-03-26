# Soccer Prediction System API Documentation

This directory contains the official documentation for the Soccer Prediction System API.

## Overview

The Soccer Prediction System API provides programmatic access to soccer match data, team information, and predictive analytics. It's built using FastAPI and follows REST principles with JSON-based responses.

## API Documentation

- **OpenAPI Specification**: Available at `/api/v1/openapi.json`
- **Swagger UI**: Available at `/api/v1/docs`
- **ReDoc**: Available at `/api/v1/redoc`

## Authentication

The API supports two authentication methods:

1. **JWT Tokens**: Obtain via the `/api/v1/auth/token` endpoint
2. **API Keys**: Configured for application-level access

### Getting a Token

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### Using a Token

```bash
curl "http://127.0.0.1:8000/api/v1/teams" \
  -H "Authorization: Bearer your_jwt_token_here"
```

## Rate Limiting

To ensure fair usage of the API, rate limiting is applied:

- Authenticated users: 100 requests per minute
- Anonymous users: 20 requests per minute

## Endpoints

The API is organized into the following categories:

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/token` | Obtain a JWT token |
| POST | `/api/v1/auth/refresh` | Refresh an expired token |

### Teams

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/teams` | Get a list of teams |
| GET | `/api/v1/teams/{team_id}` | Get details for a specific team |

### Matches

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/matches` | Get a list of matches |
| GET | `/api/v1/matches/{match_id}` | Get details for a specific match |

### Predictions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/predictions/models` | List available prediction models |
| GET | `/api/v1/predictions/match/{match_id}` | Get prediction for a specific match |
| POST | `/api/v1/predictions/custom` | Get prediction for a custom match setup |
| POST | `/api/v1/predictions/batch` | Get predictions for multiple matches (admin only) |
| GET | `/api/v1/predictions/history` | Get prediction history |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/admin/reset-model-cache` | Reset the prediction model cache |
| POST | `/api/v1/admin/reset-cache` | Reset the API response cache |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Check API health status |

## Common Parameters

Many endpoints support these common query parameters:

- `skip`: Number of items to skip (default: 0)
- `limit`: Maximum number of items to return (default: 100)
- `from_date`: Filter results from this date (YYYY-MM-DD)
- `to_date`: Filter results to this date (YYYY-MM-DD)
- `team_id`: Filter by team ID
- `competition_id`: Filter by competition ID

## Response Formats

All API responses are in JSON format. Successful responses have the appropriate data structure, while errors follow this pattern:

```json
{
  "detail": "Error description",
  "error_code": "ERROR_CODE",
  "timestamp": "2023-01-01T12:34:56"
}
```

## Caching

The API implements Redis-based caching to improve performance:

- Team data: cached for 1 hour
- Match data: cached for 5 minutes
- Health checks: cached for 10 seconds

Responses include cache headers:
- `X-Cache: HIT` or `X-Cache: MISS` to indicate cache status
- `Cache-Control: max-age=X` to indicate TTL in seconds

## Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not found |
| 429 | Too many requests |
| 500 | Server error |

## Detailed Documentation

- [Authentication](./authentication.md) - Detailed guide to authentication and authorization
- [Caching](./caching.md) - Information about the caching system
- [Models](./models.md) - Details about available prediction models
- [Error Codes](./errors.md) - Comprehensive list of error codes and meanings

## Changelog

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2023-03-01 | Initial API release |

## Support

For API support, please contact api-support@example.com or visit our [support portal](https://www.example.com/support). 