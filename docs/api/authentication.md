# Authentication & Authorization

This document explains the authentication and authorization mechanisms used in the Soccer Prediction System API.

## Overview

The API uses a token-based authentication system to secure endpoints and verify user identity. Most endpoints require authentication, with certain admin endpoints requiring additional permissions.

## Authentication Methods

The API supports two authentication methods:

1. **JWT Bearer Tokens**: JSON Web Tokens for user authentication
2. **API Keys**: For application or service authentication

### JWT Authentication

JWT (JSON Web Token) is the primary authentication method for user-based access. Each token contains encoded user information and permissions, signed with a secret key to prevent tampering.

#### Obtaining a JWT Token

To obtain a JWT token, make a POST request to the `/api/v1/auth/token` endpoint with your credentials:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

Successful response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### Using a JWT Token

Once you have a token, include it in the Authorization header of your requests:

```bash
curl "http://127.0.0.1:8000/api/v1/teams" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

#### Token Expiration and Refresh

Access tokens expire after 1 hour by default. When a token expires, you'll receive a 401 Unauthorized response with the error code `TOKEN_EXPIRED`.

To refresh an expired token without re-authenticating, use the refresh token:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."}'
```

### API Key Authentication

API keys provide an alternative authentication method, primarily intended for application-to-application integration.

#### Using an API Key

To authenticate with an API key, include it in the X-API-Key header:

```bash
curl "http://127.0.0.1:8000/api/v1/teams" \
  -H "X-API-Key: your_api_key_here"
```

#### Obtaining an API Key

API keys are issued through the admin interface or by contacting support. Unlike tokens, API keys do not expire automatically but can be revoked at any time.

## Authorization

The API uses role-based access control to determine what authenticated users can do.

### User Roles

| Role | Description | Access Level |
|------|-------------|--------------|
| user | Standard user | Basic read access to most endpoints |
| premium | Premium user | Enhanced rate limits and access to additional features |
| admin | Administrator | Full access to all endpoints, including admin operations |

### Permission Requirements

Different endpoints require different permissions:

- **Public endpoints**: No authentication required (e.g., `/api/v1/health`)
- **User endpoints**: Require valid authentication (e.g., `/api/v1/teams`)
- **Admin endpoints**: Require admin role (e.g., `/api/v1/admin/reset-cache`)

When a user attempts to access an endpoint without the required permissions, they will receive a 403 Forbidden response with the error code `PERMISSION_DENIED`.

## Rate Limiting

The API implements rate limiting based on the authentication level:

| Authentication Level | Rate Limit |
|----------------------|------------|
| No authentication | 20 requests per minute |
| User role | 100 requests per minute |
| Premium role | 200 requests per minute |
| Admin role | 500 requests per minute |

When a rate limit is exceeded, the API returns a 429 Too Many Requests response with the error code `RATE_LIMIT_EXCEEDED` and a Retry-After header indicating how long to wait before making another request.

## Security Best Practices

When working with the API, follow these security best practices:

1. **Keep credentials secure**: Never hardcode or expose credentials in client-side code.
2. **Use HTTPS**: Always use HTTPS when communicating with the API to ensure encrypted communication.
3. **Implement token refresh**: Automatically refresh access tokens when they expire.
4. **Store tokens securely**: Store tokens in secure HTTP-only cookies or secure storage.
5. **Limit token scope**: Request only the permissions your application needs.
6. **Revoke compromised tokens**: If a token is compromised, revoke it immediately.

## Troubleshooting

### Common Authentication Issues

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| 401 Unauthorized | Invalid credentials or token | Verify credentials and token format |
| 401 Unauthorized with TOKEN_EXPIRED | Token has expired | Use refresh token to obtain a new access token |
| 403 Forbidden | Insufficient permissions | Request elevated permissions if needed |
| 429 Too Many Requests | Rate limit exceeded | Implement backoff strategy and retry |

### Logging Out

To manually invalidate a token (logout), you can:

1. Client-side: Discard the token from your storage
2. Server-side: Add the token to a blacklist by calling the logout endpoint

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/auth/logout" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Client Libraries

To simplify authentication, we provide client libraries for common programming languages:

- [Python Client](https://github.com/example/soccer-prediction-python)
- [JavaScript Client](https://github.com/example/soccer-prediction-js)
- [Java Client](https://github.com/example/soccer-prediction-java)

These libraries handle token management, refresh, and retry logic automatically.

## Example: Complete Authentication Flow

```python
import requests

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

def login(username, password):
    response = requests.post(
        f"{API_BASE_URL}/auth/token",
        json={"username": username, "password": password}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Authentication failed: {response.json().get('detail')}")

def refresh_token(refresh_token):
    response = requests.post(
        f"{API_BASE_URL}/auth/refresh",
        json={"refresh_token": refresh_token}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Token refresh failed: {response.json().get('detail')}")

def get_teams(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{API_BASE_URL}/teams", headers=headers)
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401 and response.json().get("error_code") == "TOKEN_EXPIRED":
        # Handle token expiration
        new_tokens = refresh_token(stored_refresh_token)
        # Update stored tokens
        # Retry request with new access token
        return get_teams(new_tokens["access_token"])
    else:
        raise Exception(f"API request failed: {response.json().get('detail')}")
```

## Further Assistance

If you need help with authentication or authorization, please contact our support team at api-support@example.com. 