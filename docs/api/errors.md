# API Error Codes

This document provides a comprehensive list of error codes that may be returned by the Soccer Prediction System API.

## Error Response Format

When an error occurs, the API will return a JSON response with the following structure:

```json
{
  "detail": "Description of the error",
  "error_code": "ERROR_CODE",
  "timestamp": "2023-01-01T12:34:56"
}
```

The `error_code` field contains a unique identifier for the error type, which can be used for programmatic handling of errors in your application.

## HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - The request was invalid or could not be understood |
| 401 | Unauthorized - Authentication is required and has failed or not been provided |
| 403 | Forbidden - The authenticated user does not have permission to access the requested resource |
| 404 | Not Found - The requested resource was not found |
| 422 | Unprocessable Entity - Request data failed validation |
| 429 | Too Many Requests - Rate limit has been exceeded |
| 500 | Internal Server Error - Something went wrong on the server |
| 503 | Service Unavailable - The server is temporarily unavailable |

## Error Codes

### Authentication Errors (401, 403)

| Error Code | HTTP Status | Description | Resolution |
|------------|-------------|-------------|------------|
| AUTHENTICATION_FAILED | 401 | Authentication credentials are invalid | Verify credentials or request a new token |
| TOKEN_EXPIRED | 401 | The access token has expired | Use refresh token to obtain a new access token |
| TOKEN_INVALID | 401 | The token is invalid or malformed | Request a new token with valid credentials |
| PERMISSION_DENIED | 403 | User does not have permission to access this resource | Contact administrator for elevated permissions |
| RATE_LIMIT_EXCEEDED | 429 | API rate limit has been exceeded | Reduce request frequency or upgrade API tier |
| API_KEY_INVALID | 401 | The provided API key is invalid | Verify API key or request a new one |

### Request Errors (400, 404, 422)

| Error Code | HTTP Status | Description | Resolution |
|------------|-------------|-------------|------------|
| RESOURCE_NOT_FOUND | 404 | The requested resource was not found | Verify resource ID or path |
| INVALID_REQUEST | 400 | The request data is invalid or malformed | Check request format and parameters |
| VALIDATION_ERROR | 422 | Request data failed validation checks | Review validation messages and correct data |
| MISSING_PARAMETER | 400 | A required parameter is missing | Add the missing parameter to the request |
| INVALID_PARAMETER | 400 | A parameter has an invalid value | Correct the parameter value |
| UNSUPPORTED_MEDIA_TYPE | 415 | The content type is not supported | Use a supported content type (typically application/json) |
| ENDPOINT_NOT_FOUND | 404 | The requested endpoint does not exist | Check API documentation for correct endpoint URL |

### Server Errors (500, 503)

| Error Code | HTTP Status | Description | Resolution |
|------------|-------------|-------------|------------|
| SERVER_ERROR | 500 | An unexpected server error occurred | Contact support with error details |
| DATABASE_ERROR | 500 | Error connecting to or querying the database | Retry request or contact support |
| REDIS_ERROR | 500 | Error with Redis cache operations | Retry request or contact support |
| SERVICE_UNAVAILABLE | 503 | Service is temporarily unavailable | Retry after a short delay |
| MAINTENANCE_MODE | 503 | System is in maintenance mode | Check status page or retry later |

### Model Errors (500)

| Error Code | HTTP Status | Description | Resolution |
|------------|-------------|-------------|------------|
| MODEL_ERROR | 500 | Error in prediction model processing | Check input data or contact support |
| MODEL_NOT_FOUND | 404 | The requested prediction model was not found | Verify model name or use default model |
| PREDICTION_FAILED | 500 | Unable to generate prediction | Check input data or contact support |
| MODEL_TRAINING | 503 | Model is currently being trained | Retry after training completes |

### Cache Errors (500)

| Error Code | HTTP Status | Description | Resolution |
|------------|-------------|-------------|------------|
| CACHE_ERROR | 500 | Error with cache operations | Retry request or bypass cache |
| CACHE_RESET_FAILED | 500 | Failed to reset cache | Contact administrator |

## Error Handling Best Practices

When working with the Soccer Prediction System API, consider these best practices for error handling:

1. **Always check the HTTP status code** first to determine the general category of the error.
2. **Parse the error response** to extract the `error_code` and `detail` fields.
3. **Implement retry logic** for transient errors (429, 503) with exponential backoff.
4. **Log detailed error information** including the error code, message, and timestamp.
5. **Present user-friendly error messages** to end-users based on the error type.

## Example Error Handling

```python
import requests
import time

def make_api_request(url, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    
    error_data = response.json()
    error_code = error_data.get("error_code", "UNKNOWN_ERROR")
    
    if response.status_code == 429:
        # Rate limiting - implement backoff
        retry_after = int(response.headers.get("Retry-After", 60))
        print(f"Rate limited. Waiting {retry_after} seconds.")
        time.sleep(retry_after)
        return make_api_request(url, token)  # Retry
    
    elif response.status_code == 401 and error_code == "TOKEN_EXPIRED":
        # Token expired - get a new one
        new_token = refresh_token()
        return make_api_request(url, new_token)
    
    elif response.status_code >= 500:
        # Server error - log and retry with backoff
        print(f"Server error: {error_data.get('detail')}")
        time.sleep(5)
        return make_api_request(url, token)
    
    else:
        # Other errors - handle accordingly
        raise Exception(f"API Error: {error_code} - {error_data.get('detail')}")
```

## Contact Support

If you encounter persistent errors or need assistance, please contact our API support team at api-support@example.com with the following information:

- Error code and message
- Timestamp of the error
- API endpoint you were trying to access
- Request parameters (excluding sensitive data)
- Your client ID (if applicable) 