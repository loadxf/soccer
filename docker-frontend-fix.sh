#!/bin/bash
# Script to fix frontend API hostname resolution issues

echo "Fixing frontend API hostname resolution issue..."

# Create the updated api.js file
cat > src/frontend/src/services/api.js << 'EOF'
import axios from 'axios';

// Create a function to get API base URL that returns a consistent value even after reload
const getApiBaseUrl = () => {
  // When accessed from a browser, always use relative URLs to leverage Nginx proxying
  if (typeof window !== 'undefined') {
    // If a specific API URL is provided via env and not in Docker, use it (for development)
    if (process.env.REACT_APP_API_URL && process.env.REACT_APP_ENVIRONMENT !== 'docker') {
      return process.env.REACT_APP_API_URL;
    }
    // Otherwise use relative URL so Nginx can handle proxying
    return '';
  }
  
  // This code only runs in non-browser environments (Node.js, SSR)
  return process.env.REACT_APP_API_URL || 'http://localhost:8000';
};

const api = axios.create({
  baseURL: getApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
  // Increased timeout for slow connections
  timeout: 10000,
});

// Add a request interceptor
api.interceptors.request.use(
  (config) => {
    // Check if we're online before making request
    if (!navigator.onLine) {
      // Create a custom error for offline state
      const error = new Error('You are currently offline. Please check your connection.');
      error.isOffline = true;
      return Promise.reject(error);
    }
    
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    // Handle network errors
    if (error.code === 'ECONNREFUSED' || error.message?.includes('Network Error')) {
      console.warn('Network error:', error.message);
      window.dispatchEvent(new CustomEvent('app:offline'));
      return Promise.reject({
        ...error,
        isOffline: true,
        message: 'Unable to connect to API. Check if API service is running.'
      });
    }
    
    // Handle offline errors
    if (error.isOffline || !navigator.onLine) {
      console.warn('API request failed due to network issue:', error.config?.url);
      // Dispatch an offline event that components can listen for
      window.dispatchEvent(new CustomEvent('app:offline'));
      return Promise.reject({
        ...error,
        isOffline: true,
        message: 'You are offline. Please check your network connection.'
      });
    }
    
    const originalRequest = error.config;
    
    // If the error is 401 and hasn't been retried yet
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Try to refresh the token
        const refreshToken = localStorage.getItem('refreshToken');
        if (refreshToken) {
          const response = await axios.post('/api/v1/auth/refresh-token', {}, {
            headers: {
              'Authorization': `Bearer ${refreshToken}`
            }
          });
          
          const { access_token } = response.data;
          localStorage.setItem('token', access_token);
          
          // Update the authorization header
          api.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
          originalRequest.headers['Authorization'] = `Bearer ${access_token}`;
          
          // Retry the original request
          return api(originalRequest);
        }
      } catch (refreshError) {
        console.error('Error refreshing token:', refreshError);
        // If refresh fails, redirect to login
        localStorage.removeItem('token');
        localStorage.removeItem('refreshToken');
        if (window.location.pathname !== '/login') {
          window.location.href = '/login';
        }
      }
    }
    
    return Promise.reject(error);
  }
);

// Add health check function with browser-aware endpoints
api.checkHealth = async () => {
  try {
    // Prioritize relative endpoints when in browser environment
    const endpoints = [
      '/health',                 // Standard endpoint
      '/api/v1/health',          // Legacy API v1 endpoint
      '/api/health'              // Alternative endpoint
    ];
    
    for (const endpoint of endpoints) {
      try {
        const url = endpoint.startsWith('http') 
          ? endpoint 
          : `${api.defaults.baseURL}${endpoint}`;
        
        console.log(`Checking API health at: ${url}`);
        const response = await axios.get(url, { 
          timeout: 5000,
          headers: { 'Accept': 'application/json' },
          withCredentials: false
        });
        
        if (response.status === 200) {
          window.apiConnected = true;
          window.dispatchEvent(new CustomEvent('app:online'));
          return { online: true, endpoint };
        }
      } catch (endpointError) {
        console.warn(`Endpoint ${endpoint} not available:`, endpointError.message);
        // Continue to try next endpoint
      }
    }
    
    // If all endpoints failed
    window.apiConnected = false;
    window.dispatchEvent(new CustomEvent('app:offline'));
    return { online: false };
  } catch (error) {
    window.apiConnected = false;
    window.dispatchEvent(new CustomEvent('app:offline'));
    return { online: false, error: error.message };
  }
};

export default api;
EOF

echo "Rebuilding the frontend container..."
docker compose up -d --build frontend

echo -e "\nFrontend API configuration has been fixed!"
echo "Please refresh your browser and try logging in again."
echo "The app should now be able to properly communicate with the API through the Nginx proxy."

echo -e "\nTo check the frontend logs:"
echo "docker compose logs frontend"
echo -e "\nTo check the API logs:"
echo "docker compose logs app" 