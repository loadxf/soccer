import axios from 'axios';

// Create a function to get API base URL that returns a consistent value even after reload
const getApiBaseUrl = () => {
  // First check for environment variable
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  
  // Check if we're in Docker
  const isDocker = process.env.REACT_APP_ENVIRONMENT === 'docker';
  if (isDocker) {
    // In Docker, the frontend accesses API via browser, not directly container-to-container
    // Use window.location.hostname with port 8000 for API access
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    return `${protocol}//${hostname}:8000`;
  }
  
  // Then check if we're in a remote environment by checking hostname
  const hostname = window.location.hostname;
  if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
    // If we're not on localhost, assume API is same origin but on port 8000
    const protocol = window.location.protocol;
    return `${protocol}//${hostname}:8000`;
  }
  
  // Default to localhost:8000 for local development
  return 'http://localhost:8000';
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
    // Handle Docker-specific network errors
    if (process.env.REACT_APP_ENVIRONMENT === 'docker' && 
        (error.code === 'ECONNREFUSED' || error.message?.includes('app:8000'))) {
      console.warn('Docker container network error:', error.message);
      window.dispatchEvent(new CustomEvent('app:offline'));
      return Promise.reject({
        ...error,
        isOffline: true,
        message: 'Unable to connect to API container. Check if API service is running.'
      });
    }
    
    // Handle offline errors
    if (error.isOffline || !navigator.onLine || error.message === 'Network Error') {
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
          const response = await axios.get('/api/v1/auth/refresh-token', {
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

// Add health check function with Docker awareness
api.checkHealth = async () => {
  try {
    // Try multiple health endpoints
    const isDocker = process.env.REACT_APP_ENVIRONMENT === 'docker';
    
    // In Docker, try the container service name endpoints first
    const endpoints = isDocker 
      ? [
          'http://app:8000/health',  // Direct Docker service endpoint
          '/health',                 // Relative endpoint
          '/api/v1/health',          // Legacy API v1 endpoint
          '/api/health'              // Alternative endpoint
        ] 
      : [
          '/health',                 // Standard endpoint
          '/api/v1/health',          // Legacy API v1 endpoint
          '/api/health'              // Alternative endpoint
        ];
    
    for (const endpoint of endpoints) {
      try {
        // Use full URL if it starts with http, otherwise use relative endpoint
        const url = endpoint.startsWith('http') 
          ? endpoint 
          : `${api.defaults.baseURL}${endpoint}`;
        
        console.log(`Checking API health at: ${url}`);
        const response = await axios.get(url, { 
          timeout: 5000,
          // Handle cross-origin requests in Docker
          ...(isDocker && { 
            headers: { 'Accept': 'application/json' },
            withCredentials: false 
          })
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