// Frontend configuration

// API configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const ENVIRONMENT = process.env.REACT_APP_ENVIRONMENT || 'development';
const ENABLE_SERVICE_WORKER = process.env.REACT_APP_ENABLE_SERVICE_WORKER === 'true';

// Export API base URL for use in components
export const API_BASE_URL = API_URL;

// App settings
const APP_CONFIG = {
  apiUrl: API_URL,
  environment: ENVIRONMENT,
  enableServiceWorker: ENABLE_SERVICE_WORKER,
  appName: 'Soccer Prediction System',
  version: '1.0.0',
};

export default APP_CONFIG; 