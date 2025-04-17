import React, { useEffect, useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, Snackbar, Alert } from '@mui/material';
import Dashboard from './pages/Dashboard';
// import Login from './pages/Login'; // Removed login import
import Matches from './pages/Matches';
import Teams from './pages/Teams';
import Predictions from './pages/Predictions';
import Visualizations from './pages/Visualizations';
import Demo from './pages/Demo';
import NotFound from './pages/NotFound';
import Layout from './components/Layout';
// import ProtectedRoute from './components/ProtectedRoute'; // Removed ProtectedRoute import
import { AuthProvider } from './contexts/AuthContext';
import PWAInstallPrompt from './components/PWAInstallPrompt';
import OfflineIndicator from './components/OfflineIndicator';
import { initializeDB, syncOfflineData } from './utils/offlineStorage';
import api from './services/api';

function App() {
  const [apiStatus, setApiStatus] = useState({ checked: false, online: navigator.onLine });
  const [showApiError, setShowApiError] = useState(false);

  useEffect(() => {
    // Initialize IndexedDB when the app starts
    initializeDB()
      .then(() => console.log('IndexedDB initialized successfully'))
      .catch(error => console.error('Error initializing IndexedDB:', error));

    // Check API health on startup
    checkApiHealth();

    // Attempt to sync offline data when the app loads and we're online
    if (navigator.onLine) {
      syncOfflineData()
        .then(success => {
          if (success) {
            console.log('Offline data synced successfully');
          }
        })
        .catch(error => console.error('Error syncing offline data:', error));
    }

    // Register event listener to sync data when the app comes online
    const handleOnline = () => {
      // Recheck API health when we come back online
      checkApiHealth();
      
      syncOfflineData()
        .then(success => {
          if (success) {
            console.log('Offline data synced successfully after reconnecting');
          }
        })
        .catch(error => console.error('Error syncing offline data after reconnecting:', error));
    };

    // Register event listener for app-specific online/offline events
    const handleAppOffline = () => {
      setApiStatus(prev => ({ ...prev, online: false }));
      setShowApiError(true);
    };

    const handleAppOnline = () => {
      setApiStatus(prev => ({ ...prev, online: true }));
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('app:offline', handleAppOffline);
    window.addEventListener('app:online', handleAppOnline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('app:offline', handleAppOffline);
      window.removeEventListener('app:online', handleAppOnline);
    };
  }, []);

  // Function to check API health
  const checkApiHealth = async () => {
    try {
      const result = await api.checkHealth();
      setApiStatus({ checked: true, online: result.online });
      
      // If API is not available, show error message
      if (!result.online) {
        setShowApiError(true);
      }
    } catch (error) {
      console.error('Error checking API health:', error);
      setApiStatus({ checked: true, online: false });
      setShowApiError(true);
    }
  };

  // Handle closing the API error notification
  const handleCloseApiError = () => {
    setShowApiError(false);
  };

  return (
    <AuthProvider>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        {/* PWA Components */}
        <OfflineIndicator />
        <PWAInstallPrompt />
        
        {/* API Status Error Notification */}
        <Snackbar 
          open={showApiError} 
          autoHideDuration={6000} 
          onClose={handleCloseApiError}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert 
            onClose={handleCloseApiError} 
            severity="error" 
            sx={{ width: '100%' }}
          >
            Unable to connect to the API. Some features may be unavailable. Please check your network connection or contact support.
          </Alert>
        </Snackbar>
        
        <Routes>
          {/* <Route path="/login" element={<Login />} /> */ /* Removed login route */}
          <Route path="/" element={<Layout apiStatus={apiStatus} />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={
              /* <ProtectedRoute> */
                <Dashboard />
              /* </ProtectedRoute> */
            } />
            <Route path="matches" element={
              /* <ProtectedRoute> */
                <Matches />
              /* </ProtectedRoute> */
            } />
            <Route path="teams" element={
              /* <ProtectedRoute> */
                <Teams />
              /* </ProtectedRoute> */
            } />
            <Route path="predictions" element={
              /* <ProtectedRoute> */
                <Predictions />
              /* </ProtectedRoute> */
            } />
            <Route path="visualizations" element={
              /* <ProtectedRoute> */
                <Visualizations />
              /* </ProtectedRoute> */
            } />
            <Route path="demo" element={
              /* <ProtectedRoute> */
                <Demo />
              /* </ProtectedRoute> */
            } />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </Box>
    </AuthProvider>
  );
}

export default App; 