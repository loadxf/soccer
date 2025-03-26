import React, { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box } from '@mui/material';
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';
import Matches from './pages/Matches';
import Teams from './pages/Teams';
import Predictions from './pages/Predictions';
import Visualizations from './pages/Visualizations';
import Demo from './pages/Demo';
import NotFound from './pages/NotFound';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider } from './contexts/AuthContext';
import PWAInstallPrompt from './components/PWAInstallPrompt';
import OfflineIndicator from './components/OfflineIndicator';
import { initializeDB, syncOfflineData } from './utils/offlineStorage';

function App() {
  useEffect(() => {
    // Initialize IndexedDB when the app starts
    initializeDB()
      .then(() => console.log('IndexedDB initialized successfully'))
      .catch(error => console.error('Error initializing IndexedDB:', error));

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
      syncOfflineData()
        .then(success => {
          if (success) {
            console.log('Offline data synced successfully after reconnecting');
          }
        })
        .catch(error => console.error('Error syncing offline data after reconnecting:', error));
    };

    window.addEventListener('online', handleOnline);

    return () => {
      window.removeEventListener('online', handleOnline);
    };
  }, []);

  return (
    <AuthProvider>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        {/* PWA Components */}
        <OfflineIndicator />
        <PWAInstallPrompt />
        
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } />
            <Route path="matches" element={
              <ProtectedRoute>
                <Matches />
              </ProtectedRoute>
            } />
            <Route path="teams" element={
              <ProtectedRoute>
                <Teams />
              </ProtectedRoute>
            } />
            <Route path="predictions" element={
              <ProtectedRoute>
                <Predictions />
              </ProtectedRoute>
            } />
            <Route path="visualizations" element={
              <ProtectedRoute>
                <Visualizations />
              </ProtectedRoute>
            } />
            <Route path="demo" element={
              <ProtectedRoute>
                <Demo />
              </ProtectedRoute>
            } />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </Box>
    </AuthProvider>
  );
}

export default App; 