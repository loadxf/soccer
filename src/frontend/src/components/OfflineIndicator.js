import React, { useState, useEffect } from 'react';
import { Snackbar, Alert, Box, Typography } from '@mui/material';
import WifiOffIcon from '@mui/icons-material/WifiOff';
import SignalWifiStatusbar4BarIcon from '@mui/icons-material/SignalWifiStatusbar4Bar';

const OfflineIndicator = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showReconnectedMessage, setShowReconnectedMessage] = useState(false);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShowReconnectedMessage(true);
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  const handleCloseReconnectedMessage = () => {
    setShowReconnectedMessage(false);
  };

  return (
    <>
      {/* Offline indicator banner */}
      {!isOnline && (
        <Box
          sx={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            zIndex: 9999,
            bgcolor: '#f44336',
            color: 'white',
            p: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 1
          }}
        >
          <WifiOffIcon fontSize="small" />
          <Typography variant="body2">
            You are currently offline. Some features may be unavailable.
          </Typography>
        </Box>
      )}

      {/* Reconnected notification */}
      <Snackbar
        open={showReconnectedMessage}
        autoHideDuration={3000}
        onClose={handleCloseReconnectedMessage}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={handleCloseReconnectedMessage}
          severity="success"
          sx={{ width: '100%' }}
          icon={<SignalWifiStatusbar4BarIcon />}
        >
          You're back online! All features are now available.
        </Alert>
      </Snackbar>
    </>
  );
};

export default OfflineIndicator; 