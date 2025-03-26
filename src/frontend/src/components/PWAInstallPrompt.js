import React, { useState, useEffect } from 'react';
import { Button, Snackbar, Alert, Box, Typography } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';

const PWAInstallPrompt = () => {
  const [installPromptEvent, setInstallPromptEvent] = useState(null);
  const [isAppInstalled, setIsAppInstalled] = useState(false);
  const [showInstallPrompt, setShowInstallPrompt] = useState(false);

  useEffect(() => {
    // Check if the app is already installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
      setIsAppInstalled(true);
    }

    // Listen for the beforeinstallprompt event
    const handleBeforeInstallPrompt = (e) => {
      // Prevent the mini-infobar from appearing on mobile
      e.preventDefault();
      // Store the event for later use
      setInstallPromptEvent(e);
      // Show our install prompt after a delay
      setTimeout(() => {
        setShowInstallPrompt(true);
      }, 3000);
    };

    // Listen for the appinstalled event
    const handleAppInstalled = () => {
      setIsAppInstalled(true);
      setShowInstallPrompt(false);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    window.addEventListener('appinstalled', handleAppInstalled);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
      window.removeEventListener('appinstalled', handleAppInstalled);
    };
  }, []);

  const handleInstallClick = () => {
    if (!installPromptEvent) return;

    // Show the install prompt
    installPromptEvent.prompt();

    // Wait for the user to respond to the prompt
    installPromptEvent.userChoice.then((choiceResult) => {
      if (choiceResult.outcome === 'accepted') {
        console.log('User accepted the install prompt');
        setIsAppInstalled(true);
      } else {
        console.log('User dismissed the install prompt');
      }
      // We've used the prompt, and can't use it again, discard it
      setInstallPromptEvent(null);
      setShowInstallPrompt(false);
    });
  };

  const handleClosePrompt = () => {
    setShowInstallPrompt(false);
  };

  if (isAppInstalled) {
    return null; // Don't show anything if the app is already installed
  }

  return (
    <>
      {/* Floating prompt as a Snackbar */}
      <Snackbar
        open={showInstallPrompt}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        autoHideDuration={15000}
        onClose={handleClosePrompt}
      >
        <Alert 
          onClose={handleClosePrompt} 
          severity="info" 
          sx={{ width: '100%' }}
          action={
            <Button 
              color="primary" 
              size="small" 
              onClick={handleInstallClick}
              startIcon={<DownloadIcon />}
            >
              Install
            </Button>
          }
        >
          Install our app for a better experience!
        </Alert>
      </Snackbar>

      {/* Optional button that can be placed somewhere in your app */}
      {installPromptEvent && (
        <Box 
          sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            mt: 2,
            p: 2,
            border: '1px solid #e0e0e0',
            borderRadius: 2,
            bgcolor: 'background.paper'
          }}
        >
          <Typography variant="subtitle1" gutterBottom>
            Install Soccer Prediction System
          </Typography>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Install our app on your device for offline access and a better experience.
          </Typography>
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            onClick={handleInstallClick}
            sx={{ mt: 1 }}
          >
            Install App
          </Button>
        </Box>
      )}
    </>
  );
};

export default PWAInstallPrompt; 