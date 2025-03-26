import React from 'react';
import { Box, Typography, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { Helmet } from 'react-helmet-async';

const NotFound = () => {
  const navigate = useNavigate();
  
  return (
    <>
      <Helmet>
        <title>404 Not Found | Soccer Prediction System</title>
      </Helmet>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          p: 4,
          mt: 8
        }}
      >
        <Typography 
          variant="h1" 
          component="h1" 
          color="primary"
          sx={{ 
            fontSize: '8rem', 
            fontWeight: 700, 
            mb: 2,
            letterSpacing: -1
          }}
        >
          404
        </Typography>
        
        <Typography 
          variant="h4" 
          component="h2" 
          sx={{ mb: 3, fontWeight: 600 }}
        >
          Page Not Found
        </Typography>
        
        <Typography 
          variant="body1" 
          color="text.secondary"
          sx={{ 
            mb: 4, 
            maxWidth: 500 
          }}
        >
          The page you are looking for might have been removed, had its name changed, or is temporarily unavailable.
        </Typography>
        
        <Button 
          variant="contained" 
          size="large" 
          onClick={() => navigate('/dashboard')}
          sx={{ py: 1.5, px: 4, borderRadius: 2 }}
        >
          Back to Dashboard
        </Button>
      </Box>
    </>
  );
};

export default NotFound; 