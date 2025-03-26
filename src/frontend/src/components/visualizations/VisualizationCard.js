import React, { useState } from 'react';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  IconButton, 
  Menu, 
  MenuItem, 
  Box, 
  CircularProgress, 
  Typography,
  CardActions
} from '@mui/material';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import DownloadIcon from '@mui/icons-material/Download';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import RefreshIcon from '@mui/icons-material/Refresh';

/**
 * Reusable card component for displaying visualizations.
 * 
 * @param {object} props - Component props
 * @param {string} props.title - Card title
 * @param {string} props.imageData - Base64 encoded image data
 * @param {boolean} props.loading - Loading state
 * @param {string} props.error - Error message
 * @param {function} props.onRefresh - Refresh callback
 * @param {function} props.onDownload - Download callback
 * @param {function} props.onFullscreen - Fullscreen callback
 * @param {React.ReactNode} props.children - Additional content
 * @returns {React.ReactElement} Visualization card component
 */
const VisualizationCard = ({
  title,
  imageData,
  loading = false,
  error = null,
  onRefresh = null,
  onDownload = null,
  onFullscreen = null,
  children
}) => {
  const [menuAnchorEl, setMenuAnchorEl] = useState(null);
  const menuOpen = Boolean(menuAnchorEl);

  const handleMenuClick = (event) => {
    setMenuAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };

  const handleDownload = () => {
    if (onDownload) {
      onDownload();
    } else if (imageData) {
      // Default download behavior if no callback is provided
      const link = document.createElement('a');
      link.href = `data:image/png;base64,${imageData}`;
      link.download = `${title.replace(/\s+/g, '-').toLowerCase()}-${new Date().toISOString().split('T')[0]}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
    handleMenuClose();
  };

  const handleRefresh = () => {
    if (onRefresh) {
      onRefresh();
    }
    handleMenuClose();
  };

  const handleFullscreen = () => {
    if (onFullscreen) {
      onFullscreen();
    }
    handleMenuClose();
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardHeader
        title={title}
        action={
          <IconButton 
            aria-label="settings"
            onClick={handleMenuClick}
          >
            <MoreVertIcon />
          </IconButton>
        }
      />
      <Menu
        anchorEl={menuAnchorEl}
        open={menuOpen}
        onClose={handleMenuClose}
      >
        {onRefresh && (
          <MenuItem onClick={handleRefresh}>
            <RefreshIcon fontSize="small" sx={{ mr: 1 }} />
            Refresh
          </MenuItem>
        )}
        {imageData && (
          <MenuItem onClick={handleDownload}>
            <DownloadIcon fontSize="small" sx={{ mr: 1 }} />
            Download
          </MenuItem>
        )}
        {onFullscreen && (
          <MenuItem onClick={handleFullscreen}>
            <FullscreenIcon fontSize="small" sx={{ mr: 1 }} />
            Fullscreen
          </MenuItem>
        )}
      </Menu>
      
      <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
        {loading ? (
          <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height="100%">
            <CircularProgress />
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Loading visualization...
            </Typography>
          </Box>
        ) : error ? (
          <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height="100%" p={2}>
            <Typography variant="body1" color="error" align="center">
              {error}
            </Typography>
            {onRefresh && (
              <IconButton color="primary" onClick={onRefresh} sx={{ mt: 2 }}>
                <RefreshIcon /> Try Again
              </IconButton>
            )}
          </Box>
        ) : imageData ? (
          <Box 
            component="img"
            sx={{
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              transition: 'transform 0.3s ease-in-out',
              '&:hover': {
                transform: 'scale(1.02)'
              }
            }}
            src={`data:image/png;base64,${imageData}`}
            alt={title}
          />
        ) : (
          <Box display="flex" alignItems="center" justifyContent="center" height="100%">
            <Typography variant="body2" color="text.secondary">
              No visualization data available
            </Typography>
          </Box>
        )}
        
        {children}
      </CardContent>
      
      {(onRefresh || onDownload || onFullscreen) && (
        <CardActions sx={{ justifyContent: 'flex-end', p: 1 }}>
          {onRefresh && (
            <IconButton size="small" onClick={onRefresh} title="Refresh">
              <RefreshIcon fontSize="small" />
            </IconButton>
          )}
          {imageData && (
            <IconButton size="small" onClick={handleDownload} title="Download">
              <DownloadIcon fontSize="small" />
            </IconButton>
          )}
          {onFullscreen && (
            <IconButton size="small" onClick={onFullscreen} title="Fullscreen">
              <FullscreenIcon fontSize="small" />
            </IconButton>
          )}
        </CardActions>
      )}
    </Card>
  );
};

export default VisualizationCard; 