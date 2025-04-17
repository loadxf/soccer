import React, { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, 
  Box, 
  Toolbar, 
  IconButton, 
  Typography, 
  Drawer, 
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Menu,
  MenuItem,
  Tooltip,
  useMediaQuery,
  useTheme,
  Chip
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  SportsSoccer as SportsIcon,
  Groups as TeamsIcon,
  ShowChart as PredictionsIcon,
  Equalizer as VisualizationsIcon,
  AccountCircle,
  ChevronLeft,
  CloudOff,
  Cloud
} from '@mui/icons-material';
import useAuth from '../hooks/useAuth';
import SportsIcon from '@mui/icons-material/Sports';

const Layout = ({ apiStatus }) => {
  const { user, logout, isAdmin } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [drawerOpen, setDrawerOpen] = useState(!isMobile);
  const [anchorEl, setAnchorEl] = useState(null);
  
  // Close drawer by default on mobile when component mounts or screen size changes
  useEffect(() => {
    setDrawerOpen(!isMobile);
  }, [isMobile]);

  const drawerWidth = 240;

  const handleDrawerToggle = () => {
    setDrawerOpen(!drawerOpen);
  };

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    handleProfileMenuClose();
    logout();
    navigate('/login');
  };
  
  const handleNavigation = (path) => {
    navigate(path);
    // Close drawer automatically on mobile after navigation
    if (isMobile) {
      setDrawerOpen(false);
    }
  };

  const menuItems = [
    { 
      text: 'Dashboard', 
      icon: <DashboardIcon />, 
      path: '/dashboard',
      active: location.pathname === '/dashboard'
    },
    { 
      text: 'Matches', 
      icon: <SportsIcon />, 
      path: '/matches',
      active: location.pathname === '/matches'
    },
    { 
      text: 'Teams', 
      icon: <TeamsIcon />, 
      path: '/teams',
      active: location.pathname === '/teams'
    },
    { 
      text: 'Predictions', 
      icon: <PredictionsIcon />, 
      path: '/predictions',
      active: location.pathname === '/predictions'
    },
    { 
      text: 'Visualizations', 
      icon: <VisualizationsIcon />, 
      path: '/visualizations',
      active: location.pathname === '/visualizations'
    },
    { 
      text: 'Interactive Demo', 
      icon: <SportsIcon />, 
      path: '/demo',
      active: location.pathname === '/demo'
    }
  ];

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar 
        position="fixed" 
        sx={{ 
          zIndex: (theme) => theme.zIndex.drawer + 1,
          boxShadow: 1
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="toggle drawer"
            onClick={handleDrawerToggle}
            edge="start"
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography 
            variant="h6" 
            noWrap 
            component="div" 
            sx={{ 
              flexGrow: 1,
              fontSize: { xs: '1rem', sm: '1.25rem' }
            }}
          >
            {isMobile ? 'Soccer Predictions' : 'Soccer Prediction System'}
          </Typography>
          
          {/* API Status Indicator */}
          {apiStatus && (
            <Tooltip title={apiStatus.online ? "API Connected" : "API Disconnected"}>
              <Chip
                icon={apiStatus.online ? <Cloud fontSize="small" /> : <CloudOff fontSize="small" />}
                label={apiStatus.online ? "API Online" : "API Offline"}
                color={apiStatus.online ? "success" : "error"}
                size="small"
                sx={{ mr: 2 }}
              />
            </Tooltip>
          )}
          
          {user && (
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Tooltip title="Account settings">
                <IconButton
                  onClick={handleProfileMenuOpen}
                  size="large"
                  edge="end"
                  aria-label="account of current user"
                  aria-haspopup="true"
                  color="inherit"
                >
                  <AccountCircle />
                </IconButton>
              </Tooltip>
            </Box>
          )}
        </Toolbar>
      </AppBar>
      
      <Drawer
        variant={isMobile ? "temporary" : "persistent"}
        open={drawerOpen}
        onClose={isMobile ? handleDrawerToggle : undefined}
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
            background: (theme) => theme.palette.background.default,
            borderRight: '1px solid rgba(0, 0, 0, 0.08)'
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', display: 'flex', flexDirection: 'column', height: '100%' }}>
          <List sx={{ px: 1, pt: 2 }}>
            {menuItems.map((item) => (
              <ListItem 
                button 
                key={item.text}
                onClick={() => handleNavigation(item.path)}
                sx={{ 
                  mb: 1, 
                  borderRadius: 2,
                  bgcolor: item.active ? 'rgba(0, 0, 0, 0.04)' : 'transparent',
                  color: item.active ? 'primary.main' : 'text.primary',
                  '&:hover': {
                    bgcolor: 'rgba(0, 0, 0, 0.08)'
                  }
                }}
              >
                <ListItemIcon sx={{ color: item.active ? 'primary.main' : 'text.secondary' }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text} 
                  primaryTypographyProps={{ 
                    fontWeight: item.active ? 500 : 400 
                  }} 
                />
              </ListItem>
            ))}
          </List>
          
          {/* Display API status in drawer footer when offline */}
          {apiStatus && !apiStatus.online && (
            <Box sx={{ p: 2, bgcolor: 'error.light', color: 'error.contrastText', textAlign: 'center', mt: 'auto', mb: 2, mx: 2, borderRadius: 1 }}>
              <CloudOff fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
              <Typography variant="body2" component="span">
                API Offline - Limited functionality
              </Typography>
            </Box>
          )}
          
          <Box sx={{ mt: 'auto', p: 2 }}>
            {user && (
              <Box sx={{ display: 'flex', alignItems: 'center', p: 1 }}>
                <Avatar 
                  alt={user.username || 'User'} 
                  sx={{ width: 40, height: 40, mr: 2, bgcolor: 'primary.main' }}
                >
                  {user.username?.[0]?.toUpperCase() || 'U'}
                </Avatar>
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>
                    {user.username}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {isAdmin() ? 'Administrator' : 'User'}
                  </Typography>
                </Box>
              </Box>
            )}
          </Box>
        </Box>
      </Drawer>
      
      <Menu
        anchorEl={anchorEl}
        id="account-menu"
        open={Boolean(anchorEl)}
        onClose={handleProfileMenuClose}
        PaperProps={{
          elevation: 0,
          sx: {
            overflow: 'visible',
            filter: 'drop-shadow(0px 2px 8px rgba(0,0,0,0.1))',
            mt: 1.5,
            borderRadius: 2,
            minWidth: 180,
            '& .MuiMenuItem-root': {
              px: 2,
              py: 1,
            }
          },
        }}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <MenuItem onClick={handleProfileMenuClose}>
          Profile
        </MenuItem>
        <MenuItem onClick={handleProfileMenuClose}>
          Settings
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout}>
          Logout
        </MenuItem>
      </Menu>
      
      <Box
        component="main"
        sx={{ 
          flexGrow: 1, 
          p: 3, 
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          pt: { xs: 8, sm: 9 } 
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
};

export default Layout; 