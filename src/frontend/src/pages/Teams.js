import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Button,
  TextField,
  InputAdornment,
  IconButton,
  CircularProgress,
  Alert,
  Chip,
  Tabs,
  Tab,
  useMediaQuery,
  useTheme
} from '@mui/material';
import {
  Search as SearchIcon,
  Clear as ClearIcon,
  SportsSoccer as SportsIcon,
  EmojiEvents as TrophyIcon,
  Group as PlayersIcon
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';
import api from '../services/api';

const Teams = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [teams, setTeams] = useState([]);
  const [search, setSearch] = useState('');
  const [selectedTab, setSelectedTab] = useState(0);
  const [filters, setFilters] = useState({
    league: 'all',
    country: 'all'
  });
  
  // Pagination
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const teamsPerPage = 12;

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));

  const tabOptions = ['All Teams', 'Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'];
  // Shortened tab labels for mobile
  const mobileTabOptions = ['All', 'PL', 'La Liga', 'BL', 'Serie A', 'L1'];

  useEffect(() => {
    const fetchTeams = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Build query parameters
        const params = new URLSearchParams();
        params.append('page', page);
        params.append('per_page', teamsPerPage);
        
        if (search) params.append('search', search);
        
        // Add league filter based on selected tab
        if (selectedTab > 0) {
          params.append('league', tabOptions[selectedTab]);
        }
        
        // Add other filters
        if (filters.country !== 'all') {
          params.append('country', filters.country);
        }
        
        // Fetch teams data
        const response = await api.get(`/api/v1/teams?${params.toString()}`);
        
        // If first page, replace teams, otherwise append
        if (page === 1) {
          setTeams(response.data.items);
        } else {
          setTeams(prev => [...prev, ...response.data.items]);
        }
        
        // Check if more teams are available
        setHasMore(response.data.items.length === teamsPerPage);
      } catch (err) {
        console.error('Error fetching teams:', err);
        setError('Failed to load teams. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchTeams();
  }, [page, search, selectedTab, filters]);

  const handleSearchChange = (event) => {
    setSearch(event.target.value);
    setPage(1); // Reset to first page when search changes
  };

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue);
    setPage(1); // Reset to first page when tab changes
  };

  const handleLoadMore = () => {
    setPage(prev => prev + 1);
  };

  const handleClearSearch = () => {
    setSearch('');
    setPage(1);
  };

  // Default placeholder image for teams without an image
  const defaultTeamImage = 'https://via.placeholder.com/300x200?text=Team+Image';

  return (
    <>
      <Helmet>
        <title>Teams | Soccer Prediction System</title>
      </Helmet>
      <Box sx={{ px: { xs: 0, sm: 1 } }}>
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          mb: { xs: 2, md: 4 },
          px: { xs: 1, sm: 0 }
        }}>
          <Typography 
            variant={isMobile ? "h5" : "h4"} 
            component="h1" 
            sx={{ 
              fontWeight: 600,
              fontSize: { xs: '1.5rem', sm: '2rem', md: '2.125rem' }
            }}
          >
            Teams
          </Typography>
        </Box>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3, mx: { xs: 1, sm: 0 } }}>
            {error}
          </Alert>
        )}
        
        {/* Search and Filters */}
        <Paper sx={{ 
          mb: { xs: 3, md: 4 }, 
          p: { xs: 1.5, md: 2 }, 
          mx: { xs: 1, sm: 0 },
          borderRadius: 3 
        }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Search teams..."
                value={search}
                onChange={handleSearchChange}
                size={isMobile ? "small" : "medium"}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon color="action" />
                    </InputAdornment>
                  ),
                  endAdornment: search ? (
                    <InputAdornment position="end">
                      <IconButton onClick={handleClearSearch} edge="end" size="small">
                        <ClearIcon fontSize="small" />
                      </IconButton>
                    </InputAdornment>
                  ) : null
                }}
              />
            </Grid>
          </Grid>
        </Paper>
        
        {/* Tabs */}
        <Box sx={{ 
          mb: { xs: 2, md: 3 }, 
          borderBottom: 1, 
          borderColor: 'divider',
          mx: { xs: 1, sm: 0 }
        }}>
          <Tabs 
            value={selectedTab} 
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
          >
            {tabOptions.map((option, index) => (
              <Tab 
                key={index} 
                label={isMobile ? mobileTabOptions[index] : option} 
                sx={{ 
                  fontSize: { xs: '0.75rem', sm: '0.875rem' },
                  minWidth: { xs: 60, sm: 90 },
                  p: { xs: 1, sm: 2 }
                }}
              />
            ))}
          </Tabs>
        </Box>
        
        {/* Teams Grid */}
        {loading && teams.length === 0 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '300px' }}>
            <CircularProgress />
          </Box>
        ) : teams.length > 0 ? (
          <>
            <Grid container spacing={isMobile ? 2 : 3} sx={{ px: { xs: 1, sm: 0 } }}>
              {teams.map((team) => (
                <Grid item xs={6} sm={6} md={4} lg={3} key={team.id}>
                  <Card 
                    sx={{ 
                      height: '100%', 
                      display: 'flex', 
                      flexDirection: 'column',
                      borderRadius: 3,
                      transition: 'transform 0.2s',
                      '&:hover': {
                        transform: 'translateY(-4px)',
                        boxShadow: '0 8px 20px rgba(0,0,0,0.1)'
                      }
                    }}
                  >
                    <CardMedia
                      component="img"
                      height={isMobile ? "120" : "140"}
                      image={team.logo || defaultTeamImage}
                      alt={team.name}
                      sx={{ objectFit: 'contain', p: 2, bgcolor: '#f5f5f5' }}
                    />
                    <CardContent sx={{ 
                      flexGrow: 1,
                      p: { xs: 1.5, md: 2 }
                    }}>
                      <Typography 
                        variant={isMobile ? "subtitle1" : "h6"} 
                        component="div" 
                        gutterBottom 
                        sx={{ 
                          fontWeight: 600,
                          fontSize: { xs: '1rem', md: '1.25rem' },
                          lineHeight: 1.2,
                          mb: 1
                        }}
                      >
                        {team.name}
                      </Typography>
                      
                      <Box sx={{ 
                        display: 'flex',
                        flexDirection: isMobile ? 'column' : 'row',
                        alignItems: isMobile ? 'flex-start' : 'center',
                        mb: 1,
                        gap: 0.5
                      }}>
                        <Chip 
                          label={team.country} 
                          size="small" 
                          sx={{ 
                            mr: isMobile ? 0 : 1,
                            mb: isMobile ? 0.5 : 0,
                            height: { xs: 22, md: 24 },
                            fontSize: { xs: '0.7rem', md: '0.75rem' }
                          }}
                        />
                        <Chip 
                          label={team.league} 
                          size="small" 
                          color="primary"
                          variant="outlined"
                          sx={{ 
                            height: { xs: 22, md: 24 },
                            fontSize: { xs: '0.7rem', md: '0.75rem' }
                          }}
                        />
                      </Box>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: isMobile ? 1 : 2 }}>
                        <TrophyIcon sx={{ 
                          fontSize: isMobile ? 16 : 18, 
                          mr: 1, 
                          color: 'warning.main' 
                        }} />
                        <Typography 
                          variant="body2" 
                          color="text.secondary"
                          sx={{
                            fontSize: { xs: '0.75rem', md: '0.875rem' }
                          }}
                        >
                          {team.trophies} Trophies
                        </Typography>
                      </Box>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                        <PlayersIcon sx={{ 
                          fontSize: isMobile ? 16 : 18, 
                          mr: 1, 
                          color: 'info.main' 
                        }} />
                        <Typography 
                          variant="body2" 
                          color="text.secondary"
                          sx={{
                            fontSize: { xs: '0.75rem', md: '0.875rem' }
                          }}
                        >
                          {team.playerCount} Players
                        </Typography>
                      </Box>
                      
                      <Button 
                        variant="contained" 
                        fullWidth 
                        size={isMobile ? "small" : "medium"}
                        sx={{ 
                          mt: { xs: 1.5, md: 2 }, 
                          textTransform: 'none',
                          fontSize: { xs: '0.75rem', md: '0.875rem' },
                          py: { xs: 0.5, md: 1 }
                        }}
                      >
                        View Details
                      </Button>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
            {hasMore && (
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center',
                my: { xs: 3, md: 4 },
                mx: { xs: 1, sm: 0 }
              }}>
                <Button
                  variant="outlined"
                  onClick={handleLoadMore}
                  disabled={loading}
                  startIcon={loading && <CircularProgress size={20} />}
                  size={isMobile ? "small" : "medium"}
                >
                  {loading ? 'Loading...' : 'Load More Teams'}
                </Button>
              </Box>
            )}
          </>
        ) : (
          <Box sx={{ 
            textAlign: 'center', 
            py: 4,
            mx: { xs: 1, sm: 0 }
          }}>
            <SportsIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No Teams Found
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Try adjusting your search or filters.
            </Typography>
          </Box>
        )}
      </Box>
    </>
  );
};

export default Teams; 