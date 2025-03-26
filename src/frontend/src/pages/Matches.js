import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  TextField,
  InputAdornment,
  IconButton,
  Chip,
  Button,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Grid,
  useMediaQuery,
  useTheme,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Card,
  CardContent
} from '@mui/material';
import { DataGrid, GridToolbar } from '@mui/x-data-grid';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  SportsSoccer as MatchIcon,
  Schedule as ScheduleIcon,
  Clear as ClearIcon
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';
import { format, parseISO } from 'date-fns';
import api from '../services/api';

const Matches = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [matches, setMatches] = useState([]);
  const [search, setSearch] = useState('');
  const [filters, setFilters] = useState({
    league: 'all',
    season: 'all',
    status: 'all'
  });
  const [filterOptions, setFilterOptions] = useState({
    leagues: [],
    seasons: [],
    statuses: ['completed', 'scheduled', 'postponed', 'canceled']
  });
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [total, setTotal] = useState(0);
  
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));

  useEffect(() => {
    const fetchMatches = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Build query parameters
        const params = new URLSearchParams();
        params.append('page', page + 1);  // API uses 1-indexed pages
        params.append('per_page', pageSize);
        
        if (search) params.append('search', search);
        if (filters.league !== 'all') params.append('league', filters.league);
        if (filters.season !== 'all') params.append('season', filters.season);
        if (filters.status !== 'all') params.append('status', filters.status);
        
        // Fetch matches data
        const response = await api.get(`/api/v1/matches?${params.toString()}`);
        
        setMatches(response.data.items);
        setTotal(response.data.total);
        
        // Set filter options if not already set
        if (filterOptions.leagues.length === 0 && response.data.filters) {
          setFilterOptions(prev => ({
            ...prev,
            leagues: response.data.filters.leagues || [],
            seasons: response.data.filters.seasons || []
          }));
        }
      } catch (err) {
        console.error('Error fetching matches:', err);
        setError('Failed to load matches. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchMatches();
  }, [page, pageSize, search, filters]);

  const handleSearchChange = (event) => {
    setSearch(event.target.value);
    setPage(0);  // Reset to first page when search changes
  };

  const handleFilterChange = (name, value) => {
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
    setPage(0);  // Reset to first page when filters change
  };

  const handleClearFilters = () => {
    setSearch('');
    setFilters({
      league: 'all',
      season: 'all',
      status: 'all'
    });
    setPage(0);
  };

  const columns = [
    {
      field: 'date',
      headerName: 'Date',
      flex: 1,
      minWidth: 120,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <ScheduleIcon sx={{ fontSize: 18, mr: 1, color: 'text.secondary' }} />
          {format(parseISO(params.value), 'MMM d, yyyy')}
        </Box>
      )
    },
    {
      field: 'competition',
      headerName: 'Competition',
      flex: 1,
      minWidth: 150,
      hideable: true,
    },
    {
      field: 'homeTeam',
      headerName: 'Home Team',
      flex: 1.5,
      minWidth: 180
    },
    {
      field: 'awayTeam',
      headerName: 'Away Team',
      flex: 1.5,
      minWidth: 180
    },
    {
      field: 'score',
      headerName: 'Score',
      width: 120,
      align: 'center',
      renderCell: (params) => {
        const match = params.row;
        return match.status === 'completed' ? (
          <Chip 
            label={`${match.homeScore} - ${match.awayScore}`} 
            color="primary" 
            variant="outlined"
            size="small"
            sx={{ fontWeight: 600 }}
          />
        ) : (
          <Chip 
            label={match.status === 'scheduled' ? 'Upcoming' : match.status} 
            color={match.status === 'scheduled' ? 'info' : 'default'}
            size="small"
            sx={{ textTransform: 'capitalize' }}
          />
        );
      }
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 120,
      sortable: false,
      renderCell: (params) => (
        <Button
          variant="contained"
          size="small"
          sx={{ textTransform: 'none' }}
          onClick={() => {
            // Handle match details or prediction navigation
            // TODO: Implement navigation to match details
          }}
        >
          {params.row.status === 'scheduled' ? 'Predict' : 'Details'}
        </Button>
      )
    }
  ];

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setPageSize(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Mobile view match card component
  const MatchCard = ({ match }) => (
    <Card sx={{ 
      mb: 2, 
      borderRadius: 2,
      boxShadow: '0 2px 8px rgba(0,0,0,0.08)'
    }}>
      <CardContent sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <ScheduleIcon sx={{ fontSize: 16, mr: 0.5, color: 'text.secondary' }} />
            <Typography variant="caption" color="text.secondary">
              {format(parseISO(match.date), 'MMM d, yyyy')}
            </Typography>
          </Box>
          <Chip 
            label={match.competition} 
            size="small"
            variant="outlined"
            sx={{ 
              height: 20, 
              fontSize: '0.625rem',
              '& .MuiChip-label': {
                px: 1
              }
            }}
          />
        </Box>
        
        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          mb: 1.5
        }}>
          <Typography variant="body2" sx={{ fontWeight: 500, maxWidth: '40%', textAlign: 'left' }}>
            {match.homeTeam}
          </Typography>
          {match.status === 'completed' ? (
            <Chip 
              label={`${match.homeScore} - ${match.awayScore}`} 
              color="primary" 
              variant="outlined"
              size="small"
              sx={{ fontWeight: 600, fontSize: '0.75rem' }}
            />
          ) : (
            <Chip 
              label={match.status === 'scheduled' ? 'Upcoming' : match.status} 
              color={match.status === 'scheduled' ? 'info' : 'default'}
              size="small"
              sx={{ 
                textTransform: 'capitalize',
                fontSize: '0.75rem',
                height: 24
              }}
            />
          )}
          <Typography variant="body2" sx={{ fontWeight: 500, maxWidth: '40%', textAlign: 'right' }}>
            {match.awayTeam}
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            size="small"
            sx={{ 
              textTransform: 'none',
              fontSize: '0.75rem',
              py: 0.5,
              px: 2
            }}
          >
            {match.status === 'scheduled' ? 'Predict' : 'Details'}
          </Button>
        </Box>
      </CardContent>
    </Card>
  );

  if (loading && matches.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <>
      <Helmet>
        <title>Matches | Soccer Prediction System</title>
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
            Matches
          </Typography>
          {!isMobile && (
            <Box>
              <Button 
                variant="contained" 
                sx={{ 
                  textTransform: 'none', 
                  backgroundColor: 'secondary.main',
                  '&:hover': {
                    backgroundColor: 'secondary.dark',
                  }
                }}
              >
                Export Data
              </Button>
            </Box>
          )}
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3, mx: { xs: 1, sm: 0 } }}>
            {error}
          </Alert>
        )}

        {/* Filters */}
        <Paper sx={{ 
          mb: { xs: 2, md: 3 }, 
          p: { xs: 1.5, md: 3 }, 
          borderRadius: 3,
          mx: { xs: 1, sm: 0 }
        }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                variant="outlined"
                placeholder="Search teams or competitions..."
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
                      <IconButton onClick={() => setSearch('')} edge="end" size="small">
                        <ClearIcon fontSize="small" />
                      </IconButton>
                    </InputAdornment>
                  ) : null
                }}
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={2.5}>
              <FormControl fullWidth size={isMobile ? "small" : "medium"}>
                <InputLabel id="league-filter-label">League</InputLabel>
                <Select
                  labelId="league-filter-label"
                  value={filters.league}
                  onChange={(e) => handleFilterChange('league', e.target.value)}
                  label="League"
                >
                  <MenuItem value="all">All Leagues</MenuItem>
                  {filterOptions.leagues.map((league) => (
                    <MenuItem key={league} value={league}>
                      {league}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={2.5}>
              <FormControl fullWidth size={isMobile ? "small" : "medium"}>
                <InputLabel id="season-filter-label">Season</InputLabel>
                <Select
                  labelId="season-filter-label"
                  value={filters.season}
                  onChange={(e) => handleFilterChange('season', e.target.value)}
                  label="Season"
                >
                  <MenuItem value="all">All Seasons</MenuItem>
                  {filterOptions.seasons.map((season) => (
                    <MenuItem key={season} value={season}>
                      {season}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size={isMobile ? "small" : "medium"}>
                <InputLabel id="status-filter-label">Status</InputLabel>
                <Select
                  labelId="status-filter-label"
                  value={filters.status}
                  onChange={(e) => handleFilterChange('status', e.target.value)}
                  label="Status"
                >
                  <MenuItem value="all">All Statuses</MenuItem>
                  {filterOptions.statuses.map((status) => (
                    <MenuItem key={status} value={status} sx={{ textTransform: 'capitalize' }}>
                      {status}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={1}>
              <Button
                fullWidth
                variant="outlined"
                onClick={handleClearFilters}
                startIcon={<ClearIcon />}
                sx={{ height: isMobile ? '36.5px' : '40px' }}
                size={isMobile ? "small" : "medium"}
              >
                Clear
              </Button>
            </Grid>
          </Grid>
        </Paper>

        {/* Matches Data */}
        {isMobile ? (
          <Box sx={{ mx: 1 }}>
            {matches.length > 0 ? (
              <>
                {matches.map((match) => (
                  <MatchCard key={match.id} match={match} />
                ))}
                <TablePagination
                  component="div"
                  count={total}
                  page={page}
                  onPageChange={handleChangePage}
                  rowsPerPage={pageSize}
                  onRowsPerPageChange={handleChangeRowsPerPage}
                  rowsPerPageOptions={[10, 25, 50]}
                  sx={{ 
                    borderRadius: 2,
                    mt: 2,
                    '.MuiTablePagination-selectLabel, .MuiTablePagination-displayedRows': {
                      fontSize: '0.75rem'
                    },
                    '.MuiTablePagination-select': {
                      fontSize: '0.75rem'
                    }
                  }}
                />
              </>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <MatchIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  No Matches Found
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Try adjusting your search or filters.
                </Typography>
              </Box>
            )}
          </Box>
        ) : (
          <Paper 
            sx={{ 
              height: 650, 
              width: '100%', 
              borderRadius: 3,
              overflow: 'hidden',
              mb: 4
            }}
          >
            <DataGrid
              rows={matches}
              columns={columns}
              pageSize={pageSize}
              rowsPerPageOptions={[10, 25, 50, 100]}
              pagination
              paginationMode="server"
              rowCount={total}
              page={page}
              onPageChange={(newPage) => setPage(newPage)}
              onPageSizeChange={(newPageSize) => {
                setPageSize(newPageSize);
                setPage(0);
              }}
              loading={loading}
              disableSelectionOnClick
              density="standard"
              components={{
                Toolbar: GridToolbar,
              }}
              componentsProps={{
                toolbar: {
                  showQuickFilter: true,
                  quickFilterProps: { debounceMs: 500 },
                },
              }}
              sx={{
                border: 'none',
                '& .MuiDataGrid-cell:focus': {
                  outline: 'none',
                },
                '& .MuiDataGrid-columnHeaders': {
                  backgroundColor: 'rgba(0, 0, 0, 0.02)',
                },
              }}
            />
          </Paper>
        )}
      </Box>
    </>
  );
};

export default Matches; 