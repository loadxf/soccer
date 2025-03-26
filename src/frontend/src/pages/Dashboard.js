import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Grid, 
  Typography, 
  Card, 
  CardContent, 
  Divider,
  Paper,
  CircularProgress,
  Alert,
  useMediaQuery,
  useTheme
} from '@mui/material';
import { 
  SportsSoccer as MatchIcon, 
  Groups as TeamIcon, 
  ShowChart as PredictionIcon, 
  Functions as AccuracyIcon 
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';
import { Chart as ChartJS, ArcElement, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Pie, Bar } from 'react-chartjs-2';
import api from '../services/api';

// Register ChartJS components
ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    matchCount: 0,
    teamCount: 0,
    predictionCount: 0,
    averageAccuracy: 0,
    recentMatches: [],
    predictionDistribution: {
      homeWin: 0,
      awayWin: 0,
      draw: 0
    },
    modelAccuracy: []
  });
  
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch dashboard statistics
        const response = await api.get('/api/v1/dashboard/stats');
        setStats(response.data);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  const statCards = [
    {
      title: 'Total Matches',
      value: stats.matchCount,
      icon: <MatchIcon sx={{ fontSize: { xs: 30, md: 40 }, color: 'primary.main' }} />,
      color: 'primary.main'
    },
    {
      title: 'Total Teams',
      value: stats.teamCount,
      icon: <TeamIcon sx={{ fontSize: { xs: 30, md: 40 }, color: 'info.main' }} />,
      color: 'info.main'
    },
    {
      title: 'Predictions Made',
      value: stats.predictionCount,
      icon: <PredictionIcon sx={{ fontSize: { xs: 30, md: 40 }, color: 'success.main' }} />,
      color: 'success.main'
    },
    {
      title: 'Average Accuracy',
      value: `${stats.averageAccuracy}%`,
      icon: <AccuracyIcon sx={{ fontSize: { xs: 30, md: 40 }, color: 'warning.main' }} />,
      color: 'warning.main'
    }
  ];

  // Chart data for prediction distribution
  const pieChartData = {
    labels: ['Home Win', 'Away Win', 'Draw'],
    datasets: [
      {
        data: [
          stats.predictionDistribution.homeWin,
          stats.predictionDistribution.awayWin,
          stats.predictionDistribution.draw
        ],
        backgroundColor: ['#1e88e5', '#43a047', '#ffc107'],
        borderColor: ['#1976d2', '#388e3c', '#ffb300'],
        borderWidth: 1,
      },
    ],
  };

  // Chart data for model accuracy
  const barChartData = {
    labels: stats.modelAccuracy.map(model => isMobile ? model.name.substring(0, 10) : model.name),
    datasets: [
      {
        label: 'Accuracy (%)',
        data: stats.modelAccuracy.map(model => model.accuracy),
        backgroundColor: '#1e88e5',
      },
    ],
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        display: !isMobile,
      },
      title: {
        display: true,
        text: 'Model Accuracy Comparison',
        font: {
          size: isMobile ? 14 : 16
        }
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          font: {
            size: isMobile ? 10 : 12
          }
        }
      },
      x: {
        ticks: {
          font: {
            size: isMobile ? 10 : 12
          }
        }
      }
    },
  };
  
  const pieChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: isMobile ? 'bottom' : 'right',
        labels: {
          font: {
            size: isMobile ? 10 : 12
          }
        }
      }
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '70vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ mt: 4, mx: 2 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <>
      <Helmet>
        <title>Dashboard | Soccer Prediction System</title>
      </Helmet>
      <Box sx={{ px: { xs: 0, sm: 1 } }}>
        <Typography 
          variant={isMobile ? "h5" : "h4"} 
          component="h1" 
          gutterBottom 
          sx={{ 
            fontWeight: 600, 
            mb: { xs: 3, md: 4 },
            fontSize: { xs: '1.5rem', sm: '2rem', md: '2.125rem' }
          }}
        >
          Dashboard
        </Typography>

        {/* Stats Cards */}
        <Grid container spacing={2} sx={{ mb: { xs: 3, md: 5 } }}>
          {statCards.map((card, index) => (
            <Grid item xs={6} md={3} key={index}>
              <Card 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  flexDirection: 'column',
                  borderRadius: 3,
                  boxShadow: '0 4px 12px 0 rgba(0,0,0,0.05)'
                }}
              >
                <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', p: { xs: 1.5, md: 2 } }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Typography 
                      variant={isMobile ? "subtitle2" : "h6"} 
                      color="text.secondary"
                      sx={{ fontSize: { xs: '0.875rem', md: '1rem' } }}
                    >
                      {card.title}
                    </Typography>
                    {card.icon}
                  </Box>
                  <Typography 
                    variant={isMobile ? "h5" : "h3"}
                    component="div" 
                    sx={{ 
                      mt: 'auto', 
                      fontWeight: 600,
                      color: card.color,
                      fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2.5rem' }
                    }}
                  >
                    {card.value}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Charts */}
        <Grid container spacing={2} sx={{ mb: { xs: 3, md: 5 } }}>
          <Grid item xs={12} md={5}>
            <Paper 
              sx={{ 
                p: { xs: 2, md: 3 }, 
                height: { xs: 300, md: '100%' },
                borderRadius: 3,
                boxShadow: '0 4px 12px 0 rgba(0,0,0,0.05)'
              }}
            >
              <Typography 
                variant={isMobile ? "subtitle1" : "h6"} 
                gutterBottom 
                sx={{ mb: { xs: 2, md: 3 } }}
              >
                Prediction Distribution
              </Typography>
              <Box sx={{ height: { xs: 220, md: 280 }, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <Pie data={pieChartData} options={pieChartOptions} />
              </Box>
            </Paper>
          </Grid>
          <Grid item xs={12} md={7}>
            <Paper 
              sx={{ 
                p: { xs: 2, md: 3 }, 
                height: { xs: 300, md: '100%' },
                borderRadius: 3,
                boxShadow: '0 4px 12px 0 rgba(0,0,0,0.05)'
              }}
            >
              <Typography 
                variant={isMobile ? "subtitle1" : "h6"} 
                gutterBottom 
                sx={{ mb: { xs: 2, md: 3 } }}
              >
                Model Accuracy
              </Typography>
              <Box sx={{ height: { xs: 220, md: 280 } }}>
                <Bar options={barChartOptions} data={barChartData} />
              </Box>
            </Paper>
          </Grid>
        </Grid>

        {/* Recent Matches */}
        <Paper 
          sx={{ 
            p: { xs: 2, md: 3 },
            borderRadius: 3,
            boxShadow: '0 4px 12px 0 rgba(0,0,0,0.05)'
          }}
        >
          <Typography 
            variant={isMobile ? "subtitle1" : "h6"} 
            gutterBottom 
            sx={{ mb: { xs: 2, md: 3 } }}
          >
            Recent Matches
          </Typography>
          {stats.recentMatches.length > 0 ? (
            stats.recentMatches.map((match, index) => (
              <Box key={index}>
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center', 
                  py: { xs: 1.5, md: 2 },
                  flexDirection: isMobile ? 'column' : 'row',
                  alignItems: isMobile ? 'flex-start' : 'center',
                }}>
                  <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    mb: isMobile ? 1 : 0,
                    flexWrap: isMobile ? 'wrap' : 'nowrap',
                    width: isMobile ? '100%' : 'auto',
                  }}>
                    <Typography 
                      variant="body1" 
                      fontWeight={500}
                      sx={{ fontSize: { xs: '0.875rem', md: '1rem' } }}
                    >
                      {match.homeTeam}
                    </Typography>
                    <Typography 
                      variant="body1" 
                      sx={{ 
                        mx: 1, 
                        fontWeight: 600,
                        fontSize: { xs: '0.875rem', md: '1rem' }
                      }}
                    >
                      {match.homeGoals} - {match.awayGoals}
                    </Typography>
                    <Typography 
                      variant="body1" 
                      fontWeight={500}
                      sx={{ fontSize: { xs: '0.875rem', md: '1rem' } }}
                    >
                      {match.awayTeam}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                      sx={{ fontSize: { xs: '0.75rem', md: '0.875rem' } }}
                    >
                      {match.date}
                    </Typography>
                  </Box>
                </Box>
                {index < stats.recentMatches.length - 1 && <Divider />}
              </Box>
            ))
          ) : (
            <Typography variant="body1" color="text.secondary">
              No recent matches available.
            </Typography>
          )}
        </Paper>
      </Box>
    </>
  );
};

export default Dashboard; 