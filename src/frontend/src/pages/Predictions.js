import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  TextField,
  Chip,
  LinearProgress,
  Stack
} from '@mui/material';
import {
  SportsSoccer as SportsIcon,
  Analytics as AnalyticsIcon,
  Timeline as TimelineIcon,
  CompareArrows as CompareIcon
} from '@mui/icons-material';
import { Helmet } from 'react-helmet-async';
import { 
  Chart as ChartJS, 
  ArcElement, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement
} from 'chart.js';
import { Pie, Bar, Radar } from 'react-chartjs-2';
import api from '../services/api';

// Register ChartJS components
ChartJS.register(
  ArcElement, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement
);

const Predictions = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [customMatch, setCustomMatch] = useState({
    homeTeam: '',
    awayTeam: '',
    league: '',
    date: ''
  });
  const [customPrediction, setCustomPrediction] = useState(null);
  const [customPredicting, setCustomPredicting] = useState(false);
  const [upcomingMatches, setUpcomingMatches] = useState([]);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [leagues, setLeagues] = useState([]);
  const [teams, setTeams] = useState([]);

  // Tab options
  const tabOptions = ['Make Prediction', 'Upcoming Matches', 'Prediction History', 'Model Comparison'];

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Fetch available prediction models
        const modelsResponse = await api.get('/api/v1/predictions/models');
        setModels(modelsResponse.data);
        
        if (modelsResponse.data.length > 0) {
          setSelectedModel(modelsResponse.data[0].id);
        }
        
        // Fetch leagues and teams for dropdowns
        const leaguesResponse = await api.get('/api/v1/leagues');
        setLeagues(leaguesResponse.data);
        
        const teamsResponse = await api.get('/api/v1/teams?per_page=100');
        setTeams(teamsResponse.data.items);
        
        // Based on active tab, fetch additional data
        if (activeTab === 1) {
          await fetchUpcomingMatches();
        } else if (activeTab === 2) {
          await fetchPredictionHistory();
        }
      } catch (err) {
        console.error('Error fetching initial data:', err);
        setError('Failed to load data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, [activeTab]);

  const fetchUpcomingMatches = async () => {
    try {
      const response = await api.get('/api/v1/matches?status=scheduled&limit=10');
      setUpcomingMatches(response.data.items);
    } catch (err) {
      console.error('Error fetching upcoming matches:', err);
    }
  };

  const fetchPredictionHistory = async () => {
    try {
      const response = await api.get('/api/v1/predictions/history');
      setPredictionHistory(response.data);
    } catch (err) {
      console.error('Error fetching prediction history:', err);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleCustomMatchChange = (field, value) => {
    setCustomMatch(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleMakePrediction = async () => {
    try {
      setCustomPredicting(true);
      setError(null);
      
      // Check if all required fields are filled
      const { homeTeam, awayTeam, league } = customMatch;
      if (!homeTeam || !awayTeam || !league) {
        setError('Please fill in all required fields.');
        return;
      }
      
      // Make prediction request
      const response = await api.post('/api/v1/predictions/custom', {
        homeTeam,
        awayTeam,
        league,
        date: customMatch.date || new Date().toISOString().split('T')[0],
        modelId: selectedModel
      });
      
      setCustomPrediction(response.data);
    } catch (err) {
      console.error('Error making prediction:', err);
      setError('Failed to make prediction. Please try again.');
    } finally {
      setCustomPredicting(false);
    }
  };

  // Radar chart data for team comparison
  const generateTeamComparisonData = () => {
    return {
      labels: ['Attack', 'Defense', 'Possession', 'Set Pieces', 'Form', 'Home/Away'],
      datasets: [
        {
          label: customMatch.homeTeam,
          data: [85, 70, 75, 65, 80, 90],
          backgroundColor: 'rgba(30, 136, 229, 0.2)',
          borderColor: 'rgba(30, 136, 229, 1)',
          borderWidth: 1,
        },
        {
          label: customMatch.awayTeam,
          data: [70, 75, 65, 80, 75, 60],
          backgroundColor: 'rgba(67, 160, 71, 0.2)',
          borderColor: 'rgba(67, 160, 71, 1)',
          borderWidth: 1,
        },
      ],
    };
  };

  // Pie chart data for prediction probabilities
  const generatePredictionChartData = () => {
    if (!customPrediction) return null;
    
    return {
      labels: ['Home Win', 'Draw', 'Away Win'],
      datasets: [
        {
          data: [
            customPrediction.probabilities.homeWin * 100,
            customPrediction.probabilities.draw * 100,
            customPrediction.probabilities.awayWin * 100
          ],
          backgroundColor: ['#1e88e5', '#ffc107', '#43a047'],
          borderColor: ['#1976d2', '#ffb300', '#388e3c'],
          borderWidth: 1,
        },
      ],
    };
  };

  // Helper function to render progress for probabilities
  const renderProbabilityBar = (value, label, color) => (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="body2">{label}</Typography>
        <Typography variant="body2" fontWeight={500}>{Math.round(value * 100)}%</Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={value * 100}
        sx={{
          height: 10,
          borderRadius: 5,
          backgroundColor: 'rgba(0,0,0,0.05)',
          '& .MuiLinearProgress-bar': {
            borderRadius: 5,
            backgroundColor: color,
          }
        }}
      />
    </Box>
  );

  // Render content based on active tab
  const renderTabContent = () => {
    switch (activeTab) {
      case 0: // Make Prediction
        return (
          <Grid container spacing={3}>
            {/* Form Section */}
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, borderRadius: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
                  Custom Match Prediction
                </Typography>
                
                <FormControl fullWidth variant="outlined" sx={{ mb: 3 }}>
                  <InputLabel id="model-select-label">Prediction Model</InputLabel>
                  <Select
                    labelId="model-select-label"
                    id="model-select"
                    value={selectedModel}
                    onChange={handleModelChange}
                    label="Prediction Model"
                  >
                    {models.map((model) => (
                      <MenuItem key={model.id} value={model.id}>
                        {model.name} ({model.accuracy}% accuracy)
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl fullWidth variant="outlined" sx={{ mb: 3 }}>
                  <InputLabel id="home-team-label">Home Team</InputLabel>
                  <Select
                    labelId="home-team-label"
                    id="home-team"
                    value={customMatch.homeTeam}
                    onChange={(e) => handleCustomMatchChange('homeTeam', e.target.value)}
                    label="Home Team"
                  >
                    {teams.map((team) => (
                      <MenuItem key={team.id} value={team.name}>
                        {team.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl fullWidth variant="outlined" sx={{ mb: 3 }}>
                  <InputLabel id="away-team-label">Away Team</InputLabel>
                  <Select
                    labelId="away-team-label"
                    id="away-team"
                    value={customMatch.awayTeam}
                    onChange={(e) => handleCustomMatchChange('awayTeam', e.target.value)}
                    label="Away Team"
                  >
                    {teams.map((team) => (
                      <MenuItem key={team.id} value={team.name}>
                        {team.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl fullWidth variant="outlined" sx={{ mb: 3 }}>
                  <InputLabel id="league-label">League</InputLabel>
                  <Select
                    labelId="league-label"
                    id="league"
                    value={customMatch.league}
                    onChange={(e) => handleCustomMatchChange('league', e.target.value)}
                    label="League"
                  >
                    {leagues.map((league) => (
                      <MenuItem key={league.id} value={league.name}>
                        {league.name}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <TextField
                  fullWidth
                  id="date"
                  label="Match Date"
                  type="date"
                  value={customMatch.date}
                  onChange={(e) => handleCustomMatchChange('date', e.target.value)}
                  InputLabelProps={{
                    shrink: true,
                  }}
                  sx={{ mb: 3 }}
                />
                
                <Button
                  variant="contained"
                  fullWidth
                  size="large"
                  onClick={handleMakePrediction}
                  disabled={customPredicting || !customMatch.homeTeam || !customMatch.awayTeam || !customMatch.league}
                  sx={{ py: 1.5 }}
                >
                  {customPredicting ? <CircularProgress size={24} color="inherit" /> : "Predict Match Outcome"}
                </Button>
              </Paper>
            </Grid>
            
            {/* Results Section */}
            <Grid item xs={12} md={6}>
              {customPrediction ? (
                <Paper sx={{ p: 3, borderRadius: 3, height: '100%' }}>
                  <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
                    Prediction Results
                  </Typography>
                  
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                    <Chip 
                      label={`Predicted Outcome: ${customPrediction.outcome}`} 
                      color="primary"
                      sx={{ fontSize: '1rem', py: 2, px: 3 }}
                    />
                  </Box>
                  
                  <Typography variant="subtitle1" gutterBottom>
                    Outcome Probabilities
                  </Typography>
                  
                  {renderProbabilityBar(
                    customPrediction.probabilities.homeWin, 
                    `${customMatch.homeTeam} Win`, 
                    '#1e88e5'
                  )}
                  
                  {renderProbabilityBar(
                    customPrediction.probabilities.draw, 
                    'Draw', 
                    '#ffc107'
                  )}
                  
                  {renderProbabilityBar(
                    customPrediction.probabilities.awayWin, 
                    `${customMatch.awayTeam} Win`, 
                    '#43a047'
                  )}
                  
                  <Divider sx={{ my: 3 }} />
                  
                  <Typography variant="subtitle1" gutterBottom sx={{ mb: 2 }}>
                    Probability Distribution
                  </Typography>
                  
                  <Box sx={{ height: 250, display: 'flex', justifyContent: 'center' }}>
                    <Pie data={generatePredictionChartData()} />
                  </Box>
                  
                  <Divider sx={{ my: 3 }} />
                  
                  <Typography variant="subtitle1" gutterBottom sx={{ mb: 2 }}>
                    Key Factors
                  </Typography>
                  
                  <Stack spacing={1}>
                    {customPrediction.keyFactors?.map((factor, index) => (
                      <Chip 
                        key={index}
                        label={factor}
                        variant="outlined"
                        sx={{ justifyContent: 'flex-start' }}
                      />
                    ))}
                  </Stack>
                </Paper>
              ) : (
                <Paper 
                  sx={{ 
                    p: 3, 
                    borderRadius: 3, 
                    height: '100%', 
                    display: 'flex', 
                    flexDirection: 'column',
                    justifyContent: 'center',
                    alignItems: 'center',
                    textAlign: 'center'
                  }}
                >
                  <AnalyticsIcon sx={{ fontSize: 60, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    No Prediction Results Yet
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 400 }}>
                    Fill out the form on the left to get prediction results for a custom match.
                  </Typography>
                </Paper>
              )}
            </Grid>
            
            {/* Team Comparison Section */}
            {customMatch.homeTeam && customMatch.awayTeam && (
              <Grid item xs={12}>
                <Paper sx={{ p: 3, borderRadius: 3 }}>
                  <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
                    Team Comparison
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <Radar data={generateTeamComparisonData()} />
                  </Box>
                </Paper>
              </Grid>
            )}
          </Grid>
        );
        
      case 1: // Upcoming Matches
        return (
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
              Upcoming Matches
            </Typography>
            
            {loading ? (
              <CircularProgress />
            ) : upcomingMatches.length > 0 ? (
              <Grid container spacing={3}>
                {upcomingMatches.map((match) => (
                  <Grid item xs={12} sm={6} lg={4} key={match.id}>
                    <Card sx={{ borderRadius: 2 }}>
                      <CardContent>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          {match.date} • {match.league}
                        </Typography>
                        
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', my: 2 }}>
                          <Box sx={{ textAlign: 'center', flex: 1 }}>
                            <Typography variant="subtitle1" fontWeight={500}>
                              {match.homeTeam}
                            </Typography>
                          </Box>
                          
                          <Box sx={{ mx: 2 }}>
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              vs
                            </Typography>
                          </Box>
                          
                          <Box sx={{ textAlign: 'center', flex: 1 }}>
                            <Typography variant="subtitle1" fontWeight={500}>
                              {match.awayTeam}
                            </Typography>
                          </Box>
                        </Box>
                        
                        <Button 
                          variant="contained" 
                          fullWidth
                          sx={{ mt: 2 }}
                        >
                          Predict
                        </Button>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography variant="body1" color="text.secondary">
                No upcoming matches available.
              </Typography>
            )}
          </Paper>
        );
        
      case 2: // Prediction History
        return (
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
              Prediction History
            </Typography>
            
            {loading ? (
              <CircularProgress />
            ) : predictionHistory.length > 0 ? (
              <Grid container spacing={3}>
                {predictionHistory.map((prediction) => (
                  <Grid item xs={12} key={prediction.id}>
                    <Card sx={{ borderRadius: 2 }}>
                      <CardContent>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="subtitle1" fontWeight={500}>
                            {prediction.homeTeam} vs {prediction.awayTeam}
                          </Typography>
                          
                          <Chip 
                            label={prediction.accuracy ? `${Math.round(prediction.accuracy * 100)}% Accurate` : 'Pending'} 
                            color={prediction.accuracy > 0.7 ? 'success' : prediction.accuracy > 0.4 ? 'warning' : 'default'}
                            size="small"
                          />
                        </Box>
                        
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                          Predicted: {prediction.predictedOutcome} • Actual: {prediction.actualOutcome || 'Not played yet'}
                        </Typography>
                        
                        <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="body2" color="text.secondary">
                            {prediction.date} • {prediction.modelName}
                          </Typography>
                          
                          <Button size="small" color="primary">
                            Details
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography variant="body1" color="text.secondary">
                No prediction history available.
              </Typography>
            )}
          </Paper>
        );
        
      case 3: // Model Comparison
        return (
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" gutterBottom sx={{ mb: 3 }}>
              Model Comparison
            </Typography>
            
            {loading ? (
              <CircularProgress />
            ) : models.length > 0 ? (
              <Box sx={{ height: 400 }}>
                <Bar 
                  data={{
                    labels: models.map(model => model.name),
                    datasets: [
                      {
                        label: 'Accuracy (%)',
                        data: models.map(model => model.accuracy),
                        backgroundColor: '#1e88e5',
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    scales: {
                      y: {
                        beginAtZero: true,
                        max: 100,
                      }
                    }
                  }}
                />
              </Box>
            ) : (
              <Typography variant="body1" color="text.secondary">
                No model data available for comparison.
              </Typography>
            )}
          </Paper>
        );
        
      default:
        return null;
    }
  };

  return (
    <>
      <Helmet>
        <title>Predictions | Soccer Prediction System</title>
      </Helmet>
      <Box sx={{ px: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
            Predictions
          </Typography>
        </Box>
        
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}
        
        {/* Tabs Navigation */}
        <Box sx={{ mb: 3, borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
          >
            {tabOptions.map((tab, index) => (
              <Tab 
                key={index} 
                label={tab} 
                icon={
                  index === 0 ? <AnalyticsIcon /> : 
                  index === 1 ? <SportsIcon /> : 
                  index === 2 ? <TimelineIcon /> : 
                  <CompareIcon />
                }
                iconPosition="start"
              />
            ))}
          </Tabs>
        </Box>
        
        {/* Tab Content */}
        {renderTabContent()}
      </Box>
    </>
  );
};

export default Predictions; 