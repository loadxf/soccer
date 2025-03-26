import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Paper, 
  Grid, 
  TextField,
  Button,
  Divider,
  Breadcrumbs,
  Link,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
  Alert,
  Snackbar,
  useTheme,
  useMediaQuery,
  Tooltip,
  IconButton
} from '@mui/material';
import { Helmet } from 'react-helmet-async';
import HomeIcon from '@mui/icons-material/Home';
import SportsIcon from '@mui/icons-material/Sports';
import InfoIcon from '@mui/icons-material/Info';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import PredictionConfidence from '../components/visualizations/PredictionConfidence';
import FeatureImportance from '../components/visualizations/FeatureImportance';
import axios from 'axios';

const Demo = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // State for the stepper
  const [activeStep, setActiveStep] = useState(0);
  const steps = ['Select Teams', 'Enter Features', 'View Prediction'];
  
  // State for the form
  const [homeTeam, setHomeTeam] = useState('');
  const [awayTeam, setAwayTeam] = useState('');
  const [features, setFeatures] = useState({
    home_team_rank: 10,
    away_team_rank: 15,
    home_team_form: 'W-W-D-L-W',
    away_team_form: 'L-W-W-D-W',
    home_team_goals_scored_last_5: 8,
    away_team_goals_scored_last_5: 6,
    home_team_goals_conceded_last_5: 4,
    away_team_goals_conceded_last_5: 7,
    home_advantage_index: 1.2,
    days_since_last_match_home: 7,
    days_since_last_match_away: 5,
    head_to_head_wins_home: 3,
    head_to_head_wins_away: 2,
    head_to_head_draws: 1
  });
  
  // State for API data
  const [teams, setTeams] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  
  // Snackbar state
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'info'
  });

  // Fetch teams and models when component mounts
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Get teams
        const teamsResponse = await axios.get('/api/v1/teams');
        setTeams(teamsResponse.data.teams || []);
        
        // Get available models
        const modelsResponse = await axios.get('/api/v1/predictions/models');
        setModels(modelsResponse.data.models || []);
        if (modelsResponse.data.models && modelsResponse.data.models.length > 0) {
          setSelectedModel(modelsResponse.data.models[0].id);
        }
      } catch (error) {
        console.error('Error fetching initial data:', error);
        setError('Failed to load initial data. Please try again later.');
        setSnackbar({
          open: true,
          message: 'Failed to load initial data. Please try again later.',
          severity: 'error'
        });
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Handle next button in stepper
  const handleNext = () => {
    if (activeStep === 0 && (!homeTeam || !awayTeam)) {
      setSnackbar({
        open: true,
        message: 'Please select both home and away teams',
        severity: 'warning'
      });
      return;
    }
    
    if (activeStep === steps.length - 2) {
      // Submit prediction request before moving to final step
      submitPrediction();
    }
    
    setActiveStep((prevActiveStep) => prevActiveStep + 1);
  };

  // Handle back button in stepper
  const handleBack = () => {
    setActiveStep((prevActiveStep) => prevActiveStep - 1);
  };

  // Reset the form
  const handleReset = () => {
    setActiveStep(0);
    setHomeTeam('');
    setAwayTeam('');
    setFeatures({
      home_team_rank: 10,
      away_team_rank: 15,
      home_team_form: 'W-W-D-L-W',
      away_team_form: 'L-W-W-D-W',
      home_team_goals_scored_last_5: 8,
      away_team_goals_scored_last_5: 6,
      home_team_goals_conceded_last_5: 4,
      away_team_goals_conceded_last_5: 7,
      home_advantage_index: 1.2,
      days_since_last_match_home: 7,
      days_since_last_match_away: 5,
      head_to_head_wins_home: 3,
      head_to_head_wins_away: 2,
      head_to_head_draws: 1
    });
    setPrediction(null);
  };

  // Handle feature changes
  const handleFeatureChange = (e) => {
    const { name, value } = e.target;
    
    // Convert numeric values from string to number
    const numericFeatures = [
      'home_team_rank', 'away_team_rank', 
      'home_team_goals_scored_last_5', 'away_team_goals_scored_last_5', 
      'home_team_goals_conceded_last_5', 'away_team_goals_conceded_last_5',
      'home_advantage_index', 'days_since_last_match_home', 'days_since_last_match_away',
      'head_to_head_wins_home', 'head_to_head_wins_away', 'head_to_head_draws'
    ];
    
    setFeatures(prev => ({
      ...prev,
      [name]: numericFeatures.includes(name) ? Number(value) : value
    }));
  };

  // Submit prediction request
  const submitPrediction = async () => {
    setLoading(true);
    try {
      const requestData = {
        home_team: homeTeam,
        away_team: awayTeam,
        features: features,
        model_name: selectedModel || undefined
      };
      
      const response = await axios.post('/api/v1/predictions/custom', requestData);
      setPrediction(response.data);
    } catch (error) {
      console.error('Error submitting prediction:', error);
      setError('Failed to get prediction. Please try again.');
      setSnackbar({
        open: true,
        message: 'Failed to get prediction. Please try again.',
        severity: 'error'
      });
    } finally {
      setLoading(false);
    }
  };

  // Close snackbar
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Render team selection step
  const renderTeamSelection = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Select teams for prediction
        </Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Choose the home and away teams for the match you want to predict.
        </Typography>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel id="home-team-label">Home Team</InputLabel>
          <Select
            labelId="home-team-label"
            id="home-team"
            value={homeTeam}
            label="Home Team"
            onChange={(e) => setHomeTeam(e.target.value)}
          >
            {teams.map((team) => (
              <MenuItem key={team.id} value={team.id}>
                {team.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel id="away-team-label">Away Team</InputLabel>
          <Select
            labelId="away-team-label"
            id="away-team"
            value={awayTeam}
            label="Away Team"
            onChange={(e) => setAwayTeam(e.target.value)}
          >
            {teams.map((team) => (
              <MenuItem key={team.id} value={team.id}>
                {team.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      
      <Grid item xs={12}>
        <FormControl fullWidth sx={{ mt: 2 }}>
          <InputLabel id="model-label">Prediction Model</InputLabel>
          <Select
            labelId="model-label"
            id="model"
            value={selectedModel}
            label="Prediction Model"
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {models.map((model) => (
              <MenuItem key={model.id} value={model.id}>
                {model.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
    </Grid>
  );

  // Render feature input step
  const renderFeatureInput = () => (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Enter Match Features
        </Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Provide additional features to improve prediction accuracy. You can use the default values or customize them.
        </Typography>
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Home Team Rank"
          name="home_team_rank"
          type="number"
          value={features.home_team_rank}
          onChange={handleFeatureChange}
          helperText="FIFA/UEFA ranking (lower is better)"
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Away Team Rank"
          name="away_team_rank"
          type="number"
          value={features.away_team_rank}
          onChange={handleFeatureChange}
          helperText="FIFA/UEFA ranking (lower is better)"
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Home Advantage Index"
          name="home_advantage_index"
          type="number"
          inputProps={{ step: 0.1 }}
          value={features.home_advantage_index}
          onChange={handleFeatureChange}
          helperText="Strength of home advantage (1.0 is neutral)"
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Home Team Form"
          name="home_team_form"
          value={features.home_team_form}
          onChange={handleFeatureChange}
          helperText="Last 5 matches (W-L-D-W-W format)"
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Away Team Form"
          name="away_team_form"
          value={features.away_team_form}
          onChange={handleFeatureChange}
          helperText="Last 5 matches (W-L-D-W-W format)"
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Days Since Last Match (Home)"
          name="days_since_last_match_home"
          type="number"
          value={features.days_since_last_match_home}
          onChange={handleFeatureChange}
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Days Since Last Match (Away)"
          name="days_since_last_match_away"
          type="number"
          value={features.days_since_last_match_away}
          onChange={handleFeatureChange}
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Home Goals Scored (Last 5)"
          name="home_team_goals_scored_last_5"
          type="number"
          value={features.home_team_goals_scored_last_5}
          onChange={handleFeatureChange}
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Away Goals Scored (Last 5)"
          name="away_team_goals_scored_last_5"
          type="number"
          value={features.away_team_goals_scored_last_5}
          onChange={handleFeatureChange}
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Home Goals Conceded (Last 5)"
          name="home_team_goals_conceded_last_5"
          type="number"
          value={features.home_team_goals_conceded_last_5}
          onChange={handleFeatureChange}
        />
      </Grid>
      
      <Grid item xs={12} sm={6} md={4}>
        <TextField
          fullWidth
          label="Away Goals Conceded (Last 5)"
          name="away_team_goals_conceded_last_5"
          type="number"
          value={features.away_team_goals_conceded_last_5}
          onChange={handleFeatureChange}
        />
      </Grid>
      
      <Grid item xs={12}>
        <Typography variant="subtitle1" gutterBottom>
          Head-to-Head Record
        </Typography>
      </Grid>
      
      <Grid item xs={12} sm={4}>
        <TextField
          fullWidth
          label="Home Team Wins"
          name="head_to_head_wins_home"
          type="number"
          value={features.head_to_head_wins_home}
          onChange={handleFeatureChange}
        />
      </Grid>
      
      <Grid item xs={12} sm={4}>
        <TextField
          fullWidth
          label="Away Team Wins"
          name="head_to_head_wins_away"
          type="number"
          value={features.head_to_head_wins_away}
          onChange={handleFeatureChange}
        />
      </Grid>
      
      <Grid item xs={12} sm={4}>
        <TextField
          fullWidth
          label="Draws"
          name="head_to_head_draws"
          type="number"
          value={features.head_to_head_draws}
          onChange={handleFeatureChange}
        />
      </Grid>
    </Grid>
  );

  // Render prediction results
  const renderPredictionResults = () => {
    // Find team names based on IDs
    const homeTeamName = teams.find(t => t.id === homeTeam)?.name || 'Home Team';
    const awayTeamName = teams.find(t => t.id === awayTeam)?.name || 'Away Team';
    
    if (loading) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4, mb: 4 }}>
          <CircularProgress />
        </Box>
      );
    }
    
    if (!prediction) {
      return (
        <Alert severity="warning">
          No prediction data available. Please try again.
        </Alert>
      );
    }
    
    return (
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Typography variant="h6" align="center" gutterBottom>
            Match Prediction: {homeTeamName} vs {awayTeamName}
          </Typography>
        </Grid>
        
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h5" align="center" gutterBottom>
                {prediction.result === 'home_win' 
                  ? `${homeTeamName} Win` 
                  : prediction.result === 'away_win' 
                    ? `${awayTeamName} Win` 
                    : 'Draw'}
              </Typography>
              
              <Typography variant="body1" align="center" color="textSecondary" paragraph>
                Confidence: {(prediction.confidence * 100).toFixed(2)}%
              </Typography>
              
              <Grid container spacing={2} justifyContent="center" sx={{ mt: 2 }}>
                <Grid item xs={4} textAlign="center">
                  <Typography variant="h6">{(prediction.probabilities.home_win * 100).toFixed(2)}%</Typography>
                  <Typography variant="body2">Home Win</Typography>
                </Grid>
                <Grid item xs={4} textAlign="center">
                  <Typography variant="h6">{(prediction.probabilities.draw * 100).toFixed(2)}%</Typography>
                  <Typography variant="body2">Draw</Typography>
                </Grid>
                <Grid item xs={4} textAlign="center">
                  <Typography variant="h6">{(prediction.probabilities.away_win * 100).toFixed(2)}%</Typography>
                  <Typography variant="body2">Away Win</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Prediction Confidence</Typography>
              <Box sx={{ height: 300 }}>
                <PredictionConfidence 
                  data={{
                    labels: ['Home Win', 'Draw', 'Away Win'],
                    values: [
                      prediction.probabilities.home_win,
                      prediction.probabilities.draw,
                      prediction.probabilities.away_win
                    ]
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Feature Importance</Typography>
              <Box sx={{ height: 300 }}>
                <FeatureImportance 
                  data={prediction.feature_importance || {
                    labels: [
                      'Team Rank', 'Recent Form', 'Goals Scored',
                      'Goals Conceded', 'Home Advantage', 'Head-to-Head'
                    ],
                    values: [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Prediction Details</Typography>
              <Typography variant="body2" paragraph>
                This prediction was made using the {models.find(m => m.id === selectedModel)?.name || 'selected'} model.
                The model considers various factors including team rankings, recent form, scoring patterns,
                and historical head-to-head performance.
              </Typography>
              
              <Typography variant="body2">
                <strong>Expected Goals:</strong> 
                {' '}{homeTeamName}: {prediction.expected_goals?.home.toFixed(2) || '1.5'} 
                {' | '}
                {awayTeamName}: {prediction.expected_goals?.away.toFixed(2) || '1.2'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    );
  };

  // Render content based on current step
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return renderTeamSelection();
      case 1:
        return renderFeatureInput();
      case 2:
        return renderPredictionResults();
      default:
        return 'Unknown step';
    }
  };

  return (
    <Container maxWidth="lg">
      <Helmet>
        <title>Interactive Demo | Soccer Prediction System</title>
      </Helmet>
      
      <Box sx={{ mb: 4 }}>
        <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 2 }}>
          <Link
            underline="hover"
            color="inherit"
            href="/"
            sx={{ display: 'flex', alignItems: 'center' }}
          >
            <HomeIcon sx={{ mr: 0.5 }} fontSize="inherit" />
            Home
          </Link>
          <Typography
            sx={{ display: 'flex', alignItems: 'center' }}
            color="text.primary"
          >
            <SportsIcon sx={{ mr: 0.5 }} fontSize="inherit" />
            Interactive Demo
          </Typography>
        </Breadcrumbs>
        
        <Typography variant="h4" component="h1" gutterBottom>
          Interactive Match Prediction Demo
        </Typography>
        
        <Typography variant="body1" color="textSecondary" paragraph>
          This interactive demo allows you to predict the outcome of a soccer match
          using our advanced prediction models. Follow the steps to select teams, 
          customize match features, and view detailed predictions.
        </Typography>
      </Box>
      
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Stepper activeStep={activeStep} alternativeLabel={!isMobile} orientation={isMobile ? 'vertical' : 'horizontal'}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <Box sx={{ mt: 4, mb: 2 }}>
          {getStepContent(activeStep)}
        </Box>
        
        <Box sx={{ display: 'flex', flexDirection: 'row', pt: 2 }}>
          <Button
            variant="outlined"
            disabled={activeStep === 0}
            onClick={handleBack}
            sx={{ mr: 1 }}
            startIcon={<ArrowBackIcon />}
          >
            Back
          </Button>
          <Box sx={{ flex: '1 1 auto' }} />
          
          <Button 
            variant="outlined" 
            onClick={handleReset}
            sx={{ mr: 1 }}
            startIcon={<RestartAltIcon />}
          >
            Reset
          </Button>
          
          {activeStep === steps.length - 1 ? (
            <Button 
              variant="contained" 
              onClick={handleReset}
              startIcon={<RestartAltIcon />}
            >
              New Prediction
            </Button>
          ) : (
            <Button
              variant="contained"
              onClick={handleNext}
              endIcon={activeStep === steps.length - 2 ? <PlayArrowIcon /> : <ArrowForwardIcon />}
              disabled={loading}
            >
              {activeStep === steps.length - 2 ? 'Generate Prediction' : 'Next'}
            </Button>
          )}
        </Box>
      </Paper>
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Demo; 