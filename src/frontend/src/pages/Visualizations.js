import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Tab, 
  Tabs, 
  Paper, 
  Grid,
  Divider,
  Breadcrumbs,
  Link
} from '@mui/material';
import { Helmet } from 'react-helmet';
import HomeIcon from '@mui/icons-material/Home';
import EqualizerIcon from '@mui/icons-material/Equalizer';
import BarChartIcon from '@mui/icons-material/BarChart';
import ScatterPlotIcon from '@mui/icons-material/ScatterPlot';
import TimelineIcon from '@mui/icons-material/Timeline';
import RadarIcon from '@mui/icons-material/Radar';
import PersonIcon from '@mui/icons-material/Person';
import GroupsIcon from '@mui/icons-material/Groups';

import TeamPerformance from '../components/visualizations/TeamPerformance';
import ModelComparison from '../components/visualizations/ModelComparison';
import FeatureImportance from '../components/visualizations/FeatureImportance';
import PredictionConfidence from '../components/visualizations/PredictionConfidence';

// Mock data - replace with actual API calls in production
const MOCK_TEAMS = [
  'Arsenal',
  'Aston Villa',
  'Brighton',
  'Burnley',
  'Chelsea',
  'Crystal Palace',
  'Everton',
  'Fulham',
  'Leeds United',
  'Leicester City',
  'Liverpool',
  'Manchester City',
  'Manchester United',
  'Newcastle United',
  'Sheffield United',
  'Southampton',
  'Tottenham Hotspur',
  'West Brom',
  'West Ham United',
  'Wolverhampton Wanderers'
];

const MOCK_MODELS = [
  { id: 'model1', name: 'XGBoost Model' },
  { id: 'model2', name: 'LightGBM Model' },
  { id: 'model3', name: 'Neural Network Model' },
  { id: 'model4', name: 'Ensemble Model' },
  { id: 'model5', name: 'Time Series Model' }
];

const MOCK_MATCHES = [
  { id: 'match1', homeTeam: 'Arsenal', awayTeam: 'Chelsea', date: '2023-10-15' },
  { id: 'match2', homeTeam: 'Liverpool', awayTeam: 'Manchester United', date: '2023-10-16' },
  { id: 'match3', homeTeam: 'Manchester City', awayTeam: 'Tottenham Hotspur', date: '2023-10-17' },
  { id: 'match4', homeTeam: 'West Ham United', awayTeam: 'Everton', date: '2023-10-18' },
  { id: 'match5', homeTeam: 'Leicester City', awayTeam: 'Newcastle United', date: '2023-10-19' },
  { id: 'match6', homeTeam: 'Crystal Palace', awayTeam: 'Leeds United', date: '2023-10-20' },
  { id: 'match7', homeTeam: 'Aston Villa', awayTeam: 'Southampton', date: '2023-10-21' },
  { id: 'match8', homeTeam: 'Wolverhampton', awayTeam: 'Brighton', date: '2023-10-22' },
  { id: 'match9', homeTeam: 'Burnley', awayTeam: 'Fulham', date: '2023-10-23' },
  { id: 'match10', homeTeam: 'West Brom', awayTeam: 'Sheffield United', date: '2023-10-24' }
];

function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`visualization-tabpanel-${index}`}
      aria-labelledby={`visualization-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index) {
  return {
    id: `visualization-tab-${index}`,
    'aria-controls': `visualization-tabpanel-${index}`,
  };
}

const Visualizations = () => {
  const [value, setValue] = useState(0);
  const [teams, setTeams] = useState(MOCK_TEAMS);
  const [models, setModels] = useState(MOCK_MODELS);
  const [matches, setMatches] = useState(MOCK_MATCHES);
  const [loading, setLoading] = useState(false);

  // In a real implementation, you would fetch data from API
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        // Replace with real API calls
        // const teamsResponse = await api.getTeams();
        // const modelsResponse = await api.getModels();
        // const matchesResponse = await api.getMatches();
        
        // setTeams(teamsResponse.data);
        // setModels(modelsResponse.data);
        // setMatches(matchesResponse.data);
      } catch (error) {
        console.error('Error fetching data for visualizations:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  return (
    <Container maxWidth="xl">
      <Helmet>
        <title>Data Visualizations | Soccer Prediction System</title>
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
            <EqualizerIcon sx={{ mr: 0.5 }} fontSize="inherit" />
            Visualizations
          </Typography>
        </Breadcrumbs>
        
        <Typography variant="h4" component="h1" gutterBottom>
          Data Visualizations
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Explore visual representations of our soccer prediction models, team performance, and more.
        </Typography>
        <Divider />
      </Box>
      
      <Paper sx={{ width: '100%' }}>
        <Tabs
          value={value}
          onChange={handleChange}
          indicatorColor="primary"
          textColor="primary"
          variant="scrollable"
          scrollButtons="auto"
          aria-label="visualization tabs"
        >
          <Tab label="Team Performance" icon={<GroupsIcon />} {...a11yProps(0)} />
          <Tab label="Model Comparison" icon={<RadarIcon />} {...a11yProps(1)} />
          <Tab label="Feature Importance" icon={<BarChartIcon />} {...a11yProps(2)} />
          <Tab label="Prediction Confidence" icon={<ScatterPlotIcon />} {...a11yProps(3)} />
        </Tabs>
        
        <TabPanel value={value} index={0}>
          <TeamPerformance 
            availableTeams={teams} 
            defaultTeam={teams.length > 0 ? teams[0] : null}
          />
        </TabPanel>
        
        <TabPanel value={value} index={1}>
          <ModelComparison 
            availableModels={models}
          />
        </TabPanel>
        
        <TabPanel value={value} index={2}>
          <FeatureImportance 
            availableModels={models}
          />
        </TabPanel>
        
        <TabPanel value={value} index={3}>
          <PredictionConfidence 
            availableMatches={matches}
          />
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default Visualizations; 