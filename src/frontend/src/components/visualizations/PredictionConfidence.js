import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Box, 
  Paper, 
  Typography, 
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Checkbox,
  Divider,
  CircularProgress
} from '@mui/material';
import SportsSoccerIcon from '@mui/icons-material/SportsSoccer';
import VisualizationCard from './VisualizationCard';
import { fetchPredictionConfidenceChart } from '../../utils/visualization';

/**
 * Prediction confidence visualization component.
 * 
 * @param {object} props - Component props
 * @param {array} props.availableMatches - List of available matches with ids and team names
 * @param {array} props.selectedMatchIds - Optional initially selected match IDs
 * @param {number} props.maxMatches - Maximum number of matches to display at once
 * @returns {React.ReactElement} Prediction confidence component
 */
const PredictionConfidence = ({ 
  availableMatches = [],
  selectedMatchIds = [],
  maxMatches = 8
}) => {
  const [selected, setSelected] = useState(selectedMatchIds);
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [loadingMatches, setLoadingMatches] = useState(false);

  const loadVisualization = async () => {
    if (selected.length === 0) {
      setError('Please select at least one match');
      return;
    }
    
    if (selected.length > maxMatches) {
      setError(`Please select at most ${maxMatches} matches for better visualization`);
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchPredictionConfidenceChart(selected);
      setImageData(data);
    } catch (err) {
      console.error('Error loading prediction confidence visualization:', err);
      setError('Failed to load prediction confidence visualization. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Init with some selected matches if provided
  useEffect(() => {
    if (selectedMatchIds.length > 0) {
      setSelected(selectedMatchIds);
      loadVisualization();
    } else if (availableMatches.length > 0) {
      // Select first few matches by default
      const initialSelection = availableMatches
        .slice(0, Math.min(maxMatches, availableMatches.length))
        .map(match => match.id);
      setSelected(initialSelection);
    }
  }, []);

  const handleToggle = (matchId) => () => {
    const currentIndex = selected.indexOf(matchId);
    const newSelected = [...selected];

    if (currentIndex === -1) {
      newSelected.push(matchId);
    } else {
      newSelected.splice(currentIndex, 1);
    }

    setSelected(newSelected);
  };

  const getMatchLabel = (match) => {
    return `${match.homeTeam} vs ${match.awayTeam}`;
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <Typography variant="h6" gutterBottom>
            Select Matches
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Select up to {maxMatches} matches to visualize prediction confidence
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          <Box sx={{ flexGrow: 1, overflow: 'auto', maxHeight: '400px' }}>
            {loadingMatches ? (
              <Box display="flex" justifyContent="center" p={4}>
                <CircularProgress size={30} />
              </Box>
            ) : availableMatches.length === 0 ? (
              <Typography variant="body2" color="text.secondary" align="center">
                No matches available
              </Typography>
            ) : (
              <List dense>
                {availableMatches.map((match) => (
                  <ListItem
                    key={match.id}
                    button
                    onClick={handleToggle(match.id)}
                    sx={{ 
                      borderRadius: 1,
                      '&:hover': { bgcolor: 'action.hover' }
                    }}
                  >
                    <ListItemIcon>
                      <Checkbox
                        edge="start"
                        checked={selected.indexOf(match.id) !== -1}
                        tabIndex={-1}
                        disableRipple
                      />
                    </ListItemIcon>
                    <ListItemIcon>
                      <SportsSoccerIcon color="primary" />
                    </ListItemIcon>
                    <ListItemText 
                      primary={getMatchLabel(match)} 
                      secondary={match.date}
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Box>
            <Typography variant="body2" color="error" sx={{ mb: 2 }}>
              {selected.length > maxMatches && `Too many matches selected (${selected.length}/${maxMatches})`}
            </Typography>
            <Button 
              variant="contained" 
              onClick={loadVisualization}
              disabled={loading || selected.length === 0 || selected.length > maxMatches}
              fullWidth
            >
              Generate Visualization
            </Button>
          </Box>
        </Paper>
      </Grid>
      
      <Grid item xs={12} md={8}>
        <VisualizationCard
          title="Prediction Confidence by Match"
          imageData={imageData}
          loading={loading}
          error={error}
          onRefresh={loadVisualization}
        />
      </Grid>
    </Grid>
  );
};

export default PredictionConfidence; 