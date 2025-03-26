import React, { useState, useEffect } from 'react';
import { TextField, MenuItem, Grid, Box, Paper, Typography, Button } from '@mui/material';
import VisualizationCard from './VisualizationCard';
import { fetchTeamPerformanceChart } from '../../utils/visualization';

/**
 * Team Performance visualization component.
 * 
 * @param {object} props - Component props
 * @param {array} props.availableTeams - List of available teams
 * @param {string} props.defaultTeam - Default team to display
 * @param {number} props.lastNMatches - Number of last matches to display
 * @returns {React.ReactElement} Team Performance component
 */
const TeamPerformance = ({ 
  availableTeams = [], 
  defaultTeam = null,
  lastNMatches = 10
}) => {
  const [selectedTeam, setSelectedTeam] = useState(defaultTeam || (availableTeams.length > 0 ? availableTeams[0] : ''));
  const [matches, setMatches] = useState(lastNMatches);
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadVisualization = async () => {
    if (!selectedTeam) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchTeamPerformanceChart(selectedTeam, matches);
      setImageData(data);
    } catch (err) {
      console.error('Error loading team performance visualization:', err);
      setError('Failed to load team performance visualization. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedTeam) {
      loadVisualization();
    }
  }, [selectedTeam, matches]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2, mb: 2 }}>
          <Box display="flex" flexDirection={{ xs: 'column', sm: 'row' }} alignItems={{ sm: 'center' }} gap={2}>
            <TextField
              select
              label="Select Team"
              value={selectedTeam}
              onChange={(e) => setSelectedTeam(e.target.value)}
              sx={{ minWidth: 200 }}
              fullWidth={false}
            >
              {availableTeams.map((team) => (
                <MenuItem key={team} value={team}>
                  {team}
                </MenuItem>
              ))}
            </TextField>
            
            <TextField
              type="number"
              label="Last N Matches"
              value={matches}
              onChange={(e) => setMatches(Math.max(1, Math.min(50, parseInt(e.target.value) || 10)))}
              inputProps={{ min: 1, max: 50 }}
              sx={{ width: '120px' }}
            />
            
            <Button 
              variant="contained" 
              onClick={loadVisualization}
              disabled={loading || !selectedTeam}
              sx={{ ml: 'auto' }}
            >
              Update
            </Button>
          </Box>
        </Paper>
      </Grid>
      
      <Grid item xs={12}>
        <VisualizationCard
          title={`${selectedTeam} Performance - Last ${matches} Matches`}
          imageData={imageData}
          loading={loading}
          error={error}
          onRefresh={loadVisualization}
        />
      </Grid>
    </Grid>
  );
};

export default TeamPerformance; 