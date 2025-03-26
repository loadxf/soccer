import React, { useState, useEffect } from 'react';
import { 
  TextField, 
  MenuItem, 
  Grid, 
  Box, 
  Paper, 
  Typography, 
  Button,
  Slider,
  InputAdornment
} from '@mui/material';
import VisualizationCard from './VisualizationCard';
import { fetchFeatureImportanceChart } from '../../utils/visualization';

/**
 * Feature importance visualization component.
 * 
 * @param {object} props - Component props
 * @param {array} props.availableModels - List of available models with ids and names
 * @returns {React.ReactElement} Feature importance component
 */
const FeatureImportance = ({ 
  availableModels = []
}) => {
  const [selectedModel, setSelectedModel] = useState(
    availableModels.length > 0 ? availableModels[0].id : ''
  );
  const [topN, setTopN] = useState(20);
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadVisualization = async () => {
    if (!selectedModel) {
      setError('Please select a model');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchFeatureImportanceChart(selectedModel, topN);
      setImageData(data);
    } catch (err) {
      console.error('Error loading feature importance visualization:', err);
      setError('Failed to load feature importance visualization. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Load visualization when component mounts or when selected model changes
  useEffect(() => {
    if (selectedModel) {
      loadVisualization();
    }
  }, [selectedModel]);

  const handleTopNChange = (event, newValue) => {
    setTopN(newValue);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2, mb: 2 }}>
          <Box display="flex" flexDirection={{ xs: 'column', sm: 'row' }} alignItems="center" gap={2}>
            <TextField
              select
              label="Select Model"
              value={selectedModel}
              onChange={handleModelChange}
              sx={{ minWidth: 200 }}
              fullWidth={false}
            >
              {availableModels.map((model) => (
                <MenuItem key={model.id} value={model.id}>
                  {model.name}
                </MenuItem>
              ))}
            </TextField>
            
            <Box sx={{ width: 300, ml: { sm: 4 } }}>
              <Typography id="top-n-slider" gutterBottom>
                Top Features
              </Typography>
              <Slider
                value={topN}
                onChange={handleTopNChange}
                aria-labelledby="top-n-slider"
                valueLabelDisplay="auto"
                min={5}
                max={50}
                marks={[
                  { value: 5, label: '5' },
                  { value: 20, label: '20' },
                  { value: 35, label: '35' },
                  { value: 50, label: '50' }
                ]}
              />
            </Box>
            
            <Button 
              variant="contained" 
              onClick={loadVisualization}
              disabled={loading || !selectedModel}
              sx={{ ml: 'auto' }}
            >
              Update
            </Button>
          </Box>
        </Paper>
      </Grid>
      
      <Grid item xs={12}>
        <VisualizationCard
          title={`Top ${topN} Feature Importances`}
          imageData={imageData}
          loading={loading}
          error={error}
          onRefresh={loadVisualization}
        />
      </Grid>
    </Grid>
  );
};

export default FeatureImportance; 