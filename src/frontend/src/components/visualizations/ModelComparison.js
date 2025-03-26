import React, { useState, useEffect } from 'react';
import { 
  FormGroup, 
  FormControlLabel, 
  Checkbox, 
  Grid, 
  Box, 
  Paper, 
  Typography, 
  Button,
  Divider,
  Stack
} from '@mui/material';
import VisualizationCard from './VisualizationCard';
import { fetchModelComparisonChart } from '../../utils/visualization';

/**
 * Model comparison visualization component.
 * 
 * @param {object} props - Component props
 * @param {array} props.availableModels - List of available models with ids and names
 * @returns {React.ReactElement} Model comparison component
 */
const ModelComparison = ({ 
  availableModels = []
}) => {
  const [selectedModels, setSelectedModels] = useState([]);
  const [imageData, setImageData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleModelToggle = (modelId) => {
    setSelectedModels(prev => {
      if (prev.includes(modelId)) {
        return prev.filter(id => id !== modelId);
      } else {
        return [...prev, modelId];
      }
    });
  };

  const loadVisualization = async () => {
    if (selectedModels.length < 2) {
      setError('Please select at least two models to compare');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await fetchModelComparisonChart(selectedModels);
      setImageData(data);
    } catch (err) {
      console.error('Error loading model comparison visualization:', err);
      setError('Failed to load model comparison visualization. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Automatically select the first two models if available
    if (availableModels.length >= 2 && selectedModels.length === 0) {
      setSelectedModels([availableModels[0].id, availableModels[1].id]);
    }
  }, [availableModels]);

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={4}>
        <Paper sx={{ p: 2, height: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Select Models to Compare
          </Typography>
          <Divider sx={{ mb: 2 }} />
          
          <FormGroup>
            {availableModels.map((model) => (
              <FormControlLabel
                key={model.id}
                control={
                  <Checkbox 
                    checked={selectedModels.includes(model.id)}
                    onChange={() => handleModelToggle(model.id)}
                  />
                }
                label={model.name}
              />
            ))}
          </FormGroup>
          
          <Box mt={2}>
            <Stack direction="row" spacing={1}>
              <Button 
                variant="outlined" 
                size="small"
                onClick={() => setSelectedModels(availableModels.map(m => m.id))}
              >
                Select All
              </Button>
              <Button 
                variant="outlined" 
                size="small"
                onClick={() => setSelectedModels([])}
              >
                Clear All
              </Button>
            </Stack>
          </Box>
          
          <Divider sx={{ my: 2 }} />
          
          <Button 
            variant="contained" 
            onClick={loadVisualization}
            disabled={loading || selectedModels.length < 2}
            fullWidth
          >
            Compare Models
          </Button>
        </Paper>
      </Grid>
      
      <Grid item xs={12} md={8}>
        <VisualizationCard
          title="Model Performance Comparison"
          imageData={imageData}
          loading={loading}
          error={error}
          onRefresh={loadVisualization}
        />
      </Grid>
    </Grid>
  );
};

export default ModelComparison; 