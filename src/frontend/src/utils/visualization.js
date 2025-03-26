/**
 * Visualization utility functions for the frontend.
 * These functions handle API calls for visualizations and process the returned data.
 */

import axios from 'axios';
import { API_BASE_URL } from '../config';

/**
 * Fetch team performance visualization.
 * 
 * @param {string} teamName - Name of the team
 * @param {number} lastNMatches - Number of last matches to include
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchTeamPerformanceChart = async (teamName, lastNMatches = 10) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/visualizations/team-performance`, {
      params: { team_name: teamName, last_n_matches: lastNMatches }
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching team performance chart:', error);
    throw error;
  }
};

/**
 * Fetch prediction confidence visualization.
 * 
 * @param {Array} matchIds - Array of match IDs to include in visualization
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchPredictionConfidenceChart = async (matchIds) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/visualizations/prediction-confidence`, {
      match_ids: matchIds
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching prediction confidence chart:', error);
    throw error;
  }
};

/**
 * Fetch model comparison visualization.
 * 
 * @param {Array} modelIds - Array of model IDs to compare
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchModelComparisonChart = async (modelIds) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/visualizations/model-comparison`, {
      model_ids: modelIds
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching model comparison chart:', error);
    throw error;
  }
};

/**
 * Fetch feature importance visualization.
 * 
 * @param {string} modelId - ID of the model
 * @param {number} topN - Number of top features to display
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchFeatureImportanceChart = async (modelId, topN = 20) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/visualizations/feature-importance`, {
      params: { model_id: modelId, top_n: topN }
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching feature importance chart:', error);
    throw error;
  }
};

/**
 * Fetch confusion matrix visualization.
 * 
 * @param {string} modelId - ID of the model
 * @param {string} datasetId - ID of the dataset
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchConfusionMatrixChart = async (modelId, datasetId) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/visualizations/confusion-matrix`, {
      params: { model_id: modelId, dataset_id: datasetId }
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching confusion matrix chart:', error);
    throw error;
  }
};

/**
 * Fetch ROC curve visualization.
 * 
 * @param {string} modelId - ID of the model
 * @param {string} datasetId - ID of the dataset
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchRocCurveChart = async (modelId, datasetId) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/visualizations/roc-curve`, {
      params: { model_id: modelId, dataset_id: datasetId }
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching ROC curve chart:', error);
    throw error;
  }
};

/**
 * Fetch prediction history visualization.
 * 
 * @param {string} metricName - Name of the metric to plot
 * @param {string} startDate - Start date in ISO format
 * @param {string} endDate - End date in ISO format
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchPredictionHistoryChart = async (
  metricName = 'accuracy', 
  startDate = null, 
  endDate = null
) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/visualizations/prediction-history`, {
      params: { 
        metric_name: metricName,
        start_date: startDate,
        end_date: endDate
      }
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching prediction history chart:', error);
    throw error;
  }
};

/**
 * Fetch player performance visualization.
 * 
 * @param {string} playerName - Name of the player
 * @param {Array} metrics - Array of metrics to include
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchPlayerPerformanceChart = async (
  playerName, 
  metrics = ['goals', 'assists', 'minutes_played']
) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/visualizations/player-performance`, {
      params: { 
        player_name: playerName,
        metrics: metrics.join(',')
      }
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching player performance chart:', error);
    throw error;
  }
};

/**
 * Fetch league standings visualization.
 * 
 * @param {string} season - Season identifier (e.g., "2022-2023")
 * @param {number} topN - Number of top teams to display
 * @returns {Promise<string>} - Promise resolving to base64 encoded image
 */
export const fetchLeagueStandingsChart = async (season, topN = 10) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/visualizations/league-standings`, {
      params: { 
        season: season,
        top_n: topN
      }
    });
    return response.data.image;
  } catch (error) {
    console.error('Error fetching league standings chart:', error);
    throw error;
  }
}; 