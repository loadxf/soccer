"""
Tests for the model explainability module.
"""

import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from src.models.explainability import ModelExplainer, generate_model_explanations
from src.models.baseline import BaselineMatchPredictor


class TestModelExplainer:
    """Test suite for the ModelExplainer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3]])
        model.predict.return_value = np.array([0, 1])
        model.model_type = "baseline"
        model.dataset_name = "test_dataset"
        model.feature_type = "test_features"
        return model
    
    @pytest.fixture
    def test_data(self):
        """Create test data for explanations."""
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 3, 10)
        feature_names = [f"feature_{i}" for i in range(5)]
        class_names = ["Home Win", "Draw", "Away Win"]
        return X, y, feature_names, class_names
    
    @pytest.fixture
    def explainer(self, mock_model, test_data):
        """Create a model explainer instance for testing."""
        _, _, feature_names, class_names = test_data
        return ModelExplainer(mock_model, feature_names, class_names)
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_init(self, mock_model, test_data):
        """Test ModelExplainer initialization."""
        _, _, feature_names, class_names = test_data
        explainer = ModelExplainer(mock_model, feature_names, class_names)
        
        assert explainer.model == mock_model
        assert explainer.feature_names == feature_names
        assert explainer.class_names == class_names
        assert explainer.model_type == "baseline"  # Should detect the mock model type
        assert callable(explainer.predict_fn)
        assert explainer.explainers == {}  # Should start with empty explainers dict
    
    def test_determine_model_type(self, mock_model, test_data):
        """Test model type determination."""
        _, _, feature_names, class_names = test_data
        
        # Test with baseline model
        mock_model.__class__.__name__ = "BaselineMatchPredictor"
        mock_baseline = MagicMock(spec=BaselineMatchPredictor)
        explainer = ModelExplainer(mock_baseline, feature_names, class_names)
        assert explainer._determine_model_type() == "baseline"
        
        # Test with sklearn-like model
        mock_sklearn = MagicMock()
        mock_sklearn.predict_proba = MagicMock()
        explainer = ModelExplainer(mock_sklearn, feature_names, class_names)
        assert explainer._determine_model_type() in ["sklearn", "baseline"]
        
        # Test with unknown model
        mock_unknown = MagicMock()
        delattr(mock_unknown, "predict_proba")
        explainer = ModelExplainer(mock_unknown, feature_names, class_names)
        assert explainer._determine_model_type() == "unknown"
    
    @patch("src.models.explainability.shap")
    def test_explain_with_shap(self, mock_shap, explainer, test_data, temp_dir):
        """Test SHAP explanation generation."""
        X, _, _, _ = test_data
        
        # Mock SHAP explainer and values
        mock_shap_explainer = MagicMock()
        mock_shap_explainer.shap_values.return_value = [np.random.rand(1, 5) for _ in range(3)]
        mock_shap_explainer.expected_value = [0.5, 0.3, 0.2]
        mock_shap.KernelExplainer.return_value = mock_shap_explainer
        
        # Patch matplotlib to avoid actual plotting
        with patch("matplotlib.pyplot.figure"), \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"), \
             patch("src.models.explainability.PLOTS_DIR", temp_dir):
            
            # Test single sample explanation
            result = explainer.explain_with_shap(X, sample_idx=0)
            
            assert result["method"] == "shap"
            assert "timestamp" in result
            assert "plots" in result
            assert "summary_plot" in result["plots"]
            assert "metadata" in result
            
            # Test all samples explanation
            result = explainer.explain_with_shap(X)
            
            assert result["method"] == "shap"
            assert "plots" in result
            assert "summary_plot" in result["plots"]
    
    @patch("src.models.explainability.lime.lime_tabular.LimeTabularExplainer")
    def test_explain_with_lime(self, mock_lime, explainer, test_data, temp_dir):
        """Test LIME explanation generation."""
        X, _, _, _ = test_data
        
        # Mock LIME explainer and explanation
        mock_explanation = MagicMock()
        mock_explanation.as_list.return_value = [("feature_0", 0.5), ("feature_1", -0.3)]
        mock_lime_explainer = MagicMock()
        mock_lime_explainer.explain_instance.return_value = mock_explanation
        mock_lime.return_value = mock_lime_explainer
        
        # Patch matplotlib and file operations
        with patch("matplotlib.pyplot.figure"), \
             patch("src.models.explainability.PLOTS_DIR", temp_dir):
            
            result = explainer.explain_with_lime(X, sample_idx=0)
            
            assert result["method"] == "lime"
            assert "timestamp" in result
            assert "explanation" in result
            assert "plots" in result
            assert "lime_plot" in result["plots"]
            assert "metadata" in result
            assert "sample_idx" in result["metadata"]
    
    @patch("src.models.explainability.pdp")
    def test_explain_with_pdp(self, mock_pdp, explainer, test_data, temp_dir):
        """Test PDP explanation generation."""
        X, _, _, _ = test_data
        
        # Mock PDP isolation and plot
        mock_pdp_isolate = MagicMock()
        mock_pdp_isolate.pdp.mean.return_value = 0.3
        mock_pdp.pdp_isolate.return_value = mock_pdp_isolate
        
        # Patch matplotlib
        with patch("matplotlib.pyplot.figure"), \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"), \
             patch("src.models.explainability.PLOTS_DIR", temp_dir):
            
            result = explainer.explain_with_pdp(X, feature_idx=0)
            
            assert result["method"] == "pdp"
            assert "timestamp" in result
            assert "feature" in result
            assert result["feature"] == "feature_0"
            assert "plots" in result
            assert "pdp_plot" in result["plots"]
    
    @patch("src.models.explainability.permutation_importance")
    def test_explain_with_permutation_importance(self, mock_perm_importance, explainer, test_data, temp_dir):
        """Test permutation importance explanation."""
        X, y, _, _ = test_data
        
        # Mock permutation importance
        perm_result = MagicMock()
        perm_result.importances_mean = np.array([0.3, 0.2, 0.1, 0.05, 0.01])
        perm_result.importances_std = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
        mock_perm_importance.return_value = perm_result
        
        # Patch matplotlib and file operations
        with patch("matplotlib.pyplot.figure"), \
             patch("matplotlib.pyplot.savefig"), \
             patch("matplotlib.pyplot.close"), \
             patch("pandas.DataFrame.to_csv"), \
             patch("src.models.explainability.PLOTS_DIR", temp_dir), \
             patch("src.models.explainability.EXPLAINABILITY_DIR", temp_dir):
            
            result = explainer.explain_with_permutation_importance(X, y)
            
            assert result["method"] == "permutation_importance"
            assert "timestamp" in result
            assert "importance" in result
            assert isinstance(result["importance"], list)
            assert "plots" in result
            assert "importance_plot" in result["plots"]
    
    def test_explain_prediction(self, explainer, test_data):
        """Test prediction explanation for a single sample."""
        X, _, _, _ = test_data
        sample = X[0]
        
        # Mock individual explainer methods
        explainer.explain_with_shap = MagicMock(return_value={"method": "shap"})
        explainer.explain_with_lime = MagicMock(return_value={"method": "lime"})
        explainer.explain_with_anchor = MagicMock(return_value={"method": "anchor"})
        explainer.explain_with_counterfactual = MagicMock(return_value={"method": "counterfactual"})
        
        with patch("json.dump"):
            result = explainer.explain_prediction(sample, methods=["shap", "lime"])
            
            assert "timestamp" in result
            assert "prediction" in result
            assert "explanations" in result
            assert "shap" in result["explanations"]
            assert "lime" in result["explanations"]
            assert "anchor" not in result["explanations"]  # Not requested
            
            # Check that the explainer methods were called
            explainer.explain_with_shap.assert_called_once()
            explainer.explain_with_lime.assert_called_once()
            explainer.explain_with_anchor.assert_not_called()
    
    def test_generate_global_explanations(self, explainer, test_data):
        """Test generation of global explanations."""
        X, y, _, _ = test_data
        
        # Mock individual explainer methods
        explainer.explain_with_shap = MagicMock(return_value={"method": "shap"})
        explainer.explain_with_permutation_importance = MagicMock(return_value={"method": "permutation"})
        
        with patch("json.dump"):
            result = explainer.generate_global_explanations(X, y, methods=["shap", "permutation"])
            
            assert "timestamp" in result
            assert "explanations" in result
            assert "shap" in result["explanations"]
            assert "permutation" in result["explanations"]
            
            # Check that the explainer methods were called
            explainer.explain_with_shap.assert_called_once()
            explainer.explain_with_permutation_importance.assert_called_once()


@patch("src.models.explainability.ModelExplainer")
@patch("src.models.baseline.BaselineMatchPredictor.load")
@patch("src.models.explainability.load_feature_data")
def test_generate_model_explanations(mock_load_data, mock_load_model, mock_explainer, temp_dir):
    """Test the generate_model_explanations function."""
    # Mock model and data
    mock_model = MagicMock()
    mock_model.dataset_name = "test_dataset"
    mock_model.feature_type = "test_features"
    mock_model.process_data.return_value = (np.random.rand(10, 5), np.random.randint(0, 3, 10))
    mock_load_model.return_value = mock_model
    
    # Mock DataFrame
    mock_df = pd.DataFrame({
        "feature_0": np.random.rand(10),
        "feature_1": np.random.rand(10),
        "feature_2": np.random.rand(10),
        "result": np.random.randint(0, 3, 10)
    })
    mock_load_data.return_value = mock_df
    
    # Mock explainer
    mock_explainer_instance = MagicMock()
    mock_explainer_instance.generate_global_explanations.return_value = {"explanations": {}}
    mock_explainer_instance.explain_prediction.return_value = {"method": "combined"}
    mock_explainer.return_value = mock_explainer_instance
    
    with patch("json.dump"), \
         patch("src.models.explainability.EXPLAINABILITY_DIR", temp_dir):
        
        result = generate_model_explanations(
            model_path="dummy_path",
            methods=["shap", "lime"],
            num_samples=2
        )
        
        assert "model_path" in result
        assert "dataset" in result
        assert "feature_type" in result
        assert "timestamp" in result
        assert "global_explanations" in result
        assert "sample_explanations" in result
        assert "metadata" in result
        
        # Check that correct methods were called
        mock_load_model.assert_called_once_with("dummy_path")
        mock_load_data.assert_called_once()
        mock_model.process_data.assert_called_once()
        mock_explainer.assert_called_once()
        mock_explainer_instance.generate_global_explanations.assert_called_once()
        assert mock_explainer_instance.explain_prediction.call_count == 2  # For each sample 