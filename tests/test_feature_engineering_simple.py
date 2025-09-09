"""
Simplified unit tests for NFL Feature Engineering module
Focus on core functionality with minimal mocking complexity.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from core.models.feature_engineering import NFLFeatureEngineer, ModelFeatures


class TestFeatureEngineeringCore:
    """Test core feature engineering functionality without complex database mocks"""
    
    def test_prepare_training_data_basic(self):
        """Test basic training data preparation"""
        
        # Create mock feature engineer
        mock_session = Mock()
        mock_cache = Mock()
        engineer = NFLFeatureEngineer(mock_session, mock_cache)
        
        # Sample feature data
        feature_data = [
            {
                'receiving_yards_avg_3g': 85.0,
                'targets_avg_3g': 8.5,
                'snap_percentage_avg_3g': 0.75,
                'is_home': 1.0,
                'temperature': 70.0,
                'fantasy_points_ppr': 18.5
            },
            {
                'receiving_yards_avg_3g': 92.0,
                'targets_avg_3g': 9.2,
                'snap_percentage_avg_3g': 0.82,
                'is_home': 0.0,
                'temperature': 65.0,
                'fantasy_points_ppr': 21.2
            },
            {
                'receiving_yards_avg_3g': 78.0,
                'targets_avg_3g': 7.8,
                'snap_percentage_avg_3g': 0.68,
                'is_home': 1.0,
                'temperature': 75.0,
                'fantasy_points_ppr': 16.8
            }
        ]
        
        model_features = engineer.prepare_training_data(feature_data)
        
        # Verify structure
        assert isinstance(model_features, ModelFeatures)
        assert model_features.X_train.shape[0] > 0
        assert model_features.X_test.shape[0] > 0
        assert len(model_features.feature_names) == 5
        assert model_features.target_name == 'fantasy_points_ppr'
        
        # Verify scaling
        train_means = np.mean(model_features.X_train, axis=0)
        assert np.allclose(train_means, 0, atol=1e-10)
    
    def test_feature_selection_basic(self):
        """Test feature selection functionality"""
        
        mock_session = Mock()
        mock_cache = Mock()
        engineer = NFLFeatureEngineer(mock_session, mock_cache)
        
        # Create sample model features
        np.random.seed(42)
        n_samples = 50
        n_features = 15
        
        X_train = np.random.randn(n_samples, n_features)
        X_test = np.random.randn(10, n_features)
        y_train = np.random.randn(n_samples)
        y_test = np.random.randn(10)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        model_features = ModelFeatures(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            scaler=Mock(),
            target_name='target'
        )
        
        # Select top 8 features
        selected_features = engineer.select_features(model_features, k=8)
        
        # Verify selection
        assert selected_features.X_train.shape[1] == 8
        assert selected_features.X_test.shape[1] == 8
        assert len(selected_features.feature_names) == 8
    
    def test_position_weights_configuration(self):
        """Test position-specific feature weights are properly configured"""
        
        mock_session = Mock()
        mock_cache = Mock()
        engineer = NFLFeatureEngineer(mock_session, mock_cache)
        
        # Check all positions have weights
        expected_positions = ['QB', 'RB', 'WR', 'TE']
        for position in expected_positions:
            assert position in engineer.position_weights
            weights = engineer.position_weights[position]
            
            # Verify weight categories
            assert 'passing' in weights
            assert 'rushing' in weights
            assert 'receiving' in weights
            assert 'team_context' in weights
            
            # Verify weights sum to 1.0
            assert abs(sum(weights.values()) - 1.0) < 1e-10
    
    def test_lookback_windows_configuration(self):
        """Test lookback windows are properly configured"""
        
        mock_session = Mock()
        mock_cache = Mock()
        engineer = NFLFeatureEngineer(mock_session, mock_cache)
        
        # Verify lookback windows
        assert len(engineer.lookback_windows) == 4
        assert 3 in engineer.lookback_windows
        assert 5 in engineer.lookback_windows
        assert 8 in engineer.lookback_windows
        assert 16 in engineer.lookback_windows
    
    def test_error_handling_empty_data(self):
        """Test error handling with empty feature data"""
        
        mock_session = Mock()
        mock_cache = Mock()
        engineer = NFLFeatureEngineer(mock_session, mock_cache)
        
        with pytest.raises(ValueError, match="No feature data provided"):
            engineer.prepare_training_data([])
    
    def test_error_handling_missing_target(self):
        """Test error handling when target column is missing"""
        
        mock_session = Mock()
        mock_cache = Mock()
        engineer = NFLFeatureEngineer(mock_session, mock_cache)
        
        feature_data = [{'feature1': 1.0, 'feature2': 2.0}]
        
        with pytest.raises(ValueError, match="Target column .* not found"):
            engineer.prepare_training_data(feature_data, target_column='missing_target')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
