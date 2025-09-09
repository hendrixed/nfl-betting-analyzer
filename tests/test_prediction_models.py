"""
Deterministic unit tests for NFL Prediction Models
Tests model training, prediction, and ensemble functionality with reproducible results.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil

from core.models.prediction_models import (
    NFLPredictionModel, 
    EnsembleModel, 
    NFLPredictionEngine,
    ModelConfig, 
    ModelPerformance, 
    PredictionResult
)
from core.models.feature_engineering import NFLFeatureEngineer, ModelFeatures


@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for model storage"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_feature_engineer():
    """Create a mock feature engineer"""
    engineer = Mock(spec=NFLFeatureEngineer)
    engineer.select_features.return_value = Mock(spec=ModelFeatures)
    return engineer


@pytest.fixture
def sample_model_features():
    """Create sample model features for testing"""
    np.random.seed(42)  # Deterministic results
    
    n_samples = 100
    n_features = 10
    
    X_train = np.random.randn(n_samples, n_features)
    X_test = np.random.randn(20, n_features)
    y_train = np.random.randn(n_samples) * 5 + 15  # Fantasy points range
    y_test = np.random.randn(20) * 5 + 15
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    return ModelFeatures(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=Mock(),
        target_name='fantasy_points_ppr'
    )


@pytest.fixture
def sample_model_config():
    """Create a sample model configuration"""
    return ModelConfig(
        model_type='random_forest',
        target_stat='fantasy_points_ppr',
        hyperparameters={
            'n_estimators': 50,  # Smaller for faster tests
            'max_depth': 5,
            'random_state': 42
        },
        feature_selection_k=8,
        cv_folds=3,  # Smaller for faster tests
        random_state=42
    )


class TestModelConfig:
    """Test model configuration functionality"""
    
    def test_model_config_creation(self):
        """Test creating a model configuration"""
        config = ModelConfig(
            model_type='lightgbm',
            target_stat='receiving_yards',
            hyperparameters={'n_estimators': 100},
            feature_selection_k=25
        )
        
        assert config.model_type == 'lightgbm'
        assert config.target_stat == 'receiving_yards'
        assert config.hyperparameters['n_estimators'] == 100
        assert config.feature_selection_k == 25
        assert config.cv_folds == 5  # Default value
        assert config.random_state == 42  # Default value


class TestNFLPredictionModel:
    """Test individual NFL prediction models"""
    
    def test_model_initialization(self, mock_feature_engineer, sample_model_config):
        """Test model initialization with different types"""
        model = NFLPredictionModel(sample_model_config, mock_feature_engineer)
        
        assert model.config == sample_model_config
        assert model.feature_engineer == mock_feature_engineer
        assert model.model is not None
        assert not model.is_trained
        assert model.performance_metrics is None
    
    def test_model_initialization_unsupported_type(self, mock_feature_engineer):
        """Test error handling for unsupported model types"""
        config = ModelConfig(
            model_type='unsupported_model',
            target_stat='fantasy_points_ppr',
            hyperparameters={}
        )
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            NFLPredictionModel(config, mock_feature_engineer)
    
    def test_model_training(self, mock_feature_engineer, sample_model_config, sample_model_features):
        """Test model training functionality"""
        # Mock feature selection
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        model = NFLPredictionModel(sample_model_config, mock_feature_engineer)
        performance = model.train(sample_model_features)
        
        # Verify training completed
        assert model.is_trained
        assert isinstance(performance, ModelPerformance)
        assert performance.rmse > 0
        assert performance.mae > 0
        assert -1 <= performance.r2 <= 1  # R² can be negative for poor models
        assert performance.cv_score > 0
        assert performance.model_name == f"{sample_model_config.model_type}_{sample_model_config.target_stat}"
        
        # Verify feature names are stored
        assert model.feature_names == sample_model_features.feature_names
    
    def test_model_prediction(self, mock_feature_engineer, sample_model_config, sample_model_features):
        """Test model prediction functionality"""
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        model = NFLPredictionModel(sample_model_config, mock_feature_engineer)
        model.train(sample_model_features)
        
        # Test prediction
        test_features = np.random.randn(5, sample_model_features.X_train.shape[1])
        predictions, confidence_intervals = model.predict(test_features)
        
        assert len(predictions) == 5
        assert confidence_intervals.shape == (5, 2)
        assert all(ci[0] <= pred <= ci[1] for pred, ci in zip(predictions, confidence_intervals))
    
    def test_prediction_without_training(self, mock_feature_engineer, sample_model_config):
        """Test error handling when predicting without training"""
        model = NFLPredictionModel(sample_model_config, mock_feature_engineer)
        
        test_features = np.random.randn(1, 10)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(test_features)
    
    def test_feature_importance_extraction(self, mock_feature_engineer, sample_model_config, sample_model_features):
        """Test feature importance extraction"""
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        model = NFLPredictionModel(sample_model_config, mock_feature_engineer)
        model.train(sample_model_features)
        
        importance = model._get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(sample_model_features.feature_names)
        assert all(name in importance for name in sample_model_features.feature_names)
        assert all(isinstance(val, (int, float, np.number)) for val in importance.values())
    
    def test_model_save_load(self, mock_feature_engineer, sample_model_config, sample_model_features, temp_models_dir):
        """Test model saving and loading"""
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        # Train and save model
        model = NFLPredictionModel(sample_model_config, mock_feature_engineer)
        performance = model.train(sample_model_features)
        
        model_path = temp_models_dir / "test_model.joblib"
        model.save_model(model_path)
        
        assert model_path.exists()
        
        # Load model
        loaded_model = NFLPredictionModel.load_model(model_path, mock_feature_engineer)
        
        assert loaded_model.is_trained
        assert loaded_model.config.model_type == sample_model_config.model_type
        assert loaded_model.feature_names == sample_model_features.feature_names
        assert loaded_model.performance_metrics.rmse == performance.rmse
    
    def test_save_untrained_model_error(self, mock_feature_engineer, sample_model_config, temp_models_dir):
        """Test error when saving untrained model"""
        model = NFLPredictionModel(sample_model_config, mock_feature_engineer)
        
        model_path = temp_models_dir / "test_model.joblib"
        
        with pytest.raises(ValueError, match="Cannot save untrained model"):
            model.save_model(model_path)


class TestEnsembleModel:
    """Test ensemble model functionality"""
    
    def test_ensemble_initialization(self, mock_feature_engineer, sample_model_config, sample_model_features):
        """Test ensemble model initialization"""
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        # Create multiple models
        models = []
        for i in range(3):
            config = ModelConfig(
                model_type='random_forest',
                target_stat='fantasy_points_ppr',
                hyperparameters={'n_estimators': 10, 'random_state': 42 + i}
            )
            model = NFLPredictionModel(config, mock_feature_engineer)
            model.train(sample_model_features)
            models.append(model)
        
        # Create ensemble
        ensemble = EnsembleModel(models)
        
        assert len(ensemble.models) == 3
        assert len(ensemble.weights) == 3
        assert abs(sum(ensemble.weights) - 1.0) < 1e-10  # Weights sum to 1
    
    def test_ensemble_with_custom_weights(self, mock_feature_engineer, sample_model_config, sample_model_features):
        """Test ensemble with custom weights"""
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        # Create models
        models = []
        for i in range(2):
            config = ModelConfig(
                model_type='random_forest',
                target_stat='fantasy_points_ppr',
                hyperparameters={'n_estimators': 10, 'random_state': 42 + i}
            )
            model = NFLPredictionModel(config, mock_feature_engineer)
            model.train(sample_model_features)
            models.append(model)
        
        # Custom weights
        custom_weights = [0.7, 0.3]
        ensemble = EnsembleModel(models, custom_weights)
        
        assert ensemble.weights == custom_weights
    
    def test_ensemble_weight_mismatch_error(self, mock_feature_engineer, sample_model_config, sample_model_features):
        """Test error when weights don't match number of models"""
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        model = NFLPredictionModel(sample_model_config, mock_feature_engineer)
        model.train(sample_model_features)
        
        with pytest.raises(ValueError, match="Number of weights must match"):
            EnsembleModel([model], [0.5, 0.5])  # 1 model, 2 weights
    
    def test_ensemble_prediction(self, mock_feature_engineer, sample_model_config, sample_model_features):
        """Test ensemble prediction functionality"""
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        # Create ensemble
        models = []
        for i in range(2):
            config = ModelConfig(
                model_type='random_forest',
                target_stat='fantasy_points_ppr',
                hyperparameters={'n_estimators': 10, 'random_state': 42 + i}
            )
            model = NFLPredictionModel(config, mock_feature_engineer)
            model.train(sample_model_features)
            models.append(model)
        
        ensemble = EnsembleModel(models)
        
        # Test prediction
        test_features = np.random.randn(3, sample_model_features.X_train.shape[1])
        predictions, confidence_intervals = ensemble.predict(test_features)
        
        assert len(predictions) == 3
        assert confidence_intervals.shape == (3, 2)
        
        # Verify predictions are within confidence intervals
        for pred, ci in zip(predictions, confidence_intervals):
            assert ci[0] <= pred <= ci[1]
    
    def test_ensemble_performance_summary(self, mock_feature_engineer, sample_model_config, sample_model_features):
        """Test ensemble performance summary"""
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        # Create ensemble
        models = []
        for i in range(2):
            config = ModelConfig(
                model_type='random_forest',
                target_stat='fantasy_points_ppr',
                hyperparameters={'n_estimators': 10, 'random_state': 42 + i}
            )
            model = NFLPredictionModel(config, mock_feature_engineer)
            model.train(sample_model_features)
            models.append(model)
        
        ensemble = EnsembleModel(models)
        performance_summary = ensemble.get_model_performance()
        
        assert len(performance_summary) == 2
        for model_name, performance in performance_summary.items():
            assert isinstance(performance, ModelPerformance)
            assert performance.rmse > 0
            assert performance.mae > 0


class TestNFLPredictionEngine:
    """Test the main prediction engine"""
    
    def test_engine_initialization(self, temp_models_dir):
        """Test prediction engine initialization"""
        mock_session = Mock()
        
        with patch('core.models.prediction_models.NFLFeatureEngineer') as mock_fe:
            engine = NFLPredictionEngine(mock_session, temp_models_dir)
            
            assert engine.session == mock_session
            assert engine.models_dir == temp_models_dir
            assert temp_models_dir.exists()
            assert len(engine.trained_models) == 0
            assert len(engine.ensemble_models) == 0
    
    def test_create_model_configs(self, temp_models_dir):
        """Test model configuration creation"""
        mock_session = Mock()
        
        with patch('core.models.prediction_models.NFLFeatureEngineer'):
            engine = NFLPredictionEngine(mock_session, temp_models_dir)
            configs = engine.create_model_configs()
            
            # Verify configs for different stats
            expected_stats = [
                'fantasy_points_ppr', 'receiving_yards', 'rushing_yards',
                'passing_yards', 'receiving_touchdowns', 'rushing_touchdowns',
                'passing_touchdowns'
            ]
            
            for stat in expected_stats:
                assert stat in configs
                assert len(configs[stat]) >= 3  # At least 3 model types
                
                for config in configs[stat]:
                    assert isinstance(config, ModelConfig)
                    assert config.target_stat == stat
                    assert config.model_type in ['lightgbm', 'xgboost', 'random_forest']
    
    @pytest.mark.asyncio
    async def test_train_models_basic(self, temp_models_dir):
        """Test basic model training functionality"""
        mock_session = Mock()
        
        # Create sample training data
        feature_data = []
        np.random.seed(42)
        
        for i in range(50):  # Small dataset for testing
            feature_data.append({
                'receiving_yards_avg_3g': np.random.uniform(50, 120),
                'targets_avg_3g': np.random.uniform(5, 12),
                'snap_percentage_avg_3g': np.random.uniform(0.6, 0.9),
                'is_home': np.random.choice([0.0, 1.0]),
                'temperature': np.random.uniform(40, 80),
                'fantasy_points_ppr': np.random.uniform(8, 25)
            })
        
        with patch('core.models.prediction_models.NFLFeatureEngineer') as mock_fe_class:
            mock_fe = Mock()
            mock_fe_class.return_value = mock_fe
            
            # Mock feature engineering
            mock_features = ModelFeatures(
                X_train=np.random.randn(40, 5),
                X_test=np.random.randn(10, 5),
                y_train=np.random.randn(40) * 5 + 15,
                y_test=np.random.randn(10) * 5 + 15,
                feature_names=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
                scaler=Mock(),
                target_name='fantasy_points_ppr'
            )
            mock_fe.prepare_training_data.return_value = mock_features
            mock_fe.select_features.return_value = mock_features
            
            engine = NFLPredictionEngine(mock_session, temp_models_dir)
            
            # Train models for just one stat to speed up test
            limited_configs = {'fantasy_points_ppr': engine.create_model_configs()['fantasy_points_ppr'][:1]}
            
            with patch.object(engine, 'create_model_configs', return_value=limited_configs):
                performance_results = await engine.train_models(feature_data)
                
                assert len(performance_results) >= 1
                assert 'fantasy_points_ppr' in engine.ensemble_models
                
                for model_name, performance in performance_results.items():
                    assert isinstance(performance, ModelPerformance)
                    assert performance.rmse > 0


class TestDeterministicBehavior:
    """Test that models produce deterministic results"""
    
    def test_deterministic_training(self, mock_feature_engineer, sample_model_features):
        """Test that model training produces consistent results"""
        config = ModelConfig(
            model_type='random_forest',
            target_stat='fantasy_points_ppr',
            hyperparameters={
                'n_estimators': 10,
                'max_depth': 3,
                'random_state': 42
            },
            random_state=42
        )
        
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        # Train two identical models
        model1 = NFLPredictionModel(config, mock_feature_engineer)
        performance1 = model1.train(sample_model_features)
        
        model2 = NFLPredictionModel(config, mock_feature_engineer)
        performance2 = model2.train(sample_model_features)
        
        # Results should be identical
        assert abs(performance1.rmse - performance2.rmse) < 1e-10
        assert abs(performance1.mae - performance2.mae) < 1e-10
        assert abs(performance1.r2 - performance2.r2) < 1e-10
    
    def test_deterministic_predictions(self, mock_feature_engineer, sample_model_features):
        """Test that predictions are deterministic"""
        config = ModelConfig(
            model_type='random_forest',
            target_stat='fantasy_points_ppr',
            hyperparameters={
                'n_estimators': 10,
                'random_state': 42
            },
            random_state=42
        )
        
        mock_feature_engineer.select_features.return_value = sample_model_features
        
        model = NFLPredictionModel(config, mock_feature_engineer)
        model.train(sample_model_features)
        
        test_features = np.random.RandomState(42).randn(5, sample_model_features.X_train.shape[1])
        
        # Make predictions twice
        pred1, ci1 = model.predict(test_features)
        pred2, ci2 = model.predict(test_features)
        
        # Results should be identical
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(ci1, ci2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
