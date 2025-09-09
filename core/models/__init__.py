"""
NFL Models Package
Advanced machine learning models for NFL betting predictions.
"""

from .feature_engineering import NFLFeatureEngineer, ModelFeatures  # FeatureSet is internal
from .prediction_models import (
    NFLPredictionModel,
    EnsembleModel as ModelEnsemble,
    NFLPredictionEngine,
    ModelConfig,
    ModelPerformance,
    PredictionResult,
)

__all__ = [
    'NFLFeatureEngineer',
    'ModelFeatures',
    'NFLPredictionModel',
    'NFLPredictionEngine',
    'ModelEnsemble',
    'ModelConfig',
    'ModelPerformance',
    'PredictionResult'
]
