"""
NFL Prediction Models
Advanced machine learning models for accurate NFL betting predictions.
Tasks 156-174: Model development, ensemble methods, prediction engine.
"""

import logging
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

from sqlalchemy.orm import Session
from ..database_models import get_db_session
from .feature_engineering import NFLFeatureEngineer, ModelFeatures

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    rmse: float
    mae: float
    r2: float
    cv_score: float
    feature_importance: Dict[str, float]
    model_name: str


@dataclass
class PredictionResult:
    """Individual prediction result"""
    player_id: str
    player_name: str
    position: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_confidence: float
    contributing_factors: Dict[str, float]
    model_used: str


@dataclass
class ModelEnsemble:
    """Ensemble of trained models"""
    models: Dict[str, Any]
    weights: Dict[str, float]
    performance: Dict[str, ModelPerformance]
    feature_names: List[str]
    target_stat: str
    position: str


class NFLPredictionModels:
    """Advanced ML models for NFL predictions"""
    
    def __init__(self, session: Session):
        self.session = session
        self.feature_engineer = NFLFeatureEngineer(session)
        self.models_dir = Path("models/enhanced")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }
            },
            'neural_network': {
                'model': MLPRegressor,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate_init': [0.001, 0.01]
                }
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            }
        }
        
        # Ensemble weights (will be learned during training)
        self.ensemble_weights = {
            'random_forest': 0.25,
            'gradient_boosting': 0.20,
            'xgboost': 0.25,
            'lightgbm': 0.20,
            'neural_network': 0.05,
            'ridge': 0.05
        }
    
    def train_all_models(self, target_stat: str = 'fantasy_points_ppr') -> Dict[str, ModelEnsemble]:
        """Train models for all positions"""
        logger.info(f"Training models for {target_stat}")
        
        # Engineer features
        feature_set = self.feature_engineer.engineer_all_features(target_stat)
        model_features = self.feature_engineer.prepare_model_features(feature_set, target_stat)
        
        trained_ensembles = {}
        
        for position, features in model_features.items():
            logger.info(f"Training models for {position}")
            
            try:
                ensemble = self._train_position_ensemble(position, features)
                trained_ensembles[position] = ensemble
                
                # Save the ensemble
                self._save_ensemble(ensemble, position, target_stat)
                
                logger.info(f"Completed training for {position}")
                
            except Exception as e:
                logger.error(f"Error training models for {position}: {e}")
                continue
        
        return trained_ensembles
    
    def _train_position_ensemble(self, position: str, features: ModelFeatures) -> ModelEnsemble:
        """Train ensemble of models for a specific position"""
        models = {}
        performance = {}
        
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training {model_name} for {position}")
                
                # Grid search for best parameters
                model = self._train_single_model(
                    config['model'], 
                    config['params'],
                    features.X_train,
                    features.y_train,
                    features.X_test,
                    features.y_test
                )
                
                # Evaluate performance
                perf = self._evaluate_model(
                    model, 
                    features.X_test, 
                    features.y_test,
                    features.feature_names,
                    model_name
                )
                
                models[model_name] = model
                performance[model_name] = perf
                
                logger.info(f"{model_name} - R²: {perf.r2:.3f}, RMSE: {perf.rmse:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Calculate ensemble weights based on performance
        weights = self._calculate_ensemble_weights(performance)
        
        return ModelEnsemble(
            models=models,
            weights=weights,
            performance=performance,
            feature_names=features.feature_names,
            target_stat=features.target_name,
            position=position
        )
    
    def _train_single_model(self, model_class, param_grid, X_train, y_train, X_test, y_test):
        """Train a single model with hyperparameter tuning"""
        
        # Create model instance
        if model_class == MLPRegressor:
            # Neural network needs special handling
            model = model_class(max_iter=1000, random_state=42)
        elif model_class in [xgb.XGBRegressor, lgb.LGBMRegressor]:
            # Tree-based models
            model = model_class(random_state=42, verbose=0)
        else:
            model = model_class(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_
    
    def _evaluate_model(self, model, X_test, y_test, feature_names, model_name) -> ModelPerformance:
        """Evaluate model performance"""
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
        cv_score = -np.mean(cv_scores)
        
        # Feature importance
        feature_importance = {}
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                importance = np.zeros(len(feature_names))
            
            # Normalize importance
            if np.sum(importance) > 0:
                importance = importance / np.sum(importance)
            
            feature_importance = dict(zip(feature_names, importance))
            
        except Exception as e:
            logger.debug(f"Could not extract feature importance for {model_name}: {e}")
        
        return ModelPerformance(
            rmse=rmse,
            mae=mae,
            r2=r2,
            cv_score=cv_score,
            feature_importance=feature_importance,
            model_name=model_name
        )
    
    def _calculate_ensemble_weights(self, performance: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Calculate ensemble weights based on model performance"""
        
        if not performance:
            return self.ensemble_weights
        
        # Weight based on R² score (higher is better)
        r2_scores = {name: max(perf.r2, 0.0) for name, perf in performance.items()}
        total_r2 = sum(r2_scores.values())
        
        if total_r2 > 0:
            weights = {name: score / total_r2 for name, score in r2_scores.items()}
        else:
            # Fallback to equal weights
            weights = {name: 1.0 / len(performance) for name in performance.keys()}
        
        return weights
    
    def predict_player_performance(self, player_id: str, target_stat: str = 'fantasy_points_ppr') -> Optional[PredictionResult]:
        """Predict performance for a specific player"""
        
        # Get player info
        from ..database_models import Player
        player = self.session.query(Player).filter(Player.player_id == player_id).first()
        if not player:
            return None
        
        # Load ensemble for player's position
        ensemble = self._load_ensemble(player.position, target_stat)
        if not ensemble:
            logger.warning(f"No trained model found for {player.position}")
            return None
        
        try:
            # Engineer features for this player
            player_features = self.feature_engineer._engineer_player_features(player_id)
            team_features = self.feature_engineer._engineer_team_features(player.current_team)
            matchup_features = self.feature_engineer._engineer_matchup_features(player_id)
            situational_features = self.feature_engineer._engineer_situational_features(player_id)
            
            # Combine features
            all_features = {**player_features, **team_features, **matchup_features, **situational_features}
            
            # Convert to array format matching training features
            feature_vector = []
            for feature_name in ensemble.feature_names:
                feature_vector.append(all_features.get(feature_name, 0.0))
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Make ensemble prediction
            predictions = {}
            for model_name, model in ensemble.models.items():
                pred = model.predict(feature_array)[0]
                predictions[model_name] = pred
            
            # Weighted ensemble prediction
            weighted_pred = sum(
                predictions[name] * ensemble.weights.get(name, 0.0)
                for name in predictions.keys()
            )
            
            # Calculate confidence interval (simplified)
            model_std = np.std(list(predictions.values()))
            confidence_interval = (
                weighted_pred - 1.96 * model_std,
                weighted_pred + 1.96 * model_std
            )
            
            # Prediction confidence based on model agreement
            prediction_confidence = 1.0 - (model_std / max(weighted_pred, 1.0))
            prediction_confidence = max(0.0, min(1.0, prediction_confidence))
            
            # Contributing factors (top 5 features)
            contributing_factors = {}
            if ensemble.performance:
                # Use feature importance from best performing model
                best_model = max(ensemble.performance.items(), key=lambda x: x[1].r2)
                top_features = sorted(
                    best_model[1].feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                contributing_factors = dict(top_features)
            
            return PredictionResult(
                player_id=player_id,
                player_name=player.name,
                position=player.position,
                predicted_value=weighted_pred,
                confidence_interval=confidence_interval,
                prediction_confidence=prediction_confidence,
                contributing_factors=contributing_factors,
                model_used=f"{player.position}_ensemble"
            )
            
        except Exception as e:
            logger.error(f"Error predicting for player {player.name}: {e}")
            return None
    
    def _save_ensemble(self, ensemble: ModelEnsemble, position: str, target_stat: str):
        """Save trained ensemble to disk"""
        filename = f"{position}_{target_stat}_ensemble.pkl"
        filepath = self.models_dir / filename
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(ensemble, f)
            logger.info(f"Saved ensemble to {filepath}")
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
    
    def _load_ensemble(self, position: str, target_stat: str) -> Optional[ModelEnsemble]:
        """Load trained ensemble from disk"""
        filename = f"{position}_{target_stat}_ensemble.pkl"
        filepath = self.models_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'rb') as f:
                ensemble = pickle.load(f)
            return ensemble
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            return None
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all trained models"""
        summary = {
            'models_directory': str(self.models_dir),
            'available_models': [],
            'model_performance': {}
        }
        
        # Find all saved models
        for model_file in self.models_dir.glob("*.pkl"):
            parts = model_file.stem.split('_')
            if len(parts) >= 3:
                position = parts[0]
                target_stat = '_'.join(parts[1:-1])
                
                ensemble = self._load_ensemble(position, target_stat)
                if ensemble:
                    summary['available_models'].append({
                        'position': position,
                        'target_stat': target_stat,
                        'num_models': len(ensemble.models),
                        'feature_count': len(ensemble.feature_names)
                    })
                    
                    # Add performance metrics
                    summary['model_performance'][f"{position}_{target_stat}"] = {
                        name: {
                            'r2': perf.r2,
                            'rmse': perf.rmse,
                            'mae': perf.mae
                        }
                        for name, perf in ensemble.performance.items()
                    }
        
        return summary


def main():
    """Test the prediction models"""
    session = get_db_session()
    models = NFLPredictionModels(session)
    
    try:
        # Train models for fantasy points
        logger.info("Training NFL prediction models...")
        ensembles = models.train_all_models('fantasy_points_ppr')
        
        print("Training Results:")
        for position, ensemble in ensembles.items():
            print(f"\n{position} Models:")
            for model_name, perf in ensemble.performance.items():
                print(f"  {model_name}: R² = {perf.r2:.3f}, RMSE = {perf.rmse:.3f}")
        
        # Test prediction on a random active player
        from ..database_models import Player
        test_player = session.query(Player).filter(
            Player.is_active == True,
            Player.position == 'QB'
        ).first()
        
        if test_player:
            prediction = models.predict_player_performance(test_player.player_id)
            if prediction:
                print(f"\nTest Prediction for {prediction.player_name}:")
                print(f"  Predicted Fantasy Points: {prediction.predicted_value:.2f}")
                print(f"  Confidence: {prediction.prediction_confidence:.2f}")
                print(f"  Confidence Interval: {prediction.confidence_interval}")
        
        # Model summary
        summary = models.get_model_summary()
        print(f"\nModel Summary:")
        print(f"  Available Models: {len(summary['available_models'])}")
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
