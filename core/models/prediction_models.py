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


@dataclass
class ModelConfig:
    """Configuration for a prediction model"""
    model_type: str  # 'lightgbm', 'xgboost', 'random_forest', etc.
    target_stat: str  # 'fantasy_points_ppr', 'receiving_yards', etc.
    hyperparameters: Dict[str, Any]
    feature_selection_k: int = 50
    cv_folds: int = 5
    random_state: int = 42


class NFLPredictionModel:
    """Base class for NFL prediction models"""
    
    def __init__(self, config: ModelConfig, feature_engineer: NFLFeatureEngineer):
        self.config = config
        self.feature_engineer = feature_engineer
        self.model = None
        self.is_trained = False
        self.performance_metrics = None
        self.feature_names = None
        
        # Initialize model based on type
        self._initialize_model()
        
        logger.info(f"Initialized {config.model_type} model for {config.target_stat}")
    
    def _initialize_model(self):
        """Initialize the underlying ML model"""
        
        if self.config.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                random_state=self.config.random_state,
                **self.config.hyperparameters
            )
        elif self.config.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                random_state=self.config.random_state,
                **self.config.hyperparameters
            )
        elif self.config.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                random_state=self.config.random_state,
                **self.config.hyperparameters
            )
        elif self.config.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                random_state=self.config.random_state,
                **self.config.hyperparameters
            )
        elif self.config.model_type == 'ridge':
            self.model = Ridge(**self.config.hyperparameters)
        elif self.config.model_type == 'elastic_net':
            self.model = ElasticNet(
                random_state=self.config.random_state,
                **self.config.hyperparameters
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def train(self, model_features: ModelFeatures) -> ModelPerformance:
        """Train the model and return performance metrics"""
        
        try:
            # Feature selection if specified
            if self.config.feature_selection_k > 0:
                model_features = self.feature_engineer.select_features(
                    model_features, k=self.config.feature_selection_k
                )
            
            # Train the model
            self.model.fit(model_features.X_train, model_features.y_train)
            self.is_trained = True
            self.feature_names = model_features.feature_names
            
            # Evaluate performance
            y_pred = self.model.predict(model_features.X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(model_features.y_test, y_pred))
            mae = mean_absolute_error(model_features.y_test, y_pred)
            r2 = r2_score(model_features.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                self.model, model_features.X_train, model_features.y_train,
                cv=self.config.cv_folds, scoring='neg_mean_squared_error'
            )
            cv_score = np.sqrt(-cv_scores.mean())
            
            # Feature importance
            feature_importance = self._get_feature_importance()
            
            self.performance_metrics = ModelPerformance(
                rmse=rmse,
                mae=mae,
                r2=r2,
                cv_score=cv_score,
                feature_importance=feature_importance,
                model_name=f"{self.config.model_type}_{self.config.target_stat}"
            )
            
            logger.info(f"Model trained - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Base prediction
        predictions = self.model.predict(features)
        
        # Confidence intervals (simplified approach using model uncertainty)
        if hasattr(self.model, 'predict_proba'):
            # For models that support uncertainty quantification
            std_dev = np.std(predictions) * 0.5  # Simplified uncertainty
        else:
            # Use cross-validation error as uncertainty estimate
            std_dev = self.performance_metrics.cv_score * 0.5
        
        confidence_lower = predictions - 1.96 * std_dev
        confidence_upper = predictions + 1.96 * std_dev
        
        return predictions, np.column_stack([confidence_lower, confidence_upper])
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance from the trained model"""
        
        if not self.is_trained or not self.feature_names:
            return {}
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_)
            else:
                return {}
            
            return dict(zip(self.feature_names, importances))
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def save_model(self, filepath: Path):
        """Save the trained model to disk"""
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path, feature_engineer: NFLFeatureEngineer):
        """Load a trained model from disk"""
        
        model_data = joblib.load(filepath)
        
        instance = cls(model_data['config'], feature_engineer)
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.performance_metrics = model_data['performance_metrics']
        instance.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class EnsembleModel:
    """Ensemble of multiple NFL prediction models"""
    
    def __init__(self, models: List[NFLPredictionModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions"""
        
        all_predictions = []
        all_confidence_intervals = []
        
        for model in self.models:
            pred, conf_int = model.predict(features)
            all_predictions.append(pred)
            all_confidence_intervals.append(conf_int)
        
        # Weighted average of predictions
        weighted_predictions = np.average(all_predictions, axis=0, weights=self.weights)
        
        # Combine confidence intervals (conservative approach)
        combined_lower = np.average([ci[:, 0] for ci in all_confidence_intervals], axis=0, weights=self.weights)
        combined_upper = np.average([ci[:, 1] for ci in all_confidence_intervals], axis=0, weights=self.weights)
        
        # Expand intervals to account for model disagreement
        pred_std = np.std(all_predictions, axis=0)
        combined_lower -= pred_std
        combined_upper += pred_std
        
        confidence_intervals = np.column_stack([combined_lower, combined_upper])
        
        return weighted_predictions, confidence_intervals
    
    def get_model_performance(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all models in the ensemble"""
        
        return {
            model.performance_metrics.model_name: model.performance_metrics
            for model in self.models if model.performance_metrics
        }


class NFLPredictionEngine:
    """Main prediction engine for NFL player performance"""
    
    def __init__(self, session: Session, models_dir: Path = Path("models")):
        self.session = session
        self.models_dir = models_dir
        self.models_dir.mkdir(exist_ok=True)
        
        self.feature_engineer = NFLFeatureEngineer(session)
        self.trained_models: Dict[str, NFLPredictionModel] = {}
        self.ensemble_models: Dict[str, EnsembleModel] = {}
        
        logger.info("NFL Prediction Engine initialized")
    
    def create_model_configs(self) -> Dict[str, List[ModelConfig]]:
        """Create default model configurations for different stats"""
        
        target_stats = [
            'fantasy_points_ppr',
            'receiving_yards',
            'rushing_yards', 
            'passing_yards',
            'receiving_touchdowns',
            'rushing_touchdowns',
            'passing_touchdowns'
        ]
        
        configs = {}
        
        for stat in target_stats:
            configs[stat] = [
                ModelConfig(
                    model_type='lightgbm',
                    target_stat=stat,
                    hyperparameters={
                        'n_estimators': 200,
                        'max_depth': 8,
                        'learning_rate': 0.05,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8
                    }
                ),
                ModelConfig(
                    model_type='xgboost',
                    target_stat=stat,
                    hyperparameters={
                        'n_estimators': 200,
                        'max_depth': 6,
                        'learning_rate': 0.05,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8
                    }
                ),
                ModelConfig(
                    model_type='random_forest',
                    target_stat=stat,
                    hyperparameters={
                        'n_estimators': 100,
                        'max_depth': 10,
                        'min_samples_split': 5,
                        'min_samples_leaf': 2
                    }
                )
            ]
        
        return configs
    
    async def train_models(self, feature_data: List[Dict[str, Any]]) -> Dict[str, ModelPerformance]:
        """Train all models for all target statistics"""
        
        model_configs = self.create_model_configs()
        performance_results = {}
        
        for target_stat, configs in model_configs.items():
            logger.info(f"Training models for {target_stat}")
            
            # Prepare training data for this target
            model_features = self.feature_engineer.prepare_training_data(
                feature_data, target_column=target_stat
            )
            
            stat_models = []
            
            for config in configs:
                try:
                    # Create and train model
                    model = NFLPredictionModel(config, self.feature_engineer)
                    performance = model.train(model_features)
                    
                    # Save model
                    model_filename = f"{config.model_type}_{target_stat}.joblib"
                    model.save_model(self.models_dir / model_filename)
                    
                    # Store model
                    model_key = f"{config.model_type}_{target_stat}"
                    self.trained_models[model_key] = model
                    stat_models.append(model)
                    
                    performance_results[model_key] = performance
                    
                except Exception as e:
                    logger.error(f"Failed to train {config.model_type} for {target_stat}: {e}")
            
            # Create ensemble for this stat
            if stat_models:
                ensemble = EnsembleModel(stat_models)
                self.ensemble_models[target_stat] = ensemble
                logger.info(f"Created ensemble for {target_stat} with {len(stat_models)} models")
        
        return performance_results
    
    async def predict_player_performance(
        self, 
        player_id: str, 
        season: int, 
        week: int,
        target_stats: Optional[List[str]] = None
    ) -> PredictionResult:
        """Predict performance for a specific player"""
        
        target_stats = target_stats or ['fantasy_points_ppr']
        
        # Engineer features for the player
        features = await self.feature_engineer.engineer_player_features(
            player_id, season, week, target_stats
        )
        
        # Convert to numpy array for prediction
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        predictions = {}
        confidence_intervals = {}
        
        for stat in target_stats:
            if stat in self.ensemble_models:
                pred, conf_int = self.ensemble_models[stat].predict(feature_array)
                predictions[stat] = float(pred[0])
                confidence_intervals[stat] = (float(conf_int[0, 0]), float(conf_int[0, 1]))
            else:
                logger.warning(f"No trained model found for {stat}")
                predictions[stat] = 0.0
                confidence_intervals[stat] = (0.0, 0.0)
        
        # Get player info
        from ..database_models import Player
        player = self.session.query(Player).filter(Player.player_id == player_id).first()
        
        if not player:
            raise ValueError(f"Player {player_id} not found")
        
        return PredictionResult(
            player_id=player_id,
            player_name=player.name,
            position=player.position,
            team=player.current_team,
            opponent="TBD",  # Would need game info to determine
            week=week,
            season=season,
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_version="v1.0",
            prediction_timestamp=datetime.now()
        )
    
    def load_trained_models(self):
        """Load all trained models from disk"""
        
        model_files = list(self.models_dir.glob("*.joblib"))
        
        for model_file in model_files:
            try:
                model = NFLPredictionModel.load_model(model_file, self.feature_engineer)
                model_key = model_file.stem
                self.trained_models[model_key] = model
                logger.info(f"Loaded model: {model_key}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
        
        # Recreate ensembles
        self._create_ensembles_from_loaded_models()
    
    def _create_ensembles_from_loaded_models(self):
        """Create ensemble models from loaded individual models"""
        
        # Group models by target stat
        stat_models = {}
        
        for model_key, model in self.trained_models.items():
            target_stat = model.config.target_stat
            if target_stat not in stat_models:
                stat_models[target_stat] = []
            stat_models[target_stat].append(model)
        
        # Create ensembles
        for stat, models in stat_models.items():
            if len(models) > 1:
                ensemble = EnsembleModel(models)
                self.ensemble_models[stat] = ensemble
                logger.info(f"Created ensemble for {stat} with {len(models)} models")
    
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
