"""
NFL Player Performance Prediction Models
Advanced ensemble machine learning system for predicting player statistics and game outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import joblib
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Neural network imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # Create dummy nn module to prevent NameError
    class DummyNN:
        class Module:
            pass
    nn = DummyNN()
    print("PyTorch not available. Neural network models will be disabled.")

# Import our models
from database_models import PlayerPrediction, GamePrediction, ModelPerformance

# Configure logging
logging.basicConfig(level=logging.INFO)


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    model_types: List[str] = field(default_factory=lambda: ['xgboost', 'lightgbm', 'random_forest'])
    ensemble_method: str = 'weighted_average'
    validation_method: str = 'time_series_split'
    n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    save_models: bool = True
    model_directory: str = 'models'
    enable_neural_networks: bool = True
    hyperparameter_tuning: bool = True
    

@dataclass
class PredictionTarget:
    """Definition of what we're predicting."""
    name: str
    column: str
    position: str
    prediction_type: str  # 'regression' or 'classification'
    min_value: float = 0.0
    max_value: Optional[float] = None
    

class NFLPredictor:
    """Main prediction system orchestrator."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the NFL prediction system."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Prediction targets by position
        self.prediction_targets = self._initialize_prediction_targets()
        
        # Model directory setup
        self.model_dir = Path(config.model_directory)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize individual model classes
        self.model_builders = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'random_forest': RandomForestModel,
            'gradient_boosting': GradientBoostingModel
        }
        
        if PYTORCH_AVAILABLE and config.enable_neural_networks:
            self.model_builders['neural_network'] = NeuralNetworkModel
            
    def _initialize_prediction_targets(self) -> Dict[str, List[PredictionTarget]]:
        """Initialize prediction targets for each position."""
        return {
            'QB': [
                PredictionTarget('passing_yards', 'passing_yards', 'QB', 'regression', 0, 500),
                PredictionTarget('passing_tds', 'passing_touchdowns', 'QB', 'regression', 0, 7),
                PredictionTarget('interceptions', 'passing_interceptions', 'QB', 'regression', 0, 5),
                PredictionTarget('rushing_yards', 'rushing_yards', 'QB', 'regression', 0, 100),
                PredictionTarget('fantasy_points', 'fantasy_points_ppr', 'QB', 'regression', 0, 50)
            ],
            'RB': [
                PredictionTarget('rushing_yards', 'rushing_yards', 'RB', 'regression', 0, 300),
                PredictionTarget('rushing_tds', 'rushing_touchdowns', 'RB', 'regression', 0, 4),
                PredictionTarget('receptions', 'receptions', 'RB', 'regression', 0, 15),
                PredictionTarget('receiving_yards', 'receiving_yards', 'RB', 'regression', 0, 150),
                PredictionTarget('fantasy_points', 'fantasy_points_ppr', 'RB', 'regression', 0, 40)
            ],
            'WR': [
                PredictionTarget('receptions', 'receptions', 'WR', 'regression', 0, 20),
                PredictionTarget('receiving_yards', 'receiving_yards', 'WR', 'regression', 0, 250),
                PredictionTarget('receiving_tds', 'receiving_touchdowns', 'WR', 'regression', 0, 4),
                PredictionTarget('targets', 'targets', 'WR', 'regression', 0, 25),
                PredictionTarget('fantasy_points', 'fantasy_points_ppr', 'WR', 'regression', 0, 40)
            ],
            'TE': [
                PredictionTarget('receptions', 'receptions', 'TE', 'regression', 0, 15),
                PredictionTarget('receiving_yards', 'receiving_yards', 'TE', 'regression', 0, 200),
                PredictionTarget('receiving_tds', 'receiving_touchdowns', 'TE', 'regression', 0, 3),
                PredictionTarget('targets', 'targets', 'TE', 'regression', 0, 20),
                PredictionTarget('fantasy_points', 'fantasy_points_ppr', 'TE', 'regression', 0, 35)
            ]
        }
        
    def train_models(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        position: str
    ) -> Dict[str, Any]:
        """Train all models for a specific position."""
        
        self.logger.info(f"Training models for position: {position}")
        
        results = {}
        targets = self.prediction_targets.get(position, [])
        
        for target in targets:
            self.logger.info(f"Training models for {target.name}")
            
            # Prepare data for this target
            X, y = self._prepare_training_data(features_df, targets_df, target)
            
            if len(X) < 50:  # Minimum data requirement
                self.logger.warning(f"Insufficient data for {target.name}: {len(X)} samples")
                continue
                
            # Split data using time series split
            train_scores = {}
            val_scores = {}
            models = {}
            
            # Train each model type
            for model_type in self.config.model_types:
                if model_type not in self.model_builders:
                    self.logger.warning(f"Model type {model_type} not available")
                    continue
                    
                try:
                    # Build and train model
                    model_builder = self.model_builders[model_type](self.config)
                    model, train_score, val_score = model_builder.train(X, y, target)
                    
                    models[model_type] = model
                    train_scores[model_type] = train_score
                    val_scores[model_type] = val_score
                    
                    self.logger.info(f"{model_type} - Train R²: {train_score:.3f}, Val R²: {val_score:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_type} for {target.name}: {e}")
                    continue
                    
            # Create ensemble
            if len(models) > 1:
                ensemble = EnsembleModel(models, self.config.ensemble_method)
                ensemble_score = ensemble.validate(X, y)
                models['ensemble'] = ensemble
                val_scores['ensemble'] = ensemble_score
                
                self.logger.info(f"Ensemble - Val R²: {ensemble_score:.3f}")
                
            # Store models and results
            model_key = f"{position}_{target.name}"
            self.models[model_key] = models
            
            results[target.name] = {
                'train_scores': train_scores,
                'validation_scores': val_scores,
                'best_model': max(val_scores, key=val_scores.get),
                'best_score': max(val_scores.values()),
                'n_samples': len(X),
                'n_features': X.shape[1]
            }
            
            # Save models if configured
            if self.config.save_models:
                self._save_models(models, model_key)
                
        return results
        
    def predict(
        self,
        features: Dict[str, float],
        position: str,
        model_type: str = 'ensemble'
    ) -> Dict[str, Any]:
        """Make predictions for a player."""
        
        predictions = {}
        targets = self.prediction_targets.get(position, [])
        
        for target in targets:
            model_key = f"{position}_{target.name}"
            
            if model_key not in self.models:
                self.logger.warning(f"No model found for {model_key}")
                continue
                
            models = self.models[model_key]
            
            if model_type not in models:
                # Fall back to best available model
                model_type = 'ensemble' if 'ensemble' in models else list(models.keys())[0]
                
            model = models[model_type]
            
            try:
                # Prepare features for prediction
                feature_vector = self._prepare_prediction_features(features, target)
                
                # Make prediction
                prediction = model.predict([feature_vector])[0]
                
                # Apply constraints
                prediction = np.clip(prediction, target.min_value, target.max_value)
                
                # Get confidence score
                confidence = self._calculate_prediction_confidence(
                    model, feature_vector, target, models
                )
                
                predictions[target.name] = {
                    'predicted_value': float(prediction),
                    'confidence': float(confidence),
                    'model_used': model_type
                }
                
            except Exception as e:
                self.logger.error(f"Error predicting {target.name}: {e}")
                continue
                
        return predictions
        
    def _prepare_training_data(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        target: PredictionTarget
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for a specific target."""
        
        # Merge features and targets
        data = pd.merge(features_df, targets_df, on=['player_id', 'game_id'], how='inner')
        
        # Filter for position
        data = data[data['position'] == target.position]
        
        # Remove rows with missing target values
        data = data.dropna(subset=[target.column])
        
        # Get feature columns (exclude metadata columns)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_id', 'game_id', 'position', 'game_date']]
        
        X = data[feature_cols].fillna(0).values
        y = data[target.column].values
        
        return X, y
        
    def _prepare_prediction_features(
        self,
        features: Dict[str, float],
        target: PredictionTarget
    ) -> np.ndarray:
        """Prepare features for a single prediction."""
        
        # This would need to match the feature order used in training
        # For now, create a simple feature vector
        feature_vector = []
        
        # Extract relevant features based on position and target
        relevant_features = self._get_relevant_features(target)
        
        for feature_name in relevant_features:
            feature_vector.append(features.get(feature_name, 0.0))
            
        return np.array(feature_vector)
        
    def _get_relevant_features(self, target: PredictionTarget) -> List[str]:
        """Get relevant features for a specific target."""
        
        # This would be configurable based on feature importance analysis
        # For now, return a basic set
        position_features = {
            'QB': [
                'last_5_games_passing_yards_mean', 'last_5_games_passing_tds_mean',
                'completion_percentage', 'yards_per_attempt', 'season_passing_yards_trend',
                'vs_division_fantasy_avg', 'home_fantasy_avg', 'weather_temperature'
            ],
            'RB': [
                'last_5_games_rushing_yards_mean', 'last_5_games_rushing_tds_mean',
                'yards_per_carry', 'avg_carries_per_game', 'season_rushing_yards_trend',
                'vs_division_fantasy_avg', 'home_fantasy_avg'
            ],
            'WR': [
                'last_5_games_receiving_yards_mean', 'last_5_games_receptions_mean',
                'catch_rate', 'yards_per_reception', 'avg_targets_per_game',
                'vs_division_fantasy_avg', 'home_fantasy_avg'
            ],
            'TE': [
                'last_5_games_receiving_yards_mean', 'last_5_games_receptions_mean',
                'catch_rate', 'yards_per_reception', 'avg_targets_per_game',
                'vs_division_fantasy_avg', 'home_fantasy_avg'
            ]
        }
        
        return position_features.get(target.position, [])
        
    def _calculate_prediction_confidence(
        self,
        model: Any,
        feature_vector: np.ndarray,
        target: PredictionTarget,
        all_models: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for a prediction."""
        
        try:
            # Method 1: Model uncertainty (if available)
            if hasattr(model, 'predict_proba'):
                # For models with uncertainty estimation
                confidence = 0.8  # Placeholder
            else:
                # Method 2: Ensemble agreement
                if len(all_models) > 1:
                    predictions = []
                    for m in all_models.values():
                        try:
                            pred = m.predict([feature_vector])[0]
                            predictions.append(pred)
                        except:
                            continue
                            
                    if len(predictions) > 1:
                        # Higher agreement = higher confidence
                        std_dev = np.std(predictions)
                        mean_pred = np.mean(predictions)
                        cv = std_dev / (mean_pred + 1e-6)  # Coefficient of variation
                        confidence = max(0.1, 1.0 - cv)
                    else:
                        confidence = 0.7
                else:
                    confidence = 0.7
                    
            return min(0.99, max(0.1, confidence))
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {e}")
            return 0.7
            
    def _save_models(self, models: Dict[str, Any], model_key: str):
        """Save trained models to disk."""
        
        try:
            model_path = self.model_dir / f"{model_key}_models.pkl"
            joblib.dump(models, model_path)
            self.logger.info(f"Models saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            
    def load_models(self, model_key: str) -> bool:
        """Load models from disk."""
        
        try:
            model_path = self.model_dir / f"{model_key}_models.pkl"
            if model_path.exists():
                self.models[model_key] = joblib.load(model_path)
                self.logger.info(f"Models loaded from {model_path}")
                return True
            else:
                self.logger.warning(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False


class BaseMLModel:
    """Base class for all ML models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target: PredictionTarget
    ) -> Tuple[Any, float, float]:
        """Train the model and return model, train score, validation score."""
        raise NotImplementedError
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        raise NotImplementedError


class XGBoostModel(BaseMLModel):
    """XGBoost regression model."""
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target: PredictionTarget
    ) -> Tuple[xgb.XGBRegressor, float, float]:
        """Train XGBoost model."""
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        # Hyperparameters
        if self.config.hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            base_model = xgb.XGBRegressor(
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            model = grid_search.best_estimator_
            
        else:
            # Default parameters
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            model.fit(X, y)
            
        # Calculate scores
        train_score = model.score(X, y)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        val_score = np.mean(cv_scores)
        
        return model, train_score, val_score


class LightGBMModel(BaseMLModel):
    """LightGBM regression model."""
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target: PredictionTarget
    ) -> Tuple[lgb.LGBMRegressor, float, float]:
        """Train LightGBM model."""
        
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        if self.config.hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            base_model = lgb.LGBMRegressor(
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1
            )
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            model = grid_search.best_estimator_
            
        else:
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X, y)
            
        train_score = model.score(X, y)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        val_score = np.mean(cv_scores)
        
        return model, train_score, val_score


class RandomForestModel(BaseMLModel):
    """Random Forest regression model."""
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target: PredictionTarget
    ) -> Tuple[RandomForestRegressor, float, float]:
        """Train Random Forest model."""
        
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        if self.config.hyperparameter_tuning:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            base_model = RandomForestRegressor(
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X, y)
            model = grid_search.best_estimator_
            
        else:
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=7,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            model.fit(X, y)
            
        train_score = model.score(X, y)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        val_score = np.mean(cv_scores)
        
        return model, train_score, val_score


class GradientBoostingModel(BaseMLModel):
    """Gradient Boosting regression model."""
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target: PredictionTarget
    ) -> Tuple[GradientBoostingRegressor, float, float]:
        """Train Gradient Boosting model."""
        
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            random_state=self.config.random_state
        )
        
        model.fit(X, y)
        
        train_score = model.score(X, y)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        val_score = np.mean(cv_scores)
        
        return model, train_score, val_score


class NeuralNetworkModel(BaseMLModel):
    """Neural Network regression model using PyTorch."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target: PredictionTarget
    ) -> Tuple[object, float, float]:
        """Train neural network model."""
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for neural network training")
            
        # Prepare data
        if TORCH_AVAILABLE:
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        else:
            raise ImportError("PyTorch not available")
        
        # Create model
        if TORCH_AVAILABLE:
            model = NFLNeuralNetwork(X.shape[1], target).to(self.device)
            
            # Training setup
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
        else:
            raise ImportError("PyTorch not available")
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Split for validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        for epoch in range(200):
            # Training
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
            model.train()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
                    
        # Calculate scores
        model.eval()
        with torch.no_grad():
            train_pred = model(X_tensor).cpu().numpy()
            train_score = r2_score(y, train_pred)
            
            val_pred = model(X_val).cpu().numpy()
            val_score = r2_score(y_val.cpu().numpy(), val_pred)
            
        return model, train_score, val_score
        
    def predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Make predictions with neural network."""
        
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = model(X_tensor).cpu().numpy()
            
        return predictions.flatten()


class NFLNeuralNetwork(nn.Module):
    """Neural network architecture for NFL predictions."""
    
    def __init__(self, input_size: int, target: PredictionTarget):
        super(NFLNeuralNetwork, self).__init__()
        
        # Network architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.layers(x)


class EnsembleModel:
    """Ensemble model combining multiple base models."""
    
    def __init__(self, models: Dict[str, Any], ensemble_method: str = 'weighted_average'):
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = {}
        self.logger = logging.getLogger(__name__)
        
    def fit_weights(self, X: np.ndarray, y: np.ndarray):
        """Fit ensemble weights based on individual model performance."""
        
        # Calculate individual model scores
        scores = {}
        for name, model in self.models.items():
            try:
                predictions = model.predict(X)
                score = r2_score(y, predictions)
                scores[name] = max(0, score)  # Ensure non-negative
            except:
                scores[name] = 0
                
        # Normalize weights
        total_score = sum(scores.values())
        if total_score > 0:
            self.weights = {name: score/total_score for name, score in scores.items()}
        else:
            # Equal weights if all models perform poorly
            self.weights = {name: 1/len(self.models) for name in self.models.keys()}
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        
        if not self.weights:
            # Initialize with equal weights
            self.weights = {name: 1/len(self.models) for name in self.models.keys()}
            
        predictions = []
        
        if self.ensemble_method == 'weighted_average':
            # Weighted average of predictions
            weighted_sum = np.zeros(len(X))
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(X)
                    weight = self.weights.get(name, 0)
                    weighted_sum += weight * pred
                except Exception as e:
                    self.logger.warning(f"Error in model {name}: {e}")
                    continue
                    
            predictions = weighted_sum
            
        elif self.ensemble_method == 'median':
            # Median of predictions
            all_predictions = []
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(X)
                    all_predictions.append(pred)
                except:
                    continue
                    
            if all_predictions:
                predictions = np.median(all_predictions, axis=0)
            else:
                predictions = np.zeros(len(X))
                
        return predictions
        
    def validate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Validate ensemble performance."""
        
        # Fit weights on training data
        self.fit_weights(X, y)
        
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate R² score
        score = r2_score(y, predictions)
        
        return score


# Example usage and testing
def main():
    """Example usage of the ML prediction system."""
    
    # Configuration
    config = ModelConfig(
        model_types=['xgboost', 'lightgbm', 'random_forest'],
        ensemble_method='weighted_average',
        hyperparameter_tuning=True,
        save_models=True
    )
    
    # Initialize predictor
    predictor = NFLPredictor(config)
    
    # Example training data (would come from feature engineering)
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    features_df = pd.DataFrame({
        'player_id': [f'player_{i%100}' for i in range(n_samples)],
        'game_id': [f'game_{i}' for i in range(n_samples)],
        'position': ['QB'] * n_samples,
        **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)}
    })
    
    targets_df = pd.DataFrame({
        'player_id': [f'player_{i%100}' for i in range(n_samples)],
        'game_id': [f'game_{i}' for i in range(n_samples)],
        'position': ['QB'] * n_samples,
        'passing_yards': np.random.normal(250, 50, n_samples),
        'passing_touchdowns': np.random.poisson(2, n_samples),
        'passing_interceptions': np.random.poisson(1, n_samples),
        'fantasy_points_ppr': np.random.normal(20, 5, n_samples)
    })
    
    # Train models
    results = predictor.train_models(features_df, targets_df, 'QB')
    
    print("Training Results:")
    for target, result in results.items():
        print(f"\n{target}:")
        print(f"  Best Model: {result['best_model']}")
        print(f"  Best Score: {result['best_score']:.3f}")
        print(f"  Samples: {result['n_samples']}")
        
    # Make predictions
    sample_features = {f'feature_{i}': np.random.randn() for i in range(n_features)}
    
    predictions = predictor.predict(sample_features, 'QB', 'ensemble')
    
    print("\nSample Predictions:")
    for target, pred in predictions.items():
        print(f"{target}: {pred['predicted_value']:.2f} (confidence: {pred['confidence']:.2f})")


if __name__ == "__main__":
    main()