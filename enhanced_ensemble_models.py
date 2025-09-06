"""
Enhanced Ensemble Models for NFL Predictions
Implements XGBoost, LightGBM, Neural Networks with advanced ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

# ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

# Neural network imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy nn module for compatibility
    class DummyNN:
        class Module:
            pass
        class Sequential:
            def __init__(self, *args):
                pass
        class Linear:
            def __init__(self, *args):
                pass
        class ReLU:
            def __init__(self):
                pass
        class Dropout:
            def __init__(self, *args):
                pass
        class MSELoss:
            def __init__(self):
                pass
    nn = DummyNN()

from enhanced_prediction_targets import PredictionTarget, get_targets_for_position

logger = logging.getLogger(__name__)

class EnhancedEnsembleModel:
    """Advanced ensemble model with multiple algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self.performance_metrics = {}
        
    def train(self, X: np.ndarray, y: np.ndarray, target: PredictionTarget) -> Dict[str, Any]:
        """Train ensemble of models."""
        results = {}
        
        # XGBoost
        if 'xgboost' in self.config.get('model_types', []):
            xgb_model, xgb_score = self._train_xgboost(X, y, target)
            self.models['xgboost'] = xgb_model
            results['xgboost'] = xgb_score
            
        # LightGBM
        if 'lightgbm' in self.config.get('model_types', []):
            lgb_model, lgb_score = self._train_lightgbm(X, y, target)
            self.models['lightgbm'] = lgb_model
            results['lightgbm'] = lgb_score
            
        # Random Forest
        if 'random_forest' in self.config.get('model_types', []):
            rf_model, rf_score = self._train_random_forest(X, y, target)
            self.models['random_forest'] = rf_model
            results['random_forest'] = rf_score
            
        # Neural Network
        if 'neural_network' in self.config.get('model_types', []) and TORCH_AVAILABLE:
            nn_model, nn_score = self._train_neural_network(X, y, target)
            self.models['neural_network'] = nn_model
            results['neural_network'] = nn_score
            
        # Calculate ensemble weights
        self._calculate_ensemble_weights(results)
        
        return results
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, target: PredictionTarget) -> Tuple[Any, float]:
        """Train XGBoost model."""
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        # Train on full dataset
        model.fit(X, y)
        
        return model, np.mean(scores)
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, target: PredictionTarget) -> Tuple[Any, float]:
        """Train LightGBM model."""
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        model.fit(X, y)
        
        return model, np.mean(scores)
    
    def _train_random_forest(self, X: np.ndarray, y: np.ndarray, target: PredictionTarget) -> Tuple[Any, float]:
        """Train Random Forest model."""
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        model.fit(X, y)
        
        return model, np.mean(scores)
    
    def _train_neural_network(self, X: np.ndarray, y: np.ndarray, target: PredictionTarget) -> Tuple[Any, float]:
        """Train Neural Network model."""
        if not TORCH_AVAILABLE:
            return None, 0.0
            
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['neural_network'] = scaler
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # Create model
        model = NFLNeuralNetwork(X.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).numpy().flatten()
            score = r2_score(y, predictions)
        
        return model, score
    
    def _calculate_ensemble_weights(self, results: Dict[str, float]):
        """Calculate ensemble weights based on performance."""
        total_score = sum(max(0, score) for score in results.values())
        
        if total_score > 0:
            self.weights = {
                model: max(0, score) / total_score 
                for model, score in results.items()
            }
        else:
            # Equal weights if all models perform poorly
            n_models = len(results)
            self.weights = {model: 1/n_models for model in results.keys()}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        predictions = []
        
        for model_name, model in self.models.items():
            if model is None:
                continue
                
            try:
                if model_name == 'neural_network' and TORCH_AVAILABLE:
                    # Scale features for neural network
                    X_scaled = self.scalers['neural_network'].transform(X)
                    X_tensor = torch.FloatTensor(X_scaled)
                    model.eval()
                    with torch.no_grad():
                        pred = model(X_tensor).numpy().flatten()
                else:
                    pred = model.predict(X)
                
                weight = self.weights.get(model_name, 0)
                predictions.append(pred * weight)
                
            except Exception as e:
                logger.warning(f"Error in {model_name} prediction: {e}")
                continue
        
        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(len(X))

class NFLNeuralNetwork(nn.Module):
    """Neural network for NFL predictions."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class ComprehensivePredictor:
    """Main predictor class with enhanced ensemble models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.model_dir = Path(config.get('model_directory', 'models/enhanced'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def train_all_models(self, features_df: pd.DataFrame, targets_df: pd.DataFrame):
        """Train models for all positions and targets."""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            logger.info(f"Training models for {position}")
            targets = get_targets_for_position(position)
            
            for target in targets:
                try:
                    # Prepare data
                    X, y = self._prepare_data(features_df, targets_df, target, position)
                    
                    if len(X) < 50:
                        logger.warning(f"Insufficient data for {position} {target.name}")
                        continue
                    
                    # Train ensemble
                    ensemble = EnhancedEnsembleModel(self.config)
                    results = ensemble.train(X, y, target)
                    
                    # Store model
                    model_key = f"{position}_{target.name}"
                    self.models[model_key] = ensemble
                    
                    # Save to disk
                    model_path = self.model_dir / f"{model_key}.pkl"
                    joblib.dump(ensemble, model_path)
                    
                    logger.info(f"Trained {model_key}: {results}")
                    
                except Exception as e:
                    logger.error(f"Error training {position} {target.name}: {e}")
    
    def _prepare_data(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, 
                     target: PredictionTarget, position: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for a specific target."""
        # Merge features and targets
        data = pd.merge(features_df, targets_df, on=['player_id', 'game_id'], how='inner')
        
        # Filter for position
        data = data[data['position'] == position]
        
        # Remove rows with missing target values
        data = data.dropna(subset=[target.column])
        
        # Get feature columns
        feature_cols = [col for col in features_df.columns 
                       if col not in ['player_id', 'game_id', 'position']]
        
        X = data[feature_cols].fillna(0).values
        y = data[target.column].values
        
        return X, y
    
    def predict(self, features: Dict[str, float], position: str) -> Dict[str, Any]:
        """Make predictions for a player."""
        predictions = {}
        targets = get_targets_for_position(position)
        
        for target in targets:
            model_key = f"{position}_{target.name}"
            
            if model_key in self.models:
                try:
                    # Prepare feature vector
                    feature_vector = self._prepare_feature_vector(features, target)
                    
                    # Make prediction
                    ensemble = self.models[model_key]
                    prediction = ensemble.predict([feature_vector])[0]
                    
                    # Apply constraints
                    prediction = np.clip(prediction, target.min_value, target.max_value)
                    
                    predictions[target.name] = {
                        'value': float(prediction),
                        'confidence': self._calculate_confidence(ensemble, feature_vector)
                    }
                    
                except Exception as e:
                    logger.error(f"Error predicting {target.name}: {e}")
        
        return predictions
    
    def _prepare_feature_vector(self, features: Dict[str, float], target: PredictionTarget) -> np.ndarray:
        """Prepare feature vector for prediction."""
        # This would need to match training feature order
        # For now, create a basic vector
        return np.array(list(features.values())[:10])  # Take first 10 features
    
    def _calculate_confidence(self, ensemble: EnhancedEnsembleModel, feature_vector: np.ndarray) -> float:
        """Calculate prediction confidence."""
        # Simple confidence based on model agreement
        individual_predictions = []
        
        for model_name, model in ensemble.models.items():
            try:
                if model_name == 'neural_network' and TORCH_AVAILABLE:
                    X_scaled = ensemble.scalers['neural_network'].transform([feature_vector])
                    X_tensor = torch.FloatTensor(X_scaled)
                    model.eval()
                    with torch.no_grad():
                        pred = model(X_tensor).item()
                else:
                    pred = model.predict([feature_vector])[0]
                individual_predictions.append(pred)
            except:
                continue
        
        if len(individual_predictions) > 1:
            std_dev = np.std(individual_predictions)
            mean_pred = np.mean(individual_predictions)
            cv = std_dev / (abs(mean_pred) + 1e-6)
            confidence = max(0.1, 1.0 - cv)
        else:
            confidence = 0.7
        
        return min(0.95, confidence)

if __name__ == "__main__":
    # Example usage
    config = {
        'model_types': ['xgboost', 'lightgbm', 'random_forest'],
        'model_directory': 'models/enhanced'
    }
    
    predictor = ComprehensivePredictor(config)
    print("Enhanced ensemble models initialized successfully!")
