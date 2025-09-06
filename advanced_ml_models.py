"""
Advanced Machine Learning Models for NFL Betting Predictions
Enhanced with XGBoost, LightGBM, Neural Networks, and Ensemble Methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers, callbacks
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    keras = None
    TENSORFLOW_AVAILABLE = False

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)

class AdvancedMLModels:
    """Advanced machine learning models for NFL predictions."""
    
    def __init__(self, model_dir: Path = Path("models/advanced")):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1
            },
            'neural_network': {
                'hidden_layers': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32
            }
        }
    
    def create_xgboost_model(self, **kwargs) -> Optional[Any]:
        """Create XGBoost model if available."""
        if not XGBOOST_AVAILABLE or xgb is None:
            logger.warning("XGBoost not available. Install with: pip install xgboost")
            return None
        
        config = {**self.model_configs['xgboost'], **kwargs}
        return xgb.XGBRegressor(**config)
    
    def create_lightgbm_model(self, **kwargs) -> Optional[Any]:
        """Create LightGBM model if available."""
        if not LIGHTGBM_AVAILABLE or lgb is None:
            logger.warning("LightGBM not available. Install with: pip install lightgbm")
            return None
        
        config = {**self.model_configs['lightgbm'], **kwargs}
        return lgb.LGBMRegressor(**config)
    
    def create_neural_network(self, input_dim: int, **kwargs) -> Optional[Any]:
        """Create neural network model if TensorFlow available."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Install with: pip install tensorflow")
            return None
        
        config = {**self.model_configs['neural_network'], **kwargs}
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            config['hidden_layers'][0], 
            activation='relu', 
            input_shape=(input_dim,)
        ))
        model.add(layers.Dropout(config['dropout_rate']))
        
        # Hidden layers
        for units in config['hidden_layers'][1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(config['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_ensemble_model(self, X_train: np.ndarray) -> VotingRegressor:
        """Create advanced ensemble model with multiple algorithms."""
        estimators = []
        
        # Traditional models
        estimators.extend([
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('ridge', Ridge(alpha=1.0)),
            ('elastic', ElasticNet(alpha=0.1, random_state=42))
        ])
        
        # Advanced models if available
        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', self.create_xgboost_model()))
        
        if LIGHTGBM_AVAILABLE:
            estimators.append(('lgb', self.create_lightgbm_model()))
        
        # Neural network wrapper
        if TENSORFLOW_AVAILABLE:
            nn_wrapper = NeuralNetworkWrapper(input_dim=X_train.shape[1])
            estimators.append(('nn', nn_wrapper))
        
        return VotingRegressor(estimators=estimators)
    
    def create_stacked_model(self, X_train: np.ndarray) -> StackingRegressor:
        """Create stacked ensemble model."""
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42))
        ]
        
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', self.create_xgboost_model(n_estimators=50)))
        
        if LIGHTGBM_AVAILABLE:
            base_models.append(('lgb', self.create_lightgbm_model(n_estimators=50)))
        
        # Meta-learner
        meta_learner = Ridge(alpha=1.0)
        
        return StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5
        )
    
    def hyperparameter_tuning(self, model, X_train: np.ndarray, y_train: np.ndarray, 
                            param_grid: Dict) -> Any:
        """Perform hyperparameter tuning with time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_advanced_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           model_type: str = 'ensemble') -> Tuple[Any, Dict]:
        """Train advanced model with specified type."""
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        performance = {}
        
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = self.create_xgboost_model()
            
            # Hyperparameter tuning for XGBoost
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            model = self.hyperparameter_tuning(model, X_train_scaled, y_train, param_grid)
            
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = self.create_lightgbm_model()
            
            # Hyperparameter tuning for LightGBM
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2]
            }
            model = self.hyperparameter_tuning(model, X_train_scaled, y_train, param_grid)
            
        elif model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
            model = self.create_neural_network(X_train_scaled.shape[1])
            
            # Early stopping callback
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            # Train neural network
            history = model.fit(
                X_train_scaled, y_train,
                epochs=self.model_configs['neural_network']['epochs'],
                batch_size=self.model_configs['neural_network']['batch_size'],
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            performance['training_history'] = history.history
            
        elif model_type == 'ensemble':
            model = self.create_ensemble_model(X_train_scaled)
            model.fit(X_train_scaled, y_train)
            
        elif model_type == 'stacked':
            model = self.create_stacked_model(X_train_scaled)
            model.fit(X_train_scaled, y_train)
            
        else:
            # Fallback to gradient boosting
            model = GradientBoostingRegressor(n_estimators=200, random_state=42)
            model.fit(X_train_scaled, y_train)
        
        # Calculate performance metrics
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_train_scaled)
            performance.update({
                'mse': mean_squared_error(y_train, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
                'mae': mean_absolute_error(y_train, y_pred),
                'r2': r2_score(y_train, y_pred)
            })
        
        return model, scaler, performance
    
    def save_advanced_model(self, model: Any, scaler: Any, performance: Dict, 
                          model_name: str, model_type: str):
        """Save advanced model with metadata."""
        
        model_path = self.model_dir / f"{model_name}_{model_type}.pkl"
        scaler_path = self.model_dir / f"{model_name}_{model_type}_scaler.pkl"
        performance_path = self.model_dir / f"{model_name}_{model_type}_performance.json"
        
        # Save model
        if TENSORFLOW_AVAILABLE and hasattr(model, 'save'):
            # TensorFlow model
            model_dir = self.model_dir / f"{model_name}_{model_type}_tf"
            model.save(model_dir)
        else:
            # Scikit-learn compatible model
            joblib.dump(model, model_path)
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        
        # Save performance metrics
        import json
        with open(performance_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_performance = {}
            for key, value in performance.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_performance[key] = float(value)
                elif isinstance(value, dict):
                    serializable_performance[key] = value
                else:
                    serializable_performance[key] = str(value)
            
            json.dump(serializable_performance, f, indent=2)
        
        logger.info(f"Saved advanced model: {model_name}_{model_type}")
    
    def load_advanced_model(self, model_name: str, model_type: str) -> Tuple[Any, Any, Dict]:
        """Load advanced model with metadata."""
        
        model_path = self.model_dir / f"{model_name}_{model_type}.pkl"
        scaler_path = self.model_dir / f"{model_name}_{model_type}_scaler.pkl"
        performance_path = self.model_dir / f"{model_name}_{model_type}_performance.json"
        
        # Load TensorFlow model if exists
        tf_model_dir = self.model_dir / f"{model_name}_{model_type}_tf"
        if tf_model_dir.exists() and TENSORFLOW_AVAILABLE:
            model = keras.models.load_model(tf_model_dir)
        else:
            model = joblib.load(model_path)
        
        scaler = joblib.load(scaler_path)
        
        # Load performance metrics
        import json
        with open(performance_path, 'r') as f:
            performance = json.load(f)
        
        return model, scaler, performance


class NeuralNetworkWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to make TensorFlow models compatible with scikit-learn."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int] = [128, 64, 32], 
                 dropout_rate: float = 0.3, learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 32):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
    
    def fit(self, X, y):
        """Fit the neural network."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        self.model = keras.Sequential()
        
        # Input layer
        self.model.add(layers.Dense(
            self.hidden_layers[0], 
            activation='relu', 
            input_shape=(self.input_dim,)
        ))
        self.model.add(layers.Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            self.model.add(layers.Dense(units, activation='relu'))
            self.model.add(layers.Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(layers.Dense(1, activation='linear'))
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        # Train model
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()


class ModelComparison:
    """Compare performance of different model types."""
    
    def __init__(self):
        self.results = {}
    
    def compare_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Compare different model types on the same dataset."""
        
        advanced_ml = AdvancedMLModels()
        model_types = ['ensemble', 'stacked']
        
        if XGBOOST_AVAILABLE:
            model_types.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            model_types.append('lightgbm')
        if TENSORFLOW_AVAILABLE:
            model_types.append('neural_network')
        
        results = {}
        
        for model_type in model_types:
            try:
                logger.info(f"Training {model_type} model...")
                
                model, scaler, train_performance = advanced_ml.train_advanced_model(
                    X_train, y_train, model_type
                )
                
                # Test performance
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                
                test_performance = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
                
                results[model_type] = {
                    'train_performance': train_performance,
                    'test_performance': test_performance,
                    'model': model,
                    'scaler': scaler
                }
                
                logger.info(f"{model_type} - Test R²: {test_performance['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue
        
        return results
    
    def get_best_model(self, comparison_results: Dict, metric: str = 'r2') -> Tuple[str, Any, Any]:
        """Get the best performing model based on specified metric."""
        
        best_score = -np.inf if metric == 'r2' else np.inf
        best_model_type = None
        best_model = None
        best_scaler = None
        
        for model_type, results in comparison_results.items():
            score = results['test_performance'][metric]
            
            if (metric == 'r2' and score > best_score) or \
               (metric != 'r2' and score < best_score):
                best_score = score
                best_model_type = model_type
                best_model = results['model']
                best_scaler = results['scaler']
        
        logger.info(f"Best model: {best_model_type} with {metric}: {best_score:.4f}")
        
        return best_model_type, best_model, best_scaler


if __name__ == "__main__":
    # Example usage
    logger.info("Advanced ML Models initialized")
    
    # Check available libraries
    print("Available Advanced ML Libraries:")
    print(f"   XGBoost: {'Available' if XGBOOST_AVAILABLE else 'Not Available'}")
    print(f"   LightGBM: {'Available' if LIGHTGBM_AVAILABLE else 'Not Available'}")
    print(f"   TensorFlow: {'Available' if TENSORFLOW_AVAILABLE else 'Not Available'}")
    
    if not any([XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, TENSORFLOW_AVAILABLE]):
        print("\nInstall advanced libraries for enhanced performance:")
        print("   pip install xgboost lightgbm tensorflow")
