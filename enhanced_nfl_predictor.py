#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced NFL Betting Predictor - Consolidated & Improved System
Combines the best features from all modules into a single, powerful predictor.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Core imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Advanced ML imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedNFLPredictor:
    """Enhanced NFL betting predictor with advanced features."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the enhanced predictor."""
        self.config = config or self._default_config()
        self.db_url = self.config.get('database_url', "sqlite:///data/nfl_predictions.db")
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Directories
        self.model_dir = Path(self.config.get('model_directory', 'models/enhanced'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Advanced features
        self.feature_columns = self._define_feature_columns()
        self.target_columns = self._define_target_columns()
        
        # Load existing models
        self._load_models()
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            'model_types': ['random_forest', 'gradient_boosting'],
            'use_ensemble': True,
            'feature_engineering': True,
            'cross_validation': True,
            'hyperparameter_tuning': False,
            'min_samples': 50,
            'test_size': 0.2,
            'random_state': 42
        }
    
    def _define_feature_columns(self) -> Dict[str, List[str]]:
        """Define feature columns for each position."""
        return {
            'QB': [
                'is_home', 'week', 'season_week_ratio',
                'passing_attempts_avg', 'passing_completions_avg', 'completion_pct',
                'passing_yards_avg', 'passing_tds_avg', 'passing_ints_avg',
                'rushing_attempts_avg', 'rushing_yards_avg',
                'games_played', 'recent_form_3', 'recent_form_5'
            ],
            'RB': [
                'is_home', 'week', 'season_week_ratio',
                'rushing_attempts_avg', 'rushing_yards_avg', 'rushing_tds_avg',
                'targets_avg', 'receptions_avg', 'receiving_yards_avg',
                'games_played', 'recent_form_3', 'recent_form_5'
            ],
            'WR': [
                'is_home', 'week', 'season_week_ratio',
                'targets_avg', 'receptions_avg', 'receiving_yards_avg', 'receiving_tds_avg',
                'rushing_attempts_avg', 'games_played', 'recent_form_3', 'recent_form_5'
            ],
            'TE': [
                'is_home', 'week', 'season_week_ratio',
                'targets_avg', 'receptions_avg', 'receiving_yards_avg', 'receiving_tds_avg',
                'games_played', 'recent_form_3', 'recent_form_5'
            ]
        }
    
    def _define_target_columns(self) -> Dict[str, List[str]]:
        """Define target columns for each position."""
        return {
            'QB': ['fantasy_points', 'passing_yards', 'passing_touchdowns', 'passing_interceptions'],
            'RB': ['fantasy_points', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards'],
            'WR': ['fantasy_points', 'receptions', 'receiving_yards', 'receiving_touchdowns'],
            'TE': ['fantasy_points', 'receptions', 'receiving_yards', 'receiving_touchdowns']
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering."""
        if df.empty:
            return df
            
        # Sort by player and date for rolling calculations
        df = df.sort_values(['player_id', 'created_at'])
        
        # Add synthetic week feature based on created_at
        df['week'] = df.groupby('player_id').cumcount() + 1
        df['season_week_ratio'] = df['week'] / 18.0
        df['completion_pct'] = df['passing_completions'] / df['passing_attempts'].replace(0, np.nan)
        
        # Rolling averages
        for window in [3, 5]:
            for col in ['passing_yards', 'passing_touchdowns', 'rushing_yards', 'receiving_yards', 'fantasy_points_ppr']:
                if col in df.columns:
                    df[f'{col}_avg_{window}'] = df.groupby('player_id')[col].rolling(window, min_periods=1).mean().values
        
        # Recent form (last 3 and 5 games fantasy points average)
        df['recent_form_3'] = df.groupby('player_id')['fantasy_points_ppr'].rolling(3, min_periods=1).mean().values
        df['recent_form_5'] = df.groupby('player_id')['fantasy_points_ppr'].rolling(5, min_periods=1).mean().values
        
        # Games played this season
        df['games_played'] = df.groupby('player_id').cumcount() + 1
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def get_enhanced_training_data(self, position: str = None) -> pd.DataFrame:
        """Get enhanced training data with feature engineering."""
        with self.Session() as session:
            query = """
            SELECT 
                player_id,
                CASE 
                    WHEN player_id LIKE '%_qb' THEN 'QB'
                    WHEN player_id LIKE '%_rb' THEN 'RB'
                    WHEN player_id LIKE '%_wr' THEN 'WR'
                    WHEN player_id LIKE '%_te' THEN 'TE'
                    ELSE 'UNKNOWN'
                END as position,
                is_home,
                passing_attempts, passing_completions, passing_yards, passing_touchdowns, passing_interceptions,
                rushing_attempts, rushing_yards, rushing_touchdowns,
                targets, receptions, receiving_yards, receiving_touchdowns,
                fantasy_points_ppr,
                created_at
            FROM player_game_stats 
            WHERE fantasy_points_ppr > 0
            """
            
            if position:
                query += f" AND player_id LIKE '%_{position.lower()}'"
            
            query += " ORDER BY player_id, created_at"
            
            result = session.execute(text(query))
            data = result.fetchall()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = [
                'player_id', 'position', 'is_home',
                'passing_attempts', 'passing_completions', 'passing_yards', 'passing_touchdowns', 'passing_interceptions',
                'rushing_attempts', 'rushing_yards', 'rushing_touchdowns',
                'targets', 'receptions', 'receiving_yards', 'receiving_touchdowns',
                'fantasy_points_ppr', 'created_at'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            
            # Apply feature engineering
            if self.config.get('feature_engineering', True):
                df = self.engineer_features(df)
            
            return df
    
    def train_enhanced_models(self, position: str):
        """Train enhanced models with multiple algorithms."""
        logger.info(f"Training enhanced models for {position}...")
        
        df = self.get_enhanced_training_data(position)
        if df.empty or len(df) < self.config.get('min_samples', 50):
            logger.warning(f"Insufficient data for {position}: {len(df)} samples")
            return
        
        logger.info(f"Training with {len(df)} samples for {position}")
        
        # Get features and targets for this position
        feature_cols = self.feature_columns.get(position, [])
        target_cols = self.target_columns.get(position, [])
        
        # Prepare features
        available_features = [col for col in feature_cols if col in df.columns]
        if not available_features:
            logger.warning(f"No valid features found for {position}")
            return
            
        X = df[available_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[position] = scaler
        
        # Train models for each target
        position_models = {}
        for target in target_cols:
            if target == 'fantasy_points':
                target_col = 'fantasy_points_ppr'
            else:
                target_col = target
                
            if target_col not in df.columns:
                continue
                
            y = df[target_col].fillna(0)
            
            if y.std() == 0:
                logger.warning(f"No variance in {position} {target}")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=self.config.get('test_size', 0.2), 
                random_state=self.config.get('random_state', 42)
            )
            
            # Train multiple models
            models = {}
            scores = {}
            
            # Random Forest
            if 'random_forest' in self.config.get('model_types', []):
                rf = RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=42, n_jobs=-1
                )
                rf.fit(X_train, y_train)
                rf_score = r2_score(y_test, rf.predict(X_test))
                models['random_forest'] = rf
                scores['random_forest'] = rf_score
            
            # Gradient Boosting
            if 'gradient_boosting' in self.config.get('model_types', []):
                gb = GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42
                )
                gb.fit(X_train, y_train)
                gb_score = r2_score(y_test, gb.predict(X_test))
                models['gradient_boosting'] = gb
                scores['gradient_boosting'] = gb_score
            
            # XGBoost (if available)
            if 'xgboost' in self.config.get('model_types', []) and XGBOOST_AVAILABLE:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, n_jobs=-1
                )
                xgb_model.fit(X_train, y_train)
                xgb_score = r2_score(y_test, xgb_model.predict(X_test))
                models['xgboost'] = xgb_model
                scores['xgboost'] = xgb_score
            
            # Select best model or create ensemble
            if self.config.get('use_ensemble', True) and len(models) > 1:
                # Weighted ensemble based on performance
                weights = np.array(list(scores.values()))
                weights = np.maximum(weights, 0)  # Ensure non-negative
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                else:
                    weights = np.ones(len(weights)) / len(weights)
                
                ensemble_model = EnsembleModel(models, weights)
                ensemble_score = r2_score(y_test, ensemble_model.predict(X_test))
                
                position_models[target] = {
                    'model': ensemble_model,
                    'score': ensemble_score,
                    'feature_names': available_features
                }
                
                logger.info(f"✅ {position} {target}: Ensemble R² = {ensemble_score:.3f}")
            else:
                # Use best single model
                best_model_name = max(scores, key=scores.get)
                best_model = models[best_model_name]
                best_score = scores[best_model_name]
                
                position_models[target] = {
                    'model': best_model,
                    'score': best_score,
                    'feature_names': available_features
                }
                
                logger.info(f"✅ {position} {target}: {best_model_name} R² = {best_score:.3f}")
        
        # Save models
        self.models[position] = position_models
        self._save_models(position)
    
    def predict_player_enhanced(self, player_id: str, features: Dict[str, float] = None) -> Dict[str, Any]:
        """Make enhanced predictions for a player."""
        # Determine position
        position = self._get_player_position(player_id)
        if not position or position not in self.models:
            return {}
        
        # Get features
        if features is None:
            features = self._get_player_features(player_id)
        
        if not features:
            return {}
        
        # Prepare feature vector
        feature_cols = self.feature_columns.get(position, [])
        feature_vector = [features.get(col, 0) for col in feature_cols]
        
        # Scale features
        if position in self.scalers:
            feature_vector = self.scalers[position].transform([feature_vector])[0]
        
        # Make predictions
        predictions = {}
        confidence_scores = {}
        
        for target, model_info in self.models[position].items():
            try:
                pred = model_info['model'].predict([feature_vector])[0]
                predictions[target] = max(0, pred)  # Ensure non-negative
                confidence_scores[target] = min(0.95, max(0.3, model_info['score']))
            except Exception as e:
                logger.warning(f"Prediction failed for {position} {target}: {e}")
        
        return {
            'predictions': predictions,
            'confidence': confidence_scores,
            'position': position,
            'features_used': len(feature_vector)
        }
    
    def get_betting_recommendations_enhanced(self) -> List[Dict]:
        """Get enhanced betting recommendations."""
        recommendations = []
        
        # Get top players by recent performance
        with self.Session() as session:
            for position in ['QB', 'RB', 'WR', 'TE']:
                query = text("""
                    SELECT player_id, AVG(fantasy_points_ppr) as avg_points,
                           COUNT(*) as games_played
                    FROM player_game_stats 
                    WHERE player_id LIKE :pattern
                    AND fantasy_points_ppr > 0
                    AND created_at >= date('now', '-30 days')
                    GROUP BY player_id
                    HAVING games_played >= 3
                    ORDER BY avg_points DESC
                    LIMIT 5
                """)
                
                pattern = f"%_{position.lower()}"
                result = session.execute(query, {"pattern": pattern})
                players = result.fetchall()
                
                for player_id, avg_points, games_played in players:
                    prediction_result = self.predict_player_enhanced(player_id)
                    
                    if prediction_result and 'predictions' in prediction_result:
                        predictions = prediction_result['predictions']
                        confidence = prediction_result['confidence']
                        
                        if 'fantasy_points' in predictions:
                            fp_pred = predictions['fantasy_points']
                            fp_conf = confidence.get('fantasy_points', 0.5)
                            
                            # Enhanced betting logic
                            bet_rec = self._generate_betting_recommendation(
                                position, fp_pred, predictions, fp_conf
                            )
                            
                            if bet_rec:
                                recommendations.append({
                                    'player_id': player_id,
                                    'position': position,
                                    'predicted_fantasy_points': fp_pred,
                                    'confidence': fp_conf,
                                    'recent_avg': avg_points,
                                    'games_played': games_played,
                                    'recommendation': bet_rec,
                                    'all_predictions': predictions,
                                    'model_type': 'enhanced'
                                })
        
        # Sort by confidence * predicted value
        recommendations.sort(
            key=lambda x: x['confidence'] * x['predicted_fantasy_points'], 
            reverse=True
        )
        
        return recommendations[:15]
    
    def _generate_betting_recommendation(self, position: str, fp_pred: float, 
                                       all_preds: Dict, confidence: float) -> Optional[str]:
        """Generate specific betting recommendations."""
        if confidence < 0.4:
            return None
            
        recommendations = []
        
        # Fantasy points thresholds
        thresholds = {
            'QB': [(20, "Over 19.5"), (25, "Over 24.5"), (30, "Over 29.5")],
            'RB': [(15, "Over 14.5"), (20, "Over 19.5"), (25, "Over 24.5")],
            'WR': [(12, "Over 11.5"), (17, "Over 16.5"), (22, "Over 21.5")],
            'TE': [(10, "Over 9.5"), (15, "Over 14.5"), (20, "Over 19.5")]
        }
        
        for threshold, rec in thresholds.get(position, []):
            if fp_pred > threshold:
                recommendations.append(f"{rec} fantasy points")
        
        # Specific stat recommendations
        if position == 'QB':
            if all_preds.get('passing_yards', 0) > 275:
                recommendations.append("Over 274.5 passing yards")
            if all_preds.get('passing_touchdowns', 0) > 2.5:
                recommendations.append("Over 2.5 passing TDs")
        elif position == 'RB':
            if all_preds.get('rushing_yards', 0) > 80:
                recommendations.append("Over 79.5 rushing yards")
        elif position in ['WR', 'TE']:
            if all_preds.get('receiving_yards', 0) > 75:
                recommendations.append("Over 74.5 receiving yards")
            if all_preds.get('receptions', 0) > 6:
                recommendations.append("Over 5.5 receptions")
        
        return recommendations[0] if recommendations else None
    
    def _get_player_position(self, player_id: str) -> Optional[str]:
        """Get player position from ID."""
        if '_qb' in player_id.lower():
            return 'QB'
        elif '_rb' in player_id.lower():
            return 'RB'
        elif '_wr' in player_id.lower():
            return 'WR'
        elif '_te' in player_id.lower():
            return 'TE'
        return None
    
    def _get_player_features(self, player_id: str) -> Dict[str, float]:
        """Get recent features for a player."""
        with self.Session() as session:
            query = text("""
                SELECT * FROM player_game_stats 
                WHERE player_id = :player_id 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            
            result = session.execute(query, {"player_id": player_id})
            rows = result.fetchall()
            
            if not rows:
                return {}
            
            # Calculate features from recent games
            features = {
                'is_home': 0.5,  # neutral
                'week': len(rows),  # Use number of games as proxy for week
                'season_week_ratio': len(rows) / 18.0,
                'games_played': len(rows)
            }
            
            # Calculate averages
            for stat in ['passing_attempts', 'passing_completions', 'passing_yards', 
                        'passing_touchdowns', 'passing_interceptions', 'rushing_attempts',
                        'rushing_yards', 'rushing_touchdowns', 'targets', 'receptions',
                        'receiving_yards', 'receiving_touchdowns']:
                values = [getattr(row, stat, 0) or 0 for row in rows]
                features[f'{stat}_avg'] = np.mean(values) if values else 0
            
            # Completion percentage
            if features.get('passing_attempts_avg', 0) > 0:
                features['completion_pct'] = features['passing_completions_avg'] / features['passing_attempts_avg']
            else:
                features['completion_pct'] = 0
            
            # Recent form
            fp_values = [getattr(row, 'fantasy_points_ppr', 0) or 0 for row in rows]
            features['recent_form_3'] = np.mean(fp_values[:3]) if len(fp_values) >= 3 else np.mean(fp_values)
            features['recent_form_5'] = np.mean(fp_values[:5]) if len(fp_values) >= 5 else np.mean(fp_values)
            
            return features
    
    def _load_models(self):
        """Load saved models."""
        for model_file in self.model_dir.glob("*.pkl"):
            try:
                parts = model_file.stem.split('_')
                if len(parts) >= 2:
                    position = parts[0]
                    target = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                    
                    model_data = joblib.load(model_file)
                    
                    if position not in self.models:
                        self.models[position] = {}
                    
                    self.models[position][target] = model_data
                    logger.info(f"Loaded model: {position}_{target}")
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
    
    def _save_models(self, position: str):
        """Save models for a position."""
        if position not in self.models:
            return
            
        for target, model_info in self.models[position].items():
            model_path = self.model_dir / f"{position}_{target}_enhanced.pkl"
            try:
                joblib.dump(model_info, model_path)
                logger.info(f"Saved model: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save {model_path}: {e}")
        
        # Save scaler
        if position in self.scalers:
            scaler_path = self.model_dir / f"{position}_scaler.pkl"
            joblib.dump(self.scalers[position], scaler_path)
    
    def train_all_enhanced_models(self):
        """Train enhanced models for all positions."""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            try:
                self.train_enhanced_models(position)
            except Exception as e:
                logger.error(f"Failed to train {position} models: {e}")
        
        logger.info("✅ Enhanced model training completed!")
    
    def display_enhanced_recommendations(self):
        """Display enhanced betting recommendations."""
        print("🏈 ENHANCED NFL BETTING RECOMMENDATIONS")
        print("=" * 60)
        
        if not self.models:
            print("❌ No trained models found. Run train_all_enhanced_models() first.")
            return
        
        print(f"✅ Loaded enhanced models for {len(self.models)} positions")
        print()
        
        recommendations = self.get_betting_recommendations_enhanced()
        
        if not recommendations:
            print("❌ No betting recommendations available.")
            return
        
        print("🎯 TOP ENHANCED BETTING RECOMMENDATIONS:")
        print("-" * 50)
        
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        
        for i, rec in enumerate(recommendations, 1):
            emoji = position_emoji.get(rec['position'], '⚡')
            
            print(f"{i}. {rec['player_id']} ({rec['position']}) {emoji}")
            print(f"   Predicted Fantasy Points: {rec['predicted_fantasy_points']:.1f}")
            print(f"   Confidence: {rec['confidence']:.1%}")
            print(f"   Recent Average: {rec['recent_avg']:.1f} ({rec['games_played']} games)")
            print(f"   💡 RECOMMENDATION: {rec['recommendation']}")
            
            # Show additional predictions
            other_preds = rec.get('all_predictions', {})
            for stat, value in other_preds.items():
                if stat != 'fantasy_points' and value > 0:
                    print(f"   {stat.replace('_', ' ').title()}: {value:.1f}")
            print()
        
        print("⚠️  DISCLAIMER: Enhanced predictions based on advanced ML models.")
        print("   Always gamble responsibly and within your means.")


class EnsembleModel:
    """Ensemble model combining multiple algorithms."""
    
    def __init__(self, models: Dict, weights: np.ndarray):
        self.models = models
        self.weights = weights
        self.model_names = list(models.keys())
    
    def predict(self, X):
        """Make ensemble predictions."""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred


def main():
    """Main function."""
    try:
        # Initialize enhanced predictor
        config = {
            'model_types': ['random_forest', 'gradient_boosting', 'xgboost'],
            'use_ensemble': True,
            'feature_engineering': True,
            'min_samples': 30
        }
        
        predictor = EnhancedNFLPredictor(config)
        
        # Check if database exists
        if not Path("data/nfl_predictions.db").exists():
            print("❌ Database not found. Please run setup_database.py first.")
            return
        
        # Check if models exist, if not train them
        if not predictor.models:
            print("🤖 No enhanced models found. Training new models...")
            predictor.train_all_enhanced_models()
            print()
        
        # Display enhanced recommendations
        predictor.display_enhanced_recommendations()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        print("Please check the logs and try again.")


if __name__ == "__main__":
    main()
