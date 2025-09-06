#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Enhanced NFL Betting Predictor
Consolidated, clean, and fully functional system with advanced features.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalNFLPredictor:
    """Final enhanced NFL betting predictor with comprehensive features."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the predictor."""
        self.config = config or {
            'model_types': ['random_forest', 'gradient_boosting'],
            'use_ensemble': True,
            'min_samples': 30,
            'test_size': 0.2,
            'random_state': 42
        }
        
        self.db_url = "sqlite:///data/nfl_predictions.db"
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        
        # Directories
        self.model_dir = Path('models/final')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing models
        self._load_models()
    
    def get_training_data(self, position: str = None) -> pd.DataFrame:
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
                passing_attempts, passing_completions, passing_yards, 
                passing_touchdowns, passing_interceptions,
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
                'passing_attempts', 'passing_completions', 'passing_yards', 
                'passing_touchdowns', 'passing_interceptions',
                'rushing_attempts', 'rushing_yards', 'rushing_touchdowns',
                'targets', 'receptions', 'receiving_yards', 'receiving_touchdowns',
                'fantasy_points_ppr', 'created_at'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            
            # Feature engineering
            df = self._engineer_features(df)
            
            return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering."""
        if df.empty:
            return df
        
        # Sort by player and date
        df = df.sort_values(['player_id', 'created_at'])
        
        # Basic derived features
        df['completion_pct'] = df['passing_completions'] / df['passing_attempts'].replace(0, np.nan)
        df['yards_per_attempt'] = df['passing_yards'] / df['passing_attempts'].replace(0, np.nan)
        df['yards_per_carry'] = df['rushing_yards'] / df['rushing_attempts'].replace(0, np.nan)
        df['yards_per_reception'] = df['receiving_yards'] / df['receptions'].replace(0, np.nan)
        df['catch_rate'] = df['receptions'] / df['targets'].replace(0, np.nan)
        
        # Rolling averages (last 3 and 5 games)
        for window in [3, 5]:
            for col in ['fantasy_points_ppr', 'passing_yards', 'passing_touchdowns', 
                       'rushing_yards', 'receiving_yards', 'targets', 'receptions']:
                if col in df.columns:
                    df[f'{col}_avg_{window}'] = (df.groupby('player_id')[col]
                                               .rolling(window, min_periods=1)
                                               .mean().values)
        
        # Trend features (improvement/decline over last 5 games)
        for col in ['fantasy_points_ppr', 'passing_yards', 'rushing_yards', 'receiving_yards']:
            if col in df.columns:
                df[f'{col}_trend'] = (df.groupby('player_id')[col]
                                    .rolling(5, min_periods=2)
                                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                                    .values)
        
        # Games played this season
        df['games_played'] = df.groupby('player_id').cumcount() + 1
        
        # Home/away performance differential
        df['home_avg_fp'] = df.groupby(['player_id', 'is_home'])['fantasy_points_ppr'].expanding().mean().values
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def train_enhanced_models(self, position: str):
        """Train enhanced models for a position."""
        logger.info(f"Training enhanced models for {position}...")
        
        df = self.get_training_data(position)
        if df.empty or len(df) < self.config.get('min_samples', 30):
            logger.warning(f"Insufficient data for {position}: {len(df)} samples")
            return
        
        logger.info(f"Training with {len(df)} samples for {position}")
        
        # Define features based on position
        if position == 'QB':
            feature_cols = [
                'is_home', 'games_played', 'completion_pct', 'yards_per_attempt',
                'passing_yards_avg_3', 'passing_yards_avg_5', 'passing_touchdowns_avg_3',
                'fantasy_points_ppr_avg_3', 'fantasy_points_ppr_avg_5', 'fantasy_points_ppr_trend'
            ]
            target_cols = ['fantasy_points_ppr', 'passing_yards', 'passing_touchdowns']
        elif position == 'RB':
            feature_cols = [
                'is_home', 'games_played', 'yards_per_carry', 'catch_rate',
                'rushing_yards_avg_3', 'rushing_yards_avg_5', 'receiving_yards_avg_3',
                'fantasy_points_ppr_avg_3', 'fantasy_points_ppr_avg_5', 'fantasy_points_ppr_trend'
            ]
            target_cols = ['fantasy_points_ppr', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards']
        elif position == 'WR':
            feature_cols = [
                'is_home', 'games_played', 'catch_rate', 'yards_per_reception',
                'targets_avg_3', 'targets_avg_5', 'receiving_yards_avg_3', 'receiving_yards_avg_5',
                'fantasy_points_ppr_avg_3', 'fantasy_points_ppr_avg_5', 'fantasy_points_ppr_trend'
            ]
            target_cols = ['fantasy_points_ppr', 'receptions', 'receiving_yards', 'receiving_touchdowns']
        else:  # TE
            feature_cols = [
                'is_home', 'games_played', 'catch_rate', 'yards_per_reception',
                'targets_avg_3', 'receiving_yards_avg_3', 'receptions_avg_3',
                'fantasy_points_ppr_avg_3', 'fantasy_points_ppr_avg_5', 'fantasy_points_ppr_trend'
            ]
            target_cols = ['fantasy_points_ppr', 'receptions', 'receiving_yards']
        
        # Filter available features
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
            if target not in df.columns:
                continue
            
            y = df[target].fillna(0)
            
            if y.std() == 0:
                logger.warning(f"No variance in {position} {target}")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=self.config.get('test_size', 0.2), 
                random_state=self.config.get('random_state', 42)
            )
            
            # Train ensemble of models
            models = {}
            scores = {}
            
            # Random Forest
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            rf_score = r2_score(y_test, rf.predict(X_test))
            models['random_forest'] = rf
            scores['random_forest'] = rf_score
            
            # Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                min_samples_split=5, random_state=42
            )
            gb.fit(X_train, y_train)
            gb_score = r2_score(y_test, gb.predict(X_test))
            models['gradient_boosting'] = gb
            scores['gradient_boosting'] = gb_score
            
            # Create ensemble
            if self.config.get('use_ensemble', True) and len(models) > 1:
                weights = np.array(list(scores.values()))
                weights = np.maximum(weights, 0.1)  # Minimum weight
                weights = weights / weights.sum()
                
                ensemble_model = EnsembleModel(models, weights)
                ensemble_score = r2_score(y_test, ensemble_model.predict(X_test))
                
                position_models[target] = {
                    'model': ensemble_model,
                    'score': ensemble_score,
                    'features': available_features,
                    'individual_scores': scores
                }
                
                logger.info(f"✅ {position} {target}: Ensemble R² = {ensemble_score:.3f}")
            else:
                best_model_name = max(scores, key=scores.get)
                best_model = models[best_model_name]
                best_score = scores[best_model_name]
                
                position_models[target] = {
                    'model': best_model,
                    'score': best_score,
                    'features': available_features
                }
                
                logger.info(f"✅ {position} {target}: {best_model_name} R² = {best_score:.3f}")
        
        # Save models
        self.models[position] = position_models
        self._save_models(position)
    
    def predict_player(self, player_id: str) -> Dict[str, Any]:
        """Make predictions for a player."""
        position = self._get_player_position(player_id)
        if not position or position not in self.models:
            return {}
        
        # Get player features
        features = self._get_player_features(player_id, position)
        if not features:
            return {}
        
        # Make predictions
        predictions = {}
        confidence_scores = {}
        
        for target, model_info in self.models[position].items():
            try:
                feature_vector = [features.get(feat, 0) for feat in model_info['features']]
                
                # Scale features
                if position in self.scalers:
                    feature_vector = self.scalers[position].transform([feature_vector])
                else:
                    feature_vector = np.array([feature_vector])
                
                pred = model_info['model'].predict(feature_vector)[0]
                predictions[target] = max(0, pred)
                confidence_scores[target] = min(0.95, max(0.3, model_info['score']))
                
            except Exception as e:
                logger.warning(f"Prediction failed for {position} {target}: {e}")
        
        return {
            'predictions': predictions,
            'confidence': confidence_scores,
            'position': position
        }
    
    def get_betting_recommendations(self) -> List[Dict]:
        """Get comprehensive betting recommendations."""
        recommendations = []
        
        with self.Session() as session:
            for position in ['QB', 'RB', 'WR', 'TE']:
                # Get top performers with recent activity
                query = text("""
                    SELECT player_id, AVG(fantasy_points_ppr) as avg_points,
                           COUNT(*) as games_played,
                           MAX(created_at) as last_game
                    FROM player_game_stats 
                    WHERE player_id LIKE :pattern
                    AND fantasy_points_ppr > 0
                    GROUP BY player_id
                    HAVING games_played >= 3
                    ORDER BY avg_points DESC
                    LIMIT 5
                """)
                
                pattern = f"%_{position.lower()}"
                result = session.execute(query, {"pattern": pattern})
                players = result.fetchall()
                
                for player_id, avg_points, games_played, last_game in players:
                    prediction_result = self.predict_player(player_id)
                    
                    if prediction_result and 'predictions' in prediction_result:
                        predictions = prediction_result['predictions']
                        confidence = prediction_result['confidence']
                        
                        if 'fantasy_points_ppr' in predictions:
                            fp_pred = predictions['fantasy_points_ppr']
                            fp_conf = confidence.get('fantasy_points_ppr', 0.5)
                            
                            # Generate betting recommendations
                            bet_recs = self._generate_betting_recommendations(
                                position, fp_pred, predictions, fp_conf
                            )
                            
                            if bet_recs:
                                recommendations.append({
                                    'player_id': player_id,
                                    'position': position,
                                    'predicted_fantasy_points': fp_pred,
                                    'confidence': fp_conf,
                                    'recent_avg': avg_points,
                                    'games_played': games_played,
                                    'recommendations': bet_recs,
                                    'all_predictions': predictions,
                                    'value_score': fp_conf * fp_pred
                                })
        
        # Sort by value score (confidence * predicted points)
        recommendations.sort(key=lambda x: x['value_score'], reverse=True)
        return recommendations[:12]
    
    def _generate_betting_recommendations(self, position: str, fp_pred: float, 
                                        all_preds: Dict, confidence: float) -> List[str]:
        """Generate specific betting recommendations."""
        if confidence < 0.4:
            return []
        
        recommendations = []
        
        # Fantasy points recommendations
        thresholds = {
            'QB': [(18, "Over 17.5"), (22, "Over 21.5"), (26, "Over 25.5")],
            'RB': [(12, "Over 11.5"), (16, "Over 15.5"), (20, "Over 19.5")],
            'WR': [(10, "Over 9.5"), (14, "Over 13.5"), (18, "Over 17.5")],
            'TE': [(8, "Over 7.5"), (12, "Over 11.5"), (16, "Over 15.5")]
        }
        
        for threshold, rec in thresholds.get(position, []):
            if fp_pred > threshold and confidence > 0.6:
                recommendations.append(f"{rec} fantasy points")
                break
        
        # Position-specific recommendations
        if position == 'QB':
            if all_preds.get('passing_yards', 0) > 275 and confidence > 0.65:
                recommendations.append("Over 274.5 passing yards")
            if all_preds.get('passing_touchdowns', 0) > 2.2 and confidence > 0.6:
                recommendations.append("Over 1.5 passing TDs")
        elif position == 'RB':
            if all_preds.get('rushing_yards', 0) > 75 and confidence > 0.65:
                recommendations.append("Over 74.5 rushing yards")
            if all_preds.get('rushing_touchdowns', 0) > 0.8 and confidence > 0.6:
                recommendations.append("Over 0.5 rushing TDs")
        elif position in ['WR', 'TE']:
            if all_preds.get('receiving_yards', 0) > 70 and confidence > 0.65:
                recommendations.append("Over 69.5 receiving yards")
            if all_preds.get('receptions', 0) > 5.5 and confidence > 0.6:
                recommendations.append("Over 4.5 receptions")
        
        return recommendations
    
    def _get_player_position(self, player_id: str) -> Optional[str]:
        """Extract position from player ID."""
        if '_qb' in player_id.lower():
            return 'QB'
        elif '_rb' in player_id.lower():
            return 'RB'
        elif '_wr' in player_id.lower():
            return 'WR'
        elif '_te' in player_id.lower():
            return 'TE'
        return None
    
    def _get_player_features(self, player_id: str, position: str) -> Dict[str, float]:
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
            
            # Calculate features
            features = {
                'is_home': 0.5,  # neutral assumption
                'games_played': len(rows)
            }
            
            # Calculate averages and derived stats
            stats = ['passing_attempts', 'passing_completions', 'passing_yards', 
                    'passing_touchdowns', 'rushing_attempts', 'rushing_yards', 
                    'rushing_touchdowns', 'targets', 'receptions', 'receiving_yards', 
                    'receiving_touchdowns', 'fantasy_points_ppr']
            
            for stat in stats:
                values = [getattr(row, stat, 0) or 0 for row in rows]
                if values:
                    features[f'{stat}_avg_3'] = np.mean(values[:3])
                    features[f'{stat}_avg_5'] = np.mean(values[:5])
                    
                    # Trend calculation
                    if len(values) >= 3:
                        x = np.arange(len(values))
                        trend = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0
                        features[f'{stat}_trend'] = trend
            
            # Derived features
            if features.get('passing_attempts_avg_3', 0) > 0:
                features['completion_pct'] = (features.get('passing_completions_avg_3', 0) / 
                                            features['passing_attempts_avg_3'])
                features['yards_per_attempt'] = (features.get('passing_yards_avg_3', 0) / 
                                               features['passing_attempts_avg_3'])
            
            if features.get('rushing_attempts_avg_3', 0) > 0:
                features['yards_per_carry'] = (features.get('rushing_yards_avg_3', 0) / 
                                             features['rushing_attempts_avg_3'])
            
            if features.get('targets_avg_3', 0) > 0:
                features['catch_rate'] = (features.get('receptions_avg_3', 0) / 
                                        features['targets_avg_3'])
            
            if features.get('receptions_avg_3', 0) > 0:
                features['yards_per_reception'] = (features.get('receiving_yards_avg_3', 0) / 
                                                 features['receptions_avg_3'])
            
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
            model_path = self.model_dir / f"{position}_{target}_final.pkl"
            try:
                joblib.dump(model_info, model_path)
                logger.info(f"Saved model: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save {model_path}: {e}")
        
        # Save scaler
        if position in self.scalers:
            scaler_path = self.model_dir / f"{position}_scaler_final.pkl"
            joblib.dump(self.scalers[position], scaler_path)
    
    def train_all_models(self):
        """Train models for all positions."""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            try:
                self.train_enhanced_models(position)
            except Exception as e:
                logger.error(f"Failed to train {position} models: {e}")
        
        logger.info("✅ All model training completed!")
    
    def display_detailed_player_analysis(self, player_id: str):
        """Display comprehensive analysis for a specific player."""
        position = self._get_player_position(player_id)
        if not position:
            print(f"❌ Could not determine position for {player_id}")
            return
        
        print(f"🏈 DETAILED PLAYER ANALYSIS: {player_id.upper()}")
        print("=" * 80)
        
        # Get prediction results
        prediction_result = self.predict_player(player_id)
        if not prediction_result or not prediction_result.get('predictions'):
            print("❌ No predictions available for this player.")
            return
        
        predictions = prediction_result['predictions']
        confidence = prediction_result['confidence']
        
        # Get historical data
        historical_data = self._get_player_historical_data(player_id)
        
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        emoji = position_emoji.get(position, '⚡')
        
        print(f"{emoji} PLAYER: {player_id} ({position})")
        print(f"📊 GAMES ANALYZED: {len(historical_data)} recent games")
        print()
        
        # Display all predictions with confidence and historical comparison
        print("🔮 PREDICTED STATISTICS:")
        print("-" * 50)
        
        # Fantasy Points (always first)
        if 'fantasy_points_ppr' in predictions:
            fp_pred = predictions['fantasy_points_ppr']
            fp_conf = confidence.get('fantasy_points_ppr', 0.5)
            fp_avg = np.mean([g.get('fantasy_points_ppr', 0) for g in historical_data]) if historical_data else 0
            
            print(f"🏆 Fantasy Points (PPR)")
            print(f"   Predicted: {fp_pred:.1f} points")
            print(f"   Confidence: {fp_conf:.1%}")
            print(f"   Recent Avg: {fp_avg:.1f} points")
            print(f"   Difference: {fp_pred - fp_avg:+.1f} points")
            print()
        
        # Position-specific stats
        if position == 'QB':
            self._display_qb_stats(predictions, confidence, historical_data)
        elif position == 'RB':
            self._display_rb_stats(predictions, confidence, historical_data)
        elif position == 'WR':
            self._display_wr_stats(predictions, confidence, historical_data)
        elif position == 'TE':
            self._display_te_stats(predictions, confidence, historical_data)
        
        # Betting recommendations
        bet_recs = self._generate_betting_recommendations(position, fp_pred, predictions, fp_conf)
        if bet_recs:
            print("💡 BETTING RECOMMENDATIONS:")
            print("-" * 30)
            for i, rec in enumerate(bet_recs, 1):
                print(f"   {i}. {rec}")
            print()
        
        # Performance trends
        self._display_performance_trends(historical_data)
        
        print("⚠️  DISCLAIMER: Predictions based on historical data and ML models.")
        print("   Always gamble responsibly and within your means.")
    
    def _display_qb_stats(self, predictions: Dict, confidence: Dict, historical_data: List):
        """Display QB-specific statistics."""
        stats_config = [
            ('passing_yards', '🎯 Passing Yards', 'yards'),
            ('passing_touchdowns', '🏈 Passing Touchdowns', 'TDs'),
            ('passing_interceptions', '❌ Interceptions', 'INTs')
        ]
        
        for stat_key, display_name, unit in stats_config:
            if stat_key in predictions:
                pred_val = predictions[stat_key]
                conf_val = confidence.get(stat_key, 0.5)
                hist_avg = np.mean([g.get(stat_key, 0) for g in historical_data]) if historical_data else 0
                
                print(f"{display_name}")
                print(f"   Predicted: {pred_val:.1f} {unit}")
                print(f"   Confidence: {conf_val:.1%}")
                print(f"   Recent Avg: {hist_avg:.1f} {unit}")
                print(f"   Difference: {pred_val - hist_avg:+.1f} {unit}")
                print()
    
    def _display_rb_stats(self, predictions: Dict, confidence: Dict, historical_data: List):
        """Display RB-specific statistics."""
        stats_config = [
            ('rushing_yards', '🏃 Rushing Yards', 'yards'),
            ('rushing_touchdowns', '🏈 Rushing Touchdowns', 'TDs'),
            ('receiving_yards', '🙌 Receiving Yards', 'yards'),
            ('receptions', '✋ Receptions', 'catches')
        ]
        
        for stat_key, display_name, unit in stats_config:
            if stat_key in predictions:
                pred_val = predictions[stat_key]
                conf_val = confidence.get(stat_key, 0.5)
                hist_avg = np.mean([g.get(stat_key, 0) for g in historical_data]) if historical_data else 0
                
                print(f"{display_name}")
                print(f"   Predicted: {pred_val:.1f} {unit}")
                print(f"   Confidence: {conf_val:.1%}")
                print(f"   Recent Avg: {hist_avg:.1f} {unit}")
                print(f"   Difference: {pred_val - hist_avg:+.1f} {unit}")
                print()
    
    def _display_wr_stats(self, predictions: Dict, confidence: Dict, historical_data: List):
        """Display WR-specific statistics."""
        stats_config = [
            ('receiving_yards', '🙌 Receiving Yards', 'yards'),
            ('receptions', '✋ Receptions', 'catches'),
            ('receiving_touchdowns', '🏈 Receiving Touchdowns', 'TDs'),
            ('targets', '🎯 Targets', 'targets')
        ]
        
        for stat_key, display_name, unit in stats_config:
            if stat_key in predictions:
                pred_val = predictions[stat_key]
                conf_val = confidence.get(stat_key, 0.5)
                hist_avg = np.mean([g.get(stat_key, 0) for g in historical_data]) if historical_data else 0
                
                print(f"{display_name}")
                print(f"   Predicted: {pred_val:.1f} {unit}")
                print(f"   Confidence: {conf_val:.1%}")
                print(f"   Recent Avg: {hist_avg:.1f} {unit}")
                print(f"   Difference: {pred_val - hist_avg:+.1f} {unit}")
                print()
    
    def _display_te_stats(self, predictions: Dict, confidence: Dict, historical_data: List):
        """Display TE-specific statistics."""
        stats_config = [
            ('receiving_yards', '🙌 Receiving Yards', 'yards'),
            ('receptions', '✋ Receptions', 'catches'),
            ('receiving_touchdowns', '🏈 Receiving Touchdowns', 'TDs')
        ]
        
        for stat_key, display_name, unit in stats_config:
            if stat_key in predictions:
                pred_val = predictions[stat_key]
                conf_val = confidence.get(stat_key, 0.5)
                hist_avg = np.mean([g.get(stat_key, 0) for g in historical_data]) if historical_data else 0
                
                print(f"{display_name}")
                print(f"   Predicted: {pred_val:.1f} {unit}")
                print(f"   Confidence: {conf_val:.1%}")
                print(f"   Recent Avg: {hist_avg:.1f} {unit}")
                print(f"   Difference: {pred_val - hist_avg:+.1f} {unit}")
                print()
    
    def _get_player_historical_data(self, player_id: str) -> List[Dict]:
        """Get detailed historical data for a player."""
        with self.Session() as session:
            query = text("""
                SELECT * FROM player_game_stats 
                WHERE player_id = :player_id 
                ORDER BY created_at DESC 
                LIMIT 10
            """)
            
            result = session.execute(query, {"player_id": player_id})
            rows = result.fetchall()
            
            historical_data = []
            for row in rows:
                game_data = {}
                for column in row._mapping.keys():
                    game_data[column] = getattr(row, column, 0) or 0
                historical_data.append(game_data)
            
            return historical_data
    
    def _display_performance_trends(self, historical_data: List[Dict]):
        """Display performance trends analysis."""
        if len(historical_data) < 3:
            return
        
        print("📈 PERFORMANCE TRENDS (Last 10 Games):")
        print("-" * 40)
        
        # Fantasy points trend
        fp_values = [g.get('fantasy_points_ppr', 0) for g in historical_data]
        fp_trend = np.polyfit(range(len(fp_values)), fp_values[::-1], 1)[0] if len(fp_values) > 1 else 0
        
        trend_emoji = "📈" if fp_trend > 0.5 else "📉" if fp_trend < -0.5 else "➡️"
        print(f"{trend_emoji} Fantasy Points Trend: {fp_trend:+.1f} points per game")
        
        # Recent form
        recent_3 = np.mean(fp_values[:3]) if len(fp_values) >= 3 else 0
        recent_5 = np.mean(fp_values[:5]) if len(fp_values) >= 5 else 0
        season_avg = np.mean(fp_values) if fp_values else 0
        
        print(f"🔥 Last 3 Games Avg: {recent_3:.1f} points")
        print(f"📊 Last 5 Games Avg: {recent_5:.1f} points")
        print(f"📋 Season Average: {season_avg:.1f} points")
        
        # Consistency analysis
        fp_std = np.std(fp_values) if len(fp_values) > 1 else 0
        consistency = "High" if fp_std < 5 else "Medium" if fp_std < 10 else "Low"
        print(f"🎯 Consistency: {consistency} (σ = {fp_std:.1f})")
        print()
    
    def _generate_betting_recommendations(self, position: str, fp_pred: float, predictions: Dict, confidence: float) -> List[str]:
        """Generate specific betting recommendations for a player."""
        recommendations = []
        
        # Fantasy points recommendations
        if fp_pred > 20 and confidence > 0.7:
            recommendations.append(f"Strong play for fantasy lineups (20+ points predicted)")
        elif fp_pred > 15 and confidence > 0.6:
            recommendations.append(f"Good value play for tournaments")
        
        # Position-specific recommendations
        if position == 'QB':
            if predictions.get('passing_yards', 0) > 300:
                recommendations.append("Consider Over on passing yards")
            if predictions.get('passing_touchdowns', 0) > 2.5:
                recommendations.append("Consider Over on passing TDs")
        
        elif position == 'RB':
            if predictions.get('rushing_yards', 0) > 100:
                recommendations.append("Consider Over on rushing yards")
            if predictions.get('rushing_touchdowns', 0) > 1:
                recommendations.append("Consider anytime TD scorer")
        
        elif position in ['WR', 'TE']:
            if predictions.get('receiving_yards', 0) > 80:
                recommendations.append("Consider Over on receiving yards")
            if predictions.get('receptions', 0) > 6:
                recommendations.append("Consider Over on receptions")
        
        return recommendations
    
    def display_recommendations(self):
        """Display comprehensive betting recommendations."""
        print("🏈 FINAL NFL BETTING RECOMMENDATIONS")
        print("=" * 70)
        
        if not self.models:
            print("❌ No trained models found. Run train_all_models() first.")
            return
        
        print(f"✅ Loaded enhanced models for {len(self.models)} positions")
        print()
        
        recommendations = self.get_betting_recommendations()
        
        if not recommendations:
            print("❌ No betting recommendations available.")
            return
        
        print("🎯 TOP BETTING RECOMMENDATIONS:")
        print("-" * 60)
        
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        
        for i, rec in enumerate(recommendations, 1):
            emoji = position_emoji.get(rec['position'], '⚡')
            
            print(f"{i}. {rec['player_id']} ({rec['position']}) {emoji}")
            print(f"   Predicted Fantasy Points: {rec['predicted_fantasy_points']:.1f}")
            print(f"   Confidence: {rec['confidence']:.1%}")
            print(f"   Recent Average: {rec['recent_avg']:.1f} ({rec['games_played']} games)")
            print(f"   Value Score: {rec['value_score']:.1f}")
            
            if rec['recommendations']:
                print(f"   💡 RECOMMENDATIONS:")
                for bet_rec in rec['recommendations']:
                    print(f"      • {bet_rec}")
            
            # Show ALL predicted stats in organized format
            other_preds = rec.get('all_predictions', {})
            if other_preds:
                print(f"   📊 ALL PREDICTED STATS:")
                for stat, value in sorted(other_preds.items()):
                    if stat != 'fantasy_points_ppr' and value > 0:
                        formatted_stat = stat.replace('_', ' ').title()
                        print(f"      {formatted_stat}: {value:.1f}")
            print()
        
        print("\n💡 TIP: For detailed analysis of any player, use:")
        print("   predictor.display_detailed_player_analysis('player_name_position')")
        print("\n⚠️  DISCLAIMER: Advanced ML predictions for entertainment purposes.")
        print("   Always gamble responsibly and within your means.")
        print("   Past performance does not guarantee future results.")


class EnsembleModel:
    """Ensemble model combining multiple algorithms with weighted averaging."""
    
    def __init__(self, models: Dict, weights: np.ndarray):
        self.models = models
        self.weights = weights
        self.model_names = list(models.keys())
    
    def predict(self, X):
        """Make ensemble predictions."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
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
        # Initialize the enhanced predictor
        predictor = FinalNFLPredictor()
        
        # Load existing models or train new ones
        predictor._load_models()
        
        # Display comprehensive recommendations
        predictor.display_recommendations()
        
        # Demo detailed player analysis
        print("\n" + "="*80)
        print("🎯 DETAILED PLAYER ANALYSIS DEMO")
        print("="*80)
        
        # Get some sample players for detailed analysis
        with predictor.Session() as session:
            sample_query = text("""
                SELECT DISTINCT player_id, COUNT(*) as games
                FROM player_game_stats 
                WHERE fantasy_points_ppr > 10
                GROUP BY player_id 
                HAVING games >= 3
                ORDER BY AVG(fantasy_points_ppr) DESC 
                LIMIT 3
            """)
            
            result = session.execute(sample_query)
            sample_players = [row[0] for row in result.fetchall()]
        
        # Show detailed analysis for top players
        for i, player_id in enumerate(sample_players, 1):
            print(f"\n🔍 SAMPLE ANALYSIS #{i}:")
            print("-" * 50)
            predictor.display_detailed_player_analysis(player_id)
            
            if i < len(sample_players):
                print("\n" + "."*50 + "\n")
        
        print("\n💡 USAGE TIPS:")
        print("• Run predictor.display_detailed_player_analysis('player_name') for any specific player")
        print("• Use predictor.display_recommendations() for quick betting overview")
        print("• All predictions include confidence scores and historical comparisons")
        
    except Exception as e:
        logger.error(f"Error running predictor: {e}")
        print(f"❌ Error: {e}")
        print("💡 Try running: predictor.train_all_models() first")


if __name__ == "__main__":
    main()
