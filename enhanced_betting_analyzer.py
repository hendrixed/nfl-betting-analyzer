#!/usr/bin/env python3
"""
Enhanced NFL Betting Analyzer - Advanced Version
Provides comprehensive betting analysis with multiple prediction models,
confidence scoring, and detailed statistical insights.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import warnings
import json
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedBettingAnalyzer:
    """Enhanced NFL betting analyzer with advanced features."""
    
    def __init__(self):
        """Initialize the enhanced analyzer."""
        self.db_url = "sqlite:///data/nfl_predictions.db"
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("models/trained")
        self.performance_dir = Path("models/performance")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify database exists
        if not Path("data/nfl_predictions.db").exists():
            logger.error("Database not found. Please run setup_database.py first.")
            return
        
        # Load existing models
        self._load_models()
        self._load_scalers()
    
    def _load_models(self):
        """Load trained models from disk."""
        for model_file in self.model_dir.glob("*.pkl"):
            if "scaler" not in model_file.name:
                try:
                    model_name = model_file.stem
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_file}: {e}")
    
    def _load_scalers(self):
        """Load feature scalers from disk."""
        for scaler_file in self.model_dir.glob("*_scaler.pkl"):
            try:
                scaler_name = scaler_file.stem.replace("_scaler", "")
                scaler = joblib.load(scaler_file)
                self.scalers[scaler_name] = scaler
                logger.info(f"Loaded scaler: {scaler_name}")
            except Exception as e:
                logger.warning(f"Failed to load {scaler_file}: {e}")
    
    def get_enhanced_training_data(self, position: str = None) -> pd.DataFrame:
        """Get enhanced training data with additional features."""
        with self.Session() as session:
            query = """
            SELECT 
                pgs.player_id,
                CASE 
                    WHEN pgs.player_id LIKE '%_qb' THEN 'QB'
                    WHEN pgs.player_id LIKE '%_rb' THEN 'RB'
                    WHEN pgs.player_id LIKE '%_wr' THEN 'WR'
                    WHEN pgs.player_id LIKE '%_te' THEN 'TE'
                    ELSE 'UNKNOWN'
                END as position,
                COALESCE(g.week, 1) as week,
                COALESCE(g.season, 2024) as season,
                pgs.passing_attempts,
                pgs.passing_completions,
                pgs.passing_yards,
                pgs.passing_touchdowns,
                COALESCE(pgs.passing_interceptions, 0) as passing_interceptions,
                pgs.rushing_attempts,
                pgs.rushing_yards,
                pgs.rushing_touchdowns,
                pgs.targets,
                pgs.receptions,
                pgs.receiving_yards,
                pgs.receiving_touchdowns,
                pgs.fantasy_points_ppr as fantasy_points,
                pgs.is_home,
                pgs.created_at
            FROM player_game_stats pgs
            LEFT JOIN games g ON pgs.game_id = g.game_id
            WHERE pgs.fantasy_points_ppr > 0
            """
            
            if position:
                query += f" AND pgs.player_id LIKE '%_{position.lower()}'"
            
            query += " ORDER BY pgs.player_id, COALESCE(g.season, 2024), COALESCE(g.week, 1)"
            
            result = session.execute(text(query))
            data = result.fetchall()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = [
                'player_id', 'position', 'week', 'season', 'passing_attempts', 
                'passing_completions', 'passing_yards', 'passing_touchdowns', 
                'passing_interceptions', 'rushing_attempts', 'rushing_yards',
                'rushing_touchdowns', 'targets', 'receptions', 'receiving_yards',
                'receiving_touchdowns', 'fantasy_points', 'is_home', 'created_at'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            
            # Add enhanced features
            df = self._add_enhanced_features(df)
            
            return df
    
    def _add_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced features to the dataset."""
        # Sort by player and date
        df = df.sort_values(['player_id', 'season', 'week'])
        
        # Calculate rolling averages (last 3 games)
        for col in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards']:
            if col in df.columns:
                df[f'{col}_avg_3'] = df.groupby('player_id')[col].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
                df[f'{col}_avg_5'] = df.groupby('player_id')[col].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        
        # Calculate consistency metrics (standard deviation)
        for col in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards']:
            if col in df.columns:
                df[f'{col}_std_3'] = df.groupby('player_id')[col].rolling(3, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        
        # Calculate efficiency metrics
        df['completion_percentage'] = np.where(df['passing_attempts'] > 0, 
                                             df['passing_completions'] / df['passing_attempts'], 0)
        df['yards_per_attempt'] = np.where(df['passing_attempts'] > 0,
                                         df['passing_yards'] / df['passing_attempts'], 0)
        df['yards_per_carry'] = np.where(df['rushing_attempts'] > 0,
                                       df['rushing_yards'] / df['rushing_attempts'], 0)
        df['yards_per_target'] = np.where(df['targets'] > 0,
                                        df['receiving_yards'] / df['targets'], 0)
        df['catch_rate'] = np.where(df['targets'] > 0,
                                  df['receptions'] / df['targets'], 0)
        
        # Add season progress feature
        df['season_progress'] = df['week'] / 18.0
        
        # Add recent form (trend over last 3 games)
        df['recent_form'] = df.groupby('player_id')['fantasy_points'].rolling(3, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        ).reset_index(0, drop=True).fillna(0)
        
        return df
    
    def train_enhanced_models(self, position: str):
        """Train enhanced models for a specific position with multiple algorithms."""
        logger.info(f"Training enhanced models for {position}...")
        
        df = self.get_enhanced_training_data(position)
        if df.empty:
            logger.warning(f"No data found for {position}")
            return
        
        logger.info(f"Found {len(df)} samples for {position}")
        
        # Define features based on position
        base_features = ['is_home', 'season_progress', 'recent_form']
        
        if position == 'QB':
            feature_cols = base_features + [
                'passing_attempts', 'completion_percentage', 'yards_per_attempt',
                'passing_interceptions', 'rushing_attempts', 'fantasy_points_avg_3',
                'fantasy_points_std_3', 'passing_yards_avg_3'
            ]
            target_cols = ['fantasy_points', 'passing_yards', 'passing_touchdowns']
        elif position == 'RB':
            feature_cols = base_features + [
                'rushing_attempts', 'yards_per_carry', 'targets', 'catch_rate',
                'fantasy_points_avg_3', 'fantasy_points_std_3', 'rushing_yards_avg_3'
            ]
            target_cols = ['fantasy_points', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards']
        elif position == 'WR':
            feature_cols = base_features + [
                'targets', 'catch_rate', 'yards_per_target', 'rushing_attempts',
                'fantasy_points_avg_3', 'fantasy_points_std_3', 'receiving_yards_avg_3'
            ]
            target_cols = ['fantasy_points', 'receptions', 'receiving_yards', 'receiving_touchdowns']
        else:  # TE
            feature_cols = base_features + [
                'targets', 'catch_rate', 'yards_per_target',
                'fantasy_points_avg_3', 'fantasy_points_std_3', 'receiving_yards_avg_3'
            ]
            target_cols = ['fantasy_points', 'receptions', 'receiving_yards']
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].fillna(0)
        
        # Train models for each target
        for target in target_cols:
            if target not in df.columns:
                continue
                
            y = df[target].fillna(0)
            
            # Skip if no variance
            if y.std() == 0:
                logger.warning(f"No variance in {position} {target} data")
                continue
            
            # Skip if insufficient data
            if len(X) < 50:
                logger.warning(f"Insufficient data for {position} {target}: {len(X)} samples")
                continue
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train multiple models (using only compatible algorithms)
                models = {
                    'rf': RandomForestRegressor(
                        n_estimators=100, max_depth=10, min_samples_split=5,
                        min_samples_leaf=2, random_state=42, n_jobs=-1
                    ),
                    'lr': LinearRegression()
                }
                
                best_model = None
                best_score = -np.inf
                best_model_name = None
                
                for model_name, model in models.items():
                    # Train model
                    if model_name == 'lr':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Evaluate
                    score = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    logger.info(f"  {model_name.upper()}: R² = {score:.3f}, MAE = {mae:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_model_name = model_name
                
                # Save best model and scaler
                model_name = f"{position}_{target}_model"
                scaler_name = f"{position}_{target}_scaler"
                
                model_path = self.model_dir / f"{model_name}.pkl"
                scaler_path = self.model_dir / f"{scaler_name}.pkl"
                
                joblib.dump(best_model, model_path)
                joblib.dump(scaler, scaler_path)
                
                # Store in memory
                self.models[model_name] = best_model
                self.scalers[f"{position}_{target}"] = scaler
                
                # Save performance metrics
                performance = {
                    'model_type': best_model_name,
                    'r2_score': best_score,
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'features': available_features,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'timestamp': datetime.now().isoformat()
                }
                
                perf_path = self.performance_dir / f"{model_name}_performance.json"
                with open(perf_path, 'w') as f:
                    json.dump(performance, f, indent=2)
                
                logger.info(f"✅ {position} {target}: Best model = {best_model_name.upper()}, R² = {best_score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {position} {target}: {e}")
    
    def train_all_enhanced_models(self):
        """Train enhanced models for all positions."""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            self.train_enhanced_models(position)
        
        logger.info("✅ All enhanced model training completed!")
    
    def predict_player_enhanced(self, player_id: str) -> Dict[str, Any]:
        """Make enhanced predictions for a specific player."""
        # Determine position
        if '_qb' in player_id.lower():
            position = 'QB'
        elif '_rb' in player_id.lower():
            position = 'RB'
        elif '_wr' in player_id.lower():
            position = 'WR'
        elif '_te' in player_id.lower():
            position = 'TE'
        else:
            return {}
        
        # Get recent stats for features
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
            
            # Convert to DataFrame for feature engineering
            columns = [
                'player_id', 'week', 'season', 'passing_attempts', 'passing_completions',
                'passing_yards', 'passing_touchdowns', 'passing_interceptions',
                'rushing_attempts', 'rushing_yards', 'rushing_touchdowns',
                'targets', 'receptions', 'receiving_yards', 'receiving_touchdowns',
                'fantasy_points_ppr', 'is_home', 'created_at'
            ]
            
            df = pd.DataFrame([dict(zip(columns, [getattr(row, col, 0) for col in columns])) for row in rows])
            df['position'] = position
            df['fantasy_points'] = df['fantasy_points_ppr']
            
            # Add enhanced features
            df = self._add_enhanced_features(df)
            
            # Get latest features
            latest_features = df.iloc[0]
            
            # Define feature sets
            base_features = ['is_home', 'season_progress', 'recent_form']
            
            if position == 'QB':
                feature_names = base_features + [
                    'passing_attempts', 'completion_percentage', 'yards_per_attempt',
                    'passing_interceptions', 'rushing_attempts', 'fantasy_points_avg_3',
                    'fantasy_points_std_3', 'passing_yards_avg_3'
                ]
                target_cols = ['fantasy_points', 'passing_yards', 'passing_touchdowns']
            elif position == 'RB':
                feature_names = base_features + [
                    'rushing_attempts', 'yards_per_carry', 'targets', 'catch_rate',
                    'fantasy_points_avg_3', 'fantasy_points_std_3', 'rushing_yards_avg_3'
                ]
                target_cols = ['fantasy_points', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards']
            elif position == 'WR':
                feature_names = base_features + [
                    'targets', 'catch_rate', 'yards_per_target', 'rushing_attempts',
                    'fantasy_points_avg_3', 'fantasy_points_std_3', 'receiving_yards_avg_3'
                ]
                target_cols = ['fantasy_points', 'receptions', 'receiving_yards', 'receiving_touchdowns']
            else:  # TE
                feature_names = base_features + [
                    'targets', 'catch_rate', 'yards_per_target',
                    'fantasy_points_avg_3', 'fantasy_points_std_3', 'receiving_yards_avg_3'
                ]
                target_cols = ['fantasy_points', 'receptions', 'receiving_yards']
            
            # Extract features
            features = []
            for feature in feature_names:
                if feature in latest_features:
                    features.append(latest_features[feature])
                else:
                    features.append(0)
            
            features = np.array(features).reshape(1, -1)
        
        # Make predictions
        predictions = {}
        confidence_scores = {}
        
        for target in target_cols:
            model_name = f"{position}_{target}_model"
            scaler_name = f"{position}_{target}"
            
            if model_name in self.models:
                try:
                    model = self.models[model_name]
                    
                    # Scale features if linear regression
                    if scaler_name in self.scalers and hasattr(model, 'coef_'):
                        scaler = self.scalers[scaler_name]
                        features_scaled = scaler.transform(features)
                        pred = model.predict(features_scaled)[0]
                    else:
                        pred = model.predict(features)[0]
                    
                    predictions[target] = max(0, pred)
                    
                    # Calculate confidence based on model performance
                    perf_path = self.performance_dir / f"{model_name}_performance.json"
                    if perf_path.exists():
                        with open(perf_path, 'r') as f:
                            perf = json.load(f)
                        confidence_scores[target] = max(0.1, min(0.95, perf.get('r2_score', 0.5)))
                    else:
                        confidence_scores[target] = 0.5
                        
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
        
        return {
            'predictions': predictions,
            'confidence': confidence_scores,
            'player_stats': {
                'recent_avg_fp': latest_features.get('fantasy_points_avg_3', 0),
                'consistency': 1 / (1 + latest_features.get('fantasy_points_std_3', 1)),
                'recent_form': latest_features.get('recent_form', 0)
            }
        }
    
    def get_enhanced_betting_recommendations(self) -> List[Dict]:
        """Get enhanced betting recommendations with confidence scoring."""
        recommendations = []
        
        # Get top players by position
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            with self.Session() as session:
                query = text("""
                    SELECT player_id, AVG(fantasy_points_ppr) as avg_points,
                           COUNT(*) as games_played
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
                
                for player_id, avg_points, games_played in players:
                    prediction_data = self.predict_player_enhanced(player_id)
                    
                    if 'predictions' in prediction_data and 'fantasy_points' in prediction_data['predictions']:
                        predicted_fp = prediction_data['predictions']['fantasy_points']
                        confidence = prediction_data['confidence'].get('fantasy_points', 0.5)
                        player_stats = prediction_data['player_stats']
                        
                        # Enhanced confidence calculation
                        consistency_bonus = player_stats['consistency'] * 0.2
                        form_bonus = max(0, player_stats['recent_form']) * 0.1
                        games_bonus = min(0.1, games_played / 100)  # More games = more confidence
                        
                        enhanced_confidence = min(0.95, confidence + consistency_bonus + form_bonus + games_bonus)
                        
                        # Determine betting recommendation with enhanced logic
                        bet_rec = None
                        bet_value = None
                        
                        if position == 'QB' and predicted_fp > 17.5:
                            bet_rec = "Over 17.5 fantasy points"
                            bet_value = predicted_fp - 17.5
                        elif position == 'RB' and predicted_fp > 13.5:
                            bet_rec = "Over 13.5 fantasy points"
                            bet_value = predicted_fp - 13.5
                        elif position == 'WR' and predicted_fp > 10.5:
                            bet_rec = "Over 10.5 fantasy points"
                            bet_value = predicted_fp - 10.5
                        elif position == 'TE' and predicted_fp > 7.5:
                            bet_rec = "Over 7.5 fantasy points"
                            bet_value = predicted_fp - 7.5
                        
                        if bet_rec and enhanced_confidence > 0.6:  # Only recommend if confident
                            recommendations.append({
                                'player_id': player_id,
                                'position': position,
                                'predicted_fantasy_points': predicted_fp,
                                'confidence': enhanced_confidence,
                                'recommendation': bet_rec,
                                'bet_value': bet_value,
                                'historical_avg': avg_points,
                                'consistency': player_stats['consistency'],
                                'recent_form': player_stats['recent_form'],
                                'games_played': games_played,
                                'other_predictions': {k: v for k, v in prediction_data['predictions'].items() 
                                                   if k != 'fantasy_points'}
                            })
        
        # Sort by confidence * bet_value (expected value)
        recommendations.sort(key=lambda x: x['confidence'] * x['bet_value'], reverse=True)
        return recommendations[:15]
    
    def display_enhanced_recommendations(self):
        """Display enhanced betting recommendations with detailed analysis."""
        print("🏈 ENHANCED NFL BETTING ANALYZER")
        print("=" * 60)
        
        if not self.models:
            print("❌ No trained models found. Run train_all_enhanced_models() first.")
            return
        
        print(f"✅ Loaded {len(self.models)} trained models")
        print()
        
        recommendations = self.get_enhanced_betting_recommendations()
        
        if not recommendations:
            print("❌ No betting recommendations available.")
            return
        
        print("🎯 TOP ENHANCED BETTING RECOMMENDATIONS:")
        print("-" * 50)
        
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        
        for i, rec in enumerate(recommendations, 1):
            emoji = position_emoji.get(rec['position'], '⚡')
            
            print(f"{i}. {rec['player_id']} ({rec['position']}) {emoji}")
            print(f"   💰 RECOMMENDATION: {rec['recommendation']}")
            print(f"   📊 Predicted Fantasy Points: {rec['predicted_fantasy_points']:.1f}")
            print(f"   🎯 Confidence: {rec['confidence']:.1%}")
            print(f"   📈 Expected Value: {rec['bet_value']:.1f}")
            print(f"   📉 Historical Average: {rec['historical_avg']:.1f}")
            print(f"   🔄 Consistency Score: {rec['consistency']:.2f}")
            print(f"   📊 Recent Form: {rec['recent_form']:.2f}")
            print(f"   🎮 Games Played: {rec['games_played']}")
            
            # Show other predictions if available
            if rec['other_predictions']:
                print("   📋 Other Predictions:")
                for stat, value in rec['other_predictions'].items():
                    print(f"      {stat.replace('_', ' ').title()}: {value:.1f}")
            print()
        
        print("⚠️  ENHANCED DISCLAIMER:")
        print("   • Predictions based on advanced ML models and historical data")
        print("   • Confidence scores incorporate consistency and recent form")
        print("   • Expected value calculated as (prediction - line) * confidence")
        print("   • Always gamble responsibly and within your means")
        print("   • Past performance does not guarantee future results")

def main():
    """Main function for enhanced analyzer."""
    try:
        analyzer = EnhancedBettingAnalyzer()
        
        # Check if database exists
        if not Path("data/nfl_predictions.db").exists():
            print("❌ Database not found. Please run setup_database.py first.")
            return
        
        # Check if models exist, if not train them
        if not analyzer.models:
            print("🤖 No models found. Training enhanced models...")
            analyzer.train_all_enhanced_models()
            print()
        
        # Display enhanced recommendations
        analyzer.display_enhanced_recommendations()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        print("Please check the logs and try again.")

if __name__ == "__main__":
    main()
