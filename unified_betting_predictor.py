#!/usr/bin/env python3
"""
Fixed Unified NFL Betting Predictor - Database Schema Compatible
Correctly uses the actual database schema with proper table joins.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path
import warnings
import json
from datetime import datetime
import sklearn
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedUnifiedBettingPredictor:
    """Fixed NFL betting predictor with correct database schema."""
    
    def __init__(self):
        """Initialize the unified predictor."""
        self.db_url = "sqlite:///data/nfl_predictions.db"
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.models = {}
        self.model_dir = Path("models/trained")
        self.performance_dir = Path("models/performance")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
        # Check scikit-learn version
        self.sklearn_version = sklearn.__version__
        logger.info(f"Using scikit-learn version: {self.sklearn_version}")
        
        # Verify database exists
        if not Path("data/nfl_predictions.db").exists():
            logger.error("Database not found. Please run setup_database.py first.")
            return
        
        # Load existing models with error handling
        self._load_models_safely()
    
    def _load_models_safely(self):
        """Load models with compatibility checking."""
        loaded_count = 0
        failed_count = 0
        
        for model_file in self.model_dir.glob("*_model.pkl"):
            try:
                model_name = model_file.stem
                model = joblib.load(model_file)
                
                # Test prediction to ensure compatibility
                test_features = np.array([[0.5, 10, 5, 2]])  # Sample features
                _ = model.predict(test_features)
                
                self.models[model_name] = model
                logger.info(f"✅ Loaded model: {model_name}")
                loaded_count += 1
                
            except Exception as e:
                logger.warning(f"❌ Failed to load {model_file.name}: {e}")
                # Move incompatible model to backup
                backup_file = model_file.with_suffix('.pkl.backup')
                model_file.rename(backup_file)
                logger.info(f"Moved incompatible model to {backup_file.name}")
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Found {failed_count} incompatible models. They have been backed up.")
            logger.info("These models will be retrained automatically when needed.")
        
        logger.info(f"Successfully loaded {loaded_count} compatible models")
    
    def get_training_data(self, position: str = None) -> pd.DataFrame:
        """Get training data with correct schema using proper joins."""
        try:
            with self.Session() as session:
                if position:
                    # Join player_game_stats with players table to get position
                    query = text("""
                        SELECT 
                            pgs.player_id,
                            p.position,
                            pgs.is_home,
                            pgs.passing_attempts,
                            pgs.passing_completions,
                            pgs.passing_yards,
                            pgs.passing_touchdowns,
                            pgs.rushing_attempts,
                            pgs.rushing_yards,
                            pgs.rushing_touchdowns,
                            pgs.targets,
                            pgs.receptions,
                            pgs.receiving_yards,
                            pgs.receiving_touchdowns,
                            pgs.fantasy_points_standard as fantasy_points,
                            pgs.created_at
                        FROM player_game_stats pgs
                        JOIN players p ON pgs.player_id = p.player_id
                        WHERE p.position = :position
                        AND pgs.created_at IS NOT NULL
                        ORDER BY pgs.created_at DESC
                        LIMIT 10000
                    """)
                    result = session.execute(query, {"position": position})
                else:
                    query = text("""
                        SELECT 
                            pgs.player_id,
                            p.position,
                            pgs.is_home,
                            pgs.passing_attempts,
                            pgs.passing_completions,
                            pgs.passing_yards,
                            pgs.passing_touchdowns,
                            pgs.rushing_attempts,
                            pgs.rushing_yards,
                            pgs.rushing_touchdowns,
                            pgs.targets,
                            pgs.receptions,
                            pgs.receiving_yards,
                            pgs.receiving_touchdowns,
                            pgs.fantasy_points_standard as fantasy_points,
                            pgs.created_at
                        FROM player_game_stats pgs
                        JOIN players p ON pgs.player_id = p.player_id
                        WHERE pgs.created_at IS NOT NULL
                        ORDER BY pgs.created_at DESC
                        LIMIT 50000
                    """)
                    result = session.execute(query)
                
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                logger.info(f"Retrieved {len(df)} records from database")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return pd.DataFrame()
    
    def create_compatible_model(self) -> RandomForestRegressor:
        """Create a RandomForest model compatible with current scikit-learn version."""
        # Get current sklearn version for compatibility
        major, minor = sklearn.__version__.split('.')[:2]
        major, minor = int(major), int(minor)
        
        # Use parameters compatible with current version
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        return model
    
    def train_position_models(self, position: str):
        """Train models for a specific position with correct schema."""
        logger.info(f"Training models for {position}...")
        
        df = self.get_training_data(position)
        if df.empty:
            logger.warning(f"No data found for {position}")
            return
        
        logger.info(f"Found {len(df)} samples for {position}")
        
        # Define consistent features for all positions
        if position == 'QB':
            feature_cols = ['is_home', 'passing_attempts', 'passing_completions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'passing_yards', 'passing_touchdowns']
        elif position == 'RB':
            feature_cols = ['is_home', 'targets', 'receptions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards']
        elif position == 'WR':
            feature_cols = ['is_home', 'targets', 'receptions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'receptions', 'receiving_yards', 'receiving_touchdowns']
        else:  # TE
            feature_cols = ['is_home', 'targets', 'receptions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'receptions', 'receiving_yards']
        
        # Prepare features - check column availability
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < len(feature_cols):
            missing_cols = set(feature_cols) - set(available_cols)
            logger.warning(f"Missing columns for {position}: {missing_cols}")
            
        # Use available columns and fill missing ones with zeros
        X = pd.DataFrame()
        for col in feature_cols:
            if col in df.columns:
                X[col] = df[col].fillna(0)
            else:
                X[col] = 0
        
        # Train models for each target
        for target in target_cols:
            if target not in df.columns:
                logger.warning(f"Target column {target} not found for {position}")
                continue
                
            y = df[target].fillna(0)
            
            # Skip if no variance
            if y.std() == 0:
                logger.warning(f"No variance in {position} {target} data")
                continue
            
            # Skip if insufficient data
            if len(X) < 20:
                logger.warning(f"Insufficient data for {position} {target}: {len(X)} samples")
                continue
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Create compatible model
                model = self.create_compatible_model()
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = r2_score(y_train, model.predict(X_train))
                test_score = r2_score(y_test, model.predict(X_test))
                mae = mean_absolute_error(y_test, model.predict(X_test))
                
                # Save model
                model_name = f"{position}_{target}_model"
                model_path = self.model_dir / f"{model_name}.pkl"
                joblib.dump(model, model_path)
                
                # Store in memory
                self.models[model_name] = model
                
                # Save performance metrics
                performance = {
                    'model_type': 'RandomForest',
                    'sklearn_version': self.sklearn_version,
                    'r2_score': test_score,
                    'mae': mae,
                    'features': feature_cols,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'timestamp': datetime.now().isoformat()
                }
                
                perf_path = self.performance_dir / f"{model_name}_performance.json"
                with open(perf_path, 'w') as f:
                    json.dump(performance, f, indent=2)
                
                logger.info(f"✅ {position} {target}: R² = {test_score:.3f}, MAE = {mae:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to train {position} {target}: {e}")
    
    def train_all_models(self):
        """Train models for all positions."""
        logger.info("🤖 Training all models with current scikit-learn version...")
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            self.train_position_models(position)
        
        logger.info("✅ All model training completed!")
    
    def predict_player(self, player_id: str) -> Dict[str, Any]:
        """Make predictions for a specific player."""
        # Get player position from database
        with self.Session() as session:
            query = text("""
                SELECT p.position FROM players p 
                WHERE p.player_id = :player_id
            """)
            result = session.execute(query, {"player_id": player_id})
            row = result.fetchone()
            
            if not row:
                logger.warning(f"Player {player_id} not found in database")
                return {}
            
            position = row.position
        
        # Get recent stats for features
        with self.Session() as session:
            query = text("""
                SELECT pgs.* FROM player_game_stats pgs
                WHERE pgs.player_id = :player_id 
                ORDER BY pgs.created_at DESC 
                LIMIT 5
            """)
            
            result = session.execute(query, {"player_id": player_id})
            rows = result.fetchall()
            
            if not rows:
                logger.warning(f"No recent stats found for player {player_id}")
                return {}
            
            # Calculate average features (consistent with training)
            if position == 'QB':
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'passing_attempts', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'passing_completions', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                ]
                target_cols = ['fantasy_points', 'passing_yards', 'passing_touchdowns']
            else:  # RB, WR, TE
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'targets', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'receptions', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                ]
                if position == 'RB':
                    target_cols = ['fantasy_points', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards']
                elif position == 'WR':
                    target_cols = ['fantasy_points', 'receptions', 'receiving_yards', 'receiving_touchdowns']
                else:  # TE
                    target_cols = ['fantasy_points', 'receptions', 'receiving_yards']
        
        # Make predictions
        predictions = {}
        confidence_scores = {}
        
        for target in target_cols:
            model_name = f"{position}_{target}_model"
            if model_name in self.models:
                try:
                    pred = self.models[model_name].predict([features])[0]
                    predictions[target] = max(0, pred)
                    
                    # Get confidence from performance file
                    perf_path = self.performance_dir / f"{model_name}_performance.json"
                    if perf_path.exists():
                        with open(perf_path, 'r') as f:
                            perf = json.load(f)
                        confidence_scores[target] = max(0.3, min(0.9, perf.get('r2_score', 0.6)))
                    else:
                        confidence_scores[target] = 0.6
                        
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
        
        return {
            'predictions': predictions,
            'confidence': confidence_scores
        }
    
    def get_betting_recommendations(self) -> List[Dict]:
        """Get betting recommendations for top players using correct schema."""
        recommendations = []
        
        # Get recent players from database with proper join
        with self.Session() as session:
            query = text("""
                SELECT DISTINCT pgs.player_id, p.position, 
                       AVG(pgs.fantasy_points_standard) as avg_points, 
                       COUNT(*) as games_played
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                WHERE pgs.created_at > datetime('now', '-30 days')
                GROUP BY pgs.player_id, p.position
                HAVING games_played >= 3 AND avg_points > 5
                ORDER BY avg_points DESC
                LIMIT 50
            """)
            
            result = session.execute(query)
            players = result.fetchall()
            
            logger.info(f"Found {len(players)} eligible players for recommendations")
            
            for player in players:
                player_id = player.player_id
                position = player.position
                avg_points = float(player.avg_points)
                games_played = int(player.games_played)
                
                # Get predictions
                prediction_data = self.predict_player(player_id)
                
                if prediction_data and 'predictions' in prediction_data:
                    predictions = prediction_data['predictions']
                    confidence = prediction_data['confidence']
                    
                    if 'fantasy_points' in predictions:
                        predicted_fp = predictions['fantasy_points']
                        confidence_fp = confidence.get('fantasy_points', 0.5)
                        
                        # Simple betting logic
                        bet_value = predicted_fp - avg_points
                        
                        if bet_value > 2 and confidence_fp > 0.6:
                            bet_rec = f"Over {predicted_fp - 2:.1f} fantasy points"
                        elif bet_value < -2 and confidence_fp > 0.6:
                            bet_rec = f"Under {predicted_fp + 2:.1f} fantasy points"
                        else:
                            bet_rec = None
                        
                        if bet_rec and confidence_fp > 0.5:
                            recommendations.append({
                                'player_id': player_id,
                                'position': position,
                                'predicted_fantasy_points': predicted_fp,
                                'confidence': confidence_fp,
                                'recommendation': bet_rec,
                                'bet_value': bet_value,
                                'historical_avg': avg_points,
                                'games_played': games_played,
                                'other_predictions': {k: v for k, v in predictions.items() 
                                                   if k != 'fantasy_points'}
                            })
        
        # Sort by confidence * bet_value (expected value)
        recommendations.sort(key=lambda x: x['confidence'] * abs(x['bet_value']), reverse=True)
        return recommendations[:12]
    
    def display_recommendations(self):
        """Display betting recommendations."""
        print("🏈 FIXED NFL BETTING PREDICTOR")
        print("=" * 55)
        print(f"🔧 Scikit-learn version: {self.sklearn_version}")
        
        if not self.models:
            print("❌ No trained models found. Training new compatible models...")
            self.train_all_models()
            print()
        
        print(f"✅ Loaded {len(self.models)} compatible models")
        print()
        
        recommendations = self.get_betting_recommendations()
        
        if not recommendations:
            print("❌ No betting recommendations available.")
            print("💡 This might be due to:")
            print("   • Insufficient recent player data in database")
            print("   • Models need retraining")
            print("   • Database connection issues")
            print("   • No players meeting criteria (3+ games, 5+ fantasy points)")
            return
        
        print("🎯 TOP BETTING RECOMMENDATIONS:")
        print("-" * 45)
        
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        
        for i, rec in enumerate(recommendations, 1):
            emoji = position_emoji.get(rec['position'], '⚡')
            
            print(f"{i}. {rec['player_id']} ({rec['position']}) {emoji}")
            print(f"   💰 RECOMMENDATION: {rec['recommendation']}")
            print(f"   📊 Predicted Fantasy Points: {rec['predicted_fantasy_points']:.1f}")
            print(f"   🎯 Confidence: {rec['confidence']:.1%}")
            print(f"   📈 Expected Value: {rec['bet_value']:.1f}")
            print(f"   📉 Historical Average: {rec['historical_avg']:.1f}")
            print(f"   🎮 Games Played: {rec['games_played']}")
            
            # Show other predictions if available
            if rec['other_predictions']:
                print("   📋 Other Predictions:")
                for stat, value in rec['other_predictions'].items():
                    print(f"      {stat.replace('_', ' ').title()}: {value:.1f}")
            print()
        
        print("⚠️  DISCLAIMER:")
        print("   • Models retrained for current scikit-learn version")
        print("   • Uses correct database schema with proper table joins")
        print("   • Predictions based on historical player performance")
        print("   • Always gamble responsibly and within your means")

def main():
    """Main function with enhanced error handling."""
    try:
        predictor = FixedUnifiedBettingPredictor()
        
        # Check if database exists
        if not Path("data/nfl_predictions.db").exists():
            print("❌ Database not found. Please run setup_database.py first.")
            return
        
        # Display recommendations
        predictor.display_recommendations()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        print("Please check the logs and try again.")
        
        # Provide troubleshooting info
        print("\n🔧 TROUBLESHOOTING:")
        print(f"   • Current scikit-learn version: {sklearn.__version__}")
        print("   • Database schema: Uses proper joins between player_game_stats and players tables")
        print("   • Check if your database has data: SELECT COUNT(*) FROM player_game_stats;")

if __name__ == "__main__":
    main()