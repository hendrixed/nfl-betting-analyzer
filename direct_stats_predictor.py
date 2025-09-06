#!/usr/bin/env python3
"""
Direct Stats NFL Betting Predictor
Works directly with player_game_stats table, bypassing the players table mismatch.
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
import re
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DirectStatsPredictor:
    """NFL betting predictor that works directly with game stats, no players table needed."""
    
    def __init__(self):
        """Initialize the predictor."""
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
        
        # Load existing models
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
                backup_file = model_file.with_suffix('.pkl.backup')
                if not backup_file.exists():
                    model_file.rename(backup_file)
                    logger.info(f"Moved incompatible model to {backup_file.name}")
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Found {failed_count} incompatible models. They have been backed up.")
        
        logger.info(f"Successfully loaded {loaded_count} compatible models")
    
    def extract_position_from_stats(self, row) -> str:
        """Extract position from player statistics patterns."""
        # Get stat values, treating None as 0
        passing_attempts = row.get('passing_attempts', 0) or 0
        passing_yards = row.get('passing_yards', 0) or 0
        rushing_attempts = row.get('rushing_attempts', 0) or 0
        targets = row.get('targets', 0) or 0
        receptions = row.get('receptions', 0) or 0
        
        # Position logic based on stat patterns
        if passing_attempts > 0 or passing_yards > 0:
            return 'QB'
        elif rushing_attempts >= 3 and (targets == 0 or targets <= 2):
            return 'RB'
        elif targets > 0 and receptions > 0:
            if targets >= 6:  # High target volume suggests WR
                return 'WR'
            else:  # Lower targets suggests TE
                return 'TE'
        elif rushing_attempts > 0:
            return 'RB'
        else:
            # Fallback: try to extract from player_id if it has position suffix
            player_id = str(row.get('player_id', ''))
            if '_qb' in player_id.lower():
                return 'QB'
            elif '_rb' in player_id.lower():
                return 'RB'
            elif '_wr' in player_id.lower():
                return 'WR'
            elif '_te' in player_id.lower():
                return 'TE'
            else:
                return 'UNKNOWN'
    
    def test_database_connection(self):
        """Test database connection and analyze data."""
        try:
            with self.Session() as session:
                # Check basic stats
                result = session.execute(text("SELECT COUNT(*) FROM player_game_stats"))
                stats_count = result.scalar()
                logger.info(f"📈 Total game stats: {stats_count}")
                
                if stats_count == 0:
                    return False
                
                # Check recent data
                result = session.execute(text("""
                    SELECT COUNT(*) FROM player_game_stats 
                    WHERE created_at > datetime('now', '-30 days')
                """))
                recent_count = result.scalar()
                logger.info(f"📅 Recent stats (30 days): {recent_count}")
                
                # Sample some data to understand position distribution
                result = session.execute(text("""
                    SELECT player_id, passing_attempts, rushing_attempts, targets, receptions,
                           fantasy_points_standard, created_at
                    FROM player_game_stats 
                    WHERE fantasy_points_standard > 0
                    LIMIT 10
                """))
                
                samples = result.fetchall()
                logger.info("📊 Sample data with positions:")
                
                position_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'UNKNOWN': 0}
                
                for sample in samples:
                    row_dict = {
                        'player_id': sample[0],
                        'passing_attempts': sample[1],
                        'rushing_attempts': sample[2], 
                        'targets': sample[3],
                        'receptions': sample[4]
                    }
                    position = self.extract_position_from_stats(row_dict)
                    position_counts[position] += 1
                    logger.info(f"  - {sample[0]}: {position} (FP: {sample[5]})")
                
                logger.info(f"🎯 Position distribution in sample: {position_counts}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def get_training_data(self, position: str = None) -> pd.DataFrame:
        """Get training data directly from player_game_stats."""
        try:
            with self.Session() as session:
                # Get all game stats
                query = text("""
                    SELECT 
                        player_id,
                        CAST(is_home AS INTEGER) as is_home,
                        COALESCE(passing_attempts, 0) as passing_attempts,
                        COALESCE(passing_completions, 0) as passing_completions,
                        COALESCE(passing_yards, 0) as passing_yards,
                        COALESCE(passing_touchdowns, 0) as passing_touchdowns,
                        COALESCE(rushing_attempts, 0) as rushing_attempts,
                        COALESCE(rushing_yards, 0) as rushing_yards,
                        COALESCE(rushing_touchdowns, 0) as rushing_touchdowns,
                        COALESCE(targets, 0) as targets,
                        COALESCE(receptions, 0) as receptions,
                        COALESCE(receiving_yards, 0) as receiving_yards,
                        COALESCE(receiving_touchdowns, 0) as receiving_touchdowns,
                        COALESCE(fantasy_points_standard, 0) as fantasy_points,
                        created_at
                    FROM player_game_stats
                    WHERE created_at IS NOT NULL
                    AND fantasy_points_standard > 0
                    ORDER BY created_at DESC
                    LIMIT 50000
                """)
                
                result = session.execute(query)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                if df.empty:
                    logger.warning("No data found in player_game_stats")
                    return df
                
                # Extract positions from stats
                logger.info("🔍 Extracting positions from statistics...")
                df['position'] = df.apply(self.extract_position_from_stats, axis=1)
                
                # Filter by position if specified
                if position:
                    df = df[df['position'] == position]
                
                # Log position distribution
                position_counts = df['position'].value_counts()
                logger.info(f"📊 Position distribution: {position_counts.to_dict()}")
                
                logger.info(f"Retrieved {len(df)} records for position {position or 'ALL'}")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return pd.DataFrame()
    
    def create_compatible_model(self) -> RandomForestRegressor:
        """Create a RandomForest model compatible with current scikit-learn version."""
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
        """Train models for a specific position."""
        logger.info(f"Training models for {position}...")
        
        df = self.get_training_data(position)
        if df.empty:
            logger.warning(f"No data found for {position}")
            return
        
        logger.info(f"Found {len(df)} samples for {position}")
        
        # Define features and targets based on position
        if position == 'QB':
            feature_cols = ['is_home', 'passing_attempts', 'passing_completions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'passing_yards', 'passing_touchdowns']
        elif position == 'RB':
            feature_cols = ['is_home', 'targets', 'receptions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards']
        elif position == 'WR':
            feature_cols = ['is_home', 'targets', 'receptions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'receptions', 'receiving_yards', 'receiving_touchdowns']
        elif position == 'TE':
            feature_cols = ['is_home', 'targets', 'receptions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'receptions', 'receiving_yards']
        else:
            logger.warning(f"Unknown position: {position}")
            return
        
        # Prepare features
        X = pd.DataFrame()
        for col in feature_cols:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                X[col] = 0
                logger.warning(f"Column {col} not found, using zeros")
        
        # Train models for each target
        trained_count = 0
        for target in target_cols:
            if target not in df.columns:
                logger.warning(f"Target column {target} not found for {position}")
                continue
                
            y = pd.to_numeric(df[target], errors='coerce').fillna(0)
            
            # Skip if no variance or insufficient data
            if y.std() == 0:
                logger.warning(f"No variance in {position} {target} data")
                continue
            
            if len(X) < 20:
                logger.warning(f"Insufficient data for {position} {target}: {len(X)} samples")
                continue
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Create and train model
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
                trained_count += 1
                
            except Exception as e:
                logger.error(f"Failed to train {position} {target}: {e}")
        
        logger.info(f"Successfully trained {trained_count} models for {position}")
    
    def train_all_models(self):
        """Train models for all positions."""
        logger.info("🤖 Training all models directly from game stats...")
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            self.train_position_models(position)
            
        logger.info(f"✅ Model training completed! Total models: {len(self.models)}")
    
    def predict_player(self, player_id: str) -> Dict[str, Any]:
        """Make predictions for a specific player."""
        # Get recent stats for this player
        with self.Session() as session:
            query = text("""
                SELECT * FROM player_game_stats 
                WHERE player_id = :player_id 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            
            result = session.execute(query, {"player_id": player_id})
            rows = result.fetchall()
            
            if not rows:
                logger.warning(f"No recent stats found for player {player_id}")
                return {}
            
            # Extract position from most recent game
            row_dict = {
                'player_id': rows[0].player_id,
                'passing_attempts': rows[0].passing_attempts,
                'rushing_attempts': rows[0].rushing_attempts,
                'targets': rows[0].targets,
                'receptions': rows[0].receptions
            }
            position = self.extract_position_from_stats(row_dict)
            
            if position == 'UNKNOWN':
                logger.warning(f"Could not determine position for player {player_id}")
                return {}
            
            # Calculate average features
            def safe_mean(values):
                clean_values = [v for v in values if v is not None]
                return np.mean(clean_values) if clean_values else 0
            
            if position == 'QB':
                features = [
                    0.5,  # is_home (neutral)
                    safe_mean([getattr(row, 'passing_attempts', 0) for row in rows]),
                    safe_mean([getattr(row, 'passing_completions', 0) for row in rows]),
                    safe_mean([getattr(row, 'rushing_attempts', 0) for row in rows])
                ]
                target_cols = ['fantasy_points', 'passing_yards', 'passing_touchdowns']
            else:  # RB, WR, TE
                features = [
                    0.5,  # is_home (neutral)
                    safe_mean([getattr(row, 'targets', 0) for row in rows]),
                    safe_mean([getattr(row, 'receptions', 0) for row in rows]),
                    safe_mean([getattr(row, 'rushing_attempts', 0) for row in rows])
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
            'confidence': confidence_scores,
            'position': position
        }
    
    def get_betting_recommendations(self) -> List[Dict]:
        """Get betting recommendations from game stats."""
        recommendations = []
        
        # Get recent high-performing players
        with self.Session() as session:
            query = text("""
                SELECT player_id,
                       AVG(COALESCE(fantasy_points_standard, 0)) as avg_points, 
                       COUNT(*) as games_played
                FROM player_game_stats
                WHERE created_at > datetime('now', '-30 days')
                AND fantasy_points_standard > 0
                GROUP BY player_id
                HAVING games_played >= 3 AND avg_points > 5
                ORDER BY avg_points DESC
                LIMIT 50
            """)
            
            result = session.execute(query)
            players = result.fetchall()
            
            logger.info(f"Found {len(players)} eligible players for recommendations")
            
            for player in players:
                player_id = player[0]
                avg_points = float(player[1])
                games_played = int(player[2])
                
                # Get predictions
                prediction_data = self.predict_player(player_id)
                
                if prediction_data and 'predictions' in prediction_data:
                    predictions = prediction_data['predictions']
                    confidence = prediction_data['confidence']
                    position = prediction_data.get('position', 'UNKNOWN')
                    
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
        
        # Sort by confidence * bet_value
        recommendations.sort(key=lambda x: x['confidence'] * abs(x['bet_value']), reverse=True)
        return recommendations[:12]
    
    def display_recommendations(self):
        """Display betting recommendations."""
        print("🏈 DIRECT STATS NFL BETTING PREDICTOR")
        print("=" * 55)
        print(f"🔧 Scikit-learn version: {self.sklearn_version}")
        print(f"💾 Database: {self.db_url}")
        print("🎯 Working directly with game statistics (no players table needed)")
        
        # Test database connection
        if not self.test_database_connection():
            print("❌ Database connection failed or no data found")
            return
        
        if not self.models:
            print("❌ No trained models found. Training new models...")
            self.train_all_models()
            print()
        
        print(f"✅ Loaded {len(self.models)} compatible models")
        print()
        
        recommendations = self.get_betting_recommendations()
        
        if not recommendations:
            print("❌ No betting recommendations available.")
            print("💡 This might be due to:")
            print("   • Insufficient recent player data")
            print("   • Unable to determine player positions from stats")
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
        print("   • Positions determined from statistical patterns")
        print("   • Uses your real NFL data (28,019 game statistics!)")
        print("   • Bypasses players table mismatch issue")
        print("   • Always gamble responsibly and within your means")

def main():
    """Main function."""
    try:
        predictor = DirectStatsPredictor()
        predictor.display_recommendations()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        print("Please check the logs and try again.")

if __name__ == "__main__":
    main()