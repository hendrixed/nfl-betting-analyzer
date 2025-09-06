#!/usr/bin/env python3
"""
Simplified NFL Betting Predictor - Working Version
Fixes all compatibility issues and provides reliable betting recommendations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBettingPredictor:
    """Simplified, reliable NFL betting predictor."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.db_url = "sqlite:///data/nfl_predictions.db"
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.models = {}
        self.model_dir = Path("models/trained")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify database exists
        if not Path("data/nfl_predictions.db").exists():
            logger.error("Database not found. Please run setup_database.py first.")
            return
        
        # Load existing models
        self._load_models()
    
    def _load_models(self):
        """Load trained models from disk."""
        for model_file in self.model_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                model = joblib.load(model_file)
                self.models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_file}: {e}")
    
    def get_training_data(self, position: str = None) -> pd.DataFrame:
        """Get training data from database."""
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
                passing_attempts,
                passing_completions,
                passing_yards,
                passing_touchdowns,
                rushing_attempts,
                rushing_yards,
                rushing_touchdowns,
                targets,
                receptions,
                receiving_yards,
                receiving_touchdowns,
                fantasy_points_ppr as fantasy_points,
                is_home
            FROM player_game_stats 
            WHERE fantasy_points_ppr > 0
            """
            
            if position:
                query += f" AND player_id LIKE '%_{position.lower()}'"
            
            result = session.execute(text(query))
            data = result.fetchall()
            
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = [
                'player_id', 'position', 'passing_attempts', 'passing_completions',
                'passing_yards', 'passing_touchdowns', 'rushing_attempts', 'rushing_yards',
                'rushing_touchdowns', 'targets', 'receptions', 'receiving_yards',
                'receiving_touchdowns', 'fantasy_points', 'is_home'
            ]
            
            df = pd.DataFrame(data, columns=columns)
            return df
    
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
        else:  # TE
            feature_cols = ['is_home', 'targets', 'receptions', 'rushing_attempts']
            target_cols = ['fantasy_points', 'receptions', 'receiving_yards']
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        
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
            if len(X) < 20:
                logger.warning(f"Insufficient data for {position} {target}: {len(X)} samples")
                continue
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train model with improved parameters
                model = RandomForestRegressor(
                    n_estimators=50,  # Reduced for faster training
                    max_depth=8,      # Prevent overfitting
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = r2_score(y_train, model.predict(X_train))
                test_score = r2_score(y_test, model.predict(X_test))
                
                # Save model
                model_name = f"{position}_{target}_model"
                model_path = self.model_dir / f"{model_name}.pkl"
                joblib.dump(model, model_path)
                
                # Store in memory
                self.models[model_name] = model
                
                logger.info(f"✅ {position} {target}: R² = {test_score:.3f} (saved)")
                
            except Exception as e:
                logger.error(f"Failed to train {position} {target}: {e}")
    
    def train_all_models(self):
        """Train models for all positions."""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            self.train_position_models(position)
        
        logger.info("✅ All model training completed!")
    
    def predict_player(self, player_id: str) -> Dict[str, float]:
        """Make predictions for a specific player."""
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
                LIMIT 5
            """)
            
            result = session.execute(query, {"player_id": player_id})
            rows = result.fetchall()
            
            if not rows:
                return {}
            
            # Calculate average features
            if position == 'QB':
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'passing_attempts', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'passing_completions', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                ]
                model_targets = ['fantasy_points', 'passing_yards', 'passing_touchdowns']
            elif position == 'RB':
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'targets', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'receptions', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                ]
                model_targets = ['fantasy_points', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards']
            elif position == 'WR':
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'targets', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'receptions', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                ]
                model_targets = ['fantasy_points', 'receptions', 'receiving_yards', 'receiving_touchdowns']
            else:  # TE
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'targets', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'receptions', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                ]
                model_targets = ['fantasy_points', 'receptions', 'receiving_yards']
        
        # Make predictions
        predictions = {}
        for target in model_targets:
            model_name = f"{position}_{target}_model"
            if model_name in self.models:
                try:
                    pred = self.models[model_name].predict([features])[0]
                    predictions[target] = max(0, pred)  # Ensure non-negative
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
        
        return predictions
    
    def get_betting_recommendations(self) -> List[Dict]:
        """Get betting recommendations for top players."""
        recommendations = []
        
        # Get top players by position
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            with self.Session() as session:
                query = text("""
                    SELECT player_id, AVG(fantasy_points_ppr) as avg_points
                    FROM player_game_stats 
                    WHERE player_id LIKE :pattern
                    AND fantasy_points_ppr > 0
                    GROUP BY player_id
                    ORDER BY avg_points DESC
                    LIMIT 3
                """)
                
                pattern = f"%_{position.lower()}"
                result = session.execute(query, {"pattern": pattern})
                players = result.fetchall()
                
                for player_id, avg_points in players:
                    predictions = self.predict_player(player_id)
                    
                    if 'fantasy_points' in predictions:
                        predicted_fp = predictions['fantasy_points']
                        confidence = 0.65  # Base confidence
                        
                        # Determine betting recommendation
                        bet_rec = None
                        if position == 'QB' and predicted_fp > 18:
                            bet_rec = "Over 17.5 fantasy points"
                        elif position == 'RB' and predicted_fp > 14:
                            bet_rec = "Over 13.5 fantasy points"
                        elif position == 'WR' and predicted_fp > 11:
                            bet_rec = "Over 10.5 fantasy points"
                        elif position == 'TE' and predicted_fp > 8:
                            bet_rec = "Over 7.5 fantasy points"
                        
                        if bet_rec:
                            recommendations.append({
                                'player_id': player_id,
                                'position': position,
                                'predicted_fantasy_points': predicted_fp,
                                'confidence': confidence,
                                'recommendation': bet_rec,
                                'other_predictions': {k: v for k, v in predictions.items() if k != 'fantasy_points'}
                            })
        
        # Sort by predicted value
        recommendations.sort(key=lambda x: x['predicted_fantasy_points'], reverse=True)
        return recommendations[:10]
    
    def display_recommendations(self):
        """Display betting recommendations."""
        print("🏈 NFL BETTING RECOMMENDATIONS")
        print("=" * 50)
        
        if not self.models:
            print("❌ No trained models found. Run train_all_models() first.")
            return
        
        print(f"✅ Loaded {len(self.models)} trained models")
        print()
        
        recommendations = self.get_betting_recommendations()
        
        if not recommendations:
            print("❌ No betting recommendations available.")
            return
        
        print("🎯 TOP BETTING RECOMMENDATIONS:")
        print("-" * 40)
        
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        
        for i, rec in enumerate(recommendations, 1):
            emoji = position_emoji.get(rec['position'], '⚡')
            
            print(f"{i}. {rec['player_id']} ({rec['position']}) {emoji}")
            print(f"   Predicted Fantasy Points: {rec['predicted_fantasy_points']:.1f}")
            print(f"   Confidence: {rec['confidence']:.1%}")
            print(f"   💡 RECOMMENDATION: {rec['recommendation']}")
            
            # Show other predictions if available
            if rec['other_predictions']:
                for stat, value in rec['other_predictions'].items():
                    print(f"   {stat.replace('_', ' ').title()}: {value:.1f}")
            print()
        
        print("⚠️  DISCLAIMER: These are predictions based on historical data.")
        print("   Always gamble responsibly and within your means.")

def main():
    """Main function."""
    try:
        predictor = SimpleBettingPredictor()
        
        # Check if database exists
        if not Path("data/nfl_predictions.db").exists():
            print("❌ Database not found. Please run setup_database.py first.")
            return
        
        # Check if models exist, if not train them
        if not predictor.models:
            print("🤖 No models found. Training new models...")
            predictor.train_all_models()
            print()
        
        # Display recommendations
        predictor.display_recommendations()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")
        print("Please check the logs and try again.")

if __name__ == "__main__":
    main()
