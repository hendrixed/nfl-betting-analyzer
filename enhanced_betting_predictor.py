#!/usr/bin/env python3
"""
Enhanced NFL Betting Predictor - Complete Working Version
Includes QB TDs, WR receiving yards, RB rushing yards, and more betting markets.
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

class EnhancedBettingPredictor:
    """Enhanced NFL betting predictor with expanded stat predictions."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.db_url = "sqlite:///data/nfl_predictions.db"
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.models = {}
        self.model_dir = Path("models/trained")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
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
                CAST(is_home as INTEGER) as is_home
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
        """Train comprehensive models for a specific position."""
        logger.info(f"Training models for {position}...")
        
        df = self.get_training_data(position)
        if df.empty:
            logger.warning(f"No data found for {position}")
            return
        
        logger.info(f"Found {len(df)} samples for {position}")
        
        # Define features and targets based on position
        if position == 'QB':
            feature_cols = ['is_home', 'passing_attempts', 'passing_completions', 'rushing_attempts']
            target_cols = [
                'fantasy_points', 'passing_yards', 'passing_touchdowns',
                'rushing_yards', 'rushing_touchdowns'
            ]
        elif position == 'RB':
            feature_cols = ['is_home', 'rushing_attempts', 'targets', 'receptions']
            target_cols = [
                'fantasy_points', 'rushing_yards', 'rushing_touchdowns',
                'receiving_yards', 'receiving_touchdowns'
            ]
        elif position in ['WR', 'TE']:
            feature_cols = ['is_home', 'targets', 'receptions', 'rushing_attempts']
            target_cols = [
                'fantasy_points', 'receiving_yards', 'receiving_touchdowns',
                'receptions'  # Receptions as a betting market
            ]
        else:
            return
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        
        # Train models for each target
        for target in target_cols:
            if target not in df.columns:
                continue
                
            y = df[target].fillna(0)
            
            # Skip if no variance or insufficient data
            if y.std() == 0 or len(X) < 20:
                logger.warning(f"Skipping {position} {target}: insufficient variance or data")
                continue
            
            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train model with current scikit-learn version
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = r2_score(y_train, model.predict(X_train))
                test_score = r2_score(y_test, model.predict(X_test))
                
                # Only save models with reasonable performance
                if test_score > 0.1:  # At least 10% better than random
                    # Save model
                    model_name = f"{position}_{target}_model"
                    model_path = self.model_dir / f"{model_name}.pkl"
                    joblib.dump(model, model_path)
                    
                    # Store in memory
                    self.models[model_name] = model
                    
                    logger.info(f"✅ {position} {target}: R² = {test_score:.3f} (saved)")
                else:
                    logger.warning(f"⚠️  {position} {target}: Poor performance R² = {test_score:.3f} (not saved)")
                
            except Exception as e:
                logger.error(f"Failed to train {position} {target}: {e}")
    
    def train_all_models(self):
        """Train models for all positions with expanded targets."""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            self.train_position_models(position)
        
        logger.info("✅ All model training completed!")
    
    def predict_player(self, player_id: str) -> Dict[str, float]:
        """Make comprehensive predictions for a specific player."""
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
            
            # Calculate average features based on position
            if position == 'QB':
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'passing_attempts', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'passing_completions', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                ]
                model_targets = ['fantasy_points', 'passing_yards', 'passing_touchdowns', 'rushing_yards', 'rushing_touchdowns']
            elif position == 'RB':
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'targets', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'receptions', 0) or 0 for row in rows])
                ]
                model_targets = ['fantasy_points', 'rushing_yards', 'rushing_touchdowns', 'receiving_yards', 'receiving_touchdowns']
            else:  # WR, TE
                features = [
                    0.5,  # is_home (neutral)
                    np.mean([getattr(row, 'targets', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'receptions', 0) or 0 for row in rows]),
                    np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                ]
                model_targets = ['fantasy_points', 'receiving_yards', 'receiving_touchdowns', 'receptions']
        
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
        """Get comprehensive betting recommendations for all stat types."""
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
                    LIMIT 5
                """)
                
                pattern = f"%_{position.lower()}"
                result = session.execute(query, {"pattern": pattern})
                players = result.fetchall()
                
                for player_id, avg_points in players:
                    predictions = self.predict_player(player_id)
                    
                    # Generate recommendations for each prediction
                    for stat_type, predicted_value in predictions.items():
                        confidence = 0.70  # Base confidence for new models
                        
                        # Determine betting recommendation based on stat type and position
                        bet_rec = self._get_betting_recommendation(position, stat_type, predicted_value)
                        
                        if bet_rec:
                            recommendations.append({
                                'player_id': player_id,
                                'position': position,
                                'stat_type': stat_type,
                                'predicted_value': predicted_value,
                                'confidence': confidence,
                                'recommendation': bet_rec,
                                'market_type': self._get_market_type(stat_type)
                            })
        
        # Sort by predicted value within each stat type
        recommendations.sort(key=lambda x: (x['stat_type'], -x['predicted_value']))
        return recommendations[:20]  # Top 20 recommendations
    
    def _get_betting_recommendation(self, position: str, stat_type: str, predicted_value: float) -> Optional[str]:
        """Generate specific betting recommendation based on position, stat, and predicted value."""
        
        # Fantasy Points
        if stat_type == 'fantasy_points':
            if position == 'QB' and predicted_value > 18:
                return f"Over 17.5 fantasy points"
            elif position == 'RB' and predicted_value > 14:
                return f"Over 13.5 fantasy points"
            elif position == 'WR' and predicted_value > 11:
                return f"Over 10.5 fantasy points"
            elif position == 'TE' and predicted_value > 8:
                return f"Over 7.5 fantasy points"
        
        # Passing Stats (QB)
        elif stat_type == 'passing_yards' and position == 'QB':
            if predicted_value > 275:
                return f"Over 274.5 passing yards"
            elif predicted_value > 225:
                return f"Over 224.5 passing yards"
        
        elif stat_type == 'passing_touchdowns' and position == 'QB':
            if predicted_value > 2.2:
                return f"Over 2.5 passing TDs"
            elif predicted_value > 1.7:
                return f"Over 1.5 passing TDs"
        
        # Rushing Stats
        elif stat_type == 'rushing_yards':
            if position == 'RB' and predicted_value > 80:
                return f"Over 79.5 rushing yards"
            elif position == 'QB' and predicted_value > 35:
                return f"Over 34.5 rushing yards"
        
        elif stat_type == 'rushing_touchdowns':
            if predicted_value > 0.8:
                return f"Over 0.5 rushing TDs"
        
        # Receiving Stats
        elif stat_type == 'receiving_yards':
            if position == 'WR' and predicted_value > 65:
                return f"Over 64.5 receiving yards"
            elif position == 'TE' and predicted_value > 45:
                return f"Over 44.5 receiving yards"
            elif position == 'RB' and predicted_value > 25:
                return f"Over 24.5 receiving yards"
        
        elif stat_type == 'receiving_touchdowns':
            if predicted_value > 0.6:
                return f"Over 0.5 receiving TDs"
        
        elif stat_type == 'receptions':
            if position == 'WR' and predicted_value > 5.5:
                return f"Over 5.5 receptions"
            elif position == 'TE' and predicted_value > 4.5:
                return f"Over 4.5 receptions"
            elif position == 'RB' and predicted_value > 3.5:
                return f"Over 3.5 receptions"
        
        return None
    
    def _get_market_type(self, stat_type: str) -> str:
        """Get the market category for display purposes."""
        if stat_type == 'fantasy_points':
            return 'Fantasy'
        elif 'passing' in stat_type:
            return 'Passing'
        elif 'rushing' in stat_type:
            return 'Rushing'
        elif 'receiving' in stat_type or stat_type == 'receptions':
            return 'Receiving'
        else:
            return 'Other'
    
    def display_recommendations(self):
        """Display comprehensive betting recommendations."""
        print("🏈 ENHANCED NFL BETTING RECOMMENDATIONS")
        print("=" * 60)
        
        if not self.models:
            print("❌ No trained models found. Run train_all_models() first.")
            return
        
        print(f"✅ Loaded {len(self.models)} trained models")
        print()
        
        recommendations = self.get_betting_recommendations()
        
        if not recommendations:
            print("❌ No betting recommendations available.")
            return
        
        # Group recommendations by market type
        market_groups = {}
        for rec in recommendations:
            market = rec['market_type']
            if market not in market_groups:
                market_groups[market] = []
            market_groups[market].append(rec)
        
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        
        for market_type, recs in market_groups.items():
            print(f"📊 {market_type.upper()} BETS:")
            print("-" * 40)
            
            for i, rec in enumerate(recs[:5], 1):  # Top 5 per category
                emoji = position_emoji.get(rec['position'], '⚡')
                
                print(f"{i}. {rec['player_id']} ({rec['position']}) {emoji}")
                print(f"   {rec['stat_type'].replace('_', ' ').title()}: {rec['predicted_value']:.1f}")
                print(f"   Confidence: {rec['confidence']:.1%}")
                print(f"   💡 BET: {rec['recommendation']}")
                print()
        
        print("⚠️  DISCLAIMER: These are predictions based on historical data.")
        print("   Always gamble responsibly and within your means.")

def main():
    """Main function."""
    predictor = EnhancedBettingPredictor()
    
    # Check if models exist, if not train them
    if not predictor.models:
        print("🤖 No models found. Training comprehensive models...")
        predictor.train_all_models()
        print()
    
    # Display recommendations
    predictor.display_recommendations()

if __name__ == "__main__":
    main()
