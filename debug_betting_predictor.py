#!/usr/bin/env python3
"""
Debug NFL Betting Predictor
Shows exactly what's happening in the recommendation logic and relaxes thresholds.
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

class DebugBettingPredictor:
    """Debug version that shows what's happening in the recommendation logic."""
    
    def __init__(self):
        """Initialize the predictor."""
        self.db_url = "sqlite:///data/nfl_predictions.db"
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.models = {}
        self.model_dir = Path("models/trained")
        self.performance_dir = Path("models/performance")
        
        # Load trained models
        self._load_models()
    
    def _load_models(self):
        """Load the models that were just trained."""
        for model_file in self.model_dir.glob("*_model.pkl"):
            try:
                model_name = model_file.stem
                model = joblib.load(model_file)
                self.models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_file.name}: {e}")
    
    def extract_position_from_player_id(self, player_id: str) -> str:
        """Extract position from player_id suffix."""
        player_id = str(player_id).lower()
        if '_qb' in player_id:
            return 'QB'
        elif '_rb' in player_id:
            return 'RB'
        elif '_wr' in player_id:
            return 'WR'
        elif '_te' in player_id:
            return 'TE'
        else:
            return 'UNKNOWN'
    
    def predict_player_debug(self, player_id: str) -> Dict[str, Any]:
        """Make predictions with detailed debugging."""
        # Extract position from player_id
        position = self.extract_position_from_player_id(player_id)
        
        if position == 'UNKNOWN':
            print(f"   ❌ Could not determine position for {player_id}")
            return {}
        
        # Get recent stats
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
                print(f"   ❌ No recent stats for {player_id}")
                return {}
            
            # Calculate features
            def safe_mean(values):
                clean_values = [v for v in values if v is not None]
                return np.mean(clean_values) if clean_values else 0
            
            if position == 'QB':
                features = [
                    0.5,
                    safe_mean([getattr(row, 'passing_attempts', 0) for row in rows]),
                    safe_mean([getattr(row, 'passing_completions', 0) for row in rows]),
                    safe_mean([getattr(row, 'rushing_attempts', 0) for row in rows])
                ]
                target_cols = ['fantasy_points', 'passing_yards', 'passing_touchdowns']
            else:  # RB, WR, TE
                features = [
                    0.5,
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
        
        print(f"   🔮 Making predictions for {player_id} ({position})")
        print(f"   📊 Features: {[f'{f:.1f}' for f in features]}")
        
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
                    
                    print(f"   📈 {target}: {pred:.1f} (confidence: {confidence_scores[target]:.1%})")
                        
                except Exception as e:
                    print(f"   ❌ Prediction failed for {model_name}: {e}")
        
        return {
            'predictions': predictions,
            'confidence': confidence_scores,
            'position': position
        }
    
    def debug_betting_recommendations(self):
        """Debug the betting recommendation logic."""
        print("🔍 DEBUGGING BETTING RECOMMENDATIONS")
        print("=" * 50)
        
        # Get eligible players
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
                LIMIT 10
            """)
            
            result = session.execute(query)
            players = result.fetchall()
            
            print(f"Found {len(players)} eligible players")
            print()
            
            recommendations = []
            
            for i, player in enumerate(players, 1):
                player_id = player[0]
                avg_points = float(player[1])
                games_played = int(player[2])
                
                print(f"{i}. Analyzing {player_id}")
                print(f"   📊 Historical average: {avg_points:.1f} fantasy points")
                print(f"   🎮 Games played: {games_played}")
                
                # Get predictions
                prediction_data = self.predict_player_debug(player_id)
                
                if prediction_data and 'predictions' in prediction_data:
                    predictions = prediction_data['predictions']
                    confidence = prediction_data['confidence']
                    position = prediction_data.get('position', 'UNKNOWN')
                    
                    if 'fantasy_points' in predictions:
                        predicted_fp = predictions['fantasy_points']
                        confidence_fp = confidence.get('fantasy_points', 0.5)
                        bet_value = predicted_fp - avg_points
                        
                        print(f"   🎯 Predicted fantasy points: {predicted_fp:.1f}")
                        print(f"   📈 Bet value (predicted - avg): {bet_value:.1f}")
                        print(f"   🎯 Confidence: {confidence_fp:.1%}")
                        
                        # Relaxed betting logic for debugging
                        bet_rec = None
                        if bet_value > 1.5 and confidence_fp > 0.4:  # Relaxed thresholds
                            bet_rec = f"Over {predicted_fp - 1.5:.1f} fantasy points"
                            print(f"   ✅ RECOMMENDATION: {bet_rec}")
                        elif bet_value < -1.5 and confidence_fp > 0.4:
                            bet_rec = f"Under {predicted_fp + 1.5:.1f} fantasy points"
                            print(f"   ✅ RECOMMENDATION: {bet_rec}")
                        else:
                            print(f"   ❌ No recommendation:")
                            if abs(bet_value) <= 1.5:
                                print(f"      - Bet value too small: {bet_value:.1f}")
                            if confidence_fp <= 0.4:
                                print(f"      - Confidence too low: {confidence_fp:.1%}")
                        
                        if bet_rec and confidence_fp > 0.3:  # Very relaxed final threshold
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
                else:
                    print(f"   ❌ No predictions available")
                
                print()
            
            print(f"🎯 FINAL RESULTS: {len(recommendations)} recommendations generated")
            
            if recommendations:
                print("\n📋 RECOMMENDATIONS:")
                print("-" * 30)
                
                position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
                
                for i, rec in enumerate(recommendations, 1):
                    emoji = position_emoji.get(rec['position'], '⚡')
                    print(f"{i}. {rec['player_id']} ({rec['position']}) {emoji}")
                    print(f"   💰 {rec['recommendation']}")
                    print(f"   📊 Predicted: {rec['predicted_fantasy_points']:.1f} FP")
                    print(f"   🎯 Confidence: {rec['confidence']:.1%}")
                    print(f"   📈 Edge: {rec['bet_value']:.1f}")
                    print()
            else:
                print("\n❌ No recommendations met the criteria")
                print("💡 Try running with even more relaxed thresholds")

def main():
    """Main debug function."""
    predictor = DebugBettingPredictor()
    
    print(f"🏈 DEBUG NFL BETTING PREDICTOR")
    print(f"Loaded {len(predictor.models)} models")
    print()
    
    predictor.debug_betting_recommendations()

if __name__ == "__main__":
    main()