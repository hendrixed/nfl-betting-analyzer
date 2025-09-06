#!/usr/bin/env python3
"""
NFL Betting Predictor
Generate predictions for betting on player performance and game outcomes.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
import logging

from config_manager import get_config
from database_models import PlayerGameStats, Player, Game

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLBettingPredictor:
    """Generate betting predictions using trained models."""
    
    def __init__(self):
        """Initialize the betting predictor."""
        self.config = get_config()
        self.engine = create_engine(self.config.database.url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Load trained models
        self.models = self._load_models()
        
    def _load_models(self) -> Dict[str, any]:
        """Load all trained models."""
        models = {}
        model_dir = Path(self.config.models.model_directory)
        
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                model_name = model_file.stem
                try:
                    models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
        
        return models
    
    def predict_player_performance(self, player_id: str, week: int = None) -> Dict[str, float]:
        """Predict player performance for betting."""
        try:
            # Get player position from ID
            position = self._get_position_from_id(player_id)
            
            if not position:
                logger.warning(f"Could not determine position for {player_id}")
                return {}
            
            # Get recent player stats for features
            features = self._get_player_features(player_id, week)
            
            if not features:
                logger.warning(f"No features available for {player_id}")
                return {}
            
            predictions = {}
            
            # Generate predictions using available models
            for model_name, model in self.models.items():
                if position in model_name:
                    try:
                        # Prepare features for prediction
                        feature_array = np.array([list(features.values())]).reshape(1, -1)
                        prediction = model.predict(feature_array)[0]
                        
                        # Extract target name from model name
                        target = model_name.split(f"{position}_")[1].replace("_model", "")
                        predictions[target] = float(prediction)
                        
                    except Exception as e:
                        logger.warning(f"Prediction failed for {model_name}: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting for {player_id}: {e}")
            return {}
    
    def _get_position_from_id(self, player_id: str) -> Optional[str]:
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
    
    def _get_player_features(self, player_id: str, week: int = None) -> Dict[str, float]:
        """Get features for a player based on recent performance."""
        with self.Session() as session:
            # Get recent stats for the player
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
            
            # Calculate averages from recent games
            position = self._get_position_from_id(player_id)
            
            # Base features (consistent across all positions)
            features = {
                'is_home': 0.5,  # Default neutral
                'week': week or 10,  # Default current week
            }
            
            if position == 'QB':
                # QB-specific features (match training data exactly)
                avg_attempts = np.mean([getattr(row, 'passing_attempts', 0) or 0 for row in rows])
                avg_completions = np.mean([getattr(row, 'passing_completions', 0) or 0 for row in rows])
                avg_yards = np.mean([getattr(row, 'passing_yards', 0) or 0 for row in rows])
                avg_tds = np.mean([getattr(row, 'passing_touchdowns', 0) or 0 for row in rows])
                
                features.update({
                    'passing_attempts': avg_attempts,
                    'passing_completions': avg_completions,
                    'passing_yards': avg_yards,
                    'passing_touchdowns': avg_tds,
                })
                
            elif position in ['RB', 'WR', 'TE']:
                # Skill position features (match training data exactly)
                avg_rush_att = np.mean([getattr(row, 'rushing_attempts', 0) or 0 for row in rows])
                avg_rush_yds = np.mean([getattr(row, 'rushing_yards', 0) or 0 for row in rows])
                avg_targets = np.mean([getattr(row, 'targets', 0) or 0 for row in rows])
                avg_receptions = np.mean([getattr(row, 'receptions', 0) or 0 for row in rows])
                avg_rec_yds = np.mean([getattr(row, 'receiving_yards', 0) or 0 for row in rows])
                
                features.update({
                    'rushing_attempts': avg_rush_att,
                    'rushing_yards': avg_rush_yds,
                    'targets': avg_targets,
                    'receptions': avg_receptions,
                    'receiving_yards': avg_rec_yds,
                })
            
            return features
    
    def _generate_betting_recommendations(self, predictions: List[Dict]) -> List[Dict]:
        """Generate betting recommendations based on predictions."""
        recommendations = []
        
        for pred in predictions:
            confidence = pred['confidence']
            predicted_value = pred['predicted_value']
            stat_type = pred['stat_type']
            player_id = pred['player_id']
            position = self._get_position_from_id(player_id)
            
            # Position-specific betting logic based on confidence and predicted value
            if confidence > 0.6:  # 60% confidence threshold
                if stat_type == 'fantasy_points':
                    if position == 'QB':
                        if predicted_value > 20:
                            bet_type = "STRONG BET"
                            recommendation = f"Over 19.5 fantasy points"
                        elif predicted_value > 16:
                            bet_type = "BET"
                            recommendation = f"Over 15.5 fantasy points"
                        else:
                            continue
                    elif position == 'RB':
                        if predicted_value > 14.5:
                            bet_type = "STRONG BET"
                            recommendation = f"Over 14.5 fantasy points"
                        elif predicted_value > 9.5:
                            bet_type = "BET"
                            recommendation = f"Over 9.5 fantasy points"
                        else:
                            continue
                    elif position == 'WR':
                        if predicted_value > 12:
                            bet_type = "STRONG BET"
                            recommendation = f"Over 11.5 fantasy points"
                        elif predicted_value > 8:
                            bet_type = "BET"
                            recommendation = f"Over 7.5 fantasy points"
                        else:
                            continue
                    elif position == 'TE':
                        if predicted_value > 10:
                            bet_type = "STRONG BET"
                            recommendation = f"Over 9.5 fantasy points"
                        elif predicted_value > 6:
                            bet_type = "BET"
                            recommendation = f"Over 5.5 fantasy points"
                        else:
                            continue
                elif stat_type == 'passing_yards':
                    if predicted_value > 275:
                        bet_type = "STRONG BET"
                        recommendation = f"Over 274.5 passing yards"
                    elif predicted_value > 225:
                        bet_type = "BET"
                        recommendation = f"Over 224.5 passing yards"
                    else:
                        continue
                elif stat_type == 'passing_touchdowns':
                    if predicted_value > 2.2:
                        bet_type = "STRONG BET"
                        recommendation = f"Over 2.5 passing TDs"
                    elif predicted_value > 1.7:
                        bet_type = "BET"
                        recommendation = f"Over 1.5 passing TDs"
                    else:
                        continue
                else:
                    continue
                
                recommendations.append({
                    'player_id': player_id,
                    'position': position,
                    'confidence': confidence,
                    'predicted_value': predicted_value,
                    'bet_type': bet_type,
                    'recommendation': recommendation,
                    'stat_type': stat_type
                })
        
        # Sort by confidence descending, then by predicted value
        recommendations.sort(key=lambda x: (x['confidence'], x['predicted_value']), reverse=True)
        return recommendations[:15]  # Top 15 recommendations
    
    def get_betting_recommendations(self, week: int = None) -> List[Dict]:
        """Get betting recommendations for top players."""
        all_predictions = []
        
        # Get top players from all positions for predictions
        top_players = self._get_top_players(['QB', 'RB', 'WR', 'TE'], limit=15)
        
        for player_id, position in top_players:
            predictions = self.predict_player_performance(player_id, week)
            
            if predictions:
                # Convert predictions to betting format
                for stat_type, predicted_value in predictions.items():
                    confidence = self._calculate_confidence({stat_type: predicted_value}, position)
                    
                    prediction_data = {
                        'player_id': player_id,
                        'position': position,
                        'stat_type': stat_type,
                        'predicted_value': predicted_value,
                        'confidence': confidence
                    }
                    all_predictions.append(prediction_data)
        
        # Generate betting recommendations from predictions
        return self._generate_betting_recommendations(all_predictions)
    
    def _get_top_players(self, positions: List[str], limit: int = 10) -> List[Tuple[str, str]]:
        """Get top players by recent performance."""
        with self.Session() as session:
            players = []
            
            for position in positions:
                query = text("""
                    SELECT player_id, AVG(fantasy_points_ppr) as avg_points
                    FROM player_game_stats 
                    WHERE player_id LIKE :pattern
                    AND fantasy_points_ppr > 0
                    GROUP BY player_id
                    ORDER BY avg_points DESC
                    LIMIT :limit
                """)
                
                pattern = f"%_{position.lower()}"
                result = session.execute(query, {"pattern": pattern, "limit": limit//len(positions)})
                
                for row in result:
                    players.append((row.player_id, position))
            
            return players
    
    def _calculate_confidence(self, predictions: Dict[str, float], position: str) -> float:
        """Calculate confidence score for predictions."""
        # Base confidence on model accuracy and prediction values
        if position == 'QB' and 'passing_yards' in predictions:
            # QB passing yards model has 70% accuracy
            return 0.7 if predictions['passing_yards'] > 200 else 0.5
        elif position == 'RB' and 'fantasy_points' in predictions:
            # RB fantasy points model has 67% accuracy
            return 0.67 if predictions['fantasy_points'] > 10 else 0.4
        
        return 0.3  # Default low confidence
    
    def _generate_betting_advice(self, predictions: Dict[str, float], position: str) -> str:
        """Generate betting advice based on predictions."""
        advice = []
        
        if position == 'QB':
            if 'passing_yards' in predictions:
                yards = predictions['passing_yards']
                if yards > 300:
                    advice.append(f"STRONG BET: Over {int(yards-25)} passing yards")
                elif yards > 250:
                    advice.append(f"BET: Over {int(yards-15)} passing yards")
                else:
                    advice.append(f"AVOID: Under performing QB ({int(yards)} yards)")
            
            if 'passing_touchdowns' in predictions:
                tds = predictions['passing_touchdowns']
                if tds > 2.5:
                    advice.append(f"BET: Over 2.5 passing TDs")
                elif tds < 1.5:
                    advice.append(f"BET: Under 1.5 passing TDs")
        
        elif position == 'RB':
            if 'fantasy_points' in predictions:
                points = predictions['fantasy_points']
                if points > 15:
                    advice.append(f"STRONG BET: Over 14.5 fantasy points")
                elif points > 10:
                    advice.append(f"BET: Over 9.5 fantasy points")
                else:
                    advice.append(f"AVOID: Low fantasy production ({points:.1f} points)")
        
        return " | ".join(advice) if advice else "No clear betting opportunity"


def main():
    """Main function to generate betting predictions."""
    print("🏈 NFL BETTING PREDICTOR")
    print("=" * 50)
    
    predictor = NFLBettingPredictor()
    
    if not predictor.models:
        print("❌ No trained models found. Run 'python run_nfl_system.py train' first.")
        return
    
    print(f"✅ Loaded {len(predictor.models)} trained models")
    print(f"Models: {', '.join(predictor.models.keys())}")
    print()
    
    # Generate betting recommendations
    print("🎯 TOP BETTING RECOMMENDATIONS:")
    print("-" * 50)
        
    recommendations = predictor.get_betting_recommendations()
        
    for i, rec in enumerate(recommendations, 1):
        position_emoji = {'QB': '🎯', 'RB': '🏃', 'WR': '🙌', 'TE': '🎪'}
        emoji = position_emoji.get(rec.get('position', ''), '⚡')
            
        print(f"{i}. {rec['player_id']} ({rec.get('position', 'Unknown')}) {emoji}")
        print(f"   Confidence: {rec['confidence']:.1%}")
        print(f"   Predicted {rec['stat_type'].replace('_', ' ').title()}: {rec['predicted_value']:.1f}")
        print(f"   💡 {rec['bet_type']}: {rec['recommendation']}")
        print()
    
    print("⚠️  DISCLAIMER: These are predictions based on historical data.")
    print("   Always gamble responsibly and within your means.")


if __name__ == "__main__":
    main()
