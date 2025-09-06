#!/usr/bin/env python3
"""
Comprehensive Enhanced NFL Betting System
Integrates all advanced features: injury data, team matchups, weather, and prop bets.
"""

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import requests
import json
from datetime import datetime, timedelta
import sqlite3
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our systems
from streamlined_enhanced_system import StreamlinedEnhancedPredictor
from injury_data_integration import InjuryDataIntegrator
from team_matchup_analyzer import TeamMatchupAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveEnhancedSystem:
    """Comprehensive enhanced NFL betting system with all advanced features."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Initialize subsystems
        self.predictor = StreamlinedEnhancedPredictor(db_path)
        self.injury_integrator = InjuryDataIntegrator(db_path)
        self.matchup_analyzer = TeamMatchupAnalyzer(db_path)
        
        # Load models
        self.predictor.load_models()
        
        # Prop bet thresholds
        self.prop_thresholds = {
            'QB': {
                'passing_yards': [249.5, 274.5, 299.5],
                'passing_touchdowns': [1.5, 2.5],
                'rushing_yards': [19.5, 29.5]
            },
            'RB': {
                'rushing_yards': [49.5, 74.5, 99.5],
                'rushing_touchdowns': [0.5, 1.5],
                'receiving_yards': [19.5, 29.5]
            },
            'WR': {
                'receiving_yards': [39.5, 59.5, 79.5],
                'receptions': [3.5, 5.5],
                'receiving_touchdowns': [0.5]
            },
            'TE': {
                'receiving_yards': [29.5, 49.5],
                'receptions': [2.5, 4.5],
                'receiving_touchdowns': [0.5]
            }
        }
        
        # Team abbreviation mapping for player IDs
        self.team_mapping = {
            'ari': 'ARI', 'atl': 'ATL', 'bal': 'BAL', 'buf': 'BUF', 'car': 'CAR',
            'chi': 'CHI', 'cin': 'CIN', 'cle': 'CLE', 'dal': 'DAL', 'den': 'DEN',
            'det': 'DET', 'gb': 'GB', 'hou': 'HOU', 'ind': 'IND', 'jax': 'JAX',
            'kc': 'KC', 'lv': 'LV', 'lac': 'LAC', 'lar': 'LAR', 'mia': 'MIA',
            'min': 'MIN', 'ne': 'NE', 'no': 'NO', 'nyg': 'NYG', 'nyj': 'NYJ',
            'phi': 'PHI', 'pit': 'PIT', 'sf': 'SF', 'sea': 'SEA', 'tb': 'TB',
            'ten': 'TEN', 'was': 'WAS'
        }
    
    def extract_team_from_player_id(self, player_id: str) -> str:
        """Extract team abbreviation from player_id."""
        # Format: teamname_position or team_playername_position
        parts = player_id.lower().split('_')
        if len(parts) >= 2:
            team_part = parts[0]
            return self.team_mapping.get(team_part, team_part.upper())
        return 'UNK'
    
    def get_enhanced_prediction(self, player_id: str, target: str, 
                              opponent_team: str = None) -> Dict[str, Any]:
        """Get enhanced prediction with all adjustments."""
        
        # Get base prediction
        base_result = self.predictor.predict(player_id, target)
        if not base_result:
            return {}
        
        base_prediction = base_result.get('prediction', 0)
        confidence = base_result.get('confidence', 0.5)
        
        # Extract player info
        position = player_id.split('_')[-1].upper()
        player_team = self.extract_team_from_player_id(player_id)
        player_name = player_id.replace('_', ' ').title()
        
        # Get injury adjustment
        injury_factor = self.injury_integrator.calculate_injury_impact(player_name, position)
        
        # Get matchup factors if opponent provided
        matchup_factor = 1.0
        if opponent_team:
            matchup_factors = self.matchup_analyzer.get_positional_matchup_factors(
                player_team, opponent_team, position
            )
            matchup_factor = matchup_factors.get('base_matchup', 1.0)
        
        # Apply adjustments
        adjusted_prediction = base_prediction * injury_factor * matchup_factor
        adjusted_confidence = confidence * min(injury_factor + 0.5, 1.0)
        
        return {
            'player_id': player_id,
            'target': target,
            'base_prediction': base_prediction,
            'adjusted_prediction': adjusted_prediction,
            'confidence': adjusted_confidence,
            'adjustments': {
                'injury_factor': injury_factor,
                'matchup_factor': matchup_factor,
                'total_adjustment': injury_factor * matchup_factor
            },
            'player_info': {
                'name': player_name,
                'position': position,
                'team': player_team
            }
        }
    
    def calculate_prop_bet_value(self, prediction: float, line: float, 
                               confidence: float, bet_type: str = 'over') -> Dict[str, float]:
        """Calculate prop bet value and edge."""
        
        # Estimate probability
        std_dev = line * 0.25  # Assume 25% standard deviation
        z_score = (prediction - line) / std_dev
        
        if bet_type.lower() == 'over':
            # Probability of going over
            prob = 0.5 + 0.5 * np.tanh(z_score * 0.8)
        else:
            # Probability of going under
            prob = 0.5 + 0.5 * np.tanh(-z_score * 0.8)
        
        # Adjust by confidence
        adjusted_prob = prob * confidence + 0.5 * (1 - confidence)
        
        # Calculate edge (assuming -110 odds)
        breakeven = 0.5238
        edge = adjusted_prob - breakeven
        
        # Kelly criterion bet size
        if edge > 0:
            kelly_size = max(0, min(0.05, edge * 2))  # Conservative sizing
        else:
            kelly_size = 0
        
        return {
            'probability': adjusted_prob,
            'edge': edge,
            'bet_size': kelly_size,
            'value_rating': 'HIGH' if edge > 0.05 else 'MEDIUM' if edge > 0.02 else 'LOW'
        }
    
    def generate_player_recommendations(self, player_id: str, 
                                      opponent_team: str = None) -> List[Dict[str, Any]]:
        """Generate comprehensive recommendations for a player."""
        
        position = player_id.split('_')[-1].upper()
        if position not in self.prop_thresholds:
            return []
        
        recommendations = []
        
        # Get predictions for relevant targets
        position_targets = {
            'QB': ['passing_yards', 'passing_touchdowns', 'rushing_yards'],
            'RB': ['rushing_yards', 'rushing_touchdowns', 'receiving_yards'],
            'WR': ['receiving_yards', 'receptions', 'receiving_touchdowns'],
            'TE': ['receiving_yards', 'receptions', 'receiving_touchdowns']
        }
        
        targets = position_targets.get(position, [])
        
        for target in targets:
            prediction_result = self.get_enhanced_prediction(player_id, target, opponent_team)
            
            if not prediction_result:
                continue
            
            prediction = prediction_result['adjusted_prediction']
            confidence = prediction_result['confidence']
            
            # Check prop bet lines for this target
            lines = self.prop_thresholds[position].get(target, [])
            
            for line in lines:
                # Over bet
                over_value = self.calculate_prop_bet_value(prediction, line, confidence, 'over')
                if over_value['edge'] > 0.01:  # Minimum 1% edge
                    recommendations.append({
                        'player_id': player_id,
                        'player_name': prediction_result['player_info']['name'],
                        'position': position,
                        'team': prediction_result['player_info']['team'],
                        'target': target,
                        'bet_type': 'OVER',
                        'line': line,
                        'prediction': prediction,
                        'edge': over_value['edge'],
                        'probability': over_value['probability'],
                        'bet_size': over_value['bet_size'],
                        'value_rating': over_value['value_rating'],
                        'confidence': confidence,
                        'adjustments': prediction_result['adjustments']
                    })
                
                # Under bet
                under_value = self.calculate_prop_bet_value(prediction, line, confidence, 'under')
                if under_value['edge'] > 0.01:
                    recommendations.append({
                        'player_id': player_id,
                        'player_name': prediction_result['player_info']['name'],
                        'position': position,
                        'team': prediction_result['player_info']['team'],
                        'target': target,
                        'bet_type': 'UNDER',
                        'line': line,
                        'prediction': prediction,
                        'edge': under_value['edge'],
                        'probability': under_value['probability'],
                        'bet_size': under_value['bet_size'],
                        'value_rating': under_value['value_rating'],
                        'confidence': confidence,
                        'adjustments': prediction_result['adjustments']
                    })
        
        # Sort by edge
        recommendations.sort(key=lambda x: x['edge'], reverse=True)
        return recommendations
    
    def get_top_players_by_position(self, position: str, limit: int = 10) -> List[str]:
        """Get top players by position from database."""
        
        query = text(f"""
            SELECT player_id, AVG(fantasy_points_ppr) as avg_points
            FROM player_game_stats 
            WHERE player_id LIKE '%_{position.lower()}'
            AND created_at >= date('now', '-30 days')
            GROUP BY player_id
            HAVING COUNT(*) >= 3
            ORDER BY avg_points DESC
            LIMIT {limit}
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            logger.warning(f"Error getting top players: {e}")
            return []
    
    def generate_daily_recommendations(self, max_recommendations: int = 25) -> Dict[str, Any]:
        """Generate comprehensive daily recommendations."""
        
        all_recommendations = []
        
        # Get top players for each position
        for position in ['QB', 'RB', 'WR', 'TE']:
            top_players = self.get_top_players_by_position(position, 5)
            
            for player_id in top_players:
                try:
                    player_recs = self.generate_player_recommendations(player_id)
                    all_recommendations.extend(player_recs)
                except Exception as e:
                    logger.warning(f"Error generating recommendations for {player_id}: {e}")
        
        # Sort by edge and limit
        all_recommendations.sort(key=lambda x: x['edge'], reverse=True)
        top_recommendations = all_recommendations[:max_recommendations]
        
        # Generate summary statistics
        total_edge = sum(rec['edge'] for rec in top_recommendations)
        total_bet_size = sum(rec['bet_size'] for rec in top_recommendations)
        
        high_value_bets = [rec for rec in top_recommendations if rec['value_rating'] == 'HIGH']
        medium_value_bets = [rec for rec in top_recommendations if rec['value_rating'] == 'MEDIUM']
        
        position_breakdown = {}
        for rec in top_recommendations:
            pos = rec['position']
            if pos not in position_breakdown:
                position_breakdown[pos] = {'count': 0, 'total_edge': 0}
            position_breakdown[pos]['count'] += 1
            position_breakdown[pos]['total_edge'] += rec['edge']
        
        return {
            'recommendations': top_recommendations,
            'summary': {
                'total_recommendations': len(top_recommendations),
                'high_value_count': len(high_value_bets),
                'medium_value_count': len(medium_value_bets),
                'total_edge': total_edge,
                'total_bet_allocation': total_bet_size,
                'position_breakdown': position_breakdown
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Test database connection
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM player_game_stats"))
                db_records = result.fetchone()[0]
            db_status = "Connected"
        except Exception as e:
            db_records = 0
            db_status = f"Error: {e}"
        
        # Test models
        model_count = len(self.predictor.models)
        
        # Test injury data
        try:
            injury_data = self.injury_integrator.update_injury_data()
            injury_count = len(injury_data)
            injury_status = "Active"
        except Exception as e:
            injury_count = 0
            injury_status = f"Error: {e}"
        
        # Test team ratings
        try:
            team_ratings = self.matchup_analyzer.calculate_team_ratings()
            team_count = len(team_ratings)
            matchup_status = "Active"
        except Exception as e:
            team_count = 0
            matchup_status = f"Error: {e}"
        
        return {
            'database': {
                'status': db_status,
                'records': db_records
            },
            'models': {
                'loaded': model_count,
                'status': "Ready" if model_count > 0 else "No models loaded"
            },
            'injury_system': {
                'status': injury_status,
                'players_tracked': injury_count
            },
            'matchup_system': {
                'status': matchup_status,
                'teams_analyzed': team_count
            },
            'overall_status': "Operational" if all([
                db_status == "Connected",
                model_count > 0,
                injury_status == "Active"
            ]) else "Partial"
        }

def main():
    """Test the comprehensive enhanced system."""
    print("🚀 Comprehensive Enhanced NFL Betting System")
    print("=" * 60)
    
    # Initialize system
    system = ComprehensiveEnhancedSystem()
    
    # Check system status
    print("📊 System Status Check...")
    status = system.get_system_status()
    
    print(f"   Database: {status['database']['status']} ({status['database']['records']} records)")
    print(f"   Models: {status['models']['status']} ({status['models']['loaded']} loaded)")
    print(f"   Injury System: {status['injury_system']['status']} ({status['injury_system']['players_tracked']} players)")
    print(f"   Matchup System: {status['matchup_system']['status']} ({status['matchup_system']['teams_analyzed']} teams)")
    print(f"   Overall: {status['overall_status']}")
    
    # Test individual player prediction
    print("\n🏈 Testing Enhanced Predictions...")
    test_players = system.get_top_players_by_position('QB', 2)
    
    if test_players:
        test_player = test_players[0]
        print(f"   Testing player: {test_player}")
        
        prediction = system.get_enhanced_prediction(test_player, 'passing_yards', 'BUF')
        if prediction:
            print(f"   Base Prediction: {prediction['base_prediction']:.1f}")
            print(f"   Adjusted Prediction: {prediction['adjusted_prediction']:.1f}")
            print(f"   Confidence: {prediction['confidence']:.3f}")
            print(f"   Injury Factor: {prediction['adjustments']['injury_factor']:.3f}")
            print(f"   Matchup Factor: {prediction['adjustments']['matchup_factor']:.3f}")
    
    # Test recommendations
    print("\n🎯 Generating Daily Recommendations...")
    daily_recs = system.generate_daily_recommendations(10)
    
    summary = daily_recs['summary']
    print(f"   Total Recommendations: {summary['total_recommendations']}")
    print(f"   High Value Bets: {summary['high_value_count']}")
    print(f"   Total Edge: {summary['total_edge']:.3f}")
    print(f"   Bet Allocation: {summary['total_bet_allocation']:.1%}")
    
    if daily_recs['recommendations']:
        print("\n🏆 Top 5 Recommendations:")
        for i, rec in enumerate(daily_recs['recommendations'][:5]):
            print(f"   {i+1}. {rec['player_name']} {rec['bet_type']} {rec['line']} {rec['target']}")
            print(f"      Edge: {rec['edge']:.3f} | Size: {rec['bet_size']:.1%} | Value: {rec['value_rating']}")
    
    print("\n✅ Comprehensive enhanced system test complete!")

if __name__ == "__main__":
    main()
