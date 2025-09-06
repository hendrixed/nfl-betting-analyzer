#!/usr/bin/env python3
"""
Advanced NFL Prop Bet Recommendation System
Integrates injury data, team matchups, weather, and advanced analytics for comprehensive betting recommendations.
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

# Import our new systems
from injury_data_integration import InjuryDataIntegrator
from team_matchup_analyzer import TeamMatchupAnalyzer
from streamlined_enhanced_system import StreamlinedEnhancedPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPropBetSystem:
    """Advanced prop bet recommendation system with comprehensive analysis."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Initialize subsystems
        self.injury_integrator = InjuryDataIntegrator(db_path)
        self.matchup_analyzer = TeamMatchupAnalyzer(db_path)
        self.predictor = StreamlinedEnhancedPredictor(db_path)
        
        # Prop bet categories and thresholds
        self.prop_categories = {
            'QB': {
                'passing_yards': [249.5, 274.5, 299.5, 324.5],
                'passing_touchdowns': [1.5, 2.5, 3.5],
                'rushing_yards': [19.5, 29.5, 39.5],
                'completions': [19.5, 24.5, 29.5],
                'interceptions': [0.5, 1.5]
            },
            'RB': {
                'rushing_yards': [49.5, 74.5, 99.5, 124.5],
                'rushing_touchdowns': [0.5, 1.5],
                'receiving_yards': [19.5, 29.5, 39.5],
                'receptions': [2.5, 4.5, 6.5]
            },
            'WR': {
                'receiving_yards': [39.5, 59.5, 79.5, 99.5],
                'receptions': [3.5, 5.5, 7.5],
                'receiving_touchdowns': [0.5, 1.5],
                'longest_reception': [19.5, 29.5]
            },
            'TE': {
                'receiving_yards': [29.5, 49.5, 69.5],
                'receptions': [2.5, 4.5, 6.5],
                'receiving_touchdowns': [0.5, 1.5]
            }
        }
        
        # Confidence thresholds for recommendations
        self.confidence_thresholds = {
            'high': 0.75,
            'medium': 0.60,
            'low': 0.50
        }
        
        # Edge calculation parameters
        self.edge_params = {
            'min_edge': 0.05,  # Minimum 5% edge
            'max_bet_size': 0.10,  # Maximum 10% of bankroll
            'kelly_fraction': 0.25  # Conservative Kelly fraction
        }
    
    def get_comprehensive_prediction(self, player_name: str, position: str, 
                                   opponent_team: str, player_team: str) -> Dict[str, Any]:
        """Get comprehensive prediction with all adjustments."""
        
        # Get base prediction from ML model
        base_predictions = self.predictor.predict_player_performance(player_name, position)
        
        if not base_predictions:
            return {}
        
        # Get injury adjustment
        injury_adjusted = self.injury_integrator.adjust_predictions_for_injuries(
            base_predictions, player_name, position
        )
        
        # Get matchup factors
        matchup_factors = self.matchup_analyzer.get_positional_matchup_factors(
            player_team, opponent_team, position
        )
        
        # Apply matchup adjustments
        final_predictions = {}
        for stat, prediction in injury_adjusted.items():
            matchup_multiplier = matchup_factors.get('base_matchup', 1.0)
            game_script_multiplier = matchup_factors.get('game_script', 1.0)
            
            # Combine adjustments
            total_adjustment = matchup_multiplier * game_script_multiplier
            final_predictions[stat] = prediction * total_adjustment
        
        # Calculate prediction confidence
        injury_impact = self.injury_integrator.calculate_injury_impact(player_name, position)
        matchup_confidence = min(1.0, abs(matchup_factors['base_matchup'] - 1.0) + 0.5)
        
        overall_confidence = (injury_impact + matchup_confidence) / 2
        
        return {
            'predictions': final_predictions,
            'confidence': overall_confidence,
            'adjustments': {
                'injury_factor': injury_impact,
                'matchup_factor': matchup_factors['base_matchup'],
                'game_script': matchup_factors['game_script']
            },
            'base_predictions': base_predictions
        }
    
    def calculate_prop_bet_edge(self, prediction: float, prop_line: float, 
                               confidence: float, bet_type: str = 'over') -> Dict[str, float]:
        """Calculate betting edge and recommended bet size."""
        
        # Estimate probability based on prediction and confidence
        if bet_type.lower() == 'over':
            # Probability of going over the line
            z_score = (prediction - prop_line) / (prop_line * 0.2)  # Assume 20% std dev
            prob_over = 0.5 + 0.5 * np.tanh(z_score)
        else:
            # Probability of going under the line
            z_score = (prop_line - prediction) / (prop_line * 0.2)
            prob_over = 0.5 + 0.5 * np.tanh(z_score)
        
        # Adjust probability by confidence
        adjusted_prob = prob_over * confidence + 0.5 * (1 - confidence)
        
        # Calculate edge assuming -110 odds (52.38% breakeven)
        breakeven_prob = 0.5238
        edge = adjusted_prob - breakeven_prob
        
        # Kelly criterion bet sizing
        if edge > 0:
            # Assuming -110 odds: b = 0.909, p = adjusted_prob, q = 1-p
            kelly_fraction = (adjusted_prob * 0.909 - (1 - adjusted_prob)) / 0.909
            recommended_bet_size = max(0, min(self.edge_params['max_bet_size'], 
                                            kelly_fraction * self.edge_params['kelly_fraction']))
        else:
            recommended_bet_size = 0
        
        return {
            'edge': edge,
            'probability': adjusted_prob,
            'recommended_bet_size': recommended_bet_size,
            'confidence_level': 'high' if confidence > self.confidence_thresholds['high'] else
                              'medium' if confidence > self.confidence_thresholds['medium'] else 'low'
        }
    
    def generate_prop_recommendations(self, player_name: str, position: str,
                                    opponent_team: str, player_team: str) -> List[Dict[str, Any]]:
        """Generate comprehensive prop bet recommendations."""
        
        # Get comprehensive prediction
        prediction_data = self.get_comprehensive_prediction(
            player_name, position, opponent_team, player_team
        )
        
        if not prediction_data:
            return []
        
        predictions = prediction_data['predictions']
        confidence = prediction_data['confidence']
        
        recommendations = []
        
        # Get prop categories for this position
        position_props = self.prop_categories.get(position, {})
        
        for stat_name, thresholds in position_props.items():
            if stat_name in predictions:
                prediction_value = predictions[stat_name]
                
                for threshold in thresholds:
                    # Calculate over bet
                    over_edge = self.calculate_prop_bet_edge(
                        prediction_value, threshold, confidence, 'over'
                    )
                    
                    # Calculate under bet
                    under_edge = self.calculate_prop_bet_edge(
                        prediction_value, threshold, confidence, 'under'
                    )
                    
                    # Add recommendations if edge is positive
                    if over_edge['edge'] > self.edge_params['min_edge']:
                        recommendations.append({
                            'player': player_name,
                            'position': position,
                            'stat': stat_name,
                            'bet_type': 'OVER',
                            'line': threshold,
                            'prediction': prediction_value,
                            'edge': over_edge['edge'],
                            'probability': over_edge['probability'],
                            'confidence': over_edge['confidence_level'],
                            'bet_size': over_edge['recommended_bet_size'],
                            'reasoning': self._generate_reasoning(prediction_data, stat_name, threshold, 'over')
                        })
                    
                    if under_edge['edge'] > self.edge_params['min_edge']:
                        recommendations.append({
                            'player': player_name,
                            'position': position,
                            'stat': stat_name,
                            'bet_type': 'UNDER',
                            'line': threshold,
                            'prediction': prediction_value,
                            'edge': under_edge['edge'],
                            'probability': under_edge['probability'],
                            'confidence': under_edge['confidence_level'],
                            'bet_size': under_edge['recommended_bet_size'],
                            'reasoning': self._generate_reasoning(prediction_data, stat_name, threshold, 'under')
                        })
        
        # Sort by edge (highest first)
        recommendations.sort(key=lambda x: x['edge'], reverse=True)
        
        return recommendations
    
    def _generate_reasoning(self, prediction_data: Dict[str, Any], stat: str, 
                          line: float, bet_type: str) -> str:
        """Generate human-readable reasoning for the recommendation."""
        
        prediction = prediction_data['predictions'][stat]
        adjustments = prediction_data['adjustments']
        
        reasoning_parts = []
        
        # Base prediction
        reasoning_parts.append(f"Model predicts {prediction:.1f} {stat}")
        
        # Injury impact
        if adjustments['injury_factor'] < 0.9:
            reasoning_parts.append(f"Injury concerns (impact: {adjustments['injury_factor']:.2f})")
        elif adjustments['injury_factor'] > 1.0:
            reasoning_parts.append("Player healthy")
        
        # Matchup impact
        if adjustments['matchup_factor'] > 1.1:
            reasoning_parts.append("Favorable matchup")
        elif adjustments['matchup_factor'] < 0.9:
            reasoning_parts.append("Tough matchup")
        
        # Game script
        if adjustments['game_script'] > 1.05:
            reasoning_parts.append("Game script favorable")
        elif adjustments['game_script'] < 0.95:
            reasoning_parts.append("Game script unfavorable")
        
        # Prediction vs line
        diff = prediction - line
        if bet_type == 'over':
            reasoning_parts.append(f"Prediction {diff:+.1f} vs line")
        else:
            reasoning_parts.append(f"Prediction {-diff:+.1f} under line")
        
        return " | ".join(reasoning_parts)
    
    def get_daily_recommendations(self, max_recommendations: int = 20) -> List[Dict[str, Any]]:
        """Get daily prop bet recommendations across all players."""
        
        # This would integrate with actual game schedule
        # For now, generate sample recommendations for key players
        
        sample_players = [
            ('Patrick Mahomes', 'QB', 'BUF', 'KC'),
            ('Josh Allen', 'QB', 'KC', 'BUF'),
            ('Derrick Henry', 'RB', 'HOU', 'TEN'),
            ('Davante Adams', 'WR', 'KC', 'LV'),
            ('Travis Kelce', 'TE', 'BUF', 'KC')
        ]
        
        all_recommendations = []
        
        for player_name, position, opponent, team in sample_players:
            try:
                player_recs = self.generate_prop_recommendations(
                    player_name, position, opponent, team
                )
                all_recommendations.extend(player_recs)
            except Exception as e:
                logger.warning(f"Failed to generate recommendations for {player_name}: {e}")
        
        # Sort by edge and return top recommendations
        all_recommendations.sort(key=lambda x: x['edge'], reverse=True)
        
        return all_recommendations[:max_recommendations]
    
    def generate_betting_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily betting report."""
        
        recommendations = self.get_daily_recommendations()
        
        # Categorize recommendations
        high_confidence = [r for r in recommendations if r['confidence'] == 'high']
        medium_confidence = [r for r in recommendations if r['confidence'] == 'medium']
        
        # Calculate total recommended bet sizes
        total_bet_size = sum(r['bet_size'] for r in recommendations)
        
        # Position breakdown
        position_breakdown = {}
        for rec in recommendations:
            pos = rec['position']
            if pos not in position_breakdown:
                position_breakdown[pos] = {'count': 0, 'avg_edge': 0}
            position_breakdown[pos]['count'] += 1
            position_breakdown[pos]['avg_edge'] += rec['edge']
        
        for pos in position_breakdown:
            position_breakdown[pos]['avg_edge'] /= position_breakdown[pos]['count']
        
        return {
            'total_recommendations': len(recommendations),
            'high_confidence_bets': len(high_confidence),
            'medium_confidence_bets': len(medium_confidence),
            'total_bankroll_allocation': total_bet_size,
            'position_breakdown': position_breakdown,
            'top_recommendations': recommendations[:10],
            'generated_at': datetime.now().isoformat()
        }

def main():
    """Test the advanced prop bet system."""
    print("🎯 Advanced NFL Prop Bet Recommendation System")
    print("=" * 60)
    
    # Initialize system
    prop_system = AdvancedPropBetSystem()
    
    # Test individual player recommendations
    print("🏈 Testing Player Recommendations (Patrick Mahomes vs BUF)...")
    recommendations = prop_system.generate_prop_recommendations(
        "Patrick Mahomes", "QB", "BUF", "KC"
    )
    
    print(f"   ✅ Generated {len(recommendations)} recommendations")
    
    if recommendations:
        print("\n📊 Top 3 Recommendations:")
        for i, rec in enumerate(recommendations[:3]):
            print(f"   {i+1}. {rec['bet_type']} {rec['line']} {rec['stat']}")
            print(f"      Edge: {rec['edge']:.3f} | Confidence: {rec['confidence']}")
            print(f"      Bet Size: {rec['bet_size']:.1%} | Prediction: {rec['prediction']:.1f}")
            print(f"      Reasoning: {rec['reasoning']}")
            print()
    
    # Test daily report
    print("📈 Generating Daily Betting Report...")
    report = prop_system.generate_betting_report()
    
    print(f"   Total Recommendations: {report['total_recommendations']}")
    print(f"   High Confidence Bets: {report['high_confidence_bets']}")
    print(f"   Total Bankroll Allocation: {report['total_bankroll_allocation']:.1%}")
    
    print("\n📋 Position Breakdown:")
    for pos, data in report['position_breakdown'].items():
        print(f"   {pos}: {data['count']} bets, {data['avg_edge']:.3f} avg edge")
    
    print("\n🏆 Top 5 Daily Recommendations:")
    for i, rec in enumerate(report['top_recommendations'][:5]):
        print(f"   {i+1}. {rec['player']} {rec['bet_type']} {rec['line']} {rec['stat']}")
        print(f"      Edge: {rec['edge']:.3f} | Size: {rec['bet_size']:.1%}")
    
    print("\n✅ Advanced prop bet system test complete!")

if __name__ == "__main__":
    main()
