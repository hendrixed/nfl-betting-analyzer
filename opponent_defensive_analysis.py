#!/usr/bin/env python3
"""
Opponent Defensive Analysis System
Defensive rankings, matchup-specific adjustments, and opponent-based predictions.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpponentDefensiveAnalysis:
    """Comprehensive opponent defensive analysis and matchup adjustments."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Defensive position mappings
        self.defensive_categories = {
            'pass_defense': {
                'targets': ['QB', 'WR', 'TE'],
                'stats': ['passing_yards', 'passing_touchdowns', 'receiving_yards', 'receiving_touchdowns', 'receptions']
            },
            'rush_defense': {
                'targets': ['RB', 'QB'],
                'stats': ['rushing_yards', 'rushing_touchdowns', 'rushing_attempts']
            },
            'red_zone_defense': {
                'targets': ['QB', 'RB', 'WR', 'TE'],
                'stats': ['rushing_touchdowns', 'receiving_touchdowns', 'passing_touchdowns']
            }
        }
        
        # Team defensive strength tiers (updated with real data)
        self.defensive_tiers = {
            'elite': ['SF', 'BUF', 'DAL', 'PIT'],
            'good': ['BAL', 'DEN', 'MIA', 'NYJ', 'CLE'],
            'average': ['KC', 'PHI', 'SEA', 'MIN', 'LAC', 'TB', 'NO'],
            'below_average': ['GB', 'LAR', 'ATL', 'IND', 'TEN', 'NE'],
            'poor': ['WAS', 'DET', 'CAR', 'NYG', 'JAX', 'LV', 'CHI', 'ARI', 'HOU', 'CIN']
        }
        
        # Defensive multipliers by tier
        self.tier_multipliers = {
            'elite': 0.75,
            'good': 0.85,
            'average': 1.0,
            'below_average': 1.15,
            'poor': 1.30
        }
        
        # Position-specific defensive impacts
        self.position_defensive_weights = {
            'QB': {
                'pass_rush': 0.8,
                'secondary': 0.6,
                'overall_defense': 0.7
            },
            'RB': {
                'run_defense': 0.9,
                'linebacker': 0.7,
                'overall_defense': 0.8
            },
            'WR': {
                'secondary': 0.9,
                'pass_rush': 0.3,
                'overall_defense': 0.6
            },
            'TE': {
                'linebacker': 0.8,
                'secondary': 0.7,
                'overall_defense': 0.7
            }
        }
        
        # Cache for defensive calculations
        self.defensive_cache = {}
        self.cache_expiry = timedelta(hours=6)
        
    def get_team_defensive_tier(self, team: str) -> str:
        """Get defensive tier for a team."""
        for tier, teams in self.defensive_tiers.items():
            if team.upper() in teams:
                return tier
        return 'average'
    
    def calculate_defensive_rankings(self, games: int = 8) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive defensive rankings by position."""
        
        cache_key = f"defensive_rankings_{games}"
        if cache_key in self.defensive_cache:
            cache_time = self.defensive_cache[cache_key].get('timestamp', datetime.min)
            if datetime.now() - cache_time < self.cache_expiry:
                return self.defensive_cache[cache_key]['data']
        
        # Calculate defensive stats by extracting opponent team from player_id
        query = text(f"""
            SELECT 
                CASE 
                    WHEN player_id LIKE '%_qb' THEN SUBSTR(player_id, 1, LENGTH(player_id) - 3)
                    WHEN player_id LIKE '%_rb' THEN SUBSTR(player_id, 1, LENGTH(player_id) - 3)
                    WHEN player_id LIKE '%_wr' THEN SUBSTR(player_id, 1, LENGTH(player_id) - 3)
                    WHEN player_id LIKE '%_te' THEN SUBSTR(player_id, 1, LENGTH(player_id) - 3)
                    ELSE SUBSTR(player_id, 1, 3)
                END as opponent_team,
                AVG(passing_yards) as avg_pass_yards_allowed,
                AVG(passing_touchdowns) as avg_pass_tds_allowed,
                AVG(rushing_yards) as avg_rush_yards_allowed,
                AVG(rushing_touchdowns) as avg_rush_tds_allowed,
                AVG(receiving_yards) as avg_rec_yards_allowed,
                AVG(receiving_touchdowns) as avg_rec_tds_allowed,
                AVG(fantasy_points_ppr) as avg_fantasy_allowed,
                COUNT(*) as games_analyzed
            FROM player_game_stats 
            WHERE created_at >= date('now', '-{games * 7} days')
            GROUP BY opponent_team
            HAVING games_analyzed >= 3
        """)
        
        defensive_rankings = {}
        
        try:
            with self.engine.connect() as conn:
                results = conn.execute(query).fetchall()
                
                for row in results:
                    team = row[0].upper()
                    if len(team) <= 3:
                        defensive_rankings[team] = {
                            'pass_yards_allowed': row[1] or 0,
                            'pass_tds_allowed': row[2] or 0,
                            'rush_yards_allowed': row[3] or 0,
                            'rush_tds_allowed': row[4] or 0,
                            'rec_yards_allowed': row[5] or 0,
                            'rec_tds_allowed': row[6] or 0,
                            'fantasy_allowed': row[7] or 0,
                            'games_analyzed': row[8]
                        }
                
                # Calculate rankings and percentiles
                for team in defensive_rankings:
                    tier = self.get_team_defensive_tier(team)
                    defensive_rankings[team]['tier'] = tier
                    defensive_rankings[team]['tier_multiplier'] = self.tier_multipliers[tier]
                
                # Cache results
                self.defensive_cache[cache_key] = {
                    'data': defensive_rankings,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Calculated defensive rankings for {len(defensive_rankings)} teams")
                return defensive_rankings
                
        except Exception as e:
            logger.error(f"Error calculating defensive rankings: {e}")
            return {}
    
    def get_position_matchup_difficulty(self, opponent_team: str, position: str, 
                                      stat_type: str = 'fantasy_points_ppr') -> Dict[str, float]:
        """Get matchup difficulty for specific position vs opponent."""
        
        defensive_rankings = self.calculate_defensive_rankings()
        
        if opponent_team not in defensive_rankings:
            return {
                'difficulty_score': 0.5,
                'multiplier': 1.0,
                'tier': 'average',
                'rank_percentile': 0.5
            }
        
        opponent_data = defensive_rankings[opponent_team]
        tier = opponent_data['tier']
        
        # Position-specific adjustments
        if position == 'QB':
            base_allowed = opponent_data.get('pass_yards_allowed', 250)
            league_avg = 250
        elif position == 'RB':
            base_allowed = opponent_data.get('rush_yards_allowed', 100)
            league_avg = 100
        elif position in ['WR', 'TE']:
            base_allowed = opponent_data.get('rec_yards_allowed', 60)
            league_avg = 60
        else:
            base_allowed = opponent_data.get('fantasy_allowed', 10)
            league_avg = 10
        
        # Calculate difficulty (lower allowed = harder matchup)
        difficulty_ratio = base_allowed / league_avg
        difficulty_score = max(0.1, min(1.9, 2.0 - difficulty_ratio))
        
        # Apply tier multiplier
        tier_multiplier = opponent_data['tier_multiplier']
        
        return {
            'difficulty_score': difficulty_score,
            'multiplier': tier_multiplier,
            'tier': tier,
            'stats_allowed': base_allowed,
            'league_average': league_avg,
            'rank_percentile': self._calculate_defensive_percentile(opponent_team, position)
        }
    
    def _calculate_defensive_percentile(self, team: str, position: str) -> float:
        """Calculate where team ranks defensively against position (0-1 scale)."""
        
        defensive_rankings = self.calculate_defensive_rankings()
        
        if position == 'QB':
            stat_key = 'pass_yards_allowed'
        elif position == 'RB':
            stat_key = 'rush_yards_allowed'
        else:
            stat_key = 'rec_yards_allowed'
        
        team_stat = defensive_rankings.get(team, {}).get(stat_key, 0)
        all_stats = [data.get(stat_key, 0) for data in defensive_rankings.values()]
        
        if not all_stats or team_stat == 0:
            return 0.5
        
        # Lower stats allowed = better defense = higher percentile
        better_count = sum(1 for stat in all_stats if stat < team_stat)
        percentile = better_count / len(all_stats)
        
        return percentile
    
    def get_recent_defensive_trends(self, team: str, games: int = 4) -> Dict[str, float]:
        """Analyze recent defensive performance trends."""
        
        query = text(f"""
            SELECT 
                AVG(fantasy_points_ppr) as avg_fantasy_allowed,
                COUNT(*) as games,
                AVG(passing_yards + rushing_yards + receiving_yards) as avg_total_yards
            FROM player_game_stats 
            WHERE player_id LIKE '{team.lower()}_%'
            AND created_at >= date('now', '-{games * 7} days')
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchone()
                
                if result and result[1] > 0:
                    recent_allowed = result[0] or 0
                    
                    # Compare to season average
                    season_query = text(f"""
                        SELECT AVG(fantasy_points_ppr) as season_avg
                        FROM player_game_stats 
                        WHERE player_id LIKE '{team.lower()}_%'
                        AND created_at >= date('now', '-84 days')
                    """)
                    
                    season_result = conn.execute(season_query).fetchone()
                    season_avg = season_result[0] if season_result else recent_allowed
                    
                    trend_factor = recent_allowed / season_avg if season_avg > 0 else 1.0
                    
                    return {
                        'recent_fantasy_allowed': recent_allowed,
                        'season_fantasy_allowed': season_avg,
                        'trend_factor': trend_factor,
                        'trending_direction': 'worse' if trend_factor > 1.05 else 'better' if trend_factor < 0.95 else 'stable'
                    }
                    
        except Exception as e:
            logger.warning(f"Error calculating defensive trends for {team}: {e}")
        
        return {
            'recent_fantasy_allowed': 10.0,
            'season_fantasy_allowed': 10.0,
            'trend_factor': 1.0,
            'trending_direction': 'stable'
        }
    
    def calculate_coaching_adjustments(self, team: str) -> Dict[str, float]:
        """Calculate coaching tendency adjustments."""
        
        # Coaching tendencies (would be updated with real data)
        coaching_tendencies = {
            'SF': {'aggressive_defense': 1.1, 'blitz_rate': 1.2},
            'BUF': {'aggressive_defense': 1.05, 'blitz_rate': 1.1},
            'NE': {'conservative_defense': 0.95, 'situational': 1.1},
            'PIT': {'aggressive_defense': 1.08, 'blitz_rate': 1.15},
            'BAL': {'aggressive_defense': 1.06, 'blitz_rate': 1.1}
        }
        
        return coaching_tendencies.get(team, {
            'aggressive_defense': 1.0,
            'blitz_rate': 1.0,
            'conservative_defense': 1.0,
            'situational': 1.0
        })
    
    def create_opponent_adjustment_features(self, opponent_team: str, position: str,
                                          stat_type: str = 'fantasy_points_ppr') -> Dict[str, float]:
        """Create comprehensive opponent adjustment features."""
        
        # Get matchup difficulty
        matchup_data = self.get_position_matchup_difficulty(opponent_team, position, stat_type)
        
        # Get recent trends
        trend_data = self.get_recent_defensive_trends(opponent_team)
        
        # Get coaching adjustments
        coaching_data = self.calculate_coaching_adjustments(opponent_team)
        
        features = {
            # Matchup features
            'opponent_tier_elite': 1.0 if matchup_data['tier'] == 'elite' else 0.0,
            'opponent_tier_good': 1.0 if matchup_data['tier'] == 'good' else 0.0,
            'opponent_tier_poor': 1.0 if matchup_data['tier'] == 'poor' else 0.0,
            'matchup_difficulty': matchup_data['difficulty_score'],
            'opponent_multiplier': matchup_data['multiplier'],
            'defensive_rank_percentile': matchup_data['rank_percentile'],
            
            # Trend features
            'defensive_trend_factor': trend_data['trend_factor'],
            'defense_trending_worse': 1.0 if trend_data['trending_direction'] == 'worse' else 0.0,
            'defense_trending_better': 1.0 if trend_data['trending_direction'] == 'better' else 0.0,
            
            # Coaching features
            'opponent_aggressive_defense': coaching_data.get('aggressive_defense', 1.0),
            'opponent_blitz_rate': coaching_data.get('blitz_rate', 1.0),
            
            # Combined features
            'total_opponent_factor': (matchup_data['multiplier'] * 
                                    trend_data['trend_factor'] * 
                                    coaching_data.get('aggressive_defense', 1.0)),
            'matchup_advantage': 1.0 if matchup_data['multiplier'] > 1.1 else 
                               -1.0 if matchup_data['multiplier'] < 0.9 else 0.0
        }
        
        return features
    
    def get_injury_adjusted_defense(self, team: str, key_injuries: List[str] = None) -> Dict[str, float]:
        """Adjust defensive ratings based on key injuries."""
        
        if not key_injuries:
            return {'injury_adjustment': 1.0, 'adjusted_tier': self.get_team_defensive_tier(team)}
        
        # Impact of defensive injuries (simplified)
        injury_impact = {
            'CB': 0.15,    # Cornerback injury
            'S': 0.10,     # Safety injury  
            'LB': 0.12,    # Linebacker injury
            'DE': 0.08,    # Defensive end injury
            'DT': 0.06     # Defensive tackle injury
        }
        
        total_impact = 0
        for injury in key_injuries:
            for position, impact in injury_impact.items():
                if position.lower() in injury.lower():
                    total_impact += impact
        
        # Adjust defensive multiplier
        injury_adjustment = 1.0 + min(0.3, total_impact)  # Max 30% worse defense
        
        return {
            'injury_adjustment': injury_adjustment,
            'injury_impact': total_impact,
            'adjusted_tier': 'worse' if injury_adjustment > 1.1 else self.get_team_defensive_tier(team)
        }
    
    def generate_matchup_report(self, player_position: str, opponent_team: str) -> Dict[str, Any]:
        """Generate comprehensive matchup report."""
        
        matchup_data = self.get_position_matchup_difficulty(opponent_team, player_position)
        trend_data = self.get_recent_defensive_trends(opponent_team)
        features = self.create_opponent_adjustment_features(opponent_team, player_position)
        
        # Generate recommendation
        if matchup_data['multiplier'] > 1.15:
            recommendation = 'FAVORABLE'
        elif matchup_data['multiplier'] < 0.85:
            recommendation = 'DIFFICULT'
        else:
            recommendation = 'NEUTRAL'
        
        return {
            'opponent_team': opponent_team,
            'player_position': player_position,
            'matchup_rating': recommendation,
            'difficulty_score': matchup_data['difficulty_score'],
            'opponent_tier': matchup_data['tier'],
            'multiplier': matchup_data['multiplier'],
            'defensive_trend': trend_data['trending_direction'],
            'key_factors': {
                'tier_advantage': matchup_data['tier'] in ['below_average', 'poor'],
                'recent_struggles': trend_data['trending_direction'] == 'worse',
                'tough_matchup': matchup_data['tier'] in ['elite', 'good']
            },
            'confidence': min(1.0, abs(matchup_data['multiplier'] - 1.0) + 0.5)
        }

def main():
    """Test the opponent defensive analysis system."""
    print("🛡️ Opponent Defensive Analysis System")
    print("=" * 60)
    
    # Initialize system
    defense = OpponentDefensiveAnalysis()
    
    # Test defensive rankings
    print("📊 Calculating Defensive Rankings...")
    rankings = defense.calculate_defensive_rankings(8)
    print(f"   Analyzed {len(rankings)} teams")
    
    if rankings:
        # Show sample rankings
        sample_teams = list(rankings.keys())[:3]
        for team in sample_teams:
            data = rankings[team]
            print(f"   {team}: Tier {data['tier']}, Fantasy Allowed: {data['fantasy_allowed']:.1f}")
    
    # Test position matchup difficulty
    print("\n⚔️ Position Matchup Analysis (RB vs SF):")
    matchup = defense.get_position_matchup_difficulty('SF', 'RB')
    print(f"   Difficulty Score: {matchup['difficulty_score']:.3f}")
    print(f"   Multiplier: {matchup['multiplier']:.3f}")
    print(f"   Tier: {matchup['tier']}")
    print(f"   Rank Percentile: {matchup['rank_percentile']:.1%}")
    
    # Test defensive trends
    print("\n📈 Recent Defensive Trends (KC):")
    trends = defense.get_recent_defensive_trends('KC', 4)
    print(f"   Recent Fantasy Allowed: {trends['recent_fantasy_allowed']:.1f}")
    print(f"   Season Average: {trends['season_fantasy_allowed']:.1f}")
    print(f"   Trend Factor: {trends['trend_factor']:.3f}")
    print(f"   Direction: {trends['trending_direction']}")
    
    # Test opponent features
    print("\n🔧 Opponent Adjustment Features (WR vs BUF):")
    features = defense.create_opponent_adjustment_features('BUF', 'WR')
    key_features = ['matchup_difficulty', 'opponent_multiplier', 'total_opponent_factor', 'matchup_advantage']
    
    for feature in key_features:
        if feature in features:
            print(f"   {feature}: {features[feature]:.3f}")
    
    print(f"   Total Features: {len(features)}")
    
    # Test matchup report
    print("\n📋 Comprehensive Matchup Report (QB vs SF):")
    report = defense.generate_matchup_report('QB', 'SF')
    print(f"   Matchup Rating: {report['matchup_rating']}")
    print(f"   Opponent Tier: {report['opponent_tier']}")
    print(f"   Multiplier: {report['multiplier']:.3f}")
    print(f"   Confidence: {report['confidence']:.3f}")
    
    key_factors = report['key_factors']
    print("   Key Factors:")
    for factor, value in key_factors.items():
        print(f"   - {factor}: {'Yes' if value else 'No'}")
    
    print("\n✅ Opponent defensive analysis system test complete!")

if __name__ == "__main__":
    main()
