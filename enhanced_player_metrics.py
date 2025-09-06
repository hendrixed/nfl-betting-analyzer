#!/usr/bin/env python3
"""
Enhanced Player Metrics System
Snap counts, target share, red zone usage, and advanced player analytics.
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

class EnhancedPlayerMetrics:
    """Enhanced player metrics including snap counts, usage rates, and situational performance."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Position-specific metric weights
        self.position_weights = {
            'QB': {
                'snap_percentage': 0.9,
                'red_zone_usage': 0.8,
                'third_down_usage': 0.7,
                'two_minute_usage': 0.8
            },
            'RB': {
                'snap_percentage': 0.8,
                'red_zone_usage': 0.9,
                'goal_line_usage': 0.95,
                'third_down_usage': 0.6,
                'two_minute_usage': 0.7
            },
            'WR': {
                'snap_percentage': 0.7,
                'target_share': 0.9,
                'red_zone_targets': 0.85,
                'air_yards_share': 0.8,
                'slot_usage': 0.6
            },
            'TE': {
                'snap_percentage': 0.6,
                'target_share': 0.8,
                'red_zone_targets': 0.9,
                'blocking_snaps': 0.4,
                'two_minute_usage': 0.7
            }
        }
        
        # Situational performance multipliers
        self.situational_multipliers = {
            'red_zone': {
                'high_usage': 1.3,
                'medium_usage': 1.1,
                'low_usage': 0.8
            },
            'goal_line': {
                'high_usage': 1.5,
                'medium_usage': 1.2,
                'low_usage': 0.6
            },
            'third_down': {
                'high_usage': 1.2,
                'medium_usage': 1.0,
                'low_usage': 0.9
            },
            'two_minute': {
                'high_usage': 1.4,
                'medium_usage': 1.1,
                'low_usage': 0.8
            }
        }
        
        # Cache for expensive calculations
        self.metrics_cache = {}
        self.cache_expiry = timedelta(hours=4)
        
    def calculate_snap_percentage(self, player_id: str, games: int = 5) -> Dict[str, float]:
        """Calculate snap percentage and usage trends."""
        
        query = text(f"""
            SELECT 
                AVG(CASE WHEN rushing_attempts + targets + passing_attempts > 0 THEN 1.0 ELSE 0.0 END) as active_snap_rate,
                COUNT(*) as games_played,
                AVG(rushing_attempts + targets + passing_attempts) as avg_touches,
                AVG(fantasy_points_ppr) as avg_fantasy_points
            FROM player_game_stats 
            WHERE player_id = :player_id
            AND created_at >= date('now', '-{games * 7} days')
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {'player_id': player_id}).fetchone()
                
                if result and result[1] > 0:
                    # Estimate snap percentage based on usage
                    position = player_id.split('_')[-1].upper()
                    
                    if position == 'QB':
                        snap_percentage = min(0.95, result[0] * 0.9 + 0.1)
                    elif position == 'RB':
                        snap_percentage = min(0.85, result[2] / 20 * 0.7 + 0.2)
                    elif position in ['WR', 'TE']:
                        snap_percentage = min(0.90, result[2] / 10 * 0.6 + 0.3)
                    else:
                        snap_percentage = 0.5
                    
                    return {
                        'snap_percentage': snap_percentage,
                        'games_analyzed': result[1],
                        'avg_touches': result[2] or 0,
                        'usage_trend': self._calculate_usage_trend(player_id, games),
                        'snap_consistency': self._calculate_snap_consistency(player_id, games)
                    }
                    
        except Exception as e:
            logger.warning(f"Error calculating snap percentage for {player_id}: {e}")
        
        return {
            'snap_percentage': 0.5,
            'games_analyzed': 0,
            'avg_touches': 0,
            'usage_trend': 0.0,
            'snap_consistency': 0.5
        }
    
    def _calculate_usage_trend(self, player_id: str, games: int) -> float:
        """Calculate if player usage is trending up or down."""
        
        query = text(f"""
            SELECT 
                (rushing_attempts + targets + passing_attempts) as touches,
                ROW_NUMBER() OVER (ORDER BY created_at DESC) as game_rank
            FROM player_game_stats 
            WHERE player_id = :player_id
            AND created_at >= date('now', '-{games * 7} days')
            ORDER BY created_at DESC
            LIMIT {games}
        """)
        
        try:
            with self.engine.connect() as conn:
                results = conn.execute(query, {'player_id': player_id}).fetchall()
                
                if len(results) >= 3:
                    recent_touches = np.mean([r[0] for r in results[:3]])
                    older_touches = np.mean([r[0] for r in results[3:]])
                    
                    if older_touches > 0:
                        trend = (recent_touches - older_touches) / older_touches
                        return max(-0.5, min(0.5, trend))  # Clamp between -50% and +50%
                        
        except Exception as e:
            logger.warning(f"Error calculating usage trend for {player_id}: {e}")
        
        return 0.0
    
    def _calculate_snap_consistency(self, player_id: str, games: int) -> float:
        """Calculate consistency of snap usage."""
        
        query = text(f"""
            SELECT (rushing_attempts + targets + passing_attempts) as touches
            FROM player_game_stats 
            WHERE player_id = :player_id
            AND created_at >= date('now', '-{games * 7} days')
            ORDER BY created_at DESC
            LIMIT {games}
        """)
        
        try:
            with self.engine.connect() as conn:
                results = conn.execute(query, {'player_id': player_id}).fetchall()
                touches = [r[0] for r in results]
                
                if len(touches) >= 3:
                    cv = np.std(touches) / np.mean(touches) if np.mean(touches) > 0 else 1.0
                    consistency = max(0.1, 1.0 - cv)  # Lower CV = higher consistency
                    return min(1.0, consistency)
                    
        except Exception as e:
            logger.warning(f"Error calculating snap consistency for {player_id}: {e}")
        
        return 0.5
    
    def calculate_target_share(self, player_id: str, games: int = 5) -> Dict[str, float]:
        """Calculate target share and receiving metrics for pass catchers."""
        
        position = player_id.split('_')[-1].upper()
        if position not in ['WR', 'TE']:
            return {'target_share': 0.0, 'air_yards_share': 0.0, 'red_zone_target_share': 0.0}
        
        # Extract team from player_id
        team_part = player_id.split('_')[0]
        
        query = text(f"""
            SELECT 
                player_id,
                AVG(targets) as avg_targets,
                AVG(receiving_yards) as avg_receiving_yards,
                COUNT(*) as games
            FROM player_game_stats 
            WHERE player_id LIKE '{team_part}_%'
            AND player_id LIKE '%_wr' OR player_id LIKE '%_te'
            AND created_at >= date('now', '-{games * 7} days')
            GROUP BY player_id
            HAVING games >= 2
        """)
        
        try:
            with self.engine.connect() as conn:
                results = conn.execute(query).fetchall()
                
                # Calculate team totals
                team_targets = sum(r[1] for r in results)
                team_air_yards = sum(r[2] for r in results)  # Using receiving yards as proxy
                
                # Find player's share
                player_targets = 0
                player_air_yards = 0
                
                for result in results:
                    if result[0] == player_id:
                        player_targets = result[1]
                        player_air_yards = result[2]
                        break
                
                target_share = player_targets / team_targets if team_targets > 0 else 0.0
                air_yards_share = player_air_yards / team_air_yards if team_air_yards > 0 else 0.0
                
                return {
                    'target_share': target_share,
                    'air_yards_share': air_yards_share,
                    'red_zone_target_share': self._calculate_red_zone_target_share(player_id, games),
                    'target_quality': self._calculate_target_quality(player_id, games)
                }
                
        except Exception as e:
            logger.warning(f"Error calculating target share for {player_id}: {e}")
        
        return {'target_share': 0.0, 'air_yards_share': 0.0, 'red_zone_target_share': 0.0, 'target_quality': 0.5}
    
    def _calculate_red_zone_target_share(self, player_id: str, games: int) -> float:
        """Estimate red zone target share (simplified calculation)."""
        
        query = text(f"""
            SELECT AVG(receiving_touchdowns) as avg_tds, AVG(targets) as avg_targets
            FROM player_game_stats 
            WHERE player_id = :player_id
            AND created_at >= date('now', '-{games * 7} days')
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {'player_id': player_id}).fetchone()
                
                if result and result[1] > 0:
                    # Estimate red zone usage based on TD rate
                    td_rate = result[0] / result[1] if result[1] > 0 else 0
                    red_zone_share = min(0.8, td_rate * 10)  # Rough estimation
                    return red_zone_share
                    
        except Exception as e:
            logger.warning(f"Error calculating red zone target share for {player_id}: {e}")
        
        return 0.0
    
    def _calculate_target_quality(self, player_id: str, games: int) -> float:
        """Calculate quality of targets (depth, catchability)."""
        
        query = text(f"""
            SELECT 
                AVG(receiving_yards) as avg_yards,
                AVG(receptions) as avg_receptions,
                AVG(targets) as avg_targets
            FROM player_game_stats 
            WHERE player_id = :player_id
            AND created_at >= date('now', '-{games * 7} days')
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {'player_id': player_id}).fetchone()
                
                if result and result[2] > 0:
                    catch_rate = result[1] / result[2] if result[2] > 0 else 0
                    yards_per_target = result[0] / result[2] if result[2] > 0 else 0
                    
                    # Quality score based on catch rate and depth
                    quality = (catch_rate * 0.6 + min(yards_per_target / 15, 1.0) * 0.4)
                    return min(1.0, quality)
                    
        except Exception as e:
            logger.warning(f"Error calculating target quality for {player_id}: {e}")
        
        return 0.5
    
    def calculate_red_zone_usage(self, player_id: str, games: int = 5) -> Dict[str, float]:
        """Calculate red zone and goal line usage patterns."""
        
        position = player_id.split('_')[-1].upper()
        
        query = text(f"""
            SELECT 
                AVG(rushing_touchdowns + receiving_touchdowns) as avg_tds,
                AVG(rushing_attempts + targets) as avg_opportunities,
                COUNT(*) as games
            FROM player_game_stats 
            WHERE player_id = :player_id
            AND created_at >= date('now', '-{games * 7} days')
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {'player_id': player_id}).fetchone()
                
                if result and result[2] > 0:
                    td_rate = result[0] / max(result[1], 1) if result[1] > 0 else 0
                    
                    # Estimate usage categories
                    if position == 'RB':
                        if td_rate > 0.15:
                            usage_level = 'high'
                        elif td_rate > 0.08:
                            usage_level = 'medium'
                        else:
                            usage_level = 'low'
                    else:  # WR/TE
                        if td_rate > 0.12:
                            usage_level = 'high'
                        elif td_rate > 0.06:
                            usage_level = 'medium'
                        else:
                            usage_level = 'low'
                    
                    return {
                        'red_zone_usage_level': usage_level,
                        'red_zone_td_rate': td_rate,
                        'red_zone_multiplier': self.situational_multipliers['red_zone'][usage_level],
                        'goal_line_multiplier': self.situational_multipliers['goal_line'].get(usage_level, 1.0)
                    }
                    
        except Exception as e:
            logger.warning(f"Error calculating red zone usage for {player_id}: {e}")
        
        return {
            'red_zone_usage_level': 'medium',
            'red_zone_td_rate': 0.05,
            'red_zone_multiplier': 1.0,
            'goal_line_multiplier': 1.0
        }
    
    def calculate_situational_performance(self, player_id: str, games: int = 8) -> Dict[str, float]:
        """Calculate performance in different game situations."""
        
        # This would require more detailed play-by-play data
        # For now, we'll estimate based on available stats
        
        query = text(f"""
            SELECT 
                AVG(fantasy_points_ppr) as avg_fantasy,
                AVG(rushing_attempts + targets + passing_attempts) as avg_usage,
                COUNT(*) as games,
                AVG((fantasy_points_ppr - (SELECT AVG(fantasy_points_ppr) FROM player_game_stats WHERE player_id = :player_id)) * 
                    (fantasy_points_ppr - (SELECT AVG(fantasy_points_ppr) FROM player_game_stats WHERE player_id = :player_id))) as variance
            FROM player_game_stats 
            WHERE player_id = :player_id
            AND created_at >= date('now', '-{games * 7} days')
        """)
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {'player_id': player_id}).fetchone()
                
                if result and result[2] > 0:
                    avg_fantasy = result[0] or 0
                    avg_usage = result[1] or 0
                    variance = result[3] or 0
                    std_dev = variance ** 0.5 if variance > 0 else 0
                    consistency = 1.0 - min(1.0, std_dev / max(avg_fantasy, 1))
                    
                    # Estimate situational performance
                    third_down_performance = min(1.5, avg_usage / 10 + 0.8)
                    two_minute_performance = min(1.4, avg_fantasy / 15 + 0.9)
                    blowout_performance = max(0.6, 1.0 - (avg_fantasy / 20))
                    
                    return {
                        'third_down_factor': third_down_performance,
                        'two_minute_factor': two_minute_performance,
                        'blowout_factor': blowout_performance,
                        'consistency_score': consistency,
                        'clutch_performance': (third_down_performance + two_minute_performance) / 2
                    }
                    
        except Exception as e:
            logger.warning(f"Error calculating situational performance for {player_id}: {e}")
        
        return {
            'third_down_factor': 1.0,
            'two_minute_factor': 1.0,
            'blowout_factor': 1.0,
            'consistency_score': 0.5,
            'clutch_performance': 1.0
        }
    
    def create_enhanced_player_features(self, player_id: str, games: int = 5) -> Dict[str, float]:
        """Create comprehensive enhanced player features for ML models."""
        
        position = player_id.split('_')[-1].upper()
        
        # Get all metrics
        snap_metrics = self.calculate_snap_percentage(player_id, games)
        red_zone_metrics = self.calculate_red_zone_usage(player_id, games)
        situational_metrics = self.calculate_situational_performance(player_id, games)
        
        features = {
            # Snap and usage features
            'snap_percentage': snap_metrics['snap_percentage'],
            'usage_trend': snap_metrics['usage_trend'],
            'snap_consistency': snap_metrics['snap_consistency'],
            'avg_touches': min(1.0, snap_metrics['avg_touches'] / 20),  # Normalized
            
            # Red zone features
            'red_zone_usage_high': 1.0 if red_zone_metrics['red_zone_usage_level'] == 'high' else 0.0,
            'red_zone_td_rate': red_zone_metrics['red_zone_td_rate'],
            'red_zone_multiplier': red_zone_metrics['red_zone_multiplier'],
            
            # Situational features
            'third_down_factor': situational_metrics['third_down_factor'],
            'two_minute_factor': situational_metrics['two_minute_factor'],
            'consistency_score': situational_metrics['consistency_score'],
            'clutch_performance': situational_metrics['clutch_performance'],
        }
        
        # Position-specific features
        if position in ['WR', 'TE']:
            target_metrics = self.calculate_target_share(player_id, games)
            features.update({
                'target_share': target_metrics['target_share'],
                'air_yards_share': target_metrics['air_yards_share'],
                'red_zone_target_share': target_metrics['red_zone_target_share'],
                'target_quality': target_metrics['target_quality']
            })
        
        return features
    
    def get_usage_comparison(self, player_id: str, comparison_players: List[str]) -> Dict[str, Any]:
        """Compare player usage metrics against similar players."""
        
        all_players = [player_id] + comparison_players
        usage_data = {}
        
        for pid in all_players:
            snap_metrics = self.calculate_snap_percentage(pid, 5)
            red_zone_metrics = self.calculate_red_zone_usage(pid, 5)
            
            usage_data[pid] = {
                'snap_percentage': snap_metrics['snap_percentage'],
                'red_zone_multiplier': red_zone_metrics['red_zone_multiplier'],
                'avg_touches': snap_metrics['avg_touches']
            }
        
        # Calculate percentiles
        player_data = usage_data[player_id]
        comparison_data = [usage_data[pid] for pid in comparison_players]
        
        percentiles = {}
        for metric in ['snap_percentage', 'red_zone_multiplier', 'avg_touches']:
            values = [data[metric] for data in comparison_data]
            if values:
                percentile = (sum(1 for v in values if v < player_data[metric]) / len(values)) * 100
                percentiles[f'{metric}_percentile'] = percentile
        
        return {
            'player_metrics': player_data,
            'percentiles': percentiles,
            'comparison_count': len(comparison_players)
        }

def main():
    """Test the enhanced player metrics system."""
    print("📊 Enhanced Player Metrics Analysis System")
    print("=" * 60)
    
    # Initialize system
    metrics = EnhancedPlayerMetrics()
    
    # Test snap percentage calculation
    print("⚡ Snap Percentage Analysis:")
    snap_data = metrics.calculate_snap_percentage('cmccaffrey_rb', 5)
    print(f"   Snap Percentage: {snap_data['snap_percentage']:.1%}")
    print(f"   Usage Trend: {snap_data['usage_trend']:+.1%}")
    print(f"   Consistency: {snap_data['snap_consistency']:.3f}")
    print(f"   Avg Touches: {snap_data['avg_touches']:.1f}")
    
    # Test target share for WR
    print("\n🎯 Target Share Analysis (WR):")
    target_data = metrics.calculate_target_share('jjefferson_wr', 5)
    print(f"   Target Share: {target_data['target_share']:.1%}")
    print(f"   Air Yards Share: {target_data['air_yards_share']:.1%}")
    print(f"   Red Zone Target Share: {target_data['red_zone_target_share']:.1%}")
    print(f"   Target Quality: {target_data['target_quality']:.3f}")
    
    # Test red zone usage
    print("\n🏈 Red Zone Usage Analysis:")
    rz_data = metrics.calculate_red_zone_usage('cmccaffrey_rb', 5)
    print(f"   Usage Level: {rz_data['red_zone_usage_level']}")
    print(f"   TD Rate: {rz_data['red_zone_td_rate']:.1%}")
    print(f"   Red Zone Multiplier: {rz_data['red_zone_multiplier']:.3f}")
    
    # Test situational performance
    print("\n⏰ Situational Performance Analysis:")
    sit_data = metrics.calculate_situational_performance('tkelce_te', 8)
    print(f"   Third Down Factor: {sit_data['third_down_factor']:.3f}")
    print(f"   Two Minute Factor: {sit_data['two_minute_factor']:.3f}")
    print(f"   Consistency Score: {sit_data['consistency_score']:.3f}")
    print(f"   Clutch Performance: {sit_data['clutch_performance']:.3f}")
    
    # Test comprehensive features
    print("\n🔧 Enhanced Player Features:")
    features = metrics.create_enhanced_player_features('tkelce_te', 5)
    print(f"   Total Features: {len(features)}")
    
    key_features = ['snap_percentage', 'red_zone_multiplier', 'target_share', 'clutch_performance']
    for feature in key_features:
        if feature in features:
            print(f"   {feature}: {features[feature]:.3f}")
    
    print("\n✅ Enhanced player metrics system test complete!")

if __name__ == "__main__":
    main()
