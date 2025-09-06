"""
Advanced Situational Analytics for NFL Betting
Red Zone, Third Down, Goal Line, and Game Script Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from sqlalchemy import create_engine, text
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class SituationalStats:
    player_id: str
    red_zone_targets: int
    red_zone_carries: int
    red_zone_touchdowns: int
    third_down_conversions: int
    third_down_attempts: int
    goal_line_carries: int
    goal_line_targets: int
    two_minute_drill_usage: int
    garbage_time_stats: Dict[str, float]

@dataclass
class GameScript:
    game_id: str
    team: str
    expected_game_script: str  # 'positive', 'negative', 'neutral'
    pace_factor: float
    expected_pass_attempts: int
    expected_rush_attempts: int
    blowout_probability: float

class SituationalAnalyzer:
    """Analyze situational football statistics for enhanced predictions."""
    
    def __init__(self, db_path: str = "data/nfl_predictions.db"):
        self.db_path = db_path
        self.situational_cache = {}
        
        # Situational multipliers
        self.situation_multipliers = {
            'red_zone_target_share': 1.5,  # Red zone targets worth more
            'goal_line_carry_share': 2.0,  # Goal line carries very valuable
            'third_down_specialist': 1.3,  # Third down specialists get more targets
            'two_minute_drill': 1.4,      # Two minute drill increases usage
            'garbage_time_boost': 0.7,    # Garbage time stats less predictive
            'blowout_game_script': 1.2    # Positive game script helps
        }
    
    def calculate_red_zone_efficiency(self, player_id: str, position: str) -> Dict[str, float]:
        """Calculate red zone efficiency metrics for a player."""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get red zone opportunities (simulated - would need actual red zone data)
            query = """
                SELECT 
                    COUNT(*) as games,
                    AVG(CASE WHEN fantasy_points_ppr > 15 THEN 1 ELSE 0 END) as high_scoring_rate,
                    AVG(receiving_touchdowns + rushing_touchdowns + passing_touchdowns) as avg_tds
                FROM player_game_stats 
                WHERE player_id = ?
                AND fantasy_points_ppr > 0
            """
            
            result = conn.execute(query, (player_id,)).fetchone()
            
            if not result or result[0] == 0:
                return {}
            
            games, high_scoring_rate, avg_tds = result
            
            # Estimate red zone metrics based on touchdowns and position
            if position == 'RB':
                estimated_rz_carries = avg_tds * 3.5  # RBs get ~3.5 carries per TD
                estimated_rz_efficiency = min(avg_tds / max(estimated_rz_carries, 1), 0.4)
            elif position in ['WR', 'TE']:
                estimated_rz_targets = avg_tds * 4.2  # WRs get ~4.2 targets per TD
                estimated_rz_efficiency = min(avg_tds / max(estimated_rz_targets, 1), 0.3)
            else:  # QB
                estimated_rz_efficiency = min(avg_tds / max(games, 1), 3.0)
            
            return {
                'red_zone_efficiency': estimated_rz_efficiency,
                'red_zone_touchdown_rate': avg_tds,
                'high_scoring_game_rate': high_scoring_rate,
                'red_zone_multiplier': 1.0 + (estimated_rz_efficiency * 0.5)
            }
    
    def calculate_third_down_impact(self, player_id: str, position: str) -> Dict[str, float]:
        """Calculate third down conversion impact for a player."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Estimate third down usage based on receptions and position
            
            # Get player's average targets per game
            cursor.execute("""
                SELECT AVG(pgs.targets) as avg_targets
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                WHERE pgs.player_id = ? AND p.position = ?
            """, (player_id, position))
            
            result = cursor.fetchone()
            avg_targets = result[0] if result and result[0] else 0
            
            # Initialize variables
            third_down_target_share = 0.0
            estimated_3rd_targets = 0.0
            third_down_conversion_rate = 0.0
            
            if position in ['WR', 'TE']:
                # Slot receivers and TEs typically get more third down work
                third_down_target_share = 0.25 if position == 'TE' else 0.20
                estimated_3rd_targets = avg_targets * third_down_target_share
                third_down_conversion_rate = 0.4  # League average ~40%
            elif position == 'RB':
                # RBs get some third down work as checkdown options
                third_down_target_share = 0.15
                estimated_3rd_targets = avg_targets * third_down_target_share
                third_down_conversion_rate = 0.35
            
            return {
                'third_down_target_share': third_down_target_share,
                'estimated_third_down_targets': estimated_3rd_targets,
                'third_down_conversion_rate': third_down_conversion_rate,
                'third_down_multiplier': 1.0 + (third_down_target_share * 0.3)
            }
    
    def analyze_goal_line_usage(self, player_id: str, position: str) -> Dict[str, float]:
        """Analyze goal line usage patterns."""
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    AVG(rushing_touchdowns + receiving_touchdowns) as avg_tds,
                    AVG(rushing_attempts) as avg_carries,
                    AVG(targets) as avg_targets,
                    COUNT(*) as games
                FROM player_game_stats 
                WHERE player_id = ?
            """
            
            result = conn.execute(query, (player_id,)).fetchone()
            
            if not result or result[3] == 0:
                return {}
            
            avg_tds, avg_carries, avg_targets, games = result
            
            # Initialize variables
            goal_line_carry_share = 0.0
            goal_line_target_share = 0.0
            goal_line_multiplier = 1.0
            
            # Estimate goal line usage
            if position == 'RB':
                goal_line_carry_share = min(avg_tds / max(avg_carries or 1, 1) * 5, 0.8)
                goal_line_multiplier = 1.0 + (goal_line_carry_share * 0.4)
            elif position in ['WR', 'TE']:
                goal_line_target_share = min(avg_tds / max(avg_targets or 1, 1) * 8, 0.3)
                goal_line_multiplier = 1.0 + (goal_line_target_share * 0.3)
            elif position == 'QB':
                # QBs can have rushing TDs near goal line
                goal_line_carry_share = min(avg_tds * 0.2, 0.1)  # Conservative estimate
                goal_line_multiplier = 1.0 + (goal_line_carry_share * 0.2)
            
            usage_rate = goal_line_carry_share if position in ['RB', 'QB'] else goal_line_target_share
            
            return {
                'goal_line_usage_rate': usage_rate,
                'goal_line_touchdown_rate': avg_tds or 0.0,
                'goal_line_multiplier': goal_line_multiplier
            }
    
    def calculate_game_script_impact(self, team: str, opponent: str) -> GameScript:
        """Calculate expected game script and its impact."""
        
        # Simplified game script calculation (would use team strength ratings in reality)
        with sqlite3.connect(self.db_path) as conn:
            # Get team performance metrics
            team_query = """
                SELECT 
                    AVG(fantasy_points_ppr) as avg_team_performance
                FROM player_game_stats 
                WHERE player_id LIKE ?
                GROUP BY player_id
                HAVING COUNT(*) > 5
            """
            
            team_result = conn.execute(team_query, (f"%_{team.lower()}",)).fetchall()
            opp_result = conn.execute(team_query, (f"%_{opponent.lower()}",)).fetchall()
            
            team_strength = np.mean([r[0] for r in team_result]) if team_result else 15.0
            opp_strength = np.mean([r[0] for r in opp_result]) if opp_result else 15.0
            
            strength_diff = team_strength - opp_strength
            
            # Determine game script
            if strength_diff > 3:
                expected_script = 'positive'
                pace_factor = 1.1
                pass_rush_ratio = 0.6  # More passing when ahead
                blowout_prob = 0.3
            elif strength_diff < -3:
                expected_script = 'negative'
                pace_factor = 1.2  # Faster pace when behind
                pass_rush_ratio = 0.7  # Much more passing when behind
                blowout_prob = 0.2
            else:
                expected_script = 'neutral'
                pace_factor = 1.0
                pass_rush_ratio = 0.6
                blowout_prob = 0.1
            
            # Estimate play volume
            base_plays = 65
            expected_plays = int(base_plays * pace_factor)
            expected_passes = int(expected_plays * pass_rush_ratio)
            expected_rushes = expected_plays - expected_passes
            
            return GameScript(
                game_id=f"{team}_vs_{opponent}",
                team=team,
                expected_game_script=expected_script,
                pace_factor=pace_factor,
                expected_pass_attempts=expected_passes,
                expected_rush_attempts=expected_rushes,
                blowout_probability=blowout_prob
            )
    
    def get_situational_multiplier(self, player_id: str, position: str, 
                                 game_script: GameScript) -> float:
        """Calculate overall situational multiplier for a player."""
        
        # Get all situational metrics
        rz_metrics = self.calculate_red_zone_efficiency(player_id, position)
        third_down_metrics = self.calculate_third_down_impact(player_id, position)
        goal_line_metrics = self.analyze_goal_line_usage(player_id, position)
        
        # Base multiplier
        multiplier = 1.0
        
        # Red zone impact
        if rz_metrics:
            multiplier *= rz_metrics.get('red_zone_multiplier', 1.0)
        
        # Third down impact
        if third_down_metrics:
            multiplier *= third_down_metrics.get('third_down_multiplier', 1.0)
        
        # Goal line impact
        if goal_line_metrics:
            multiplier *= goal_line_metrics.get('goal_line_multiplier', 1.0)
        
        # Game script impact
        if game_script.expected_game_script == 'positive':
            if position == 'RB':
                multiplier *= 1.15  # RBs benefit from positive game script
            elif position == 'QB':
                multiplier *= 0.95  # QBs throw less when ahead
        elif game_script.expected_game_script == 'negative':
            if position in ['QB', 'WR', 'TE']:
                multiplier *= 1.1   # Passing players benefit from negative script
            elif position == 'RB':
                multiplier *= 0.9   # RBs get less work when behind
        
        # Pace factor impact
        multiplier *= game_script.pace_factor
        
        return min(max(multiplier, 0.7), 1.4)  # Cap between 0.7 and 1.4


class AdvancedPlayerMetrics:
    """Calculate advanced player metrics beyond basic stats."""
    
    def __init__(self, db_path: str = "data/nfl_predictions.db"):
        self.db_path = db_path
    
    def calculate_target_share(self, player_id: str) -> Dict[str, float]:
        """Calculate target share and air yards metrics."""
        
        with sqlite3.connect(self.db_path) as conn:
            # Get player targets and team context
            query = """
                SELECT 
                    targets,
                    receptions,
                    receiving_yards,
                    receiving_touchdowns,
                    fantasy_points_ppr
                FROM player_game_stats 
                WHERE player_id = ?
                AND targets > 0
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            result = conn.execute(query, (player_id,)).fetchall()
            
            if not result:
                return {}
            
            # Calculate metrics
            targets = [r[0] for r in result]
            receptions = [r[1] for r in result]
            rec_yards = [r[2] for r in result]
            rec_tds = [r[3] for r in result]
            
            avg_targets = np.mean(targets)
            catch_rate = np.mean([r/t if t > 0 else 0 for r, t in zip(receptions, targets)])
            yards_per_target = np.mean([y/t if t > 0 else 0 for y, t in zip(rec_yards, targets)])
            yards_per_reception = np.mean([y/r if r > 0 else 0 for y, r in zip(rec_yards, receptions)])
            
            # Estimate air yards (would need actual data)
            estimated_air_yards = yards_per_target * 1.4  # Air yards typically 1.4x yards per target
            
            return {
                'avg_targets_per_game': avg_targets,
                'catch_rate': catch_rate,
                'yards_per_target': yards_per_target,
                'yards_per_reception': yards_per_reception,
                'estimated_air_yards_per_target': estimated_air_yards,
                'target_quality_score': catch_rate * yards_per_target
            }
    
    def calculate_snap_count_impact(self, player_id: str, position: str) -> Dict[str, float]:
        """Estimate snap count impact on production."""
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    fantasy_points_ppr,
                    targets + rushing_attempts as opportunities
                FROM player_game_stats 
                WHERE player_id = ?
                AND fantasy_points_ppr > 0
                ORDER BY created_at DESC
                LIMIT 15
            """
            
            result = conn.execute(query, (player_id,)).fetchall()
            
            if not result:
                return {}
            
            fp_values = [r[0] for r in result]
            opportunities = [r[1] for r in result]
            
            # Estimate snap count based on opportunities
            if position == 'RB':
                estimated_snap_rate = min(np.mean(opportunities) / 25, 1.0)  # 25 touches = full workload
            elif position in ['WR', 'TE']:
                estimated_snap_rate = min(np.mean(opportunities) / 12, 1.0)  # 12 targets = full workload
            else:  # QB
                estimated_snap_rate = 1.0  # QBs play most snaps
            
            points_per_snap = np.mean(fp_values) / max(estimated_snap_rate, 0.1)
            
            return {
                'estimated_snap_rate': estimated_snap_rate,
                'points_per_snap': points_per_snap,
                'opportunity_share': estimated_snap_rate,
                'snap_count_multiplier': 0.5 + (estimated_snap_rate * 0.5)
            }
    
    def calculate_route_running_metrics(self, player_id: str, position: str) -> Dict[str, float]:
        """Calculate route running and usage pattern metrics."""
        
        if position not in ['WR', 'TE']:
            return {}
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    receptions,
                    targets,
                    receiving_yards,
                    receiving_touchdowns
                FROM player_game_stats 
                WHERE player_id = ?
                AND targets > 0
                ORDER BY created_at DESC
                LIMIT 12
            """
            
            result = conn.execute(query, (player_id,)).fetchall()
            
            if not result:
                return {}
            
            receptions = [r[0] for r in result]
            targets = [r[1] for r in result]
            yards = [r[2] for r in result]
            tds = [r[3] for r in result]
            
            # Calculate advanced metrics
            separation_score = np.mean([r/t if t > 0 else 0 for r, t in zip(receptions, targets)])
            big_play_rate = np.mean([1 if y > 20 else 0 for y in yards])
            red_zone_target_rate = np.mean([1 if td > 0 else 0 for td in tds])
            
            # Route diversity score (estimated)
            route_diversity = min(separation_score + big_play_rate, 1.0)
            
            return {
                'separation_score': separation_score,
                'big_play_rate': big_play_rate,
                'red_zone_target_rate': red_zone_target_rate,
                'route_diversity_score': route_diversity,
                'route_running_grade': self._grade_route_running(separation_score, big_play_rate)
            }
    
    def _grade_route_running(self, separation: float, big_play_rate: float) -> str:
        """Grade route running ability."""
        combined_score = (separation * 0.7) + (big_play_rate * 0.3)
        
        if combined_score >= 0.8:
            return 'ELITE'
        elif combined_score >= 0.65:
            return 'GOOD'
        elif combined_score >= 0.5:
            return 'AVERAGE'
        else:
            return 'BELOW_AVERAGE'


class CoachingAnalytics:
    """Analyze coaching tendencies and play-calling patterns."""
    
    def __init__(self, db_path: str = "data/nfl_predictions.db"):
        self.db_path = db_path
        
        # Coaching tendencies by team (simplified)
        self.coaching_tendencies = {
            'aggressive_red_zone': ['kc', 'buf', 'gb'],
            'run_heavy': ['bal', 'cle', 'ten'],
            'pass_heavy': ['mia', 'cin', 'lac'],
            'conservative': ['pit', 'ne', 'nyg'],
            'creative_play_calling': ['sf', 'phi', 'det']
        }
    
    def get_coaching_multiplier(self, player_id: str, position: str) -> float:
        """Get coaching impact multiplier for a player."""
        
        # Extract team from player_id
        team = player_id.split('_')[-1] if '_' in player_id else 'unk'
        
        multiplier = 1.0
        
        # Red zone aggressiveness
        if team in self.coaching_tendencies['aggressive_red_zone']:
            if position in ['WR', 'TE']:
                multiplier *= 1.1  # More red zone targets
            elif position == 'RB':
                multiplier *= 1.05  # More goal line carries
        
        # Run/pass tendencies
        if team in self.coaching_tendencies['run_heavy']:
            if position == 'RB':
                multiplier *= 1.15
            elif position in ['WR', 'TE']:
                multiplier *= 0.95
        elif team in self.coaching_tendencies['pass_heavy']:
            if position in ['QB', 'WR', 'TE']:
                multiplier *= 1.1
            elif position == 'RB':
                multiplier *= 0.9
        
        # Creative play calling
        if team in self.coaching_tendencies['creative_play_calling']:
            multiplier *= 1.05  # Slight boost for all positions
        
        return multiplier
    
    def analyze_situational_coaching(self, team: str, situation: str) -> Dict[str, float]:
        """Analyze coaching decisions in specific situations."""
        
        tendencies = {
            'red_zone_pass_rate': 0.6,  # Default 60% pass in red zone
            'third_down_aggression': 0.7,
            'fourth_down_aggression': 0.3,
            'two_minute_drill_efficiency': 0.75
        }
        
        # Adjust based on team
        if team in self.coaching_tendencies['aggressive_red_zone']:
            tendencies['red_zone_pass_rate'] = 0.7
            tendencies['fourth_down_aggression'] = 0.5
        
        if team in self.coaching_tendencies['conservative']:
            tendencies['red_zone_pass_rate'] = 0.5
            tendencies['fourth_down_aggression'] = 0.2
        
        return tendencies


# Example usage and integration
def main():
    """Example usage of advanced situational analytics."""
    
    print("Advanced Situational Analytics Demo")
    print("=" * 50)
    
    # Initialize analyzers
    situational = SituationalAnalyzer()
    player_metrics = AdvancedPlayerMetrics()
    coaching = CoachingAnalytics()
    
    # Example player analysis
    player_id = "cmccaffrey_rb"
    position = "RB"
    
    print(f"\nAnalyzing {player_id}:")
    
    # Red zone efficiency
    rz_metrics = situational.calculate_red_zone_efficiency(player_id, position)
    if rz_metrics:
        print(f"Red Zone Efficiency: {rz_metrics['red_zone_efficiency']:.3f}")
        print(f"Red Zone Multiplier: {rz_metrics['red_zone_multiplier']:.3f}")
    
    # Target share metrics
    target_metrics = player_metrics.calculate_target_share(player_id)
    if target_metrics:
        print(f"Target Quality Score: {target_metrics.get('target_quality_score', 0):.3f}")
    
    # Snap count impact
    snap_metrics = player_metrics.calculate_snap_count_impact(player_id, position)
    if snap_metrics:
        print(f"Estimated Snap Rate: {snap_metrics['estimated_snap_rate']:.3f}")
    
    # Game script analysis
    game_script = situational.calculate_game_script_impact("sf", "dal")
    print(f"\nGame Script: {game_script.expected_game_script}")
    print(f"Pace Factor: {game_script.pace_factor:.2f}")
    
    # Overall situational multiplier
    overall_multiplier = situational.get_situational_multiplier(player_id, position, game_script)
    print(f"Overall Situational Multiplier: {overall_multiplier:.3f}")
    
    print("\nAdvanced situational analytics ready!")


if __name__ == "__main__":
    main()
