"""
Comprehensive Stats Engine

This module expands statistical coverage to include ALL possible NFL statistics
for complete player analysis and betting insights.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import nfl_data_py as nfl

from simplified_database_models import Player, PlayerGameStats
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

@dataclass
class ComprehensivePlayerStats:
    """Complete statistical profile for a player"""
    
    # Player Identity
    player_id: str
    name: str
    position: str
    team: str
    
    # Basic Passing Stats (for all positions)
    passing_attempts: int = 0
    passing_completions: int = 0
    passing_yards: int = 0
    passing_touchdowns: int = 0
    passing_interceptions: int = 0
    passing_sacks: int = 0
    passing_sack_yards: int = 0
    passing_rating: float = 0.0
    passing_qbr: float = 0.0
    passing_first_downs: int = 0
    passing_completion_percentage: float = 0.0
    passing_yards_per_attempt: float = 0.0
    passing_air_yards: int = 0
    passing_yards_after_catch: int = 0
    
    # Basic Rushing Stats (for all positions)
    rushing_attempts: int = 0
    rushing_yards: int = 0
    rushing_touchdowns: int = 0
    rushing_fumbles: int = 0
    rushing_first_downs: int = 0
    rushing_yards_per_carry: float = 0.0
    rushing_long: int = 0
    rushing_20_plus: int = 0
    rushing_breakaway_runs: int = 0
    
    # Basic Receiving Stats (for all positions)
    targets: int = 0
    receptions: int = 0
    receiving_yards: int = 0
    receiving_touchdowns: int = 0
    receiving_fumbles: int = 0
    receiving_first_downs: int = 0
    receiving_yards_per_target: float = 0.0
    receiving_yards_per_reception: float = 0.0
    receiving_catch_percentage: float = 0.0
    receiving_long: int = 0
    receiving_20_plus: int = 0
    receiving_air_yards: int = 0
    receiving_yards_after_catch: int = 0
    
    # Advanced Receiving Stats
    target_share: float = 0.0
    air_yards_share: float = 0.0
    separation: float = 0.0
    drop_rate: float = 0.0
    contested_catch_rate: float = 0.0
    
    # Red Zone Stats
    red_zone_targets: int = 0
    red_zone_receptions: int = 0
    red_zone_touchdowns: int = 0
    red_zone_rushing_attempts: int = 0
    red_zone_rushing_touchdowns: int = 0
    red_zone_passing_attempts: int = 0
    red_zone_passing_touchdowns: int = 0
    
    # Situational Stats
    third_down_conversions: int = 0
    third_down_attempts: int = 0
    fourth_down_conversions: int = 0
    fourth_down_attempts: int = 0
    two_minute_warning_stats: Dict[str, float] = field(default_factory=dict)
    
    # Snap Count and Usage
    snap_count: int = 0
    snap_percentage: float = 0.0
    routes_run: int = 0
    route_participation: float = 0.0
    
    # Fantasy Stats (all formats)
    fantasy_points_standard: float = 0.0
    fantasy_points_ppr: float = 0.0
    fantasy_points_half_ppr: float = 0.0
    fantasy_points_super_draft: float = 0.0
    
    # Betting-Relevant Stats
    over_under_yards: float = 0.0
    touchdown_scorer_probability: float = 0.0
    first_touchdown_probability: float = 0.0
    anytime_touchdown_probability: float = 0.0
    
    # Performance Quality Metrics
    pff_grade: float = 0.0
    pff_passing_grade: float = 0.0
    pff_rushing_grade: float = 0.0
    pff_receiving_grade: float = 0.0
    
    # Game Context
    home_away_split: str = ""
    opponent: str = ""
    game_script: str = ""  # positive/negative/neutral
    weather_conditions: str = ""
    temperature: Optional[int] = None
    wind_speed: Optional[int] = None
    
    # Trends and Projections
    last_5_games_trend: str = ""  # up/down/stable
    season_projection: Dict[str, float] = field(default_factory=dict)
    rest_days: int = 0
    
    # Confidence and Reliability
    data_completeness: float = 0.0
    prediction_confidence: float = 0.0

class ComprehensiveStatsEngine:
    """Engine for collecting and calculating comprehensive NFL statistics"""
    
    def __init__(self, session: Session):
        self.session = session
        
        # Define position-specific stat priorities
        self.position_stat_priorities = {
            'QB': [
                'passing_attempts', 'passing_completions', 'passing_yards', 'passing_touchdowns',
                'passing_interceptions', 'passing_rating', 'rushing_yards', 'rushing_touchdowns',
                'fantasy_points_standard', 'fantasy_points_ppr'
            ],
            'RB': [
                'rushing_attempts', 'rushing_yards', 'rushing_touchdowns', 'targets', 'receptions',
                'receiving_yards', 'receiving_touchdowns', 'snap_percentage', 'red_zone_touches',
                'fantasy_points_standard', 'fantasy_points_ppr'
            ],
            'WR': [
                'targets', 'receptions', 'receiving_yards', 'receiving_touchdowns', 'target_share',
                'air_yards_share', 'separation', 'red_zone_targets', 'snap_percentage',
                'fantasy_points_standard', 'fantasy_points_ppr'
            ],
            'TE': [
                'targets', 'receptions', 'receiving_yards', 'receiving_touchdowns', 'target_share',
                'red_zone_targets', 'snap_percentage', 'route_participation',
                'fantasy_points_standard', 'fantasy_points_ppr'
            ]
        }
    
    def get_comprehensive_player_stats(self, player_id: str, weeks_back: int = 5) -> ComprehensivePlayerStats:
        """Get comprehensive statistics for a player"""
        
        # Get player info
        player = self.session.query(Player).filter(Player.player_id == player_id).first()
        
        if not player:
            raise ValueError(f"Player {player_id} not found")
        
        # Get recent game stats
        recent_stats = self.session.query(PlayerGameStats).filter(
            PlayerGameStats.player_id == player_id
        ).order_by(PlayerGameStats.stat_id.desc()).limit(weeks_back).all()
        
        if not recent_stats:
            # Return basic stats structure
            return ComprehensivePlayerStats(
                player_id=player_id,
                name=player.name,
                position=player.position,
                team=player.current_team or "FA"
            )
        
        # Calculate comprehensive stats
        stats = self._calculate_comprehensive_stats(player, recent_stats)
        
        # Add advanced calculations
        stats = self._add_advanced_calculations(stats, recent_stats)
        
        # Add position-specific analysis
        stats = self._add_position_specific_analysis(stats, player.position)
        
        # Add betting-relevant calculations
        stats = self._add_betting_calculations(stats)
        
        return stats
    
    def _calculate_comprehensive_stats(self, player: Player, recent_stats: List[PlayerGameStats]) -> ComprehensivePlayerStats:
        """Calculate comprehensive statistics from recent games"""
        
        stats = ComprehensivePlayerStats(
            player_id=player.player_id,
            name=player.name,
            position=player.position,
            team=player.current_team or "FA"
        )
        
        # Calculate totals and averages
        games_played = len(recent_stats)
        
        if games_played == 0:
            return stats
        
        # Passing stats
        stats.passing_attempts = sum(s.passing_attempts for s in recent_stats)
        stats.passing_completions = sum(s.passing_completions for s in recent_stats)
        stats.passing_yards = sum(s.passing_yards for s in recent_stats)
        stats.passing_touchdowns = sum(s.passing_touchdowns for s in recent_stats)
        stats.passing_interceptions = sum(s.passing_interceptions for s in recent_stats)
        stats.passing_sacks = sum(getattr(s, 'passing_sacks', 0) for s in recent_stats)
        stats.passing_first_downs = sum(getattr(s, 'passing_first_downs', 0) for s in recent_stats)
        
        # Rushing stats
        stats.rushing_attempts = sum(s.rushing_attempts for s in recent_stats)
        stats.rushing_yards = sum(s.rushing_yards for s in recent_stats)
        stats.rushing_touchdowns = sum(s.rushing_touchdowns for s in recent_stats)
        stats.rushing_fumbles = sum(getattr(s, 'rushing_fumbles', 0) for s in recent_stats)
        stats.rushing_first_downs = sum(getattr(s, 'rushing_first_downs', 0) for s in recent_stats)
        
        # Receiving stats
        stats.targets = sum(s.targets for s in recent_stats)
        stats.receptions = sum(s.receptions for s in recent_stats)
        stats.receiving_yards = sum(s.receiving_yards for s in recent_stats)
        stats.receiving_touchdowns = sum(s.receiving_touchdowns for s in recent_stats)
        stats.receiving_fumbles = sum(getattr(s, 'receiving_fumbles', 0) for s in recent_stats)
        stats.receiving_first_downs = sum(getattr(s, 'receiving_first_downs', 0) for s in recent_stats)
        
        # Fantasy stats
        stats.fantasy_points_standard = sum(getattr(s, 'fantasy_points_standard', 0) for s in recent_stats)
        stats.fantasy_points_ppr = sum(s.fantasy_points_ppr for s in recent_stats)
        stats.fantasy_points_half_ppr = sum(getattr(s, 'fantasy_points_half_ppr', 0) for s in recent_stats)
        
        # Snap counts
        stats.snap_count = sum(getattr(s, 'snap_count', 0) for s in recent_stats)
        
        # Calculate half-PPR if not available
        if recent_stats and not hasattr(recent_stats[0], 'fantasy_points_half_ppr'):
            stats.fantasy_points_half_ppr = (stats.fantasy_points_standard + stats.fantasy_points_ppr) / 2
        
        return stats
    
    def _add_advanced_calculations(self, stats: ComprehensivePlayerStats, recent_stats: List[PlayerGameStats]) -> ComprehensivePlayerStats:
        """Add advanced calculated statistics"""
        
        games_played = len(recent_stats)
        
        if games_played == 0:
            return stats
        
        # Passing efficiency
        if stats.passing_attempts > 0:
            stats.passing_completion_percentage = (stats.passing_completions / stats.passing_attempts) * 100
            stats.passing_yards_per_attempt = stats.passing_yards / stats.passing_attempts
            
            # Calculate passer rating
            stats.passing_rating = self._calculate_passer_rating(
                stats.passing_attempts, stats.passing_completions, 
                stats.passing_yards, stats.passing_touchdowns, stats.passing_interceptions
            )
        
        # Rushing efficiency
        if stats.rushing_attempts > 0:
            stats.rushing_yards_per_carry = stats.rushing_yards / stats.rushing_attempts
        
        # Receiving efficiency
        if stats.targets > 0:
            stats.receiving_catch_percentage = (stats.receptions / stats.targets) * 100
            stats.receiving_yards_per_target = stats.receiving_yards / stats.targets
        
        if stats.receptions > 0:
            stats.receiving_yards_per_reception = stats.receiving_yards / stats.receptions
        
        # Per-game averages
        stats.fantasy_points_ppr = stats.fantasy_points_ppr / games_played
        stats.fantasy_points_standard = stats.fantasy_points_standard / games_played
        stats.fantasy_points_half_ppr = stats.fantasy_points_half_ppr / games_played
        
        # Calculate data completeness
        total_fields = 25  # Key statistical fields
        completed_fields = 0
        
        if stats.passing_attempts > 0: completed_fields += 5
        if stats.rushing_attempts > 0: completed_fields += 5
        if stats.targets > 0: completed_fields += 5
        if stats.fantasy_points_ppr > 0: completed_fields += 3
        if stats.snap_count > 0: completed_fields += 2
        
        stats.data_completeness = min(1.0, completed_fields / total_fields)
        
        return stats
    
    def _add_position_specific_analysis(self, stats: ComprehensivePlayerStats, position: str) -> ComprehensivePlayerStats:
        """Add position-specific advanced analysis"""
        
        if position == 'QB':
            # QB-specific metrics
            total_touchdowns = stats.passing_touchdowns + stats.rushing_touchdowns
            total_yards = stats.passing_yards + stats.rushing_yards
            
            # Calculate red zone efficiency
            if stats.passing_attempts > 0:
                stats.red_zone_passing_attempts = int(stats.passing_attempts * 0.15)  # Estimate
                stats.red_zone_passing_touchdowns = int(stats.passing_touchdowns * 0.6)  # Estimate
            
        elif position == 'RB':
            # RB-specific metrics
            total_touches = stats.rushing_attempts + stats.receptions
            total_yards = stats.rushing_yards + stats.receiving_yards
            total_touchdowns = stats.rushing_touchdowns + stats.receiving_touchdowns
            
            # Estimate snap percentage
            if total_touches > 0:
                stats.snap_percentage = min(100.0, (total_touches / 5) * 15)  # Rough estimate
            
        elif position in ['WR', 'TE']:
            # WR/TE-specific metrics
            if stats.targets > 0:
                # Estimate target share (would need team data)
                stats.target_share = min(35.0, (stats.targets / 5) * 2.5)  # Rough estimate
                
                # Estimate red zone usage
                stats.red_zone_targets = int(stats.targets * 0.2)  # Estimate
                stats.red_zone_receptions = int(stats.receptions * 0.25)  # Estimate
                stats.red_zone_touchdowns = int(stats.receiving_touchdowns * 0.8)  # Estimate
        
        return stats
    
    def _add_betting_calculations(self, stats: ComprehensivePlayerStats) -> ComprehensivePlayerStats:
        """Add betting-relevant calculations"""
        
        # Calculate touchdown probabilities based on recent performance
        games_played = 5  # Assuming 5 recent games
        
        if games_played > 0:
            total_touchdowns = stats.passing_touchdowns + stats.rushing_touchdowns + stats.receiving_touchdowns
            stats.anytime_touchdown_probability = min(0.95, (total_touchdowns / games_played) * 0.8)
            stats.first_touchdown_probability = stats.anytime_touchdown_probability * 0.15
        
        # Calculate over/under projections
        stats.over_under_yards = stats.passing_yards + stats.rushing_yards + stats.receiving_yards
        
        # Set prediction confidence based on data completeness and consistency
        stats.prediction_confidence = stats.data_completeness * 0.8
        
        return stats
    
    def _calculate_passer_rating(self, attempts: int, completions: int, yards: int, tds: int, ints: int) -> float:
        """Calculate NFL passer rating"""
        
        if attempts == 0:
            return 0.0
        
        try:
            a = max(0, min(2.375, ((completions / attempts) - 0.3) * 5))
            b = max(0, min(2.375, ((yards / attempts) - 3) * 0.25))
            c = max(0, min(2.375, (tds / attempts) * 20))
            d = max(0, min(2.375, 2.375 - ((ints / attempts) * 25)))
            
            rating = ((a + b + c + d) / 6) * 100
            return round(rating, 1)
        except:
            return 0.0
    
    def get_all_position_comprehensive_stats(self, position: str, limit: int = 20) -> List[ComprehensivePlayerStats]:
        """Get comprehensive stats for all players at a position"""
        
        players = self.session.query(Player).filter(
            Player.position == position,
            Player.is_active == True
        ).limit(limit).all()
        
        comprehensive_stats = []
        
        for player in players:
            try:
                stats = self.get_comprehensive_player_stats(player.player_id)
                comprehensive_stats.append(stats)
            except Exception as e:
                logger.warning(f"Error getting stats for {player.name}: {e}")
                continue
        
        # Sort by fantasy points
        comprehensive_stats.sort(key=lambda x: x.fantasy_points_ppr, reverse=True)
        
        return comprehensive_stats

def main():
    """Test comprehensive stats engine"""
    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///nfl_predictions.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    stats_engine = ComprehensiveStatsEngine(session)
    
    # Test with a known player
    try:
        # Get a QB for testing
        qb_player = session.query(Player).filter(
            Player.position == 'QB',
            Player.is_active == True
        ).first()
        
        if qb_player:
            comprehensive_stats = stats_engine.get_comprehensive_player_stats(qb_player.player_id)
            
            print("🏈 COMPREHENSIVE STATS TEST")
            print("=" * 50)
            print(f"Player: {comprehensive_stats.name} ({comprehensive_stats.position})")
            print(f"Team: {comprehensive_stats.team}")
            print()
            
            print("📊 PASSING STATS:")
            print(f"   Attempts: {comprehensive_stats.passing_attempts}")
            print(f"   Completions: {comprehensive_stats.passing_completions}")
            print(f"   Yards: {comprehensive_stats.passing_yards}")
            print(f"   Touchdowns: {comprehensive_stats.passing_touchdowns}")
            print(f"   Interceptions: {comprehensive_stats.passing_interceptions}")
            print(f"   Rating: {comprehensive_stats.passing_rating}")
            print(f"   Completion %: {comprehensive_stats.passing_completion_percentage:.1f}%")
            print(f"   Yards/Attempt: {comprehensive_stats.passing_yards_per_attempt:.1f}")
            print()
            
            print("🏃 RUSHING STATS:")
            print(f"   Attempts: {comprehensive_stats.rushing_attempts}")
            print(f"   Yards: {comprehensive_stats.rushing_yards}")
            print(f"   Touchdowns: {comprehensive_stats.rushing_touchdowns}")
            print(f"   Yards/Carry: {comprehensive_stats.rushing_yards_per_carry:.1f}")
            print()
            
            print("🎯 FANTASY STATS:")
            print(f"   Standard: {comprehensive_stats.fantasy_points_standard:.1f}")
            print(f"   PPR: {comprehensive_stats.fantasy_points_ppr:.1f}")
            print(f"   Half-PPR: {comprehensive_stats.fantasy_points_half_ppr:.1f}")
            print()
            
            print("💰 BETTING STATS:")
            print(f"   Anytime TD Prob: {comprehensive_stats.anytime_touchdown_probability:.1%}")
            print(f"   Total Yards: {comprehensive_stats.over_under_yards}")
            print(f"   Data Completeness: {comprehensive_stats.data_completeness:.1%}")
            print(f"   Prediction Confidence: {comprehensive_stats.prediction_confidence:.1%}")
        else:
            print("❌ No QB players found for testing")
        
    except Exception as e:
        print(f"❌ Error testing comprehensive stats: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
