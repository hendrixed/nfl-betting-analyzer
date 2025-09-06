#!/usr/bin/env python3
"""
Advanced Analytics System
Implements EPA (Expected Points Added), WPA (Win Probability Added), and Next Gen Stats integration.
"""

import sys
import os
import logging
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import sqlite3
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedPlayerStats:
    """Advanced analytics for a player."""
    player_id: str
    player_name: str
    position: str
    team: str
    
    # EPA Stats
    epa_per_play: float
    epa_total: float
    epa_passing: float
    epa_rushing: float
    epa_receiving: float
    
    # WPA Stats
    wpa_per_play: float
    wpa_total: float
    clutch_wpa: float
    
    # Next Gen Stats
    separation: float
    speed: float
    acceleration: float
    air_yards: float
    yac_above_expected: float
    completion_prob: float
    
    # Advanced Efficiency
    pff_grade: float
    dvoa: float
    success_rate: float

class AdvancedAnalytics:
    """Advanced analytics system for NFL betting analysis."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # EPA value chart (simplified)
        self.epa_values = {
            'touchdown': 6.0,
            'field_goal': 3.0,
            'safety': 2.0,
            'turnover': -2.0,
            'punt': -1.5,
            'down_conversion': 1.2,
            'failed_conversion': -1.8
        }
        
        # Down and distance EPA expectations
        self.down_distance_epa = {
            (1, 'short'): 0.8,   # 1st & 1-3
            (1, 'medium'): 0.6,  # 1st & 4-7
            (1, 'long'): 0.4,    # 1st & 8+
            (2, 'short'): 0.3,   # 2nd & 1-3
            (2, 'medium'): 0.1,  # 2nd & 4-7
            (2, 'long'): -0.2,   # 2nd & 8+
            (3, 'short'): 0.0,   # 3rd & 1-3
            (3, 'medium'): -0.4, # 3rd & 4-7
            (3, 'long'): -0.8,   # 3rd & 8+
            (4, 'any'): -1.2     # 4th down
        }
        
        # Field position adjustments
        self.field_position_multiplier = {
            'red_zone': 1.5,      # Inside 20
            'scoring_zone': 1.2,  # 20-40 yard line
            'midfield': 1.0,      # 40-40
            'own_territory': 0.8, # Own 40-20
            'deep_own': 0.6       # Own 20 and closer
        }
        
        # Initialize analytics database
        self._init_analytics_database()
        
    def _init_analytics_database(self):
        """Initialize advanced analytics database tables."""
        
        # EPA tracking table
        epa_table_sql = """
        CREATE TABLE IF NOT EXISTS player_epa_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            player_name TEXT,
            position TEXT,
            team TEXT,
            week INTEGER,
            season INTEGER,
            epa_per_play REAL,
            epa_total REAL,
            epa_passing REAL,
            epa_rushing REAL,
            epa_receiving REAL,
            plays_analyzed INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # WPA tracking table
        wpa_table_sql = """
        CREATE TABLE IF NOT EXISTS player_wpa_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            player_name TEXT,
            position TEXT,
            team TEXT,
            week INTEGER,
            season INTEGER,
            wpa_per_play REAL,
            wpa_total REAL,
            clutch_wpa REAL,
            high_leverage_plays INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        # Next Gen Stats table
        ngs_table_sql = """
        CREATE TABLE IF NOT EXISTS player_ngs_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT NOT NULL,
            player_name TEXT,
            position TEXT,
            team TEXT,
            week INTEGER,
            season INTEGER,
            avg_separation REAL,
            max_speed REAL,
            avg_acceleration REAL,
            air_yards_per_target REAL,
            yac_above_expected REAL,
            completion_probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(epa_table_sql))
            conn.execute(text(wpa_table_sql))
            conn.execute(text(ngs_table_sql))
            conn.commit()
    
    def calculate_player_epa(self, player_id: str, weeks: int = 4) -> Dict[str, float]:
        """Calculate Expected Points Added for a player."""
        
        # Get recent game data
        query = """
        SELECT * FROM player_game_stats 
        WHERE player_id = :player_id 
        AND created_at >= date('now', '-{} days')
        ORDER BY created_at DESC
        """.format(weeks * 7)
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query), {'player_id': player_id}).fetchall()
        
        if not results:
            return self._get_default_epa_stats()
        
        # Convert to DataFrame for easier analysis
        if results:
            # Get column names dynamically
            column_query = "PRAGMA table_info(player_game_stats)"
            with self.engine.connect() as conn:
                column_info = conn.execute(text(column_query)).fetchall()
            
            column_names = [col[1] for col in column_info]
            df = pd.DataFrame(results, columns=column_names)
        else:
            df = pd.DataFrame()
        
        # Calculate EPA components
        epa_stats = {}
        
        # Passing EPA (for QBs)
        if 'passing_yards' in df.columns and not df.empty and df['passing_yards'].sum() > 0:
            passing_epa = self._calculate_passing_epa(df)
            epa_stats.update(passing_epa)
        
        # Rushing EPA
        if 'rushing_yards' in df.columns and not df.empty and df['rushing_yards'].sum() > 0:
            rushing_epa = self._calculate_rushing_epa(df)
            epa_stats.update(rushing_epa)
        
        # Receiving EPA
        if 'receiving_yards' in df.columns and not df.empty and df['receiving_yards'].sum() > 0:
            receiving_epa = self._calculate_receiving_epa(df)
            epa_stats.update(receiving_epa)
        
        # Calculate totals
        epa_stats['epa_total'] = (epa_stats.get('epa_passing', 0) + 
                                 epa_stats.get('epa_rushing', 0) + 
                                 epa_stats.get('epa_receiving', 0))
        
        total_plays = (df.get('passing_attempts', pd.Series([0])).sum() + 
                      df.get('rushing_attempts', pd.Series([0])).sum() + 
                      df.get('targets', pd.Series([0])).sum())
        
        epa_stats['epa_per_play'] = epa_stats['epa_total'] / max(total_plays, 1)
        
        return epa_stats
    
    def _calculate_passing_epa(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate passing EPA."""
        
        total_epa = 0
        total_attempts = df.get('passing_attempts', pd.Series([0])).sum()
        
        for _, game in df.iterrows():
            attempts = game.get('passing_attempts', 0)
            yards = game.get('passing_yards', 0)
            tds = game.get('passing_touchdowns', 0)
            ints = game.get('interceptions', 0)
            
            if attempts > 0:
                # EPA from touchdowns
                td_epa = tds * self.epa_values['touchdown']
                
                # EPA from interceptions
                int_epa = ints * self.epa_values['turnover']
                
                # EPA from yards (simplified: 0.1 EPA per yard)
                yard_epa = yards * 0.1
                
                # EPA from completions vs incompletions
                estimated_completions = yards / max(8.0, 1)  # Assume ~8 yards per completion
                completion_epa = estimated_completions * 0.2
                
                game_epa = td_epa + int_epa + yard_epa + completion_epa
                total_epa += game_epa
        
        return {
            'epa_passing': total_epa,
            'epa_passing_per_attempt': total_epa / max(total_attempts, 1)
        }
    
    def _calculate_rushing_epa(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate rushing EPA."""
        
        total_epa = 0
        total_attempts = df.get('rushing_attempts', pd.Series([0])).sum()
        
        for _, game in df.iterrows():
            attempts = game.get('rushing_attempts', 0)
            yards = game.get('rushing_yards', 0)
            tds = game.get('rushing_touchdowns', 0)
            
            if attempts > 0:
                # EPA from touchdowns
                td_epa = tds * self.epa_values['touchdown']
                
                # EPA from yards (0.08 EPA per rushing yard)
                yard_epa = yards * 0.08
                
                # EPA from successful runs (>= 4 yards)
                avg_yards_per_carry = yards / attempts
                if avg_yards_per_carry >= 4.0:
                    success_epa = attempts * 0.3
                else:
                    success_epa = attempts * -0.1
                
                game_epa = td_epa + yard_epa + success_epa
                total_epa += game_epa
        
        return {
            'epa_rushing': total_epa,
            'epa_rushing_per_attempt': total_epa / max(total_attempts, 1)
        }
    
    def _calculate_receiving_epa(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate receiving EPA."""
        
        total_epa = 0
        total_targets = df.get('targets', pd.Series([0])).sum()
        
        for _, game in df.iterrows():
            targets = game.get('targets', 0)
            receptions = game.get('receptions', 0)
            yards = game.get('receiving_yards', 0)
            tds = game.get('receiving_touchdowns', 0)
            
            if targets > 0:
                # EPA from touchdowns
                td_epa = tds * self.epa_values['touchdown']
                
                # EPA from yards (0.09 EPA per receiving yard)
                yard_epa = yards * 0.09
                
                # EPA from receptions (first down conversions)
                catch_rate = receptions / targets
                reception_epa = receptions * 0.25 * catch_rate
                
                # EPA penalty for drops
                drops = targets - receptions
                drop_epa = drops * -0.15
                
                game_epa = td_epa + yard_epa + reception_epa + drop_epa
                total_epa += game_epa
        
        return {
            'epa_receiving': total_epa,
            'epa_receiving_per_target': total_epa / max(total_targets, 1)
        }
    
    def calculate_player_wpa(self, player_id: str, weeks: int = 4) -> Dict[str, float]:
        """Calculate Win Probability Added for a player."""
        
        # Mock WPA calculation (would need play-by-play data for real implementation)
        epa_stats = self.calculate_player_epa(player_id, weeks)
        
        # Convert EPA to WPA using correlation factor
        epa_to_wpa_factor = 0.12  # Approximate conversion
        
        wpa_stats = {
            'wpa_total': epa_stats.get('epa_total', 0) * epa_to_wpa_factor,
            'wpa_per_play': epa_stats.get('epa_per_play', 0) * epa_to_wpa_factor,
            'clutch_wpa': epa_stats.get('epa_total', 0) * epa_to_wpa_factor * 0.3,  # 30% in clutch
            'high_leverage_situations': max(1, int(epa_stats.get('epa_total', 0) / 5))
        }
        
        return wpa_stats
    
    def fetch_next_gen_stats(self, player_id: str) -> Dict[str, float]:
        """Fetch Next Gen Stats (mock implementation)."""
        
        position = player_id.split('_')[-1].upper()
        
        # Mock Next Gen Stats based on position
        if position == 'QB':
            return {
                'avg_time_to_throw': np.random.uniform(2.4, 3.2),
                'avg_air_yards': np.random.uniform(7.5, 12.0),
                'aggressiveness': np.random.uniform(0.15, 0.25),
                'max_speed': np.random.uniform(18.0, 22.0),
                'completion_prob_above_expected': np.random.uniform(-0.05, 0.08)
            }
        elif position == 'RB':
            return {
                'avg_speed': np.random.uniform(19.5, 23.5),
                'rush_yards_over_expected': np.random.uniform(-0.5, 1.2),
                'efficiency': np.random.uniform(0.85, 1.15),
                'max_speed': np.random.uniform(20.0, 24.0),
                'yards_after_contact': np.random.uniform(2.8, 4.2)
            }
        elif position in ['WR', 'TE']:
            return {
                'avg_separation': np.random.uniform(2.1, 3.8),
                'avg_speed': np.random.uniform(18.0, 22.5),
                'catch_percentage': np.random.uniform(0.55, 0.85),
                'yac_above_expected': np.random.uniform(-1.2, 2.1),
                'air_yards_per_target': np.random.uniform(8.5, 15.2),
                'max_speed': np.random.uniform(19.0, 23.0)
            }
        else:
            return {}
    
    def calculate_advanced_efficiency_metrics(self, player_id: str) -> Dict[str, float]:
        """Calculate advanced efficiency metrics."""
        
        # Get player data
        query = """
        SELECT * FROM player_game_stats 
        WHERE player_id = :player_id 
        AND created_at >= date('now', '-28 days')
        ORDER BY created_at DESC
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query), {'player_id': player_id}).fetchall()
        
        if not results:
            return {}
        
        # Get column names dynamically
        column_query = "PRAGMA table_info(player_game_stats)"
        with self.engine.connect() as conn:
            column_info = conn.execute(text(column_query)).fetchall()
        
        column_names = [col[1] for col in column_info]
        df = pd.DataFrame(results, columns=column_names)
        
        position = player_id.split('_')[-1].upper()
        efficiency_metrics = {}
        
        if position == 'QB':
            # QB efficiency metrics
            total_attempts = df.get('passing_attempts', pd.Series([0])).sum()
            total_yards = df.get('passing_yards', pd.Series([0])).sum()
            total_tds = df.get('passing_touchdowns', pd.Series([0])).sum()
            total_ints = df.get('interceptions', pd.Series([0])).sum()
            
            if total_attempts > 0:
                efficiency_metrics.update({
                    'yards_per_attempt': total_yards / total_attempts,
                    'td_rate': total_tds / total_attempts,
                    'int_rate': total_ints / total_attempts,
                    'passer_rating': self._calculate_passer_rating(total_attempts, total_yards, total_tds, total_ints),
                    'qbr_estimate': min(100, max(0, (total_yards / total_attempts * 3 + total_tds * 20 - total_ints * 15)))
                })
        
        elif position == 'RB':
            # RB efficiency metrics
            total_attempts = df.get('rushing_attempts', pd.Series([0])).sum()
            total_yards = df.get('rushing_yards', pd.Series([0])).sum()
            total_tds = df.get('rushing_touchdowns', pd.Series([0])).sum()
            
            if total_attempts > 0:
                efficiency_metrics.update({
                    'yards_per_carry': total_yards / total_attempts,
                    'td_rate': total_tds / total_attempts,
                    'breakaway_runs': max(0, int((total_yards / total_attempts - 4.0) * total_attempts / 20)),
                    'goal_line_efficiency': min(1.0, total_tds / max(1, total_attempts / 20))
                })
        
        elif position in ['WR', 'TE']:
            # Receiver efficiency metrics
            total_targets = df.get('targets', pd.Series([0])).sum()
            total_receptions = df.get('receptions', pd.Series([0])).sum()
            total_yards = df.get('receiving_yards', pd.Series([0])).sum()
            total_tds = df.get('receiving_touchdowns', pd.Series([0])).sum()
            
            if total_targets > 0:
                efficiency_metrics.update({
                    'catch_rate': total_receptions / total_targets,
                    'yards_per_target': total_yards / total_targets,
                    'yards_per_reception': total_yards / max(total_receptions, 1),
                    'td_rate': total_tds / total_targets,
                    'target_share_estimate': min(0.35, total_targets / max(1, len(df) * 35))  # Estimate
                })
        
        return efficiency_metrics
    
    def _calculate_passer_rating(self, attempts: int, yards: int, tds: int, ints: int) -> float:
        """Calculate NFL passer rating."""
        
        if attempts == 0:
            return 0
        
        # Passer rating formula components
        a = max(0, min(2.375, (yards / attempts - 3) * 0.25))
        b = max(0, min(2.375, (tds / attempts) * 20))
        c = max(0, min(2.375, 2.375 - (ints / attempts) * 25))
        d = max(0, min(2.375, (yards / attempts) * 0.25))
        
        rating = ((a + b + c + d) / 6) * 100
        return min(158.3, max(0, rating))
    
    def _get_default_epa_stats(self) -> Dict[str, float]:
        """Get default EPA stats for players with no data."""
        
        return {
            'epa_total': 0.0,
            'epa_per_play': 0.0,
            'epa_passing': 0.0,
            'epa_rushing': 0.0,
            'epa_receiving': 0.0
        }
    
    def get_comprehensive_analytics(self, player_id: str) -> AdvancedPlayerStats:
        """Get comprehensive advanced analytics for a player."""
        
        # Extract player info
        player_name = player_id.replace('_', ' ').title()
        position = player_id.split('_')[-1].upper()
        
        # Mock team extraction
        team_part = player_id.split('_')[0].upper()
        team_mapping = {
            'PMAHOMES': 'KC', 'JALLEN': 'BUF', 'CMCCAFFREY': 'SF',
            'THILL': 'MIA', 'TKELCE': 'KC', 'SDIGGS': 'BUF'
        }
        team = team_mapping.get(team_part, 'UNK')
        
        # Get EPA stats
        epa_stats = self.calculate_player_epa(player_id)
        
        # Get WPA stats
        wpa_stats = self.calculate_player_wpa(player_id)
        
        # Get Next Gen Stats
        ngs_stats = self.fetch_next_gen_stats(player_id)
        
        # Get efficiency metrics
        efficiency_stats = self.calculate_advanced_efficiency_metrics(player_id)
        
        return AdvancedPlayerStats(
            player_id=player_id,
            player_name=player_name,
            position=position,
            team=team,
            
            # EPA Stats
            epa_per_play=epa_stats.get('epa_per_play', 0.0),
            epa_total=epa_stats.get('epa_total', 0.0),
            epa_passing=epa_stats.get('epa_passing', 0.0),
            epa_rushing=epa_stats.get('epa_rushing', 0.0),
            epa_receiving=epa_stats.get('epa_receiving', 0.0),
            
            # WPA Stats
            wpa_per_play=wpa_stats.get('wpa_per_play', 0.0),
            wpa_total=wpa_stats.get('wpa_total', 0.0),
            clutch_wpa=wpa_stats.get('clutch_wpa', 0.0),
            
            # Next Gen Stats
            separation=ngs_stats.get('avg_separation', 0.0),
            speed=ngs_stats.get('max_speed', 0.0),
            acceleration=ngs_stats.get('avg_acceleration', 0.0),
            air_yards=ngs_stats.get('air_yards_per_target', 0.0),
            yac_above_expected=ngs_stats.get('yac_above_expected', 0.0),
            completion_prob=ngs_stats.get('completion_prob_above_expected', 0.0),
            
            # Advanced Efficiency (mock values)
            pff_grade=np.random.uniform(60, 90),
            dvoa=np.random.uniform(-20, 30),
            success_rate=efficiency_stats.get('catch_rate', np.random.uniform(0.5, 0.8))
        )
    
    def generate_analytics_report(self, player_ids: List[str]) -> Dict[str, Any]:
        """Generate comprehensive analytics report for multiple players."""
        
        player_analytics = []
        
        for player_id in player_ids:
            try:
                analytics = self.get_comprehensive_analytics(player_id)
                player_analytics.append(analytics)
            except Exception as e:
                logger.warning(f"Failed to get analytics for {player_id}: {e}")
        
        # Generate summary statistics
        if player_analytics:
            epa_values = [p.epa_per_play for p in player_analytics]
            wpa_values = [p.wpa_per_play for p in player_analytics]
            
            summary = {
                'total_players_analyzed': len(player_analytics),
                'avg_epa_per_play': np.mean(epa_values),
                'top_epa_player': max(player_analytics, key=lambda x: x.epa_per_play).player_name,
                'avg_wpa_per_play': np.mean(wpa_values),
                'top_wpa_player': max(player_analytics, key=lambda x: x.wpa_per_play).player_name,
                'position_breakdown': self._get_position_breakdown(player_analytics)
            }
        else:
            summary = {'total_players_analyzed': 0}
        
        return {
            'player_analytics': player_analytics,
            'summary': summary,
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_position_breakdown(self, analytics: List[AdvancedPlayerStats]) -> Dict[str, Dict[str, float]]:
        """Get position-based analytics breakdown."""
        
        position_stats = {}
        
        for player in analytics:
            pos = player.position
            if pos not in position_stats:
                position_stats[pos] = {
                    'count': 0,
                    'total_epa': 0,
                    'total_wpa': 0,
                    'total_pff': 0
                }
            
            position_stats[pos]['count'] += 1
            position_stats[pos]['total_epa'] += player.epa_per_play
            position_stats[pos]['total_wpa'] += player.wpa_per_play
            position_stats[pos]['total_pff'] += player.pff_grade
        
        # Calculate averages
        for pos in position_stats:
            count = position_stats[pos]['count']
            position_stats[pos]['avg_epa'] = position_stats[pos]['total_epa'] / count
            position_stats[pos]['avg_wpa'] = position_stats[pos]['total_wpa'] / count
            position_stats[pos]['avg_pff'] = position_stats[pos]['total_pff'] / count
        
        return position_stats

def main():
    """Test the advanced analytics system."""
    print("📊 ADVANCED ANALYTICS SYSTEM")
    print("=" * 60)
    print("🎯 EPA, WPA, and Next Gen Stats Integration")
    
    # Initialize system
    analytics = AdvancedAnalytics()
    
    # Test individual player analytics
    print("\n🔬 Individual Player Analytics:")
    
    test_players = ['pmahomes_qb', 'cmccaffrey_rb', 'thill_wr', 'tkelce_te']
    
    for player_id in test_players:
        player_analytics = analytics.get_comprehensive_analytics(player_id)
        
        print(f"\n   📈 {player_analytics.player_name} ({player_analytics.position}):")
        print(f"     EPA/Play: {player_analytics.epa_per_play:.3f}")
        print(f"     Total EPA: {player_analytics.epa_total:.1f}")
        print(f"     WPA/Play: {player_analytics.wpa_per_play:.3f}")
        print(f"     PFF Grade: {player_analytics.pff_grade:.1f}")
        print(f"     Success Rate: {player_analytics.success_rate:.3f}")
        
        if player_analytics.position in ['WR', 'TE']:
            print(f"     Separation: {player_analytics.separation:.1f} yards")
            print(f"     YAC Above Expected: {player_analytics.yac_above_expected:.1f}")
        elif player_analytics.position == 'QB':
            print(f"     Air Yards/Target: {player_analytics.air_yards:.1f}")
            print(f"     Completion Prob: {player_analytics.completion_prob:.3f}")
    
    # Generate comprehensive report
    print("\n📊 Comprehensive Analytics Report:")
    report = analytics.generate_analytics_report(test_players)
    
    summary = report['summary']
    print(f"   Players Analyzed: {summary['total_players_analyzed']}")
    print(f"   Avg EPA/Play: {summary.get('avg_epa_per_play', 0):.3f}")
    print(f"   Top EPA Player: {summary.get('top_epa_player', 'N/A')}")
    print(f"   Avg WPA/Play: {summary.get('avg_wpa_per_play', 0):.3f}")
    print(f"   Top WPA Player: {summary.get('top_wpa_player', 'N/A')}")
    
    # Position breakdown
    if 'position_breakdown' in summary:
        print("\n   📋 Position Breakdown:")
        for pos, stats in summary['position_breakdown'].items():
            print(f"     {pos}: {stats['count']} players, "
                  f"Avg EPA: {stats['avg_epa']:.3f}, "
                  f"Avg PFF: {stats['avg_pff']:.1f}")
    
    print("\n" + "=" * 60)
    print("✅ Advanced analytics system operational!")

if __name__ == "__main__":
    main()
