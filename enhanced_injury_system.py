#!/usr/bin/env python3
"""
Enhanced Injury and Status Information System
Advanced injury tracking with severity analysis, recovery predictions, and impact modeling.
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
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InjuryRecord:
    """Structured injury record."""
    player_name: str
    player_id: str
    injury_type: str
    body_part: str
    severity: str
    status: str
    date_reported: datetime
    expected_return: Optional[datetime]
    games_missed: int
    practice_status: str
    trend: str
    impact_score: float

class EnhancedInjurySystem:
    """Enhanced injury tracking and analysis system."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # Injury severity mapping with recovery times
        self.injury_severity_map = {
            'concussion': {'severity': 'high', 'avg_recovery_days': 14, 'variance': 7},
            'hamstring': {'severity': 'medium', 'avg_recovery_days': 10, 'variance': 5},
            'ankle': {'severity': 'medium', 'avg_recovery_days': 12, 'variance': 6},
            'knee': {'severity': 'high', 'avg_recovery_days': 21, 'variance': 14},
            'shoulder': {'severity': 'medium', 'avg_recovery_days': 8, 'variance': 4},
            'back': {'severity': 'medium', 'avg_recovery_days': 9, 'variance': 5},
            'groin': {'severity': 'low', 'avg_recovery_days': 7, 'variance': 3},
            'quad': {'severity': 'low', 'avg_recovery_days': 6, 'variance': 3},
            'calf': {'severity': 'low', 'avg_recovery_days': 5, 'variance': 2},
            'wrist': {'severity': 'low', 'avg_recovery_days': 4, 'variance': 2},
            'finger': {'severity': 'minimal', 'avg_recovery_days': 3, 'variance': 1},
            'toe': {'severity': 'minimal', 'avg_recovery_days': 2, 'variance': 1}
        }
        
        # Position-specific injury impacts
        self.position_injury_impacts = {
            'QB': {
                'shoulder': 0.7, 'elbow': 0.6, 'wrist': 0.5, 'finger': 0.3,
                'knee': 0.4, 'ankle': 0.3, 'concussion': 0.8, 'back': 0.5
            },
            'RB': {
                'knee': 0.8, 'ankle': 0.7, 'hamstring': 0.8, 'shoulder': 0.4,
                'concussion': 0.9, 'back': 0.6, 'groin': 0.6, 'quad': 0.7
            },
            'WR': {
                'hamstring': 0.8, 'ankle': 0.6, 'knee': 0.7, 'shoulder': 0.5,
                'concussion': 0.9, 'finger': 0.3, 'groin': 0.5, 'quad': 0.6
            },
            'TE': {
                'knee': 0.7, 'ankle': 0.6, 'shoulder': 0.6, 'back': 0.5,
                'concussion': 0.9, 'hamstring': 0.7, 'finger': 0.4, 'wrist': 0.4
            }
        }
        
        # Practice status impact
        self.practice_status_impact = {
            'full': 1.0,
            'limited': 0.85,
            'did_not_participate': 0.3,
            'questionable': 0.7,
            'doubtful': 0.2,
            'out': 0.0
        }
        
        # Initialize injury database
        self._init_injury_database()
        
    def _init_injury_database(self):
        """Initialize injury tracking database."""
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS enhanced_injury_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            player_id TEXT,
            team TEXT,
            position TEXT,
            injury_type TEXT,
            body_part TEXT,
            severity TEXT,
            status TEXT,
            practice_status TEXT,
            date_reported DATE,
            expected_return DATE,
            games_missed INTEGER DEFAULT 0,
            impact_score REAL,
            trend TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()
    
    def fetch_comprehensive_injury_data(self) -> List[InjuryRecord]:
        """Fetch comprehensive injury data from multiple sources."""
        
        all_injuries = []
        
        # Source 1: ESPN Injury Report
        espn_injuries = self._fetch_espn_injuries()
        all_injuries.extend(espn_injuries)
        
        # Source 2: NFL.com Injury Report
        nfl_injuries = self._fetch_nfl_injuries()
        all_injuries.extend(nfl_injuries)
        
        # Source 3: Team Practice Reports
        practice_injuries = self._fetch_practice_reports()
        all_injuries.extend(practice_injuries)
        
        # Deduplicate and merge
        merged_injuries = self._merge_injury_reports(all_injuries)
        
        # Store in database
        self._store_injury_reports(merged_injuries)
        
        return merged_injuries
    
    def _fetch_espn_injuries(self) -> List[InjuryRecord]:
        """Fetch injuries from ESPN (mock implementation)."""
        
        # Mock ESPN injury data
        mock_injuries = [
            {
                'player_name': 'Patrick Mahomes',
                'player_id': 'pmahomes_qb',
                'team': 'KC',
                'position': 'QB',
                'injury_type': 'ankle sprain',
                'body_part': 'ankle',
                'severity': 'questionable',
                'practice_status': 'limited',
                'date_reported': datetime.now() - timedelta(days=2)
            },
            {
                'player_name': 'Christian McCaffrey',
                'player_id': 'cmccaffrey_rb',
                'team': 'SF',
                'position': 'RB',
                'injury_type': 'hamstring strain',
                'body_part': 'hamstring',
                'severity': 'probable',
                'practice_status': 'full',
                'date_reported': datetime.now() - timedelta(days=1)
            },
            {
                'player_name': 'Tyreek Hill',
                'player_id': 'thill_wr',
                'team': 'MIA',
                'position': 'WR',
                'injury_type': 'quad contusion',
                'body_part': 'quad',
                'severity': 'questionable',
                'practice_status': 'limited',
                'date_reported': datetime.now() - timedelta(days=3)
            }
        ]
        
        injuries = []
        for injury_data in mock_injuries:
            injury = self._create_injury_record(injury_data, 'ESPN')
            injuries.append(injury)
        
        return injuries
    
    def _fetch_nfl_injuries(self) -> List[InjuryRecord]:
        """Fetch injuries from NFL.com (mock implementation)."""
        
        # Mock NFL.com data with additional details
        mock_injuries = [
            {
                'player_name': 'Josh Allen',
                'player_id': 'jallen_qb',
                'team': 'BUF',
                'position': 'QB',
                'injury_type': 'shoulder soreness',
                'body_part': 'shoulder',
                'severity': 'probable',
                'practice_status': 'full',
                'date_reported': datetime.now() - timedelta(days=1)
            },
            {
                'player_name': 'Travis Kelce',
                'player_id': 'tkelce_te',
                'team': 'KC',
                'position': 'TE',
                'injury_type': 'knee maintenance',
                'body_part': 'knee',
                'severity': 'probable',
                'practice_status': 'limited',
                'date_reported': datetime.now() - timedelta(days=1)
            }
        ]
        
        injuries = []
        for injury_data in mock_injuries:
            injury = self._create_injury_record(injury_data, 'NFL.com')
            injuries.append(injury)
        
        return injuries
    
    def _fetch_practice_reports(self) -> List[InjuryRecord]:
        """Fetch practice participation reports."""
        
        # Mock practice report data
        mock_practice = [
            {
                'player_name': 'Stefon Diggs',
                'player_id': 'sdiggs_wr',
                'team': 'BUF',
                'position': 'WR',
                'injury_type': 'rest day',
                'body_part': 'general',
                'severity': 'probable',
                'practice_status': 'did_not_participate',
                'date_reported': datetime.now()
            }
        ]
        
        injuries = []
        for injury_data in mock_practice:
            injury = self._create_injury_record(injury_data, 'Practice Report')
            injuries.append(injury)
        
        return injuries
    
    def _create_injury_record(self, injury_data: Dict[str, Any], source: str) -> InjuryRecord:
        """Create structured injury record."""
        
        body_part = injury_data['body_part'].lower()
        position = injury_data['position']
        
        # Calculate severity and impact
        severity_info = self.injury_severity_map.get(body_part, {
            'severity': 'medium', 'avg_recovery_days': 7, 'variance': 3
        })
        
        # Position-specific impact
        position_impacts = self.position_injury_impacts.get(position, {})
        base_impact = position_impacts.get(body_part, 0.5)
        
        # Practice status adjustment
        practice_impact = self.practice_status_impact.get(
            injury_data['practice_status'], 0.5
        )
        
        # Final impact score
        impact_score = 1.0 - (base_impact * (1.0 - practice_impact))
        
        # Estimate return date
        avg_days = severity_info['avg_recovery_days']
        variance = severity_info['variance']
        recovery_days = max(1, int(np.random.normal(avg_days, variance)))
        expected_return = injury_data['date_reported'] + timedelta(days=recovery_days)
        
        # Determine trend
        days_since_report = (datetime.now() - injury_data['date_reported']).days
        if days_since_report <= 1:
            trend = 'new'
        elif practice_impact > 0.8:
            trend = 'improving'
        elif practice_impact < 0.4:
            trend = 'worsening'
        else:
            trend = 'stable'
        
        return InjuryRecord(
            player_name=injury_data['player_name'],
            player_id=injury_data['player_id'],
            injury_type=injury_data['injury_type'],
            body_part=body_part,
            severity=severity_info['severity'],
            status=injury_data['severity'],
            date_reported=injury_data['date_reported'],
            expected_return=expected_return,
            games_missed=0,  # Would be calculated from historical data
            practice_status=injury_data['practice_status'],
            trend=trend,
            impact_score=impact_score
        )
    
    def _merge_injury_reports(self, injuries: List[InjuryRecord]) -> List[InjuryRecord]:
        """Merge duplicate injury reports from different sources."""
        
        # Group by player
        player_injuries = {}
        for injury in injuries:
            key = injury.player_id or injury.player_name
            if key not in player_injuries:
                player_injuries[key] = []
            player_injuries[key].append(injury)
        
        # Merge duplicates
        merged = []
        for player_id, player_injury_list in player_injuries.items():
            if len(player_injury_list) == 1:
                merged.append(player_injury_list[0])
            else:
                # Merge multiple reports for same player
                primary = player_injury_list[0]
                for other in player_injury_list[1:]:
                    # Use most recent report date
                    if other.date_reported > primary.date_reported:
                        primary.date_reported = other.date_reported
                    
                    # Use most restrictive practice status
                    if self.practice_status_impact[other.practice_status] < \
                       self.practice_status_impact[primary.practice_status]:
                        primary.practice_status = other.practice_status
                    
                    # Average impact scores
                    primary.impact_score = (primary.impact_score + other.impact_score) / 2
                
                merged.append(primary)
        
        return merged
    
    def _store_injury_reports(self, injuries: List[InjuryRecord]):
        """Store injury reports in database."""
        
        for injury in injuries:
            insert_sql = """
            INSERT OR REPLACE INTO enhanced_injury_reports 
            (player_name, player_id, injury_type, body_part, severity, status,
             practice_status, date_reported, expected_return, impact_score, trend, source)
            VALUES (:player_name, :player_id, :injury_type, :body_part, :severity, :status,
                    :practice_status, :date_reported, :expected_return, :impact_score, :trend, :source)
            """
            
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), {
                    'player_name': injury.player_name,
                    'player_id': injury.player_id,
                    'injury_type': injury.injury_type,
                    'body_part': injury.body_part,
                    'severity': injury.severity,
                    'status': injury.status,
                    'practice_status': injury.practice_status,
                    'date_reported': injury.date_reported,
                    'expected_return': injury.expected_return,
                    'impact_score': injury.impact_score,
                    'trend': injury.trend,
                    'source': 'Enhanced System'
                })
                conn.commit()
    
    def get_player_injury_analysis(self, player_id: str) -> Dict[str, Any]:
        """Get comprehensive injury analysis for a player."""
        
        query = """
        SELECT * FROM enhanced_injury_reports 
        WHERE player_id = :player_id OR player_name LIKE :player_name
        ORDER BY date_reported DESC
        LIMIT 5
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query), {
                'player_id': player_id, 
                'player_name': f"%{player_id.replace('_', ' ')}%"
            }).fetchall()
        
        if not results:
            return {'status': 'healthy', 'impact_factor': 1.0}
        
        current_injury = results[0]
        
        # Calculate current impact
        days_since_injury = (datetime.now() - datetime.fromisoformat(str(current_injury[10]))).days
        
        # Recovery curve (exponential decay)
        recovery_factor = np.exp(-days_since_injury / 7)  # 7-day half-life
        current_impact = 1.0 - (1.0 - current_injury[13]) * recovery_factor
        
        # Historical injury pattern
        injury_frequency = len(results)
        injury_severity_avg = np.mean([r[13] for r in results])
        
        return {
            'status': current_injury[8] if current_injury[8] != 'probable' else 'active',
            'current_injury': {
                'type': current_injury[5],
                'body_part': current_injury[6],
                'severity': current_injury[7],
                'practice_status': current_injury[9],
                'days_since_report': days_since_injury,
                'trend': current_injury[15]
            },
            'impact_factor': current_impact,
            'injury_history': {
                'frequency': injury_frequency,
                'avg_severity': injury_severity_avg,
                'injury_prone': injury_frequency > 3
            },
            'recommendation': self._get_injury_recommendation(current_impact, injury_frequency)
        }
    
    def _get_injury_recommendation(self, impact_factor: float, frequency: int) -> str:
        """Get betting recommendation based on injury analysis."""
        
        if impact_factor < 0.7:
            return 'AVOID - High injury impact'
        elif impact_factor < 0.85 and frequency > 2:
            return 'CAUTION - Injury concerns'
        elif impact_factor < 0.9:
            return 'MONITOR - Minor impact'
        else:
            return 'CLEAR - No significant concerns'
    
    def get_team_injury_report(self, team: str) -> Dict[str, Any]:
        """Get comprehensive team injury report."""
        
        query = """
        SELECT position, COUNT(*) as injury_count, AVG(impact_score) as avg_impact
        FROM enhanced_injury_reports 
        WHERE team = :team AND date_reported >= date('now', '-7 days')
        GROUP BY position
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query), {'team': team}).fetchall()
        
        position_impacts = {}
        total_injuries = 0
        
        for row in results:
            position = row[0] or 'UNKNOWN'
            count = row[1]
            avg_impact = row[2] or 1.0
            
            position_impacts[position] = {
                'injury_count': count,
                'avg_impact_score': avg_impact,
                'health_rating': 'POOR' if avg_impact < 0.7 else 'FAIR' if avg_impact < 0.85 else 'GOOD'
            }
            total_injuries += count
        
        # Overall team health score
        if position_impacts:
            overall_impact = np.mean([pos['avg_impact_score'] for pos in position_impacts.values()])
            team_health = 'EXCELLENT' if overall_impact > 0.9 else \
                         'GOOD' if overall_impact > 0.8 else \
                         'FAIR' if overall_impact > 0.7 else 'POOR'
        else:
            overall_impact = 1.0
            team_health = 'EXCELLENT'
        
        return {
            'team': team,
            'overall_health': team_health,
            'overall_impact_score': overall_impact,
            'total_injuries': total_injuries,
            'position_breakdown': position_impacts,
            'injury_trend': self._calculate_injury_trend(team)
        }
    
    def _calculate_injury_trend(self, team: str) -> str:
        """Calculate team injury trend over time."""
        
        query = """
        SELECT DATE(date_reported) as report_date, COUNT(*) as daily_injuries
        FROM enhanced_injury_reports 
        WHERE team = :team AND date_reported >= date('now', '-14 days')
        GROUP BY DATE(date_reported)
        ORDER BY report_date
        """
        
        with self.engine.connect() as conn:
            results = conn.execute(text(query), {'team': team}).fetchall()
        
        if len(results) < 2:
            return 'stable'
        
        # Simple trend calculation
        recent_avg = np.mean([r[1] for r in results[-3:]])
        earlier_avg = np.mean([r[1] for r in results[:-3]]) if len(results) > 3 else recent_avg
        
        if recent_avg > earlier_avg * 1.5:
            return 'worsening'
        elif recent_avg < earlier_avg * 0.7:
            return 'improving'
        else:
            return 'stable'
    
    def generate_injury_insights(self) -> Dict[str, Any]:
        """Generate league-wide injury insights."""
        
        # Fetch recent injury data
        injuries = self.fetch_comprehensive_injury_data()
        
        # Position injury analysis
        position_stats = {}
        for injury in injuries:
            pos = injury.player_id.split('_')[-1].upper() if injury.player_id else 'UNKNOWN'
            if pos not in position_stats:
                position_stats[pos] = {'count': 0, 'avg_impact': 0, 'injuries': []}
            
            position_stats[pos]['count'] += 1
            position_stats[pos]['injuries'].append(injury.impact_score)
        
        # Calculate averages
        for pos in position_stats:
            if position_stats[pos]['injuries']:
                position_stats[pos]['avg_impact'] = np.mean(position_stats[pos]['injuries'])
        
        # Most injury-prone body parts
        body_part_freq = {}
        for injury in injuries:
            body_part_freq[injury.body_part] = body_part_freq.get(injury.body_part, 0) + 1
        
        return {
            'total_active_injuries': len(injuries),
            'position_injury_rates': position_stats,
            'most_common_injuries': sorted(body_part_freq.items(), key=lambda x: x[1], reverse=True)[:5],
            'high_impact_injuries': len([i for i in injuries if i.impact_score < 0.7]),
            'injury_severity_distribution': {
                'high': len([i for i in injuries if i.severity == 'high']),
                'medium': len([i for i in injuries if i.severity == 'medium']),
                'low': len([i for i in injuries if i.severity == 'low']),
                'minimal': len([i for i in injuries if i.severity == 'minimal'])
            }
        }

def main():
    """Test the enhanced injury system."""
    print("🏥 ENHANCED INJURY & STATUS SYSTEM")
    print("=" * 60)
    
    # Initialize system
    injury_system = EnhancedInjurySystem()
    
    # Fetch comprehensive injury data
    print("📊 Fetching Comprehensive Injury Data...")
    injuries = injury_system.fetch_comprehensive_injury_data()
    print(f"   Found {len(injuries)} active injury reports")
    
    # Test player analysis
    print("\n🔍 Player Injury Analysis:")
    test_players = ['pmahomes_qb', 'cmccaffrey_rb', 'thill_wr']
    
    for player_id in test_players:
        analysis = injury_system.get_player_injury_analysis(player_id)
        print(f"\n   {player_id}:")
        print(f"     Status: {analysis['status']}")
        print(f"     Impact Factor: {analysis['impact_factor']:.3f}")
        print(f"     Recommendation: {analysis['recommendation']}")
        
        if 'current_injury' in analysis:
            injury = analysis['current_injury']
            print(f"     Current: {injury['type']} ({injury['body_part']}) - {injury['trend']}")
    
    # Test team analysis
    print("\n🏈 Team Injury Reports:")
    test_teams = ['KC', 'SF', 'BUF', 'MIA']
    
    for team in test_teams:
        report = injury_system.get_team_injury_report(team)
        print(f"\n   {team}:")
        print(f"     Health: {report['overall_health']}")
        print(f"     Impact Score: {report['overall_impact_score']:.3f}")
        print(f"     Total Injuries: {report['total_injuries']}")
        print(f"     Trend: {report['injury_trend']}")
    
    # Generate insights
    print("\n📈 League Injury Insights:")
    insights = injury_system.generate_injury_insights()
    
    print(f"   Total Active Injuries: {insights['total_active_injuries']}")
    print(f"   High Impact Injuries: {insights['high_impact_injuries']}")
    
    print("\n   Most Common Injuries:")
    for body_part, count in insights['most_common_injuries']:
        print(f"     {body_part}: {count}")
    
    print("\n   Position Injury Rates:")
    for pos, stats in insights['position_injury_rates'].items():
        print(f"     {pos}: {stats['count']} injuries (avg impact: {stats['avg_impact']:.3f})")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced injury system operational!")

if __name__ == "__main__":
    main()
