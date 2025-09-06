#!/usr/bin/env python3
"""
NFL Injury Data Integration System
Real-time injury reports, player status, and impact analysis for betting predictions.
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

class InjuryDataIntegrator:
    """Comprehensive injury data integration and analysis system."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # API endpoints for injury data
        self.injury_sources = {
            'espn': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams',
            'nfl_api': 'https://api.sleeper.app/v1/players/nfl',
            'fantasy_pros': 'https://www.fantasypros.com/nfl/reports/leaders/injury-report.php'
        }
        
        # Injury severity mapping
        self.injury_severity = {
            'OUT': 0.0,           # Player will not play
            'DOUBTFUL': 0.25,     # Unlikely to play
            'QUESTIONABLE': 0.65, # May or may not play
            'PROBABLE': 0.85,     # Likely to play
            'HEALTHY': 1.0,       # Full participation
            'IR': 0.0,            # Injured Reserve
            'PUP': 0.0,           # Physically Unable to Perform
            'SUSPENDED': 0.0,     # Suspended
            'COVID': 0.0          # COVID-19 list
        }
        
        # Position impact weights
        self.position_impact = {
            'QB': 0.8,   # High impact
            'RB': 0.7,   # High impact
            'WR': 0.6,   # Medium-high impact
            'TE': 0.5,   # Medium impact
            'K': 0.3,    # Low impact
            'DEF': 0.4   # Medium impact
        }
        
        self.injury_cache = {}
        self.cache_expiry = timedelta(hours=2)  # Cache for 2 hours
        
    def fetch_espn_injury_data(self) -> Dict[str, Any]:
        """Fetch injury data from ESPN API."""
        try:
            response = requests.get(self.injury_sources['espn'], timeout=10)
            if response.status_code == 200:
                data = response.json()
                injury_data = {}
                
                for team in data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
                    team_name = team.get('team', {}).get('abbreviation', '')
                    
                    # Get team roster with injury status
                    roster_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team.get('team', {}).get('id', '')}/roster"
                    roster_response = requests.get(roster_url, timeout=10)
                    
                    if roster_response.status_code == 200:
                        roster_data = roster_response.json()
                        
                        for athlete in roster_data.get('athletes', []):
                            for player in athlete.get('items', []):
                                player_name = player.get('fullName', '')
                                position = player.get('position', {}).get('abbreviation', '')
                                
                                # Check for injury status
                                injury_status = 'HEALTHY'
                                if player.get('injuries'):
                                    injury_status = player['injuries'][0].get('status', 'QUESTIONABLE')
                                
                                injury_data[player_name] = {
                                    'team': team_name,
                                    'position': position,
                                    'status': injury_status,
                                    'availability': self.injury_severity.get(injury_status, 0.5),
                                    'source': 'ESPN',
                                    'updated': datetime.now()
                                }
                
                return injury_data
                
        except Exception as e:
            logger.warning(f"Failed to fetch ESPN injury data: {e}")
            return {}
    
    def fetch_sleeper_injury_data(self) -> Dict[str, Any]:
        """Fetch injury data from Sleeper API."""
        try:
            response = requests.get(self.injury_sources['nfl_api'], timeout=10)
            if response.status_code == 200:
                players = response.json()
                injury_data = {}
                
                for player_id, player_info in players.items():
                    if player_info.get('sport') == 'nfl' and player_info.get('active'):
                        player_name = f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}"
                        position = player_info.get('position', '')
                        team = player_info.get('team', '')
                        
                        # Check injury status
                        injury_status = player_info.get('injury_status', 'HEALTHY')
                        if not injury_status:
                            injury_status = 'HEALTHY'
                        
                        injury_data[player_name] = {
                            'team': team,
                            'position': position,
                            'status': injury_status,
                            'availability': self.injury_severity.get(injury_status, 1.0),
                            'source': 'Sleeper',
                            'updated': datetime.now()
                        }
                
                return injury_data
                
        except Exception as e:
            logger.warning(f"Failed to fetch Sleeper injury data: {e}")
            return {}
    
    def get_cached_injury_data(self) -> Dict[str, Any]:
        """Get cached injury data if still valid."""
        if not self.injury_cache:
            return {}
        
        cache_time = self.injury_cache.get('timestamp', datetime.min)
        if datetime.now() - cache_time > self.cache_expiry:
            self.injury_cache = {}
            return {}
        
        return self.injury_cache.get('data', {})
    
    def update_injury_data(self) -> Dict[str, Any]:
        """Update injury data from all sources."""
        logger.info("Updating injury data from multiple sources...")
        
        # Check cache first
        cached_data = self.get_cached_injury_data()
        if cached_data:
            logger.info("Using cached injury data")
            return cached_data
        
        # Fetch from multiple sources
        all_injury_data = {}
        
        # ESPN data
        espn_data = self.fetch_espn_injury_data()
        all_injury_data.update(espn_data)
        logger.info(f"Fetched {len(espn_data)} players from ESPN")
        
        # Sleeper data
        sleeper_data = self.fetch_sleeper_injury_data()
        
        # Merge data, prioritizing more recent/reliable sources
        for player_name, player_data in sleeper_data.items():
            if player_name in all_injury_data:
                # Keep the more severe injury status
                existing_availability = all_injury_data[player_name]['availability']
                new_availability = player_data['availability']
                
                if new_availability < existing_availability:
                    all_injury_data[player_name] = player_data
            else:
                all_injury_data[player_name] = player_data
        
        logger.info(f"Total injury data: {len(all_injury_data)} players")
        
        # Cache the data
        self.injury_cache = {
            'data': all_injury_data,
            'timestamp': datetime.now()
        }
        
        return all_injury_data
    
    def calculate_injury_impact(self, player_name: str, position: str) -> float:
        """Calculate injury impact factor for a player."""
        injury_data = self.update_injury_data()
        
        if player_name not in injury_data:
            return 1.0  # No injury data, assume healthy
        
        player_injury = injury_data[player_name]
        availability = player_injury['availability']
        position_weight = self.position_impact.get(position, 0.5)
        
        # Calculate impact: (availability * position_importance)
        impact_factor = availability * position_weight + (1 - position_weight)
        
        return max(0.1, min(1.0, impact_factor))  # Clamp between 0.1 and 1.0
    
    def get_team_injury_report(self, team_abbr: str) -> Dict[str, Any]:
        """Get comprehensive injury report for a team."""
        injury_data = self.update_injury_data()
        
        team_injuries = {}
        for player_name, player_info in injury_data.items():
            if player_info.get('team') and player_info['team'].upper() == team_abbr.upper():
                team_injuries[player_name] = player_info
        
        # Calculate team health score
        if not team_injuries:
            health_score = 1.0
        else:
            total_impact = 0
            total_weight = 0
            
            for player_name, player_info in team_injuries.items():
                position = player_info['position']
                availability = player_info['availability']
                weight = self.position_impact.get(position, 0.5)
                
                total_impact += availability * weight
                total_weight += weight
            
            health_score = total_impact / total_weight if total_weight > 0 else 1.0
        
        return {
            'team': team_abbr,
            'health_score': health_score,
            'injured_players': team_injuries,
            'injury_count': len(team_injuries),
            'key_injuries': [
                player for player, info in team_injuries.items()
                if info['availability'] < 0.5 and self.position_impact.get(info['position'], 0) > 0.6
            ]
        }
    
    def adjust_predictions_for_injuries(self, predictions: Dict[str, float], 
                                     player_name: str, position: str) -> Dict[str, float]:
        """Adjust predictions based on injury status."""
        injury_impact = self.calculate_injury_impact(player_name, position)
        
        adjusted_predictions = {}
        for stat_name, prediction in predictions.items():
            # Apply injury impact to predictions
            if injury_impact < 1.0:
                # Reduce predictions for injured players
                adjustment_factor = 0.5 + (injury_impact * 0.5)  # Scale between 0.5 and 1.0
                adjusted_predictions[stat_name] = prediction * adjustment_factor
            else:
                adjusted_predictions[stat_name] = prediction
        
        return adjusted_predictions
    
    def get_injury_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze injury trends over time."""
        # This would require historical injury data storage
        # For now, return current snapshot analysis
        injury_data = self.update_injury_data()
        
        position_injuries = {}
        team_injuries = {}
        
        for player_name, player_info in injury_data.items():
            position = player_info['position']
            team = player_info['team']
            status = player_info['status']
            
            if position not in position_injuries:
                position_injuries[position] = {'total': 0, 'severe': 0}
            if team not in team_injuries:
                team_injuries[team] = {'total': 0, 'severe': 0}
            
            position_injuries[position]['total'] += 1
            team_injuries[team]['total'] += 1
            
            if player_info['availability'] < 0.5:
                position_injuries[position]['severe'] += 1
                team_injuries[team]['severe'] += 1
        
        return {
            'position_analysis': position_injuries,
            'team_analysis': team_injuries,
            'total_injured': len(injury_data),
            'severe_injuries': sum(1 for p in injury_data.values() if p['availability'] < 0.5)
        }
    
    def create_injury_features(self, player_name: str, team: str) -> Dict[str, float]:
        """Create injury-based features for ML models."""
        injury_data = self.update_injury_data()
        team_report = self.get_team_injury_report(team)
        
        features = {
            'player_injury_status': 1.0,  # Default healthy
            'team_health_score': team_report['health_score'],
            'team_injury_count': team_report['injury_count'],
            'has_key_injuries': 1.0 if team_report['key_injuries'] else 0.0
        }
        
        # Player-specific injury status
        if player_name in injury_data:
            player_info = injury_data[player_name]
            features['player_injury_status'] = player_info['availability']
        
        return features

def main():
    """Test the injury data integration system."""
    print("🏥 NFL Injury Data Integration System")
    print("=" * 50)
    
    # Initialize system
    integrator = InjuryDataIntegrator()
    
    # Update injury data
    print("📊 Fetching injury data...")
    injury_data = integrator.update_injury_data()
    print(f"   ✅ Found injury data for {len(injury_data)} players")
    
    # Test team injury report
    print("\n🏈 Team Injury Analysis (Sample: KC)...")
    kc_report = integrator.get_team_injury_report('KC')
    print(f"   Team Health Score: {kc_report['health_score']:.3f}")
    print(f"   Injured Players: {kc_report['injury_count']}")
    print(f"   Key Injuries: {len(kc_report['key_injuries'])}")
    
    # Test injury trends
    print("\n📈 Injury Trends Analysis...")
    trends = integrator.get_injury_trends()
    print(f"   Total Injured: {trends['total_injured']}")
    print(f"   Severe Injuries: {trends['severe_injuries']}")
    
    # Test prediction adjustment
    print("\n🎯 Testing Prediction Adjustments...")
    sample_predictions = {
        'passing_yards': 275.0,
        'passing_touchdowns': 2.1,
        'rushing_yards': 45.0
    }
    
    adjusted = integrator.adjust_predictions_for_injuries(
        sample_predictions, "Patrick Mahomes", "QB"
    )
    
    print("   Original vs Adjusted Predictions:")
    for stat, orig_val in sample_predictions.items():
        adj_val = adjusted[stat]
        change = ((adj_val - orig_val) / orig_val) * 100
        print(f"   {stat}: {orig_val:.1f} → {adj_val:.1f} ({change:+.1f}%)")
    
    print("\n✅ Injury data integration system test complete!")

if __name__ == "__main__":
    main()
