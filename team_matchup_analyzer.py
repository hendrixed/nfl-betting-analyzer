#!/usr/bin/env python3
"""
Advanced Team Matchup Analysis System
Comprehensive team strength ratings, defensive rankings, pace analysis, and matchup-specific predictions.
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
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class TeamStrength:
    team: str
    offensive_rating: float
    defensive_rating: float
    pace_factor: float
    home_field_advantage: float
    recent_form: float
    injury_impact: float

@dataclass
class DefensiveRankings:
    team: str
    pass_defense_rank: int
    rush_defense_rank: int
    red_zone_defense_rank: int
    third_down_defense_rank: int
    points_allowed_per_game: float
    yards_allowed_per_game: float
    takeaway_rate: float

@dataclass
class MatchupAnalysis:
    home_team: str
    away_team: str
    projected_total: float
    projected_spread: float
    pace_projection: int
    weather_impact: float
    key_matchup_advantages: List[str]
    betting_edges: List[str]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TeamMatchupAnalyzer:
    """Comprehensive team strength and matchup analysis system."""
    
    def __init__(self, db_path: str = "sqlite:///data/nfl_predictions.db"):
        self.db_path = db_path
        self.engine = create_engine(db_path)
        
        # NFL team mappings
        self.nfl_teams = {
            'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
            'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
            'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
            'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
            'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
            'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers',
            'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
            'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
            'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
            'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
            'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
        }
        
        # Defensive position mappings
        self.defensive_stats = {
            'pass_defense': ['passing_yards_allowed', 'passing_tds_allowed', 'interceptions', 'sacks'],
            'rush_defense': ['rushing_yards_allowed', 'rushing_tds_allowed', 'tackles_for_loss'],
            'red_zone_defense': ['red_zone_attempts', 'red_zone_scores', 'goal_line_stands'],
            'turnover_creation': ['interceptions', 'fumbles_recovered', 'defensive_tds']
        }
        
        # Offensive efficiency metrics
        self.offensive_stats = {
            'passing_offense': ['passing_yards', 'passing_tds', 'completion_percentage', 'qb_rating'],
            'rushing_offense': ['rushing_yards', 'rushing_tds', 'yards_per_carry'],
            'red_zone_offense': ['red_zone_efficiency', 'goal_line_efficiency'],
            'third_down_offense': ['third_down_conversions', 'third_down_attempts']
        }
        
        self.team_ratings_cache = {}
        self.matchup_cache = {}
        self.cache_expiry = timedelta(hours=6)
        
    def calculate_team_ratings(self) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive team strength ratings."""
        if self.team_ratings_cache and \
           datetime.now() - self.team_ratings_cache.get('timestamp', datetime.min) < self.cache_expiry:
            return self.team_ratings_cache['ratings']
        
        logger.info("Calculating team strength ratings...")
        
        # Get team performance data from database
        query = text("""
            SELECT 
                CASE 
                    WHEN player_id LIKE '%_qb' THEN SUBSTR(player_id, 1, LENGTH(player_id) - 3)
                    WHEN player_id LIKE '%_rb' THEN SUBSTR(player_id, 1, LENGTH(player_id) - 3)
                    WHEN player_id LIKE '%_wr' THEN SUBSTR(player_id, 1, LENGTH(player_id) - 3)
                    WHEN player_id LIKE '%_te' THEN SUBSTR(player_id, 1, LENGTH(player_id) - 3)
                    ELSE SUBSTR(player_id, 1, 3)
                END as team,
                AVG(passing_yards) as avg_passing_yards,
                AVG(passing_touchdowns) as avg_passing_tds,
                AVG(rushing_yards) as avg_rushing_yards,
                AVG(rushing_touchdowns) as avg_rushing_tds,
                AVG(receiving_yards) as avg_receiving_yards,
                AVG(receiving_touchdowns) as avg_receiving_tds,
                COUNT(*) as games_played
            FROM player_game_stats 
            WHERE created_at >= date('now', '-60 days')
            GROUP BY team
            HAVING games_played >= 5
        """)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn)
            
            team_ratings = {}
            
            for _, row in df.iterrows():
                team = row['team'].upper()
                if len(team) <= 3 and team in self.nfl_teams:
                    
                    # Calculate offensive ratings (0-100 scale)
                    passing_rating = min(100, (row['avg_passing_yards'] or 0) / 3.5 + 
                                       (row['avg_passing_tds'] or 0) * 15)
                    rushing_rating = min(100, (row['avg_rushing_yards'] or 0) / 1.5 + 
                                       (row['avg_rushing_tds'] or 0) * 20)
                    receiving_rating = min(100, (row['avg_receiving_yards'] or 0) / 2.0 + 
                                         (row['avg_receiving_tds'] or 0) * 18)
                    
                    # Overall offensive rating
                    offensive_rating = (passing_rating * 0.4 + rushing_rating * 0.3 + 
                                      receiving_rating * 0.3)
                    
                    # Defensive rating (inverse of points allowed - simulated)
                    # In real implementation, this would use actual defensive stats
                    defensive_rating = 75 + np.random.normal(0, 10)  # Placeholder
                    defensive_rating = max(0, min(100, defensive_rating))
                    
                    # Special teams rating (placeholder)
                    special_teams_rating = 50 + np.random.normal(0, 15)
                    special_teams_rating = max(0, min(100, special_teams_rating))
                    
                    # Overall team rating
                    overall_rating = (offensive_rating * 0.45 + defensive_rating * 0.45 + 
                                    special_teams_rating * 0.1)
                    
                    team_ratings[team] = {
                        'overall': overall_rating,
                        'offense': offensive_rating,
                        'defense': defensive_rating,
                        'special_teams': special_teams_rating,
                        'passing_offense': passing_rating,
                        'rushing_offense': rushing_rating,
                        'receiving_offense': receiving_rating,
                        'games_analyzed': row['games_played']
                    }
            
            # Cache the ratings
            self.team_ratings_cache = {
                'ratings': team_ratings,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Calculated ratings for {len(team_ratings)} teams")
            return team_ratings
            
        except Exception as e:
            logger.error(f"Error calculating team ratings: {e}")
            return {}
    
    def get_defensive_matchup_rating(self, offensive_team: str, defensive_team: str, 
                                   stat_category: str) -> float:
        """Get defensive matchup rating for specific stat category."""
        team_ratings = self.calculate_team_ratings()
        
        if offensive_team not in team_ratings or defensive_team not in team_ratings:
            return 1.0  # Neutral matchup
        
        offense_rating = team_ratings[offensive_team].get(f'{stat_category}_offense', 50)
        defense_rating = team_ratings[defensive_team].get('defense', 50)
        
        # Calculate matchup advantage
        # Higher offensive rating vs lower defensive rating = favorable matchup
        matchup_score = (offense_rating - defense_rating + 100) / 100
        
        # Clamp between 0.5 and 1.5 for reasonable adjustments
        return max(0.5, min(1.5, matchup_score))
    
    def analyze_head_to_head(self, team1: str, team2: str) -> Dict[str, Any]:
        """Analyze head-to-head matchup between two teams."""
        team_ratings = self.calculate_team_ratings()
        
        if team1 not in team_ratings or team2 not in team_ratings:
            return {
                'favorite': team1,
                'spread_estimate': 0.0,
                'confidence': 0.5,
                'key_matchups': []
            }
        
        team1_rating = team_ratings[team1]['overall']
        team2_rating = team_ratings[team2]['overall']
        
        # Determine favorite and spread
        if team1_rating > team2_rating:
            favorite = team1
            spread_estimate = (team1_rating - team2_rating) * 0.3  # Rough conversion
        else:
            favorite = team2
            spread_estimate = (team2_rating - team1_rating) * 0.3
        
        # Calculate confidence based on rating difference
        rating_diff = abs(team1_rating - team2_rating)
        confidence = min(0.95, 0.5 + (rating_diff / 100))
        
        # Key matchup analysis
        key_matchups = []
        
        # Passing matchup
        team1_pass = team_ratings[team1].get('passing_offense', 50)
        team2_defense = team_ratings[team2].get('defense', 50)
        pass_advantage = team1_pass - team2_defense
        
        if abs(pass_advantage) > 10:
            key_matchups.append({
                'type': 'passing',
                'advantage_team': team1 if pass_advantage > 0 else team2,
                'advantage_score': abs(pass_advantage)
            })
        
        # Rushing matchup
        team1_rush = team_ratings[team1].get('rushing_offense', 50)
        rush_advantage = team1_rush - team2_defense
        
        if abs(rush_advantage) > 10:
            key_matchups.append({
                'type': 'rushing',
                'advantage_team': team1 if rush_advantage > 0 else team2,
                'advantage_score': abs(rush_advantage)
            })
        
        return {
            'favorite': favorite,
            'spread_estimate': spread_estimate,
            'confidence': confidence,
            'key_matchups': key_matchups,
            'team_ratings': {
                team1: team_ratings[team1],
                team2: team_ratings[team2]
            }
        }
    
    def get_positional_matchup_factors(self, player_team: str, opponent_team: str, 
                                     position: str) -> Dict[str, float]:
        """Get matchup factors for specific position."""
        team_ratings = self.calculate_team_ratings()
        
        factors = {
            'base_matchup': 1.0,
            'pace_factor': 1.0,
            'game_script': 1.0,
            'weather_factor': 1.0
        }
        
        if player_team not in team_ratings or opponent_team not in team_ratings:
            return factors
        
        player_team_data = team_ratings[player_team]
        opponent_team_data = team_ratings[opponent_team]
        
        # Position-specific matchup analysis
        if position == 'QB':
            # QB vs pass defense
            qb_rating = player_team_data.get('passing_offense', 50)
            pass_def = opponent_team_data.get('defense', 50)
            factors['base_matchup'] = (qb_rating - pass_def + 100) / 100
            
        elif position in ['RB']:
            # RB vs run defense
            rush_rating = player_team_data.get('rushing_offense', 50)
            rush_def = opponent_team_data.get('defense', 50)
            factors['base_matchup'] = (rush_rating - rush_def + 100) / 100
            
        elif position in ['WR', 'TE']:
            # WR/TE vs pass defense
            rec_rating = player_team_data.get('receiving_offense', 50)
            pass_def = opponent_team_data.get('defense', 50)
            factors['base_matchup'] = (rec_rating - pass_def + 100) / 100
        
        # Game script factor (based on team strength difference)
        team_diff = player_team_data['overall'] - opponent_team_data['overall']
        
        if position == 'QB':
            # QBs benefit from being behind (more passing)
            factors['game_script'] = 1.0 + (team_diff * -0.002)
        elif position == 'RB':
            # RBs benefit from being ahead (more rushing)
            factors['game_script'] = 1.0 + (team_diff * 0.002)
        
        # Clamp factors
        for key in factors:
            factors[key] = max(0.5, min(1.5, factors[key]))
        
        return factors
    
    def get_strength_of_schedule(self, team: str, weeks_ahead: int = 4) -> Dict[str, float]:
        """Calculate strength of schedule for upcoming games."""
        team_ratings = self.calculate_team_ratings()
        
        if team not in team_ratings:
            return {'overall': 0.5, 'offensive': 0.5, 'defensive': 0.5}
        
        # This would require actual schedule data
        # For now, return average difficulty
        return {
            'overall': 0.5,  # 0 = easiest, 1 = hardest
            'offensive': 0.5,
            'defensive': 0.5,
            'upcoming_opponents': []  # Would list actual opponents
        }
    
    def create_matchup_features(self, player_team: str, opponent_team: str, 
                              position: str) -> Dict[str, float]:
        """Create matchup-based features for ML models."""
        team_ratings = self.calculate_team_ratings()
        matchup_factors = self.get_positional_matchup_factors(player_team, opponent_team, position)
        
        features = {
            'team_overall_rating': team_ratings.get(player_team, {}).get('overall', 50) / 100,
            'opponent_overall_rating': team_ratings.get(opponent_team, {}).get('overall', 50) / 100,
            'team_offense_rating': team_ratings.get(player_team, {}).get('offense', 50) / 100,
            'opponent_defense_rating': team_ratings.get(opponent_team, {}).get('defense', 50) / 100,
            'matchup_advantage': matchup_factors['base_matchup'],
            'game_script_factor': matchup_factors['game_script'],
            'pace_factor': matchup_factors['pace_factor']
        }
        
        # Position-specific features
        if position == 'QB':
            features['passing_matchup'] = self.get_defensive_matchup_rating(
                player_team, opponent_team, 'passing'
            )
        elif position == 'RB':
            features['rushing_matchup'] = self.get_defensive_matchup_rating(
                player_team, opponent_team, 'rushing'
            )
        elif position in ['WR', 'TE']:
            features['receiving_matchup'] = self.get_defensive_matchup_rating(
                player_team, opponent_team, 'receiving'
            )
        
        return features

def main():
    """Test the team matchup analysis system."""
    print("🏈 NFL Team Matchup Analysis System")
    print("=" * 50)
    
    # Initialize system
    analyzer = TeamMatchupAnalyzer()
    
    # Calculate team ratings
    print("📊 Calculating team strength ratings...")
    team_ratings = analyzer.calculate_team_ratings()
    print(f"   ✅ Calculated ratings for {len(team_ratings)} teams")
    
    # Show top teams
    if team_ratings:
        sorted_teams = sorted(team_ratings.items(), key=lambda x: x[1]['overall'], reverse=True)
        print("\n🏆 Top 5 Teams by Overall Rating:")
        for i, (team, ratings) in enumerate(sorted_teams[:5]):
            print(f"   {i+1}. {team}: {ratings['overall']:.1f} (O: {ratings['offense']:.1f}, D: {ratings['defense']:.1f})")
    
    # Test head-to-head analysis
    print("\n⚔️  Head-to-Head Analysis (KC vs BUF)...")
    h2h = analyzer.analyze_head_to_head('KC', 'BUF')
    print(f"   Favorite: {h2h['favorite']}")
    print(f"   Estimated Spread: {h2h['spread_estimate']:.1f}")
    print(f"   Confidence: {h2h['confidence']:.3f}")
    print(f"   Key Matchups: {len(h2h['key_matchups'])}")
    
    # Test matchup factors
    print("\n🎯 Positional Matchup Analysis (KC QB vs BUF)...")
    factors = analyzer.get_positional_matchup_factors('KC', 'BUF', 'QB')
    print("   Matchup Factors:")
    for factor, value in factors.items():
        print(f"   {factor}: {value:.3f}")
    
    # Test matchup features
    print("\n🔧 Matchup Features for ML Models...")
    features = analyzer.create_matchup_features('KC', 'BUF', 'QB')
    print("   Generated Features:")
    for feature, value in features.items():
        print(f"   {feature}: {value:.3f}")
    
    print("\n✅ Team matchup analysis system test complete!")

if __name__ == "__main__":
    main()
