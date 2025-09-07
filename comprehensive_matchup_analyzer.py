"""
Comprehensive Matchup Analyzer

This module provides comprehensive matchup analysis including historical
team performance, offensive vs defensive matchups, and predictive
matchup factors for enhanced player predictions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from sqlalchemy.orm import Session
from sqlalchemy import text, func
import json

from database_models import PlayerGameStats, Game, Player

logger = logging.getLogger(__name__)

@dataclass
class TeamMatchupProfile:
    """Comprehensive team matchup profile"""
    team: str
    season: int
    
    # Offensive Metrics
    points_per_game: float = 0.0
    total_yards_per_game: float = 0.0
    passing_yards_per_game: float = 0.0
    rushing_yards_per_game: float = 0.0
    red_zone_efficiency: float = 0.0
    third_down_conversion_rate: float = 0.0
    
    # Defensive Metrics  
    points_allowed_per_game: float = 0.0
    yards_allowed_per_game: float = 0.0
    passing_yards_allowed_per_game: float = 0.0
    rushing_yards_allowed_per_game: float = 0.0
    red_zone_defense: float = 0.0
    third_down_defense: float = 0.0
    
    # Advanced Metrics
    pace_of_play: float = 0.0           # Plays per game
    time_of_possession: float = 0.0     # Minutes per game
    turnover_differential: float = 0.0
    
    # Positional Strength Ratings (0-100 scale)
    qb_rating: float = 50.0
    rb_rating: float = 50.0
    wr_rating: float = 50.0
    te_rating: float = 50.0
    
    # Defensive Position Ratings (lower = better for offense)
    pass_defense_rating: float = 50.0
    run_defense_rating: float = 50.0
    
    # Context Factors
    home_field_advantage: float = 0.0   # Point differential at home
    weather_impact_factor: float = 1.0
    rest_advantage: int = 0             # Days since last game
    
    # Data Quality
    games_analyzed: int = 0
    data_confidence: float = 0.0

@dataclass
class PositionalMatchup:
    """Specific positional matchup analysis"""
    offense_team: str
    defense_team: str
    position: str
    
    # Historical Performance
    avg_points_allowed: float = 0.0
    avg_yards_allowed: float = 0.0
    avg_touchdowns_allowed: float = 0.0
    
    # Matchup Factors
    scheme_advantage: float = 1.0       # Offensive scheme vs defensive scheme
    personnel_advantage: float = 1.0    # Player talent advantage
    coaching_advantage: float = 1.0     # Coaching/game plan advantage
    
    # Situational Factors
    pace_impact: float = 1.0           # Impact of game pace
    game_script_impact: float = 1.0    # Impact of expected game flow
    weather_impact: float = 1.0        # Weather impact on position
    
    # Overall Matchup Rating
    matchup_rating: str = "neutral"     # poor/below_avg/neutral/above_avg/excellent
    confidence: float = 0.5             # Confidence in rating

@dataclass
class ComprehensiveMatchupReport:
    """Complete matchup analysis report"""
    home_team: str
    away_team: str
    week: int
    season: int
    
    # Team Profiles
    home_team_profile: TeamMatchupProfile
    away_team_profile: TeamMatchupProfile
    
    # Key Matchups
    qb_matchup: PositionalMatchup
    rb_matchup: PositionalMatchup
    wr_matchup: PositionalMatchup
    te_matchup: PositionalMatchup
    
    # Game-Level Predictions
    expected_total_points: float = 0.0
    expected_pace: float = 0.0
    game_script_prediction: str = "balanced"  # run_heavy/pass_heavy/balanced
    
    # Key Factors
    key_advantages: List[str] = field(default_factory=list)
    key_concerns: List[str] = field(default_factory=list)
    
    # Overall Assessment
    overall_confidence: float = 0.0
    recommendation: str = "moderate"    # avoid/low/moderate/high/excellent

class ComprehensiveMatchupAnalyzer:
    """Analyze comprehensive matchup data for enhanced predictions"""
    
    def __init__(self, session: Session):
        self.session = session
        
        # Matchup analysis parameters
        self.historical_window = 8  # Games to analyze for recent form
        self.seasonal_window = 16   # Games for seasonal analysis
        self.multi_season_window = 32  # Games across seasons
        
        # Rating scales
        self.rating_percentiles = {
            'elite': 90,      # Top 10%
            'good': 75,       # Top 25%
            'average': 50,    # Average
            'below_avg': 25,  # Bottom 25%
            'poor': 10        # Bottom 10%
        }
        
    async def generate_comprehensive_matchup_report(self, home_team: str, away_team: str, 
                                                  week: int, season: int) -> ComprehensiveMatchupReport:
        """Generate complete matchup analysis report"""
        
        logger.info(f"Generating comprehensive matchup report: {away_team} @ {home_team}, Week {week}")
        
        try:
            # Generate team profiles
            home_profile = await self._generate_team_profile(home_team, season, week)
            away_profile = await self._generate_team_profile(away_team, season, week)
            
            # Analyze positional matchups
            qb_matchup = await self._analyze_positional_matchup(away_team, home_team, 'QB', season)
            rb_matchup = await self._analyze_positional_matchup(away_team, home_team, 'RB', season)
            wr_matchup = await self._analyze_positional_matchup(away_team, home_team, 'WR', season)
            te_matchup = await self._analyze_positional_matchup(away_team, home_team, 'TE', season)
            
            # Game-level analysis
            game_analysis = await self._analyze_game_level_factors(home_team, away_team, season, week)
            
            # Create comprehensive report
            report = ComprehensiveMatchupReport(
                home_team=home_team,
                away_team=away_team,
                week=week,
                season=season,
                home_team_profile=home_profile,
                away_team_profile=away_profile,
                qb_matchup=qb_matchup,
                rb_matchup=rb_matchup,
                wr_matchup=wr_matchup,
                te_matchup=te_matchup,
                **game_analysis
            )
            
            # Add overall assessment
            report = await self._add_overall_assessment(report)
            
            logger.info(f"Matchup report generated with {report.overall_confidence:.3f} confidence")
            return report
            
        except Exception as e:
            logger.error(f"Error generating matchup report: {e}")
            # Return basic report with default values
            return ComprehensiveMatchupReport(
                home_team=home_team,
                away_team=away_team,
                week=week,
                season=season,
                home_team_profile=TeamMatchupProfile(home_team, season),
                away_team_profile=TeamMatchupProfile(away_team, season),
                qb_matchup=PositionalMatchup(away_team, home_team, 'QB'),
                rb_matchup=PositionalMatchup(away_team, home_team, 'RB'),
                wr_matchup=PositionalMatchup(away_team, home_team, 'WR'),
                te_matchup=PositionalMatchup(away_team, home_team, 'TE')
            )
    
    async def _generate_team_profile(self, team: str, season: int, week: int) -> TeamMatchupProfile:
        """Generate comprehensive team profile"""
        
        try:
            # Query team statistics from recent games
            query = text("""
                SELECT 
                    COUNT(DISTINCT game_id) as games_played,
                    AVG(CASE WHEN team = :team THEN 
                        passing_yards + rushing_yards + receiving_yards 
                        ELSE 0 END) as avg_offense_yards,
                    AVG(CASE WHEN opponent = :team THEN 
                        passing_yards + rushing_yards + receiving_yards 
                        ELSE 0 END) as avg_defense_yards,
                    AVG(CASE WHEN team = :team THEN passing_yards ELSE 0 END) as avg_passing_yards,
                    AVG(CASE WHEN team = :team THEN rushing_yards ELSE 0 END) as avg_rushing_yards,
                    AVG(CASE WHEN team = :team THEN fantasy_points_ppr ELSE 0 END) as avg_fantasy_production,
                    AVG(CASE WHEN team = :team THEN 
                        passing_touchdowns + rushing_touchdowns + receiving_touchdowns 
                        ELSE 0 END) as avg_touchdowns,
                    AVG(CASE WHEN opponent = :team THEN 
                        passing_touchdowns + rushing_touchdowns + receiving_touchdowns 
                        ELSE 0 END) as avg_touchdowns_allowed
                FROM player_game_stats 
                WHERE (team = :team OR opponent = :team)
                AND created_at >= date('now', '-{} days')
            """.format(self.historical_window * 7))
            
            result = self.session.execute(query, {"team": team}).fetchone()
            
            if result and result[0] > 0:  # Has data
                profile = TeamMatchupProfile(
                    team=team,
                    season=season,
                    total_yards_per_game=float(result[1] or 0),
                    yards_allowed_per_game=float(result[2] or 0),
                    passing_yards_per_game=float(result[3] or 0),
                    rushing_yards_per_game=float(result[4] or 0),
                    games_analyzed=int(result[0] or 0)
                )
                
                # Calculate additional metrics
                profile.points_per_game = float(result[6] or 0) * 6  # Rough TD to points conversion
                profile.points_allowed_per_game = float(result[7] or 0) * 6
                
                # Calculate ratings and additional metrics
                profile = await self._enhance_team_profile(profile)
                
                return profile
            
        except Exception as e:
            logger.warning(f"Error generating team profile for {team}: {e}")
        
        # Return default profile
        return TeamMatchupProfile(team=team, season=season)
    
    async def _enhance_team_profile(self, profile: TeamMatchupProfile) -> TeamMatchupProfile:
        """Enhance team profile with calculated ratings and metrics"""
        
        # Calculate positional ratings based on statistical production
        total_offense = profile.passing_yards_per_game + profile.rushing_yards_per_game
        
        # Offensive ratings based on yards per game
        if total_offense > 400:
            profile.qb_rating = 85
            profile.rb_rating = 80
            profile.wr_rating = 85
            profile.te_rating = 75
        elif total_offense > 350:
            profile.qb_rating = 70
            profile.rb_rating = 65
            profile.wr_rating = 70
            profile.te_rating = 60
        elif total_offense > 300:
            profile.qb_rating = 55
            profile.rb_rating = 50
            profile.wr_rating = 55
            profile.te_rating = 45
        else:
            profile.qb_rating = 40
            profile.rb_rating = 35
            profile.wr_rating = 40
            profile.te_rating = 30
        
        # Defensive ratings (inverse of yards allowed)
        if profile.yards_allowed_per_game < 300:
            profile.pass_defense_rating = 85
            profile.run_defense_rating = 85
        elif profile.yards_allowed_per_game < 350:
            profile.pass_defense_rating = 65
            profile.run_defense_rating = 65
        elif profile.yards_allowed_per_game < 400:
            profile.pass_defense_rating = 50
            profile.run_defense_rating = 50
        else:
            profile.pass_defense_rating = 35
            profile.run_defense_rating = 35
        
        # Calculate pace and efficiency metrics
        profile.pace_of_play = 65.0  # Default NFL average
        profile.red_zone_efficiency = 0.6  # Default 60%
        profile.third_down_conversion_rate = 0.4  # Default 40%
        
        # Calculate data confidence
        profile.data_confidence = min(1.0, profile.games_analyzed / 8.0)
        
        return profile
    
    async def _analyze_positional_matchup(self, offense_team: str, defense_team: str, 
                                        position: str, season: int) -> PositionalMatchup:
        """Analyze specific positional matchup"""
        
        try:
            # Query historical performance vs this defense
            query = text("""
                SELECT 
                    AVG(fantasy_points_ppr) as avg_fantasy,
                    AVG(passing_yards + rushing_yards + receiving_yards) as avg_yards,
                    AVG(passing_touchdowns + rushing_touchdowns + receiving_touchdowns) as avg_tds,
                    COUNT(*) as games,
                    AVG(snap_count) as avg_snaps
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                WHERE p.position = :position
                AND pgs.opponent = :defense_team
                AND pgs.created_at >= date('now', '-{} days')
                AND pgs.fantasy_points_ppr > 0
            """.format(self.multi_season_window * 7))
            
            result = self.session.execute(query, {
                "position": position, 
                "defense_team": defense_team
            }).fetchone()
            
            matchup = PositionalMatchup(
                offense_team=offense_team,
                defense_team=defense_team,
                position=position
            )
            
            if result and result[3] > 0:  # Has historical data
                matchup.avg_points_allowed = float(result[0] or 0)
                matchup.avg_yards_allowed = float(result[1] or 0)
                matchup.avg_touchdowns_allowed = float(result[2] or 0)
                matchup.confidence = min(1.0, float(result[3]) / 10.0)
                
                # Determine matchup rating based on fantasy points allowed
                if matchup.avg_points_allowed > 18:
                    matchup.matchup_rating = "excellent"
                elif matchup.avg_points_allowed > 15:
                    matchup.matchup_rating = "above_avg"
                elif matchup.avg_points_allowed > 12:
                    matchup.matchup_rating = "neutral"
                elif matchup.avg_points_allowed > 9:
                    matchup.matchup_rating = "below_avg"
                else:
                    matchup.matchup_rating = "poor"
                
                # Calculate scheme and personnel advantages
                matchup.scheme_advantage = self._calculate_scheme_advantage(offense_team, defense_team, position)
                matchup.personnel_advantage = self._calculate_personnel_advantage(offense_team, defense_team, position)
                
            return matchup
            
        except Exception as e:
            logger.warning(f"Error analyzing {position} matchup: {e}")
            return PositionalMatchup(offense_team, defense_team, position)
    
    def _calculate_scheme_advantage(self, offense_team: str, defense_team: str, position: str) -> float:
        """Calculate scheme advantage factor"""
        
        # Simplified scheme analysis - would be enhanced with actual scheme data
        scheme_matchups = {
            'QB': {
                ('KC', 'NE'): 1.2,  # High-powered offense vs bend-don't-break defense
                ('BUF', 'MIA'): 1.1,
                ('TB', 'NO'): 0.9,
            },
            'RB': {
                ('BAL', 'CIN'): 1.3,  # Run-heavy offense vs weak run defense
                ('SF', 'ARI'): 1.2,
            },
            'WR': {
                ('KC', 'LV'): 1.2,  # Pass-heavy offense vs weak secondary
                ('LAR', 'SEA'): 1.1,
            }
        }
        
        position_matchups = scheme_matchups.get(position, {})
        return position_matchups.get((offense_team, defense_team), 1.0)
    
    def _calculate_personnel_advantage(self, offense_team: str, defense_team: str, position: str) -> float:
        """Calculate personnel advantage factor"""
        
        # Simplified personnel analysis - would use actual player ratings
        elite_offenses = ['KC', 'BUF', 'CIN', 'LAR', 'TB']
        elite_defenses = ['SF', 'BUF', 'DAL', 'NE', 'PIT']
        
        offense_tier = 1.2 if offense_team in elite_offenses else 1.0
        defense_tier = 0.8 if defense_team in elite_defenses else 1.0
        
        return offense_tier * defense_tier
    
    async def _analyze_game_level_factors(self, home_team: str, away_team: str, 
                                        season: int, week: int) -> Dict[str, Any]:
        """Analyze game-level factors like pace, script, totals"""
        
        try:
            # Query historical head-to-head data
            h2h_query = text("""
                SELECT 
                    AVG(fantasy_points_ppr) as avg_points,
                    COUNT(DISTINCT game_id) as games
                FROM player_game_stats 
                WHERE ((team = :home_team AND opponent = :away_team) 
                       OR (team = :away_team AND opponent = :home_team))
                AND created_at >= date('now', '-1095 days')  -- 3 years
            """)
            
            h2h_result = self.session.execute(h2h_query, {
                "home_team": home_team,
                "away_team": away_team
            }).fetchone()
            
            # Calculate expected totals and pace
            expected_total = float(h2h_result[0] or 45.0) if h2h_result else 45.0
            expected_pace = 65.0  # NFL average plays per game
            
            # Determine game script
            game_script = "balanced"  # Default
            
            # Key advantages and concerns
            key_advantages = []
            key_concerns = []
            
            # Add home field advantage
            key_advantages.append(f"{home_team} home field advantage")
            
            # Weather considerations (simplified)
            if week >= 12:  # Late season
                key_concerns.append("Potential weather impact")
            
            return {
                'expected_total_points': expected_total,
                'expected_pace': expected_pace,
                'game_script_prediction': game_script,
                'key_advantages': key_advantages,
                'key_concerns': key_concerns
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing game-level factors: {e}")
            return {
                'expected_total_points': 45.0,
                'expected_pace': 65.0,
                'game_script_prediction': 'balanced',
                'key_advantages': ['Home field advantage'],
                'key_concerns': ['Weather conditions']
            }
    
    async def _add_overall_assessment(self, report: ComprehensiveMatchupReport) -> ComprehensiveMatchupReport:
        """Add overall assessment and recommendation"""
        
        # Calculate overall confidence based on individual matchup confidences
        confidences = [
            report.qb_matchup.confidence,
            report.rb_matchup.confidence,
            report.wr_matchup.confidence,
            report.te_matchup.confidence,
            report.home_team_profile.data_confidence,
            report.away_team_profile.data_confidence
        ]
        
        report.overall_confidence = np.mean(confidences)
        
        # Determine recommendation based on matchup ratings
        excellent_matchups = sum(1 for m in [report.qb_matchup, report.rb_matchup, 
                                           report.wr_matchup, report.te_matchup] 
                               if m.matchup_rating == "excellent")
        
        good_matchups = sum(1 for m in [report.qb_matchup, report.rb_matchup, 
                                      report.wr_matchup, report.te_matchup] 
                          if m.matchup_rating in ["excellent", "above_avg"])
        
        if excellent_matchups >= 2:
            report.recommendation = "excellent"
        elif excellent_matchups >= 1 or good_matchups >= 3:
            report.recommendation = "high"
        elif good_matchups >= 2:
            report.recommendation = "moderate"
        elif good_matchups >= 1:
            report.recommendation = "low"
        else:
            report.recommendation = "avoid"
        
        return report
    
    def get_historical_head_to_head(self, team1: str, team2: str, 
                                   seasons: int = 3) -> Dict[str, Any]:
        """Get historical head-to-head performance between teams"""
        
        try:
            query = text("""
                SELECT 
                    COUNT(DISTINCT game_id) as total_games,
                    AVG(CASE WHEN team = :team1 THEN fantasy_points_ppr ELSE 0 END) as team1_avg,
                    AVG(CASE WHEN team = :team2 THEN fantasy_points_ppr ELSE 0 END) as team2_avg,
                    AVG(fantasy_points_ppr) as avg_total_points
                FROM player_game_stats 
                WHERE ((team = :team1 AND opponent = :team2) 
                       OR (team = :team2 AND opponent = :team1))
                AND created_at >= date('now', '-{} days')
            """.format(seasons * 365))
            
            result = self.session.execute(query, {
                "team1": team1,
                "team2": team2
            }).fetchone()
            
            if result:
                return {
                    'total_games': int(result[0] or 0),
                    'team1_avg_points': float(result[1] or 0),
                    'team2_avg_points': float(result[2] or 0),
                    'avg_total_points': float(result[3] or 0),
                    'recent_trend': 'even'  # Simplified
                }
            
        except Exception as e:
            logger.warning(f"Error getting head-to-head data: {e}")
        
        # Return default data
        return {
            'total_games': 6,
            'team1_avg_points': 21.0,
            'team2_avg_points': 21.0,
            'avg_total_points': 42.0,
            'recent_trend': 'even'
        }
    
    def get_positional_strength_rankings(self, season: int) -> Dict[str, Dict[str, float]]:
        """Get positional strength rankings for all teams"""
        
        try:
            query = text("""
                SELECT 
                    team,
                    p.position,
                    AVG(fantasy_points_ppr) as avg_points,
                    COUNT(*) as games
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                WHERE p.position IN ('QB', 'RB', 'WR', 'TE')
                AND pgs.created_at >= date('now', '-120 days')  -- Recent form
                GROUP BY team, p.position
                HAVING COUNT(*) >= 5
                ORDER BY team, p.position, avg_points DESC
            """)
            
            results = self.session.execute(query).fetchall()
            
            rankings = {}
            for result in results:
                team = result[0]
                position = result[1]
                avg_points = float(result[2])
                
                if team not in rankings:
                    rankings[team] = {}
                
                rankings[team][position] = avg_points
            
            return rankings
            
        except Exception as e:
            logger.warning(f"Error getting positional rankings: {e}")
            return {}
    
    def calculate_matchup_multiplier(self, player_position: str, offense_team: str, 
                                   defense_team: str) -> float:
        """Calculate matchup multiplier for player projections"""
        
        # This would integrate with the main prediction system
        # to provide matchup-adjusted projections
        
        base_multiplier = 1.0
        
        # Get recent performance vs this defense
        try:
            query = text("""
                SELECT AVG(fantasy_points_ppr) as recent_avg
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                WHERE p.position = :position
                AND pgs.team = :offense_team
                AND pgs.opponent = :defense_team
                AND pgs.created_at >= date('now', '-365 days')
            """)
            
            result = self.session.execute(query, {
                "position": player_position,
                "offense_team": offense_team,
                "defense_team": defense_team
            }).fetchone()
            
            if result and result[0]:
                recent_avg = float(result[0])
                
                # Compare to position average
                if recent_avg > 15:
                    base_multiplier = 1.2
                elif recent_avg > 12:
                    base_multiplier = 1.1
                elif recent_avg < 8:
                    base_multiplier = 0.8
                elif recent_avg < 10:
                    base_multiplier = 0.9
                    
        except Exception as e:
            logger.warning(f"Error calculating matchup multiplier: {e}")
        
        return base_multiplier
