"""
Historical Trend Analyzer

This module identifies and analyzes historical trends and patterns in NFL data
to enhance predictive accuracy and identify key performance indicators.
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
from collections import defaultdict

from database_models import PlayerGameStats, Player, Game

logger = logging.getLogger(__name__)

@dataclass
class PlayerTrend:
    """Individual player performance trend"""
    player_id: str
    player_name: str
    position: str
    
    # Trend metrics
    recent_form_score: float = 0.0      # Last 4 games performance
    seasonal_trend: str = "stable"      # improving/declining/stable/volatile
    consistency_score: float = 0.0      # 0-1, higher = more consistent
    
    # Performance patterns
    home_vs_away_differential: float = 0.0
    divisional_performance: float = 0.0
    weather_impact_factor: float = 1.0
    rest_performance_factor: float = 1.0
    
    # Matchup trends
    vs_strong_defenses: float = 0.0
    vs_weak_defenses: float = 0.0
    ceiling_games_frequency: float = 0.0  # % of games with 20+ fantasy points
    floor_games_frequency: float = 0.0    # % of games with <5 fantasy points
    
    # Predictive indicators
    breakout_probability: float = 0.0
    bust_probability: float = 0.0
    injury_risk_score: float = 0.0
    
    # Data quality
    games_analyzed: int = 0
    trend_confidence: float = 0.0

@dataclass
class TeamTrend:
    """Team-level performance trends"""
    team: str
    season: int
    
    # Offensive trends
    scoring_trend: str = "stable"
    pace_trend: str = "stable"
    passing_volume_trend: str = "stable"
    rushing_volume_trend: str = "stable"
    
    # Defensive trends
    points_allowed_trend: str = "stable"
    yards_allowed_trend: str = "stable"
    turnover_trend: str = "stable"
    
    # Situational trends
    red_zone_efficiency_trend: str = "stable"
    third_down_trend: str = "stable"
    home_field_advantage_trend: float = 0.0
    
    # Advanced metrics
    strength_of_schedule_remaining: float = 0.0
    injury_impact_score: float = 0.0
    coaching_adjustment_factor: float = 1.0
    
    # Predictive factors
    playoff_probability: float = 0.0
    desperation_factor: float = 0.0  # Late season must-win situations
    
    games_analyzed: int = 0
    trend_confidence: float = 0.0

@dataclass
class SeasonalPattern:
    """League-wide seasonal patterns"""
    season: int
    week: int
    
    # Scoring patterns
    avg_points_per_game: float = 0.0
    high_scoring_game_frequency: float = 0.0
    defensive_game_frequency: float = 0.0
    
    # Position trends
    qb_performance_trend: float = 0.0
    rb_usage_trend: float = 0.0
    wr_target_share_trend: float = 0.0
    te_involvement_trend: float = 0.0
    
    # Weather and situational
    weather_impact_games: int = 0
    primetime_performance_differential: float = 0.0
    divisional_game_intensity: float = 0.0
    
    # Injury and rest patterns
    avg_injury_rate: float = 0.0
    short_rest_impact: float = 0.0
    bye_week_bounce_back: float = 0.0

class HistoricalTrendAnalyzer:
    """Analyze historical trends and patterns for predictive insights"""
    
    def __init__(self, session: Session):
        self.session = session
        
        # Analysis parameters
        self.recent_games_window = 4
        self.trend_analysis_window = 8
        self.seasonal_analysis_window = 16
        self.multi_season_window = 48
        
        # Trend thresholds
        self.improvement_threshold = 0.15  # 15% improvement
        self.decline_threshold = -0.15     # 15% decline
        self.volatility_threshold = 0.3    # 30% coefficient of variation
        
    async def analyze_player_trends(self, player_id: str, season: int) -> PlayerTrend:
        """Analyze comprehensive trends for a specific player"""
        
        logger.info(f"Analyzing trends for player {player_id}")
        
        try:
            # Get player basic info
            player_info = self.session.query(Player).filter_by(player_id=player_id).first()
            if not player_info:
                logger.warning(f"Player {player_id} not found")
                return PlayerTrend(player_id=player_id, player_name="Unknown", position="UNK")
            
            # Get recent game data
            recent_games = await self._get_player_recent_games(player_id, self.trend_analysis_window)
            
            if len(recent_games) < 3:
                logger.warning(f"Insufficient data for player {player_id}")
                return PlayerTrend(
                    player_id=player_id,
                    player_name=player_info.name,
                    position=player_info.position,
                    games_analyzed=len(recent_games)
                )
            
            trend = PlayerTrend(
                player_id=player_id,
                player_name=player_info.name,
                position=player_info.position,
                games_analyzed=len(recent_games)
            )
            
            # Calculate recent form
            trend.recent_form_score = await self._calculate_recent_form(recent_games)
            
            # Determine seasonal trend
            trend.seasonal_trend = await self._determine_seasonal_trend(recent_games)
            
            # Calculate consistency
            trend.consistency_score = await self._calculate_consistency(recent_games)
            
            # Analyze situational performance
            trend = await self._analyze_situational_performance(trend, player_id)
            
            # Calculate predictive indicators
            trend = await self._calculate_predictive_indicators(trend, recent_games)
            
            # Set trend confidence
            trend.trend_confidence = min(1.0, len(recent_games) / self.trend_analysis_window)
            
            logger.info(f"Player trend analysis complete: {trend.seasonal_trend} trend, {trend.recent_form_score:.2f} form")
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing player trends: {e}")
            return PlayerTrend(player_id=player_id, player_name="Error", position="UNK")
    
    async def _get_player_recent_games(self, player_id: str, num_games: int) -> List[Dict]:
        """Get recent games for a player"""
        
        query = text("""
            SELECT 
                game_id, team, opponent, is_home,
                fantasy_points_ppr, passing_yards, rushing_yards, receiving_yards,
                passing_touchdowns, rushing_touchdowns, receiving_touchdowns,
                targets, receptions, snap_count, snap_percentage,
                created_at
            FROM player_game_stats
            WHERE player_id = :player_id
            AND fantasy_points_ppr > 0
            ORDER BY created_at DESC
            LIMIT :num_games
        """)
        
        results = self.session.execute(query, {
            "player_id": player_id,
            "num_games": num_games
        }).fetchall()
        
        games = []
        for result in results:
            games.append({
                'game_id': result[0],
                'team': result[1],
                'opponent': result[2],
                'is_home': result[3],
                'fantasy_points': float(result[4] or 0),
                'passing_yards': int(result[5] or 0),
                'rushing_yards': int(result[6] or 0),
                'receiving_yards': int(result[7] or 0),
                'passing_tds': int(result[8] or 0),
                'rushing_tds': int(result[9] or 0),
                'receiving_tds': int(result[10] or 0),
                'targets': int(result[11] or 0),
                'receptions': int(result[12] or 0),
                'snap_count': int(result[13] or 0) if result[13] else None,
                'snap_percentage': float(result[14] or 0) if result[14] else None,
                'date': result[15]
            })
        
        return games
    
    async def _calculate_recent_form(self, games: List[Dict]) -> float:
        """Calculate recent form score (0-100)"""
        
        if not games:
            return 0.0
        
        # Take most recent games for form calculation
        recent_games = games[:self.recent_games_window]
        fantasy_points = [game['fantasy_points'] for game in recent_games]
        
        if not fantasy_points:
            return 0.0
        
        # Calculate weighted average (more recent games weighted higher)
        weights = [0.4, 0.3, 0.2, 0.1][:len(fantasy_points)]
        weighted_avg = sum(fp * w for fp, w in zip(fantasy_points, weights)) / sum(weights)
        
        # Normalize to 0-100 scale (20+ fantasy points = 100)
        form_score = min(100.0, (weighted_avg / 20.0) * 100)
        
        return form_score
    
    async def _determine_seasonal_trend(self, games: List[Dict]) -> str:
        """Determine if player is improving, declining, stable, or volatile"""
        
        if len(games) < 4:
            return "insufficient_data"
        
        fantasy_points = [game['fantasy_points'] for game in reversed(games)]  # Chronological order
        
        # Calculate trend using linear regression slope
        x = np.arange(len(fantasy_points))
        slope = np.polyfit(x, fantasy_points, 1)[0]
        
        # Calculate coefficient of variation for volatility
        cv = np.std(fantasy_points) / np.mean(fantasy_points) if np.mean(fantasy_points) > 0 else 0
        
        # Determine trend
        if cv > self.volatility_threshold:
            return "volatile"
        elif slope > self.improvement_threshold:
            return "improving"
        elif slope < self.decline_threshold:
            return "declining"
        else:
            return "stable"
    
    async def _calculate_consistency(self, games: List[Dict]) -> float:
        """Calculate consistency score (0-1, higher = more consistent)"""
        
        if not games:
            return 0.0
        
        fantasy_points = [game['fantasy_points'] for game in games]
        
        if len(fantasy_points) < 2:
            return 0.0
        
        # Use coefficient of variation (lower = more consistent)
        mean_points = np.mean(fantasy_points)
        if mean_points == 0:
            return 0.0
        
        cv = np.std(fantasy_points) / mean_points
        
        # Convert to consistency score (0-1, where 1 = perfectly consistent)
        consistency = max(0.0, 1.0 - min(cv, 1.0))
        
        return consistency
    
    async def _analyze_situational_performance(self, trend: PlayerTrend, player_id: str) -> PlayerTrend:
        """Analyze performance in different situations"""
        
        try:
            # Home vs Away performance
            home_away_query = text("""
                SELECT 
                    is_home,
                    AVG(fantasy_points_ppr) as avg_points,
                    COUNT(*) as games
                FROM player_game_stats
                WHERE player_id = :player_id
                AND fantasy_points_ppr > 0
                AND created_at >= date('now', '-365 days')
                GROUP BY is_home
            """)
            
            home_away_results = self.session.execute(home_away_query, {"player_id": player_id}).fetchall()
            
            home_avg = away_avg = 0.0
            for result in home_away_results:
                if result[0]:  # is_home = True
                    home_avg = float(result[1])
                else:  # is_home = False
                    away_avg = float(result[1])
            
            trend.home_vs_away_differential = home_avg - away_avg
            
            # Divisional performance
            divisional_query = text("""
                SELECT AVG(fantasy_points_ppr) as avg_points
                FROM player_game_stats pgs
                WHERE pgs.player_id = :player_id
                AND pgs.fantasy_points_ppr > 0
                AND pgs.created_at >= date('now', '-365 days')
                AND pgs.opponent IN (
                    SELECT DISTINCT team 
                    FROM player_game_stats 
                    WHERE team != pgs.team 
                    LIMIT 3  -- Simplified divisional check
                )
            """)
            
            divisional_result = self.session.execute(divisional_query, {"player_id": player_id}).fetchone()
            trend.divisional_performance = float(divisional_result[0] or 0) if divisional_result else 0.0
            
            # Defense strength analysis
            strong_def_query = text("""
                SELECT AVG(fantasy_points_ppr) as avg_points
                FROM player_game_stats pgs
                WHERE pgs.player_id = :player_id
                AND pgs.fantasy_points_ppr > 0
                AND pgs.created_at >= date('now', '-365 days')
                AND pgs.opponent IN (
                    SELECT opponent
                    FROM player_game_stats
                    GROUP BY opponent
                    HAVING AVG(fantasy_points_ppr) < 12  -- Strong defenses allow <12 avg
                )
            """)
            
            strong_def_result = self.session.execute(strong_def_query, {"player_id": player_id}).fetchone()
            trend.vs_strong_defenses = float(strong_def_result[0] or 0) if strong_def_result else 0.0
            
            weak_def_query = text("""
                SELECT AVG(fantasy_points_ppr) as avg_points
                FROM player_game_stats pgs
                WHERE pgs.player_id = :player_id
                AND pgs.fantasy_points_ppr > 0
                AND pgs.created_at >= date('now', '-365 days')
                AND pgs.opponent IN (
                    SELECT opponent
                    FROM player_game_stats
                    GROUP BY opponent
                    HAVING AVG(fantasy_points_ppr) > 16  -- Weak defenses allow >16 avg
                )
            """)
            
            weak_def_result = self.session.execute(weak_def_query, {"player_id": player_id}).fetchone()
            trend.vs_weak_defenses = float(weak_def_result[0] or 0) if weak_def_result else 0.0
            
        except Exception as e:
            logger.warning(f"Error analyzing situational performance: {e}")
        
        return trend
    
    async def _calculate_predictive_indicators(self, trend: PlayerTrend, games: List[Dict]) -> PlayerTrend:
        """Calculate predictive indicators for future performance"""
        
        if not games:
            return trend
        
        fantasy_points = [game['fantasy_points'] for game in games]
        
        # Ceiling and floor game frequencies
        ceiling_games = sum(1 for fp in fantasy_points if fp >= 20)
        floor_games = sum(1 for fp in fantasy_points if fp < 5)
        
        trend.ceiling_games_frequency = ceiling_games / len(fantasy_points)
        trend.floor_games_frequency = floor_games / len(fantasy_points)
        
        # Breakout probability (based on recent improvement and usage)
        if trend.seasonal_trend == "improving" and trend.recent_form_score > 70:
            trend.breakout_probability = 0.3
        elif trend.seasonal_trend == "improving":
            trend.breakout_probability = 0.15
        else:
            trend.breakout_probability = 0.05
        
        # Bust probability (based on decline and inconsistency)
        if trend.seasonal_trend == "declining" and trend.consistency_score < 0.3:
            trend.bust_probability = 0.4
        elif trend.seasonal_trend == "volatile":
            trend.bust_probability = 0.25
        else:
            trend.bust_probability = 0.1
        
        # Injury risk (simplified - based on snap count trends)
        snap_counts = [game.get('snap_count', 0) for game in games if game.get('snap_count')]
        if snap_counts and len(snap_counts) >= 3:
            recent_snaps = np.mean(snap_counts[:3])
            earlier_snaps = np.mean(snap_counts[3:]) if len(snap_counts) > 3 else recent_snaps
            
            if recent_snaps < earlier_snaps * 0.8:  # 20% decline in snaps
                trend.injury_risk_score = 0.3
            else:
                trend.injury_risk_score = 0.1
        
        return trend
    
    async def analyze_team_trends(self, team: str, season: int) -> TeamTrend:
        """Analyze comprehensive trends for a team"""
        
        logger.info(f"Analyzing trends for team {team}")
        
        try:
            trend = TeamTrend(team=team, season=season)
            
            # Get team's recent performance data
            team_query = text("""
                SELECT 
                    AVG(fantasy_points_ppr) as avg_points,
                    AVG(passing_yards + rushing_yards + receiving_yards) as avg_yards,
                    COUNT(DISTINCT game_id) as games
                FROM player_game_stats
                WHERE team = :team
                AND created_at >= date('now', '-120 days')
                GROUP BY team
            """)
            
            team_result = self.session.execute(team_query, {"team": team}).fetchone()
            
            if team_result:
                trend.games_analyzed = int(team_result[2] or 0)
                
                # Analyze scoring trends (simplified)
                avg_points = float(team_result[0] or 0)
                if avg_points > 25:
                    trend.scoring_trend = "improving"
                elif avg_points < 18:
                    trend.scoring_trend = "declining"
                else:
                    trend.scoring_trend = "stable"
                
                # Calculate trend confidence
                trend.trend_confidence = min(1.0, trend.games_analyzed / 8.0)
            
            logger.info(f"Team trend analysis complete: {trend.scoring_trend} scoring trend")
            return trend
            
        except Exception as e:
            logger.error(f"Error analyzing team trends: {e}")
            return TeamTrend(team=team, season=season)
    
    async def analyze_seasonal_patterns(self, season: int, week: int) -> SeasonalPattern:
        """Analyze league-wide seasonal patterns"""
        
        logger.info(f"Analyzing seasonal patterns for {season} Week {week}")
        
        try:
            pattern = SeasonalPattern(season=season, week=week)
            
            # Get league-wide statistics for this week
            league_query = text("""
                SELECT 
                    AVG(fantasy_points_ppr) as avg_points,
                    COUNT(*) as total_performances,
                    COUNT(CASE WHEN fantasy_points_ppr >= 20 THEN 1 END) as high_scoring,
                    COUNT(CASE WHEN fantasy_points_ppr < 8 THEN 1 END) as low_scoring
                FROM player_game_stats
                WHERE created_at >= date('now', '-7 days')
                AND fantasy_points_ppr > 0
            """)
            
            league_result = self.session.execute(league_query).fetchone()
            
            if league_result:
                pattern.avg_points_per_game = float(league_result[0] or 0)
                total_performances = int(league_result[1] or 1)
                pattern.high_scoring_game_frequency = int(league_result[2] or 0) / total_performances
                pattern.defensive_game_frequency = int(league_result[3] or 0) / total_performances
            
            # Position-specific trends (simplified)
            position_query = text("""
                SELECT 
                    p.position,
                    AVG(pgs.fantasy_points_ppr) as avg_points
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                WHERE pgs.created_at >= date('now', '-30 days')
                AND pgs.fantasy_points_ppr > 0
                AND p.position IN ('QB', 'RB', 'WR', 'TE')
                GROUP BY p.position
            """)
            
            position_results = self.session.execute(position_query).fetchall()
            
            for result in position_results:
                position = result[0]
                avg_points = float(result[1] or 0)
                
                if position == 'QB':
                    pattern.qb_performance_trend = avg_points
                elif position == 'RB':
                    pattern.rb_usage_trend = avg_points
                elif position == 'WR':
                    pattern.wr_target_share_trend = avg_points
                elif position == 'TE':
                    pattern.te_involvement_trend = avg_points
            
            logger.info(f"Seasonal pattern analysis complete: {pattern.avg_points_per_game:.1f} avg points")
            return pattern
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
            return SeasonalPattern(season=season, week=week)
    
    def get_trend_based_multipliers(self, player_trend: PlayerTrend) -> Dict[str, float]:
        """Get projection multipliers based on trend analysis"""
        
        multipliers = {
            'base': 1.0,
            'form': 1.0,
            'trend': 1.0,
            'consistency': 1.0,
            'situation': 1.0
        }
        
        # Form-based multiplier
        if player_trend.recent_form_score > 80:
            multipliers['form'] = 1.15
        elif player_trend.recent_form_score > 60:
            multipliers['form'] = 1.05
        elif player_trend.recent_form_score < 30:
            multipliers['form'] = 0.85
        elif player_trend.recent_form_score < 50:
            multipliers['form'] = 0.95
        
        # Trend-based multiplier
        if player_trend.seasonal_trend == "improving":
            multipliers['trend'] = 1.1
        elif player_trend.seasonal_trend == "declining":
            multipliers['trend'] = 0.9
        elif player_trend.seasonal_trend == "volatile":
            multipliers['trend'] = 0.95
        
        # Consistency multiplier (more consistent = slight boost)
        if player_trend.consistency_score > 0.7:
            multipliers['consistency'] = 1.05
        elif player_trend.consistency_score < 0.3:
            multipliers['consistency'] = 0.95
        
        return multipliers
    
    def identify_breakout_candidates(self, position: str, min_games: int = 5) -> List[Dict[str, Any]]:
        """Identify players with high breakout potential"""
        
        try:
            query = text("""
                SELECT 
                    pgs.player_id,
                    p.name,
                    p.position,
                    AVG(pgs.fantasy_points_ppr) as avg_points,
                    COUNT(*) as games,
                    AVG(pgs.snap_percentage) as avg_snap_pct
                FROM player_game_stats pgs
                JOIN players p ON pgs.player_id = p.player_id
                WHERE p.position = :position
                AND pgs.created_at >= date('now', '-60 days')
                AND pgs.fantasy_points_ppr > 0
                GROUP BY pgs.player_id, p.name, p.position
                HAVING COUNT(*) >= :min_games
                AND AVG(pgs.fantasy_points_ppr) BETWEEN 8 AND 15  -- Breakout range
                AND AVG(pgs.snap_percentage) > 50  -- Significant usage
                ORDER BY AVG(pgs.snap_percentage) DESC, AVG(pgs.fantasy_points_ppr) DESC
                LIMIT 10
            """)
            
            results = self.session.execute(query, {
                "position": position,
                "min_games": min_games
            }).fetchall()
            
            candidates = []
            for result in results:
                candidates.append({
                    'player_id': result[0],
                    'name': result[1],
                    'position': result[2],
                    'avg_points': float(result[3]),
                    'games': int(result[4]),
                    'avg_snap_percentage': float(result[5] or 0),
                    'breakout_score': float(result[5] or 0) * 0.6 + float(result[3]) * 0.4
                })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error identifying breakout candidates: {e}")
            return []
