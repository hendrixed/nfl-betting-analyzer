"""
NFL Statistical Computing Engine
Advanced statistical analysis and computation for NFL betting predictions.
Tasks 126-139: Statistical modeling, trend analysis, performance metrics, predictive features.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..database_models import Player, Team, Game, PlayerGameStats, get_db_session

logger = logging.getLogger(__name__)


@dataclass
class PlayerTrends:
    """Player performance trends and statistics"""
    player_id: str
    games_analyzed: int
    avg_fantasy_points: float
    trend_direction: str  # 'improving', 'declining', 'stable'
    consistency_score: float
    ceiling: float  # 90th percentile performance
    floor: float    # 10th percentile performance
    recent_form: float  # Last 4 games average
    matchup_difficulty: float


@dataclass
class TeamAnalytics:
    """Team-level analytics and metrics"""
    team_code: str
    offensive_efficiency: float
    defensive_efficiency: float
    pace_factor: float
    red_zone_efficiency: float
    turnover_differential: float
    home_field_advantage: float


@dataclass
class GameContext:
    """Game situational context and factors"""
    game_id: str
    weather_impact: float
    divisional_game: bool
    prime_time: bool
    rest_days_home: int
    rest_days_away: int
    playoff_implications: bool
    revenge_game: bool


class NFLStatisticalComputingEngine:
    """Advanced statistical computing engine for NFL analysis"""
    
    def __init__(self, session: Session):
        self.session = session
        self.current_season = 2025
        
        # Statistical parameters
        self.trend_window = 8  # Games to analyze for trends
        self.recent_form_window = 4  # Games for recent form
        self.min_games_for_analysis = 3
        
        # Weights for composite scores
        self.consistency_weights = {
            'variance': 0.4,
            'coefficient_variation': 0.3,
            'boom_bust_ratio': 0.3
        }
        
        # Position-specific scoring expectations
        self.position_expectations = {
            'QB': {'avg': 18.5, 'ceiling': 35.0, 'floor': 8.0},
            'RB': {'avg': 12.8, 'ceiling': 28.0, 'floor': 3.0},
            'WR': {'avg': 11.2, 'ceiling': 25.0, 'floor': 2.0},
            'TE': {'avg': 8.9, 'ceiling': 20.0, 'floor': 1.5}
        }
    
    def calculate_player_trends(self, player_id: str, games_back: int = None) -> PlayerTrends:
        """Calculate comprehensive player performance trends"""
        if games_back is None:
            games_back = self.trend_window
        
        # Get recent player stats
        stats = self.session.query(PlayerGameStats).filter(
            PlayerGameStats.player_id == player_id
        ).order_by(PlayerGameStats.game_date.desc()).limit(games_back).all()
        
        if len(stats) < self.min_games_for_analysis:
            return self._default_player_trends(player_id)
        
        # Extract fantasy points
        fantasy_points = [stat.fantasy_points_ppr for stat in stats if stat.fantasy_points_ppr is not None]
        
        if not fantasy_points:
            return self._default_player_trends(player_id)
        
        # Calculate basic statistics
        avg_points = np.mean(fantasy_points)
        std_points = np.std(fantasy_points)
        
        # Calculate percentiles
        ceiling = np.percentile(fantasy_points, 90)
        floor = np.percentile(fantasy_points, 10)
        
        # Calculate recent form (last 4 games)
        recent_games = min(self.recent_form_window, len(fantasy_points))
        recent_form = np.mean(fantasy_points[:recent_games])
        
        # Determine trend direction
        if len(fantasy_points) >= 6:
            first_half = np.mean(fantasy_points[len(fantasy_points)//2:])
            second_half = np.mean(fantasy_points[:len(fantasy_points)//2])
            
            if second_half > first_half * 1.1:
                trend_direction = 'improving'
            elif second_half < first_half * 0.9:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'stable'
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(fantasy_points)
        
        # Calculate matchup difficulty (placeholder)
        matchup_difficulty = 0.5  # Would integrate with defensive rankings
        
        return PlayerTrends(
            player_id=player_id,
            games_analyzed=len(fantasy_points),
            avg_fantasy_points=avg_points,
            trend_direction=trend_direction,
            consistency_score=consistency_score,
            ceiling=ceiling,
            floor=floor,
            recent_form=recent_form,
            matchup_difficulty=matchup_difficulty
        )
    
    def calculate_team_analytics(self, team_code: str) -> TeamAnalytics:
        """Calculate comprehensive team analytics"""
        
        # Get team's recent games
        recent_games = self.session.query(Game).filter(
            or_(Game.home_team == team_code, Game.away_team == team_code),
            Game.season == self.current_season
        ).order_by(Game.game_date.desc()).limit(8).all()
        
        if not recent_games:
            return self._default_team_analytics(team_code)
        
        # Calculate offensive efficiency
        total_points_scored = 0
        total_points_allowed = 0
        home_games = 0
        
        for game in recent_games:
            if game.home_team == team_code:
                if game.home_score is not None:
                    total_points_scored += game.home_score
                    total_points_allowed += game.away_score or 0
                    home_games += 1
            else:
                if game.away_score is not None:
                    total_points_scored += game.away_score
                    total_points_allowed += game.home_score or 0
        
        games_played = len([g for g in recent_games if g.game_status == 'completed'])
        
        if games_played > 0:
            offensive_efficiency = total_points_scored / games_played
            defensive_efficiency = 35.0 - (total_points_allowed / games_played)  # Inverse scoring
        else:
            offensive_efficiency = 21.0  # League average
            defensive_efficiency = 21.0
        
        # Calculate other metrics (placeholders for now)
        pace_factor = 1.0  # Would calculate from play counts
        red_zone_efficiency = 0.6  # Would calculate from red zone data
        turnover_differential = 0.0  # Would calculate from turnover data
        home_field_advantage = 2.5 if home_games > 0 else 0.0
        
        return TeamAnalytics(
            team_code=team_code,
            offensive_efficiency=offensive_efficiency,
            defensive_efficiency=defensive_efficiency,
            pace_factor=pace_factor,
            red_zone_efficiency=red_zone_efficiency,
            turnover_differential=turnover_differential,
            home_field_advantage=home_field_advantage
        )
    
    def analyze_game_context(self, game_id: str) -> GameContext:
        """Analyze situational context for a game"""
        
        game = self.session.query(Game).filter(Game.game_id == game_id).first()
        if not game:
            return self._default_game_context(game_id)
        
        # Calculate rest days (placeholder logic)
        rest_days_home = 7  # Standard week
        rest_days_away = 7
        
        # Determine if divisional game
        divisional_game = self._is_divisional_matchup(game.home_team, game.away_team)
        
        # Determine if prime time
        prime_time = False
        if game.game_date:
            hour = game.game_date.hour
            prime_time = hour >= 20 or hour <= 1  # 8 PM or later, or Monday night
        
        # Other contextual factors (placeholders)
        weather_impact = 0.0  # Would integrate weather data
        playoff_implications = False  # Would analyze standings
        revenge_game = False  # Would track previous matchups
        
        return GameContext(
            game_id=game_id,
            weather_impact=weather_impact,
            divisional_game=divisional_game,
            prime_time=prime_time,
            rest_days_home=rest_days_home,
            rest_days_away=rest_days_away,
            playoff_implications=playoff_implications,
            revenge_game=revenge_game
        )
    
    def calculate_position_rankings(self, position: str, scoring_type: str = 'ppr') -> List[Dict]:
        """Calculate position rankings for fantasy purposes"""
        
        # Map scoring type to database field
        scoring_field_map = {
            'ppr': PlayerGameStats.fantasy_points_ppr,
            'standard': PlayerGameStats.fantasy_points_standard,
            'half_ppr': PlayerGameStats.fantasy_points_half_ppr
        }
        
        scoring_field = scoring_field_map.get(scoring_type, PlayerGameStats.fantasy_points_ppr)
        
        # Get average points per game for position
        rankings = self.session.query(
            Player.player_id,
            Player.name,
            Player.current_team,
            func.avg(scoring_field).label('avg_points'),
            func.count(PlayerGameStats.game_id).label('games_played')
        ).join(PlayerGameStats).filter(
            Player.position == position,
            Player.is_active == True,
            PlayerGameStats.season == self.current_season
        ).group_by(
            Player.player_id, Player.name, Player.current_team
        ).having(
            func.count(PlayerGameStats.game_id) >= self.min_games_for_analysis
        ).order_by(
            func.avg(scoring_field).desc()
        ).limit(50).all()
        
        # Convert to list of dictionaries
        result = []
        for rank, (player_id, name, team, avg_points, games) in enumerate(rankings, 1):
            result.append({
                'rank': rank,
                'player_id': player_id,
                'name': name,
                'team': team,
                'avg_points': round(float(avg_points), 2),
                'games_played': games,
                'total_points': round(float(avg_points) * games, 2)
            })
        
        return result
    
    def calculate_strength_of_schedule(self, team_code: str) -> Dict[str, float]:
        """Calculate strength of schedule metrics"""
        
        # Get team's opponents
        games = self.session.query(Game).filter(
            or_(Game.home_team == team_code, Game.away_team == team_code),
            Game.season == self.current_season
        ).all()
        
        opponents = []
        for game in games:
            opponent = game.away_team if game.home_team == team_code else game.home_team
            opponents.append(opponent)
        
        if not opponents:
            return {'overall': 0.5, 'remaining': 0.5, 'played': 0.5}
        
        # Calculate opponent strength (simplified)
        opponent_strengths = []
        for opponent in opponents:
            # Get opponent's record or use placeholder
            strength = 0.5  # Would calculate from wins/losses
            opponent_strengths.append(strength)
        
        overall_sos = np.mean(opponent_strengths)
        
        return {
            'overall': overall_sos,
            'remaining': overall_sos,  # Would separate played vs remaining
            'played': overall_sos
        }
    
    def generate_advanced_metrics(self, player_id: str) -> Dict[str, Any]:
        """Generate advanced metrics for a player"""
        
        # Get player and recent stats
        player = self.session.query(Player).filter(Player.player_id == player_id).first()
        if not player:
            return {}
        
        stats = self.session.query(PlayerGameStats).filter(
            PlayerGameStats.player_id == player_id,
            PlayerGameStats.season == self.current_season
        ).order_by(PlayerGameStats.game_date.desc()).all()
        
        if not stats:
            return {}
        
        metrics = {
            'player_id': player_id,
            'position': player.position,
            'games_played': len(stats)
        }
        
        # Position-specific metrics
        if player.position == 'QB':
            metrics.update(self._calculate_qb_metrics(stats))
        elif player.position == 'RB':
            metrics.update(self._calculate_rb_metrics(stats))
        elif player.position in ['WR', 'TE']:
            metrics.update(self._calculate_receiver_metrics(stats))
        
        return metrics
    
    def _calculate_consistency_score(self, fantasy_points: List[float]) -> float:
        """Calculate consistency score based on variance and boom/bust ratio"""
        if len(fantasy_points) < 2:
            return 0.5
        
        # Calculate coefficient of variation
        mean_points = np.mean(fantasy_points)
        std_points = np.std(fantasy_points)
        cv = std_points / mean_points if mean_points > 0 else 1.0
        
        # Calculate boom/bust ratio
        boom_threshold = mean_points * 1.5
        bust_threshold = mean_points * 0.5
        
        booms = sum(1 for p in fantasy_points if p >= boom_threshold)
        busts = sum(1 for p in fantasy_points if p <= bust_threshold)
        
        boom_bust_ratio = booms / max(busts, 1)
        
        # Combine metrics (lower CV and higher boom/bust ratio = more consistent)
        consistency = (1 / (1 + cv)) * 0.7 + min(boom_bust_ratio / 2, 1.0) * 0.3
        
        return min(consistency, 1.0)
    
    def _calculate_qb_metrics(self, stats: List[PlayerGameStats]) -> Dict[str, float]:
        """Calculate QB-specific advanced metrics"""
        total_attempts = sum(s.passing_attempts for s in stats if s.passing_attempts)
        total_completions = sum(s.passing_completions for s in stats if s.passing_completions)
        total_yards = sum(s.passing_yards for s in stats if s.passing_yards)
        total_tds = sum(s.passing_touchdowns for s in stats if s.passing_touchdowns)
        total_ints = sum(s.passing_interceptions for s in stats if s.passing_interceptions)
        
        if total_attempts == 0:
            return {}
        
        return {
            'completion_percentage': (total_completions / total_attempts) * 100,
            'yards_per_attempt': total_yards / total_attempts,
            'td_percentage': (total_tds / total_attempts) * 100,
            'int_percentage': (total_ints / total_attempts) * 100,
            'passer_rating': self._calculate_passer_rating(
                total_attempts, total_completions, total_yards, total_tds, total_ints
            )
        }
    
    def _calculate_rb_metrics(self, stats: List[PlayerGameStats]) -> Dict[str, float]:
        """Calculate RB-specific advanced metrics"""
        total_carries = sum(s.rushing_attempts for s in stats if s.rushing_attempts)
        total_yards = sum(s.rushing_yards for s in stats if s.rushing_yards)
        total_tds = sum(s.rushing_touchdowns for s in stats if s.rushing_touchdowns)
        total_targets = sum(s.targets for s in stats if s.targets)
        total_receptions = sum(s.receptions for s in stats if s.receptions)
        
        metrics = {}
        
        if total_carries > 0:
            metrics['yards_per_carry'] = total_yards / total_carries
            metrics['td_rate'] = total_tds / total_carries
        
        if total_targets > 0:
            metrics['target_share'] = total_targets / len(stats)  # Per game
            metrics['catch_rate'] = (total_receptions / total_targets) * 100
        
        return metrics
    
    def _calculate_receiver_metrics(self, stats: List[PlayerGameStats]) -> Dict[str, float]:
        """Calculate WR/TE-specific advanced metrics"""
        total_targets = sum(s.targets for s in stats if s.targets)
        total_receptions = sum(s.receptions for s in stats if s.receptions)
        total_yards = sum(s.receiving_yards for s in stats if s.receiving_yards)
        total_tds = sum(s.receiving_touchdowns for s in stats if s.receiving_touchdowns)
        
        metrics = {}
        
        if total_targets > 0:
            metrics['catch_rate'] = (total_receptions / total_targets) * 100
            metrics['targets_per_game'] = total_targets / len(stats)
        
        if total_receptions > 0:
            metrics['yards_per_reception'] = total_yards / total_receptions
            metrics['td_rate'] = total_tds / total_receptions
        
        return metrics
    
    def _calculate_passer_rating(self, attempts: int, completions: int, yards: int, tds: int, ints: int) -> float:
        """Calculate NFL passer rating"""
        if attempts == 0:
            return 0.0
        
        # NFL passer rating formula
        comp_pct = (completions / attempts - 0.3) * 5
        yards_per_att = (yards / attempts - 3) * 0.25
        td_pct = (tds / attempts) * 20
        int_pct = 2.375 - (ints / attempts * 25)
        
        # Clamp values
        comp_pct = max(0, min(2.375, comp_pct))
        yards_per_att = max(0, min(2.375, yards_per_att))
        td_pct = max(0, min(2.375, td_pct))
        int_pct = max(0, min(2.375, int_pct))
        
        rating = ((comp_pct + yards_per_att + td_pct + int_pct) / 6) * 100
        
        return round(rating, 1)
    
    def _is_divisional_matchup(self, team1: str, team2: str) -> bool:
        """Check if two teams are in the same division"""
        divisions = {
            'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC West': ['ARI', 'LAR', 'SF', 'SEA']
        }
        
        for division_teams in divisions.values():
            if team1 in division_teams and team2 in division_teams:
                return True
        
        return False
    
    def _default_player_trends(self, player_id: str) -> PlayerTrends:
        """Return default player trends when insufficient data"""
        return PlayerTrends(
            player_id=player_id,
            games_analyzed=0,
            avg_fantasy_points=0.0,
            trend_direction='stable',
            consistency_score=0.5,
            ceiling=0.0,
            floor=0.0,
            recent_form=0.0,
            matchup_difficulty=0.5
        )
    
    def _default_team_analytics(self, team_code: str) -> TeamAnalytics:
        """Return default team analytics when insufficient data"""
        return TeamAnalytics(
            team_code=team_code,
            offensive_efficiency=21.0,
            defensive_efficiency=21.0,
            pace_factor=1.0,
            red_zone_efficiency=0.6,
            turnover_differential=0.0,
            home_field_advantage=2.5
        )
    
    def _default_game_context(self, game_id: str) -> GameContext:
        """Return default game context when insufficient data"""
        return GameContext(
            game_id=game_id,
            weather_impact=0.0,
            divisional_game=False,
            prime_time=False,
            rest_days_home=7,
            rest_days_away=7,
            playoff_implications=False,
            revenge_game=False
        )


def main():
    """Test the statistical computing engine"""
    session = get_db_session()
    engine = NFLStatisticalComputingEngine(session)
    
    try:
        # Test player trends
        players = session.query(Player).filter(Player.is_active == True).limit(5).all()
        
        for player in players:
            trends = engine.calculate_player_trends(player.player_id)
            print(f"\nPlayer: {player.name} ({player.position})")
            print(f"  Games Analyzed: {trends.games_analyzed}")
            print(f"  Avg Fantasy Points: {trends.avg_fantasy_points:.2f}")
            print(f"  Trend: {trends.trend_direction}")
            print(f"  Consistency: {trends.consistency_score:.2f}")
        
        # Test position rankings
        qb_rankings = engine.calculate_position_rankings('QB')
        print(f"\nTop 5 QBs:")
        for qb in qb_rankings[:5]:
            print(f"  {qb['rank']}. {qb['name']} ({qb['team']}): {qb['avg_points']} pts/game")
        
    finally:
        session.close()


if __name__ == "__main__":
    main()
