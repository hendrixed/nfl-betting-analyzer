"""
Game Prediction Engine

This module provides comprehensive game-level predictions including scores,
all player stats, and complete game analysis.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date

from simplified_database_models import Game, Player, PlayerGameStats
from real_time_nfl_system import RealTimeNFLSystem
from comprehensive_stats_engine import ComprehensiveStatsEngine, ComprehensivePlayerStats
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

@dataclass
class GameScorePrediction:
    """Predicted game score and analysis"""
    
    game_id: str
    home_team: str
    away_team: str
    game_date: date
    
    # Score Predictions
    home_score: float
    away_score: float
    total_points: float
    spread: float  # Positive means home team favored
    
    # Game Analysis
    pace_factor: float  # Expected plays per game
    game_script: str    # "high_scoring", "defensive", "balanced"
    weather_impact: str # "favorable", "neutral", "adverse"
    
    # Confidence Metrics
    score_confidence: float
    spread_confidence: float
    total_confidence: float
    
    # Key Factors
    key_matchups: List[str] = field(default_factory=list)
    injury_impacts: List[str] = field(default_factory=list)
    trends: List[str] = field(default_factory=list)

@dataclass
class GamePlayerPredictions:
    """All player predictions for a specific game"""
    
    game_info: GameScorePrediction
    home_roster_predictions: List[ComprehensivePlayerStats]
    away_roster_predictions: List[ComprehensivePlayerStats]
    
    # Game Totals
    total_passing_yards: float = 0.0
    total_rushing_yards: float = 0.0
    total_touchdowns: int = 0
    
    # Top Performers Projected
    top_fantasy_performers: List[Tuple[str, float]] = field(default_factory=list)
    top_passing_yards: List[Tuple[str, float]] = field(default_factory=list)
    top_rushing_yards: List[Tuple[str, float]] = field(default_factory=list)
    top_receiving_yards: List[Tuple[str, float]] = field(default_factory=list)

class GamePredictionEngine:
    """Engine for comprehensive game-level predictions"""
    
    def __init__(self, session: Session):
        self.session = session
        self.nfl_system = RealTimeNFLSystem()
        self.stats_engine = ComprehensiveStatsEngine(session)
    
    async def predict_complete_game(self, game_id: str) -> GamePlayerPredictions:
        """Generate complete game predictions with all players and score"""
        
        logger.info(f"Generating complete game prediction for {game_id}")
        
        # Get game info
        game = self.session.query(Game).filter(Game.game_id == game_id).first()
        
        if not game:
            raise ValueError(f"Game {game_id} not found")
        
        # Predict game score
        score_prediction = await self._predict_game_score(game)
        
        # Get all players for both teams
        home_players = await self._get_team_active_players(game.home_team)
        away_players = await self._get_team_active_players(game.away_team)
        
        # Generate predictions for all players
        home_predictions = []
        away_predictions = []
        
        for player in home_players:
            try:
                player_stats = self.stats_engine.get_comprehensive_player_stats(player.player_id)
                # Add game-specific adjustments
                player_stats = await self._adjust_for_game_context(player_stats, game, is_home=True)
                home_predictions.append(player_stats)
            except Exception as e:
                logger.warning(f"Error predicting {player.name}: {e}")
        
        for player in away_players:
            try:
                player_stats = self.stats_engine.get_comprehensive_player_stats(player.player_id)
                # Add game-specific adjustments
                player_stats = await self._adjust_for_game_context(player_stats, game, is_home=False)
                away_predictions.append(player_stats)
            except Exception as e:
                logger.warning(f"Error predicting {player.name}: {e}")
        
        # Create complete game prediction
        game_prediction = GamePlayerPredictions(
            game_info=score_prediction,
            home_roster_predictions=home_predictions,
            away_roster_predictions=away_predictions
        )
        
        # Calculate game totals and top performers
        game_prediction = self._calculate_game_totals(game_prediction)
        
        return game_prediction
    
    async def _predict_game_score(self, game: Game) -> GameScorePrediction:
        """Predict game score and analysis"""
        
        # Get team offensive and defensive stats
        home_offensive_avg = await self._get_team_offensive_average(game.home_team)
        away_offensive_avg = await self._get_team_offensive_average(game.away_team)
        home_defensive_avg = await self._get_team_defensive_average(game.home_team)
        away_defensive_avg = await self._get_team_defensive_average(game.away_team)
        
        # Calculate predicted scores
        # Home team score = (home offense + away defense weakness) + home field advantage
        home_score = (home_offensive_avg + (30 - away_defensive_avg)) * 0.5 + 2.5
        
        # Away team score = (away offense + home defense weakness)
        away_score = (away_offensive_avg + (30 - home_defensive_avg)) * 0.5
        
        # Apply realistic bounds
        home_score = max(10, min(50, home_score))
        away_score = max(10, min(50, away_score))
        
        total_points = home_score + away_score
        spread = home_score - away_score
        
        # Determine game script
        if total_points > 50:
            game_script = "high_scoring"
        elif total_points < 40:
            game_script = "defensive"
        else:
            game_script = "balanced"
        
        return GameScorePrediction(
            game_id=game.game_id,
            home_team=game.home_team,
            away_team=game.away_team,
            game_date=game.game_date,
            home_score=home_score,
            away_score=away_score,
            total_points=total_points,
            spread=spread,
            pace_factor=65.0,  # Average NFL plays per game
            game_script=game_script,
            weather_impact="neutral",
            score_confidence=0.7,
            spread_confidence=0.65,
            total_confidence=0.75,
            key_matchups=[f"{game.away_team} offense vs {game.home_team} defense"],
            trends=["Recent team form analysis"]
        )
    
    async def _get_team_offensive_average(self, team: str) -> float:
        """Get team's average offensive performance"""
        
        # Query recent team offensive stats
        # This is simplified - would calculate from actual team data
        return 24.0  # Average NFL points per game
    
    async def _get_team_defensive_average(self, team: str) -> float:
        """Get team's average defensive performance (points allowed)"""
        
        # Query recent team defensive stats
        # This is simplified - would calculate from actual team data
        return 22.0  # Average NFL points allowed per game
    
    async def _get_team_active_players(self, team: str) -> List[Player]:
        """Get all active players for a team"""
        
        return self.session.query(Player).filter(
            Player.current_team == team,
            Player.position.in_(['QB', 'RB', 'WR', 'TE']),
            Player.is_active == True
        ).all()
    
    async def _adjust_for_game_context(self, player_stats: ComprehensivePlayerStats, 
                                     game: Game, is_home: bool) -> ComprehensivePlayerStats:
        """Adjust player stats for specific game context"""
        
        # Home field advantage
        if is_home:
            adjustment_factor = 1.05
        else:
            adjustment_factor = 0.98
        
        # Apply adjustment to key stats
        player_stats.fantasy_points_ppr *= adjustment_factor
        player_stats.fantasy_points_standard *= adjustment_factor
        
        # Position-specific adjustments
        if player_stats.position == 'QB':
            player_stats.passing_yards = int(player_stats.passing_yards * adjustment_factor)
            player_stats.passing_touchdowns = player_stats.passing_touchdowns * adjustment_factor
        elif player_stats.position == 'RB':
            player_stats.rushing_yards = int(player_stats.rushing_yards * adjustment_factor)
            player_stats.receiving_yards = int(player_stats.receiving_yards * adjustment_factor)
        elif player_stats.position in ['WR', 'TE']:
            player_stats.receiving_yards = int(player_stats.receiving_yards * adjustment_factor)
            player_stats.receptions = int(player_stats.receptions * adjustment_factor)
        
        return player_stats
    
    def _calculate_game_totals(self, game_prediction: GamePlayerPredictions) -> GamePlayerPredictions:
        """Calculate game totals and identify top performers"""
        
        all_players = game_prediction.home_roster_predictions + game_prediction.away_roster_predictions
        
        # Calculate totals
        game_prediction.total_passing_yards = sum(p.passing_yards for p in all_players)
        game_prediction.total_rushing_yards = sum(p.rushing_yards for p in all_players)
        game_prediction.total_touchdowns = sum(
            p.passing_touchdowns + p.rushing_touchdowns + p.receiving_touchdowns 
            for p in all_players
        )
        
        # Find top performers
        game_prediction.top_fantasy_performers = sorted(
            [(p.name, p.fantasy_points_ppr) for p in all_players],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        game_prediction.top_passing_yards = sorted(
            [(p.name, p.passing_yards) for p in all_players if p.passing_yards > 0],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        game_prediction.top_rushing_yards = sorted(
            [(p.name, p.rushing_yards) for p in all_players if p.rushing_yards > 0],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        game_prediction.top_receiving_yards = sorted(
            [(p.name, p.receiving_yards) for p in all_players if p.receiving_yards > 0],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return game_prediction

async def main():
    """Test game prediction engine"""
    
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///nfl_predictions.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    game_engine = GamePredictionEngine(session)
    
    # Get a sample game
    game = session.query(Game).first()
    
    if game:
        print("🏈 COMPLETE GAME PREDICTION TEST")
        print("=" * 60)
        
        try:
            prediction = await game_engine.predict_complete_game(game.game_id)
            
            print(f"Game: {prediction.game_info.away_team} @ {prediction.game_info.home_team}")
            print(f"Predicted Score: {prediction.game_info.away_team} {prediction.game_info.away_score:.0f} - {prediction.game_info.home_team} {prediction.game_info.home_score:.0f}")
            print(f"Total Points: {prediction.game_info.total_points:.0f}")
            print(f"Spread: {prediction.game_info.home_team} {prediction.game_info.spread:+.1f}")
            print()
            
            print("🏆 TOP FANTASY PERFORMERS:")
            for i, (name, points) in enumerate(prediction.top_fantasy_performers[:3], 1):
                print(f"   {i}. {name}: {points:.1f} pts")
            
            print(f"\n📊 GAME TOTALS:")
            print(f"   Total Passing Yards: {prediction.total_passing_yards:.0f}")
            print(f"   Total Rushing Yards: {prediction.total_rushing_yards:.0f}")
            print(f"   Total Touchdowns: {prediction.total_touchdowns:.0f}")
            
        except Exception as e:
            print(f"❌ Error testing game prediction: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ No games found in database")

if __name__ == "__main__":
    asyncio.run(main())
