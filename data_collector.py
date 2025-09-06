"""
NFL Data Collection System
Comprehensive data collection from multiple sources for player and game predictions.
"""

import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any
from sqlalchemy import create_engine, select, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert
import nfl_data_py as nfl
import requests
from dataclasses import dataclass
from pathlib import Path
import json
import time

# Import our models
from database_models import (
    Player, Team, Game, PlayerGameStats, BettingLine,
    Base, create_all_tables
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class DataCollectionConfig:
    """Configuration for data collection."""
    database_url: str
    api_keys: Dict[str, str]
    data_sources: Dict[str, bool]
    seasons: List[int]
    current_season: int
    current_week: int
    enable_live_data: bool = True
    batch_size: int = 1000
    rate_limit_delay: float = 1.0


class NFLDataCollector:
    """Main data collection orchestrator."""
    
    def __init__(self, config: DataCollectionConfig):
        """Initialize the data collector."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        self.engine = create_engine(config.database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize collectors
        self.player_collector = PlayerDataCollector(self)
        self.game_collector = GameDataCollector(self)
        self.stats_collector = StatsDataCollector(self)
        self.betting_collector = BettingDataCollector(self)
        
        # NFL team mapping
        self.nfl_teams = {
            'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons',
            'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
            'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
            'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
            'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos',
            'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
            'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts',
            'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
            'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams',
            'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
            'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots',
            'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
            'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles',
            'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks',
            'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers',
            'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
        }
        
    async def initialize_database(self):
        """Initialize database tables."""
        try:
            create_all_tables(self.engine)
            await self._populate_teams()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
            
    async def _populate_teams(self):
        """Populate teams table with NFL teams."""
        with self.Session() as session:
            for team_id, team_name in self.nfl_teams.items():
                # Check if team exists
                existing_team = session.query(Team).filter(Team.team_id == team_id).first()
                if not existing_team:
                    # Extract city and name
                    parts = team_name.rsplit(' ', 1)
                    city = parts[0] if len(parts) > 1 else team_name
                    name = parts[1] if len(parts) > 1 else team_name
                    
                    # Determine conference and division (simplified)
                    conference, division = self._get_team_division(team_id)
                    
                    team = Team(
                        team_id=team_id,
                        team_name=name,
                        city=city,
                        conference=conference,
                        division=division
                    )
                    session.add(team)
            
            session.commit()
            
    def _get_team_division(self, team_id: str) -> tuple[str, str]:
        """Get conference and division for a team."""
        divisions = {
            'AFC': {
                'North': ['BAL', 'CIN', 'CLE', 'PIT'],
                'South': ['HOU', 'IND', 'JAX', 'TEN'],
                'East': ['BUF', 'MIA', 'NE', 'NYJ'],
                'West': ['DEN', 'KC', 'LAC', 'LV']
            },
            'NFC': {
                'North': ['CHI', 'DET', 'GB', 'MIN'],
                'South': ['ATL', 'CAR', 'NO', 'TB'],
                'East': ['DAL', 'NYG', 'PHI', 'WAS'],
                'West': ['ARI', 'LAR', 'SF', 'SEA']
            }
        }
        
        for conf, divs in divisions.items():
            for div, teams in divs.items():
                if team_id in teams:
                    return conf, div
        return 'NFC', 'Unknown'  # Default
        
    async def run_full_collection(self):
        """Run complete data collection pipeline."""
        try:
            self.logger.info("Starting full data collection...")
            
            # Initialize database
            await self.initialize_database()
            
            # Collect historical data
            if self.config.seasons:
                for season in self.config.seasons:
                    await self.collect_season_data(season)
            
            # Collect current season data
            await self.collect_current_season()
            
            # Collect live data if enabled
            if self.config.enable_live_data:
                await self.collect_live_data()
                
            self.logger.info("Full data collection completed successfully")
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            raise
            
    async def collect_season_data(self, season: int):
        """Collect all data for a specific season."""
        self.logger.info(f"Collecting data for {season} season...")
        
        # Collect in order of dependencies
        await self.player_collector.collect_players(season)
        await self.game_collector.collect_schedule(season)
        await self.stats_collector.collect_weekly_stats(season)
        
        self.logger.info(f"Season {season} data collection completed")
        
    async def collect_current_season(self):
        """Collect current season data up to current week."""
        season = self.config.current_season
        self.logger.info(f"Collecting current season data: {season}")
        
        await self.collect_season_data(season)
        
    async def collect_live_data(self):
        """Collect live/current data."""
        self.logger.info("Collecting live data...")
        
        # Collect current week betting lines
        await self.betting_collector.collect_current_lines()
        
        # Update player statuses/injuries
        await self.player_collector.update_player_status()
        
        self.logger.info("Live data collection completed")


class PlayerDataCollector:
    """Collects player information and roster data."""
    
    def __init__(self, main_collector: NFLDataCollector):
        self.main = main_collector
        self.logger = logging.getLogger(f"{__name__}.PlayerCollector")
        
    async def collect_players(self, season: int):
        """Collect player data for a season."""
        try:
            self.logger.info(f"Collecting players for {season}")
            
            # Get roster data from nfl_data_py
            rosters = nfl.import_seasonal_rosters([season])
            
            # Process players
            await self._process_player_data(rosters, season)
            
        except Exception as e:
            self.logger.error(f"Error collecting players for {season}: {e}")
            raise
            
    async def _process_player_data(self, rosters: pd.DataFrame, season: int):
        """Process and store player data."""
        with self.main.Session() as session:
            for _, row in rosters.iterrows():
                try:
                    # Create player ID from name and position
                    player_id = self._generate_player_id(row)
                    
                    # Check if player exists
                    existing_player = session.query(Player).filter(
                        Player.player_id == player_id
                    ).first()
                    
                    if existing_player:
                        # Update existing player
                        self._update_player_fields(existing_player, row, season)
                    else:
                        # Create new player
                        player = self._create_player_from_row(row, player_id, season)
                        session.add(player)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing player {row.get('full_name', 'Unknown')}: {e}")
                    continue
                    
            session.commit()
            self.logger.info(f"Processed {len(rosters)} players for {season}")
            
    def _generate_player_id(self, row: pd.Series) -> str:
        """Generate consistent player ID."""
        name = str(row.get('full_name', '')).lower().replace(' ', '_').replace('.', '')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        return f"{name}_{row.get('position', 'unknown').lower()}"
        
    def _create_player_from_row(self, row: pd.Series, player_id: str, season: int) -> Player:
        """Create Player object from roster row."""
        return Player(
            player_id=player_id,
            name=str(row.get('full_name', '')),
            first_name=str(row.get('first_name', '')),
            last_name=str(row.get('last_name', '')),
            position=str(row.get('position', '')),
            current_team=str(row.get('team', '')),
            height_inches=self._convert_height(row.get('height')),
            weight_lbs=int(row.get('weight', 0)) if pd.notna(row.get('weight')) else None,
            draft_year=int(row.get('draft_year', 0)) if pd.notna(row.get('draft_year')) else None,
            draft_round=int(row.get('draft_round', 0)) if pd.notna(row.get('draft_round')) else None,
            draft_pick=int(row.get('draft_pick', 0)) if pd.notna(row.get('draft_pick')) else None,
            college=str(row.get('college', '')) if pd.notna(row.get('college')) else None,
            years_experience=int(row.get('years_exp', 0)) if pd.notna(row.get('years_exp')) else None,
            espn_id=str(row.get('espn_id', '')) if pd.notna(row.get('espn_id')) else None,
            yahoo_id=str(row.get('yahoo_id', '')) if pd.notna(row.get('yahoo_id')) else None,
            pfr_id=str(row.get('pfr_id', '')) if pd.notna(row.get('pfr_id')) else None,
            is_active=True
        )
        
    def _update_player_fields(self, player: Player, row: pd.Series, season: int):
        """Update existing player with new information."""
        player.current_team = str(row.get('team', player.current_team))
        player.updated_at = datetime.now()
        
        # Update other fields if they're missing
        if not player.height_inches and pd.notna(row.get('height')):
            player.height_inches = self._convert_height(row.get('height'))
            
        if not player.weight_lbs and pd.notna(row.get('weight')):
            player.weight_lbs = int(row.get('weight'))
            
    def _convert_height(self, height_str: str) -> Optional[int]:
        """Convert height string (e.g., '6-2') to inches."""
        try:
            if pd.isna(height_str) or not height_str:
                return None
            height_str = str(height_str)
            if '-' in height_str:
                feet, inches = height_str.split('-')
                return int(feet) * 12 + int(inches)
            return None
        except:
            return None
            
    async def update_player_status(self):
        """Update current player statuses and injuries."""
        # This would integrate with injury report APIs
        self.logger.info("Updating player statuses...")
        # Implementation would go here


class GameDataCollector:
    """Collects game schedules and results."""
    
    def __init__(self, main_collector: NFLDataCollector):
        self.main = main_collector
        self.logger = logging.getLogger(f"{__name__}.GameCollector")
        
    async def collect_schedule(self, season: int):
        """Collect game schedule for a season."""
        try:
            self.logger.info(f"Collecting schedule for {season}")
            
            # Get schedule data from nfl_data_py
            schedule = nfl.import_schedules([season])
            
            await self._process_schedule_data(schedule, season)
            
        except Exception as e:
            self.logger.error(f"Error collecting schedule for {season}: {e}")
            raise
            
    async def _process_schedule_data(self, schedule: pd.DataFrame, season: int):
        """Process and store schedule data."""
        with self.main.Session() as session:
            for _, row in schedule.iterrows():
                try:
                    game_id = str(row.get('game_id', ''))
                    if not game_id:
                        continue
                        
                    # Check if game exists
                    existing_game = session.query(Game).filter(
                        Game.game_id == game_id
                    ).first()
                    
                    if existing_game:
                        # Update existing game
                        self._update_game_fields(existing_game, row)
                    else:
                        # Create new game
                        game = self._create_game_from_row(row, season)
                        session.add(game)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing game {row.get('game_id', 'Unknown')}: {e}")
                    continue
                    
            session.commit()
            self.logger.info(f"Processed {len(schedule)} games for {season}")
            
    def _create_game_from_row(self, row: pd.Series, season: int) -> Game:
        """Create Game object from schedule row."""
        return Game(
            game_id=str(row.get('game_id', '')),
            season=season,
            week=int(row.get('week', 0)),
            game_type=str(row.get('game_type', 'REG')),
            game_date=pd.to_datetime(row.get('gameday')).date() if pd.notna(row.get('gameday')) else date.today(),
            home_team=str(row.get('home_team', '')),
            away_team=str(row.get('away_team', '')),
            home_score=int(row.get('home_score', 0)) if pd.notna(row.get('home_score')) else None,
            away_score=int(row.get('away_score', 0)) if pd.notna(row.get('away_score')) else None,
            game_status='completed' if pd.notna(row.get('home_score')) else 'scheduled',
            stadium=str(row.get('location', '')) if pd.notna(row.get('location')) else None,
            is_overtime=bool(row.get('overtime', False))
        )
        
    def _update_game_fields(self, game: Game, row: pd.Series):
        """Update existing game with new information."""
        # Update scores if available
        if pd.notna(row.get('home_score')):
            game.home_score = int(row.get('home_score'))
            game.away_score = int(row.get('away_score', 0))
            game.game_status = 'completed'
            
        game.updated_at = datetime.now()


class StatsDataCollector:
    """Collects player statistics."""
    
    def __init__(self, main_collector: NFLDataCollector):
        self.main = main_collector
        self.logger = logging.getLogger(f"{__name__}.StatsCollector")
        
    async def collect_weekly_stats(self, season: int):
        """Collect weekly player statistics."""
        try:
            self.logger.info(f"Collecting weekly stats for {season}")
            
            # Get weekly stats from nfl_data_py
            weekly_stats = nfl.import_weekly_data([season])
            
            await self._process_weekly_stats(weekly_stats, season)
            
        except Exception as e:
            self.logger.error(f"Error collecting weekly stats for {season}: {e}")
            raise
            
    async def _process_weekly_stats(self, stats: pd.DataFrame, season: int):
        """Process and store weekly statistics."""
        with self.main.Session() as session:
            for _, row in stats.iterrows():
                try:
                    # Generate player ID and game ID
                    player_id = self._get_player_id_from_stats(row)
                    game_id = self._get_game_id_from_stats(row, season)
                    
                    if not player_id or not game_id:
                        continue
                        
                    # Check if stats exist
                    existing_stats = session.query(PlayerGameStats).filter(
                        and_(
                            PlayerGameStats.player_id == player_id,
                            PlayerGameStats.game_id == game_id
                        )
                    ).first()
                    
                    if existing_stats:
                        # Update existing stats
                        self._update_stats_fields(existing_stats, row)
                    else:
                        # Create new stats
                        stats_obj = self._create_stats_from_row(row, player_id, game_id)
                        session.add(stats_obj)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing stats for player {row.get('player_name', 'Unknown')}: {e}")
                    continue
                    
            session.commit()
            self.logger.info(f"Processed {len(stats)} stat records for {season}")
            
    def _get_player_id_from_stats(self, row: pd.Series) -> Optional[str]:
        """Get player ID from stats row."""
        if pd.isna(row.get('player_name')) or pd.isna(row.get('position')):
            return None
            
        name = str(row.get('player_name', '')).lower().replace(' ', '_').replace('.', '')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        position = str(row.get('position', 'unknown')).lower()
        return f"{name}_{position}"
        
    def _get_game_id_from_stats(self, row: pd.Series, season: int) -> Optional[str]:
        """Generate game ID from stats row."""
        # This would need to match the game_id format from schedule data
        # For now, create a simple format
        week = row.get('week', 0)
        team = row.get('recent_team', '')
        opponent = row.get('opponent_team', '')
        
        if not all([week, team, opponent]):
            return None
            
        return f"{season}_{week:02d}_{team}_{opponent}"
        
    def _create_stats_from_row(self, row: pd.Series, player_id: str, game_id: str) -> PlayerGameStats:
        """Create PlayerGameStats object from stats row."""
        # Calculate fantasy points
        fantasy_std = self._calculate_fantasy_points(row, 'standard')
        fantasy_ppr = self._calculate_fantasy_points(row, 'ppr')
        fantasy_half_ppr = self._calculate_fantasy_points(row, 'half_ppr')
        
        return PlayerGameStats(
            player_id=player_id,
            game_id=game_id,
            team=str(row.get('recent_team', '')),
            opponent=str(row.get('opponent_team', '')),
            is_home=bool(row.get('is_home', False)),
            
            # Passing stats
            passing_attempts=int(row.get('attempts', 0)) if pd.notna(row.get('attempts')) else 0,
            passing_completions=int(row.get('completions', 0)) if pd.notna(row.get('completions')) else 0,
            passing_yards=int(row.get('passing_yards', 0)) if pd.notna(row.get('passing_yards')) else 0,
            passing_touchdowns=int(row.get('passing_tds', 0)) if pd.notna(row.get('passing_tds')) else 0,
            passing_interceptions=int(row.get('interceptions', 0)) if pd.notna(row.get('interceptions')) else 0,
            passing_sacks=int(row.get('sacks', 0)) if pd.notna(row.get('sacks')) else 0,
            passing_sack_yards=int(row.get('sack_yards', 0)) if pd.notna(row.get('sack_yards')) else 0,
            
            # Rushing stats
            rushing_attempts=int(row.get('carries', 0)) if pd.notna(row.get('carries')) else 0,
            rushing_yards=int(row.get('rushing_yards', 0)) if pd.notna(row.get('rushing_yards')) else 0,
            rushing_touchdowns=int(row.get('rushing_tds', 0)) if pd.notna(row.get('rushing_tds')) else 0,
            rushing_fumbles=int(row.get('rushing_fumbles', 0)) if pd.notna(row.get('rushing_fumbles')) else 0,
            rushing_first_downs=int(row.get('rushing_first_downs', 0)) if pd.notna(row.get('rushing_first_downs')) else 0,
            
            # Receiving stats
            targets=int(row.get('targets', 0)) if pd.notna(row.get('targets')) else 0,
            receptions=int(row.get('receptions', 0)) if pd.notna(row.get('receptions')) else 0,
            receiving_yards=int(row.get('receiving_yards', 0)) if pd.notna(row.get('receiving_yards')) else 0,
            receiving_touchdowns=int(row.get('receiving_tds', 0)) if pd.notna(row.get('receiving_tds')) else 0,
            receiving_fumbles=int(row.get('receiving_fumbles', 0)) if pd.notna(row.get('receiving_fumbles')) else 0,
            receiving_first_downs=int(row.get('receiving_first_downs', 0)) if pd.notna(row.get('receiving_first_downs')) else 0,
            
            # Fantasy points
            fantasy_points_standard=fantasy_std,
            fantasy_points_ppr=fantasy_ppr,
            fantasy_points_half_ppr=fantasy_half_ppr
        )
        
    def _calculate_fantasy_points(self, row: pd.Series, scoring_type: str) -> float:
        """Calculate fantasy points based on scoring type."""
        points = 0.0
        
        # Passing (1 point per 25 yards, 4 points per TD, -2 per INT)
        passing_yards = float(row.get('passing_yards', 0)) if pd.notna(row.get('passing_yards')) else 0
        passing_tds = float(row.get('passing_tds', 0)) if pd.notna(row.get('passing_tds')) else 0
        interceptions = float(row.get('interceptions', 0)) if pd.notna(row.get('interceptions')) else 0
        
        points += passing_yards * 0.04  # 1 point per 25 yards
        points += passing_tds * 4
        points += interceptions * -2
        
        # Rushing (1 point per 10 yards, 6 points per TD)
        rushing_yards = float(row.get('rushing_yards', 0)) if pd.notna(row.get('rushing_yards')) else 0
        rushing_tds = float(row.get('rushing_tds', 0)) if pd.notna(row.get('rushing_tds')) else 0
        
        points += rushing_yards * 0.1  # 1 point per 10 yards
        points += rushing_tds * 6
        
        # Receiving (1 point per 10 yards, 6 points per TD)
        receiving_yards = float(row.get('receiving_yards', 0)) if pd.notna(row.get('receiving_yards')) else 0
        receiving_tds = float(row.get('receiving_tds', 0)) if pd.notna(row.get('receiving_tds')) else 0
        receptions = float(row.get('receptions', 0)) if pd.notna(row.get('receptions')) else 0
        
        points += receiving_yards * 0.1  # 1 point per 10 yards  
        points += receiving_tds * 6
        
        # Reception bonuses based on scoring type
        if scoring_type == 'ppr':
            points += receptions * 1.0  # Full PPR
        elif scoring_type == 'half_ppr':
            points += receptions * 0.5  # Half PPR
            
        # Fumbles (-2 points)
        fumbles = float(row.get('fumbles_lost', 0)) if pd.notna(row.get('fumbles_lost')) else 0
        points += fumbles * -2
        
        return round(points, 2)
        
    def _update_stats_fields(self, stats: PlayerGameStats, row: pd.Series):
        """Update existing stats with new information."""
        # Update all stat fields
        # This would involve updating all the statistical fields
        stats.updated_at = datetime.now()


class BettingDataCollector:
    """Collects betting lines and props."""
    
    def __init__(self, main_collector: NFLDataCollector):
        self.main = main_collector
        self.logger = logging.getLogger(f"{__name__}.BettingCollector")
        
    async def collect_current_lines(self):
        """Collect current betting lines."""
        self.logger.info("Collecting current betting lines...")
        
        # This would integrate with betting APIs
        # For now, this is a placeholder
        pass


# Example usage
async def main():
    """Example usage of the data collection system."""
    
    # Configuration
    config = DataCollectionConfig(
        database_url="postgresql://user:password@localhost/nfl_predictions",
        api_keys={
            "sports_betting_api": "your_api_key_here",
            "weather_api": "your_weather_key_here"
        },
        data_sources={
            "nfl_data_py": True,
            "betting_apis": True,
            "weather_apis": True
        },
        seasons=[2020, 2021, 2022, 2023, 2024],
        current_season=2024,
        current_week=10,
        enable_live_data=True
    )
    
    # Initialize collector
    collector = NFLDataCollector(config)
    
    # Run collection
    await collector.run_full_collection()
    
    print("Data collection completed!")


if __name__ == "__main__":
    asyncio.run(main())