#!/usr/bin/env python3
"""
NFL Data Ingestion Adapters
Unified adapters for NFL data sources with caching, schema validation, and error handling.
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import aiohttp
import requests
import nfl_data_py as nfl
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Data Schemas
@dataclass
class PlayerRoster:
    """Schema for player roster data"""
    player_id: str
    name: str
    position: str
    team: str
    jersey_number: Optional[int]
    status: str  # active, inactive, injured_reserve, etc.
    depth_chart_rank: Optional[int]
    snap_percentage: Optional[float]
    last_updated: datetime

@dataclass
class GameSchedule:
    """Schema for game schedule data"""
    game_id: str
    season: int
    week: int
    game_date: datetime
    home_team: str
    away_team: str
    stadium: str
    weather_conditions: Optional[Dict[str, Any]]
    game_status: str  # scheduled, in_progress, completed, postponed

@dataclass
class PlayerStats:
    """Schema for player statistics"""
    player_id: str
    game_id: str
    week: int
    season: int
    team: str
    opponent: str
    position: str
    # Passing stats
    passing_attempts: Optional[int] = None
    passing_completions: Optional[int] = None
    passing_yards: Optional[int] = None
    passing_touchdowns: Optional[int] = None
    interceptions: Optional[int] = None
    # Rushing stats
    rushing_attempts: Optional[int] = None
    rushing_yards: Optional[int] = None
    rushing_touchdowns: Optional[int] = None
    # Receiving stats
    targets: Optional[int] = None
    receptions: Optional[int] = None
    receiving_yards: Optional[int] = None
    receiving_touchdowns: Optional[int] = None
    # Snap counts
    offensive_snaps: Optional[int] = None
    snap_percentage: Optional[float] = None

@dataclass
class InjuryReport:
    """Schema for injury reports"""
    player_id: str
    name: str
    team: str
    position: str
    injury_status: str  # out, doubtful, questionable, probable
    injury_description: str
    report_date: datetime
    week: int
    season: int

@dataclass
class WeatherData:
    """Schema for weather data"""
    game_id: str
    stadium: str
    temperature: Optional[float]
    humidity: Optional[float]
    wind_speed: Optional[float]
    wind_direction: Optional[str]
    precipitation: Optional[float]
    conditions: str  # clear, cloudy, rain, snow, etc.
    timestamp: datetime

@dataclass
class BettingLine:
    """Schema for betting lines"""
    game_id: str
    market_type: str  # spread, total, moneyline, player_props
    player_id: Optional[str]  # for player props
    line_value: float
    odds: int
    sportsbook: str
    timestamp: datetime

class CacheManager:
    """Manages data caching to disk"""
    
    def __init__(self, cache_dir: str = "data/snapshots"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, data_type: str, date_str: str) -> Path:
        """Get cache file path for specific data type and date"""
        date_dir = self.cache_dir / date_str
        date_dir.mkdir(exist_ok=True)
        return date_dir / f"{data_type}.parquet"
    
    def is_cached(self, data_type: str, date_str: str, max_age_hours: int = 6) -> bool:
        """Check if data is cached and not stale"""
        cache_path = self.get_cache_path(data_type, date_str)
        if not cache_path.exists():
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)
    
    def save_to_cache(self, data: pd.DataFrame, data_type: str, date_str: str) -> None:
        """Save data to cache"""
        cache_path = self.get_cache_path(data_type, date_str)
        try:
            data.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(data)} records to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to cache data to {cache_path}: {e}")
    
    def load_from_cache(self, data_type: str, date_str: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        cache_path = self.get_cache_path(data_type, date_str)
        try:
            if cache_path.exists():
                data = pd.read_parquet(cache_path)
                logger.info(f"Loaded {len(data)} records from cache: {cache_path}")
                return data
        except Exception as e:
            logger.error(f"Failed to load cache from {cache_path}: {e}")
        return None

class BaseAdapter(ABC):
    """Base class for data adapters"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.session = None
    
    @abstractmethod
    async def fetch_data(self, **kwargs) -> pd.DataFrame:
        """Fetch data from source"""
        pass
    
    def validate_schema(self, data: pd.DataFrame, schema_class) -> bool:
        """Validate data against schema"""
        try:
            # Check required columns exist
            required_fields = [field.name for field in schema_class.__dataclass_fields__.values()]
            missing_cols = set(required_fields) - set(data.columns)
            if missing_cols:
                logger.warning(f"Missing columns in {schema_class.__name__}: {missing_cols}")
                return False
            return True
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False

class NFLDataPyAdapter(BaseAdapter):
    """Adapter for nfl_data_py library"""
    
    async def fetch_rosters(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch roster data"""
        date_str = f"{season}-W{week or 'current'}"
        
        if self.cache_manager.is_cached("rosters", date_str):
            cached_data = self.cache_manager.load_from_cache("rosters", date_str)
            if cached_data is not None:
                return cached_data
        
        try:
            logger.info(f"Fetching roster data for {season}")
            rosters = nfl.import_rosters([season])
            
            if week:
                # Filter to specific week if provided
                rosters = rosters[rosters['week'] == week]
            
            # Normalize to our schema
            normalized_data = self._normalize_roster_data(rosters)
            
            # Cache the data
            self.cache_manager.save_to_cache(normalized_data, "rosters", date_str)
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch roster data: {e}")
            return pd.DataFrame()
    
    async def fetch_schedules(self, season: int) -> pd.DataFrame:
        """Fetch schedule data"""
        date_str = f"{season}-schedule"
        
        if self.cache_manager.is_cached("schedules", date_str, max_age_hours=24):
            cached_data = self.cache_manager.load_from_cache("schedules", date_str)
            if cached_data is not None:
                return cached_data
        
        try:
            logger.info(f"Fetching schedule data for {season}")
            schedules = nfl.import_schedules([season])
            
            # Normalize to our schema
            normalized_data = self._normalize_schedule_data(schedules)
            
            # Cache the data
            self.cache_manager.save_to_cache(normalized_data, "schedules", date_str)
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch schedule data: {e}")
            return pd.DataFrame()
    
    async def fetch_weekly_stats(self, season: int, week: int) -> pd.DataFrame:
        """Fetch weekly player statistics"""
        date_str = f"{season}-W{week}"
        
        if self.cache_manager.is_cached("weekly_stats", date_str):
            cached_data = self.cache_manager.load_from_cache("weekly_stats", date_str)
            if cached_data is not None:
                return cached_data
        
        try:
            logger.info(f"Fetching weekly stats for {season} Week {week}")
            weekly_data = nfl.import_weekly_data([season])
            weekly_data = weekly_data[weekly_data['week'] == week]
            
            # Normalize to our schema
            normalized_data = self._normalize_stats_data(weekly_data)
            
            # Cache the data
            self.cache_manager.save_to_cache(normalized_data, "weekly_stats", date_str)
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch weekly stats: {e}")
            return pd.DataFrame()
    
    async def fetch_snap_counts(self, season: int, week: int) -> pd.DataFrame:
        """Fetch snap count data"""
        date_str = f"{season}-W{week}-snaps"
        
        if self.cache_manager.is_cached("snap_counts", date_str):
            cached_data = self.cache_manager.load_from_cache("snap_counts", date_str)
            if cached_data is not None:
                return cached_data
        
        try:
            logger.info(f"Fetching snap counts for {season} Week {week}")
            snap_data = nfl.import_snap_counts([season])
            snap_data = snap_data[snap_data['week'] == week]
            
            # Normalize snap count data
            normalized_data = self._normalize_snap_data(snap_data)
            
            # Cache the data
            self.cache_manager.save_to_cache(normalized_data, "snap_counts", date_str)
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to fetch snap counts: {e}")
            return pd.DataFrame()
    
    def _normalize_roster_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize roster data to our schema"""
        try:
            normalized = pd.DataFrame({
                'player_id': raw_data.get('player_id', ''),
                'name': raw_data.get('player_name', ''),
                'position': raw_data.get('position', ''),
                'team': raw_data.get('team', ''),
                'jersey_number': raw_data.get('jersey_number'),
                'status': raw_data.get('status', 'active'),
                'depth_chart_rank': raw_data.get('depth_chart_order'),
                'snap_percentage': None,  # Will be filled from snap count data
                'last_updated': datetime.now()
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize roster data: {e}")
            return pd.DataFrame()
    
    def _normalize_schedule_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize schedule data to our schema"""
        try:
            normalized = pd.DataFrame({
                'game_id': raw_data.get('game_id', ''),
                'season': raw_data.get('season', 0),
                'week': raw_data.get('week', 0),
                'game_date': pd.to_datetime(raw_data.get('gameday', '')),
                'home_team': raw_data.get('home_team', ''),
                'away_team': raw_data.get('away_team', ''),
                'stadium': raw_data.get('stadium', ''),
                'weather_conditions': None,  # Will be populated separately
                'game_status': raw_data.get('game_type', 'scheduled')
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize schedule data: {e}")
            return pd.DataFrame()
    
    def _normalize_stats_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize stats data to our schema"""
        try:
            normalized = pd.DataFrame({
                'player_id': raw_data.get('player_id', ''),
                'game_id': raw_data.get('game_id', ''),
                'week': raw_data.get('week', 0),
                'season': raw_data.get('season', 0),
                'team': raw_data.get('recent_team', ''),
                'opponent': raw_data.get('opponent_team', ''),
                'position': raw_data.get('position', ''),
                'passing_attempts': raw_data.get('passing_attempts'),
                'passing_completions': raw_data.get('completions'),
                'passing_yards': raw_data.get('passing_yards'),
                'passing_touchdowns': raw_data.get('passing_tds'),
                'interceptions': raw_data.get('interceptions'),
                'rushing_attempts': raw_data.get('carries'),
                'rushing_yards': raw_data.get('rushing_yards'),
                'rushing_touchdowns': raw_data.get('rushing_tds'),
                'targets': raw_data.get('targets'),
                'receptions': raw_data.get('receptions'),
                'receiving_yards': raw_data.get('receiving_yards'),
                'receiving_touchdowns': raw_data.get('receiving_tds'),
                'offensive_snaps': None,  # From snap count data
                'snap_percentage': None
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize stats data: {e}")
            return pd.DataFrame()
    
    def _normalize_snap_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Normalize snap count data"""
        try:
            normalized = pd.DataFrame({
                'player_id': raw_data.get('player_id', ''),
                'game_id': raw_data.get('game_id', ''),
                'week': raw_data.get('week', 0),
                'season': raw_data.get('season', 0),
                'team': raw_data.get('team', ''),
                'position': raw_data.get('position', ''),
                'offensive_snaps': raw_data.get('offense_snaps'),
                'defensive_snaps': raw_data.get('defense_snaps'),
                'special_teams_snaps': raw_data.get('st_snaps'),
                'offense_pct': raw_data.get('offense_pct'),
                'defense_pct': raw_data.get('defense_pct'),
                'st_pct': raw_data.get('st_pct')
            })
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize snap data: {e}")
            return pd.DataFrame()

class WeatherAdapter(BaseAdapter):
    """Adapter for weather data from National Weather Service"""
    
    def __init__(self, cache_manager: CacheManager):
        super().__init__(cache_manager)
        self.base_url = "https://api.weather.gov"
        # Stadium coordinates (lat, lon)
        self.stadium_coords = {
            'ARI': (33.5276, -112.2626),  # State Farm Stadium
            'ATL': (33.7555, -84.4006),   # Mercedes-Benz Stadium
            'BAL': (39.2780, -76.6227),   # M&T Bank Stadium
            'BUF': (42.7738, -78.7870),   # Highmark Stadium
            'CAR': (35.2258, -80.8528),   # Bank of America Stadium
            'CHI': (41.8623, -87.6167),   # Soldier Field
            'CIN': (39.0955, -84.5160),   # Paycor Stadium
            'CLE': (41.5061, -81.6995),   # Cleveland Browns Stadium
            'DAL': (32.7473, -97.0945),   # AT&T Stadium
            'DEN': (39.7439, -105.0200),  # Empower Field at Mile High
            'DET': (42.3400, -83.0456),   # Ford Field (dome)
            'GB': (44.5013, -88.0622),    # Lambeau Field
            'HOU': (29.6847, -95.4107),   # NRG Stadium (dome)
            'IND': (39.7601, -86.1639),   # Lucas Oil Stadium (dome)
            'JAX': (30.3240, -81.6374),   # TIAA Bank Field
            'KC': (39.0489, -94.4839),    # Arrowhead Stadium
            'LV': (36.0909, -115.1833),   # Allegiant Stadium (dome)
            'LAC': (33.8644, -118.2611),  # SoFi Stadium (dome)
            'LAR': (33.8644, -118.2611),  # SoFi Stadium (dome)
            'MIA': (25.9580, -80.2389),   # Hard Rock Stadium
            'MIN': (44.9778, -93.2581),   # U.S. Bank Stadium (dome)
            'NE': (42.0909, -71.2643),    # Gillette Stadium
            'NO': (29.9511, -90.0812),    # Caesars Superdome (dome)
            'NYG': (40.8135, -74.0745),   # MetLife Stadium
            'NYJ': (40.8135, -74.0745),   # MetLife Stadium
            'PHI': (39.9008, -75.1675),   # Lincoln Financial Field
            'PIT': (40.4468, -80.0158),   # Heinz Field
            'SF': (37.4032, -121.9698),   # Levi's Stadium
            'SEA': (47.5952, -122.3316),  # Lumen Field
            'TB': (27.9759, -82.5034),    # Raymond James Stadium
            'TEN': (36.1665, -86.7713),   # Nissan Stadium
            'WAS': (38.9076, -76.8645)    # FedExField
        }
    
    async def fetch_weather(self, game_id: str, stadium: str, game_date: datetime) -> Optional[WeatherData]:
        """Fetch weather data for a specific game"""
        date_str = game_date.strftime("%Y-%m-%d")
        cache_key = f"weather_{game_id}"
        
        if self.cache_manager.is_cached(cache_key, date_str, max_age_hours=12):
            cached_data = self.cache_manager.load_from_cache(cache_key, date_str)
            if cached_data is not None and len(cached_data) > 0:
                return self._dataframe_to_weather(cached_data.iloc[0])
        
        # Check if stadium is domed (no weather impact)
        domed_stadiums = {'DET', 'HOU', 'IND', 'LV', 'LAC', 'LAR', 'MIN', 'NO'}
        team = self._stadium_to_team(stadium)
        
        if team in domed_stadiums:
            weather_data = WeatherData(
                game_id=game_id,
                stadium=stadium,
                temperature=72.0,  # Controlled environment
                humidity=50.0,
                wind_speed=0.0,
                wind_direction="N/A",
                precipitation=0.0,
                conditions="dome",
                timestamp=datetime.now()
            )
        else:
            # Fetch actual weather data
            weather_data = await self._fetch_nws_weather(game_id, stadium, game_date)
        
        if weather_data:
            # Cache the weather data
            df = pd.DataFrame([asdict(weather_data)])
            self.cache_manager.save_to_cache(df, cache_key, date_str)
        
        return weather_data
    
    async def _fetch_nws_weather(self, game_id: str, stadium: str, game_date: datetime) -> Optional[WeatherData]:
        """Fetch weather from National Weather Service API"""
        try:
            team = self._stadium_to_team(stadium)
            if team not in self.stadium_coords:
                logger.warning(f"No coordinates found for stadium: {stadium}")
                return None
            
            lat, lon = self.stadium_coords[team]
            
            async with aiohttp.ClientSession() as session:
                # Get grid point
                grid_url = f"{self.base_url}/points/{lat},{lon}"
                async with session.get(grid_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get grid point: {response.status}")
                        return None
                    
                    grid_data = await response.json()
                    forecast_url = grid_data['properties']['forecast']
                
                # Get forecast
                async with session.get(forecast_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get forecast: {response.status}")
                        return None
                    
                    forecast_data = await response.json()
                    periods = forecast_data['properties']['periods']
                    
                    # Find the period closest to game time
                    game_period = self._find_closest_period(periods, game_date)
                    
                    if game_period:
                        return WeatherData(
                            game_id=game_id,
                            stadium=stadium,
                            temperature=game_period.get('temperature'),
                            humidity=None,  # NWS doesn't provide humidity in basic forecast
                            wind_speed=self._extract_wind_speed(game_period.get('windSpeed', '')),
                            wind_direction=self._extract_wind_direction(game_period.get('windDirection', '')),
                            precipitation=0.0,  # Would need detailed forecast for this
                            conditions=game_period.get('shortForecast', '').lower(),
                            timestamp=datetime.now()
                        )
        
        except Exception as e:
            logger.error(f"Failed to fetch NWS weather data: {e}")
        
        return None
    
    def _stadium_to_team(self, stadium: str) -> str:
        """Convert stadium name to team abbreviation"""
        # This is a simplified mapping - in production, you'd have a comprehensive lookup
        stadium_map = {
            'State Farm Stadium': 'ARI',
            'Mercedes-Benz Stadium': 'ATL',
            'M&T Bank Stadium': 'BAL',
            'Highmark Stadium': 'BUF',
            # Add more mappings as needed
        }
        return stadium_map.get(stadium, stadium[:3].upper())
    
    def _find_closest_period(self, periods: List[Dict], game_date: datetime) -> Optional[Dict]:
        """Find forecast period closest to game time"""
        # Simplified - just return first period for now
        return periods[0] if periods else None
    
    def _extract_wind_speed(self, wind_speed_str: str) -> Optional[float]:
        """Extract wind speed from string like '10 mph'"""
        try:
            return float(wind_speed_str.split()[0])
        except:
            return None
    
    def _extract_wind_direction(self, wind_direction_str: str) -> str:
        """Extract wind direction"""
        return wind_direction_str.strip()
    
    def _dataframe_to_weather(self, row: pd.Series) -> WeatherData:
        """Convert DataFrame row to WeatherData object"""
        return WeatherData(
            game_id=row['game_id'],
            stadium=row['stadium'],
            temperature=row['temperature'],
            humidity=row['humidity'],
            wind_speed=row['wind_speed'],
            wind_direction=row['wind_direction'],
            precipitation=row['precipitation'],
            conditions=row['conditions'],
            timestamp=pd.to_datetime(row['timestamp'])
        )

class UnifiedDataIngestion:
    """Unified data ingestion coordinator"""
    
    def __init__(self, session: Session, cache_dir: str = "data/snapshots"):
        self.session = session
        self.cache_manager = CacheManager(cache_dir)
        
        # Initialize adapters
        self.nfl_adapter = NFLDataPyAdapter(self.cache_manager)
        self.weather_adapter = WeatherAdapter(self.cache_manager)
        
        logger.info("Unified data ingestion system initialized")
    
    async def ingest_weekly_data(self, season: int, week: int) -> Dict[str, Any]:
        """Ingest all data for a specific week"""
        logger.info(f"Starting weekly data ingestion for {season} Week {week}")
        
        results = {
            'season': season,
            'week': week,
            'timestamp': datetime.now(),
            'data_sources': {},
            'errors': []
        }
        
        try:
            # Fetch rosters
            rosters = await self.nfl_adapter.fetch_rosters(season, week)
            results['data_sources']['rosters'] = len(rosters)
            
            # Fetch schedules
            schedules = await self.nfl_adapter.fetch_schedules(season)
            week_schedules = schedules[schedules['week'] == week] if len(schedules) > 0 else pd.DataFrame()
            results['data_sources']['schedules'] = len(week_schedules)
            
            # Fetch weekly stats
            stats = await self.nfl_adapter.fetch_weekly_stats(season, week)
            results['data_sources']['stats'] = len(stats)
            
            # Fetch snap counts
            snaps = await self.nfl_adapter.fetch_snap_counts(season, week)
            results['data_sources']['snap_counts'] = len(snaps)
            
            # Fetch weather data for games
            weather_data = []
            for _, game in week_schedules.iterrows():
                weather = await self.weather_adapter.fetch_weather(
                    game['game_id'], 
                    game['stadium'], 
                    game['game_date']
                )
                if weather:
                    weather_data.append(weather)
            
            results['data_sources']['weather'] = len(weather_data)
            
            logger.info(f"Weekly data ingestion complete: {results['data_sources']}")
            
        except Exception as e:
            error_msg = f"Weekly data ingestion failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results
    
    async def ingest_season_foundation(self, season: int) -> Dict[str, Any]:
        """Ingest foundational season data (schedules, rosters)"""
        logger.info(f"Starting season foundation ingestion for {season}")
        
        results = {
            'season': season,
            'timestamp': datetime.now(),
            'data_sources': {},
            'errors': []
        }
        
        try:
            # Fetch full season schedules
            schedules = await self.nfl_adapter.fetch_schedules(season)
            results['data_sources']['schedules'] = len(schedules)
            
            # Fetch current rosters
            rosters = await self.nfl_adapter.fetch_rosters(season)
            results['data_sources']['rosters'] = len(rosters)
            
            logger.info(f"Season foundation ingestion complete: {results['data_sources']}")
            
        except Exception as e:
            error_msg = f"Season foundation ingestion failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
        
        return results

# Example usage and testing
async def main():
    """Example usage of the ingestion system"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create test database session
    engine = create_engine("sqlite:///test_ingestion.db")
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Initialize ingestion system
    ingestion = UnifiedDataIngestion(session)
    
    # Test weekly data ingestion
    results = await ingestion.ingest_weekly_data(2024, 1)
    print("Weekly ingestion results:", results)
    
    # Test season foundation ingestion
    foundation_results = await ingestion.ingest_season_foundation(2024)
    print("Foundation ingestion results:", foundation_results)
    
    session.close()

if __name__ == "__main__":
    asyncio.run(main())
